#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

What this does:
- Per-feature normalization for EPS stability (by default computed ONCE from the initial train window)
- Triple loss per batch:
    (1) reconstruction of observed triples,
    (2) predict next time-embedding from history,
    (3) EPS forecast loss at the validation step (next time step after train)
- Rolling-origin expanding-window training:
    Window k: Train 0..train_end, Validate = train_end+1 (used for the loss), Test = train_end+2 (metrics).
    Repeat for 'epochs_per_window' epochs, then slide window by one time step and continue.
- Progress bars: outer (windows) and inner (epochs)
- PCA + Sundial

Adds at end:
- Final across-all-tests summary: RMSE / R² / Kelly R², loss-mixture weights, and first 50 rows
  of the aggregated GroundTruth vs Forecasted; saves the full table to CSV.
"""

import os, sys, math, argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from typing import Tuple, List, Optional, Dict, Any
import wandb

# add near the other imports
from pathlib import Path
import re

import random

# from transformers import AutoModelForCausalLM  # (import) Sundial HF model loader
import timesfm  # (library) Google TimesFM: pretrained time-series model (PyTorch backend)

# ADD ↓↓↓
import hashlib  # (std lib) string hashing for unique file names
import time  # (std lib) wall-clock time; used in run hash to avoid collision
import warnings

# Matplotlib is used for saving plots on headless servers (HPC).
import matplotlib  # plotting backend

matplotlib.use("Agg")  # use non-interactive backend so savefig works on cluster nodes
import matplotlib.pyplot as plt  # plotting API
from typing import List

# ADD ↑↑↑


torch.backends.cuda.matmul.allow_tf32 = (
    True  # prints: (no shape); meaning: allow fast TF32 matmuls
)
torch.set_float32_matmul_precision(
    "high"
)  # prints: (no shape); meaning: favor speed for FP32 matmuls
# ADD ↓↓↓  (orthogonality / decorrelation helpers)


# --- helper for clean folder names from floats ---
def _fmt(x: float) -> str:
    """
    Format floats compactly for directory names:
    - keeps scientific notation like 1e-06
    - avoids spaces/extra zeros
    """
    return f"{float(x):g}"


def _center_time(embedding_over_time: torch.Tensor) -> torch.Tensor:
    """
    Center each embedding dimension by its mean over time.
    embedding_over_time: (T, R) or (B, T, R) float32
    returns             : same shape; zero-mean along time axis
    """
    if embedding_over_time.dim() == 2:  # (T, R)
        # embedding_over_time.mean(dim=0, keepdim=True): [1, R] float
        return embedding_over_time - embedding_over_time.mean(dim=0, keepdim=True)
    elif embedding_over_time.dim() == 3:  # (B, T, R)
        # embedding_over_time.mean(dim=1, keepdim=True): [B, 1, R] float
        return embedding_over_time - embedding_over_time.mean(dim=1, keepdim=True)
    else:
        raise ValueError("Expected shape (T, R) or (B, T, R)")


def _covariance_centered(centered_over_time: torch.Tensor) -> torch.Tensor:
    """
    Empirical covariance across time (biased, divide by T).
    centered_over_time: (T, R) or (B, T, R) float32
    returns           : (R, R) or (B, R, R) float32
    """
    if centered_over_time.dim() == 2:
        # T: int (time length)
        T = centered_over_time.shape[0]
        # centered_over_time.T @ centered_over_time: [R, T] @ [T, R] -> [R, R] float
        return (centered_over_time.T @ centered_over_time) / max(T, 1)
    else:
        # B, T, R: int
        B, T, R = centered_over_time.shape
        # centered_over_time.transpose(1, 2): [B, R, T]
        # torch.matmul: [B, R, T] @ [B, T, R] -> [B, R, R]
        return torch.matmul(
            centered_over_time.transpose(1, 2), centered_over_time
        ) / max(T, 1)


def _zero_diagonal(square_matrix: torch.Tensor) -> torch.Tensor:
    """
    Zero out diagonal entries; leave off-diagonals as-is.
    square_matrix: (R, R) or (B, R, R) float32
    returns      : same shape float32
    """
    if square_matrix.dim() == 2:
        # torch.diag(square_matrix.diag()): [R, R] diag made from diagonal
        return square_matrix - torch.diag(torch.diag(square_matrix))
    else:
        # torch.diag_embed(...) makes diag tensors batched: [B, R, R]
        return square_matrix - torch.diag_embed(
            torch.diagonal(square_matrix, dim1=-2, dim2=-1)
        )


def decorrelation_L2(embedding_over_time: torch.Tensor) -> torch.Tensor:
    """
    L2 penalty on off-diagonal covariance (after centering over time).
    embedding_over_time: (T, R) or (B, T, R)
    returns             : scalar float32 tensor
    """
    centered = _center_time(embedding_over_time)  # same shape as input
    covariance = _covariance_centered(centered)  # (R, R) or (B, R, R)
    offdiag = _zero_diagonal(covariance)  # diagonal set to 0
    if offdiag.dim() == 2:
        return (offdiag.pow(2)).mean()  # scalar
    else:
        return (offdiag.pow(2)).mean(dim=(1, 2)).mean()  # scalar (mean over batch)


def decorrelation_L1(embedding_over_time: torch.Tensor) -> torch.Tensor:
    """
    L1 penalty on off-diagonal covariance (after centering over time).
    embedding_over_time: (T, R) or (B, T, R)
    returns             : scalar float32 tensor
    """
    centered = _center_time(embedding_over_time)  # same shape as input
    covariance = _covariance_centered(centered)  # (R, R) or (B, R, R)
    offdiag = _zero_diagonal(covariance)  # diagonal set to 0
    if offdiag.dim() == 2:
        return offdiag.abs().mean()  # scalar
    else:
        return offdiag.abs().mean(dim=(1, 2)).mean()  # scalar


# ADD ↑↑↑


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    v = str(v).lower()
    if v in ("yes", "y", "true", "t", "1"):
        return True
    if v in ("no", "n", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (at some perf cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # ensure each DataLoader worker gets a different, but
    # deterministic, seed
    torch.utils.data.get_worker_info()
    g = torch.Generator()
    g.manual_seed(seed)
    return g  # return generator for use in DataLoader


# ---------------- Metrics ----------------
def r2(gt: torch.Tensor, pred: torch.Tensor) -> float:
    ss_res = torch.sum((gt - pred) ** 2)
    ss_tot = torch.sum((gt - torch.mean(gt)) ** 2)
    return (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0


def r2_kelly(gt: torch.Tensor, pred: torch.Tensor) -> float:
    ss_res = torch.sum((gt - pred) ** 2)
    ss_tot = torch.sum(gt**2)
    return (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0


def rmse_metric(gt: torch.Tensor, pred: torch.Tensor) -> float:
    gt = gt.float()
    pred = pred.float()
    if gt.numel() == 0:
        return float("nan")
    diff = gt - pred
    return float(torch.sqrt(torch.mean(diff * diff)).item())


def mae_metric(gt: torch.Tensor, pred: torch.Tensor) -> float:
    gt = gt.float()
    pred = pred.float()
    if gt.numel() == 0:
        return float("nan")
    return float(torch.mean(torch.abs(gt - pred)).item())


def mape_metric(gt: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> float:
    # Why eps: skip entries with |gt| <= eps so near-zero truths don't blow up MAPE.
    gt = gt.float()
    pred = pred.float()
    if gt.numel() == 0:
        return float("nan")
    mask = torch.abs(gt) > eps
    if not torch.any(mask):
        return float("nan")
    return float((torch.mean(torch.abs((gt[mask] - pred[mask]) / gt[mask])) * 100.0).item())


def build_xgboost_eligibility_by_date(
    csv_path: str,
) -> Dict[pd.Timestamp, set[str]]:
    """
    Reproduce the row eligibility used by examples/xgboost_expanding_1.py.

    XGBoost first drops missing EPS targets, then creates previous-observed-row
    lags per ticker, then drops rows with any missing lagged predictor. Filtering
    Tensor-TimesFM OOS rows to this mask makes final metrics use the same
    GroundTruth sample.
    """
    raw_feature_cols = [
        "lmean",
        "atq",
        "ni",
        "dv",
        "acc",
        "invest",
        "mc",
        "bm",
        "dinvt",
        "dar",
        "capx",
        "gm",
        "sga",
    ]
    same_quarter_feature_cols = ["lmean"]
    lag_base_cols = [
        col for col in raw_feature_cols if col not in same_quarter_feature_cols
    ]
    lagged_feature_cols = [f"{col}_lag1" for col in lag_base_cols]

    id_col = "ticker"
    date_col = "fpedats"
    target_col = "lvalue"

    df = pd.read_csv(csv_path, dtype={id_col: str})
    missing = [
        col
        for col in [id_col, date_col, target_col] + raw_feature_cols
        if col not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Cannot build XGBoost eligibility mask; missing columns: {missing}"
        )

    df[id_col] = df[id_col].astype(str).str.strip()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    for col in raw_feature_cols + [target_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df_model = df[[id_col, date_col, target_col] + raw_feature_cols].copy()
    df_model = df_model.dropna(subset=[id_col, date_col, target_col])
    df_model = df_model.sort_values([id_col, date_col]).reset_index(drop=True)

    lagged_features = df_model.groupby(id_col)[lag_base_cols].shift(1)
    lagged_features.columns = lagged_feature_cols
    df_model = pd.concat([df_model, lagged_features], axis=1)
    df_model = df_model.dropna(subset=lagged_feature_cols)
    df_model = df_model.sort_values([date_col, id_col]).reset_index(drop=True)

    eligible_by_date: Dict[pd.Timestamp, set[str]] = {}
    for date_value, tickers_for_date in df_model.groupby(date_col)[id_col]:
        eligible_by_date[pd.Timestamp(date_value).normalize()] = set(
            tickers_for_date.astype(str).str.strip()
        )
    return eligible_by_date


def infer_tensor_time_axis_dates_from_csv(csv_path: str) -> np.ndarray:
    """
    Reproduce the tensor-prep time axis: sorted quarterly Periods from fpedats.
    Return normalized quarter-end Timestamps indexed by tensor t.
    """
    df = pd.read_csv(csv_path, usecols=["fpedats"])
    fpedats_dt = pd.to_datetime(df["fpedats"], errors="coerce")
    unique_quarters = np.sort(fpedats_dt.dt.to_period("Q").dropna().unique())
    return np.array(
        [pd.Period(q, freq="Q").to_timestamp(how="end").normalize() for q in unique_quarters],
        dtype=object,
    )


def masked_mse(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute mean squared error over non-NaN targets.
    predicted : [*, *] float; meaning: model predictions
    target    : [*, *] float; meaning: ground-truth (NaN allowed)
    returns   : scalar float tensor; meaning: MSE over observed entries
    """
    valid_mask = torch.isfinite(
        target
    )  # prints: same shape; meaning: True where target is observed and finite
    if not torch.any(valid_mask):
        return torch.tensor(
            0.0, device=predicted.device
        )  # prints: scalar; meaning: no labels → zero loss
    if not torch.isfinite(predicted[valid_mask]).all():
        raise ValueError("masked_mse received non-finite predictions on observed targets.")
    diff = (
        predicted[valid_mask] - target[valid_mask]
    )  # prints: [n] float; meaning: residuals on observed entries
    return torch.mean(diff * diff)  # prints: scalar float; meaning: MSE over observed


# ------------- Dataset (finite observed triples) -------------
class FilteredTensor3DDataset(Dataset):
    def __init__(self, tensor3d: torch.Tensor):
        # keep on CPU; DataLoader will move batches to GPU
        cpu = tensor3d.detach().cpu()
        mask = torch.isfinite(cpu)
        self.idxs = mask.nonzero(as_tuple=False).long()  # [N,3]
        self.values = cpu[mask].float()  # [N]

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.idxs[i], self.values[i]


"""
sparse tensor: give self.idxs
"""


# ---------------- Model ----------------
class TensorTimesFM(nn.Module):
    """
    Pure predictor with three modes:
      1) mode="reconstruction":
           inputs:  time_index [batch_size],
                    firm_index [batch_size],
                    feature_index [batch_size]
           output:  predicted_values [batch_size]   (normalized space)

      2) mode="prediction":
           inputs:  history_time_indices [history_length]  or [batch_size, history_length]
           output:  next_time_embedding
                    shape [time_embedding_size]  or [batch_size, time_embedding_size]

      3) mode="forecast":
           inputs:  predicted_time_embedding [time_embedding_size] or [batch_size, time_embedding_size],
                    firm_index [num_targets],
                    feature_index [num_targets]  (or a scalar int for “same feature for all”)
           output:  predicted_values [num_targets]   (normalized space)
    """

    def __init__(
        self,
        shape,  # (time_count, firm_count, feature_count)
        time_embedding_size,  # Rank_CP
        mlp_hidden=1024,
        dropout_p=0.1,
        device="cpu",
        use_timesfm: bool = True,  # NEW (bool)
        timesfm_repo_id: str = "google/timesfm-2.0-500m-pytorch",  # NEW (str)
        timesfm_horizon_len: int = 1,  # NEW (int)
        timesfm_context_cap: int = 2048,  # NEW (int)
        timesfm_per_core_batch_size: int = 32,  # NEW (int)
        timesfm_freq_category: int = 2,  # NEW (int) 0/1/2 → EPS quarterly=2
    ):
        super().__init__()
        time_count, firm_count, feature_count = shape
        self.device = device

        # embeddings
        self.time_embeddings = nn.Embedding(time_count, time_embedding_size)
        self.firm_embeddings = nn.Embedding(firm_count, time_embedding_size)
        self.feature_embeddings = nn.Embedding(feature_count, time_embedding_size)

        # GRU over time-embedding sequences
        self.sequence_model = nn.GRU(
            input_size=time_embedding_size,
            hidden_size=time_embedding_size,
            num_layers=2,
            batch_first=True,
        )

        # 2-layer adapter after TimesFM (learnable)
        # Shape:
        #   input : [R]  (TimesFM one-step forecast of the time embedding)
        #   output: [R]  (mapped into the model's time-embedding space)
        #
        # NOTE (stability):
        #   In predict_next_time_embedding_with_adapter() we use a residual form:
        #       adapted = raw + adapter_mlp(raw)
        #   and we initialize adapter_mlp to output ~0 at the start, so adapted ≈ raw.
        self.time_prediction_adapter_hidden_size = int(
            time_embedding_size
        )  # (int) hidden width of adapter MLP

        self.time_prediction_adapter = nn.Sequential(
            nn.Linear(
                time_embedding_size, self.time_prediction_adapter_hidden_size, bias=True
            ),
            nn.ReLU(),
            nn.Linear(
                self.time_prediction_adapter_hidden_size, time_embedding_size, bias=True
            ),
        )

        # Initialize adapter so that at the start: adapter(raw) ≈ 0, i.e. adapted ≈ raw.
        # Why we do NOT zero both Linear layers (the previous bug):
        #   If both layers are zero, the first layer outputs zero for any input,
        #   ReLU stays zero, and the second layer also receives zero input.
        #   The first-layer weights then get no useful gradient signal, so the
        #   adapter never learns input-dependent corrections — only a constant bias.
        # Correct strategy:
        #   - First Linear: standard kaiming_uniform_ (PyTorch default for Linear),
        #     so it produces a non-trivial, input-dependent intermediate vector.
        #   - Second Linear: weights and bias zeroed, so the adapter's overall
        #     output is exactly 0 at initialization.
        # Combined with the residual form `adapted = raw + adapter(raw)`
        # (see predict_next_time_embedding_with_adapter), this gives adapted ≈ raw
        # at init while keeping every parameter on a path that receives gradient.
        adapter_first_linear = self.time_prediction_adapter[0]
        adapter_second_linear = self.time_prediction_adapter[2]
        assert isinstance(adapter_first_linear, nn.Linear)
        assert isinstance(adapter_second_linear, nn.Linear)
        with torch.no_grad():
            # First Linear layer: alive, kaiming-uniform weights.
            # nn.init.kaiming_uniform_ : in-place He-style init; `a=math.sqrt(5)`
            #   matches PyTorch's nn.Linear default reset_parameters().
            nn.init.kaiming_uniform_(
                adapter_first_linear.weight, a=math.sqrt(5)
            )  # shape: [hidden, R] float32
            adapter_first_linear.bias.zero_()  # shape: [hidden] float32
            # Second Linear layer: zeroed → adapter output = 0 at init.
            adapter_second_linear.weight.zero_()  # shape: [R, hidden] float32
            adapter_second_linear.bias.zero_()  # shape: [R] float32

        # two task heads (reconstruction vs forecast): both map [time_vec | firm_vec | feature_vec] -> scalar
        # We keep the same architecture for both heads, but they do NOT share weights.
        self.reconstruction_head = self._build_scalar_head(
            input_dim=3 * time_embedding_size,
            mlp_hidden=mlp_hidden,
            dropout_p=dropout_p,
        )
        self.forecast_head = self._build_scalar_head(
            input_dim=3 * time_embedding_size,
            mlp_hidden=mlp_hidden,
            dropout_p=dropout_p,
        )

        self.eps_regressor_hidden_size = 1024  # prints: int; meaning: width of the regression MLP that maps drivers→EPS (tunable)
        self.eps_regressor = None  # prints: None; meaning: we will build the regressor lazily once we know input dim K
        self.dropout_p = (
            dropout_p  # prints: float; meaning: reuse same dropout prob as the main MLP
        )
        self.use_timesfm = bool(use_timesfm)  # (bool) enable TimesFM path
        self.timesfm_freq_category = int(timesfm_freq_category)  # (int) 0/1/2
        self.timesfm_horizon_len = int(timesfm_horizon_len)  # (int) usually 1
        self.timesfm_context_cap = int(timesfm_context_cap)  # (int) 2048
        self.timesfm_per_core_batch_size = int(timesfm_per_core_batch_size)  # (int)

        # Build a single TimesFM object (PyTorch backend). Keep it frozen.
        if self.use_timesfm:
            self.timesfm = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend=("gpu" if torch.cuda.is_available() else "cpu"),
                    per_core_batch_size=self.timesfm_per_core_batch_size,
                    horizon_len=self.timesfm_horizon_len,
                    context_len=self.timesfm_context_cap,
                    input_patch_len=32,
                    output_patch_len=128,
                    num_layers=50,
                    model_dims=1280,
                    use_positional_embedding=False,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=timesfm_repo_id
                ),
            )
            # Freeze TimesFM parameters so optimizer never updates them
            # Freeze the underlying PyTorch model (TimesFm wrapper is not an nn.Module)
            base_model = getattr(self.timesfm, "_model", None)
            if base_model is None:
                raise RuntimeError(
                    "TimesFM PyTorch backend did not expose _model; upgrade the timesfm package."
                )

            for p in base_model.parameters():
                p.requires_grad_(False)
            assert not any(
                p.requires_grad for p in base_model.parameters()
            ), "TimesFM not frozen"
            base_model.eval()  # inference mode

        else:
            self.timesfm = None

    def _build_scalar_head(
        self,
        input_dim: int,  # (int) = 3 * rank; meaning: concat dim [time|firm|feature]
        mlp_hidden: int,  # (int) width of hidden layers
        dropout_p: float,  # (float) dropout probability
    ) -> nn.Module:
        """
        Build a small MLP that outputs ONE scalar per row.
        Shape:
            input  : [B, input_dim]
            output : [B, 1]
        """
        return nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(mlp_hidden, 1),
        )

    def _predict_next_time_embedding_on_graph_from_matrix(
        self, time_embedding_history: torch.Tensor
    ) -> torch.Tensor:
        """
        Layer B: given an EXPLICIT history embedding matrix [L, R], run GRU or
        TimesFM (frozen) and return the next raw embedding [R].

        Skips the `self.time_embeddings(...)` lookup so callers can supply a
        history whose last row is a *predicted* embedding (e.g., final-test
        forecast where slot p-1 is e_hat_{T-1} rather than a table value).

        time_embedding_history: torch.Tensor [L, R]
            L = history length; R = time_embedding_size.
        Returns:
            next_time_embedding_raw: torch.Tensor [R], dtype matches input.
        """
        # (1) shape: [L, R] → batch by rank: [B=R, T=L]
        L, R = int(time_embedding_history.shape[0]), int(
            time_embedding_history.shape[1]
        )
        inputs_bt = time_embedding_history.transpose(0, 1).contiguous()  # [R, L]

        if not self.use_timesfm:
            inputs_btr = time_embedding_history.unsqueeze(0)  # [1, L, R]
            _, last_hidden = self.sequence_model(inputs_btr)  # [layers, 1, R]
            return last_hidden[-1].squeeze(0)  # [R]

        # TimesFM path (PyTorch patched decoder)
        base_model = self.timesfm._model
        horizon = int(self.timesfm_horizon_len)
        cap = int(self.timesfm_context_cap)

        # --- respect context cap: keep most recent window of history, leaving room for horizon ---
        max_hist = max(1, cap - horizon)
        if L > max_hist:
            inputs_bt = inputs_bt[:, -max_hist:]
            L = int(inputs_bt.size(1))

        # --- align history length to the model's patch size by LEFT-padding ---
        # TimesFM reshapes input into patches of length `patch_len` (usually 32).
        P = int(getattr(getattr(base_model, "config", None), "patch_len", 32))
        need = (P - (L % P)) % P

        # If adding left-pad would exceed cap, drop extra tokens from the left first.
        if L + need > max_hist:
            drop = (L + need) - max_hist
            if drop > 0:
                inputs_bt = inputs_bt[:, drop:]
                L = int(inputs_bt.size(1))
                need = (P - (L % P)) % P  # recompute after drop

        if need:
            # left-pad with zeros; these positions will be masked via paddings
            inputs_bt = torch.nn.functional.pad(inputs_bt, (need, 0), "constant", 0.0)
            L += need

        # --- paddings must be length (L + horizon) ---
        # mark left-pad (if any) as padded, and the future/horizon on the right as padded
        padd_len = L + horizon
        paddings_bt = torch.zeros(
            (R, padd_len), dtype=inputs_bt.dtype, device=inputs_bt.device
        )
        if need:
            paddings_bt[:, :need] = 1.0
        paddings_bt[:, L:] = 1.0  # pad horizon steps

        # frequency per series (3 categories in patched decoder; 2 = quarter/year)
        freq_bt = torch.full(
            (R, 1),
            int(self.timesfm_freq_category),
            device=inputs_bt.device,
            dtype=torch.int64,
        )

        # dtype alignment with model weights
        mdl_dtype = next(base_model.parameters()).dtype
        inputs_bt = inputs_bt.to(dtype=mdl_dtype)
        paddings_bt = paddings_bt.to(dtype=mdl_dtype)

        # debug prints (optional)
        print("L_hist:", L, "horizon:", horizon, "padd_len:", padd_len)
        print(
            "inputs_bt:",
            inputs_bt.shape,
            "paddings_bt:",
            paddings_bt.shape,
            "freq_bt:",
            freq_bt.shape,
        )

        # decode returns (mean, full); we want the first-step mean
        decoded = base_model.decode(inputs_bt, paddings_bt, freq_bt, horizon)
        if isinstance(decoded, tuple):
            mean_bt, _full_bt = decoded
        else:
            # very old versions might return only mean
            mean_bt = decoded

        # first-step forecast
        next_values_b = mean_bt[:, 0]  # [R]
        return next_values_b.to(dtype=time_embedding_history.dtype)  # [R]

    def predict_next_time_embedding_on_graph(
        self, history_time_indices: torch.Tensor
    ) -> torch.Tensor:
        # Layer A: indices -> embedding matrix [L, R] via the learnable table.
        time_embedding_history = self.time_embeddings(history_time_indices)  # [L, R]
        # Layer B: explicit matrix -> next raw embedding [R].
        return self._predict_next_time_embedding_on_graph_from_matrix(
            time_embedding_history
        )

    def predict_next_time_embedding_with_adapter(
        self, history_time_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict next time embedding using:
          - GRU (if use_timesfm=False), OR
          - frozen TimesFM + trainable residual adapter (if use_timesfm=True).

        Returns:
            next_time_embedding_adapted: [R] float32
        """
        next_time_embedding_raw = self.predict_next_time_embedding_on_graph(
            history_time_indices=history_time_indices
        )  # [R] float32
        if self.use_timesfm:
            # Adapter MLP expects [B, R], so we temporarily add batch dim B=1.
            time_prediction_adapter_input_batch = next_time_embedding_raw.unsqueeze(
                0
            )  # [1, R] float32
            time_prediction_adapter_output_batch = self.time_prediction_adapter(
                time_prediction_adapter_input_batch
            )  # [1, R] float32
            time_prediction_adapter_output = (
                time_prediction_adapter_output_batch.squeeze(0)
            )  # [R] float32

            # Residual: start from raw, then add a learnable correction.
            next_time_embedding_adapted = (
                next_time_embedding_raw + time_prediction_adapter_output
            )  # [R] float32
            return next_time_embedding_adapted

        # GRU path: no TimesFM adapter is used.
        return next_time_embedding_raw  # [R] float32

    def predict_next_time_embedding_from_history_embedding_matrix_with_adapter(
        self, history_time_embedding_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Like predict_next_time_embedding_with_adapter, but takes an EXPLICIT
        history embedding matrix instead of time indices. Used at final-test
        time when the last row of the history is a model-predicted embedding
        e_hat_{T-1} rather than a table entry self.time_embeddings(T-1).

        Reuses the same Layer-B core (TimesFM/GRU + dtype handling) and the
        same 2-layer residual adapter as the index-based wrapper above; this
        is a NEW ENTRY POINT, not a new architecture.

        history_time_embedding_matrix: torch.Tensor [context_length, R]
        Returns:
            predicted_next_time_embedding: torch.Tensor [R] float32
        """
        # Layer B (no Layer-A lookup): explicit matrix -> raw next embedding.
        next_time_embedding_raw = (
            self._predict_next_time_embedding_on_graph_from_matrix(
                history_time_embedding_matrix
            )
        )  # [R] float32 — raw TimesFM/GRU output, before adapter
        if self.use_timesfm:
            # Same adapter pathway as the index-based method.
            time_prediction_adapter_input_batch = next_time_embedding_raw.unsqueeze(
                0
            )  # [1, R] float32
            time_prediction_adapter_output_batch = self.time_prediction_adapter(
                time_prediction_adapter_input_batch
            )  # [1, R] float32
            time_prediction_adapter_output = (
                time_prediction_adapter_output_batch.squeeze(0)
            )  # [R] float32
            # Residual: raw + learnable correction (init-zero second linear).
            return next_time_embedding_raw + time_prediction_adapter_output  # [R]

        # GRU path: no adapter.
        return next_time_embedding_raw  # [R] float32

    # helper: combine parts to a value
    # helper: combine parts to a value
    def predict_value_from_parts(
        self,
        time_vector: torch.Tensor,  # [B, R] float; meaning: either true time embedding e_t or predicted e_hat
        firm_index: torch.Tensor,  # [B] long; meaning: firm ids
        feature_index: torch.Tensor,  # [B] long; meaning: feature ids
        head: str,  # (str) either 'reconstruction' or 'forecast'
    ) -> torch.Tensor:
        """
        returns:
            predicted normalized value: [B] float
        """
        # pick which head to use
        if head == "reconstruction":
            scalar_head = self.reconstruction_head
        elif head == "forecast":
            scalar_head = self.forecast_head
        else:
            raise ValueError(f"Unknown head: {head!r}")

        firm_vector = self.firm_embeddings(firm_index)  # [B, R]
        feature_vector = self.feature_embeddings(feature_index)  # [B, R]
        concat = torch.cat([time_vector, firm_vector, feature_vector], dim=1)  # [B, 3R]
        return scalar_head(concat).squeeze(1)  # [B]

    def forecast_features_matrix(
        self,
        predicted_time_embedding: torch.Tensor,  # prints: [R] or [1,R] float; meaning: predicted next time vector
        firm_index: torch.Tensor,  # prints: [M] long;  meaning: firm indices to forecast for
        feature_indices: torch.Tensor,  # prints: [K] long;  meaning: which features (drivers) to forecast
    ) -> torch.Tensor:
        """
        Return normalized driver predictions for T+1 as a dense matrix [M, K].
        Stage A: use the forecast head on each feature requested.
        """
        # ensure shape [1, R] so we can expand to all firms
        if predicted_time_embedding.dim() == 1:
            predicted_time_embedding = predicted_time_embedding.unsqueeze(
                0
            )  # prints: [1,R]; meaning: add batch dim for expand

        # repeat time vector for each firm row
        repeated_time_vectors = predicted_time_embedding.expand(
            firm_index.size(0), -1
        )  # prints: [M,R]; meaning: align time vec to firms
        firm_vectors = self.firm_embeddings(
            firm_index
        )  # prints: [M,R] float; meaning: firm embedding rows

        # forecast each requested feature column using the forecast head
        predicted_columns = (
            []
        )  # prints: list of [M] tensors; meaning: we will stack into [M,K]
        for one_feature_index in feature_indices:
            # get feature embedding row and expand it to [M,R]
            feature_vector = self.feature_embeddings(one_feature_index.view(1)).expand(
                firm_index.size(0), -1
            )  # prints: [M,R]
            # concat the three parts → [M,3R]
            concatenated_inputs = torch.cat(
                [repeated_time_vectors, firm_vectors, feature_vector], dim=1
            )  # prints: [M,3R]
            # pass through predictor → scalar per row
            one_column_prediction = self.forecast_head(concatenated_inputs).squeeze(
                1
            )  # prints: [M] float
            predicted_columns.append(one_column_prediction)

        # stack each [M] column into [M,K]
        return torch.stack(
            predicted_columns, dim=1
        )  # prints: [M,K] float; meaning: drivers_{T+1} (normalized)

    def forward(self, mode: str, **inputs):
        if mode == "reconstruction":
            # inputs required: time_index, firm_index, feature_index (all [batch_size])
            time_index = inputs["time_index"]
            firm_index = inputs["firm_index"]
            feature_index = inputs["feature_index"]
            time_vector = self.time_embeddings(
                time_index
            )  # [batch_size, time_embedding_size]
            return self.predict_value_from_parts(
                time_vector, firm_index, feature_index, head="reconstruction"
            )

        elif mode == "prediction":
            # inputs required: history_time_indices ([history_length] or [batch_size, history_length])
            history_time_indices = inputs["history_time_indices"]
            if history_time_indices.dim() == 1:
                # [history_length] -> [1, history_length]
                history_time_indices = history_time_indices.unsqueeze(0)
            # embed each time step: [batch_size, history_length, time_embedding_size]
            time_sequence = self.time_embeddings(history_time_indices)
            # run GRU; last hidden state is our "next time embedding"
            _, last_hidden = self.sequence_model(
                time_sequence
            )  # last_hidden: [num_layers, batch_size, time_embedding_size]
            next_time_embedding = last_hidden[-1]  # [batch_size, time_embedding_size]
            # if batch_size is 1, return [time_embedding_size]
            return (
                next_time_embedding.squeeze(0)
                if next_time_embedding.size(0) == 1
                else next_time_embedding
            )

        elif mode == "forecast":
            """
            08/18/25
            Do not touch the time factor, only touch the "firm, features". Freeze everything, just let it pass gradient.
            shared_layers are embedding
            ONLY MODIFY THE forecast task, do not modify time factor anymore.
            """

            """
            Forecast all features, not just target. --> Feedforward Network (for loop), MLP (this is regression here) --> mse(predicted_EPS, ground_truth_EPS)
            
            """
            # inputs required: predicted_time_embedding ([time_embedding_size] or [batch_size, time_embedding_size]),
            #                  firm_index [num_targets],
            #                  feature_index [num_targets] or scalar
            predicted_time_embedding = inputs["predicted_time_embedding"]
            firm_index = inputs["firm_index"]
            feature_index_input = inputs["feature_index"]

            if predicted_time_embedding.dim() == 1:
                predicted_time_embedding = predicted_time_embedding.unsqueeze(
                    0
                )  # [1, time_embedding_size]

            # if feature_index is a Python int, broadcast it
            if torch.is_tensor(feature_index_input):
                feature_index = feature_index_input
            else:
                feature_index = torch.full_like(firm_index, int(feature_index_input))

            # repeat predicted time vector to match target count
            repeated_time_vectors = predicted_time_embedding.expand(
                firm_index.size(0), -1
            )  # [num_targets, time_embedding_size]
            return self.predict_value_from_parts(
                repeated_time_vectors, firm_index, feature_index, head="forecast"
            )
        elif mode == "forecast_regression":
            # kept for backward-compatibility (unused by new G1/G2 path)
            predicted_time_embedding = inputs["predicted_time_embedding"]
            firm_index = inputs["firm_index"]
            driver_feature_indices = inputs["driver_feature_indices"]

            predicted_drivers_normalized = self.forecast_features_matrix(
                predicted_time_embedding=predicted_time_embedding,
                firm_index=firm_index,
                feature_indices=driver_feature_indices,
            )
            regression_head = self._ensure_eps_regressor(
                input_dim_drivers=predicted_drivers_normalized.size(1)
            )
            return regression_head(predicted_drivers_normalized).squeeze(1)

        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    def _ensure_eps_regressor(self, input_dim_drivers: int) -> nn.Module:
        """
        Lazily build the drivers→EPS regressor with correct input dim K.
        Returns a torch.nn.Sequential on self.device.
        """
        # if not built yet, or K changed, (re)build
        need_rebuild = (
            self.eps_regressor is None
            or getattr(self.eps_regressor[0], "in_features", None) != input_dim_drivers
        )  # prints: bool; meaning: check whether we must rebuild
        if need_rebuild:
            self.eps_regressor = nn.Sequential(
                nn.Linear(
                    input_dim_drivers, self.eps_regressor_hidden_size
                ),  # prints: Linear(K, H); meaning: first layer
                nn.ReLU(),  # prints: activation; meaning: nonlinearity
                nn.Dropout(self.dropout_p),  # prints: dropout; meaning: regularization
                nn.Linear(
                    self.eps_regressor_hidden_size, self.eps_regressor_hidden_size
                ),  # prints: Linear(H,H)
                nn.ReLU(),  # prints: activation
                nn.Dropout(self.dropout_p),  # prints: dropout
                nn.Linear(
                    self.eps_regressor_hidden_size, 1
                ),  # prints: Linear(H,1); meaning: EPŜ (normalized)
            ).to(self.device)
        return self.eps_regressor  # prints: nn.Sequential; meaning: ready to use


# ---------- evaluate_one_step (CHANGED to use G1 forecast + G2 ground truth) ----------
def evaluate_one_step(
    model,
    full_raw_tensor,
    feat_means,
    feat_stds,
    train_T: int,
    test_idx: int,
    target_feature_index: int,
    consensus_feature_index: int,
    device,
    use_two_stage_forecast: bool,
    regression_feature_indices_list: list[int],
    tickers_array: np.ndarray,
    use_timesfm: bool,  # ✅ add
    timesfm_repo_id: str,
    timesfm_freq_category: int,
    timesfm_horizon_len: int,
    timesfm_context_cap: int,
    timesfm_per_core_batch_size: int,
    plot_feature_index: int,  # <--- ADD THIS PARAMETER
    tensor_time_axis_dates: Optional[np.ndarray] = None,
    xgboost_eligible_by_date: Optional[Dict[pd.Timestamp, set[str]]] = None,
    override_predicted_time_embedding_for_test: Optional[torch.Tensor] = None,
):
    """
    Evaluate a single test step 'test_idx' (>= train_T+1).

    ### Note on leakage:
    Using analyst *consensus* at t for t+1 is not peeking into the future;
    consensus is compiled *before* the earnings print. So at step t+1,
    consensus is already known at time t. We therefore treat it as ground truth
    input (not a forecasted driver).
    """
    steps_ahead = test_idx - train_T
    if steps_ahead <= 0:
        return {"rmse": 0.0, "r2": 0.0, "r2_kelly": 0.0}, pd.DataFrame()

    model.eval()
    with torch.no_grad():
        if override_predicted_time_embedding_for_test is not None:
            # May7 fix: caller already computed e_hat_T from the corrected
            # length-p history [e_{T-p}, ..., e_{T-2}, e_hat_{T-1}].
            time_embedding_for_test = (
                override_predicted_time_embedding_for_test
            )  # torch.Tensor [R] float32; predicted embedding for test target T
        else:
            # Backward-compatible path (matches May6 behavior exactly).
            # history up to and INCLUDING validation time train_T to forecast test (train_T+1)
            history_time_indices = torch.arange(0, train_T + 1, device=device)
            time_embedding_for_test = model.predict_next_time_embedding_with_adapter(
                history_time_indices=history_time_indices
            )  # [R]

        # labels/masks
        slice_true_raw = full_raw_tensor[test_idx]  # [N, F] raw
        j = target_feature_index
        valid_target_mask = torch.isfinite(slice_true_raw[:, j])  # [N] bool
        if use_two_stage_forecast:
            # Stage-B consumes observed analyst consensus at t+1 as an input.
            # This is allowed because the consensus forecast for t+1 is formed
            # before the t+1 realization.
            valid_target_mask = valid_target_mask & torch.isfinite(
                slice_true_raw[:, consensus_feature_index]
            )  # [N] bool; EPS label and analyst-consensus input both finite
        valid_i = torch.nonzero(valid_target_mask, as_tuple=True)[0]  # [M]
        if valid_i.numel() == 0:
            return {"rmse": 0.0, "r2": 0.0, "r2_kelly": 0.0}, pd.DataFrame()

        test_date = None
        if tensor_time_axis_dates is not None:
            test_date = pd.Timestamp(tensor_time_axis_dates[test_idx]).normalize()

        if xgboost_eligible_by_date is not None:
            if test_date is None:
                raise ValueError(
                    "tensor_time_axis_dates is required when filtering to XGBoost eligibility."
                )
            eligible_tickers = xgboost_eligible_by_date.get(test_date, set())
            if not eligible_tickers:
                return {"rmse": 0.0, "r2": 0.0, "r2_kelly": 0.0}, pd.DataFrame()
            candidate_tickers = tickers_array[valid_i.cpu().numpy()]
            keep_np = np.array(
                [str(ticker).strip() in eligible_tickers for ticker in candidate_tickers],
                dtype=bool,
            )
            if not np.any(keep_np):
                return {"rmse": 0.0, "r2": 0.0, "r2_kelly": 0.0}, pd.DataFrame()
            keep = torch.from_numpy(keep_np).to(valid_i.device)
            valid_i = valid_i[keep]

        # consensus ground truth at test step (raw), used by Stage-B and baseline reporting.
        cons_raw = slice_true_raw[valid_i, consensus_feature_index].cpu()  # [M]
        mean_c = feat_means[consensus_feature_index].item()
        std_c = feat_stds[consensus_feature_index].item()
        cons_norm = (cons_raw - mean_c) / std_c  # [M] normalized analyst consensus

        if use_two_stage_forecast:
            # --- forecast all selected non-consensus features at t+1 ---
            regression_feature_indices_tensor = torch.tensor(
                regression_feature_indices_list,
                device=device,
                dtype=torch.long,
            )  # [K_sel]
            forecasted_selected_features_norm = model.forecast_features_matrix(
                predicted_time_embedding=time_embedding_for_test,  # [R]
                firm_index=valid_i.to(device),  # [M]
                feature_indices=regression_feature_indices_tensor,  # [K_sel]
            )  # [M,K_sel]

            # EPS regressor consumes features in original feature order, except the
            # consensus column is observed rather than forecasted:
            #   [X_hat[..., 0:consensus], X[..., consensus], X_hat[..., consensus+1:]]
            cons_norm_dev = cons_norm.to(device=device)
            features_before_consensus_mask = (
                regression_feature_indices_tensor < int(consensus_feature_index)
            )  # [K_sel] bool
            features_after_consensus_mask = (
                regression_feature_indices_tensor > int(consensus_feature_index)
            )  # [K_sel] bool
            regression_input_norm = torch.cat(
                [
                    forecasted_selected_features_norm[:, features_before_consensus_mask],
                    cons_norm_dev.unsqueeze(1),
                    forecasted_selected_features_norm[:, features_after_consensus_mask],
                ],
                dim=1,
            )  # [M,K_sel+1], ordered as [forecast 0:12, observed 12, forecast 13]
            reg_head = model._ensure_eps_regressor(regression_input_norm.size(1))
            pred_norm = reg_head(regression_input_norm).squeeze(1)  # [M]
        else:
            # direct EPS head
            j_idx_target = torch.full_like(valid_i, j)
            pred_norm = model(
                mode="forecast",
                predicted_time_embedding=time_embedding_for_test,
                firm_index=valid_i.to(device),
                feature_index=j_idx_target.to(device),
            )
        if not torch.isfinite(pred_norm).all():
            raise ValueError(
                f"evaluate_one_step produced non-finite EPS predictions for test_idx={test_idx}."
            )

        # de-normalize EPS
        mean_j = feat_means[j].item()
        std_j = feat_stds[j].item()
        y_pred = (pred_norm.cpu() * std_j) + mean_j  # [M]
        y_true = slice_true_raw[valid_i, j].cpu()  # [M]

        mse = nn.functional.mse_loss(y_pred, y_true).item()
        tickers_valid = tickers_array[valid_i.cpu().numpy()]
        # === NEW: 1-Step Rolling Forecast specifically for the plot feature ===
        j_plot = torch.full_like(valid_i, plot_feature_index)
        pred_plot_norm = model(
            mode="forecast",
            predicted_time_embedding=time_embedding_for_test,
            firm_index=valid_i.to(device),
            feature_index=j_plot.to(device),
        )
        # De-normalize and get raw truth for the plot feature
        mean_plot = feat_means[plot_feature_index].item()
        std_plot = feat_stds[plot_feature_index].item()
        plot_pred_raw = (pred_plot_norm.cpu() * std_plot) + mean_plot
        plot_true_raw = slice_true_raw[valid_i, plot_feature_index].cpu()
        # ======================================================================

        # === Build full per-feature panel for downstream Stage-B usage ===
        F_total = full_raw_tensor.shape[2]
        features_to_dump = [
            i for i in range(F_total) if i != consensus_feature_index
        ]

        dump_idx_dev = torch.tensor(
            features_to_dump, device=device, dtype=torch.long
        )  # [K]
        pred_dump_norm = model.forecast_features_matrix(
            predicted_time_embedding=time_embedding_for_test,
            firm_index=valid_i.to(device),
            feature_indices=dump_idx_dev,
        )  # [M, K] normalized
        pred_dump_norm_cpu = pred_dump_norm.cpu().numpy()  # [M, K]

        orig_t_idx = test_idx - 1  # >= train_T because steps_ahead > 0
        valid_i_cpu = valid_i.cpu()
        orig_raw_all = full_raw_tensor[orig_t_idx, valid_i_cpu, :].numpy()  # [M, F]
        feat_means_np = feat_means.numpy()
        feat_stds_np = feat_stds.numpy()

        dump_means = feat_means_np[features_to_dump]  # [K]
        dump_stds = feat_stds_np[features_to_dump]  # [K]
        pred_dump_raw_cpu = (
            pred_dump_norm_cpu * dump_stds[None, :] + dump_means[None, :]
        )

        out_cols = {
            "t": test_idx,
            "original_time_t": test_idx - 1,
            "target_time_tplus1": test_idx,
            "time": test_date if test_date is not None else pd.NaT,
            "firm": valid_i_cpu.numpy(),
            "Ticker": tickers_valid,
            "GroundTruth": y_true.numpy(),
            "Forecasted": y_pred.numpy(),
            "Consensus": cons_raw.numpy(),
            "PlotFeatureGroundTruth": plot_true_raw.numpy(),
            "PlotFeatureForecast": plot_pred_raw.numpy(),
        }
        for X in features_to_dump:
            out_cols[f"OrigFeature_{X}_t_raw"] = orig_raw_all[:, X]
            out_cols[f"OrigFeature_{X}_t_norm"] = (
                (orig_raw_all[:, X] - feat_means_np[X]) / feat_stds_np[X]
            )
        for k, X in enumerate(features_to_dump):
            out_cols[f"PredFeature_{X}_tplus1_norm"] = pred_dump_norm_cpu[:, k]
            out_cols[f"PredFeature_{X}_tplus1_raw"] = pred_dump_raw_cpu[:, k]

        out = pd.DataFrame(out_cols)

        return {
            "rmse": math.sqrt(mse),
            "r2": r2(y_true, y_pred),
            "r2_kelly": r2_kelly(y_true, y_pred),
        }, out


def _safe_r2_from_sse(sse: float, sum_y: float, sum_y2: float, n: int):
    """Compute R² and Kelly’s R² from aggregates; return (r2, r2_kelly)."""
    if n <= 0:
        return float("nan"), float("nan")
    # SS_tot = sum((y - mean)^2) = sum_y2 - sum_y^2 / n
    ss_tot = sum_y2 - (sum_y * sum_y) / max(n, 1)
    r2 = 1.0 - (sse / ss_tot) if ss_tot > 0 else float("nan")
    r2_k = 1.0 - (sse / sum_y2) if sum_y2 > 0 else float("nan")
    return r2, r2_k


# ---------- compute_epoch_forecast_aggregates (CHANGED to use G1+G2) ----------
@torch.no_grad()
def compute_epoch_forecast_aggregates(
    model,
    full_norm_dev: torch.Tensor,
    train_T: int,
    target_feature_index: int,
    use_two_stage_forecast: bool,
    regression_feature_indices_list: list[int],
    consensus_feature_index: int,
    device=None,
    use_timesfm: bool = True,
    timesfm_repo_id: str = "google/timesfm-2.0-500m-pytorch",
    timesfm_freq_category: int = 2,
    timesfm_horizon_len: int = 1,
    timesfm_context_cap: int = 2048,
    timesfm_per_core_batch_size: int = 32,
):
    """
    Returns (sse, sum_y, sum_y2, n) for EPS at validation time (normalized space).
    Uses forecasted selected features plus observed analyst consensus as the
    EPS-regressor input.
    """
    device = full_norm_dev.device if device is None else device

    # next time embedding predicts val at train_T
    history_time_indices = torch.arange(0, train_T, device=device)
    next_time_embedding = model.predict_next_time_embedding_with_adapter(
        history_time_indices=history_time_indices
    )  # [R]

    j = target_feature_index
    val_slice = full_norm_dev[train_T]  # [N,F] normalized
    valid_target_mask = torch.isfinite(val_slice[:, j])  # [N] bool
    if use_two_stage_forecast:
        valid_target_mask = valid_target_mask & torch.isfinite(
            val_slice[:, consensus_feature_index]
        )  # [N] bool; EPS label and analyst-consensus input both finite
    valid_firms = torch.nonzero(valid_target_mask, as_tuple=True)[0]
    if valid_firms.numel() == 0:
        return 0.0, 0.0, 0.0, 0

    y_true = val_slice[valid_firms, j]  # [M]

    if use_two_stage_forecast:
        regression_feature_indices_tensor = torch.tensor(
            regression_feature_indices_list, device=device, dtype=torch.long
        )  # [K_sel]
        forecasted_selected_features = model.forecast_features_matrix(
            predicted_time_embedding=next_time_embedding,
            firm_index=valid_firms,
            feature_indices=regression_feature_indices_tensor,
        )  # [M,K_sel]

        observed_consensus_norm = val_slice[
            valid_firms, consensus_feature_index
        ]  # [M] normalized
        features_before_consensus_mask = (
            regression_feature_indices_tensor < int(consensus_feature_index)
        )  # [K_sel] bool
        features_after_consensus_mask = (
            regression_feature_indices_tensor > int(consensus_feature_index)
        )  # [K_sel] bool
        regression_input = torch.cat(
            [
                forecasted_selected_features[:, features_before_consensus_mask],
                observed_consensus_norm.unsqueeze(1),
                forecasted_selected_features[:, features_after_consensus_mask],
            ],
            dim=1,
        )  # [M,K_sel+1], ordered as [forecast 0:12, observed 12, forecast 13]
        reg_head = model._ensure_eps_regressor(regression_input.size(1))
        y_pred = reg_head(regression_input).squeeze(1)  # [M]
    else:
        y_pred = model(
            mode="forecast",
            predicted_time_embedding=next_time_embedding,
            firm_index=valid_firms,
            feature_index=torch.full_like(valid_firms, j),
        )

    if not torch.isfinite(y_pred).all():
        raise ValueError(
            f"compute_epoch_forecast_aggregates produced non-finite EPS predictions for validation index {train_T}."
        )

    diff = y_pred - y_true
    sse = float(torch.sum(diff * diff).item())
    sum_y = float(torch.sum(y_true).item())
    sum_y2 = float(torch.sum(y_true * y_true).item())
    n = int(y_true.numel())
    return sse, sum_y, sum_y2, n


# ------------------------ plotting helpers ------------------------
def fill_nan_with_linear_interpolation_1d_numpy(
    raw_value_array_1d: np.ndarray,
) -> np.ndarray:
    """
    Fill NaN values in a 1-D numpy array using **linear interpolation**.

    This follows the common TimesFM usage guidance: fill missing values before forecasting.

    Parameters
    ----------
    raw_value_array_1d : np.ndarray
        Shape: [time]
        Meaning: one univariate time series, possibly with np.nan for missing values.

    Returns
    -------
    filled_value_array_1d : np.ndarray
        Shape: [time]
        Meaning:
          - internal NaNs are linearly interpolated,
          - leading / trailing NaNs are filled by the nearest observed endpoint,
          - if the whole series is NaN, we return all zeros.
    """
    # (shape) ensure 1-D
    if raw_value_array_1d.ndim != 1:
        raise ValueError(f"Expected 1-D array, got shape {raw_value_array_1d.shape}")

    # (shape) copy so we do not mutate the caller's array
    filled_value_array_1d = raw_value_array_1d.astype(np.float32).copy()  # [time]

    time_index_array_1d = np.arange(
        filled_value_array_1d.shape[0], dtype=np.float32
    )  # [time]
    valid_mask_array_1d = ~np.isnan(filled_value_array_1d)  # [time] bool

    if valid_mask_array_1d.sum() == 0:
        # Edge case: all values are missing.
        return np.zeros_like(filled_value_array_1d, dtype=np.float32)  # [time]

    valid_time_index_array_1d = time_index_array_1d[valid_mask_array_1d]  # [num_valid]
    valid_value_array_1d = filled_value_array_1d[valid_mask_array_1d]  # [num_valid]

    # np.interp does linear interpolation and fills outside-range by endpoints.
    filled_value_array_1d = np.interp(
        x=time_index_array_1d,  # [time]
        xp=valid_time_index_array_1d,  # [num_valid]
        fp=valid_value_array_1d,  # [num_valid]
    ).astype(
        np.float32
    )  # [time]

    return filled_value_array_1d  # [time]


class TimesFMOfficialZeroShotForecaster:
    """
    This class uses the official TimesFM wrapper for zero-shot forecasting.
    It does not use inner hacks. It uses the standard `.forecast()` function.
    """

    def __init__(
        self,
        timesfm_repo_id: str,
        timesfm_context_cap: int,
        timesfm_horizon_len: int,
        timesfm_per_core_batch_size: int,
        timesfm_freq_category: int,
        device: torch.device,
    ):
        # type: int; meaning: frequency category (0=high, 1=medium, 2=low/quarterly)
        self.timesfm_freq_category = int(timesfm_freq_category)

        # Build the official wrapper from HuggingFace
        # timesfm.TimesFm: The official Google function to load the model safely.
        self.official_timesfm_wrapper = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                # type: string; meaning: tells TimesFM whether to run on GPU or CPU
                backend=("gpu" if device.type == "cuda" else "cpu"),
                # type: int; meaning: how many series the GPU processes at the exact same time
                per_core_batch_size=int(timesfm_per_core_batch_size),
                # type: int; meaning: forecast horizon (how many steps into the future, we use 1)
                horizon_len=int(timesfm_horizon_len),
                # type: int; meaning: max history length allowed (usually 2048)
                context_len=int(timesfm_context_cap),
                # The next 5 settings are required constants for the 500m model architecture
                input_patch_len=32,
                output_patch_len=128,
                num_layers=50,
                model_dims=1280,
                use_positional_embedding=False,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=timesfm_repo_id),
        )

    def forecast_massive_batch(self, list_of_history_1d_arrays: list) -> np.ndarray:
        """
        Takes a huge list of 1D arrays (histories) and predicts the next step for all of them at once.
        """
        # type: int; meaning: How many total series (histories) we are predicting right now.
        total_series_count = len(list_of_history_1d_arrays)

        # If the list is empty, return an empty array safely
        if total_series_count == 0:
            # np.array: NumPy function to create an array. Shape: [0] (empty).
            return np.array([], dtype=np.float32)

        # type: Python list; shape: [total_series_count];
        # meaning: TimesFM requires a list of frequencies, one for every single item in the batch.
        frequency_list = [self.timesfm_freq_category] * total_series_count

        # self.official_timesfm_wrapper.forecast(): Official zero-shot inference function.
        # It takes our standard Python list of arrays, handles all complex padding automatically,
        # pushes it to the GPU, and returns the predictions. NO python for-loops are used inside.
        # type: Tuple; meaning: returns (forecasts_array, extra_info_dict). We only want the first part.
        forecast_results, _ = self.official_timesfm_wrapper.forecast(
            inputs=list_of_history_1d_arrays, freq=frequency_list
        )

        # forecast_results shape: [total_series_count, horizon_len].
        # forecast_results[:, 0]: Extracts all rows, but only the first prediction step (T+1).
        # .astype(np.float32): Converts to standard 32-bit float numbers to save memory.
        # type: 1D NumPy array; shape: [total_series_count].
        final_predictions_1d_array = forecast_results[:, 0].astype(np.float32)

        return final_predictions_1d_array


def run_vanilla_timesfm_baseline_expanding_window(
    full_raw_value_tensor_time_firm_feature: np.ndarray,
    forecast_time_index_array_1d: np.ndarray,
    firm_index_list_per_time: list,
    timesfm_official_forecaster: TimesFMOfficialZeroShotForecaster,
    plot_feature_index: int,
    target_feature_index: int,  # <--- ADDED TARGET INDEX
) -> Tuple[list, list]:  # <--- NOW RETURNS TWO LISTS
    """
    Vanilla approach: For time T, build a list of [Firm1_F1, Firm1_F2... FirmM_F14],
    predict all at once, then extract both the plot and target feature indices.
    """
    num_time_total = int(full_raw_value_tensor_time_firm_feature.shape[0])
    num_firm_total = int(full_raw_value_tensor_time_firm_feature.shape[1])
    num_feature_total = int(full_raw_value_tensor_time_firm_feature.shape[2])

    print(
        "[TimesFM Baseline] Running expanding window. Extracting Plot and Target features simultaneously..."
    )

    list_of_plot_predictions = []
    list_of_target_predictions = []

    for step_index, target_time_T in enumerate(
        tqdm(
            forecast_time_index_array_1d.tolist(),
            desc="TimesFM Baseline OOS",
            unit="step",
        )
    ):

        valid_firms_array_1d = firm_index_list_per_time[step_index]
        count_of_valid_firms = len(valid_firms_array_1d)

        if target_time_T <= 0 or count_of_valid_firms == 0:
            empty_predictions_array = np.full(
                (count_of_valid_firms,), np.nan, dtype=np.float32
            )
            list_of_plot_predictions.append(empty_predictions_array)
            list_of_target_predictions.append(empty_predictions_array)
            continue

        raw_history_block_3d_array: np.ndarray = (
            full_raw_value_tensor_time_firm_feature[
                0:target_time_T, valid_firms_array_1d, :
            ]
        )
        history_block_3d_array: np.ndarray = np.empty_like(raw_history_block_3d_array)

        for current_feature_index in trange(num_feature_total):
            history_block_3d_array[:, :, current_feature_index] = (
                fill_nan_with_linear_interpolation_2d_numpy_per_column(
                    raw_history_block_3d_array[:, :, current_feature_index]
                )
            )

        history_transposed_3d_array: np.ndarray = np.transpose(
            history_block_3d_array, (1, 2, 0)
        )
        history_flattened_2d_array: np.ndarray = np.reshape(
            history_transposed_3d_array,
            (count_of_valid_firms * num_feature_total, target_time_T),
        )

        history_batch_list: list = list(history_flattened_2d_array)
        all_features_predictions_1d_array: np.ndarray = (
            timesfm_official_forecaster.forecast_massive_batch(history_batch_list)
        )

        predictions_organized_2d_array: np.ndarray = np.reshape(
            all_features_predictions_1d_array, (count_of_valid_firms, num_feature_total)
        )

        # --- EXTRACT BOTH SIMULTANEOUSLY (NO DOUBLE COMPUTE) ---
        plot_predictions_array_1d: np.ndarray = predictions_organized_2d_array[
            :, plot_feature_index
        ]
        target_predictions_array_1d: np.ndarray = predictions_organized_2d_array[
            :, target_feature_index
        ]

        list_of_plot_predictions.append(plot_predictions_array_1d)
        list_of_target_predictions.append(target_predictions_array_1d)

    return list_of_plot_predictions, list_of_target_predictions


def fill_nan_with_linear_interpolation_2d_numpy_per_column(
    raw_value_matrix_time_firm: np.ndarray,
) -> np.ndarray:
    """
    Fill NaNs in a 2-D matrix by applying 1-D linear interpolation *per column*.

    Input:
        raw_value_matrix_time_firm: [time, num_firms]
    Output:
        filled_value_matrix_time_firm: [time, num_firms]
    """
    if raw_value_matrix_time_firm.ndim != 2:
        raise ValueError(
            f"Expected 2-D matrix [time, firm], got {raw_value_matrix_time_firm.shape}"
        )

    num_time_steps = int(raw_value_matrix_time_firm.shape[0])  # (int) T
    num_firms = int(raw_value_matrix_time_firm.shape[1])  # (int) N

    filled_value_matrix_time_firm = np.empty(
        (num_time_steps, num_firms), dtype=np.float32
    )  # [T,N]
    for firm_index in range(num_firms):
        raw_series_value_array_1d = raw_value_matrix_time_firm[:, firm_index].astype(
            np.float32
        )  # [T]
        filled_series_value_array_1d = fill_nan_with_linear_interpolation_1d_numpy(
            raw_value_array_1d=raw_series_value_array_1d
        )  # [T]
        filled_value_matrix_time_firm[:, firm_index] = (
            filled_series_value_array_1d  # write back
        )
    return filled_value_matrix_time_firm  # [T,N]


# ------------------------ main ------------------------
def main():
    parser = argparse.ArgumentParser()
    # data / model
    parser.add_argument("--epochs_per_window", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4096 * 32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--Rank_CP", type=int, default=100)
    parser.add_argument("--mlp_hidden", type=int, default=1024)
    parser.add_argument("--dropout_p", type=float, default=0.1)  # dropout 0.1
    parser.add_argument("--target_feature_index", type=int, default=13)  # EPS column
    parser.add_argument(
        "--consensus_feature_index",
        type=int,
        default=12,
        help="Column index of the analyst consensus forecast",
    )
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="EPS_gru_debug_trainable_weights_freeze_forecast_Aug_28",
    )
    parser.add_argument("--num_workers", type=int, default=16)
    # ===== Loss weights (either fixed or trainable softmax over 4 components) =====
    parser.add_argument(
        "--w_recon",
        type=float,
        default=0.0,
        help="Fixed weight for reconstruction_loss (if not using --trainable_weights)",
    )
    parser.add_argument(
        "--w_predict",
        type=float,
        default=0.10,
        help="Fixed weight for prediction_loss (if not using --trainable_weights)",
    )
    parser.add_argument(
        "--w_forecast_drivers",
        type=float,
        default=0.50,
        help="Fixed weight for drivers_forecast_loss at t+1 (if not using --trainable_weights)",
    )
    parser.add_argument(
        "--w_regression",
        type=float,
        default=1.00,
        help="Fixed weight for regression_loss (EPS from drivers) at t+1 (if not using --trainable_weights)",
    )

    parser.add_argument(
        "--trainable_weights",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, learn 4-way softmax over losses.",
    )
    parser.add_argument(
        "--use_two_stage_forecast",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, Stage-A drivers + Stage-B EPS regression.",
    )
    parser.add_argument(
        "--driver_feature_indices",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9,10,11,12",
        help=(
            "Legacy comma-separated driver indices used for config compatibility. "
            "The local rolling forecast/regression losses are controlled by "
            "--selected_feature_indices."
        ),
    )

    # Local rolling-window context length p.
    # The training loop now iterates current_time_index in
    #     range(context_length, number_of_time_periods - 1)
    # and at each i:
    #     - reconstruction target window: tensor[i-p+1 : i+1, :, :]   (length p)
    #     - prediction context window  : tensor[i-p+1 : i+1, :, :]   (length p) -> i+1
    #     - forecast target            : tensor[i+1, :, :]
    #     - regression target          : tensor[i+1, firm, eps]
    # We need exactly p history steps ending at i. With the current indexing
    # convention, the first supervised instance is:
    #     input  = [1, ..., p]
    #     target = p + 1
    # so i starts at p.
    parser.add_argument(
        "--minimum_context_length",
        type=int,
        default=20,
        help=(
            "Local context length p. At each current_time_index i in "
            "[context_length, number_of_time_periods-2], we use history window "
            "[i-p+1 : i+1] and forecast at i+1."
        ),
    )

    # ===================== NEW: local-rolling minibatch sampling =====================
    parser.add_argument(
        "--reconstruction_batch_size",
        type=int,
        default=4096,
        help=(
            "How many (firm, feature) fibers to sample per reconstruction "
            "minibatch. Each sampled fiber reconstructs all ordered local time "
            "positions in [i-p+1 : i+1]."
        ),
    )
    parser.add_argument(
        "--number_of_reconstruction_minibatches",
        type=int,
        default=4,
        help="How many reconstruction minibatches to draw per current_time_index i.",
    )
    parser.add_argument(
        "--forecast_batch_size",
        type=int,
        default=4096,
        help=(
            "How many (firm, feature) pairs to sample per forecast minibatch at i+1; "
            "the same firm batch is reused for the EPS regression loss at i+1."
        ),
    )
    parser.add_argument(
        "--number_of_forecast_minibatches",
        type=int,
        default=4,
        help="How many forecast/regression minibatches to draw per current_time_index i.",
    )
    parser.add_argument(
        "--selected_feature_indices",
        type=str,
        default="all_except_consensus",
        help=(
            "Comma-separated feature indices used for reconstruction sampling AND "
            "forecast sampling, or 'all_except_consensus'. Default forecasts every "
            "tensor feature except the analyst-consensus feature."
        ),
    )

    # split control
    parser.add_argument(
        "--start_split_ratio",
        type=float,
        default=0.8,
        help="Initial train length ratio; e.g., 0.8 for 80% train.",
    )
    parser.add_argument(
        "--reset_each_window",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, reinit model+optim each window.",
    )
    parser.add_argument(
        "--recompute_norm_each_window",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, recompute winsor/stats each window.",
    )
    parser.add_argument(
        "--use_timesfm",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="If true, use TimesFM zero-shot for the next time embedding (and skip GRU prediction loss).",
    )
    parser.add_argument(
        "--timesfm_repo_id",
        type=str,
        default="google/timesfm-2.0-500m-pytorch",
        help="Hugging Face repo id for the PyTorch TimesFM-2.0 500M checkpoint.",
    )
    parser.add_argument(
        "--timesfm_horizon_len",
        type=int,
        default=1,
        help="How many steps ahead TimesFM predicts each call (we use 1).",
    )
    parser.add_argument(
        "--timesfm_context_cap",
        type=int,
        default=2048,
        help="Max history length sent to TimesFM (model was trained up to 2048).",
    )
    parser.add_argument(
        "--timesfm_per_core_batch_size",
        type=int,
        default=32,
        help="Internal TimesFM mini-batch size.",
    )
    parser.add_argument(
        "--timesfm_freq_category",
        type=int,
        default=2,
        help="Frequency: 0=high (hour/day), 1=week/month, 2=quarter/year. EPS quarterly → 2.",
    )
    # ADD ↓↓↓ (after existing args)
    parser.add_argument(
        "--lambda_ortho_time",
        type=float,
        default=1e-4,  # small; sweep over [1e-6, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3]
        help="Weight for time-embedding orthogonality penalty. Recommend sweep in [1e-6 ... 1e3].",
    )
    parser.add_argument(
        "--ortho_penalty_norm",
        type=str,
        default="L2",  # "L2" or "L1"
        choices=["L2", "L1"],
        help="Which decorrelation norm to use for time embeddings (L2 or L1).",
    )
    # --- Frobenius-norm regularizer on the time-embedding matrix E ---
    # Complements the L2/L1 off-diagonal covariance penalty. At high rank_CP
    # (e.g. 200), the off-diag penalty alone scales with ||E||, so the optimizer
    # can game it by letting ||E|| drift upward. Adding λ_F * mean(E**2) bounds
    # ||E||, which "tightens" the Frobenius norm of E while the L2/L1 penalty
    # continues to decorrelate directions. Together they enable safe
    # rank_CP=200 double-descent runs.
    parser.add_argument(
        "--lambda_frobenius_time",
        type=float,
        default=0.0,  # off by default; sweep in [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        help=(
            "Weight for Frobenius-norm regularizer on time embeddings: "
            "λ_F * mean(E**2). Bounds ||E|| so the L2/L1 off-diagonal "
            "covariance penalty is not gamed by scale drift. Set 0 to disable."
        ),
    )
    # ---- data file paths (NEW) ----
    parser.add_argument(
        "--tensor_npy_path",  # str flag written on CLI
        type=str,
        default="./fundamentals_analyst_forecast_EPS_regression.npy",
        help="Path to 3-D tensor .npy of shape [time, firm, feature].",
    )
    parser.add_argument(
        "--ticker_sidecar_npy_path",  # str flag written on CLI
        type=str,
        default="./fundamentals_analyst_forecast_EPS_regression_ticker.npy",
        help="Path to 1-D ticker .npy with shape [firm].",
    )
    parser.add_argument(
        "--xgboost_eval_csv_path",
        type=str,
        default="./fundamentals_analyst_forecast_EPS_regression_ready.csv",
        help=(
            "CSV used to reproduce XGBoost row eligibility for apples-to-apples "
            "Tensor-TimesFM vs XGBoost OOS metrics."
        ),
    )
    parser.add_argument(
        "--filter_oos_to_xgboost_eligible",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help=(
            "If true, filter Tensor-TimesFM OOS rows to the exact (date,ticker) "
            "rows kept by xgboost_expanding_1.py after lag-missing drops."
        ),
    )

    # ---- plotting (NEW) ----
    parser.add_argument(
        "--make_fiber_forecast_plot",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="If true, after rolling evaluation, save a 3-line plot: (1) raw fiber, (2) TimesFM baseline, (3) Tensor-TimesFM.",
    )
    parser.add_argument(
        "--plot_firm_index",
        type=int,
        default=0,
        help="0-based firm index for plotting. 'firm 1' in your description corresponds     to index 0 here.",
    )
    parser.add_argument(
        "--plot_feature_index",
        type=int,
        default=0,
        help="0-based feature index to plot (e.g., 0 to 13). You can plot ANY feature, not just the target.",
    )
    parser.add_argument(
        "--plot_png_filename",
        type=str,
        default="",
        help="Optional filename for the PNG. If empty, we auto-name it inside output_dir.",
    )

    parser.add_argument(
        "--plot_all_firms",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="If true, plot a multi-firm fiber (time-major: all valid firms inside each time). Overrides --plot_firm_index for plotting.",
    )
    parser.add_argument(
        "--timesfm_baseline_decode_batch_size",
        type=int,
        default=256,
        help="Batch size used when calling TimesFM decode() for the TimesFM baseline plot line.",
    )
    parser.add_argument(
        "--timesfm_baseline_use_gpu",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, run TimesFM baseline on GPU for the plot (faster but uses VRAM). Default False to avoid OOM.",
    )
    parser.add_argument(
        "--timesfm_baseline_forecast_all_features",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help=(
            "If true, the TimesFM baseline for the multi-firm fiber plot will *compute* forecasts for ALL features "
            "(feature dimension) in batches over firms, but only *plot* the selected plot_feature_index. "
            "This matches the advisor requirement: firm-batched, univariate per feature, expanding window. "
            "If false, we only forecast plot_feature_index (faster). Default True."
        ),
    )
    # ADD ↓↓↓ (seed argument for reproducible math)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for torch, numpy, and random to ensure reproducibility.",
    )  # prints: none; meaning: accept an integer seed from the terminal commands
    # ADD ↑↑↑
    # ADD ↑↑↑

    args = parser.parse_args()
    # ADD ↓↓↓ (unique run hash to suffix CSVs)
    # make a small dict out of args for hashing (all keys sorted for stability)
    global g_dl
    g_dl = set_seed(args.seed)
    args_as_dict_for_hash = {
        k: str(getattr(args, k)) for k in sorted(vars(args).keys())
    }  # dict[str->str]
    # include wall-clock seconds to avoid accidental collisions across separate runs
    args_as_dict_for_hash["_timestamp_unix"] = str(
        int(time.time())
    )  # str seconds since epoch
    # json.dumps(...).encode(): bytes; hashlib.sha1(...).hexdigest(): str hex; [:8]: short 8-char tag
    run_hash = hashlib.sha1(
        json.dumps(args_as_dict_for_hash, sort_keys=True).encode()
    ).hexdigest()[
        :8
    ]  # 'a1b2c3d4'
    print(f"[INFO] run_hash = {run_hash}")  # human-friendly print
    # keep in args for reuse later
    args.run_hash = run_hash  # attach to args
    # ADD ↑↑↑

    # # -------- [W&B] init (sweep-safe) --------
    # run = wandb.init(  # (no shape) / type: wandb Run / why: start a W&B run
    #     project=os.environ.get(
    #         "WANDB_PROJECT", "eps-sundial-tensor"
    #     ),  # (str) / which project in W&B
    #     mode=os.environ.get(
    #         "WANDB_MODE", "online"
    #     ),  # (str) / "online" or "offline" or "disabled"
    #     config=vars(
    #         args
    #     ),  # (dict[str->value]) / give argparse defaults; sweeps will override
    # )
    # cfg = run.config  # (mapping-like) / sweep-merged config lives here
    
    # -------- [W&B] init (sweep-safe) --------

    wandb_name_from_env = os.environ.get("WANDB_NAME")
    wandb_group_from_env = os.environ.get("WANDB_RUN_GROUP")

    wandb_init_kwargs = {
        "project": os.environ.get("WANDB_PROJECT", "eps-sundial-tensor"),
        "mode": os.environ.get("WANDB_MODE", "online"),
        "config": vars(args),
    }

    if wandb_name_from_env:
        wandb_init_kwargs["name"] = wandb_name_from_env

    if wandb_group_from_env:
        wandb_init_kwargs["group"] = wandb_group_from_env

    run = wandb.init(**wandb_init_kwargs)

    cfg = run.config

    def _coerce_type(dst_val, src_val):
        # keep bools sane (avoid bool("0") == True surprises)
        if isinstance(dst_val, bool):
            return str2bool(src_val) if isinstance(src_val, str) else bool(src_val)
        # ints can arrive as "3" or 3.0; allow float->int as well
        if isinstance(dst_val, int) and not isinstance(src_val, bool):
            return int(float(src_val))
        if isinstance(dst_val, float):
            return float(src_val)
        return src_val

    # copy sweep-merged config back into argparse, so the rest of your code uses the right values
    for key in cfg.keys():  # (iterable of str) / loop over keys in W&B config
        if hasattr(args, key):
            current = getattr(args, key)  # (bool) / only copy known argparse fields
            setattr(
                args, key, _coerce_type(current, cfg[key])
            )  # (no shape) / set args.key = sweep value (or default)

    # ---- TimesFM path: we still allow a prediction loss via the adapter layer ----
    # (TimesFM stays frozen and NOT detached; gradients will flow to time_embeddings and the adapter.)

    # # optional naming/grouping from env (purely cosmetic in UI)
    # wandb_name_from_env = os.environ.get(
    #     "WANDB_NAME"
    # )  # (str or None) / human-friendly run name
    # wandb_group_from_env = os.environ.get(
    #     "WANDB_RUN_GROUP"
    # )  # (str or None) / group name for runs
    # if wandb_name_from_env:
    #     wandb.run.name = wandb_name_from_env  # (no shape) / apply if provided
    # if wandb_group_from_env:
    #     wandb.run.group = wandb_group_from_env  # (no shape) / apply if provided
    # ------------------------------------------
    # ---- (optional) define a global step and metrics tied to it ----
    wandb.define_metric(
        "global_step"
    )  # (no shape) / type: define / why: create a step axis
    wandb.define_metric(
        "*", step_metric="global_step"
    )  # (no shape) / type: define / why: all metrics use that step

    wandb.log({"hparams/epochs_per_window": int(args.epochs_per_window)}, step=0)
    # ----------------- parse & CHECK drivers vs consensus -----------------
    drivers_all = [
        int(x.strip())
        for x in args.driver_feature_indices.split(",")
        if x.strip() != ""
    ]
    print(f"[CHECK] driver_feature_indices (all): {drivers_all}")
    print(f"[CHECK] consensus_feature_index: {args.consensus_feature_index}")

    if args.consensus_feature_index not in drivers_all:
        raise ValueError(
            f"consensus_feature_index={args.consensus_feature_index} must be in driver_feature_indices={drivers_all}"
        )
    # G1 = all but consensus
    drivers_no_consensus = [d for d in drivers_all if d != args.consensus_feature_index]
    print(
        f"[CHECK] G1 (no consensus) size={len(drivers_no_consensus)} indices={drivers_no_consensus}"
    )
    print(
        f"[CHECK] Legacy G1 (no consensus) size={len(drivers_no_consensus)}; "
        "forecast/regression feature set is resolved from --selected_feature_indices "
        "after tensor shape is known."
    )

    # ============================================================================
    # NEW: keep selected_feature_indices as a string for now.
    # It may be the symbolic value "all_except_consensus", which requires D2
    # (the tensor feature dimension) and is resolved immediately after tensor load.
    # ============================================================================
    selected_feature_indices_spec = str(args.selected_feature_indices).strip()
    print(f"[CHECK] selected_feature_indices spec: {selected_feature_indices_spec}")

    # ============================================================================
    # NEW: hard-code weight_prediction = 0.0 for this experiment.
    # The prediction loss path (TimesFM + 2-layer MLP -> next time embedding)
    # stays in the model and is still computed for logging/debugging, but it
    # contributes ZERO to the gradient update. We also forbid trainable_weights
    # because softmax cannot represent a hard zero.
    # ============================================================================
    weight_prediction = 0.0  # type: float; LOCKED for this experiment
    if abs(weight_prediction) > 1e-12:
        raise ValueError(
            "For this experiment, weight_prediction must be hard-coded to 0."
        )
    if bool(args.trainable_weights):
        raise ValueError(
            "trainable_weights=True is incompatible with the hard-coded "
            "weight_prediction=0.0 used in this experiment. Re-run with "
            "--trainable_weights=False."
        )
    if abs(float(args.w_predict)) > 1e-12:
        print(
            f"[INFO] CLI passed --w_predict={args.w_predict}, but weight_prediction "
            f"is hard-coded to 0.0 for this experiment; CLI value ignored."
        )
    args.w_predict = 0.0  # keep argparse / W&B in sync with the hard-coded value
    # --- make a filesystem-safe tag from the *tensor* file name ---
    tensor_tag = Path(args.tensor_npy_path).name  # e.g. "fundamentals.npy"
    tensor_tag = re.sub(
        r"(\.npy(\.gz)?|\.npz)$", "", tensor_tag, flags=re.IGNORECASE
    )  # strip known suffixes
    tensor_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", tensor_tag).strip(
        "_"
    )  # keep it folder-safe

    # --- include tensor_tag as a subdir under args.output_dir ---
    run_output_dir = os.path.join(
        args.output_dir,
        tensor_tag,
        (
            "wr_{wr}_wp_{wp}_wdrv_{wdrv}_wreg_{wreg}_lambda_{lam}".format(
                wr=_fmt(args.w_recon),
                wp=_fmt(args.w_predict),
                wdrv=_fmt(args.w_forecast_drivers),
                wreg=_fmt(args.w_regression),
                lam=_fmt(args.lambda_ortho_time),
            )
        ),
    )

    os.makedirs(run_output_dir, exist_ok=True)
    args.output_dir = run_output_dir
    print(f"[INFO] Outputs will be saved under: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Paths for NEW tensor + ticker sidecar ----
    # resolved input paths from CLI
    tensor_npy_path: str = args.tensor_npy_path  # str; path to [T,N,F] tensor
    ticker_sidecar_npy_path: str = (
        args.ticker_sidecar_npy_path
    )  # str; path to [N] tickers

    if not os.path.exists(tensor_npy_path):
        sys.exit(f"Error: {tensor_npy_path} not found")
    if not os.path.exists(ticker_sidecar_npy_path):
        sys.exit(f"Error: {ticker_sidecar_npy_path} not found")

    # load data
    full_raw_cpu = torch.from_numpy(np.load(tensor_npy_path)).float()  # [T,N,F] float32
    tickers = np.load(ticker_sidecar_npy_path, allow_pickle=True)  # [N] object/str
    tickers = np.array([str(ticker).strip() for ticker in tickers], dtype=object)

    if tickers.ndim != 1 or tickers.shape[0] != full_raw_cpu.shape[1]:
        sys.exit(
            f"Ticker sidecar shape mismatch: got {tickers.shape}, expected length {full_raw_cpu.shape[1]}"
        )  # prints: msg; why: safety

    # ---- Load raw (keep on device for math, but quantiles computed on CPU) ----

    # trial subset: keep first 50 timesteps for quick validation
    trail = full_raw_cpu.shape[0]
    # trail = 50
    full_raw_cpu = (
        full_raw_cpu[:trail] if trail < full_raw_cpu.shape[0] else full_raw_cpu
    )
    nT, D1, D2 = full_raw_cpu.shape
    full_raw_dev = full_raw_cpu.to(device, non_blocking=True)  # on device for fast math
    print(f"Loaded raw tensor shape: {full_raw_cpu.shape}")

    # Resolve the feature set used by the local rolling reconstruction and
    # next-slice forecast losses. The default is deliberately tied to the actual
    # tensor width D2 so EPS/target columns are included when they are not the
    # analyst-consensus column.
    if selected_feature_indices_spec.lower() in {
        "",
        "auto",
        "all_except_consensus",
    }:
        selected_feature_indices_list = [
            feature_index
            for feature_index in range(D2)
            if feature_index != int(args.consensus_feature_index)
        ]
    else:
        selected_feature_indices_list = [
            int(token.strip())
            for token in selected_feature_indices_spec.split(",")
            if token.strip() != ""
        ]

    if len(selected_feature_indices_list) == 0:
        raise ValueError(
            "--selected_feature_indices resolved to an empty list; refusing to train."
        )
    duplicate_selected_features = sorted(
        {
            feature_index
            for feature_index in selected_feature_indices_list
            if selected_feature_indices_list.count(feature_index) > 1
        }
    )
    if duplicate_selected_features:
        raise ValueError(
            f"--selected_feature_indices contains duplicates: {duplicate_selected_features}"
        )
    out_of_range_selected_features = [
        feature_index
        for feature_index in selected_feature_indices_list
        if feature_index < 0 or feature_index >= D2
    ]
    if out_of_range_selected_features:
        raise ValueError(
            f"--selected_feature_indices contains feature indices outside [0, {D2 - 1}]: "
            f"{out_of_range_selected_features}"
        )
    if int(args.consensus_feature_index) in selected_feature_indices_list:
        raise ValueError(
            f"consensus_feature_index={args.consensus_feature_index} must not appear "
            f"in selected_feature_indices={selected_feature_indices_list}."
        )
    if int(args.target_feature_index) in selected_feature_indices_list:
        print(
            f"[CHECK] target_feature_index={args.target_feature_index} is included in "
            "selected_feature_indices, so the forecast loss trains the direct EPS "
            "fiber forecast as requested."
        )

    args.selected_feature_indices = ",".join(
        str(feature_index) for feature_index in selected_feature_indices_list
    )
    print(
        "[CHECK] selected_feature_indices resolved for reconstruction + next-slice "
        f"forecast loss: size={len(selected_feature_indices_list)} "
        f"indices={selected_feature_indices_list}"
    )
    print(
        "[CHECK] EPS regressor input uses forecasted selected features plus "
        f"observed consensus: K={len(selected_feature_indices_list) + 1}"
    )
    wandb.config.update(
        {
            "selected_feature_indices_resolved": args.selected_feature_indices,
            "selected_feature_policy": "all_except_consensus"
            if selected_feature_indices_spec.lower()
            in {"", "auto", "all_except_consensus"}
            else "explicit",
            "eps_regressor_input_dim": len(selected_feature_indices_list) + 1,
        },
        allow_val_change=True,
    )

    if bool(args.filter_oos_to_xgboost_eligible) and not os.path.exists(
        args.xgboost_eval_csv_path
    ):
        sys.exit(f"Error: {args.xgboost_eval_csv_path} not found")

    tensor_time_axis_dates = None
    if os.path.exists(args.xgboost_eval_csv_path):
        tensor_time_axis_dates = infer_tensor_time_axis_dates_from_csv(
            args.xgboost_eval_csv_path
        )
        if len(tensor_time_axis_dates) < nT:
            sys.exit(
                "XGBoost eval CSV has fewer quarterly dates than the tensor time axis: "
                f"{len(tensor_time_axis_dates)} < {nT}"
            )
        tensor_time_axis_dates = tensor_time_axis_dates[:nT]

    xgboost_eligible_by_date = None
    if bool(args.filter_oos_to_xgboost_eligible):
        xgboost_eligible_by_date = build_xgboost_eligibility_by_date(
            args.xgboost_eval_csv_path
        )
        total_eligible_pairs = sum(
            len(tickers_for_date)
            for tickers_for_date in xgboost_eligible_by_date.values()
        )
        print(
            "[INFO] Tensor-TimesFM OOS metrics will be filtered to "
            f"XGBoost-eligible rows from {args.xgboost_eval_csv_path} "
            f"({total_eligible_pairs} date/ticker pairs before OOS split)."
        )
    else:
        print("[INFO] Tensor-TimesFM OOS metrics use all rows with observed target.")

    # ---- Determine initial window indexes ----
    start_split = int(args.start_split_ratio * nT)  # e.g., 40 when nT=50 and ratio=0.8
    # We will iterate train_end from (start_split-1) up to (nT-3) so that test = train_end+2 <= nT-1
    first_train_end = start_split - 1
    last_train_end = nT - 3
    if first_train_end < 1 or last_train_end < first_train_end:
        sys.exit("Sequence too short for rolling windows with given split ratio.")

    xgboost_expected_oos_pairs = None
    if xgboost_eligible_by_date is not None and tensor_time_axis_dates is not None:
        ticker_set_for_tensor = set(str(ticker).strip() for ticker in tickers.tolist())
        test_index_range = range(first_train_end + 2, last_train_end + 3)
        xgboost_expected_oos_pairs = 0
        for test_time_index in test_index_range:
            test_date = pd.Timestamp(
                tensor_time_axis_dates[test_time_index]
            ).normalize()
            xgboost_expected_oos_pairs += len(
                xgboost_eligible_by_date.get(test_date, set())
                & ticker_set_for_tensor
            )
        print(
            "[AUDIT] Expected Tensor-TimesFM OOS rows after XGBoost eligibility "
            f"filter: {xgboost_expected_oos_pairs}"
        )

    # ---- Helper: compute stats (winsorize train slice only), return (feat_means, feat_stds) on device ----
    def compute_stats_from_train_slice(train_end_inclusive: int):
        train_raw = full_raw_cpu[: train_end_inclusive + 1]  # CPU tensor
        train_vals = train_raw[torch.isfinite(train_raw)].numpy()
        if train_vals.size == 0:
            # degenerate; fallback to zeros/ones
            means = torch.zeros(D2, dtype=torch.float32, device=device)
            stds = torch.ones(D2, dtype=torch.float32, device=device)
            return means, stds
        lower = np.quantile(train_vals, 0.01)
        upper = np.quantile(train_vals, 0.99)
        train_wins = torch.clamp(train_raw, min=lower, max=upper).numpy()
        feat_stds = torch.from_numpy(np.nanstd(train_wins, axis=(0, 1))).float()
        feat_means = torch.from_numpy(np.nanmean(train_wins, axis=(0, 1))).float()
        missing_mean_mask = ~torch.isfinite(feat_means)
        missing_std_mask = (~torch.isfinite(feat_stds)) | (feat_stds <= 0)
        if torch.any(missing_mean_mask):
            missing_features = torch.nonzero(missing_mean_mask, as_tuple=True)[0].tolist()
            print(
                f"[WARN] normalization mean missing for feature indices {missing_features}; using 0.0."
            )
            feat_means[missing_mean_mask] = 0.0
        if torch.any(missing_std_mask):
            missing_features = torch.nonzero(missing_std_mask, as_tuple=True)[0].tolist()
            print(
                f"[WARN] normalization std missing/zero for feature indices {missing_features}; using 1.0."
            )
            feat_stds[missing_std_mask] = 1.0
        return feat_means.to(device), feat_stds.to(device)

    # ---- Initial stats (fixed by default) ----
    init_means_dev, init_stds_dev = compute_stats_from_train_slice(first_train_end)
    full_norm_dev = (full_raw_dev - init_means_dev[None, None, :]) / init_stds_dev[
        None, None, :
    ]

    # ---- Model + optimizer (init once; we update model.train_T per window) ----
    def build_tensor_timesfm():
        model = TensorTimesFM(
            full_raw_cpu.shape,
            args.Rank_CP,
            mlp_hidden=args.mlp_hidden,
            dropout_p=args.dropout_p,
            device=device,
            use_timesfm=args.use_timesfm,
            timesfm_repo_id=args.timesfm_repo_id,
            timesfm_horizon_len=args.timesfm_horizon_len,
            timesfm_context_cap=args.timesfm_context_cap,
            timesfm_per_core_batch_size=args.timesfm_per_core_batch_size,
            timesfm_freq_category=args.timesfm_freq_category,
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        return model, optimizer

    (
        model,
        optimizer,
    ) = (
        build_tensor_timesfm()
    )  # prints: (model, opt); why: initial objects (opt will be replaced below anyway)
    model.to(device)  # prints: none;    why: move to GPU/CPU

    # Light watch; gradients can be noisy with torch.compile
    wandb.watch(model, log=None)

    # Ensure the EPS regressor exists BEFORE optimizer/compile.
    # Training Step 3 uses [forecasted selected features at i+1 | observed
    # analyst consensus at i+1] -> EPS, so these parameters must be present
    # before optimizer construction.
    model._ensure_eps_regressor(
        input_dim_drivers=len(selected_feature_indices_list) + 1
    )

    model = torch.compile(  # prints: none;          why: compile AFTER regressor exists
        model, mode="reduce-overhead"
    )

    # ===== Trainable loss weights (trainer side, now 4 components) =====
    if args.trainable_weights:
        loss_logits = nn.Parameter(
            torch.zeros(4, device=device)
        )  # prints: [4]; meaning: learn mixture over (recon, predict, drivers_forecast, regression)
    else:
        loss_logits = None  # prints: None; meaning: use fixed weights

    def current_loss_weights():
        if args.trainable_weights:
            return torch.softmax(loss_logits, dim=0)
        else:
            w = [
                args.w_recon,
                args.w_predict,
                args.w_forecast_drivers,
                args.w_regression,
            ]
            w = [float(x) for x in w]  # <— coerce any "1e-3" strings to float
            return torch.tensor(w, device=device, dtype=torch.float32)

    def build_optimizer(m: nn.Module):
        """
        Builds Adam over model parameters (+ loss_logits if trainable).
        Must be called AFTER all submodules (like eps_regressor) exist.
        """
        model_params = [
            p for p in m.parameters()
        ]  # prints: list[param]; why: includes eps_regressor if created
        if loss_logits is not None:
            return torch.optim.Adam(
                [
                    {"params": model_params, "weight_decay": args.weight_decay},
                    {
                        "params": [loss_logits],
                        "weight_decay": 0.0,
                    },  # no decay on logits
                ],
                lr=args.learning_rate,
            )
        else:
            return torch.optim.Adam(
                model_params, lr=args.learning_rate, weight_decay=args.weight_decay
            )

    # (re)build optimizer so it includes loss_logits (if any) + regressor params
    optimizer = build_optimizer(
        model
    )  # prints: torch.optim;   why: final optimizer with all params
    if model.eps_regressor is not None:
        num_reg_params = sum(p.numel() for p in model.eps_regressor.parameters())
        print(f"[DEBUG] EPS regressor params included in optimizer: {num_reg_params}")

    # ---- Rolling windows ----
    total_windows = last_train_end - first_train_end + 1
    print(
        f"\n--- Rolling training: windows={total_windows}, "
        f"initial Train=0..{first_train_end}, Val={first_train_end+1}, Test={first_train_end+2} ---"
    )

    # Logs
    metrics_log_path = os.path.join(args.output_dir, "rolling_metrics_log.jsonl")
    if os.path.exists(metrics_log_path):
        os.remove(metrics_log_path)
    # NEW: per-epoch CSV path
    # ADD ↓↓↓
    epoch_csv_path = os.path.join(
        args.output_dir,
        f"epoch_training_metrics_{args.run_hash}_w_predict_{args.w_predict}_rank_cp_{args.Rank_CP}_epochs_{args.epochs_per_window}.csv",  # str; unique per run
    )  # (str) path to per-epoch CSV with run hash
    # ADD ↑↑↑
    if not os.path.exists(epoch_csv_path):
        with open(epoch_csv_path, "w") as fcsv:
            fcsv.write(
                "window,epoch,train_end,val_idx,test_idx,"
                "train_recon_mse,train_pred_mse,train_drivers_mse,train_regression_mse,"
                "train_recon_r2,train_recon_r2_kelly,train_regression_r2,train_regression_r2_kelly,"
                "test_rmse,test_r2,test_r2_kelly,"
                "w_recon,w_pred,w_forecast_drivers,w_regression,"
                "mean_abs_offdiag\n"  # NEW column
            )

    # --- Collect last-epoch test predictions per window for final metrics ---
    all_test_rows = (
        []
    )  # each element: DataFrame with columns [t, firm, GroundTruth, Forecasted]

    for w_idx, train_end in enumerate(
        tqdm(range(first_train_end, last_train_end + 1), desc="Windows", position=0)
    ):
        val_idx = train_end + 1
        test_idx = train_end + 2
        train_T = train_end + 1  # history length used inside the model

        # Optionally recompute normalization each window (strict but slower).
        # May6 treats val_idx = test_idx - 1 as the last supervised training
        # target for forecasting test_idx. Therefore normalization may use data
        # through val_idx without peeking at the test target.
        if args.recompute_norm_each_window:
            feat_means_dev, feat_stds_dev = compute_stats_from_train_slice(val_idx)
            full_norm_dev = (
                full_raw_dev - feat_means_dev[None, None, :]
            ) / feat_stds_dev[None, None, :]
        else:
            feat_means_dev, feat_stds_dev = init_means_dev, init_stds_dev

        # ─────────────────────────────────────────────────────────────────────
        # NOTE: The OLD design built a `FilteredTensor3DDataset` over the entire
        # training cube full_norm_dev[:train_T] and a `DataLoader` to feed a
        # GLOBAL reconstruction loss (Step 1). That global block is REMOVED in
        # favor of a LOCAL rolling design where reconstruction is sampled
        # *inside the window [i-p+1 : i+1]* together with forecast and EPS
        # regression at i+1, all aligned to the SAME current_time_index = i.
        #
        # The full normalized tensor `full_norm_dev` (shape [T, N, F], on
        # device) is read directly during the new inner loop below — no
        # DataLoader is required because each per-i window is small enough to
        # mask + sample on the GPU.
        # ─────────────────────────────────────────────────────────────────────

        if args.reset_each_window and (w_idx > 0):
            model, _ = build_tensor_timesfm()  # prints: fresh model
            model.to(device)  # prints: none

            model._ensure_eps_regressor(  # prints: nn.Sequential; why: create K→1 head so optimizer sees params
                input_dim_drivers=len(selected_feature_indices_list) + 1
            )

            model = torch.compile(
                model, mode="reduce-overhead"
            )  # prints: none; why: same perf path as first window
            optimizer = build_optimizer(
                model
            )  # prints: Adam;  why: final optimizer with regressor + (maybe) loss_logits

        # no else needed

        # ---- Train for epochs_per_window epochs on this window ----
        inner = tqdm(
            range(1, args.epochs_per_window + 1),
            desc=f"[W{w_idx:03d}] Train=0..{train_end} Val={val_idx} Test={test_idx}",
            leave=False,
            position=1,
        )
        for ep in inner:
            model.train()

            # ════════════════════════════════════════════════════════════════════
            # NEW DESIGN — LOCAL ROLLING TRAINING LOOP
            # ────────────────────────────────────────────────────────────────────
            # Replaces:
            #   (a) the old global Step-1 reconstruction `for batch in train_ld:`
            #       block, which used a DataLoader over the full training cube,
            #   (b) the old Step 2-3 expanding-window block over
            #       target_time_index in [p, train_T) which forecast a different
            #       slice each iteration but reconstructed globally.
            #
            # Both old blocks ran ONE optimizer.step() per batch (Step 1) plus
            # ONE per epoch (Step 2-3 with gradient accumulation). The new
            # design instead runs ONE optimizer.step() per current_time_index i
            # with a TOTAL loss that aligns reconstruction, prediction,
            # forecast, and EPS regression to the SAME i:
            #
            #     for i in range(context_length, number_of_time_periods - 1):
            #         loss_total = w_recon * loss_reconstruction(window i)
            #                    + w_pred  * loss_prediction(i -> i+1)   # w_pred ≡ 0
            #                    + w_fcst  * loss_forecast(i+1)
            #                    + w_reg   * loss_regression(i+1)
            #                    + λ_ortho * orthogonality_penalty
            #                    + λ_frob  * frobenius_penalty
            #         optimizer.step()
            #
            # Note: this means more optimizer steps per epoch than before. The
            # learning rate / epochs_per_window may need re-tuning after the
            # first run.
            # ════════════════════════════════════════════════════════════════════

            # ---- epoch accumulators for unbiased component means (normalized space) ----
            recon_sse = 0.0  # float; sum of (pred - true)^2 over all reconstruction minibatches in this epoch
            recon_n = 0  # int; total count of reconstruction labels seen
            pred_sse = 0.0  # float; sum of (pred_emb - true_emb)^2 across i (logging only — w_pred=0)
            pred_n = 0  # int; total count of embedding-dim entries seen for prediction
            drivers_sse = 0.0  # float; LEGACY name kept for log/CSV — populated from the new forecast loss
            drivers_n = 0  # int; matches drivers_sse
            regression_sse = 0.0  # float; sum of (pred_eps - true_eps)^2 across forecast minibatches
            regression_n = 0  # int; total count of EPS regression labels seen

            # NEW (parallel to drivers_*): forecast SSE/count tracked separately so the
            # block-by-block explanation can audit the new forecast loss in isolation.
            forecast_sse = 0.0  # float
            forecast_n = 0  # int

            # opt_loss_running mirrors the previous tqdm display; we now treat each
            # current_time_index as a "step" for averaging purposes.
            opt_loss_running = 0.0
            
            opt_loss_running = 0.0

            # New two-stage bookkeeping.
            stage1_loss_running = 0.0
            stage2_loss_running = 0.0

            # accumulators for normalized-space R² of reconstruction
            recon_sum_y = 0.0  # float
            recon_sum_y2 = 0.0  # float

            mse_loss = nn.MSELoss()  # nn.MSELoss (PyTorch): scalar mean(|pred - target|^2)

            # ---- core loop indices/dimensions in the user's notation ----
            context_length = int(args.minimum_context_length)  # int = p
            # May6 leakage rule:
            #   For test index t = test_idx, the last supervised training
            #   target is t-1 = val_idx. Since train_T = train_end + 1 = t-1,
            #   set number_of_time_periods = train_T + 1 = t. The local loop
            #   below uses target i+1 and stops at i = t-2, so it never trains
            #   on target t.
            number_of_time_periods = int(train_T + 1)  # int = test_idx; available supervised data are 0..t-1
            number_of_firms = int(D1)  # int = N (firm dim)
            number_of_features = int(D2)  # int = F (feature dim)
            feature_index_target = int(args.target_feature_index)  # int; EPS supervised label column
            consensus_feature_index = int(args.consensus_feature_index)  # int; analyst consensus column

            # ---- selected_feature_indices used for BOTH reconstruction sampling AND forecast sampling ----
            # default = all tensor features except consensus; this includes the EPS target feature.
            selected_feature_indices_tensor = torch.tensor(
                selected_feature_indices_list, device=device, dtype=torch.long
            )  # [K_sel] long; K_sel = number of selected features
            number_of_selected_features = int(selected_feature_indices_tensor.numel())  # int = K_sel

            # ---- regression input feature indices ----
            # Shape: [K_sel] long. These are the forecasted non-consensus features
            # at i+1. Stage-B appends observed analyst consensus separately.
            regression_feature_indices_tensor = selected_feature_indices_tensor

            # ---- runtime sanity check on the loop bound: range(p, T-1) must be non-empty ----
            # Here T means test_idx. We need at least one current_time_index i
            # such that p <= i <= T-2. The first i = p has:
            #   input  = [1, ..., p]
            #   target = p+1
            # The final i = T-2 has:
            #   input  = [T-p-1, ..., T-2]
            #   target = T-1
            if context_length >= (number_of_time_periods - 1):
                raise ValueError(
                    f"[Window {w_idx:03d} | Epoch {ep:02d}] ERROR: "
                    f"context_length ({context_length}) >= "
                    f"number_of_time_periods-1 ({number_of_time_periods - 1}); "
                    f"the local rolling loop range(p, T-1) is empty. "
                    f"Lower --minimum_context_length or wait for a larger window."
                )

            # ---- weight constants (weight_prediction is hard-coded to 0 above) ----
            weight_reconstruction = float(args.w_recon)  # float
            # weight_prediction was already validated and locked to 0.0 above.
            # We rebind it locally so the code reads as a self-contained recipe:
            local_weight_prediction = 0.0  # float; LOCKED for this experiment
            weight_forecast = float(args.w_forecast_drivers)  # float
            weight_regression = float(args.w_regression)  # float
            lambda_orthogonality_time_value = float(args.lambda_ortho_time)  # float
            lambda_frobenius_time_value = float(args.lambda_frobenius_time)  # float

            # ---- inner tqdm bar over current_time_index for visibility ----
            time_index_bar = tqdm(
                range(context_length, number_of_time_periods - 1),
                desc=f"[W{w_idx:03d}|E{ep:02d}] i in [{context_length}..{number_of_time_periods - 2}]",
                leave=False,
                position=2,
            )
            number_of_time_steps = (number_of_time_periods - 1) - context_length  # int
            number_of_time_steps = max(number_of_time_steps, 1)  # avoid divide-by-zero in display

            for current_time_index in time_index_bar:
                i = int(current_time_index)  # int alias — matches user's i variable

                # ─────────────────────────────────────────────────────────────
                # Common local history indices.
                # History window = [i - p + 1, ..., i], length p.
                # Target time    = i + 1.
                # ─────────────────────────────────────────────────────────────
                history_time_indices_tensor = torch.arange(
                    i - context_length + 1,
                    i + 1,
                    device=device,
                    dtype=torch.long,
                )  # [p] long

                # ═════════════════════════════════════════════════════════════
                # STAGE 1: RECONSTRUCTION + TIME-EMBEDDING REGULARIZATION
                #
                # This stage updates embeddings and reconstruction_head.
                # It does NOT compute prediction/forecast/regression.
                # It includes:
                #   w_recon * reconstruction_loss
                #   + lambda_ortho * orthogonality_penalty
                #   + lambda_frobenius * frobenius_penalty
                # ═════════════════════════════════════════════════════════════

                # reconstruction_window_normalized: [p, N, K_sel] float32.
                # We restrict the third axis to selected_feature_indices.
                reconstruction_window_normalized = full_norm_dev[
                    i - context_length + 1 : i + 1
                ][
                    :, :, selected_feature_indices_tensor
                ]  # [p, N, K_sel] float32, NaN allowed

                # Mask + any(dim=0) gives the list of (firm, feature) fibers
                # with at least one observed value in the local p-length window.
                reconstruction_observed_mask = torch.isfinite(
                    reconstruction_window_normalized
                )  # [p, N, K_sel] bool
                reconstruction_observed_pair_mask = reconstruction_observed_mask.any(
                    dim=0
                )  # [N, K_sel] bool

                reconstruction_observed_pair_indices = torch.nonzero(
                    reconstruction_observed_pair_mask, as_tuple=False
                )  # [num_observed_pairs, 2] long; columns are (firm, k_local)

                number_of_observed_firm_feature_pairs = int(
                    reconstruction_observed_pair_indices.size(0)
                )

                reconstruction_minibatch_loss_list = []

                if number_of_observed_firm_feature_pairs == 0:
                    # Edge case: no observed local fibers in this window.
                    loss_reconstruction = torch.zeros((), device=device)
                else:
                    reconstruction_local_time_index_batch = torch.arange(
                        0,
                        context_length,
                        device=device,
                        dtype=torch.long,
                    )  # [p] long; ordered relative positions 0, ..., p-1

                    reconstruction_absolute_time_index_batch = torch.arange(
                        i - context_length + 1,
                        i + 1,
                        device=device,
                        dtype=torch.long,
                    )  # [p] long; absolute time indices i-p+1, ..., i

                    for reconstruction_minibatch_index in range(
                        int(args.number_of_reconstruction_minibatches)
                    ):
                        # Uniform sample WITH replacement over observed local fibers.
                        reconstruction_sample_pair_indices = torch.randint(
                            low=0,
                            high=number_of_observed_firm_feature_pairs,
                            size=(int(args.reconstruction_batch_size),),
                            device=device,
                        )  # [B_recon] long

                        sampled_firm_feature_pairs = reconstruction_observed_pair_indices[
                            reconstruction_sample_pair_indices
                        ]  # [B_recon, 2] long; columns are (firm, k_local)

                        reconstruction_firm_index_batch = sampled_firm_feature_pairs[
                            :, 0
                        ]  # [B_recon] long

                        reconstruction_feature_local_batch = sampled_firm_feature_pairs[
                            :, 1
                        ]  # [B_recon] long

                        reconstruction_feature_index_batch = selected_feature_indices_tensor[
                            reconstruction_feature_local_batch
                        ]  # [B_recon] long

                        # Build ordered time-by-fiber grid.
                        reconstruction_time_index_matrix = (
                            reconstruction_absolute_time_index_batch[:, None].expand(
                                -1,
                                reconstruction_firm_index_batch.numel(),
                            )
                        )  # [p, B_recon] long

                        reconstruction_firm_index_matrix = (
                            reconstruction_firm_index_batch[None, :].expand(
                                context_length,
                                -1,
                            )
                        )  # [p, B_recon] long

                        reconstruction_feature_index_matrix = (
                            reconstruction_feature_index_batch[None, :].expand(
                                context_length,
                                -1,
                            )
                        )  # [p, B_recon] long

                        # Predict reconstruction values.
                        predicted_reconstruction_value_matrix = model(
                            mode="reconstruction",
                            time_index=reconstruction_time_index_matrix.reshape(-1),
                            firm_index=reconstruction_firm_index_matrix.reshape(-1),
                            feature_index=reconstruction_feature_index_matrix.reshape(-1),
                        ).view(
                            context_length,
                            -1,
                        )  # [p, B_recon] float32

                        # True ordered fibers from local window.
                        true_reconstruction_value_matrix = reconstruction_window_normalized[
                            reconstruction_local_time_index_batch[:, None],
                            reconstruction_firm_index_batch,
                            reconstruction_feature_local_batch,
                        ]  # [p, B_recon] float32, NaN allowed

                        reconstruction_observed_fiber_mask = torch.isfinite(
                            true_reconstruction_value_matrix
                        )  # [p, B_recon] bool

                        if reconstruction_observed_fiber_mask.any():
                            loss_reconstruction_minibatch = mse_loss(
                                predicted_reconstruction_value_matrix[
                                    reconstruction_observed_fiber_mask
                                ],
                                true_reconstruction_value_matrix[
                                    reconstruction_observed_fiber_mask
                                ],
                            )  # scalar
                        else:
                            loss_reconstruction_minibatch = torch.zeros((), device=device)

                        reconstruction_minibatch_loss_list.append(
                            loss_reconstruction_minibatch
                        )

                        # Bookkeeping only, unweighted, normalized-space.
                        observed_reconstruction_count = int(
                            reconstruction_observed_fiber_mask.sum().item()
                        )
                        recon_sse += (
                            float(loss_reconstruction_minibatch.detach().item())
                            * observed_reconstruction_count
                        )
                        recon_n += observed_reconstruction_count

                        with torch.no_grad():
                            observed_true_reconstruction_values = true_reconstruction_value_matrix[
                                reconstruction_observed_fiber_mask
                            ]
                            recon_sum_y += float(
                                observed_true_reconstruction_values.sum().item()
                            )
                            recon_sum_y2 += float(
                                (
                                    observed_true_reconstruction_values
                                    * observed_true_reconstruction_values
                                ).sum().item()
                            )

                    loss_reconstruction = torch.stack(
                        reconstruction_minibatch_loss_list
                    ).mean()  # scalar

                # Regularization terms for Stage 1 only.
                # Keep your existing definition: computed over full supervised history rows.
                time_indices_for_regularization = torch.arange(
                    0,
                    number_of_time_periods,
                    device=device,
                    dtype=torch.long,
                )  # [number_of_time_periods] long

                time_embedding_matrix_over_time = model.time_embeddings(
                    time_indices_for_regularization
                )  # [T, R] float32

                if args.ortho_penalty_norm == "L2":
                    orthogonality_penalty_value = decorrelation_L2(
                        time_embedding_matrix_over_time
                    )
                else:
                    orthogonality_penalty_value = decorrelation_L1(
                        time_embedding_matrix_over_time
                    )

                frobenius_penalty_value = time_embedding_matrix_over_time.pow(2).mean()

                loss_stage1 = (
                    weight_reconstruction * loss_reconstruction
                    + lambda_orthogonality_time_value * orthogonality_penalty_value
                    + lambda_frobenius_time_value * frobenius_penalty_value
                )

                if not torch.isfinite(loss_stage1):
                    raise ValueError(
                        f"Non-finite loss_stage1 at i={i}: "
                        f"recon={loss_reconstruction.item():.6e}, "
                        f"ortho={float(orthogonality_penalty_value.item()):.6e}, "
                        f"frob={float(frobenius_penalty_value.item()):.6e}"
                    )

                # First optimizer update: reconstruction + regularization only.
                if loss_stage1.requires_grad:
                    optimizer.zero_grad(set_to_none=True)
                    loss_stage1.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                # ═════════════════════════════════════════════════════════════
                # STAGE 2: TIME-PREDICTION LOGGING + FEATURE FORECAST + EPS REGRESSION
                #
                # Important:
                #   We recompute predicted_time_factor_at_next_time AFTER Stage 1,
                #   because Stage 1 has just updated the embeddings.
                #
                # This stage includes:
                #   w_predict * prediction_loss
                #   + w_forecast * forecast_loss
                #   + w_regression * regression_loss
                #
                # It does NOT include orthogonality or Frobenius penalties.
                # ═════════════════════════════════════════════════════════════

                if args.use_timesfm:
                    predicted_time_factor_at_next_time = (
                        model.predict_next_time_embedding_with_adapter(
                            history_time_indices=history_time_indices_tensor,
                        )
                    )  # [R] float32
                else:
                    predicted_time_factor_at_next_time = model(
                        mode="prediction",
                        history_time_indices=history_time_indices_tensor,
                    )  # [R] float32

                if not torch.isfinite(predicted_time_factor_at_next_time).all():
                    raise ValueError(
                        f"Non-finite predicted_time_factor_at_next_time at i={i}."
                    )

                # Prediction loss: logging only if local_weight_prediction == 0.
                # The target label is detached, so gradient does not update the target row.
                true_time_factor_at_next_time = model.time_embeddings(
                    torch.tensor([i + 1], device=device, dtype=torch.long)
                )[0].detach()  # [R] float32

                loss_prediction = mse_loss(
                    predicted_time_factor_at_next_time,
                    true_time_factor_at_next_time,
                )  # scalar

                if not torch.isfinite(loss_prediction):
                    raise ValueError(f"Non-finite loss_prediction at i={i}.")

                pred_sse += float(loss_prediction.detach().item()) * int(
                    true_time_factor_at_next_time.numel()
                )
                pred_n += int(true_time_factor_at_next_time.numel())

                # Forecast + regression target slice at i+1.
                target_slice_at_next_time = full_norm_dev[i + 1]  # [N, F] float32, NaN allowed

                forecast_minibatch_loss_list = []
                regression_minibatch_loss_list = []

                for forecast_minibatch_index in range(
                    int(args.number_of_forecast_minibatches)
                ):
                    # Sample firm batch.
                    forecast_firm_index_batch = torch.randint(
                        low=0,
                        high=number_of_firms,
                        size=(int(args.forecast_batch_size),),
                        device=device,
                    )  # [B_fcst] long

                    # Sample feature batch from selected_feature_indices.
                    forecast_feature_local_batch = torch.randint(
                        low=0,
                        high=number_of_selected_features,
                        size=(int(args.forecast_batch_size),),
                        device=device,
                    )  # [B_fcst] long

                    forecast_feature_index_batch = selected_feature_indices_tensor[
                        forecast_feature_local_batch
                    ]  # [B_fcst] long

                    # Feature forecast loss at i+1.
                    predicted_forecast_value_batch = model(
                        mode="forecast",
                        predicted_time_embedding=predicted_time_factor_at_next_time,
                        firm_index=forecast_firm_index_batch,
                        feature_index=forecast_feature_index_batch,
                    )  # [B_fcst] float32

                    true_forecast_value_batch = target_slice_at_next_time[
                        forecast_firm_index_batch,
                        forecast_feature_index_batch,
                    ]  # [B_fcst] float32, NaN allowed

                    forecast_observed_mask_batch = torch.isfinite(
                        true_forecast_value_batch
                    )  # [B_fcst] bool

                    if forecast_observed_mask_batch.any():
                        loss_forecast_minibatch = mse_loss(
                            predicted_forecast_value_batch[forecast_observed_mask_batch],
                            true_forecast_value_batch[forecast_observed_mask_batch],
                        )

                        observed_forecast_count = int(
                            forecast_observed_mask_batch.sum().item()
                        )

                        forecast_sse += (
                            float(loss_forecast_minibatch.detach().item())
                            * observed_forecast_count
                        )
                        forecast_n += observed_forecast_count

                        # Keep legacy drivers_* logging.
                        drivers_sse += (
                            float(loss_forecast_minibatch.detach().item())
                            * observed_forecast_count
                        )
                        drivers_n += observed_forecast_count
                    else:
                        loss_forecast_minibatch = torch.zeros((), device=device)

                    forecast_minibatch_loss_list.append(loss_forecast_minibatch)

                    # ─── EPS regression on the same firm batch ───
                    true_eps_at_firm_batch = target_slice_at_next_time[
                        forecast_firm_index_batch,
                        feature_index_target,
                    ]  # [B_fcst] float32, NaN allowed

                    observed_consensus_at_firm_batch = target_slice_at_next_time[
                        forecast_firm_index_batch,
                        consensus_feature_index,
                    ]  # [B_fcst] float32, NaN allowed

                    valid_eps_mask_batch = torch.isfinite(
                        true_eps_at_firm_batch
                    ) & torch.isfinite(
                        observed_consensus_at_firm_batch
                    )  # [B_fcst] bool

                    if valid_eps_mask_batch.any():
                        valid_firm_index_batch = forecast_firm_index_batch[
                            valid_eps_mask_batch
                        ]  # [B'] long

                        # Forecast all selected non-consensus features for these firms.
                        forecasted_selected_features_at_next_time = (
                            model.forecast_features_matrix(
                                predicted_time_embedding=predicted_time_factor_at_next_time,
                                firm_index=valid_firm_index_batch,
                                feature_indices=regression_feature_indices_tensor,
                            )
                        )  # [B', K_sel] float32

                        observed_consensus_at_next_time = observed_consensus_at_firm_batch[
                            valid_eps_mask_batch
                        ]  # [B'] float32

                        features_before_consensus_mask = (
                            regression_feature_indices_tensor < int(consensus_feature_index)
                        )  # [K_sel] bool

                        features_after_consensus_mask = (
                            regression_feature_indices_tensor > int(consensus_feature_index)
                        )  # [K_sel] bool

                        regression_input_batch = torch.cat(
                            [
                                forecasted_selected_features_at_next_time[
                                    :, features_before_consensus_mask
                                ],
                                observed_consensus_at_next_time.unsqueeze(1),
                                forecasted_selected_features_at_next_time[
                                    :, features_after_consensus_mask
                                ],
                            ],
                            dim=1,
                        )  # [B', K_sel + 1]

                        regression_head_module = model._ensure_eps_regressor(
                            regression_input_batch.size(1)
                        )

                        predicted_eps_batch = regression_head_module(
                            regression_input_batch
                        ).squeeze(1)  # [B'] float32

                        true_eps_batch = true_eps_at_firm_batch[
                            valid_eps_mask_batch
                        ]  # [B'] float32

                        if not torch.isfinite(predicted_eps_batch).all():
                            raise ValueError(
                                f"Non-finite predicted_eps_batch at i={i}, "
                                f"forecast_minibatch_index={forecast_minibatch_index}."
                            )

                        loss_regression_minibatch = mse_loss(
                            predicted_eps_batch,
                            true_eps_batch,
                        )

                        observed_regression_count = int(
                            valid_eps_mask_batch.sum().item()
                        )

                        regression_sse += (
                            float(loss_regression_minibatch.detach().item())
                            * observed_regression_count
                        )
                        regression_n += observed_regression_count
                    else:
                        loss_regression_minibatch = torch.zeros((), device=device)

                    regression_minibatch_loss_list.append(loss_regression_minibatch)

                if forecast_minibatch_loss_list:
                    loss_forecast = torch.stack(forecast_minibatch_loss_list).mean()
                else:
                    loss_forecast = torch.zeros((), device=device)

                if regression_minibatch_loss_list:
                    loss_regression = torch.stack(regression_minibatch_loss_list).mean()
                else:
                    loss_regression = torch.zeros((), device=device)

                loss_stage2 = (
                    local_weight_prediction * loss_prediction
                    + weight_forecast * loss_forecast
                    + weight_regression * loss_regression
                )

                if not torch.isfinite(loss_stage2):
                    raise ValueError(
                        f"Non-finite loss_stage2 at i={i}: "
                        f"pred={loss_prediction.item():.6e}, "
                        f"fcst={loss_forecast.item():.6e}, "
                        f"reg={loss_regression.item():.6e}"
                    )

                # Second optimizer update: prediction/forecast/regression only.
                # No orthogonality/Frobenius here.
                if loss_stage2.requires_grad:
                    optimizer.zero_grad(set_to_none=True)
                    loss_stage2.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                # Display/bookkeeping.
                loss_two_stage_total_for_display = (
                    float(loss_stage1.detach().item())
                    + float(loss_stage2.detach().item())
                )

                opt_loss_running += loss_two_stage_total_for_display
                stage1_loss_running += float(loss_stage1.detach().item())
                stage2_loss_running += float(loss_stage2.detach().item())

                time_index_bar.set_postfix(
                    {
                        "L_s1": f"{float(loss_stage1.detach().item()):.3f}",
                        "L_s2": f"{float(loss_stage2.detach().item()):.3f}",
                        "L_recon": f"{float(loss_reconstruction.detach().item()):.3f}",
                        "L_fcst": f"{float(loss_forecast.detach().item()):.3f}",
                        "L_reg": f"{float(loss_regression.detach().item()):.3f}",
                        "L_ortho": f"{float(orthogonality_penalty_value.detach().item()):.3e}",
                        "L_frob": f"{float(frobenius_penalty_value.detach().item()):.3e}",
                    }
                )
            # ════════════════════════════════════════════════════════════════════
            # END OF LOCAL ROLLING TRAINING LOOP. Below: epoch metric finalization
            # uses recon_sse / drivers_sse / regression_sse accumulated above.
            # `num_batches` (used by the legacy display) is replaced by the per-i
            # step count so the existing avg_opt_loss / weights printout still
            # behaves sensibly.
            # ════════════════════════════════════════════════════════════════════
            num_batches = number_of_time_steps  # int; reused by the legacy display

            # === END OF LOCAL ROLLING LOOP: finalize epoch metrics (normalized space) ===
            recon_epoch_mse = (
                (recon_sse / recon_n) if recon_n > 0 else float("nan")
            )  # prints: float; mean MSE for reconstruction
            pred_epoch_mse = (pred_sse / pred_n) if pred_n > 0 else float("nan")
            # prints: float; mean MSE for time-embedding pred

            drivers_epoch_mse = (
                (drivers_sse / drivers_n) if drivers_n > 0 else float("nan")
            )  # prints: float; mean MSE for drivers forecast
            regression_epoch_mse = (
                (regression_sse / regression_n) if regression_n > 0 else float("nan")
            )  # prints: float; mean MSE for EPS regression
            # ADD ↓↓↓  (end of epoch, after drivers_epoch_mse / regression_epoch_mse computed)

            # (1) build the time-embedding matrix for observed data through val_idx.
            # This excludes the current test target test_idx, but includes the last
            # supervised training target val_idx = test_idx - 1.
            time_indices_for_history = torch.arange(
                0, train_T + 1, device=device
            )  # [train_T+1] long; 0..val_idx
            time_embedding_matrix_over_time = model.time_embeddings(
                time_indices_for_history
            )  # [train_T+1, R] float32

            # (2) center over time and compute covariance
            centered_time_embedding_matrix = _center_time(
                time_embedding_matrix_over_time
            )  # [train_T, R] float32
            covariance_centered_over_time = _covariance_centered(
                centered_time_embedding_matrix
            )  # [R, R] float32

            # (3) zero the diagonal, then take absolute mean across off-diagonals
            covariance_offdiag_only = _zero_diagonal(
                covariance_centered_over_time
            )  # [R, R] float32 (diag=0)
            mean_abs_offdiag_epoch = (
                covariance_offdiag_only.abs().mean().detach().item()
            )  # float; scalar metric
            # ADD ↓↓↓ (compute the scalar penalty at epoch end, using same norm)
            if args.ortho_penalty_norm == "L2":
                orthogonality_penalty_epoch = (
                    decorrelation_L2(time_embedding_matrix_over_time).detach().item()
                )  # float
            else:
                orthogonality_penalty_epoch = (
                    decorrelation_L1(time_embedding_matrix_over_time).detach().item()
                )  # float

            # (4) log + (optionally) include in progress bar line
            print(
                f"[Window {w_idx:03d} | Epoch {ep:02d}] mean_abs_offdiag = {mean_abs_offdiag_epoch:.6e}"
            )  # human eye
            # ADD ↑↑↑

            # training R² for reconstruction (normalized) — uses accumulators you already keep
            train_recon_r2, train_recon_r2_k = _safe_r2_from_sse(
                sse=recon_sse, sum_y=recon_sum_y, sum_y2=recon_sum_y2, n=recon_n
            )  # prints: floats; meaning: R² and Kelly R² for reconstruction

            # training R² for EPS at validation (normalized) — reuse your helper (once/epoch)
            # AFTER (clean)
            fc_sse, fc_sum_y, fc_sum_y2, fc_n = compute_epoch_forecast_aggregates(
                model=model,
                full_norm_dev=full_norm_dev,
                train_T=train_T,
                target_feature_index=args.target_feature_index,
                use_two_stage_forecast=args.use_two_stage_forecast,
                regression_feature_indices_list=selected_feature_indices_list,
                consensus_feature_index=args.consensus_feature_index,  # G2
                device=device,
                use_timesfm=args.use_timesfm,
                timesfm_repo_id=args.timesfm_repo_id,
                timesfm_freq_category=args.timesfm_freq_category,
                timesfm_horizon_len=args.timesfm_horizon_len,
                timesfm_context_cap=args.timesfm_context_cap,
                timesfm_per_core_batch_size=args.timesfm_per_core_batch_size,
            )

            train_regression_r2, train_regression_r2_k = _safe_r2_from_sse(
                fc_sse, fc_sum_y, fc_sum_y2, fc_n
            )  # prints: floats; meaning: EPS-at-val R² and Kelly R²

            # progress bar & console
            # include epoch-level orthogonality penalty in the displayed average (for visibility)
            # avg_opt_loss = (opt_loss_running / max(num_batches, 1)) + (
            #     lambda_orthogonality_time_value * orthogonality_penalty_epoch
            # )
            
            avg_opt_loss = opt_loss_running / max(num_batches, 1)

            avg_stage1_loss = stage1_loss_running / max(num_batches, 1)
            avg_stage2_loss = stage2_loss_running / max(num_batches, 1)
            # (float) add λ * penalty so the tqdm line tracks what you actually optimize

            w_now = (
                current_loss_weights().detach().cpu().tolist()
            )  # prints: [4] float; current weights

            inner.set_postfix(
                {
                    "avg_opt_loss": f"{avg_opt_loss:.3f}",
                    "avg_s1_loss": f"{avg_stage1_loss:.3f}",
                    "avg_s2_loss": f"{avg_stage2_loss:.3f}",
                    "recon": f"{recon_epoch_mse:.3f}",
                    "pred": f"{pred_epoch_mse:.3f}",
                    "drv": f"{drivers_epoch_mse:.3f}",
                    "reg": f"{regression_epoch_mse:.3f}",
                    "w": f"[{w_now[0]:.3f},{w_now[1]:.3f},{w_now[2]:.3f},{w_now[3]:.3f}]",
                    # ADD one key to the dict:
                    "mean_offdiag": f"{mean_abs_offdiag_epoch:.2e}",
                }
            )

            print(
                f"[Window {w_idx:03d} | Epoch {ep:02d}] "
                f"reconstruction loss = {recon_epoch_mse:.6f}, "
                f"prediction loss = {pred_epoch_mse:.6f}, "
                f"drivers-forecast loss = {drivers_epoch_mse:.6f}, "
                f"regression(EPS) loss = {regression_epoch_mse:.6f}"
            )

            # ---------- Build corrected final-test history (May7 fix) ----------
            # Goal: feed TimesFM `[e_{T-p}, ..., e_{T-2}, e_hat_{T-1}]` (length p),
            # not `[..., e_{T-1}_table]`. e_hat_{T-1} is recomputed from the
            # FINAL model parameters at this epoch (post optimizer.step()).
            #
            # Variable map (user-spec name → May7 var):
            #   T         = test_target_time_index   = test_idx       (= train_T + 1)
            #   T-1       = last_training_target_idx = train_T
            #   p         = context_length           = int(args.minimum_context_length)
            last_training_target_time_index = int(train_T)              # int; user's "T-1"
            test_target_time_index = int(test_idx)                      # int; user's "T"
            assert test_target_time_index == last_training_target_time_index + 1, (
                "test_idx must be exactly one step past train_T."
            )

            with torch.no_grad():
                # ----- (a) Recompute e_hat_{T-1} from history [T-p-1, ..., T-2] -----
                # In May7 indexing: indices [train_T - p, train_T) — length p.
                # All of these indices are valid TRAINING-history positions.
                last_training_context_start_index = (
                    last_training_target_time_index - context_length
                )  # int; inclusive
                last_training_context_end_exclusive = last_training_target_time_index  # int
                last_training_history_time_indices = torch.arange(
                    last_training_context_start_index,
                    last_training_context_end_exclusive,
                    device=device,
                    dtype=torch.long,
                )  # torch.LongTensor [p]; time indices for predicting e_hat_{T-1}

                predicted_time_embedding_for_last_training_target = (
                    model.predict_next_time_embedding_with_adapter(
                        history_time_indices=last_training_history_time_indices
                    )
                )  # torch.Tensor [R]; predicted last-training-target embedding e_hat_{T-1}

                # ----- (b) Build the length-p final-test history. -----
                # First (p-1) slots: TABLE embeddings at [T-p, ..., T-2].
                # Last slot:        PREDICTED embedding e_hat_{T-1}.
                ground_truth_history_time_indices_without_last = torch.arange(
                    test_target_time_index - context_length,    # T-p (inclusive)
                    test_target_time_index - 1,                 # T-1 (exclusive) -> ends at T-2
                    device=device,
                    dtype=torch.long,
                )  # torch.LongTensor [p-1]; indices [T-p, ..., T-2]
                assert (
                    ground_truth_history_time_indices_without_last.numel() == 0
                    or ground_truth_history_time_indices_without_last.max().item()
                       <= test_target_time_index - 2
                ), "Final-test history must not include T or T-1 from the table."

                ground_truth_history_embedding_matrix_without_last = model.time_embeddings(
                    ground_truth_history_time_indices_without_last
                )  # torch.Tensor [p-1, R]; LEARNED TABLE embeddings for [T-p, ..., T-2]

                final_history_embedding_matrix = torch.cat(
                    [
                        ground_truth_history_embedding_matrix_without_last,            # [p-1, R]
                        predicted_time_embedding_for_last_training_target.unsqueeze(0),  # [1, R]
                    ],
                    dim=0,
                )  # torch.Tensor [p, R]; FINAL TimesFM input history at test time
                assert final_history_embedding_matrix.shape[0] == context_length

                # ----- (c) Predict e_hat_T from the corrected history. -----
                predicted_time_embedding_for_test = (
                    model.predict_next_time_embedding_from_history_embedding_matrix_with_adapter(
                        history_time_embedding_matrix=final_history_embedding_matrix
                    )
                )  # torch.Tensor [R]; predicted embedding for the final test target T

                # ----- (d) Debug stats: predicted-vs-table at the last slot. -----
                # `_table_last_history_embedding` is what May6 used in slot p-1.
                _table_last_history_embedding = model.time_embeddings(
                    torch.tensor(
                        [last_training_target_time_index],
                        device=device,
                        dtype=torch.long,
                    )
                ).squeeze(0)  # torch.Tensor [R]; table value e_{T-1}
                _predicted_last_norm = float(
                    torch.norm(predicted_time_embedding_for_last_training_target).item()
                )
                _table_last_norm = float(torch.norm(_table_last_history_embedding).item())
                _diff_last_norm = float(
                    torch.norm(
                        predicted_time_embedding_for_last_training_target
                        - _table_last_history_embedding
                    ).item()
                )

            # Evaluate on the single test step for this window
            metrics, oos_df = evaluate_one_step(
                model,
                full_raw_cpu,
                feat_means_dev.detach().cpu(),
                feat_stds_dev.detach().cpu(),
                train_T,
                test_idx,
                args.target_feature_index,
                args.consensus_feature_index,
                device,
                use_two_stage_forecast=args.use_two_stage_forecast,
                regression_feature_indices_list=selected_feature_indices_list,
                tickers_array=tickers,
                use_timesfm=args.use_timesfm,
                timesfm_repo_id=args.timesfm_repo_id,
                timesfm_freq_category=args.timesfm_freq_category,
                timesfm_horizon_len=args.timesfm_horizon_len,
                timesfm_context_cap=args.timesfm_context_cap,
                timesfm_per_core_batch_size=args.timesfm_per_core_batch_size,
                plot_feature_index=args.plot_feature_index,  # <--- ADD THIS LINE HERE!
                tensor_time_axis_dates=tensor_time_axis_dates,
                xgboost_eligible_by_date=xgboost_eligible_by_date,
                override_predicted_time_embedding_for_test=predicted_time_embedding_for_test,
            )

            # Console summary
            print(
                f"\n[Window {w_idx:03d} | Epoch {ep:02d}] "
                f"Test t={test_idx} -> RMSE={metrics['rmse']:.6f}, "
                f"R²={metrics['r2']:.6f}, Kelly R²={metrics['r2_kelly']:.6f}"
            )
            if not oos_df.empty:
                print(oos_df.head(10).to_string(index=False))
            # ---------------- W&B per-epoch log ----------------
            # (friendly: builds a small dict and streams it to your dashboard)

            # compute analyst/consensus baseline for THIS test step using the oos_df table
            if not oos_df.empty:
                # y_true_tensor: torch.Tensor[float] shape: [M]; meaning: GT EPS at test step
                y_true_tensor = torch.from_numpy(oos_df["GroundTruth"].values).float()
                # y_cons_tensor: torch.Tensor[float] shape: [M]; meaning: analyst consensus at test step
                y_cons_tensor = torch.from_numpy(oos_df["Consensus"].values).float()

                # rmse_consensus: float; meaning: baseline RMSE (= sqrt of MSE)
                rmse_consensus = math.sqrt(
                    nn.functional.mse_loss(y_cons_tensor, y_true_tensor).item()
                )
                # r2_consensus: float; meaning: standard R² for consensus
                r2_consensus = r2(y_true_tensor, y_cons_tensor)
                # r2k_consensus: float; meaning: Kelly’s R² for consensus
                r2k_consensus = r2_kelly(y_true_tensor, y_cons_tensor)
            else:
                # handle rare case with no rows at this step
                rmse_consensus = float("nan")
                r2_consensus = float("nan")
                r2k_consensus = float("nan")

            # w_now_list: list[float] len=4; meaning: current mixture weights (recon,pred,drivers,regression)
            w_now_list = current_loss_weights().detach().cpu().tolist()
            # w_r, w_p, w_drv, w_reg: floats; meaning: unpacked weights
            w_r, w_p, w_drv, w_reg = w_now_list

            # global_step: int; meaning: single monotonically-increasing step for nice charts
            global_step = (w_idx * args.epochs_per_window) + ep

            # log_row: dict[str, number]; meaning: everything you want to see in W&B for this epoch
            log_row = {
                # bookkeeping / position
                "window": int(w_idx),  # int; which rolling window
                "epoch_in_window": int(ep),  # int; epoch number inside this window
                "global_step": int(
                    global_step
                ),  # int; continuous step across all windows
                "train_end": int(train_end),  # int; last train index
                "val_idx": int(val_idx),  # int; validation time index
                "test_idx": int(test_idx),  # int; test time index
                # training losses (normalized space)
                "loss/reconstruction": float(recon_epoch_mse),  # float; MSE
                "loss/prediction": float(pred_epoch_mse),  # float; MSE
                "loss/drivers_forecast": float(drivers_epoch_mse),  # float; MSE
                "loss/eps_regression": float(regression_epoch_mse),  # float; MSE
                # training R² at epoch (normalized space)
                "train/r2_reconstruction": float(train_recon_r2),  # float; R²
                "train/r2_reconstruction_kelly": float(
                    train_recon_r2_k
                ),  # float; Kelly R²
                "train/r2_eps_at_val": float(train_regression_r2),  # float; R²
                "train/r2_eps_at_val_kelly": float(
                    train_regression_r2_k
                ),  # float; Kelly R²
                # model forecast metrics on THIS test step (raw space)
                "test/rmse": float(metrics["rmse"]),  # float
                "test/r2": float(metrics["r2"]),  # float
                "test/r2_kelly": float(metrics["r2_kelly"]),  # float
                # consensus baseline on THIS test step (raw space)
                "consensus/test_rmse": float(rmse_consensus),  # float
                "consensus/test_r2": float(r2_consensus),  # float
                "consensus/test_r2_kelly": float(r2k_consensus),  # float
                # mixture weights
                "weights/w_recon": float(w_r),  # float
                "weights/w_pred": float(w_p),  # float
                "weights/w_forecast_drivers": float(w_drv),  # float
                "weights/w_regression": float(w_reg),  # float
            }

            # (optional) advisor-friendly aliases so names match your CSV/final printouts
            log_row.update(
                {
                    "Forecasted_RMSE": log_row["test/rmse"],  # float; alias
                    "Forecasted_R2": log_row["test/r2"],  # float; alias
                    "Forecasted_Kelly R2": log_row["test/r2_kelly"],  # float; alias
                    "Analyst_RMSE": log_row["consensus/test_rmse"],  # float; alias
                    "Analyst_R2": log_row["consensus/test_r2"],  # float; alias
                    "Analyst_Kelly R2": log_row[
                        "consensus/test_r2_kelly"
                    ],  # float; alias
                    "w_recon": log_row["weights/w_recon"],  # float; alias
                    "w_pred": log_row["weights/w_pred"],  # float; alias
                    "w_forecast_drivers": log_row[
                        "weights/w_forecast_drivers"
                    ],  # float; alias
                    "w_regression": log_row["weights/w_regression"],  # float; alias
                }
            )
            log_row["hparams/epochs_per_window"] = int(args.epochs_per_window)
            log_row["epoch_progress"] = float(ep / args.epochs_per_window)  # 0..1
            # ADD ↓↓↓ (extra keys in log_row)
            log_row["reg/mean_abs_offdiag"] = float(
                mean_abs_offdiag_epoch
            )  # float; decorrelation metric
            log_row["reg/lambda_ortho_time"] = float(
                args.lambda_ortho_time
            )  # float; penalty weight
            # Frobenius-norm tracking: mean(E**2) of the time-embedding matrix
            # (same E used for the orthogonality penalty). Lets the sweep see
            # whether λ_F is actually holding ||E|| in place.
            log_row["reg/frobenius_mean_sq"] = float(
                (time_embedding_matrix_over_time.detach() ** 2).mean().item()
            )  # float; ||E||_F^2 / (T*R)
            log_row["reg/lambda_frobenius_time"] = float(
                args.lambda_frobenius_time
            )  # float; F-norm regularizer weight
            log_row["reg/penalty_norm"] = str(
                args.ortho_penalty_norm
            )  # str; "L2" or "L1"
            # ADD ↑↑↑
            # --- add orthogonality diagnostics to the W&B row ---
            log_row["reg/ortho_penalty_epoch"] = float(
                orthogonality_penalty_epoch
            )  # float; penalty value (norm as selected)
            log_row["reg/mean_abs_offdiag"] = float(
                mean_abs_offdiag_epoch
            )  # float; |offdiag(C)| mean
            log_row["reg/lambda_ortho_time"] = float(
                args.lambda_ortho_time
            )  # float; hyperparameter weight
            log_row["reg/penalty_norm"] = str(
                args.ortho_penalty_norm
            )  # str; "L2" or "L1"

            # ---- May7 debug: predicted-vs-table last-history-slot ----
            log_row["debug/use_predicted_last_history_embedding_for_final_test"] = True
            log_row["debug/last_training_target_time_index"] = int(
                last_training_target_time_index
            )
            log_row["debug/test_target_time_index"] = int(test_target_time_index)
            log_row["debug/final_history_embedding_matrix_shape"] = int(
                final_history_embedding_matrix.shape[0]
            )
            log_row["debug/ground_truth_history_part_shape"] = int(
                ground_truth_history_embedding_matrix_without_last.shape[0]
            )
            log_row["debug/predicted_last_history_embedding_norm"] = _predicted_last_norm
            log_row["debug/table_last_history_embedding_norm"] = _table_last_norm
            log_row["debug/difference_between_predicted_and_table_last_history_embedding_norm"] = (
                _diff_last_norm
            )

            # actually stream it to W&B (step makes charts align)
            wandb.log(
                log_row, step=global_step
            )  # returns: None; meaning: sends a row to the run
            # ----------------------------------------------------

            # Persist metrics
            with open(metrics_log_path, "a") as f:
                w_now = (
                    current_loss_weights().detach().cpu().tolist()
                )  # prints: [4] float
                w_r, w_p, w_drv, w_reg = w_now  # prints: 4 floats

                rec = {
                    "window": w_idx,
                    "epoch": ep,
                    "train_end": train_end,
                    "val_idx": val_idx,
                    "test_idx": test_idx,
                    # --- TEST metrics (raw scale) ---
                    "test_rmse": float(metrics["rmse"]),
                    "test_r2": float(metrics["r2"]),
                    "test_r2_k": float(metrics["r2_kelly"]),
                    # --- TRAIN metrics (normalized scale) ---
                    "train_recon_mse": float(recon_epoch_mse),
                    "train_pred_mse": float(pred_epoch_mse),
                    "train_drivers_mse": float(drivers_epoch_mse),
                    "train_regression_mse": float(regression_epoch_mse),
                    # R² (normalized): recon + EPS-at-val (call above assigned)
                    "train_recon_r2": float(train_recon_r2),
                    "train_recon_r2_kelly": float(train_recon_r2_k),
                    "train_regression_r2": float(train_regression_r2),
                    "train_regression_r2_kelly": float(train_regression_r2_k),
                    # --- loss weights at this epoch ---
                    "w_recon": float(w_r),
                    "w_pred": float(w_p),
                    "w_forecast_drivers": float(w_drv),
                    "w_regression": float(w_reg),
                }

                f.write(json.dumps(rec) + "\n")
            # NEW: append a CSV row per epoch (training normalized metrics + test raw metrics)
            with open(epoch_csv_path, "a") as fcsv:
                fcsv.write(
                    f"{w_idx},{ep},{train_end},{val_idx},{test_idx},"
                    f"{recon_epoch_mse:.8f},{pred_epoch_mse:.8f},{drivers_epoch_mse:.8f},{regression_epoch_mse:.8f},"
                    f"{train_recon_r2:.8f},{train_recon_r2_k:.8f},{train_regression_r2:.8f},{train_regression_r2_k:.8f},"
                    f"{metrics['rmse']:.8f},{metrics['r2']:.8f},{metrics['r2_kelly']:.8f},"
                    f"{w_r:.8f},{w_p:.8f},{w_drv:.8f},{w_reg:.8f},"
                    f"{mean_abs_offdiag_epoch:.8e}\n"  # scientific format for small values
                )

            # If this is the last epoch for the current window, keep the test preds for final summary
            if ep == args.epochs_per_window and not oos_df.empty:
                all_test_rows.append(oos_df.copy(deep=True))

        # end epochs for this window

    # -------- Final across-all-tests summary --------
    if all_test_rows:
        all_df = pd.concat(
            all_test_rows, ignore_index=True
        )  # prints: DataFrame; why: stack all windows
        assert (
            len(tickers) == D1
        ), f"Ticker array length {len(tickers)} != firm dim {D1}"  # prints: msg; why: alignment safety

        all_df["Ticker"] = tickers[
            all_df["firm"].to_numpy()
        ]  # prints: none;    why: readable firm id
        all_df = all_df.sort_values(["t", "Ticker"]).reset_index(
            drop=True
        )  # prints: none; why: stable nice order

        y_true = torch.from_numpy(all_df["GroundTruth"].values).float()
        y_pred = torch.from_numpy(all_df["Forecasted"].values).float()
        actual_oos_rows = int(len(all_df))
        mse = nn.functional.mse_loss(y_pred, y_true).item()
        final_rmse = math.sqrt(mse)
        final_r2 = r2(y_true, y_pred)
        final_r2_kelly = r2_kelly(y_true, y_pred)

        # ► NEW: analyst-consensus metrics
        cons_tensor = torch.from_numpy(all_df["Consensus"].values).float()

        mse_cons = nn.functional.mse_loss(cons_tensor, y_true).item()
        cons_rmse = math.sqrt(mse_cons)
        cons_r2 = r2(y_true, cons_tensor)
        cons_r2_k = r2_kelly(y_true, cons_tensor)

        print("\n===== FINAL ROLLING-TEST SUMMARY (all windows) =====")
        print(f"RMSE  : {final_rmse:.6f}")
        print(f"R²    : {final_r2:.6f}")
        print(f"Kelly : {final_r2_kelly:.6f}")

        print("\n===== FINAL ROLLING-TEST SUMMARY (all windows) =====")
        print(f"Model-RMSE   : {final_rmse:.6f}")
        print(f"Model-R²     : {final_r2:.6f}")
        print(f"Model-Kelly  : {final_r2_kelly:.6f}")

        print(f"Analyst-RMSE : {cons_rmse:.6f}")
        print(f"Analyst-R²   : {cons_r2:.6f}")
        print(f"Analyst-Kelly: {cons_r2_k:.6f}")
        print(
            f"Eval rows    : {actual_oos_rows}"
            + (
                f" / expected XGBoost-eligible {xgboost_expected_oos_pairs}"
                if xgboost_expected_oos_pairs is not None
                else ""
            )
        )
        if (
            xgboost_expected_oos_pairs is not None
            and actual_oos_rows != int(xgboost_expected_oos_pairs)
        ):
            print(
                "[WARN] Actual Tensor-TimesFM OOS row count does not match the "
                "XGBoost-eligible mask. Do not compare R2 until this is resolved."
            )

        # Final loss mixture weights (from the last model state)
        # with torch.no_grad():
        #     w = torch.softmax(-model.loss_params, dim=0).cpu().numpy()
        # print final weights (learned or fixed) and use them to build the log filename
        w_final = current_loss_weights().detach().cpu().tolist()  # prints: [4] float
        w_recon_f, w_pred_f, w_drv_f, w_reg_f = w_final
        print(
            f"Final weights -> "
            f"w_recon={w_recon_f:.8f}, w_pred={w_pred_f:.8f}, "
            f"w_forecast_drivers={w_drv_f:.8f}, w_regression={w_reg_f:.8f}"
        )

        # -------- [W&B] final summary --------
        wandb.summary["final/model_rmse"] = float(final_rmse)
        wandb.summary["final/model_r2"] = float(final_r2)
        wandb.summary["final/model_r2_kelly"] = float(final_r2_kelly)

        wandb.summary["final/consensus_rmse"] = float(cons_rmse)
        wandb.summary["final/consensus_r2"] = float(cons_r2)
        wandb.summary["final/consensus_r2_kelly"] = float(cons_r2_k)
        wandb.summary["final/eval_rows"] = int(actual_oos_rows)
        if xgboost_expected_oos_pairs is not None:
            wandb.summary["final/xgboost_expected_oos_rows"] = int(
                xgboost_expected_oos_pairs
            )

        wandb.summary["final/w_recon"] = w_recon_f
        wandb.summary["final/w_pred"] = w_pred_f
        wandb.summary["final/w_forecast_drivers"] = w_drv_f
        wandb.summary["final/w_regression"] = w_reg_f

        # Save your CSVs/artifacts to the run (optional but handy)
        if os.path.exists(epoch_csv_path):
            wandb.save(epoch_csv_path)
        # --------------------------------------

        # ============================================================================
        # Extra metrics: RMSE / MAE / MAPE / R² / Kelly R² for
        #   (tensor-timesfm vs timesfm-baseline) x (target feature, plot feature)
        # Baseline inference is the expensive part — run it once here so both the
        # CSV below and the fiber plot below can consume the same numbers.
        # ============================================================================
        plot_feature_index = int(args.plot_feature_index)
        target_feature_index = int(args.target_feature_index)

        # Tensor model on target feature: extend existing (RMSE/R²/Kelly) with MAE/MAPE.
        target_tensor_rmse = float(final_rmse)
        target_tensor_r2 = float(final_r2)
        target_tensor_r2k = float(final_r2_kelly)
        target_tensor_mae = mae_metric(y_true, y_pred)
        target_tensor_mape = mape_metric(y_true, y_pred)

        # Sort once; shared by plot-feature metric calc and by the fiber plot.
        all_df_sorted = all_df.sort_values(
            by=["t", "firm"], ascending=[True, True]
        ).reset_index(drop=True)

        # Tensor model on the plot feature (always computable — no baseline needed).
        valid_plot_mask_np = ~all_df_sorted["PlotFeatureGroundTruth"].isna().to_numpy()
        plot_df = all_df_sorted.loc[valid_plot_mask_np].copy()
        if len(plot_df) > 0:
            gt_plot_t = torch.from_numpy(
                plot_df["PlotFeatureGroundTruth"].values.astype(np.float32)
            )
            pred_tensor_plot_t = torch.from_numpy(
                plot_df["PlotFeatureForecast"].values.astype(np.float32)
            )
            plot_tensor_rmse = rmse_metric(gt_plot_t, pred_tensor_plot_t)
            plot_tensor_mae = mae_metric(gt_plot_t, pred_tensor_plot_t)
            plot_tensor_mape = mape_metric(gt_plot_t, pred_tensor_plot_t)
            plot_tensor_r2 = r2(gt_plot_t, pred_tensor_plot_t)
            plot_tensor_r2k = r2_kelly(gt_plot_t, pred_tensor_plot_t)
        else:
            gt_plot_t = None
            pred_tensor_plot_t = None
            plot_tensor_rmse = plot_tensor_mae = plot_tensor_mape = float("nan")
            plot_tensor_r2 = plot_tensor_r2k = float("nan")

        # Baseline metrics: default NaN; fill in if baseline is actually run.
        target_timesfm_rmse = target_timesfm_mae = target_timesfm_mape = float("nan")
        target_timesfm_r2 = target_timesfm_r2k = float("nan")
        plot_timesfm_rmse = plot_timesfm_mae = plot_timesfm_mape = float("nan")
        plot_timesfm_r2 = plot_timesfm_r2k = float("nan")
        pred_timesfm_plot_t = None
        pred_timesfm_target_t = None
        timesfm_plot_fiber_full = None
        timesfm_target_fiber_full = None
        full_time_index_array_1d = None
        full_firm_index_list_per_time = None
        baseline_ok = False

        if bool(args.make_fiber_forecast_plot):
            full_time_index_array_1d = np.sort(
                all_df_sorted["t"].astype(int).unique()
            )
            if full_time_index_array_1d.size == 0:
                print("[WARN] all_df has no valid time indices; skip baseline run.")
            else:
                full_firm_index_list_per_time = [
                    all_df_sorted.loc[all_df_sorted["t"] == t, "firm"]
                    .values.astype(np.int64)
                    for t in full_time_index_array_1d
                ]
                full_raw_value_tensor_time_firm_feature = (
                    full_raw_cpu.detach().cpu().numpy().astype(np.float32)
                )
                timesfm_baseline_device = (
                    torch.device("cuda")
                    if (bool(args.timesfm_baseline_use_gpu) and torch.cuda.is_available())
                    else torch.device("cpu")
                )
                try:
                    timesfm_official_forecaster = TimesFMOfficialZeroShotForecaster(
                        timesfm_repo_id=str(args.timesfm_repo_id),
                        timesfm_context_cap=int(args.timesfm_context_cap),
                        timesfm_horizon_len=int(args.timesfm_horizon_len),
                        timesfm_per_core_batch_size=int(
                            args.timesfm_per_core_batch_size
                        ),
                        timesfm_freq_category=int(args.timesfm_freq_category),
                        device=timesfm_baseline_device,
                    )
                    plot_list_per_time_full, target_list_per_time_full = (
                        run_vanilla_timesfm_baseline_expanding_window(
                            full_raw_value_tensor_time_firm_feature=full_raw_value_tensor_time_firm_feature,
                            forecast_time_index_array_1d=full_time_index_array_1d,
                            firm_index_list_per_time=full_firm_index_list_per_time,
                            timesfm_official_forecaster=timesfm_official_forecaster,
                            plot_feature_index=plot_feature_index,
                            target_feature_index=target_feature_index,
                        )
                    )
                    timesfm_plot_fiber_full = np.concatenate(
                        plot_list_per_time_full, axis=0
                    ).astype(np.float32)
                    timesfm_target_fiber_full = np.concatenate(
                        target_list_per_time_full, axis=0
                    ).astype(np.float32)

                    gt_target_t = torch.from_numpy(
                        all_df_sorted["GroundTruth"].values.astype(np.float32)
                    )
                    pred_timesfm_target_t = torch.from_numpy(timesfm_target_fiber_full)

                    target_timesfm_rmse = rmse_metric(gt_target_t, pred_timesfm_target_t)
                    target_timesfm_mae = mae_metric(gt_target_t, pred_timesfm_target_t)
                    target_timesfm_mape = mape_metric(gt_target_t, pred_timesfm_target_t)
                    target_timesfm_r2 = r2(gt_target_t, pred_timesfm_target_t)
                    target_timesfm_r2k = r2_kelly(gt_target_t, pred_timesfm_target_t)

                    if gt_plot_t is not None:
                        pred_timesfm_plot_t = torch.from_numpy(
                            timesfm_plot_fiber_full[valid_plot_mask_np]
                        )
                        plot_timesfm_rmse = rmse_metric(gt_plot_t, pred_timesfm_plot_t)
                        plot_timesfm_mae = mae_metric(gt_plot_t, pred_timesfm_plot_t)
                        plot_timesfm_mape = mape_metric(gt_plot_t, pred_timesfm_plot_t)
                        plot_timesfm_r2 = r2(gt_plot_t, pred_timesfm_plot_t)
                        plot_timesfm_r2k = r2_kelly(gt_plot_t, pred_timesfm_plot_t)

                    baseline_ok = True
                except Exception as e:  # why: don't drop CSV just because baseline crashed
                    print(f"[WARN] TimesFM baseline run failed: {e}")

        # Pre-build the stats text block used by the fiber plot (and handy in logs).
        stats_text = (
            f"--- Plot Feature (Idx {plot_feature_index}) ---\n"
            f"TimesFM Baseline  R²: {plot_timesfm_r2:7.4f} | Kelly: {plot_timesfm_r2k:7.4f} | "
            f"RMSE: {plot_timesfm_rmse:8.4f} | MAE: {plot_timesfm_mae:8.4f} | MAPE: {plot_timesfm_mape:8.2f}%\n"
            f"Tensor-TimesFM    R²: {plot_tensor_r2:7.4f} | Kelly: {plot_tensor_r2k:7.4f} | "
            f"RMSE: {plot_tensor_rmse:8.4f} | MAE: {plot_tensor_mae:8.4f} | MAPE: {plot_tensor_mape:8.2f}%\n\n"
            f"--- Target Feature (Idx {target_feature_index}) ---\n"
            f"TimesFM Baseline  R²: {target_timesfm_r2:7.4f} | Kelly: {target_timesfm_r2k:7.4f} | "
            f"RMSE: {target_timesfm_rmse:8.4f} | MAE: {target_timesfm_mae:8.4f} | MAPE: {target_timesfm_mape:8.2f}%\n"
            f"Tensor-TimesFM    R²: {target_tensor_r2:7.4f} | Kelly: {target_tensor_r2k:7.4f} | "
            f"RMSE: {target_tensor_rmse:8.4f} | MAE: {target_tensor_mae:8.4f} | MAPE: {target_tensor_mape:8.2f}%"
        )
        print("\n" + stats_text)

        # Mirror the extra metrics into W&B summary as well.
        wandb.summary["final/target_tensor_mae"] = float(target_tensor_mae)
        wandb.summary["final/target_tensor_mape"] = float(target_tensor_mape)
        wandb.summary["final/target_timesfm_rmse"] = float(target_timesfm_rmse)
        wandb.summary["final/target_timesfm_mae"] = float(target_timesfm_mae)
        wandb.summary["final/target_timesfm_mape"] = float(target_timesfm_mape)
        wandb.summary["final/target_timesfm_r2"] = float(target_timesfm_r2)
        wandb.summary["final/target_timesfm_r2_kelly"] = float(target_timesfm_r2k)
        wandb.summary["final/plot_feature_index"] = int(plot_feature_index)
        wandb.summary["final/plot_tensor_rmse"] = float(plot_tensor_rmse)
        wandb.summary["final/plot_tensor_mae"] = float(plot_tensor_mae)
        wandb.summary["final/plot_tensor_mape"] = float(plot_tensor_mape)
        wandb.summary["final/plot_tensor_r2"] = float(plot_tensor_r2)
        wandb.summary["final/plot_tensor_r2_kelly"] = float(plot_tensor_r2k)
        wandb.summary["final/plot_timesfm_rmse"] = float(plot_timesfm_rmse)
        wandb.summary["final/plot_timesfm_mae"] = float(plot_timesfm_mae)
        wandb.summary["final/plot_timesfm_mape"] = float(plot_timesfm_mape)
        wandb.summary["final/plot_timesfm_r2"] = float(plot_timesfm_r2)
        wandb.summary["final/plot_timesfm_r2_kelly"] = float(plot_timesfm_r2k)

        log_name = f"results_{args.run_hash}.csv"
        log_path = os.path.join(args.output_dir, log_name)

        header = (
            "Forecasted_RMSE,Forecasted_R2,Forecasted_Kelly R2,"
            "Analyst_RMSE,Analyst_R2,Analyst_Kelly R2,"
            # Target feature (tensor model already logged above; add MAE/MAPE)
            "Target_Tensor_MAE,Target_Tensor_MAPE,"
            # Target feature (TimesFM baseline)
            "Target_TimesFM_RMSE,Target_TimesFM_MAE,Target_TimesFM_MAPE,"
            "Target_TimesFM_R2,Target_TimesFM_Kelly_R2,"
            # Plot feature index + tensor-model metrics on the plot feature
            "Plot_Feature_Index,"
            "Plot_Tensor_RMSE,Plot_Tensor_MAE,Plot_Tensor_MAPE,"
            "Plot_Tensor_R2,Plot_Tensor_Kelly_R2,"
            # Plot feature (TimesFM baseline)
            "Plot_TimesFM_RMSE,Plot_TimesFM_MAE,Plot_TimesFM_MAPE,"
            "Plot_TimesFM_R2,Plot_TimesFM_Kelly_R2,"
            "Eval_Filter,N_Predictions,Expected_XGBoost_Eligible_OOS_Rows,"
            "w_recon,w_pred,w_forecast_drivers,w_regression,"
            "lambda_ortho_time,lambda_frobenius_time\n"
        )
        eval_filter_name = (
            "xgboost_eligible"
            if bool(args.filter_oos_to_xgboost_eligible)
            else "observed_target"
        )
        expected_oos_for_csv = (
            int(xgboost_expected_oos_pairs)
            if xgboost_expected_oos_pairs is not None
            else ""
        )
        row = (
            f"{final_rmse:.6f},{final_r2:.6f},{final_r2_kelly:.6f},"
            f"{cons_rmse:.6f},{cons_r2:.6f},{cons_r2_k:.6f},"
            f"{target_tensor_mae:.6f},{target_tensor_mape:.6f},"
            f"{target_timesfm_rmse:.6f},{target_timesfm_mae:.6f},{target_timesfm_mape:.6f},"
            f"{target_timesfm_r2:.6f},{target_timesfm_r2k:.6f},"
            f"{plot_feature_index},"
            f"{plot_tensor_rmse:.6f},{plot_tensor_mae:.6f},{plot_tensor_mape:.6f},"
            f"{plot_tensor_r2:.6f},{plot_tensor_r2k:.6f},"
            f"{plot_timesfm_rmse:.6f},{plot_timesfm_mae:.6f},{plot_timesfm_mape:.6f},"
            f"{plot_timesfm_r2:.6f},{plot_timesfm_r2k:.6f},"
            f"{eval_filter_name},{actual_oos_rows},{expected_oos_for_csv},"
            f"{w_recon_f:.8f},{w_pred_f:.8f},{w_drv_f:.8f},{w_reg_f:.8f},"
            f"{float(args.lambda_ortho_time):.8f},{float(args.lambda_frobenius_time):.8f}\n"
        )

        write_header = not os.path.exists(log_path)
        with open(log_path, "a") as f:
            if write_header:
                f.write(header)
            f.write(row)
        print(f"\nSaved metrics log → {log_path}")

        # Save and preview first 50 rows of all test predictions
        # ADD ↓↓↓
        out_all_csv = os.path.join(
            args.output_dir,
            f"4_losses_rolling_{trail}_test_predictions_{args.run_hash}.csv",  # str; unique
        )
        # ADD ↑↑↑
        all_df.to_csv(out_all_csv, index=False)
        print("\nFirst 50 GroundTruth vs Forecasted rows (across all test steps):")
        print(all_df.head(50).to_string(index=False))
        print(f"\nSaved full table to: {out_all_csv}")
        # ------------------------ 3-line OOS plot (UPDATED: multi-firm fiber) ------------------------
        #
        # Advisor requirement:
        #   Plot, over OUT-OF-SAMPLE times, the feature series for ALL firms (excluding firms whose
        #   ground-truth at that time is missing), in a "flattened fiber" ordering:
        #
        #     [ firm1@t0, firm2@t0, ..., firmM@t0,  firm1@t1, firm2@t1, ..., ]
        #
        #   where t0 = first OOS time index (in this script, min(all_df["t"])) and firm indices are
        #   sorted ascending within each time.
        #
        # We plot 3 lines:
        #   (1) Raw baseline (Ground Truth)
        #   (2) TimesFM baseline (0-shot, univariate per firm, expanding window)
        #   Firm 1, predict all feature; firm 2, predict all feature, ..., firm M, predict all feature; then t+1. No for loop.
        #   cross-section too big. Company by company. 
        #   (3) Tensor-TimesFM (your model) predictions (already in all_df["Forecasted"])
        #
        # ------------------------ 3-line OOS plot (UPDATED: multi-firm fiber) ------------------------
        if args.make_fiber_forecast_plot:
            # Baseline inference + all metrics are already computed above.
            # This block only renders the fiber plot using the stored tensors.
            if (
                len(all_test_rows) > 0
                and baseline_ok
                and gt_plot_t is not None
                and pred_timesfm_plot_t is not None
            ):
                raw_truth_fiber_value_array_1d = gt_plot_t.numpy()
                tensor_fiber_value_array_1d = pred_tensor_plot_t.numpy()
                timesfm_plot_fiber_value_array_1d = pred_timesfm_plot_t.numpy()

                plot_time_index_array_1d = np.sort(
                    plot_df["t"].astype(int).unique()
                )
                plot_firm_index_list_per_time = [
                    plot_df.loc[plot_df["t"] == t, "firm"].values
                    for t in plot_time_index_array_1d
                ]

                segment_start_position_list = [0]
                running_position = 0
                for firms in plot_firm_index_list_per_time:
                    running_position += len(firms)
                    segment_start_position_list.append(running_position)
                segment_start_position_list = segment_start_position_list[:-1]

                flattened_index_array_1d = np.arange(
                    raw_truth_fiber_value_array_1d.shape[0], dtype=np.int64
                )

                plt.figure(figsize=(16, 6))
                plt.plot(
                    flattened_index_array_1d,
                    raw_truth_fiber_value_array_1d,
                    label="Raw baseline (Ground Truth)",
                    linewidth=1.2,
                )
                plt.plot(
                    flattened_index_array_1d,
                    timesfm_plot_fiber_value_array_1d,
                    label="TimesFM baseline (0-shot)",
                    linewidth=1.2,
                )
                plt.plot(
                    flattened_index_array_1d,
                    tensor_fiber_value_array_1d,
                    label="Tensor-TimesFM (your model)",
                    linewidth=1.2,
                )

                for boundary_position in segment_start_position_list[1:]:
                    plt.axvline(x=int(boundary_position), linewidth=0.5, alpha=0.15)

                plt.title(
                    f"Multi-firm OOS fiber plot | feature_index={plot_feature_index} | "
                    f"t in [{int(plot_time_index_array_1d[0])}, {int(plot_time_index_array_1d[-1])}]"
                )
                plt.xlabel("Flattened index (time-major; firms inside each time)")
                plt.ylabel("Feature value")
                plt.legend(loc="upper left")

                ax = plt.gca()
                ax.text(
                    0.99, 0.96, stats_text,
                    transform=ax.transAxes,
                    fontsize=8,
                    family="monospace",
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="gray"),
                )

                plt.tight_layout()

                plot_png_filename = (
                    args.plot_png_filename.strip()
                    if args.plot_png_filename.strip()
                    else f"oos_multifirm_fiber_plot_feat{plot_feature_index}_hash_{args.run_hash}.png"
                )
                plot_png_path = os.path.join(args.output_dir, plot_png_filename)
                plt.savefig(plot_png_path, dpi=600)
                plt.close()
                print(
                    f"[PLOT] Saved strictly 1-step rolling multi-firm 3-line plot to: {plot_png_path}"
                )
            else:
                print(
                    "[PLOT] Skipping fiber plot: no test rows, baseline failed, "
                    "or no valid plot-feature data."
                )
    print("\nDone.")

    wandb.summary["run/hash"] = args.run_hash
    wandb.finish()


if __name__ == "__main__":
    main()

    # w_recon = 0. --> if still >= 74% r^2, questions about the tensor, is that useful?

    # random shuffle

    # w_recon=1e-5 to 1e-1, w_pred=1e-5 to 1e-1. w_forecast = 1 fixed.

    # big discrepency

"""
https://wrds-www.wharton.upenn.edu/pages/get-data/beta-suite-wrds/beta-suite-by-wrds/
"""
