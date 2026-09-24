#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoupledTensorReconstruction_CV.py

This program performs coupled reconstruction on three tensors:
  - T1 (time, item_sales, store): Reconstructed by learned embeddings + MLP_T1.
  - T2 (time, item_prices, store): Reconstructed by MLP_T2 on concatenated time/item/store embeddings.
  - T3 (time, features): Reconstructed by MLP_T3 on the concatenation of time embedding and an effective feature embedding.

Forecasting is done with **TimesFM** (pretrained, zero-shot). The overall training loss combines:
  - loss_recon: coupled reconstruction (T1/T2/T3)
  - loss_forecast: a TimesFM-driven forecast consistency term over short windows

Weights:
  - Either trainable via softmax so w_recon + w_forecast = 1,
  - Or fixed (grid search).

Key adjustments in this version:
  1. Two-loss design: total_loss = w_recon * loss_recon + w_forecast * loss_forecast (+ small orthogonality reg).
  2. TimesFM is inference-only; no diffusion/prediction losses or MR-Diffusion params anywhere.
  3. Clearer names: no weight_rnn/weight_pred. We use (w_recon, w_forecast) only.
  4. Storage filenames use a TimesFM prefix (no MR-Diff params).
  5. Small bugfix: n_time = min(original_time, 150) for quick runs.

Nov 17, 2025:

1. Turn off Tensor 2 (items_prices), Tensor 3 (time, event). Focus on Tensor 1 only for debugging. (Tensor 1 \rightarrow Targets)
2. Check the results compared to the couple tensor.

3. Later on ==> concat (items_prices, items_sales) as Tensor 1 ==> Target is still items_sales.

4. Later on ==> matrix = time x event_types (17)
"""

import argparse
import math
import numpy as np
import os
import datetime

# ============================================================
# [BLOCK 1 FIX] Force HuggingFace Offline Mode
# ============================================================
# TimesFM is ALREADY cached (you checked on Sep 13), so force offline mode
import os

# type: str / meaning: directory where HF models are cached persistently
HF_CACHE_DIR: str = os.path.expanduser("~/.cache/huggingface")

# Verify the cache exists (should print True)
cache_exists: bool = os.path.exists(
    os.path.join(HF_CACHE_DIR, "hub", "models--google--timesfm-2.0-500m-pytorch")
)
print(f"[CACHE] TimesFM cache exists: {cache_exists}")

if not cache_exists:
    # [DEBUG FIX] This is now just a warning. We will allow the
    # script to download if needed by *not* forcing offline mode.
    print(
        "[CACHE WARNING] TimesFM not cached! The script will now attempt to download it."
    )
    # import sys
    # sys.exit(1) # Do not exit, allow download

# Environment variables that control HuggingFace behavior
os.environ["HF_HOME"] = HF_CACHE_DIR  # (str) main HF cache location
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR  # (str) transformers cache
os.environ["HF_DATASETS_CACHE"] = HF_CACHE_DIR  # (str) datasets cache

# ✅ ENABLE OFFLINE MODE (model is already cached)
# [DEBUG FIX] Disabling forced offline mode to allow download on first run.
# os.environ['TRANSFORMERS_OFFLINE'] = '1'  # (str) '1' = offline mode
# os.environ['HF_HUB_OFFLINE'] = '1'  # (str) '1' = offline mode

print(f"[CACHE] HuggingFace cache directory: {HF_CACHE_DIR}")
print(f"[CACHE] Offline mode: DISABLED (will download if needed)")
print(f"[CACHE] This prevents any network downloads during training")
# ============================================================
# ============================================================
import sys
import time
import csv
import random
from itertools import cycle
from tqdm import tqdm
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import wandb
import timesfm  # pretrained TimesFM – inference only (zero-shot). See HF model card.

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = (
    True  # prints: (no shape); meaning: allow fast TF32 matmuls
)
torch.set_float32_matmul_precision(
    "high"
)  # prints: (no shape); meaning: favor speed for FP32 matmuls
# ADD ↓↓↓  (orthogonality / decorrelation helpers)
# ------------------------------------------------------------
# WandB Initialization
# ------------------------------------------------------------
wandb.init(project="coupled_tensor_forecasting_cv")
config = wandb.config

# ============================================================
# [DEBUG DELETE] Removed the [TEST] block that loaded TimesFM.
# We now load it *once* in the main() function to fix the
# "load again and again" problem.
# ============================================================

print("[DEBUG] TimesFM will be loaded once inside the main() function.")


# ============================================================
# [BLOCK 2 FIX] Epoch Timer for Diagnostics
# ============================================================
class EpochTimer:
    """
    Simple timer to track what's taking time during training.
    Prints warnings if any operation takes >5 minutes.
    """

    def __init__(self):
        # type: float / meaning: timestamp when epoch started
        self.epoch_start: float = 0.0
        # type: float / meaning: timestamp of last operation
        self.last_op: float = 0.0
        # type: str / meaning: name of last operation
        self.last_op_name: str = ""

    def start_epoch(self, epoch_num: int):
        """Call at the beginning of each epoch."""
        # type: int / meaning: current epoch number
        self.epoch_num: int = epoch_num
        # time.time(): (function) returns current Unix timestamp (float seconds)
        self.epoch_start = time.time()
        self.last_op = time.time()
        self.last_op_name = "epoch_start"
        print(f"\n[TIMER] Epoch {epoch_num} started at {time.strftime('%H:%M:%S')}")

    def checkpoint(self, operation_name: str):
        """
        Call after each major operation (e.g., 'reconstruction', 'forecast', 'validation').
        Prints a warning if the operation took >5 minutes.
        """
        # type: float / meaning: current timestamp
        now: float = time.time()
        # type: float / meaning: seconds elapsed since last checkpoint
        elapsed: float = now - self.last_op

        # Print timing for this operation
        print(f"[TIMER]   {operation_name}: {elapsed:.1f}s")

        # Warning if operation took >5 minutes (300 seconds)
        if elapsed > 300:
            print(
                f"[TIMER WARNING] ⚠️  {operation_name} took {elapsed/60:.1f} minutes!"
            )
            print(f"[TIMER WARNING]     Previous operation: {self.last_op_name}")

            # Check GPU memory
            if torch.cuda.is_available():
                # torch.cuda.memory_allocated(): (function) returns bytes of GPU memory in use
                gpu_mem_gb: float = torch.cuda.memory_allocated() / 1e9
                print(f"[TIMER WARNING]     GPU memory: {gpu_mem_gb:.2f} GB")

        # Update for next checkpoint
        self.last_op = now
        self.last_op_name = operation_name

    def end_epoch(self):
        """Call at the end of each epoch."""
        # type: float / meaning: total seconds for this epoch
        total_time: float = time.time() - self.epoch_start
        print(f"[TIMER] Epoch {self.epoch_num} total: {total_time/60:.1f} minutes\n")


# ============================================================


# ------------------------------------------------------------
# Metric Functions (for T1 only)
# ------------------------------------------------------------
def rmse_torch(y_true, y_pred):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


def r2_torch(y_true, y_pred, eps=1e-12):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    mean_y = torch.mean(y_true)
    ss_tot = torch.sum((y_true - mean_y) ** 2)
    ss_tot = torch.clamp(ss_tot, min=eps)
    return 1.0 - ss_res / ss_tot


def r2_kelly_torch(y_true, y_pred):
    diff = torch.sum((y_true - y_pred) ** 2) / torch.sum(y_true**2)
    return 1 - diff


def wrmsse_torch(indices, y_true, y_pred, train_data, eps=1e-8):
    unique_series = torch.unique(indices[:, 1:], dim=0)
    rmsse_list = []
    for key in unique_series:
        item = key[0].item()
        store = key[1].item()
        mask = (indices[:, 1] == key[0]) & (indices[:, 2] == key[1])
        if mask.sum() == 0:
            continue
        y_true_series = y_true[mask]
        y_pred_series = y_pred[mask]
        mse_forecast = torch.mean((y_true_series - y_pred_series) ** 2)
        rmse_forecast = torch.sqrt(mse_forecast)
        series_train = train_data[:, item, store]
        nonzero = (series_train != 0).nonzero(as_tuple=False)
        if nonzero.numel() == 0:
            scale = torch.tensor(1.0, device=series_train.device)
        else:
            first_nonzero = nonzero[0].item()
            if first_nonzero >= series_train.numel() - 1:
                scale = torch.tensor(1.0, device=series_train.device)
            else:
                diffs = (
                    series_train[first_nonzero + 1 :] - series_train[first_nonzero:-1]
                )
                if diffs.numel() == 0:
                    scale = torch.tensor(1.0, device=series_train.device)
                else:
                    scale = torch.sqrt(torch.mean(diffs**2))
                    if scale.item() == 0:
                        scale = torch.tensor(1.0, device=series_train.device)
        rmsse = rmse_forecast / (scale + eps)
        rmsse_list.append(rmsse)
    if len(rmsse_list) == 0:
        return torch.tensor(0.0, device=train_data.device)
    overall_wrmsse = sum(rmsse_list) / len(rmsse_list)
    return overall_wrmsse


# ------------------------------------------------------------
# Additional Helper Functions
# ------------------------------------------------------------
def measure_component_times(model, batch_T1, batch_T2, batch_T3, forecast_steps):
    """Measures inference time for reconstruction vs. forecasting steps."""
    model.eval()
    with torch.no_grad():
        # type: float / meaning: start timestamp
        start_time = time.time()
        # type: tuple(torch.Tensor) / meaning: (loss, ortho_penalty)
        _ = model(
            mode="reconstruction_loss",
            batch_T1=batch_T1,
            batch_T2=batch_T2,
            batch_T3=batch_T3,
        )
        # type: float / meaning: elapsed time for reconstruction
        cp_time = time.time() - start_time

        # --- Get history indices for forecast call ---
        # type: int / meaning: history length
        hist_T: int = int(model.train_time)
        hist_T = max(1, min(hist_T, model.emb_time.num_embeddings))
        # type: torch.Tensor / shape: [L] / meaning: [0, ..., L-1]
        history_indices: torch.Tensor = torch.arange(
            hist_T,
            device=model.device,
            dtype=torch.long,
        )

        start_time = time.time()
        # type: torch.Tensor / shape: [H, R] / meaning: predicted embeddings
        _ = model.get_future_time_embeddings_on_graph(
            history_time_indices=history_indices, forecast_length=forecast_steps
        )
        # type: float / meaning: elapsed time for TimesFM forecast
        timesfm_time = time.time() - start_time
    return cp_time, timesfm_time


def count_missing_values(tensor):
    return torch.isnan(tensor).sum().item()


def sliding_window_validation(
    model, T1_train, lag_p, forecast_horizon=28, device="cpu"
):
    """
    Use a sliding (rolling) validation on the training range.
    For each window, call `compute_forecast_loss` to get an MSE loss
    between TimesFM-based forecasts and the ground truth in T1_train.
    """
    # type: torch.nn.Module / meaning: model with TimesFM + embeddings
    model.eval()  # put model in eval mode (no dropout / batchnorm update)

    # type: int / meaning: number of time steps in this training tensor
    train_time = T1_train.shape[0]

    # type: float / meaning: sum of validation losses over all windows
    total_loss: float = 0.0

    # type: int / meaning: how many validation windows we actually used
    count: int = 0

    # tqdm: common helper to show a progress bar in the terminal
    for start in tqdm(
        range(0, train_time - lag_p, forecast_horizon),
        desc="Lag p validation windows",
        leave=False,
    ):
        # type: int / meaning: how many steps to forecast in this window
        # `start + lag_p` is the last history index; we ensure that the
        # future block stays inside [0, train_time - 1]
        steps: int = min(forecast_horizon, train_time - (start + lag_p) - 1)

        # if we have no room to forecast, stop
        if steps <= 0:
            break

        # We now reuse the helper that already does:
        #  - build history indices,
        #  - call TimesFM on-graph (get_future_time_embeddings_on_graph),
        #  - run model(mode="forecast_T1", ...),
        #  - compute MSE on the window [start_idx+1 ... start_idx+steps].
        #
        # type: torch.Tensor / shape: scalar / meaning: validation loss as MSE on this block
        loss_tensor = compute_forecast_loss(
            model=model,
            T1_train=T1_train,
            device=device,
            window=steps,
            # `start + lag_p` = same "start_idx" you used before:
            #   history up to this time, forecast the next `steps` days
            start_idx=start + lag_p,
        )

        # .item(): common PyTorch method to turn a 0-dim tensor into a Python float
        total_loss += float(loss_tensor.item())
        count += 1

    # avoid division by zero; if no window, return +inf as before
    return total_loss / count if count > 0 else float("inf")


# ------------------------------------------------------------
# Dataset Classes
# ------------------------------------------------------------
class Tensor3DDataset(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor
        self.indices = np.array(list(np.ndindex(tensor.shape)))
        self.values = tensor.detach().cpu().numpy().flatten()

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return torch.tensor(self.indices[idx], dtype=torch.long), self.values[idx]


class Tensor2DDataset(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor
        self.indices = np.array(list(np.ndindex(tensor.shape)))
        self.values = tensor.detach().cpu().numpy().flatten()

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return torch.tensor(self.indices[idx], dtype=torch.long), self.values[idx]


# ===================== TimesFM helpers + orthogonality =====================
# (import is now at top of file)


def _center_time(embedding_over_time: torch.Tensor) -> torch.Tensor:
    if embedding_over_time.dim() == 2:
        return embedding_over_time - embedding_over_time.mean(dim=0, keepdim=True)
    elif embedding_over_time.dim() == 3:
        return embedding_over_time - embedding_over_time.mean(dim=1, keepdim=True)
    else:
        raise ValueError("Expected (T,R) or (B,T,R)")


def _covariance_centered(centered_over_time: torch.Tensor) -> torch.Tensor:
    if centered_over_time.dim() == 2:
        T = centered_over_time.shape[0]
        return (centered_over_time.T @ centered_over_time) / max(T, 1)
    else:
        B, T, R = centered_over_time.shape
        return torch.matmul(
            centered_over_time.transpose(1, 2), centered_over_time
        ) / max(T, 1)


def _zero_diagonal(square_matrix: torch.Tensor) -> torch.Tensor:
    if square_matrix.dim() == 2:
        return square_matrix - torch.diag(torch.diag(square_matrix))
    else:
        return square_matrix - torch.diag_embed(
            torch.diagonal(square_matrix, dim1=-2, dim2=-1)
        )


def decorrelation_L2(embedding_over_time: torch.Tensor) -> torch.Tensor:
    centered = _center_time(embedding_over_time)
    covariance = _covariance_centered(centered)
    offdiag = _zero_diagonal(covariance)
    return (
        (offdiag.pow(2)).mean()
        if offdiag.dim() == 2
        else (offdiag.pow(2)).mean(dim=(1, 2)).mean()
    )


# ===============================================================================


class CoupledCPDecomposition(nn.Module):
    def __init__(
        self,
        shape_t1,
        shape_t3,
        rank_cp,
        train_time,
        # [DEBUG REFACTOR] Add pre-loaded TimesFM model and hparams
        timesfm_model_instance,
        timesfm_hparams_instance,
        mlp_hidden_T1=1024,
        mlp_hidden_T2=512,
        mlp_hidden_T3=256,
        # --- NEW: Argument for total layers in MLP_T1 ---
        # (int) / meaning: The total number of Linear layers for MLP_T1 (must be >= 2)
        total_mlp_layers_t1=3,
        # --- End New ---
        lambda_param=1.0,
        w1=0.33,
        w2=0.33,
        device="cpu",
        static_features_T3=None,
        static_dim_T3=None,
        dropout_prob=0.5,
        train_weights=True,
        w_recon_fixed=1.0,
    ):
        super(CoupledCPDecomposition, self).__init__()
        self.device = device
        self.train_weights: bool = bool(
            train_weights
        )  # controls trainable vs fixed weights
        # --- NEW: Store MLP_T1 layer count and validate it ---
        # (int) / meaning: Total number of Linear layers for MLP_T1.
        # We use max(2, ...) to ensure it's at least 2 (one input-to-hidden, one hidden-to-output).
        self.total_mlp_layers_t1 = max(2, int(total_mlp_layers_t1))
        # --- End New ---
        n_time, n_item, n_store = shape_t1
        n_time2, n_features = shape_t3
        assert n_time == n_time2, "T1/T2 and T3 must share the same time dimension."
        self.train_time = train_time

        # Embeddings
        self.emb_time = nn.Embedding(n_time, rank_cp)
        self.emb_item_T1 = nn.Embedding(n_item, rank_cp)
        self.emb_store_T1 = nn.Embedding(n_store, rank_cp)
        self.emb_item_T2 = nn.Embedding(n_item, rank_cp)
        self.emb_store_T2 = nn.Embedding(n_store, rank_cp)
        self.emb_feature = nn.Embedding(n_features, rank_cp)

        if static_features_T3 is None or static_dim_T3 is None:
            raise ValueError("Must supply static_features_T3 and static_dim_T3 for T3.")
        self.register_buffer("static_T3", static_features_T3)
        self.static_proj_T3 = nn.Linear(static_dim_T3, rank_cp)

        for emb in [
            self.emb_time,
            self.emb_item_T1,
            self.emb_store_T1,
            self.emb_item_T2,
            self.emb_store_T2,
            self.emb_feature,
        ]:
            nn.init.xavier_uniform_(emb.weight)

        # (float) / meaning: Probability of an element to be zeroed.
        self.dropout_prob = dropout_prob

        # --- NEW: Build MLP_T1 dynamically (Goal 1: n adjustable layers) ---

        # (list[nn.Module]) / meaning: An empty list to hold the layers for MLP_T1.
        mlp_t1_layers_list = []

        # (int) / meaning: The input feature size for T1 (time + item + store).
        input_dimension_t1 = 3 * rank_cp
        # (int) / meaning: The hidden feature size for T1, e.g., 1024.
        hidden_dimension_t1 = mlp_hidden_T1

        # --- First Layer (Input -> Hidden) ---
        # (nn.Linear) / shape: [Batch, 3*rank_cp] -> [Batch, mlp_hidden_T1]
        first_layer = nn.Linear(input_dimension_t1, hidden_dimension_t1)
        # Add the first layer to our list.
        mlp_t1_layers_list.append(first_layer)
        # (nn.ReLU) / meaning: Rectified Linear Unit activation function.
        mlp_t1_layers_list.append(nn.ReLU())
        # (nn.Dropout) / meaning: Dropout layer for regularization.
        mlp_t1_layers_list.append(nn.Dropout(self.dropout_prob))

        # --- Intermediate Hidden Layers ---
        # This loop creates the adjustable number of hidden layers.
        # We subtract 2 (for the first layer we just added and the final output layer).
        # (int) / meaning: The number of "hidden-to-hidden" layers to create.
        number_of_intermediate_layers = self.total_mlp_layers_t1 - 2

        # This loop will run 'number_of_intermediate_layers' times.
        # If total_mlp_layers_t1 is 2 or 3, this loop is skipped.
        for i in range(number_of_intermediate_layers):
            # (nn.Linear) / shape: [Batch, mlp_hidden_T1] -> [Batch, mlp_hidden_T1]
            hidden_layer = nn.Linear(hidden_dimension_t1, hidden_dimension_t1)
            # Add a hidden-to-hidden layer.
            mlp_t1_layers_list.append(hidden_layer)
            # (nn.ReLU) / meaning: ReLU activation function.
            mlp_t1_layers_list.append(nn.ReLU())
            # (nn.Dropout) / meaning: Dropout layer.
            mlp_t1_layers_list.append(nn.Dropout(self.dropout_prob))

        # --- Last Layer (Hidden -> Output) ---
        # (nn.Linear) / shape: [Batch, mlp_hidden_T1] -> [Batch, 1]
        output_layer = nn.Linear(hidden_dimension_t1, 1)
        # Add the final output layer to the list.
        mlp_t1_layers_list.append(output_layer)

        # (nn.Sequential) / meaning: A container that chains modules together in order.
        # nn.Sequential: uncommon function, it takes a list of modules and runs data
        # through them sequentially. The * unpacks the list.
        self.mlp_T1 = nn.Sequential(*mlp_t1_layers_list)
        # --- End of MLP_T1 build ---

        # --- NEW: Build MLP_T2 (Goal 2: 3 hidden layers) ---
        # (int) / meaning: Input feature size for T2 (time + item + store).
        input_dimension_t2 = 3 * rank_cp
        # (int) / meaning: Hidden feature size for T2, e.g., 512.
        hidden_dimension_t2 = mlp_hidden_T2

        self.mlp_T2 = nn.Sequential(
            # Layer 1 (Input -> Hidden1)
            # (nn.Linear) / shape: [Batch, 3*rank_cp] -> [Batch, mlp_hidden_T2]
            nn.Linear(input_dimension_t2, hidden_dimension_t2),
            # (nn.Tanh) / meaning: Tanh activation function, scales output to [-1, 1].
            nn.Tanh(),
            # (nn.Dropout) / meaning: Dropout layer.
            nn.Dropout(self.dropout_prob),
            # Layer 2 (Hidden1 -> Hidden2)
            # (nn.Linear) / shape: [Batch, mlp_hidden_T2] -> [Batch, mlp_hidden_T2]
            nn.Linear(hidden_dimension_t2, hidden_dimension_t2),
            # (nn.Tanh) / meaning: Tanh activation function.
            nn.Tanh(),
            # (nn.Dropout) / meaning: Dropout layer.
            nn.Dropout(self.dropout_prob),
            # Layer 3 (Hidden2 -> Hidden3)
            # (nn.Linear) / shape: [Batch, mlp_hidden_T2] -> [Batch, mlp_hidden_T2]
            nn.Linear(hidden_dimension_t2, hidden_dimension_t2),
            # (nn.Tanh) / meaning: Tanh activation function.
            nn.Tanh(),
            # (nn.Dropout) / meaning: Dropout layer.
            nn.Dropout(self.dropout_prob),
            # Layer 4 (Hidden3 -> Output)
            # (nn.Linear) / shape: [Batch, mlp_hidden_T2] -> [Batch, 1]
            nn.Linear(hidden_dimension_t2, 1),
        )
        # --- End of MLP_T2 build ---

        # --- NEW: Build MLP_T3 (Goal 2: 3 hidden layers) ---
        # (int) / meaning: Input feature size for T3 (time + feature).
        input_dimension_t3 = 2 * rank_cp
        # (int) / meaning: Hidden feature size for T3, e.g., 256.
        hidden_dimension_t3 = mlp_hidden_T3

        self.mlp_T3 = nn.Sequential(
            # Layer 1 (Input -> Hidden1)
            # (nn.Linear) / shape: [Batch, 2*rank_cp] -> [Batch, mlp_hidden_T3]
            nn.Linear(input_dimension_t3, hidden_dimension_t3),
            # (nn.LeakyReLU) / meaning: Leaky ReLU activation, allows small negative values.
            nn.LeakyReLU(negative_slope=0.2),
            # (nn.Dropout) / meaning: Dropout layer.
            nn.Dropout(self.dropout_prob),
            # Layer 2 (Hidden1 -> Hidden2)
            # (nn.Linear) / shape: [Batch, mlp_hidden_T3] -> [Batch, mlp_hidden_T3]
            nn.Linear(hidden_dimension_t3, hidden_dimension_t3),
            # (nn.LeakyReLU) / meaning: Leaky ReLU activation.
            nn.LeakyReLU(negative_slope=0.2),
            # (nn.Dropout) / meaning: Dropout layer.
            nn.Dropout(self.dropout_prob),
            # Layer 3 (Hidden2 -> Hidden3)
            # (nn.Linear) / shape: [Batch, mlp_hidden_T3] -> [Batch, mlp_hidden_T3]
            nn.Linear(hidden_dimension_t3, hidden_dimension_t3),
            # (nn.LeakyReLU) / meaning: Leaky ReLU activation.
            nn.LeakyReLU(negative_slope=0.2),
            # (nn.Dropout) / meaning: Dropout layer.
            nn.Dropout(self.dropout_prob),
            # Layer 4 (Hidden3 -> Output)
            # (nn.Linear) / shape: [Batch, mlp_hidden_T3] -> [Batch, 1]
            nn.Linear(hidden_dimension_t3, 1),
        )
        # --- End of MLP_T3 build ---

        # --- [DEBUG REFACTOR] ---
        # Removed the TimesFM loading logic from here.
        # We now accept the pre-loaded model as an argument.

        # Raw nn.Module (pre-loaded, frozen, on-device)
        self.timesfm_model = timesfm_model_instance
        # Hparams (pre-loaded)
        self.timesfm_hparams = timesfm_hparams_instance

        # --- End of refactor ---

        # Small attribute: we will reuse this default H if caller does not pass one
        self.default_forecast_horizon: int = 28  # (int) Walmart uses H=28

        # TimesFM config (inference only)
        self.timesfm_repo_id: str = "google/timesfm-2.0-500m-pytorch"
        self.timesfm_context_cap: int = 2048
        self.timesfm_per_core_bsz: int = 32
        self.timesfm_freq_category: int = 0  # daily-like

        # Orthogonality regularization on time factors (history)
        self.lambda_ortho_time: float = 0.01

        # Loss & weighting
        self.mse = nn.MSELoss()
        self.lambda_param = lambda_param
        # self.w1 = w1
        # self.w2 = w2
        # New: Set up weights for reconstruction losses (for T1, T2, T3)
        desired_reconstruction_weights_vector = torch.tensor(
            [w1, w2, 1 - w1 - w2], dtype=torch.float32
        )  # type: torch.Tensor, shape: (3,), meaning: starting weights for T1/T2/T3 losses, add to 1
        print(w1, w2)
        if self.train_weights:
            self.reconstruction_logits_parameters = nn.Parameter(
                torch.log(desired_reconstruction_weights_vector + 1e-10)
            )  # type: torch.nn.Parameter, shape: (3,), meaning: trainable starting logs for softmax to get weights; nn.Parameter: uncommon, makes this updateable by optimizer
        else:
            self.register_buffer(
                "reconstruction_weights_fixed_buffer",
                desired_reconstruction_weights_vector,
            )  # type: torch.Tensor, shape: (3,), meaning: fixed weights if not trainable; register_buffer: uncommon, stores non-trainable tensor on device

        # Change: For overall weights (recon vs forecast), init to match w_recon_fixed
        desired_overall_weights_vector = torch.tensor(
            [w_recon_fixed, 1], dtype=torch.float32
        )  # type: torch.Tensor, shape: (2,), meaning: starting weights for recon/forecast, add to 1
        if self.train_weights:
            self.loss_params = nn.Parameter(
                torch.log(desired_overall_weights_vector + 1e-10)
            )  # type: torch.nn.Parameter, shape: (2,), meaning: trainable logs for overall softmax
        else:
            # fixed weights (already in code, no change here)
            w_recon = float(w_recon_fixed)
            w_forecast = max(1.0, 1.0 - w_recon)
            self.register_buffer(
                "fixed_weights",
                torch.tensor([w_recon, w_forecast], dtype=torch.float32),
            )

        self.to(device)

    def get_future_time_embeddings_on_graph(
        self,
        history_time_indices: torch.Tensor,  # (LongTensor) [L] times used as context
        forecast_length: int,  # (int) H steps to predict
    ) -> torch.Tensor:
        """
        Use frozen TimesFM to forecast future time-embedding vectors ON-GRAPH.
        Grads flow to self.emb_time via history lookup. TimesFM params stay frozen.
        Returns: (FloatTensor) [H, R] where R = rank_cp
        """
        # 1) Take learned history embeddings: [L, R]
        time_embedding_history: torch.Tensor = self.emb_time(
            history_time_indices
        )  # [L, R] float

        # 2) TimesFM treats batch as 'R' series; transpose to [R, L]
        input_rank_by_len: torch.Tensor = time_embedding_history.transpose(
            0, 1
        )  # [R, L]

        # 3) Pad or truncate to model context length C
        C_model: int = int(self.timesfm_hparams.context_len)  # (int)
        R_rank: int = int(input_rank_by_len.shape[0])  # (int)
        L_hist: int = int(input_rank_by_len.shape[1])  # (int)

        if L_hist < C_model:
            # left pad with zeros to length C: result [R, C]
            input_ts: torch.Tensor = F.pad(
                input_rank_by_len, (C_model - L_hist, 0), "constant", 0.0
            )
            paddings_hist: torch.Tensor = F.pad(  # [R, C]
                torch.zeros(R_rank, L_hist, device=self.device, dtype=torch.float32),
                (C_model - L_hist, 0),
                "constant",
                1.0,
            )
        else:
            # keep last C entries: result [R, C]
            input_ts = input_rank_by_len[:, -C_model:]  # [R, C]
            paddings_hist = torch.zeros(
                R_rank, C_model, device=self.device, dtype=torch.float32
            )  # [R, C]

        # 4) Make paddings for the forecast part [R, H]
        H: int = int(forecast_length)  # (int)
        paddings_future: torch.Tensor = torch.zeros(
            R_rank, H, device=self.device, dtype=torch.float32
        )  # [R, H]

        # 5) Full paddings [R, C+H]
        paddings_full: torch.Tensor = torch.cat(
            [paddings_hist, paddings_future], dim=1
        )  # [R, C+H]

        # 6) Frequency tag (R rows)
        freq_tensor: torch.Tensor = torch.full(
            (R_rank, 1),
            int(self.timesfm_freq_category),
            device=self.device,
            dtype=torch.long,
        )  # [R, 1] long

        # 7) Decode on-graph (TimesFM weights are frozen, but inputs track grad)
        mean_output, _ = self.timesfm_model.decode(
            input_ts=input_ts,  # [R, C]
            paddings=paddings_full,  # [R, C+H]
            freq=freq_tensor,  # [R, 1]
            horizon_len=H,  # (int) H
            output_patch_len=int(self.timesfm_hparams.output_patch_len),  # (int)
            max_len=C_model,  # (int)
            return_forecast_on_context=False,  # (bool)
        )  # returns ( [R, H], ... )

        # 8) Back to [H, R] to match your _forward_T1 usage
        future_matrix_hr: torch.Tensor = mean_output.transpose(
            0, 1
        ).contiguous()  # [H, R] float
        return future_matrix_hr

    # --- helpers ---
    def get_effective_weights(self):
        if self.train_weights:
            eff = torch.softmax(self.loss_params, dim=0)  # sum to 1
            return eff[0], eff[1]
        else:
            s = self.fixed_weights.sum()
            s = 1
            return (self.fixed_weights[0] / s), (self.fixed_weights[1] / s)

    def get_reconstruction_weights(self):
        if self.train_weights:
            return torch.softmax(
                self.reconstruction_logits_parameters, dim=0
            )  # type: torch.Tensor, shape: (3,), meaning: compute positive weights summing to 1 from logits; torch.softmax: uncommon, turns any numbers to probabilities
        else:
            return (
                self.reconstruction_weights_fixed_buffer
            )  # type: torch.Tensor, shape: (3,), meaning: return fixed weights

    # --- tensor heads ---
    def _forward_T1(self, indices, forecasted_embeddings=None):
        time_idx = indices[:, 0].long()
        item_idx = indices[:, 1].long()
        store_idx = indices[:, 2].long()
        if forecasted_embeddings is None:
            time_emb = self.emb_time(time_idx)
        else:
            time_emb = forecasted_embeddings[time_idx]
        item_emb = self.emb_item_T1(item_idx)
        store_emb = self.emb_store_T1(store_idx)
        concat = torch.cat([time_emb, item_emb, store_emb], dim=1)
        pred = self.mlp_T1(concat).squeeze(1)
        return pred

    def _forward_T2(self, indices):
        time_idx = indices[:, 0].long()
        item_idx = indices[:, 1].long()
        store_idx = indices[:, 2].long()
        time_emb = self.emb_time(time_idx)
        item_emb = self.emb_item_T2(item_idx)
        store_emb = self.emb_store_T2(store_idx)
        concat = torch.cat([time_emb, item_emb, store_emb], dim=1)
        pred = self.mlp_T2(concat).squeeze(1)
        return pred

    def _forward_T3(self, indices):
        time_idx = indices[:, 0].long()
        feat_idx = indices[:, 1].long()
        time_emb = self.emb_time(time_idx)
        feat_emb = self.emb_feature(feat_idx)
        static_feat = self.static_T3[feat_idx]
        proj_feat = self.static_proj_T3(static_feat)
        feat_eff = feat_emb + proj_feat
        concat = torch.cat([time_emb, feat_eff], dim=1)
        pred = self.mlp_T3(concat).squeeze(1)
        return pred

    # === NEW: Multi-modal forward method (replaces old 'forward') ===
    def forward(self, mode: str, **inputs):
        """
        Main forward method, mimics the 'Rideshare' script logic.

        Args:
            mode (str): One of 'reconstruction_loss', 'forecast_T1'.
            **inputs: Keyword arguments based on mode.

        Returns:
            - if mode == 'reconstruction_loss': (loss_recon, loss_ortho)
            - if mode == 'forecast_T1': (predictions)
        """
        # --- Mode 1: Compute Reconstruction Loss ---
        if mode == "reconstruction_loss":
            # type: torch.Tensor / shape: scalar / meaning: accumulator for T1 loss
            loss_T1 = torch.tensor(0.0, device=self.device)
            # type: torch.Tensor / shape: scalar / meaning: accumulator for T2 loss
            loss_T2 = torch.tensor(0.0, device=self.device)
            # type: torch.Tensor / shape: scalar / meaning: accumulator for T3 loss
            loss_T3 = torch.tensor(0.0, device=self.device)

            if "batch_T1" in inputs and inputs["batch_T1"] is not None:
                # type: torch.Tensor / shape: [B, 3] / meaning: (t, item, store) indices
                indices_T1: torch.Tensor = inputs["batch_T1"][0].to(self.device)
                # type: torch.Tensor / shape: [B] / meaning: ground truth sales values
                values_T1: torch.Tensor = inputs["batch_T1"][1].to(self.device)
                # type: torch.Tensor / shape: [B] / meaning: mask of non-NaN values
                valid_mask_T1 = ~torch.isnan(values_T1)
                if valid_mask_T1.sum() > 0:
                    # type: torch.Tensor / shape: [B_valid] / meaning: predictions for valid inputs
                    pred_T1 = self._forward_T1(
                        indices_T1[valid_mask_T1],
                        forecasted_embeddings=None,  # Use learned self.emb_time
                    )
                    # type: torch.Tensor / shape: scalar / meaning: MSE loss for T1
                    loss_T1 = self.mse(
                        pred_T1, values_T1[valid_mask_T1].float().view(-1)
                    )

            if "batch_T2" in inputs and inputs["batch_T2"] is not None:
                # type: torch.Tensor / shape: [B, 3] / meaning: (t, item, store) indices
                indices_T2: torch.Tensor = inputs["batch_T2"][0].to(self.device)
                # type: torch.Tensor / shape: [B] / meaning: ground truth price values
                values_T2: torch.Tensor = inputs["batch_T2"][1].to(self.device)
                # type: torch.Tensor / shape: [B] / meaning: mask of non-NaN values
                valid_mask_T2 = ~torch.isnan(values_T2)
                if valid_mask_T2.sum() > 0:
                    # type: torch.Tensor / shape: [B_valid] / meaning: predictions for valid inputs
                    pred_T2 = self._forward_T2(indices_T2[valid_mask_T2])
                    # type: torch.Tensor / shape: scalar / meaning: MSE loss for T2
                    loss_T2 = self.mse(
                        pred_T2, values_T2[valid_mask_T2].float().view(-1)
                    )

            if "batch_T3" in inputs and inputs["batch_T3"] is not None:
                # type: torch.Tensor / shape: [B, 2] / meaning: (t, feature) indices
                indices_T3: torch.Tensor = inputs["batch_T3"][0].to(self.device)
                # type: torch.Tensor / shape: [B] / meaning: ground truth feature values
                values_T3: torch.Tensor = inputs["batch_T3"][1].to(self.device)
                # type: torch.Tensor / shape: [B] / meaning: mask of non-NaN values
                valid_mask_T3 = ~torch.isnan(values_T3)
                if valid_mask_T3.sum() > 0:
                    # type: torch.Tensor / shape: [B_valid] / meaning: predictions for valid inputs
                    pred_T3 = self._forward_T3(indices_T3[valid_mask_T3])
                    # type: torch.Tensor / shape: scalar / meaning: MSE loss for T3
                    loss_T3 = self.mse(
                        pred_T3, values_T3[valid_mask_T3].float().view(-1)
                    )

            # --- Combine reconstruction losses ---
            # type: torch.Tensor / shape: (3,) / meaning: [w_T1, w_T2, w_T3]
            w_t1, w_t2, w_t3 = self.get_reconstruction_weights()
            # type: torch.Tensor / shape: scalar / meaning: final weighted reconstruction loss
            recon_loss = self.lambda_param * (
                w_t1 * loss_T1 + w_t2 * loss_T2 + w_t3 * loss_T3
            )

            # --- Orthogonality penalty ---
            # type: torch.Tensor / shape: [T_train, R] / meaning: history embeddings
            time_history = self.emb_time.weight[: self.train_time]
            # type: torch.Tensor / shape: scalar / meaning: L2 decorrelation penalty
            ortho_penalty = self.lambda_ortho_time * decorrelation_L2(time_history)

            return recon_loss, ortho_penalty

        # --- Mode 2: Compute T1 Forecast ---
        elif mode == "forecast_T1":
            # type: torch.Tensor / shape: [N, 3] / meaning: (t_local, item, store) indices
            indices_T1: torch.Tensor = inputs["indices_T1"]
            # type: torch.Tensor / shape: [H, R] / meaning: pre-computed future embeddings
            embeddings_block: torch.Tensor = inputs["predicted_time_embeddings_block"]

            # Call the internal helper, passing the predicted embeddings
            # type: torch.Tensor / shape: [N] / meaning: final T1 sales predictions
            predictions = self._forward_T1(
                indices_T1,
                forecasted_embeddings=embeddings_block,  # Use TimesFM-predicted embeddings
            )
            return predictions

        else:
            raise ValueError(f"Unknown mode for CoupledCPDecomposition: {mode}")


# ------------------------------------------------------------
# Per-epoch forecast loss helper (TimesFM-driven)
# ------------------------------------------------------------
def compute_forecast_loss(model, T1_train, device, window=28, start_idx=None):
    """Compute a forecast consistency loss over a short window in the training range."""
    # type: int / meaning: total time steps in the training data for this fold
    T_total = T1_train.shape[0]
    if start_idx is None:
        if T_total <= 2:
            return torch.tensor(0.0, device=device)
        # type: int / meaning: sampled start index for the forecast window
        start = max(1, min(T_total - window - 1, model.train_time - 1))
    else:
        start = int(start_idx)

    # type: int / meaning: number of steps to forecast (H), clamped to available data
    steps = min(window, T_total - (start + 1))
    if steps <= 0:
        return torch.tensor(0.0, device=device)

    # --- This logic was moved from the deleted 'forecast_time_embeddings' method ---
    # type: int / meaning: length of history (L) to use as context
    hist_T: int = int(start)
    hist_T = max(
        1, min(hist_T, model.emb_time.num_embeddings)
    )  # Clamp to [1, max_time]
    # type: torch.Tensor / shape: [L] / meaning: indices [0, 1, ..., L-1]
    history_indices: torch.Tensor = torch.arange(
        hist_T,
        device=device,
        dtype=torch.long,
    )
    # --- End moved logic ---

    # 1. Get future embeddings from TimesFM (ON-GRAPH)
    # type: torch.Tensor / shape: [H, R] / meaning: TimesFM-predicted embeddings
    femb = model.get_future_time_embeddings_on_graph(
        history_time_indices=history_indices,  # [L]
        forecast_length=steps,  # H
    )

    # 2. Build grid of indices for T1 prediction
    # type: int / meaning: number of items
    n_item = T1_train.shape[1]
    # type: int / meaning: number of stores
    n_store = T1_train.shape[2]
    # type: torch.Tensor / shape: [H] / meaning: local time indices [0, ..., H-1]
    time_indices = torch.arange(steps, device=device, dtype=torch.long)
    # type: torch.Tensor / shape: [I] / meaning: item indices [0, ..., I-1]
    item_indices = torch.arange(n_item, device=device, dtype=torch.long)
    # type: torch.Tensor / shape: [S] / meaning: store indices [0, ..., S-1]
    store_indices = torch.arange(n_store, device=device, dtype=torch.long)

    # type: tuple(torch.Tensor) / meaning: 3 tensors, each shape [H, I, S]
    t_idx, i_idx, j_idx = torch.meshgrid(
        time_indices, item_indices, store_indices, indexing="ij"
    )
    # type: torch.Tensor / shape: [H*I*S, 3] / meaning: (t_local, item, store)
    forecast_indices = torch.stack(
        [t_idx.flatten(), i_idx.flatten(), j_idx.flatten()], dim=1
    )

    # 3. Call the new 'forecast_T1' mode
    # type: torch.Tensor / shape: [H*I*S] / meaning: flattened sales predictions
    preds_flat = model(
        mode="forecast_T1",
        indices_T1=forecast_indices,
        predicted_time_embeddings_block=femb,
    )
    # type: torch.Tensor / shape: [H, I, S] / meaning: reshaped sales predictions
    preds = preds_flat.view(steps, n_item, n_store)

    # 4. Get ground truth and compute loss
    # type: torch.Tensor / shape: [H, I, S] / meaning: ground truth sales
    gt = T1_train[(start + 1) : (start + steps + 1), :, :].to(device)

    # type: torch.Tensor / shape: scalar / meaning: MSE forecast loss
    return torch.mean((preds - gt) ** 2)


# ------------------------------------------------------------
# Sliding Window Forecast for Coupled Tensors (Test)
# ------------------------------------------------------------
def sliding_window_forecast(
    T1,
    T2,
    T3,
    initial_train_size,
    rank_cp,
    learning_rate,
    num_epochs,
    batch_size,
    device,
    static_features_T3,
    static_dim_T3,
    # [DEBUG REFACTOR] Accept pre-loaded model/hparams
    timesfm_model_main,
    timesfm_hparams_main,
    # --- NEW: Accept mlp_layers_t1 ---
    # (int) / meaning: The total number of Linear layers for MLP_T1.
    total_mlp_layers_t1=3,
    # --- End New ---
    forecast_horizon=28,
    lag_p=10,
    train_weights=True,
    w_recon_fixed=1.0,
    w1=0.33,  # <--- ADD THIS
    w2=0.33,  # <--- ADD THIS
    dropout_prob=0.5,  # <--- NEW PARAMETER HERE
):
    # type: int / meaning: total time steps in dataset
    time_total = T1.shape[0]
    # type: int / meaning: first time index of test set
    test_start = initial_train_size
    # type: int / meaning: end time index of test set
    test_end = time_total
    # type: torch.Tensor / shape: [T_test, I, S] / meaning: full test ground truth
    gt_full = T1[test_start:test_end, :, :]
    # type: int / meaning: num test steps, items, stores
    num_test, n_item, n_store = gt_full.shape
    # type: torch.Tensor / shape: [T_test, I, S] / meaning: accumulator for sum of predictions
    sum_preds = torch.zeros((num_test, n_item, n_store), device=device)
    # type: torch.Tensor / shape: [T_test, I, S] / meaning: accumulator for count of predictions
    count_preds = torch.zeros((num_test, n_item, n_store), device=device)
    # type: list[float] / meaning: list of w_recon values from each fold
    weights_recon_list = []
    # type: list[float] / meaning: list of w_forecast values from each fold
    weights_forecast_list = []
    # NEW: track reconstruction weights (T1/T2/T3) and last-seen values
    # type: list[float] / meaning: list of w_T1 values
    reconstruction_weight_T1_list = []  # type: list[float]
    # type: list[float] / meaning: list of w_T2 values
    reconstruction_weight_T2_list = []  # type: list[float]
    # type: list[float] / meaning: list of w_T3 values
    reconstruction_weight_T3_list = []  # type: list[float]
    # type: float / meaning: final w_T1 value
    last_reconstruction_weight_T1: float = float("nan")
    # type: float / meaning: final w_T2 value
    last_reconstruction_weight_T2: float = float("nan")
    # type: float / meaning: final w_T3 value
    last_reconstruction_weight_T3: float = float("nan")
    # type: float / meaning: final w_recon value
    last_weight_recon_overall: float = float("nan")
    # type: float / meaning: final w_forecast value
    last_weight_forecast_overall: float = float("nan")

    # type: int / meaning: start time index for the current test block
    for t in tqdm(
        range(test_start, test_end, forecast_horizon), desc="Sliding Window Forecast"
    ):
        # type: int / meaning: length of this test block (H)
        block_length = min(forecast_horizon, test_end - t)
        # type: int / meaning: end time index for this test block
        fold_end = t + block_length

        # --- Create training data for this fold ---
        # type: torch.Tensor / shape: [t, I, S] / meaning: T1 train data
        T1_train_full = T1[:t, :, :]
        # type: torch.Tensor / shape: [t, I, S] / meaning: T2 train data
        T2_train_full = T2[:t, :, :]
        # type: torch.Tensor / shape: [t, F] / meaning: T3 train data
        T3_train_full = T3[:t, :]

        # type: DataLoader / meaning: dataloader for T1
        train_loader_T1 = DataLoader(
            Tensor3DDataset(T1_train_full),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=16,
        )
        # type: DataLoader / meaning: dataloader for T2
        train_loader_T2 = DataLoader(
            Tensor3DDataset(T2_train_full),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=16,
        )
        # type: DataLoader / meaning: dataloader for T3
        train_loader_T3 = DataLoader(
            Tensor2DDataset(T3_train_full),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=16,
        )

        # type: tuple / meaning: shape of T1 data [t, I, S]
        shape_t1_fold = T1_train_full.shape
        # type: tuple / meaning: shape of T3 data [t, F]
        shape_t3_fold = T3_train_full.shape

        model = CoupledCPDecomposition(
            shape_t1=shape_t1_fold,
            shape_t3=shape_t3_fold,
            rank_cp=rank_cp,
            train_time=t,  # Pass current train length 't'
            # [DEBUG REFACTOR] Pass the pre-loaded model/hparams
            timesfm_model_instance=timesfm_model_main,
            timesfm_hparams_instance=timesfm_hparams_main,
            mlp_hidden_T1=1024,
            mlp_hidden_T2=512,
            mlp_hidden_T3=256,
            # --- NEW: Pass total_mlp_layers_t1 to the model ---
            # Pass the variable from the function argument into the class constructor.
            total_mlp_layers_t1=total_mlp_layers_t1,
            # --- End New ---
            lambda_param=1.0,
            w1=w1,
            w2=w2,
            device=device,
            static_features_T3=static_features_T3,
            static_dim_T3=static_dim_T3,
            dropout_prob=dropout_prob,
            train_weights=train_weights,
            w_recon_fixed=w_recon_fixed,
        )

        # type: torch.optim.Adam / meaning: Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # type: torch.amp.GradScaler / meaning: gradient scaler for mixed precision
        scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

        # type: float / meaning: best validation loss so far
        best_val_loss = float("inf")
        # type: dict / meaning: state dictionary of the best model
        best_model_state = None
        timer = EpochTimer()
        # type: int / meaning: epoch number
        for epoch in tqdm(range(num_epochs), desc=f"Block t={t} Epochs", leave=False):
            timer.start_epoch(epoch)
            # timing (one quick sample)
            # type: tuple / meaning: sample batch for timing
            sample_batch_T1 = next(iter(train_loader_T1))
            sample_batch_T2 = next(iter(train_loader_T2))
            sample_batch_T3 = next(iter(train_loader_T3))
            # type: tuple(float) / meaning: (recon_time, forecast_time)
            cp_time, timesfm_time = measure_component_times(
                model,
                sample_batch_T1,
                sample_batch_T2,
                sample_batch_T3,
                forecast_steps=block_length,
            )
            timer.checkpoint("component_timing")
            wandb.log(
                {
                    "cp_decomposition_time_epoch": cp_time,
                    "timesfm_prediction_time_epoch": timesfm_time,
                    "epoch": epoch,
                }
            )

            # === training over reconstruction batches ===
            model.train()
            # type: float / meaning: accumulator for epoch reconstruction loss
            epoch_recon = 0.0
            # itertools.cycle: common, creates an infinite iterator
            # type: iterator / meaning: infinite iterator over T2 dataloader
            iter_T2 = cycle(train_loader_T2)
            # type: iterator / meaning: infinite iterator over T3 dataloader
            iter_T3 = cycle(train_loader_T3)

            # type: tuple / meaning: (indices, values) for T1
            for batch_T1 in train_loader_T1:
                # type: tuple / meaning: (indices, values) for T2
                batch_T2 = next(iter_T2)
                # type: tuple / meaning: (indices, values) for T3
                batch_T3 = next(iter_T3)

                # --- Step 1: Reconstruction Loss ---
                optimizer.zero_grad(set_to_none=True)
                # torch.amp.autocast: common, enables automatic mixed precision
                # nullcontext: common, a no-op context manager
                # type: torch.amp.autocast or nullcontext / meaning: enables mixed precision
                ctx = (
                    torch.amp.autocast(device_type="cuda")
                    if model.device.type == "cuda"
                    else nullcontext()
                )
                with ctx:
                    # ** This is "Mode 1" **
                    # type: tuple(torch.Tensor) / meaning: (scalar loss, scalar penalty)
                    loss_recon, ortho = model(
                        mode="reconstruction_loss",  # Use the new mode
                        batch_T1=batch_T1,
                        batch_T2=batch_T2,
                        batch_T3=batch_T3,
                    )
                    # type: tuple(torch.Tensor) / meaning: (w_recon, w_forecast)
                    w_recon, w_forecast = model.get_effective_weights()
                    # type: torch.Tensor / shape: scalar / meaning: total loss for this step
                    total = w_recon * loss_recon + ortho

                # type: None / meaning: scales loss and computes gradients
                scaler.scale(total).backward()
                # torch.nn.utils.clip_grad_norm_: common, clips gradient norm
                # type: None / meaning: clips gradient norm to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                # type: None / meaning: updates optimizer weights
                scaler.step(optimizer)
                # type: None / meaning: updates gradient scaler
                scaler.update()

                epoch_recon += loss_recon.item()
            epoch_recon /= len(train_loader_T1)
            # ============================================================
            # [BLOCK 3 FIX] Checkpoint after reconstruction
            # ============================================================
            timer.checkpoint("reconstruction_training")
            # ============================================================
            # === [NEW] Gradient Check Setup ===
            # We will check if the forecast loss step updates the time embeddings.
            # We do this only on the first epoch of the first window (t == test_start)
            # to avoid spamming the log.

            # type: float / meaning: Store the sum of the first time embedding *before* the forecast step.
            sum_embedding_before_forecast_step = 0.0

            # type: bool / meaning: Flag to activate the check
            run_gradient_check_this_epoch: bool = (t == test_start) and (epoch <= 1)

            if run_gradient_check_this_epoch:
                with torch.no_grad():
                    # (torch.Tensor) scalar, sum of the first time vector's weights
                    sum_embedding_before_forecast_step = float(
                        model.emb_time.weight[0].sum().item()
                    )
                print(f"\n--- [GRADIENT CHECK (Epoch 0, Fold t={t})] ---")
                print(
                    f"  emb_time[0] sum BEFORE forecast step: {sum_embedding_before_forecast_step:.6f}"
                )
            # === [END NEW] ===
            # ============================================================
            # [BLOCK 3 FIX] Checkpoint after gradient check
            # ============================================================
            timer.checkpoint("gradient_check")
            # ============================================================
            # === Step 2: Per-epoch forecast consistency step (TimesFM) ===
            optimizer.zero_grad(set_to_none=True)
            with (
                torch.amp.autocast(device_type="cuda")
                if model.device.type == "cuda"
                else nullcontext()
            ):
                # This function now correctly calls the on-graph methods
                # ("Mode 2" is called inside here)
                # type: torch.Tensor / shape: scalar / meaning: MSE loss on a future window
                loss_forecast = compute_forecast_loss(
                    model, T1_train_full, device, window=min(28, block_length)
                )
                # type: tuple(torch.Tensor) / meaning: (w_recon, w_forecast)
                w_recon, w_forecast = model.get_effective_weights()
                # type: torch.Tensor / shape: scalar / meaning: total loss for this step
                total_forecast = w_forecast * loss_forecast

            # This backpropagation flows through TimesFM to model.emb_time
            # type: None / meaning: scales loss and computes gradients
            scaler.scale(total_forecast).backward()
            # type: None / meaning: clips gradient norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            # type: None / meaning: updates optimizer weights (emb_time is updated here)
            scaler.step(optimizer)
            # type: None / meaning: updates gradient scaler
            scaler.update()
            # ============================================================
            # [BLOCK 3 FIX] Checkpoint after forecast step
            # ============================================================
            timer.checkpoint("forecast_step")
            # ============================================================
            # === [NEW] Gradient Check Result ===
            if run_gradient_check_this_epoch:
                with torch.no_grad():
                    # (torch.Tensor) scalar, sum of the *same* vector *after* the update
                    sum_embedding_after_forecast_step = float(
                        model.emb_time.weight[0].sum().item()
                    )
                print(
                    f"  emb_time[0] sum AFTER forecast step:  {sum_embedding_after_forecast_step:.6f}"
                )
                if (
                    sum_embedding_before_forecast_step
                    != sum_embedding_after_forecast_step
                ):
                    print("  ✅ SUCCESS: emb_time was updated by the forecast loss.")
                else:
                    print(
                        "  ❌ FAILURE: emb_time was NOT updated (this is an error if w_forecast > 0)."
                    )
                print("--- [END GRADIENT CHECK] ---\n")
            # === [END NEW] ===

            # validation with sliding window
            # type: float / meaning: validation loss for this epoch
            val_loss = sliding_window_validation(
                model,
                T1_train_full,
                lag_p,
                forecast_horizon=forecast_horizon,
                device=device,
            )
            # ============================================================
            # [BLOCK 3 FIX] Checkpoint after validation
            # ============================================================
            timer.checkpoint("validation")
            # ============================================================
            wandb.log(
                {
                    "epoch_train_recon": epoch_recon,
                    "epoch_forecast_loss": loss_forecast.item(),
                    "epoch_val_loss": val_loss,
                    "block_train_end": t - 1,
                    "epoch": epoch,
                    "w_recon": float(w_recon.detach().cpu()),
                    "w_forecast": float(w_forecast.detach().cpu()),
                }
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
            # ============================================================
            # [BLOCK 3 FIX] End epoch timing
            # ============================================================
            timer.end_epoch()
            # ============================================================
        # [BLOCK 5 FIX] Cleanup GPU Memory After All Epochs
        # ============================================================
        if torch.cuda.is_available():
            # type: None / meaning: forces GPU to release unused memory
            torch.cuda.empty_cache()

        # import gc: (library) Python's garbage collector
        import gc

        # gc.collect(): (function) forces Python to free unused memory
        gc.collect()

        # Log memory usage
        if torch.cuda.is_available():
            # type: float / meaning: GB of GPU memory currently allocated
            gpu_mem_gb: float = torch.cuda.memory_allocated() / 1e9
            # type: float / meaning: GB of GPU memory reserved by PyTorch
            gpu_reserved_gb: float = torch.cuda.memory_reserved() / 1e9
            print(f"[MEMORY] GPU allocated: {gpu_mem_gb:.2f} GB")
            print(f"[MEMORY] GPU reserved:  {gpu_reserved_gb:.2f} GB")
        # ============================================================

        if best_model_state is None:
            best_model_state = model.state_dict()
        model.load_state_dict(best_model_state)

        # === simpler, TimesFM-focused checkpoint/file names ===
        # type: str / meaning: file path for checkpoint
        run_identifier = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            "Long-term_Forecasting/trials",
            f"{run_identifier}_timesfm_coupled_cv_checkpoint_RankCP_{rank_cp}_lr_{learning_rate}_epochs_{num_epochs}_bs_{batch_size}_lag_{lag_p}_block_{t}.pt",
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(best_model_state, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        model.eval()
        with torch.no_grad():
            # --- This logic was moved from the deleted 'forecast_time_embeddings' method ---
            # We use the full training history (length 't') for the final test forecast
            # type: int / meaning: length of history (L) to use as context
            hist_T: int = int(t)  # 't' is the train_end index for this fold
            hist_T = max(1, min(hist_T, model.emb_time.num_embeddings))  # Clamp
            # type: torch.Tensor / shape: [L] / meaning: indices [0, 1, ..., L-1]
            history_indices: torch.Tensor = torch.arange(
                hist_T,
                device=device,
                dtype=torch.long,
            )
            # --- End moved logic ---

            # 1. Get future embeddings from TimesFM (NO_GRAD context)
            # type: torch.Tensor / shape: [H, R] / meaning: TimesFM-predicted embeddings
            forecasted_embeddings = model.get_future_time_embeddings_on_graph(
                history_time_indices=history_indices,  # [L]
                forecast_length=block_length,  # H
            )

            # 2. Build index grid
            # type: int / meaning: number of items
            n_item = T1.shape[1]
            # type: int / meaning: number of stores
            n_store = T1.shape[2]
            # type: torch.Tensor / shape: [H] / meaning: local time [0, ..., H-1]
            time_indices = torch.arange(block_length, device=device, dtype=torch.long)
            # type: torch.Tensor / shape: [I] / meaning: item indices [0, ..., I-1]
            item_indices = torch.arange(n_item, device=device, dtype=torch.long)
            # type: torch.Tensor / shape: [S] / meaning: store indices [0, ..., S-1]
            store_indices = torch.arange(n_store, device=device, dtype=torch.long)

            # type: tuple(torch.Tensor) / meaning: 3 tensors, each shape [H, I, S]
            t_idx, i_idx, j_idx = torch.meshgrid(
                time_indices, item_indices, store_indices, indexing="ij"
            )
            # type: torch.Tensor / shape: [H*I*S, 3] / meaning: (t_local, item, store)
            forecast_indices = torch.stack(
                [t_idx.flatten(), i_idx.flatten(), j_idx.flatten()], dim=1
            )

            # 3. Call the new 'forecast_T1' mode
            # type: torch.Tensor / shape: [H*I*S] / meaning: flattened sales predictions
            preds_flat = model(
                mode="forecast_T1",
                indices_T1=forecast_indices,
                predicted_time_embeddings_block=forecasted_embeddings,
            )
            # type: torch.Tensor / shape: [H, I, S] / meaning: reshaped sales predictions
            preds = preds_flat.view(block_length, n_item, n_store)

        # type: int / meaning: loop counter for test block steps
        for i in range(block_length):
            # type: int / meaning: global test time index
            global_idx = t - test_start + i
            sum_preds[global_idx] += preds[i]
            count_preds[global_idx] += 1

        # type: tuple(torch.Tensor) / meaning: (w_recon, w_forecast)
        w_recon, w_forecast = model.get_effective_weights()
        weights_recon_list.append(float(w_recon.detach().cpu()))
        weights_forecast_list.append(float(w_forecast.detach().cpu()))

        # NEW: also snapshot the three reconstruction weights (T1/T2/T3)
        with torch.no_grad():
            # type: torch.Tensor / shape: [3] / meaning: [w_T1, w_T2, w_T3]
            reconstruction_weights_vector = (
                model.get_reconstruction_weights().detach().cpu()
            )
        reconstruction_weight_T1_list.append(float(reconstruction_weights_vector[0]))
        reconstruction_weight_T2_list.append(float(reconstruction_weights_vector[1]))
        reconstruction_weight_T3_list.append(float(reconstruction_weights_vector[2]))
        print(
            f"reconstruction weights: T1={reconstruction_weights_vector[0]:.4f}, T2={reconstruction_weights_vector[1]:.4f}, T3={reconstruction_weights_vector[2]:.4f}"
        )

        # NEW: keep "final" (last-block) values
        last_reconstruction_weight_T1 = float(reconstruction_weights_vector[0])
        last_reconstruction_weight_T2 = float(reconstruction_weights_vector[1])
        last_reconstruction_weight_T3 = float(reconstruction_weights_vector[2])
        last_weight_recon_overall = float(w_recon.detach().cpu())
        last_weight_forecast_overall = float(w_forecast.detach().cpu())

        wandb.log(
            {
                "block_train_end": t - 1,
                "forecast_horizon": block_length,
                "val_loss": best_val_loss,
                "w_recon": float(w_recon.detach().cpu()),
                "w_forecast": float(w_forecast.detach().cpu()),
                "w1_T1": float(reconstruction_weights_vector[0]),
                "w2_T2": float(reconstruction_weights_vector[1]),
                "w3_T3": float(reconstruction_weights_vector[2]),
                # "block_train_end": t - 1, # Duplicate key removed
            }
        )

    # --- End of sliding window loop ---

    # Build full indices from gt_full shape for WRMSSE
    # type: int / meaning: num test steps, items, stores
    num_test, n_item, n_store = gt_full.shape
    # type: torch.Tensor / shape: [T_test*I*S, 3] / meaning: (t_global, item, store)
    full_indices = torch.stack(
        [
            torch.arange(num_test, device=gt_full.device).repeat_interleave(
                n_item * n_store
            ),
            torch.arange(n_item, device=gt_full.device).repeat(num_test * n_store),
            torch.arange(n_store, device=gt_full.device).repeat(
                num_test * n_item
            ),  # Bugfix: was n_item
        ],
        dim=1,
    )
    # type: torch.Tensor / shape: [T_test*I*S, 3] / meaning: alias for full_indices
    indices = full_indices
    # type: torch.Tensor / shape: [T_test, I, S] / meaning: averaged predictions
    avg_preds = sum_preds / count_preds.clamp(min=1)

    # --- Compute final metrics ---
    # type: torch.Tensor / shape: scalar / meaning: overall MSE
    overall_mse = nn.MSELoss()(avg_preds, gt_full)
    # type: torch.Tensor / shape: scalar / meaning: overall RMSE
    overall_rmse = rmse_torch(gt_full, avg_preds)
    # type: torch.Tensor / shape: scalar / meaning: overall R2
    overall_r2 = r2_torch(gt_full, avg_preds)
    # type: torch.Tensor / shape: scalar / meaning: overall Kelly R2
    overall_r2k = r2_kelly_torch(gt_full, avg_preds)
    # type: torch.Tensor / shape: scalar / meaning: overall WRMSSE
    overall_wrmsse = wrmsse_torch(
        indices, gt_full.flatten(), avg_preds.flatten(), T1[:initial_train_size]
    )
    # type: float / meaning: average w_recon across folds
    avg_weight_recon = (
        np.mean(weights_recon_list) if len(weights_recon_list) else float("nan")
    )
    # type: float / meaning: average w_forecast across folds
    avg_weight_forecast = (
        np.mean(weights_forecast_list) if len(weights_forecast_list) else float("nan")
    )
    # Compute averages across all blocks (now we include the last block too)
    # type: float / meaning: average w_T1
    avg_reconstruction_weight_T1: float = (
        np.mean(reconstruction_weight_T1_list)
        if reconstruction_weight_T1_list
        else float("nan")
    )
    # type: float / meaning: average w_T2
    avg_reconstruction_weight_T2: float = (
        np.mean(reconstruction_weight_T2_list)
        if reconstruction_weight_T2_list
        else float("nan")
    )
    # type: float / meaning: average w_T3
    avg_reconstruction_weight_T3: float = (
        np.mean(reconstruction_weight_T3_list)
        if reconstruction_weight_T3_list
        else float("nan")
    )

    return {
        "overall_mse": overall_mse.item(),
        "overall_rmse": overall_rmse.item(),
        "overall_r2": overall_r2.item(),
        "overall_r2k": overall_r2k.item(),
        "overall_wrmsse": overall_wrmsse.item(),
        "avg_weight_recon": avg_weight_recon,
        "avg_weight_forecast": avg_weight_forecast,
        "avg_w1_T1": avg_reconstruction_weight_T1,  # NEW: average w1 (T1)
        "avg_w2_T2": avg_reconstruction_weight_T2,  # NEW: average w2 (T2)
        "avg_w3_T3": avg_reconstruction_weight_T3,  # NEW: average w3 (T3)
        "last_w1_T1": last_reconstruction_weight_T1,  # NEW: final w1 (T1)
        "last_w2_T2": last_reconstruction_weight_T2,  # NEW: final w2 (T2)
        "last_w3_T3": last_reconstruction_weight_T3,  # NEW: final w3 (T3)
        "last_w_recon": last_weight_recon_overall,  # NEW: final overall w_recon
        "last_w_forecast": last_weight_forecast_overall,  # NEW: final overall w_forecast (== w_predict)
        "all_preds": avg_preds.cpu(),
        "all_gt": gt_full.cpu(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs per forecast block",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8192 * 2, help="Mini-batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--Rank_CP", type=int, default=100, help="CP rank (embedding dimension)"
    )
    # --- NEW: Argument for MLP_T1 total layers ---
    # (int) / meaning: The total number of Linear layers for the T1 MLP.
    # For example, 3 means 1 input layer, 1 hidden layer, and 1 output layer.
    parser.add_argument(
        "--mlp_layers_t1",
        type=int,
        default=3,
        help="Total number of Linear layers for MLP_T1 (e.g., 3 means 2 hidden layers)",
    )
    # --- End New ---
    parser.add_argument(
        "--lambda_param",
        type=float,
        default=1.0,
        help="Weight multiplier for overall loss",
    )
    parser.add_argument(
        "--w1", type=float, default=0.33, help="Weight for T1 loss in reconstruction"
    )
    parser.add_argument(
        "--w2", type=float, default=0.33, help="Weight for T2 loss in reconstruction"
    )
    parser.add_argument(
        "--train_weights",
        type=int,
        default=1,
        help="1=trainable (softmax) weights, 0=fixed",
    )
    parser.add_argument(
        "--w_recon_fixed",
        type=float,
        default=1.0,
        help="If train_weights=0, w_forecast=1-w_recon_fixed",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=1,
        help="Use CUDA if available (1 for yes, 0 for no)",
    )
    parser.add_argument(
        "--forecast_horizon", type=int, default=28, help="Forecast horizon (steps)"
    )
    parser.add_argument(
        "--lag_p",
        type=int,
        default=730,
        help="Lag p used for sliding window validation",
    )
    parser.add_argument(
        "--dropout_prob",
        type=float,
        default=0.15,
        help="Dropout probability for MLP layers (default: 0.5)",
    )
    args = parser.parse_args()

    device = (
        torch.device("cuda:0")
        if args.cuda and torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"\033[92mUsing device: {device}\033[0m")

    # === [DEBUG REFACTOR] Load TimesFM ONCE at the start ===
    print("\n[DEBUG] Loading TimesFM model... (This happens only once)")

    # Build hparams for the PyTorch TimesFM-2.0 500M checkpoint
    timesfm_hparams = timesfm.TimesFmHparams(
        backend=(
            "gpu" if device.type == "cuda" else "cpu"
        ),  # (str) run on GPU if avail
        per_core_batch_size=32,  # (int) internal mini-batch
        horizon_len=28,  # (int) model forecast length; we still pass horizon_len at decode-time too
        context_len=2048,  # (int) context window
        input_patch_len=32,  # (int) fixed for this model
        output_patch_len=128,  # (int) fixed for this model
        num_layers=50,  # (int) fixed
        model_dims=1280,  # (int) fixed
        use_positional_embedding=False,  # (bool) fixed
        num_heads=16,  # (int) fixed
    )

    # High-level wrapper (gives access to ._model)
    # This line will DOWNLOAD if cache is empty, or LOAD from cache if present.
    _timesfm_wrapper = timesfm.TimesFm(
        hparams=timesfm_hparams,
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
        ),
    )

    # Raw nn.Module we will call directly
    timesfm_model_main = _timesfm_wrapper._model.to(
        device
    ).eval()  # (nn.Module) on device, eval mode
    timesfm_hparams_main = (
        _timesfm_wrapper.hparams
    )  # (TimesFmHparams) keep for decode args

    # Freeze all TimesFM parameters (we do not train it)
    for param in timesfm_model_main.parameters():  # (nn.Parameter iterator)
        param.requires_grad = False  # (bool) freeze

    print("[DEBUG] ✅ TimesFM model loaded and frozen.\n")
    # === End of TimesFM loading block ===

    def safe_load_tensor(path, device):
        if not os.path.exists(path):
            print(f"\033[91mError: Missing {path}\033[0m")
            sys.exit(1)
        return torch.load(path, map_location=device).float()

    T1 = safe_load_tensor("./tensor_1_units.pt", device)
    T2 = safe_load_tensor("./tensor_2_price.pt", device)
    T3 = safe_load_tensor("./tensor_3_event.pt", device)

    print(f"Missing values in T1: {count_missing_values(T1)}")
    print(f"Missing values in T2: {count_missing_values(T2)}")
    print(f"Missing values in T3: {count_missing_values(T3)}")

    original_time = T1.shape[0]
    n_time = max(original_time, 150)  # quick run cap (bugfixed from max->min)
    T1 = T1[:n_time, :, :]
    T2 = T2[:n_time, :, :]
    T3 = T3[:n_time, :]

    train_time_initial = int(0.8 * n_time)
    test_time = n_time - train_time_initial
    print(f"Train time: {train_time_initial}, Test time: {test_time}")

    shape_t1 = T1[:train_time_initial, :, :].shape
    shape_t3 = T3[:train_time_initial, :].shape

    static_features_T3 = torch.eye(shape_t3[1], device=device)
    static_dim_T3 = shape_t3[1]

    print(
        "Using TimesFM zero-shot forecasting; only (recon, forecast) losses are used."
    )
    run_identifier = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # === cleaner filenames (no MR-Diff params) ===
    base_suffix = f"timesfm_coupled_cv_RankCP_{args.Rank_CP}_lr_{args.learning_rate}_epochs_{args.num_epochs}_bs_{args.batch_size}_lag_{args.lag_p}"
    file_suffix = f"{base_suffix}_{run_identifier}"
    results_csv = os.path.join(
        "Long-term_Forecasting/trials", f"{file_suffix}_results.csv"
    )
    loss_plot = os.path.join("Long-term_Forecasting/trials", f"{file_suffix}_rmse.png")
    os.makedirs("Long-term_Forecasting/trials", exist_ok=True)

    metrics = sliding_window_forecast(
        T1,
        T2,
        T3,
        train_time_initial,
        args.Rank_CP,
        args.learning_rate,
        args.num_epochs,
        args.batch_size,
        device,
        static_features_T3,
        static_dim_T3,
        # [DEBUG REFACTOR] Pass the pre-loaded model/hparams
        timesfm_model_main=timesfm_model_main,
        timesfm_hparams_main=timesfm_hparams_main,
        # --- NEW: Pass mlp_layers_t1 from args ---
        # This sends the command-line argument to the sliding window function.
        total_mlp_layers_t1=args.mlp_layers_t1,
        # --- End New ---
        forecast_horizon=args.forecast_horizon,
        lag_p=args.lag_p,
        train_weights=bool(args.train_weights),
        w_recon_fixed=args.w_recon_fixed,
        w1=args.w1,  # <--- ADD THIS
        w2=args.w2,  # <--- ADD THIS
        dropout_prob=args.dropout_prob,  # <--- ADD THIS
    )

    print("\nOverall Test Metrics (Coupled Tensors, T1 Forecast):")
    print(f"  MSE        : {metrics['overall_mse']:.4f}")
    print(f"  RMSE       : {metrics['overall_rmse']:.4f}")
    print(f"  R²         : {metrics['overall_r2']:.4f}")
    print(f"  Kelly R²   : {metrics['overall_r2k']:.4f}")
    print(f"  WRMSSE     : {metrics['overall_wrmsse']:.4f}")
    print(
        f"  Avg Effective Weights (Recon, Forecast): {metrics['avg_weight_recon']:.4f}, {metrics['avg_weight_forecast']:.4f}"
    )

    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "MSE",
                "RMSE",
                "R2",
                "Kelly_R2",
                "WRMSSE",
                "Avg_Weight_Recon",
                "Avg_Weight_Forecast",
                # NEW: reconstruction weights (averages)
                "Avg_w1_T1",
                "Avg_w2_T2",
                "Avg_w3_T3",
                # NEW: reconstruction weights (final/last block)
                "Final_w1_T1",
                "Final_w2_T2",
                "Final_w3_T3",
                # NEW: overall weights (final/last block)
                "Final_w_recon",
                "Final_w_forecast",
                "Final_w_predict",  # predict == forecast
            ]
        )
        writer.writerow(
            [
                metrics["overall_mse"],
                metrics["overall_rmse"],
                metrics["overall_r2"],
                metrics["overall_r2k"],
                metrics["overall_wrmsse"],
                metrics["avg_weight_recon"],
                metrics["avg_weight_forecast"],
                # NEW: averages
                metrics.get("avg_w1_T1", float("nan")),
                metrics.get("avg_w2_T2", float("nan")),
                metrics.get("avg_w3_T3", float("nan")),
                # NEW: finals (last block)
                metrics.get("last_w1_T1", float("nan")),
                metrics.get("last_w2_T2", float("nan")),
                metrics.get("last_w3_T3", float("nan")),
                metrics.get("last_w_recon", float("nan")),
                metrics.get("last_w_forecast", float("nan")),
                metrics.get(
                    "last_w_forecast", float("nan")
                ),  # Final_w_predict == Final_w_forecast
            ]
        )

    print(f"\033[92mOverall test metrics saved to {results_csv}\033[0m")

    all_preds = metrics["all_preds"]
    gt = metrics["all_gt"]
    # mask_sparse = gt == 0
    # mask_regular = gt != 0

    # if mask_sparse.sum() > 0:
    #     mse_sparse = nn.MSELoss()(all_preds[mask_sparse], gt[mask_sparse]).item()
    #     rmse_sparse = rmse_torch(gt[mask_sparse], all_preds[mask_sparse]).item()
    #     full_indices = torch.stack(
    #         [
    #             torch.arange(gt.shape[0], device=gt.device).repeat_interleave(
    #                 gt.shape[1] * gt.shape[2]
    #             ),
    #             torch.arange(gt.shape[1], device=gt.device).repeat(
    #                 gt.shape[0] * gt.shape[2]
    #             ),
    #             torch.arange(gt.shape[2], device=gt.device).repeat(
    #                 gt.shape[0] * gt.shape[1]
    #             ),
    #         ],
    #         dim=1,
    #     )
    #     indices_sparse = full_indices[mask_sparse.flatten()]
    #     wrmsse_sparse = wrmsse_torch(
    #         indices_sparse,
    #         gt[mask_sparse].flatten(),
    #         all_preds[mask_sparse].flatten(),
    #         T1[:train_time_initial],
    #     ).item()
    # else:
    #     mse_sparse = rmse_sparse = wrmsse_sparse = float("nan")

    # if mask_regular.sum() > 0:
    #     mse_regular = nn.MSELoss()(all_preds[mask_regular], gt[mask_regular]).item()
    #     rmse_regular = rmse_torch(gt[mask_regular], all_preds[mask_regular]).item()
    #     r2_regular = r2_torch(gt[mask_regular], all_preds[mask_regular]).item()
    #     r2k_regular = r2_kelly_torch(gt[mask_regular], all_preds[mask_regular]).item()
    #     indices_regular = torch.stack(
    #         [
    #             torch.arange(gt.shape[0], device=gt.device).repeat_interleave(
    #                 gt.shape[1] * gt.shape[2]
    #             ),
    #             torch.arange(gt.shape[1], device=gt.device).repeat(
    #                 gt.shape[0] * gt.shape[2]
    #             ),
    #             torch.arange(gt.shape[2], device=gt.device).repeat(
    #                 gt.shape[0] * gt.shape[1]
    #             ),
    #         ],
    #         dim=1,
    #     )
    #     indices_regular = indices_regular[mask_regular.flatten()]
    #     wrmsse_regular = wrmsse_torch(
    #         indices_regular,
    #         gt[mask_regular].flatten(),
    #         all_preds[mask_regular].flatten(),
    #         T1[:train_time_initial],
    #     ).item()
    # else:
    #     mse_regular = rmse_regular = r2_regular = r2k_regular = wrmsse_regular = float(
    #         "nan"
    #     )

    # sparse_csv = os.path.join(
    #     "Long-term_Forecasting/trials", f"{file_suffix}_sparse.csv"
    # )
    # regular_csv = os.path.join(
    #     "Long-term_Forecasting/trials", f"{file_suffix}_regular.csv"
    # )
    # with open(sparse_csv, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["MSE", "RMSE", "WRMSSE"])
    #     writer.writerow([mse_sparse, rmse_sparse, wrmsse_sparse])
    # with open(regular_csv, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["MSE", "RMSE", "R2", "Kelly_R2", "WRMSSE"])
    #     writer.writerow(
    #         [mse_regular, rmse_regular, r2_regular, r2k_regular, wrmsse_regular]
    #     )
    # print(f"Results for sparse group saved to {sparse_csv}")
    # print(f"Results for regular group saved to {regular_csv}")

    window_rmse = []
    window_idx = []
    for i in range(all_preds.shape[0]):
        rmse_val = rmse_torch(gt[i], all_preds[i])
        window_rmse.append(rmse_val.item())
        window_idx.append(train_time_initial + i)
    plt.figure(figsize=(10, 6))
    plt.plot(window_idx, window_rmse, marker="o", linestyle="-", label="RMSE")
    plt.xlabel("Forecast Time Index")
    plt.ylabel("RMSE")
    plt.title("Forecast RMSE per Test Time Step")
    plt.legend()
    plt.savefig(loss_plot, dpi=600)
    plt.close()
    print(f"\033[92mSliding window RMSE visualization saved as {loss_plot}\033[0m")
    wandb.finish()


if __name__ == "__main__":
    main()
