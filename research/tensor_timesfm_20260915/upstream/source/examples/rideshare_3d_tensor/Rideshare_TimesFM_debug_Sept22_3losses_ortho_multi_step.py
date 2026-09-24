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
- PCA + TimesFM

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
from tqdm import tqdm
from typing import Tuple
import wandb

import random

# from transformers import AutoModelForCausalLM  # (import) TimesFM HF model loader
import timesfm  # (library) Google TimesFM: pretrained time-series model (PyTorch backend)

# ADD ↓↓↓
import hashlib  # (std lib) string hashing for unique file names
import time  # (std lib) wall-clock time; used in run hash to avoid collision

# ADD ↑↑↑


torch.backends.cuda.matmul.allow_tf32 = (
    True  # prints: (no shape); meaning: allow fast TF32 matmuls
)
torch.set_float32_matmul_precision(
    "high"
)  # prints: (no shape); meaning: favor speed for FP32 matmuls


def resolve_window_stride(arg: str, C: int, H: int) -> int:
    s = str(arg).strip().upper()
    if s in ("AUTO", "C+H"):
        return C + H
    if s == "C":
        return C
    if s == "H":
        return H
    # allow plain positive integers
    try:
        k = int(s)
        if k >= 1:
            return k
    except ValueError:
        pass
    raise ValueError("Invalid --window_stride. Use an integer, 'C', 'H', or 'C+H'.")


# ADD ↓↓↓  (orthogonality / decorrelation helpers)
@torch.no_grad()
def append_test_rows_for_block(
    agg_list: list,  # list to append dict rows to
    *,
    model: nn.Module,
    full_norm_tensor: torch.Tensor,  # [T, N_trip, F] normalized
    test_start_index: int,
    test_end_index: int,
    target_feature_indices_list: list[int],
    context_len: int,
    device: torch.device,
    use_timesfm: bool,
    timesfm_repo_id: str,
    timesfm_freq_category: int,
    timesfm_context_cap: int,
    timesfm_per_core_batch_size: int,
    feat_means_dev: torch.Tensor,
    feat_stds_dev: torch.Tensor,
):
    """Append (t, trip, feature, GroundTruth, Forecasted) rows in RAW space."""
    for t in range(int(test_start_index), int(test_end_index) + 1):
        hist_start_local = max((t - 1) - int(context_len) + 1, 0)
        history_time_indices = torch.arange(hist_start_local, t, device=device)
        time_emb = get_future_time_embeddings_via_gru_or_timesfm(
            model=model,
            history_time_indices=history_time_indices,
            steps_ahead=1,
            use_timesfm=use_timesfm,
            timesfm_repo_id=timesfm_repo_id,
            timesfm_freq_category=int(timesfm_freq_category),
            timesfm_context_cap=int(timesfm_context_cap),
            timesfm_per_core_batch_size=int(timesfm_per_core_batch_size),
        )[
            0
        ]  # [R]
        gt_slice = full_norm_tensor[t]  # [N_trip, F]
        all_trip_indices = torch.arange(
            gt_slice.shape[0], device=device, dtype=torch.long
        )

        for fidx in target_feature_indices_list:
            gt_norm = gt_slice[:, fidx]  # [N_trip]
            valid_mask = ~torch.isnan(gt_norm)
            if not torch.any(valid_mask):
                continue
            pred_norm = model(
                mode="forecast",
                predicted_time_embedding=time_emb,
                trip_index=all_trip_indices,
                feature_index=torch.tensor(
                    [int(fidx)], device=device, dtype=torch.long
                ),
            ).squeeze(-1)
            gt_raw = gt_norm[valid_mask] * feat_stds_dev[fidx] + feat_means_dev[fidx]
            pred_raw = (
                pred_norm[valid_mask] * feat_stds_dev[fidx] + feat_means_dev[fidx]
            )
            for trip_i, y, yhat in zip(
                torch.where(valid_mask)[0].tolist(),
                gt_raw.detach().cpu().tolist(),
                pred_raw.detach().cpu().tolist(),
            ):
                agg_list.append(
                    {
                        "t": int(t),
                        "trip": int(trip_i),
                        "feature": int(fidx),
                        "GroundTruth": float(y),
                        "Forecasted": float(yhat),
                    }
                )


def forecast_mse_targets_over_block(
    *,
    model: nn.Module,
    full_norm_tensor: torch.Tensor,
    predicted_time_embeddings_block: torch.Tensor,  # [H_block, R]
    validation_start_index: int,
    validation_end_index: int,
    target_feature_indices_list: list[int],
    device: torch.device,
) -> torch.Tensor:
    block_len = int(validation_end_index - validation_start_index + 1)
    if block_len <= 0:
        # Return a zero *with* grad so backward() is a no-op but valid
        return torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)

    all_trip_indices = torch.arange(
        full_norm_tensor.shape[1], device=device, dtype=torch.long
    )

    # Tensor accumulators (stay on-graph for SSE; count can be tensor or python int)
    sse = torch.zeros((), device=device, dtype=torch.float32)
    count = 0

    for h in range(block_len):
        t_future = int(validation_start_index + h)
        time_emb_h = predicted_time_embeddings_block[h]  # [R]

        gt_slice = full_norm_tensor[t_future]  # [N_trip, F]
        for fidx in target_feature_indices_list:
            gt = gt_slice[:, fidx]  # [N_trip]
            valid = ~torch.isnan(gt)
            if not torch.any(valid):
                continue

            pred = model(
                mode="forecast",
                predicted_time_embedding=time_emb_h,
                trip_index=all_trip_indices,
                feature_index=torch.tensor(
                    [int(fidx)], device=device, dtype=torch.long
                ),
            ).squeeze(
                -1
            )  # [N_trip]

            diff = pred[valid] - gt[valid]  # [N_valid]
            sse = sse + torch.dot(diff, diff)
            count += int(diff.numel())

    if count == 0:
        # Same rationale: return a zero that won't break backward()
        return torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)

    # Mean squared error (still a leaf with grad_fn)
    return sse / float(count)


from contextlib import contextmanager


@contextmanager
def freeze_modules(*modules: nn.Module):
    prev = []
    for m in modules:
        ms = []
        for p in m.parameters():
            ms.append(p.requires_grad)
            p.requires_grad_(False)
        prev.append(ms)
    try:
        yield
    finally:
        for m, states in zip(modules, prev):
            for p, rg in zip(m.parameters(), states):
                p.requires_grad_(rg)


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

# ================= PCA + TimesFM helpers (shapes/types/meaning on each line) =================

import warnings

# ---- PCA whitener (fit on time-embedding history) ----


@torch.no_grad()
def fit_pca_whitener_from_time_history(
    time_embedding_history: torch.Tensor,  # [L, R] float32; meaning: time-table rows for times 0..T (or 0..T+1)
    pca_epsilon: float = 1e-6,  # (float) small diagonal stabilizer for eigenvalues
):
    """
    Fit PCA whitening on the history and return pieces needed to (un)whiten.
    Shapes:
      L: number of historical time steps
      R: time embedding size (Rank_CP)
    """
    # (1) center over time → zero-mean per dimension
    mean_vector: torch.Tensor = torch.mean(
        time_embedding_history, dim=0, keepdim=True
    )  # [1, R] float32
    centered_history: torch.Tensor = (
        time_embedding_history - mean_vector
    )  # [L, R] float32

    # (2) covariance over time
    L_eff = max(int(time_embedding_history.shape[0]) - 1, 1)  # (int) avoid div-by-zero
    covariance_matrix: torch.Tensor = (centered_history.T @ centered_history) / float(
        L_eff
    )  # [R, R] float32

    # (3) symmetric eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(
        covariance_matrix
    )  # eigenvalues: [R], eigenvectors: [R, R]

    # (4) whitening and unwhitening matrices
    inv_sqrt_eigs: torch.Tensor = torch.rsqrt(eigenvalues + pca_epsilon)  # [R] float32
    whitening_matrix: torch.Tensor = eigenvectors @ torch.diag(
        inv_sqrt_eigs
    )  # [R, R] float32

    sqrt_eigs: torch.Tensor = torch.sqrt(eigenvalues + pca_epsilon)  # [R] float32
    unwhitening_matrix: torch.Tensor = eigenvectors @ torch.diag(
        sqrt_eigs
    )  # [R, R] float32

    return {
        "mean_vector": mean_vector,  # [1, R] float32; add back after unwhitening
        "whitening_matrix": whitening_matrix,  # [R, R] float32; right-multiply centered row vectors
        "unwhitening_matrix": unwhitening_matrix,  # [R, R] float32; right-multiply whitened row vectors
    }


@torch.no_grad()
def apply_whitening(
    time_embedding_history: torch.Tensor,  # [L, R] float32
    whitener: dict,
):
    """
    Z = (X - mu) @ W  → whitened with ~identity covariance
    """
    centered = time_embedding_history - whitener["mean_vector"]  # [L, R] float32
    whitened = centered @ whitener["whitening_matrix"]  # [L, R] float32
    return whitened


@torch.no_grad()
def unwhiten_single(
    whitened_next: torch.Tensor,  # [R] float32
    whitener: dict,
):
    """
    r_next = z_next @ W_inv + mu
    """
    r_next = (whitened_next @ whitener["unwhitening_matrix"].T) + whitener[
        "mean_vector"
    ].squeeze(
        0
    )  # [R] float32
    return r_next


# ===================== TimesFM (simple, no PCA, no standardize) =====================

_TIMESFM_MODEL_SINGLETON = None  # (global) keep single TimesFM object

_TIMESFM_CFG = None  # (repo_id, horizon_len, per_core_batch_size, context_cap)


def _get_timesfm_model(
    prefer_device: torch.device,
    huggingface_repo_id: str,
    horizon_len: int,
    per_core_batch_size: int,
    context_cap: int,
):
    """
    Build or return a TimesFM singleton keyed by (repo_id, horizon_len, bsz, context_cap).
    If any of these differ from the cached instance, rebuild so the output length matches.
    """
    global _TIMESFM_MODEL_SINGLETON, _TIMESFM_CFG
    req_cfg = (
        str(huggingface_repo_id),
        int(horizon_len),
        int(per_core_batch_size),
        int(context_cap),
    )

    # Rebuild if we have no model yet or the requested horizon/config changed
    if (_TIMESFM_MODEL_SINGLETON is None) or (_TIMESFM_CFG != req_cfg):
        _TIMESFM_MODEL_SINGLETON = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=("gpu" if torch.cuda.is_available() else "cpu"),
                per_core_batch_size=int(per_core_batch_size),
                horizon_len=int(horizon_len),
                context_len=int(context_cap),
                # required fixed params for this checkpoint
                input_patch_len=32,
                output_patch_len=128,
                num_layers=50,
                model_dims=1280,
                use_positional_embedding=False,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=str(huggingface_repo_id)
            ),
        )
        _TIMESFM_CFG = req_cfg
    return _TIMESFM_MODEL_SINGLETON


@torch.no_grad()
def _timesfm_forecast_next_value_for_one_series_simple(
    series_1d: np.ndarray,  # shape [L], dtype float32/float64; 1-D history for ONE component
    freq_category: int,  # int in {0,1,2}; for EPS (quarterly) → 2
    horizon_len: int,  # int, we use 1
    context_cap: int,  # int, e.g. 2048; we will truncate history to last context_cap points
    prefer_device: torch.device,  # torch.device; not strictly used, kept for symmetry/logging
    huggingface_repo_id: str,  # str; HF model id
    per_core_batch_size: int,  # int; internal TimesFM micro-batch
) -> float:
    """
    SIMPLE path:
      - no PCA, no z-score
      - just truncate long history and call TimesFM
      - return ONE float (the next value)
    """
    # Ensure float32 numpy array for TimesFM.
    series_array_float32 = np.asarray(series_1d, dtype=np.float32)  # [L] float32

    # Truncate if longer than model context.
    if series_array_float32.shape[0] > context_cap:  # (bool) length check
        series_array_float32 = series_array_float32[
            -context_cap:
        ]  # keep last context_cap values

    # Get or create the singleton TimesFM model.
    timesfm_model_singleton = _get_timesfm_model(
        prefer_device=prefer_device,
        huggingface_repo_id=huggingface_repo_id,
        horizon_len=horizon_len,
        per_core_batch_size=per_core_batch_size,
        context_cap=context_cap,
    )  # (TimesFm object)

    # TimesFM array API takes a list of arrays and a list of freq categories
    point_forecast_array, _ = timesfm_model_singleton.forecast(
        inputs=[series_array_float32],  # List[np.ndarray], each [L_i] float32
        freq=[int(freq_category)],  # List[int], values in {0,1,2}
    )  # point_forecast_array: np.ndarray with shape [batch=1, horizon_len]

    # Take first (only) series, first (only) step
    next_value_float = float(point_forecast_array[0, 0])  # (float)
    return next_value_float


@torch.no_grad()
def _timesfm_forecast_values_for_one_series_multi_step(
    series_1d: np.ndarray,  # [L] float32/float64; meaning: history for ONE dimension
    freq_category: int,  # int in {0,1,2}; quarterly/yearly → 2
    horizon_len: int,  # int H; how many steps ahead we want
    context_cap: int,  # int; cap history length
    prefer_device: torch.device,  # torch.device; not used directly (TimesFM manages device)
    huggingface_repo_id: str,  # str; HF checkpoint id
    per_core_batch_size: int,  # int; internal micro-batch
) -> np.ndarray:
    """
    Return H-step point forecasts for one series.
    Output: np.ndarray shape [horizon_len], dtype float32; steps 1..H ahead.
    """
    series_array_float32 = np.asarray(series_1d, dtype=np.float32)  # [L] float32
    if series_array_float32.shape[0] > context_cap:
        series_array_float32 = series_array_float32[
            -context_cap:
        ]  # keep last context_cap

    # Build or reuse the singleton with horizon_len = H
    timesfm_model_singleton = _get_timesfm_model(
        prefer_device=prefer_device,
        huggingface_repo_id=huggingface_repo_id,
        horizon_len=horizon_len,  # IMPORTANT: H, not 1
        per_core_batch_size=per_core_batch_size,
        context_cap=context_cap,
    )
    point_forecast_array, _ = timesfm_model_singleton.forecast(
        inputs=[series_array_float32],  # List[np.ndarray], each [L]
        freq=[int(freq_category)],  # List[int]
    )  # shape: [batch=1, horizon_len]
    return np.asarray(point_forecast_array[0], dtype=np.float32)  # [H] float32


@torch.no_grad()
def timesfm_next_time_embedding_simple(
    time_embedding_history: torch.Tensor,  # shape [L, R] float32 on ANY device
    freq_category: int,  # int {0,1,2}
    horizon_len: int,  # int (we use 1)
    context_cap: int,  # int
    prefer_device: torch.device,  # torch.device
    huggingface_repo_id: str,  # str
    per_core_batch_size: int,  # int
) -> torch.Tensor:
    """
    Predict ONE-STEP-AHEAD time embedding vector:
      for cp_index = 0..R-1:
        - take column cp_index → history [L]
        - feed TimesFM → next scalar
      stack → [R] (same device as input)
    """
    device_of_input_tensor = time_embedding_history.device  # torch.device
    length_L, rank_R = time_embedding_history.shape  # ints

    next_values_list = []  # Python list[float], length rank_R

    for cp_index in range(rank_R):
        # Grab 1-D history for this CP dim on CPU as numpy (TimesFM uses numpy arrays)
        one_component_history_numpy = (
            time_embedding_history[:, cp_index]  # tensor shape [L], dtype float32
            .detach()  # cut autograd graph; no gradients
            .cpu()  # move storage to CPU RAM
            .numpy()  # convert to numpy array (shares memory when possible)
        )  # numpy array shape [L], dtype float32

        # Zero-shot 1-step forecast with TimesFM
        next_value_float = _timesfm_forecast_next_value_for_one_series_simple(
            series_1d=one_component_history_numpy,  # np.ndarray [L]
            freq_category=freq_category,  # int
            horizon_len=horizon_len,  # int
            context_cap=context_cap,  # int
            prefer_device=prefer_device,  # torch.device
            huggingface_repo_id=huggingface_repo_id,  # str
            per_core_batch_size=per_core_batch_size,  # int
        )  # returns float

        next_values_list.append(next_value_float)  # append Python float

    # Build a tensor on the SAME device and SAME dtype as input
    next_time_embedding_vector_tensor = torch.tensor(
        next_values_list,
        dtype=time_embedding_history.dtype,
        device=device_of_input_tensor,
    )  # tensor shape [R], dtype float32

    return next_time_embedding_vector_tensor  # [R]


@torch.no_grad()
def timesfm_time_embeddings_multi_step(
    time_embedding_history: torch.Tensor,  # [L, R] float32 on any device; history up to time T
    freq_category: int,  # int {0,1,2}
    horizon_len: int,  # int H; how many steps ahead
    context_cap: int,  # int; context cap
    prefer_device: torch.device,  # torch.device
    huggingface_repo_id: str,  # str; HF repo id
    per_core_batch_size: int,  # int
) -> torch.Tensor:
    """
    Predict H future time-embedding vectors at once.
    Output: [H, R] float32 on the SAME device as input.
      row h-1 corresponds to time T+h (1-indexed horizon).
    """
    device_out = time_embedding_history.device  # torch.device; keep device
    length_L, rank_R = (
        time_embedding_history.shape
    )  # ints: history length and embedding rank

    # We'll fill a Python list of [R] tensors, then stack → [H, R].
    horizon_vectors = []  # list[torch.Tensor [R] float32]

    # For each CP dimension r, call TimesFM to get [H] future values and then stack.
    # We'll build a matrix of shape [R, H] first, then transpose to [H, R].
    future_by_dim = []  # list[np.ndarray [H] float32], length = R
    for cp_index in range(rank_R):
        series_cpu_numpy = (
            time_embedding_history[:, cp_index].detach().cpu().numpy()
        )  # np.ndarray [L], float32
        # H-step forecast for this CP dimension
        future_values_h = _timesfm_forecast_values_for_one_series_multi_step(
            series_1d=series_cpu_numpy,
            freq_category=int(freq_category),
            horizon_len=int(horizon_len),
            context_cap=int(context_cap),
            prefer_device=prefer_device,
            huggingface_repo_id=huggingface_repo_id,
            per_core_batch_size=int(per_core_batch_size),
        )  # np.ndarray [H] float32
        future_values_h = future_values_h[: int(horizon_len)]
        future_by_dim.append(future_values_h)  # accumulate per-dim [H]

    # future_by_dim is length R, each element [H]; stack→ [R, H], then transpose→ [H, R]
    future_matrix_hr = torch.tensor(
        np.stack(future_by_dim, axis=0).T,  # transpose while converting: [H, R]
        dtype=time_embedding_history.dtype,
        device=device_out,
    )  # torch.Tensor [H, R] float32
    return future_matrix_hr


@torch.no_grad()
def get_future_time_embeddings_via_gru_or_timesfm(
    model: nn.Module,  # CP2DWithTimesFM
    history_time_indices: torch.Tensor,  # [L] long; times 0..T (inclusive or exclusive per caller)
    steps_ahead: int,  # int H; how many steps into the future we need
    use_timesfm: bool,  # bool
    timesfm_repo_id: str,
    timesfm_freq_category: int,
    timesfm_context_cap: int,
    timesfm_per_core_batch_size: int,
):
    """
    Return [steps_ahead, R] of future time embeddings.
      - TimesFM path: H-step block via timesfm_time_embeddings_multi_step.
      - GRU path: use model.forecast_future_time_embeddings (autoregressive).
    """
    device = next(model.parameters()).device
    if not use_timesfm:
        # Autoregressive GRU future → [H, R]
        return model.forecast_future_time_embeddings(
            history_time_indices=history_time_indices, steps_ahead=int(steps_ahead)
        )  # [H, R] float32

    # Build [L, R] history from the embedding table
    time_embedding_history = model.time_embeddings(
        history_time_indices
    )  # [L, R] float32
    # H-step block via TimesFM for each CP dimension → [H, R]
    return timesfm_time_embeddings_multi_step(
        time_embedding_history=time_embedding_history,  # [L, R]
        freq_category=int(timesfm_freq_category),
        horizon_len=int(steps_ahead),  # H
        context_cap=int(timesfm_context_cap),
        prefer_device=device,
        huggingface_repo_id=timesfm_repo_id,
        per_core_batch_size=int(timesfm_per_core_batch_size),
    )  # [H, R] float32


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
    # torch.utils.data.get_worker_info()
    g = torch.Generator()
    g.manual_seed(seed)
    return g  # return generator for use in DataLoader


g_dl = set_seed(42)


# ---------------- Metrics ----------------
def r2(gt: torch.Tensor, pred: torch.Tensor) -> float:
    ss_res = torch.sum((gt - pred) ** 2)
    ss_tot = torch.sum((gt - torch.mean(gt)) ** 2)
    return (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0


def r2_kelly(gt: torch.Tensor, pred: torch.Tensor) -> float:
    ss_res = torch.sum((gt - pred) ** 2)
    ss_tot = torch.sum(gt**2)
    return (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0


def masked_mse(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute mean squared error over non-NaN targets.
    predicted : [*, *] float; meaning: model predictions
    target    : [*, *] float; meaning: ground-truth (NaN allowed)
    returns   : scalar float tensor; meaning: MSE over observed entries
    """
    valid_mask = ~torch.isnan(
        target
    )  # prints: same shape; meaning: True where target observed
    if not torch.any(valid_mask):
        return torch.tensor(
            0.0, device=predicted.device
        )  # prints: scalar; meaning: no labels → zero loss
    diff = (
        predicted[valid_mask] - target[valid_mask]
    )  # prints: [n] float; meaning: residuals on observed entries
    return torch.mean(diff * diff)  # prints: scalar float; meaning: MSE over observed


# NEW: 3D dataset (time, trip, feature)
class FilteredTensor3DDataset(Dataset):
    """
    Sparse sampler over observed cells in a 3D tensor [T, N_trip, F].
    Stores only indices with non-NaN values for reconstruction training.
    """

    def __init__(self, tensor3d: torch.Tensor, t_offset: int = 0):
        # tensor3d: [T, N_trip, F] float32 with NaN allowed
        cpu = tensor3d.detach().cpu()  # [T, N_trip, F] on CPU
        mask = ~torch.isnan(cpu)  # [T, N_trip, F] bool
        self.idxs = mask.nonzero(as_tuple=False).long()  # [N, 3] (t, trip, feat)
        self.idxs[:, 0] += int(t_offset)  # make time absolute
        self.values = cpu[mask].float()  # [N] values at those cells

    def __len__(self) -> int:
        return self.values.shape[0]  # N observed entries

    def __getitem__(self, i: int):
        # returns: ([3] long), ([]) float
        return self.idxs[i], self.values[i]


"""
sparse tensor: give self.idxs
"""


# NEW: 3D model (time, trip, feature)
class CP3DWithTimesFM(nn.Module):
    """
    Canonical-Polyadic style factorization learned with a predictor head.
    Embeddings:
      - time_embeddings:   [T, R]
      - trip_embeddings:   [N_trip, R]
      - feature_embeddings:[F, R]
    Prediction head consumes [time_vec | trip_vec | feature_vec] → scalar.
    """

    def __init__(
        self,
        shape: Tuple[int, int, int],  # (time_count, trip_count, feature_count)
        time_embedding_size: int,  # R
        mlp_hidden: int = 1024,
        dropout_p: float = 0.1,
        device: str = "cpu",
        mlp_layers: int = 4,  # NEW: total Linear layers (>=2)
    ):
        super().__init__()
        time_count, trip_count, feature_count = shape
        self.device = device

        # (1) embeddings
        self.time_embeddings = nn.Embedding(time_count, time_embedding_size)  # [T, R]
        self.trip_embeddings = nn.Embedding(
            trip_count, time_embedding_size
        )  # [N_trip, R]
        self.feature_embeddings = nn.Embedding(
            feature_count, time_embedding_size
        )  # [F, R]

        # (2) optional GRU (kept only for parity; disabled when using TimesFM)
        self.sequence_model = nn.GRU(
            input_size=time_embedding_size,
            hidden_size=time_embedding_size,
            num_layers=2,
            batch_first=True,
        )

        # (3) predictor: [time_vec | trip_vec | feature_vec] -> scalar (normalized)
        input_feature_dim: int = 3 * time_embedding_size  # int; 3R
        total_linear_layers: int = max(2, int(mlp_layers))  # int; at least input+output
        module_layers: list[nn.Module] = []  # Python list of modules

        # 1) Input → first hidden
        module_layers.append(
            nn.Linear(input_feature_dim, mlp_hidden)
        )  # [B, 3R] → [B, H]; fully-connected map (Linear). :contentReference[oaicite:1]{index=1}
        module_layers.append(
            nn.ReLU()
        )  # [B, H] → [B, H]; element-wise max(0,x). :contentReference[oaicite:2]{index=2}
        module_layers.append(
            nn.Dropout(dropout_p)
        )  # [B, H] → [B, H]; zero activations with prob p during training. :contentReference[oaicite:3]{index=3}

        # 2) Middle hidden blocks (repeat)
        for _ in range(
            total_linear_layers - 2
        ):  # e.g., 4→2 repeats (H1,H2,H3 then output)
            module_layers.append(nn.Linear(mlp_hidden, mlp_hidden))  # [B, H] → [B, H]
            module_layers.append(nn.ReLU())  # [B, H]
            module_layers.append(nn.Dropout(dropout_p))  # [B, H]

        # 3) Final output layer
        module_layers.append(nn.Linear(mlp_hidden, 1))  # [B, H] → [B, 1]; scalar

        # Chain in this exact order at runtime
        self.predictor = nn.Sequential(
            *module_layers
        )  # runs layers in the order they were added. :contentReference[oaicite:4]{index=4}

        self.dropout_p = dropout_p

    def _predict_value_from_parts(
        self,
        time_vector: torch.Tensor,  # [B, R]
        trip_index: torch.Tensor,  # [B] long
        feature_index: torch.Tensor,  # [B] long
    ) -> torch.Tensor:  # returns [B]
        trip_vector = self.trip_embeddings(trip_index)  # [B, R]
        feature_vector = self.feature_embeddings(feature_index)  # [B, R]
        concat_3r = torch.cat(
            [time_vector, trip_vector, feature_vector], dim=1
        )  # [B, 3R]
        return self.predictor(concat_3r).squeeze(1)  # [B]

    def forward(self, mode: str, **inputs):
        if mode == "reconstruction":
            # inputs: time_index [B], trip_index [B], feature_index [B]
            time_index = inputs["time_index"]  # [B] long
            trip_index = inputs["trip_index"]  # [B] long
            feature_index = inputs["feature_index"]  # [B] long
            time_vector = self.time_embeddings(time_index)  # [B, R]
            return self._predict_value_from_parts(
                time_vector, trip_index, feature_index
            )

        elif mode == "prediction":  # GRU path (unused if TimesFM)
            history_time_indices = inputs["history_time_indices"]  # [L] or [B, L]
            if history_time_indices.dim() == 1:
                history_time_indices = history_time_indices.unsqueeze(0)  # [1, L]
            seq = self.time_embeddings(history_time_indices)  # [B, L, R]
            _, last_hidden = self.sequence_model(seq)  # [layers, B, R]
            next_time_embedding = last_hidden[-1]  # [B, R]
            return (
                next_time_embedding.squeeze(0)
                if next_time_embedding.size(0) == 1
                else next_time_embedding
            )

        elif (
            mode == "forecast"
        ):  # predict ONE feature for many trips at a given future time embedding
            predicted_time_embedding = inputs[
                "predicted_time_embedding"
            ]  # [R] or [1, R]
            trip_index_input = inputs["trip_index"]  # [B] long (one or many trips)
            feature_index_input = inputs["feature_index"]  # scalar or [B]
            if predicted_time_embedding.dim() == 1:
                predicted_time_embedding = predicted_time_embedding.unsqueeze(
                    0
                )  # [1, R]
            if not torch.is_tensor(trip_index_input):
                trip_index = torch.tensor(
                    [int(trip_index_input)], device=predicted_time_embedding.device
                )
            else:
                trip_index = trip_index_input
            if not torch.is_tensor(feature_index_input):
                feature_index = torch.tensor(
                    [int(feature_index_input)], device=predicted_time_embedding.device
                )
            else:
                feature_index = feature_index_input

            B = trip_index.size(0)
            if feature_index.dim() == 0:  # scalar -> [B]
                feature_index = feature_index.expand(B)
            elif feature_index.numel() == 1:  # [1] -> [B]
                feature_index = feature_index.expand(B)

            time_vec = predicted_time_embedding.expand(B, -1)  # [B, R]
            trip_vector = self.trip_embeddings(trip_index)  # [B, R]
            feature_vector = self.feature_embeddings(feature_index)  # [B, R]
            concat_3r = torch.cat([time_vec, trip_vector, feature_vector], dim=1)

            # return self._predict_value_from_parts(time_vec, trip_index, feature_index)  # [B]
            return self.predictor(concat_3r).squeeze(1)  # [B]

        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    @torch.no_grad()
    def forecast_future_time_embeddings(
        self, history_time_indices: torch.Tensor, steps_ahead: int  # [L] long  # int H
    ) -> torch.Tensor:
        """
        Autoregressive GRU forecast of H future time embeddings.
        Returns: torch.Tensor [H, R] on the same device as the module.
        """
        if history_time_indices.dim() != 1:
            history_time_indices = history_time_indices.reshape(-1)  # [L]

        device_out = next(self.parameters()).device
        R: int = self.time_embeddings.embedding_dim

        # 1) get history embeddings: [1, L, R]
        seq: torch.Tensor = self.time_embeddings(history_time_indices).unsqueeze(0)

        # 2) pass through GRU to get last hidden: [num_layers, 1, R] -> take last layer → [1, R]
        _, hidden = self.sequence_model(seq)  # standard torch.nn.GRU forward
        last: torch.Tensor = hidden[-1]  # [1, R]

        # 3) roll H steps: feed last each time (simple, vanilla)
        out_future = []
        cur = last  # [1, R]
        for _ in range(int(steps_ahead)):
            out_future.append(cur.squeeze(0))  # append [R]
            # for a vanilla approach, re-use 'cur' as if it were the last hidden
            # if you want teacher-forcing style, embed an index table instead.
            cur = cur  # no-op; keep it simple

        return torch.stack(out_future, dim=0).to(device_out)  # [H, R]

    # @torch.no_grad()
    # def forecast_features_vector_for_all_trips(
    #     self,
    #     predicted_time_embedding: torch.Tensor,  # [R] or [1, R]
    #     trip_indices: torch.Tensor,  # [B] long
    #     feature_indices: torch.Tensor,  # [K] long
    # ) -> torch.Tensor:
    #     """
    #     Vectorized forecast for a feature-set across many trips at one future time.
    #     Returns: [B, K]
    #     """
    #     if predicted_time_embedding.dim() == 1:
    #         predicted_time_embedding = predicted_time_embedding.unsqueeze(0)  # [1, R]
    #     B = trip_indices.size(0)
    #     K = feature_indices.size(0)
    #     time_vec = predicted_time_embedding.expand(B, -1)  # [B, R]
    #     feat_vec = (
    #         self.feature_embeddings(feature_indices).unsqueeze(0).expand(B, -1, -1)
    #     )  # [B, K, R]
    #     trip_vec = (
    #         self.trip_embeddings(trip_indices).unsqueeze(1).expand(-1, K, -1)
    #     )  # [B, K, R]
    #     concat_3r = torch.cat(
    #         [time_vec.unsqueeze(1).expand(-1, K, -1), trip_vec, feat_vec], dim=2
    #     )  # [B, K, 3R]
    #     preds = self.predictor(concat_3r.reshape(B * K, -1)).reshape(B, K)  # [B, K]
    #     return preds


@torch.no_grad()
def evaluate_test_block_3d_target_only(
    *,
    model: nn.Module,
    full_norm_tensor: torch.Tensor,  # [T, N_trip, F] normalized
    test_start_index: int,
    test_end_index: int,
    target_feature_indices_list: list[int],
    context_len: int,
    device: torch.device,
    use_timesfm: bool,
    timesfm_repo_id: str,
    timesfm_freq_category: int,
    timesfm_context_cap: int,
    timesfm_per_core_batch_size: int,
    feat_means_dev: (
        torch.Tensor | None
    ) = None,  # [F] raw means for THIS window (optional)
    feat_stds_dev: (
        torch.Tensor | None
    ) = None,  # [F] raw stds  for THIS window (optional)
) -> dict:
    """
    Computes RMSE/MAE/R² over TARGETS ONLY in **normalized space** by default.
    If feat_means_dev & feat_stds_dev are provided, converts both pred & gt to raw space first.
    """
    use_raw = (feat_means_dev is not None) and (feat_stds_dev is not None)

    sum_squared_error = 0.0
    sum_abs_error = 0.0
    sum_y = 0.0
    sum_y_squared = 0.0
    count = 0

    for t in range(int(test_start_index), int(test_end_index) + 1):
        train_end_index = t - 1
        hist_start_local = max(train_end_index - int(context_len) + 1, 0)
        history_time_indices = torch.arange(
            hist_start_local, train_end_index + 1, device=device
        )

        future_time_embeddings_h1 = get_future_time_embeddings_via_gru_or_timesfm(
            model=model,
            history_time_indices=history_time_indices,
            steps_ahead=1,
            use_timesfm=use_timesfm,
            timesfm_repo_id=timesfm_repo_id,
            timesfm_freq_category=int(timesfm_freq_category),
            timesfm_context_cap=int(timesfm_context_cap),
            timesfm_per_core_batch_size=int(timesfm_per_core_batch_size),
        )  # [1, R]
        predicted_time_embedding_t1 = future_time_embeddings_h1[0]  # [R]

        gt_slice_trips_by_feature = full_norm_tensor[t]  # [N_trip, F]
        all_trip_indices = torch.arange(
            gt_slice_trips_by_feature.shape[0], device=device, dtype=torch.long
        )

        for target_feature_index in target_feature_indices_list:
            gt_norm = gt_slice_trips_by_feature[:, target_feature_index]  # [N_trip]
            valid_mask = ~torch.isnan(gt_norm)
            if not torch.any(valid_mask):
                continue

            pred_norm = model(
                mode="forecast",
                predicted_time_embedding=predicted_time_embedding_t1,
                trip_index=all_trip_indices,
                feature_index=torch.tensor(
                    [int(target_feature_index)], device=device, dtype=torch.long
                ),
            ).squeeze(
                -1
            )  # [N_trip]

            gt_valid = gt_norm[valid_mask]
            pred_valid = pred_norm[valid_mask]

            if use_raw:
                # x_raw = x_norm * std + mean
                std_f = float(feat_stds_dev[target_feature_index].item())
                mean_f = float(feat_means_dev[target_feature_index].item())
                gt_valid = gt_valid * std_f + mean_f
                pred_valid = pred_valid * std_f + mean_f

            residuals = pred_valid - gt_valid
            sum_squared_error += float(torch.dot(residuals, residuals))
            sum_abs_error += float(torch.sum(torch.abs(residuals)))
            sum_y += float(torch.sum(gt_valid))
            sum_y_squared += float(torch.dot(gt_valid, gt_valid))
            count += int(gt_valid.numel())

    if count == 0:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "r2_kelly": float("nan"),
        }

    rmse = math.sqrt(sum_squared_error / float(count))
    mae = sum_abs_error / float(count)
    ss_tot = sum_y_squared - (sum_y * sum_y) / float(count)
    r2_val = (1.0 - (sum_squared_error / ss_tot)) if ss_tot > 0 else float("nan")
    r2_kelly_val = (
        (1.0 - (sum_squared_error / sum_y_squared))
        if sum_y_squared > 0
        else float("nan")
    )
    return {"rmse": rmse, "mae": mae, "r2": r2_val, "r2_kelly": r2_kelly_val}


def _safe_r2_from_sse(sse: float, sum_y: float, sum_y2: float, n: int):
    """Compute R² and Kelly’s R² from aggregates; return (r2, r2_kelly)."""
    if n <= 0:
        return float("nan"), float("nan")
    # SS_tot = sum((y - mean)^2) = sum_y2 - sum_y^2 / n
    ss_tot = sum_y2 - (sum_y * sum_y) / max(n, 1)
    r2 = 1.0 - (sse / ss_tot) if ss_tot > 0 else float("nan")
    r2_k = 1.0 - (sse / sum_y2) if sum_y2 > 0 else float("nan")
    return r2, r2_k


@torch.no_grad()
def compute_epoch_forecast_aggregates_multi_horizon_3d(
    model: nn.Module,  # PyTorch module doing forecast()
    full_norm_tensor: torch.Tensor,  # shape [T_total, N_trip, F_total], float32
    train_start_index: int,  # inclusive history start (int)
    train_end_index: int,  # inclusive history end (int) — last t in context
    context_len: int,  # history length C (int)
    target_feature_indices_list: list[
        int
    ],  # the ONLY features we score, e.g. [13,14,...]
    device: torch.device | None = None,  # CUDA or CPU
    use_timesfm: bool = True,  # whether to use TimesFM to push time embeddings
    timesfm_repo_id: str = "google/timesfm-2.0-500m-pytorch",
    timesfm_freq_category: int = 0,
    timesfm_context_cap: int = 2048,
    timesfm_per_core_batch_size: int = 32,
    forecast_horizon: int = 4,  # H future steps
):
    """
    Return sums to compute RMSE/R² over TARGETS ONLY across next H steps.
    We do NOT use covariate errors in the loss.
    """
    device = full_norm_tensor.device if device is None else device

    # Derive the history time index vector with length <= context_len
    C_local: int = int(context_len)  # scalar
    hist_start_local: int = max(
        int(train_end_index) - C_local + 1, int(train_start_index)
    )
    history_time_indices: torch.Tensor = torch.arange(  # shape [C_hist]
        hist_start_local, int(train_end_index) + 1, device=device
    )

    # Ask GRU/TimesFM for H future time embeddings; returns [H, R]
    future_time_embeddings: torch.Tensor = (
        get_future_time_embeddings_via_gru_or_timesfm(
            model=model,
            history_time_indices=history_time_indices,
            steps_ahead=int(forecast_horizon),
            use_timesfm=use_timesfm,
            timesfm_repo_id=timesfm_repo_id,
            timesfm_freq_category=timesfm_freq_category,
            timesfm_context_cap=timesfm_context_cap,
            timesfm_per_core_batch_size=timesfm_per_core_batch_size,
        )
    )  # [H, R]

    # Accumulators for SSE and sums to compute R² later (over all targets and trips)
    sum_squared_error: float = 0.0
    sum_y: float = 0.0
    sum_y_squared: float = 0.0
    count: int = 0
    all_trip_indices = torch.arange(
        full_norm_tensor.shape[1], device=device, dtype=torch.long
    )
    # Iterate each horizon step
    for h in range(int(forecast_horizon)):
        t_future: int = int(train_end_index) + 1 + h
        if t_future >= full_norm_tensor.size(0):  # guard end of series
            break

        predicted_time_embedding_h: torch.Tensor = future_time_embeddings[h]  # [R]

        # Loop over target feature ids; compute forecast and MSE only for these
        for target_feature_index in target_feature_indices_list:
            # Ground truth across all trips at this (t_future, feature)
            # Shape: [N_trip]
            ground_truth_values: torch.Tensor = full_norm_tensor[
                t_future, :, target_feature_index
            ]

            # Skip if all NaN
            valid_mask: torch.Tensor = ~torch.isnan(
                ground_truth_values
            )  # [N_trip], bool
            if not torch.any(valid_mask):
                continue

            # Model forecast for this (time_embed, feature); broadcast across trips internally
            # Your model's "forecast" mode should return a vector for this feature
            # Here we keep consistent with your API: feature_index expects a 1-dim tensor
            predicted_values: torch.Tensor = model(
                mode="forecast",
                predicted_time_embedding=predicted_time_embedding_h,  # [R]
                trip_index=all_trip_indices,  # FIX: required
                feature_index=torch.tensor(
                    [int(target_feature_index)], device=device, dtype=torch.long  # [1]
                ),
            ).squeeze(
                0
            )  # [N_trip] or [ ] -> ensure [N_trip]

            # Select only valid entries
            gt_valid: torch.Tensor = ground_truth_values[valid_mask]  # [N_valid]
            pred_valid: torch.Tensor = predicted_values[valid_mask]  # [N_valid]

            # Accumulate SSE and sums (float() to detach)
            diff: torch.Tensor = pred_valid - gt_valid  # [N_valid]
            sum_squared_error += float(torch.dot(diff, diff))  # scalar
            sum_y += float(torch.sum(gt_valid))
            sum_y_squared += float(torch.dot(gt_valid, gt_valid))
            count += int(gt_valid.numel())

    return sum_squared_error, sum_y, sum_y_squared, count


@torch.no_grad()
def compute_mase_denominator_raw_fibers(
    train_raw_tensor: torch.Tensor,  # [T_train, N_trip, F_total] float32, may contain NaN
    target_feature_indices_list: list[
        int
    ],  # list of feature indices to evaluate (targets only)
    seasonal_period: int = 24,  # m in MASE(m); 1 = lag-1 naive
) -> float:
    """
    Compute the MASE denominator over a 3D panel by treating each (trip, feature) 'fiber'
    as a separate time series and averaging absolute lag-m differences across ALL valid fibers.

    MASE(m) denominator = mean over training data of |y_t - y_{t-m}|.

    We:
      - loop fibers: for each trip_index in [0..N_trip-1] and each target feature_index
      - take 1-D series over time: y = train_raw_tensor[:, trip_index, feature_index]  # [T_train]
      - build a validity mask for pairs (y[t], y[t-m]) to avoid NaNs
      - accumulate sum(|y[t] - y[t-m]|) and the count of valid pairs
    Finally:
      denom = total_sum_abs_diff / total_count
    """
    # ---------- shapes ----------
    # T_train: int length of the training time slice
    # N_trip : int number of trips (aka 'entities')
    # F_total: int number of features
    # --------------------------------
    if not isinstance(train_raw_tensor, torch.Tensor):
        raise TypeError(
            "train_raw_tensor must be a torch.Tensor of shape [T_train, N_trip, F_total]."
        )

    if train_raw_tensor.dim() != 3:
        raise ValueError(
            f"Expected [T_train, N_trip, F_total], got {tuple(train_raw_tensor.shape)}."
        )

    T_train, N_trip, F_total = train_raw_tensor.shape
    m: int = int(seasonal_period)

    # Not enough history for lag-m pairs
    if T_train <= m:
        return float("nan")

    # Work on CPU for simpler indexing; dtype float32 is fine
    series_3d_cpu: torch.Tensor = train_raw_tensor.detach().cpu()

    total_abs_diff_sum: float = (
        0.0  # sum over all |y_t - y_{t-m}| (Python float accumulator)
    )
    total_valid_pairs: int = 0  # count of valid pairs overall (Python int)

    # -------- iterate fibers: (trip, feature) --------
    for trip_index in range(N_trip):
        # Optional micro-optimization: slice once per trip → [T_train, F_total]
        trip_matrix_tf: torch.Tensor = series_3d_cpu[
            :, trip_index, :
        ]  # [T_train, F_total]

        for feature_index in target_feature_indices_list:
            if feature_index < 0 or feature_index >= F_total:
                continue  # skip out-of-range safely

            # 1-D time series for this fiber
            y_t: torch.Tensor = trip_matrix_tf[:, feature_index]  # [T_train] float32

            # Build lagged pairs using slicing
            #   y_t[m:]     aligns with times m..T_train-1
            #   y_t[:-m]    aligns with times 0..T_train-m-1
            cur_vals: torch.Tensor = y_t[m:]  # [T_train - m]
            prev_vals: torch.Tensor = y_t[:-m]  # [T_train - m]

            # Valid where both are finite (not NaN)
            valid_mask: torch.Tensor = (~torch.isnan(cur_vals)) & (
                ~torch.isnan(prev_vals)
            )  # [T_train - m] bool

            # If nothing valid for this fiber, skip
            if not torch.any(valid_mask):
                continue

            # Absolute differences on valid pairs
            diffs: torch.Tensor = torch.abs(
                cur_vals[valid_mask] - prev_vals[valid_mask]
            )  # [n_valid_pairs] float32

            # Accumulate into Python scalars
            total_abs_diff_sum += float(diffs.sum().item())
            total_valid_pairs += int(diffs.numel())

    # Final mean; guard divide-by-zero
    if total_valid_pairs == 0:
        return float("nan")

    mase_denominator: float = total_abs_diff_sum / float(total_valid_pairs)
    return mase_denominator


@torch.no_grad()
def evaluate_test_block_blockforecast_raw(
    *,
    model: nn.Module,
    full_norm_tensor: torch.Tensor,  # [T, N_trip, F] normalized
    feat_means_dev: torch.Tensor,  # [F] raw means
    feat_stds_dev: torch.Tensor,  # [F] raw stds
    target_feature_indices_list: list[int],
    t_test: int,
    t_end: int,
    predicted_time_embeddings_block: torch.Tensor,  # [H, R]
    device: torch.device,
    mase_denominator: float | None = None,
) -> dict:
    """
    Aggregate RAW-space metrics over the test block [t_test..t_end] using the
    supplied block of future time-embeddings (length H=t_end - t_test + 1).
    """
    H = int(t_end - t_test + 1)
    assert predicted_time_embeddings_block.size(0) >= H

    N_trip = full_norm_tensor.shape[1]
    all_trip_indices = torch.arange(N_trip, device=device, dtype=torch.long)

    # Accumulators in RAW space
    sse = 0.0
    sae = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    n = 0

    for h in range(H):
        t = t_test + h
        time_emb = predicted_time_embeddings_block[h]  # [R]
        gt_slice_norm = full_norm_tensor[t]  # [N_trip, F]

        for fidx in target_feature_indices_list:
            gt_norm = gt_slice_norm[:, fidx]  # [N_trip]
            valid = ~torch.isnan(gt_norm)
            if not torch.any(valid):
                continue

            # Predict normalized
            pred_norm = model(
                mode="forecast",
                predicted_time_embedding=time_emb,
                trip_index=all_trip_indices,
                feature_index=torch.tensor(
                    [int(fidx)], device=device, dtype=torch.long
                ),
            ).squeeze(
                -1
            )  # [N_trip]

            # Convert to RAW
            std_f = float(feat_stds_dev[fidx].item())
            mean_f = float(feat_means_dev[fidx].item())
            gt_raw = gt_norm[valid] * std_f + mean_f
            pred_raw = pred_norm[valid] * std_f + mean_f

            diff = pred_raw - gt_raw
            sse += float(torch.dot(diff, diff).item())
            sae += float(torch.sum(torch.abs(diff)).item())
            sum_y += float(torch.sum(gt_raw).item())
            sum_y2 += float(torch.dot(gt_raw, gt_raw).item())
            n += int(gt_raw.numel())

    rmse = math.sqrt(sse / n) if n > 0 else float("nan")
    mae = (sae / n) if n > 0 else float("nan")
    r2 = (
        (1.0 - sse / (sum_y2 - (sum_y * sum_y) / n))
        if (n > 0 and (sum_y2 - (sum_y * sum_y) / n) > 0)
        else float("nan")
    )
    r2_k = (1.0 - sse / sum_y2) if (n > 0 and sum_y2 > 0) else float("nan")

    mase = float("nan")
    if (
        n > 0
        and mase_denominator is not None
        and np.isfinite(mase_denominator)
        and mase_denominator > 0
    ):
        mase = mae / mase_denominator

    return {"rmse": rmse, "mae": mae, "mase": mase, "r2": r2, "r2_kelly": r2_k}


@torch.no_grad()
def append_test_rows_from_block_embeddings_raw(
    agg_list: list,
    *,
    model: nn.Module,
    full_norm_tensor: torch.Tensor,  # [T, N_trip, F] normalized
    feat_means_dev: torch.Tensor,
    feat_stds_dev: torch.Tensor,
    target_feature_indices_list: list[int],
    t_test: int,
    t_end: int,
    predicted_time_embeddings_block: torch.Tensor,  # [H, R]
    device: torch.device,
):
    """
    Append per-cell RAW rows for the test block using a precomputed [H,R] time-embedding block.
    """
    H = int(t_end - t_test + 1)
    N_trip = full_norm_tensor.shape[1]
    all_trip_indices = torch.arange(N_trip, device=device, dtype=torch.long)

    for h in range(H):
        t = t_test + h
        time_emb = predicted_time_embeddings_block[h]  # [R]
        gt_slice_norm = full_norm_tensor[t]  # [N_trip, F]

        for fidx in target_feature_indices_list:
            gt_norm = gt_slice_norm[:, fidx]  # [N_trip]
            valid = ~torch.isnan(gt_norm)
            if not torch.any(valid):
                continue

            pred_norm = model(
                mode="forecast",
                predicted_time_embedding=time_emb,
                trip_index=all_trip_indices,
                feature_index=torch.tensor(
                    [int(fidx)], device=device, dtype=torch.long
                ),
            ).squeeze(
                -1
            )  # [N_trip]

            std_f = float(feat_stds_dev[fidx].item())
            mean_f = float(feat_means_dev[fidx].item())
            gt_raw = gt_norm[valid] * std_f + mean_f
            pred_raw = pred_norm[valid] * std_f + mean_f

            for trip_i, y, yhat in zip(
                torch.where(valid)[0].tolist(),
                gt_raw.detach().cpu().tolist(),
                pred_raw.detach().cpu().tolist(),
            ):
                agg_list.append(
                    {
                        "t": int(t),
                        "trip": int(trip_i),
                        "feature": int(fidx),
                        "GroundTruth": float(y),
                        "Forecasted": float(yhat),
                    }
                )


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
        "--trainable_weights",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, learn 4-way softmax over losses.",
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
        default=0,
        help="Frequency: 0=high (hour/day), 1=week/month, 2=quarter/year. ETTh1 hourly → 0.",
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
    # near your other parser.add_argument(...) calls
    parser.add_argument(
        "--window_stride",
        type=str,
        default="C+H",
        help="How far to advance the window start between folds. "
        "Accepts an integer (e.g. '1', '128') or one of 'C', 'H', 'C+H'. "
        "Default: C+H (no overlap between windows).",
    )

    # ADD ↑↑↑
    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--forecast_horizon", type=int, default=96)  # or 192
    # --- argparse additions (near the other feature args) ---
    parser.add_argument(
        "--target_feature_indices",
        type=str,
        default=None,
        help='Comma-separated list of target feature indices (e.g. "13,14,15"). '
        "If set, this overrides --target_feature_index.",
    )
    parser.add_argument(
        "--w_forecast",
        type=float,
        default=1.0,
        help="Weight for the target-forecast loss over the validation block.",
    )
    parser.add_argument(
        "--mlp_layers",
        type=int,
        default=3,  # total Linear layers (>=2): input→hidden...→output
        help="Total number of Linear layers in the predictor MLP. "
        "Example: 3 => two hidden layers + output; 5 => four hidden layers + output.",
    )

    args = parser.parse_args()
    # --- normalize targets to a list ---
    if args.target_feature_indices and args.target_feature_indices.strip():
        args.target_feature_indices_list = [
            int(x.strip())
            for x in args.target_feature_indices.split(",")
            if x.strip() != ""
        ]
    else:
        # backward compatible single-target default
        args.target_feature_indices_list = [int(args.target_feature_index)]

    # Guardrail: make it explicit we only score targets here
    assert (
        isinstance(args.target_feature_indices_list, list)
        and len(args.target_feature_indices_list) > 0
    ), "target_feature_indices_list must be a non-empty list of target feature indices (no covariates)."

    # ADD ↓↓↓ (unique run hash to suffix CSVs)
    # make a small dict out of args for hashing (all keys sorted for stability)
    args_as_dict_for_hash = {
        k: str(getattr(args, k)) for k in sorted(vars(args).keys())
    }  # dict[str->str]
    # # Make TimesFM produce H-step forecasts (we'll still truncate at boundaries if needed)
    args.timesfm_horizon_len = int(
        args.forecast_horizon
    )  # (int) keep a single horizon len everywhere

    def _print_mode_summary(args):
        """
        Explain run mode in plain words (ESL-friendly).
        """
        # using_timesfm: True → TimesFM zero-shot; False → GRU autoregressive
        using_timesfm: bool = bool(args.use_timesfm)

        # forecast_horizon_steps: how many future steps per window (H)
        forecast_horizon_steps: int = int(args.forecast_horizon)

        # orthogonality_norm_name: "L2" (vanilla) or "L1"
        orthogonality_norm_name: str = str(args.ortho_penalty_norm)

        # orthogonality_weight_value: lambda for decorrelation penalty
        orthogonality_weight_value: float = float(args.lambda_ortho_time)

        # mode_text: human-readable mode name for console
        mode_text: str = (
            "TimesFM (zero-shot)" if using_timesfm else "GRU (autoregressive)"
        )

        # Final one-line summary print
        print(
            f"[MODE] {mode_text} | H={forecast_horizon_steps} | "
            f"ortho={orthogonality_norm_name} (lambda={orthogonality_weight_value:g})"
        )

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

    # -------- [W&B] init (sweep-safe) --------
    run = wandb.init(  # (no shape) / type: wandb Run / why: start a W&B run
        project=os.environ.get(
            "WANDB_PROJECT", "eps-sundial-tensor"
        ),  # (str) / which project in W&B
        mode=os.environ.get(
            "WANDB_MODE", "online"
        ),  # (str) / "online" or "offline" or "disabled"
        config=vars(
            args
        ),  # (dict[str->value]) / give argparse defaults; sweeps will override
    )
    cfg = run.config  # (mapping-like) / sweep-merged config lives here

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
    # Re-sync: we want TimesFM to use the same horizon as our rolling window
    args.timesfm_horizon_len = int(args.forecast_horizon)

    # ---- TimesFM implies: no GRU prediction loss (we do not train GRU to predict next time embedding) ----
    # If using TimesFM, kill the GRU prediction loss (advisor request)
    if args.use_timesfm:
        args.w_predict = 0.0  # (float) always zero when TimesFM is used
        print("[INFO] Using TimesFM: set w_predict=0.0; prediction loss disabled.")

    _print_mode_summary(args)  # <-- [NEW] print final, sweep-merged mode summary
    # optional naming/grouping from env (purely cosmetic in UI)
    wandb_name_from_env = os.environ.get(
        "WANDB_NAME"
    )  # (str or None) / human-friendly run name
    wandb_group_from_env = os.environ.get(
        "WANDB_RUN_GROUP"
    )  # (str or None) / group name for runs
    if wandb_name_from_env:
        wandb.run.name = wandb_name_from_env  # (no shape) / apply if provided
    if wandb_group_from_env:
        wandb.run.group = wandb_group_from_env  # (no shape) / apply if provided
    # ------------------------------------------
    # ---- (optional) define a global step and metrics tied to it ----
    wandb.define_metric(
        "global_step"
    )  # (no shape) / type: define / why: create a step axis
    wandb.define_metric(
        "*", step_metric="global_step"
    )  # (no shape) / type: define / why: all metrics use that step

    wandb.log({"hparams/epochs_per_window": int(args.epochs_per_window)}, step=0)

    # ---- (optional) make a unique output dir using current weight values ----
    run_output_dir = os.path.join(
        args.output_dir, f"wr_{args.w_recon}_wp_{args.w_predict}_wf_{args.w_forecast}"
    )

    os.makedirs(
        run_output_dir, exist_ok=True
    )  # (no shape) / make that folder if missing
    args.output_dir = (
        run_output_dir  # (no shape) / from now on, write under this run folder
    )
    print(
        f"[INFO] Outputs will be saved under: {args.output_dir}"
    )  # (no shape) / friendly print
    # --- NEW: create per-epoch logging file paths and containers (used across the whole run) ---
    # (str) unique JSON Lines file for one-record-per-epoch logging
    metrics_log_path = os.path.join(
        args.output_dir,
        f"per_epoch_metrics_{args.run_hash}.jsonl",  # e.g., '.../per_epoch_metrics_a1b2c3d4.jsonl'
    )

    # (str) unique CSV for easy spreadsheet viewing of per-epoch metrics
    epoch_csv_path = os.path.join(
        args.output_dir,
        f"per_epoch_metrics_{args.run_hash}.csv",  # e.g., '.../per_epoch_metrics_a1b2c3d4.csv'
    )

    # (list[pd.DataFrame]) keep all test-step prediction tables across windows for the final summary
    all_test_rows = []

    # (bool) if the CSV does not exist yet, write a header the first time we append rows later
    if not os.path.exists(epoch_csv_path):
        with open(epoch_csv_path, "w") as csv_header_file:
            csv_header_file.write(
                # CSV header columns (plain text, comma-separated)
                "window,epoch,train_end,"
                "val_block_start,val_block_end,"
                "test_block_start,test_block_end,"
                "train_recon_mse,train_pred_mse,"
                "train_r2_reconstruction,train_r2_reconstruction_kelly,"
                "test_rmse,test_mae,test_r2,test_r2_kelly,"
                "w_recon,w_pred,w_forecast,"
                "mean_abs_offdiag\n"
            )
    # --- END NEW ---

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- TimesFM warm-up: do one tiny call so weights are loaded and cached ---
    if args.use_timesfm:
        try:
            tfm = _get_timesfm_model(
                prefer_device=device,
                huggingface_repo_id=args.timesfm_repo_id,
                horizon_len=int(args.timesfm_horizon_len),
                per_core_batch_size=int(args.timesfm_per_core_batch_size),
                context_cap=int(args.timesfm_context_cap),
            )
            # Make a tiny dummy forecast: one flat series of length 8, horizon 1
            _ = tfm.forecast(
                inputs=[np.zeros(8, dtype=np.float32)],
                freq=[int(args.timesfm_freq_category)],
            )
            print("[INFO] TimesFM warmed.")
        except Exception as e:
            warnings.warn(f"[WARN] TimesFM warm-up skipped: {e}")

    # === 3-D tensor (time, trip, feature) ===
    # Expect a numpy file shaped [T_total, N_trip, F_total]
    tensor_3d_path = "./rideshare_tensor_masked_0.9.npy"  # <— change to your file
    if not os.path.exists(tensor_3d_path):
        sys.exit(f"Error: {tensor_3d_path} not found")

    full_raw_cpu = torch.from_numpy(
        np.load(tensor_3d_path)
    ).float()  # [T_total, N_trip, F_total]
    if full_raw_cpu.dim() != 3:
        sys.exit(
            f"3-D tensor must be [T, N_trip, F], got shape={tuple(full_raw_cpu.shape)}"
        )

    T_total, N_trip, F_total = full_raw_cpu.shape
    full_raw_dev = full_raw_cpu.to(device, non_blocking=True)

    # Choose your target feature and drivers (indexes in last dimension)
    # (Example below assumes EPS-like target at column 13; change to your real indices)
    # DO NOT include the target in drivers.
    # Keep CLI overrides working:

    # nT, D1 = full_raw_cpu.shape
    nT, N_trip, F_total = full_raw_cpu.shape
    # remove: nT, D1 = full_raw_cpu.shape
    nT = T_total

    full_raw_dev = full_raw_cpu.to(device, non_blocking=True)  # on device for fast math
    print(f"Loaded raw tensor shape: {full_raw_cpu.shape}")
    print(f"T={T_total}, N_trip={N_trip}, F={F_total}")

    # ---- Determine initial window indexes ----
    start_split = int(args.start_split_ratio * nT)  # e.g., 40 when nT=50 and ratio=0.8
    # We will iterate train_end from (start_split-1) up to (nT-3) so that test = train_end+2 <= nT-1
    first_train_end = start_split - 1
    last_train_end = nT - 3
    if first_train_end < 1 or last_train_end < first_train_end:
        sys.exit("Sequence too short for rolling windows with given split ratio.")

    def compute_stats_from_train_slice(train_end_inclusive: int):
        # [T_train, N_trip, F]
        train_raw = full_raw_cpu[: train_end_inclusive + 1]
        arr = train_raw.numpy()

        # Winsorize per feature across both time & trips
        lowers = np.nanquantile(arr, 0.01, axis=(0, 1))  # [F]
        uppers = np.nanquantile(arr, 0.99, axis=(0, 1))  # [F]

        arr_w = np.clip(arr, lowers[None, None, :], uppers[None, None, :])

        feat_means = np.nanmean(arr_w, axis=(0, 1))  # [F]
        feat_stds = np.nanstd(arr_w, axis=(0, 1))  # [F]

        # Safety
        feat_means = np.where(np.isfinite(feat_means), feat_means, 0.0)
        feat_stds = np.where((feat_stds > 0) & np.isfinite(feat_stds), feat_stds, 1.0)

        feat_means = torch.from_numpy(feat_means).float().to(device)  # [F]
        feat_stds = torch.from_numpy(feat_stds).float().to(device)  # [F]
        return feat_means, feat_stds

    # ---- Initial stats (fixed by default) ----
    init_means_dev, init_stds_dev = compute_stats_from_train_slice(int(first_train_end))

    full_norm_dev = (full_raw_dev - init_means_dev[None, None, :]) / init_stds_dev[
        None, None, :
    ]

    # ---- Model + optimizer (init once; we update model.train_T per window) ----
    def make_model_and_optim():
        model = CP3DWithTimesFM(
            (T_total, N_trip, F_total),  # (time_count, trip_count, feature_count)
            time_embedding_size=args.Rank_CP,
            mlp_hidden=args.mlp_hidden,
            dropout_p=args.dropout_p,
            device=str(device),
            mlp_layers=int(args.mlp_layers),  # NEW
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        return model, optimizer

    model, optimizer = (
        make_model_and_optim()
    )  # prints: (model, opt); why: initial objects (opt will be replaced below anyway)
    model.to(device)  # prints: none;    why: move to GPU/CPU

    # Light watch; gradients can be noisy with torch.compile
    wandb.watch(model, log=None)

    # --- NEW: ensure the EPS regressor exists BEFORE optimizer/compile ---

    model = torch.compile(  # prints: none;          why: compile AFTER regressor exists
        model, mode="reduce-overhead"
    )

    def current_loss_weights() -> torch.Tensor:
        """
        Returns fixed (non-trainable) weights in a 3-vector:
        index 0 -> w_recon      (float; reconstruction loss weight)
        index 1 -> w_prediction (float; forced 0.0 when TimesFM is used)
        index 2 -> w_forecast   (float; target-only forecast loss weight)
        dtype: float32
        device: same CUDA/CPU as the model.
        """
        # Scalars -> Python floats
        w_recon_float: float = float(args.w_recon)
        w_pred_float: float = 0.0 if bool(args.use_timesfm) else float(args.w_predict)
        w_forecast_float: float = float(args.w_forecast)

        # Return 1-D tensor [3] on the right device for later broadcast/mul
        return torch.tensor(
            [w_recon_float, w_pred_float, w_forecast_float],
            device=device,
            dtype=torch.float32,
        )

    def build_optimizer(m: nn.Module):
        """Build plain Adam over model parameters (3-loss setup)."""
        return torch.optim.Adam(
            m.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

    # (re)build optimizer so it includes loss_logits (if any) + regressor params
    optimizer = build_optimizer(
        model
    )  # prints: torch.optim;   why: final optimizer with all params

    # Rolling by forecast_horizon (stride = H). We stop when there are no validation steps left.
    window_indices = []  # list[(train_end, val_start, val_end, test_start, test_end)]
    # current_train_end = first_train_end
    # ---------------- Single train/val/test split ----------------
    C = int(args.context_len)  # context length
    H = int(args.forecast_horizon)  # forecast horizon
    t_end = T_total - 1
    t_test = t_end - H + 1  # test starts H steps from the end

    # === Your requested assertions ===
    # 1) assert: C + H > t_end - H
    if not (C + H > (t_end - H)):
        sys.exit(
            f"Split check failed: require C+H > (t_end - H). "
            f"Got C={C}, H={H}, t_end={t_end} -> {C+H} ≤ {t_end-H}."
        )

    # 2) validation block: [C+1 .. t_test-1], must be shorter than H
    val_start = C + 1
    val_end = t_test - 1
    val_len = max(0, val_end - val_start + 1)
    if not (val_len < H):
        sys.exit(
            f"Validation length must be < H. Got val_len={val_len}, H={H}. "
            f"(C={C}, t_test={t_test}, val=[{val_start}..{val_end}])"
        )

    # ---- Normalization computed from the TRAIN slice [0..C] (inclusive) ----
    def compute_stats_from_train_slice_inclusive(end_idx_inclusive: int):
        train_raw = full_raw_cpu[: end_idx_inclusive + 1]  # [C+1, N_trip, F]
        arr = train_raw.numpy()
        lowers = np.nanquantile(arr, 0.01, axis=(0, 1))
        uppers = np.nanquantile(arr, 0.99, axis=(0, 1))
        arr_w = np.clip(arr, lowers[None, None, :], uppers[None, None, :])
        feat_means = np.nanmean(arr_w, axis=(0, 1))
        feat_stds = np.nanstd(arr_w, axis=(0, 1))
        feat_means = np.where(np.isfinite(feat_means), feat_means, 0.0)
        feat_stds = np.where((feat_stds > 0) & np.isfinite(feat_stds), feat_stds, 1.0)
        feat_means = torch.from_numpy(feat_means).float().to(device)
        feat_stds = torch.from_numpy(feat_stds).float().to(device)
        return feat_means, feat_stds

    feat_means_dev, feat_stds_dev = compute_stats_from_train_slice_inclusive(C)
    full_norm_dev = (full_raw_dev - feat_means_dev[None, None, :]) / feat_stds_dev[
        None, None, :
    ]

    # ---- Build train dataset over [0..C] only ----
    train_norm_cpu = full_norm_dev[: C + 1].detach().cpu()  # [C+1, N_trip, F]
    train_ds = FilteredTensor3DDataset(train_norm_cpu, t_offset=0)

    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "generator": g_dl,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 8
    train_ld = DataLoader(train_ds, **dataloader_kwargs)

    # ---- Precompute history for VAL and TEST blocks ----
    # For validation block forecasts, use the last C points ending at t=C.
    # History indices length==C: [C-C+1 .. C] if C>0 else [].
    hist_start_val = max(0, C - C + 1)  # typically 1
    history_time_indices_val = torch.arange(hist_start_val, C + 1, device=device)

    # For test block, context = [t_test - C .. t_test-1] (length C)
    hist_start_test = t_test - C
    if hist_start_test < 0:
        sys.exit(f"Test context would start before t=0: t_test={t_test}, C={C}.")
    history_time_indices_test = torch.arange(hist_start_test, t_test, device=device)

    # ---- MASE denominator on RAW training slice [0..C] ----
    train_raw_slice = full_raw_dev[: C + 1]  # device tensor
    mase_den = compute_mase_denominator_raw_fibers(
        train_raw_slice, args.target_feature_indices_list, 24
    )

    # ---- Train for epochs_per_window epochs on the TRAIN slice ----
    mse_loss = nn.MSELoss()
    model.train()
    inner = tqdm(
        range(1, args.epochs_per_window + 1),
        desc=f"[SingleSplit] Train=0..{C}  Val={val_start}..{val_end}  Test={t_test}..{t_end}",
        leave=False,
        position=1,
    )

    for ep in inner:
        recon_sse = 0.0
        recon_n = 0
        pred_sse = 0.0
        pred_n = 0
        recon_sum_y = 0.0
        recon_sum_y2 = 0.0
        opt_loss_running = 0.0

        # (Step-3) build once per epoch: block future embeddings over VAL (length val_len)
        if val_len > 0:
            with torch.no_grad():
                pred_time_emb_val_blk = get_future_time_embeddings_via_gru_or_timesfm(
                    model=model,
                    history_time_indices=history_time_indices_val,
                    steps_ahead=val_len,
                    use_timesfm=args.use_timesfm,
                    timesfm_repo_id=args.timesfm_repo_id,
                    timesfm_freq_category=int(args.timesfm_freq_category),
                    timesfm_context_cap=int(args.timesfm_context_cap),
                    timesfm_per_core_batch_size=int(args.timesfm_per_core_batch_size),
                ).detach()  # [val_len, R]
        else:
            pred_time_emb_val_blk = None

        for idxs, vals in train_ld:
            idxs = idxs.to(device, non_blocking=True)
            vals = vals.to(device, non_blocking=True)
            time_index_batch, trip_index_batch, feature_index_batch = idxs.T

            # weights (recon, pred, forecast)
            w = current_loss_weights()

            # ---- STEP 1: reconstruction on observed triples ----
            optimizer.zero_grad(set_to_none=True)
            pred_vals = model(
                mode="reconstruction",
                time_index=time_index_batch,
                trip_index=trip_index_batch,
                feature_index=feature_index_batch,
            )
            loss_recon = mse_loss(pred_vals, vals)

            # orthogonality penalty on history [0..C] (use last C points)
            time_indices_history = torch.arange(hist_start_val, C + 1, device=device)
            time_mat = model.time_embeddings(time_indices_history)  # [C, R]
            ortho_pen = (
                decorrelation_L2(time_mat)
                if args.ortho_penalty_norm == "L2"
                else decorrelation_L1(time_mat)
            )
            lam = torch.tensor(
                float(args.lambda_ortho_time), device=device, dtype=time_mat.dtype
            )

            total_step1 = (w[0] * loss_recon) + lam * ortho_pen
            total_step1.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            opt_loss_running += float(total_step1.detach().item())

            # bookkeeping (normalized space)
            n_batch = vals.numel()
            recon_sse += loss_recon.item() * n_batch
            recon_n += n_batch
            recon_sum_y += float(vals.sum().item())
            recon_sum_y2 += float((vals * vals).sum().item())

            # ---- STEP 2: GRU prediction loss (skip in TimesFM mode) ----
            if not args.use_timesfm:
                optimizer.zero_grad(set_to_none=True)
                next_pred = model(
                    mode="prediction", history_time_indices=history_time_indices_val
                )
                true_next = model.time_embeddings(torch.tensor([C + 1], device=device))[
                    0
                ].detach()
                loss_pred = mse_loss(next_pred, true_next)
                (
                    w[1] * loss_pred
                ).backward()  # backprop prediction loss? Will double check.
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                pred_sse += loss_pred.item() * next_pred.numel()
                pred_n += int(next_pred.numel())
            else:
                loss_pred = torch.tensor(0.0, device=device)

            # ---- STEP 3: target-only forecast MSE over VAL block (no grads into time/GRU) ----
            """
            ChatGPT: I want this sequential model to be frozen, obut I also want to update the time embeddings.
            """
            for p in model.time_embeddings.parameters():
                p.requires_grad_(False)
            for p in model.sequence_model.parameters():
                p.requires_grad_(False)
            optimizer.zero_grad(set_to_none=True)
            if val_len > 0:
                loss_forecast = forecast_mse_targets_over_block(
                    model=model,
                    full_norm_tensor=full_norm_dev,
                    predicted_time_embeddings_block=pred_time_emb_val_blk,
                    validation_start_index=val_start,
                    validation_end_index=val_end,
                    target_feature_indices_list=args.target_feature_indices_list,
                    device=device,
                )
            else:
                loss_forecast = torch.tensor(0.0, device=device, dtype=torch.float32)
            # TODO: check TimesFM's parameter: Compute gradient, not update TimesFM, update time embeddings.
            # TimesFM freeze:
            # Design Check: Quality Check.
            (w[2] * loss_forecast).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for p in model.time_embeddings.parameters():
                p.requires_grad_(True)
            for p in model.sequence_model.parameters():
                p.requires_grad_(True)

        # ---- epoch-end diagnostics ----
        recon_mse = (recon_sse / recon_n) if recon_n > 0 else float("nan")
        pred_mse = (
            0.0
            if args.use_timesfm
            else ((pred_sse / pred_n) if pred_n > 0 else float("nan"))
        )

        # decorrelation diagnostics
        time_mat = model.time_embeddings(
            torch.arange(hist_start_val, C + 1, device=device)
        )
        offdiag_mean_abs = (
            _zero_diagonal(_covariance_centered(_center_time(time_mat)))
            .abs()
            .mean()
            .detach()
            .item()
        )
        ortho_epoch = (
            (
                decorrelation_L2(time_mat)
                if args.ortho_penalty_norm == "L2"
                else decorrelation_L1(time_mat)
            )
            .detach()
            .item()
        )

        # R² on reconstruction (normalized)
        def _safe_r2_from_sse(sse, sum_y, sum_y2, n):
            if n <= 0:
                return float("nan"), float("nan")
            ss_tot = sum_y2 - (sum_y * sum_y) / n
            r2 = 1.0 - (sse / ss_tot) if ss_tot > 0 else float("nan")
            r2_k = 1.0 - (sse / sum_y2) if sum_y2 > 0 else float("nan")
            return r2, r2_k

        train_recon_r2, train_recon_r2_k = _safe_r2_from_sse(
            recon_sse, recon_sum_y, recon_sum_y2, recon_n
        )

        w_now = current_loss_weights().detach().cpu().tolist()
        inner.set_postfix(
            {
                "avg_opt_loss": f"{opt_loss_running/ max(1,len(train_ld)) + float(args.lambda_ortho_time)*ortho_epoch:.3f}",
                "recon": f"{recon_mse:.3f}",
                "pred": f"{pred_mse:.3f}",
                "w": f"[{w_now[0]:.3f},{w_now[1]:.3f},{w_now[2]:.3f}]",
                "mean_offdiag": f"{offdiag_mean_abs:.2e}",
            }
        )
        # end of each epoch (right after you compute recon_mse, pred_mse, train_recon_r2, etc.)
        with open(epoch_csv_path, "a") as f:
            f.write(
                f"0,{ep},{C},{val_start},{val_end},{t_test},{t_end},"
                f"{recon_mse:.6f},{pred_mse:.6f},"
                f"{train_recon_r2:.6f},{train_recon_r2_k:.6f},"
                f"NA,NA,NA,NA,"  # placeholders for test metrics at epoch time
                f"{w_now[0]:.6f},{w_now[1]:.6f},{w_now[2]:.6f},"
                f"{offdiag_mean_abs:.6e}\n"
            )

    # ---------------- Test evaluation (block forecast) ----------------
    # Predict H-step block of time-embeddings from the test context
    future_time_embeddings_test = get_future_time_embeddings_via_gru_or_timesfm(
        model=model,
        history_time_indices=history_time_indices_test,
        steps_ahead=H,
        use_timesfm=args.use_timesfm,
        timesfm_repo_id=args.timesfm_repo_id,
        timesfm_freq_category=int(args.timesfm_freq_category),
        timesfm_context_cap=int(args.timesfm_context_cap),
        timesfm_per_core_batch_size=int(args.timesfm_per_core_batch_size),
    )  # [H, R]

    # right after t_test = t_end - H + 1
    assert (
        t_end - t_test + 1
    ) == H, f"Expected test block length H={H}, got {t_end - t_test + 1}."
    assert (
        t_test - C
    ) >= 0, f"Context would start before t=0 (C={C}, t_test={t_test})."

    # right after future_time_embeddings_test = get_future_time_embeddings_via_...
    assert (
        future_time_embeddings_test.size(0) >= H
    ), f"Got only {future_time_embeddings_test.size(0)} future embeddings; expected at least H={H}."

    # Aggregate metrics (RAW space)
    metrics = evaluate_test_block_blockforecast_raw(
        model=model,
        full_norm_tensor=full_norm_dev,
        feat_means_dev=feat_means_dev,
        feat_stds_dev=feat_stds_dev,
        target_feature_indices_list=args.target_feature_indices_list,
        t_test=int(t_test),
        t_end=int(t_end),
        predicted_time_embeddings_block=future_time_embeddings_test,
        device=device,
        mase_denominator=mase_den,
    )

    # Save per-cell rows (RAW) for inspection
    all_test_rows = []
    append_test_rows_from_block_embeddings_raw(
        all_test_rows,
        model=model,
        full_norm_tensor=full_norm_dev,
        feat_means_dev=feat_means_dev,
        feat_stds_dev=feat_stds_dev,
        target_feature_indices_list=args.target_feature_indices_list,
        t_test=int(t_test),
        t_end=int(t_end),
        predicted_time_embeddings_block=future_time_embeddings_test,
        device=device,
    )

    print(
        f"\n=== TEST [{t_test}..{t_end}] ===\n"
        f"RMSE={metrics['rmse']:.6f}  MAE={metrics['mae']:.6f}  MASE={metrics['mase']:.6f}  "
        f"R²={metrics['r2']:.6f}  Kelly R²={metrics['r2_kelly']:.6f}"
    )

    # Optionally persist summary and rows (like your original)
    if all_test_rows:
        all_df = (
            pd.DataFrame(all_test_rows)
            .sort_values(["t", "trip", "feature"])
            .reset_index(drop=True)
        )
        out_all_csv = os.path.join(
            args.output_dir,
            f"single_split_H{H}_test_predictions_{args.run_hash}_mlp_layer_{args.mlp_layers}_rank_{args.Rank_CP}.csv",
        )
        all_df.to_csv(out_all_csv, index=False)
        print(f"Saved per-cell test table → {out_all_csv}")

        # final line (CSV one-liner summary)
        final_csv = os.path.join(
            args.output_dir,
            f"single_split_metrics_{args.run_hash}_mlp_layer_{args.mlp_layers}_rank_{args.Rank_CP}.csv",
        )
        write_header = not os.path.exists(final_csv)
        with open(final_csv, "a") as f:
            if write_header:
                f.write("H,C,t_test,t_end,RMSE,MAE,MASE,R2,R2_Kelly\n")
            f.write(
                f"{H},{C},{t_test},{t_end},{metrics['rmse']:.6f},{metrics['mae']:.6f},{metrics['mase']:.6f},{metrics['r2']:.6f},{metrics['r2_kelly']:.6f}\n"
            )
        print(f"Saved test metrics → {final_csv}")

    print("\nDone.")
    # ADD ↓↓↓ (just before wandb.finish())
    wandb.summary["run/hash"] = args.run_hash  # record hash in summary
    # ADD ↑↑↑

    wandb.finish()  # prints: None; meaning: closes the run cleanly


if __name__ == "__main__":
    main()

    # w_recon = 0. --> if still >= 74% r^2, questions about the tensor, is that useful?

    # random shuffle

    # w_recon=1e-5 to 1e-1, w_pred=1e-5 to 1e-1. w_forecast = 1 fixed.

    # big discrepency
