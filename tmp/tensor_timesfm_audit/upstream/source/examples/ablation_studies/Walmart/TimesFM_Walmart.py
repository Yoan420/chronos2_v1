#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TimesFM-only baseline (zero-shot, rolling windows) on T1 sales units.

What this script does (baseline, no tensors/MLP learning):
1) Read three tensors T1/T2/T3 (we only use T1's sales units for y).
2) Convert T1 into a single long DataFrame with columns: [unique_id, ds, y].
   - unique_id := "item_{i}_store_{j}"
   - ds := daily timestamp (increase monotonically)
   - y  := sales units
3) Save that DataFrame to CSV (so the dataset is standardized).
4) Initialize TimesFM (2.0 checkpoint) and do rolling zero-shot forecasts on the last 20%:
   - window stride = 28, context_len = 365, horizon_len = 28 (clipped at series end)
5) Compute test metrics across the whole test period: RMSE, R², Kelly-R², WRMSSE.
   - WRMSSE is computed per (item,store) using training history for scale denominators.

Usage:
  pip install "timesfm==1.3.0"  # to use v1/v2 API for 2.0 checkpoints
  python timesfm_baseline_rolling.py --cuda 1

References:
- HF model card (DataFrame format, API usage, freq flags, context/horizon):
  google/timesfm-2.0-500m-pytorch  (see unique_id/ds/y and forecast_on_df examples)
- TimesFM GitHub (v1/v2 code path + pip pin):
  "1.0 and 2.0 relevant code archived in sub directory v1; pip install timesfm==1.3.0"
"""

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np  # type: module, array ops
import pandas as pd  # type: module, tabular data
import torch  # type: module, tensor ops (used for metrics convenience)
from tqdm.auto import tqdm

# Import TimesFM v1/v2 inference API:
# - class TimesFm: main inference driver (v1/v2 API)
# - class TimesFmHparams: configure backend, batch size, horizon, context, dims
# - class TimesFmCheckpoint: tells it which HF repo checkpoint to load
import timesfm  # type: module; comes from pip install timesfm==1.3.0
from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint  # uncommon import names

torch.set_float32_matmul_precision("high")  # improve FP32 matmul speed (safe)


# ----------------------------
# Metrics (vanilla, torch-based)
# ----------------------------
# --- helper (put near your other utils) ---
# 1) Keep this helper near your other utils
def left_pad_to_multiple(x: np.ndarray, base: int = 32) -> np.ndarray:
    """Pad on the LEFT to the next multiple of `base` by repeating the first value."""
    L = int(x.shape[0])
    need = (-L) % base
    if need == 0:
        return x.astype(np.float32, copy=False)
    pad_val = float(x[0]) if L > 0 else 0.0
    pad = np.full((need,), pad_val, dtype=np.float32)
    return np.concatenate([pad, x.astype(np.float32, copy=False)], axis=0)


def rmse_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    y_true: FloatTensor [...], ground truth
    y_pred: FloatTensor [...], predictions (same shape)
    returns: scalar FloatTensor = sqrt(mean((y_pred - y_true)^2))
    """
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


def r2_torch(
    y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    Standard R^2 = 1 - SS_res / SS_tot.
    Clamp denominator to avoid divide-by-zero.
    """
    ss_res = torch.sum((y_true - y_pred) ** 2)
    mean_y = torch.mean(y_true)
    ss_tot = torch.sum((y_true - mean_y) ** 2)
    ss_tot = torch.clamp(ss_tot, min=eps)
    return 1.0 - ss_res / ss_tot


def r2_kelly_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Kelly R^2 = 1 - sum((y - yhat)^2) / sum(y^2)
    """
    diff = torch.sum((y_true - y_pred) ** 2) / torch.sum(y_true**2)
    return 1 - diff


def wrmsse_torch(
    indices_3d: torch.LongTensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    train_tensor_T1: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    indices_3d: LongTensor [N, 3] = (t_idx, item_idx, store_idx) for the test region flattened
    y_true    : FloatTensor [N]
    y_pred    : FloatTensor [N]
    train_tensor_T1: FloatTensor [T_train, I, S] used for RMSSE scale (differences of sales in training only)
    returns   : scalar FloatTensor = WRMSSE averaged across series (equal weights)

    Implementation follows the standard RMSSE idea:
      RMSSE(series) = RMSE_forecast / sqrt(mean((train[t] - train[t-1])^2))
    """
    unique_pairs = torch.unique(
        indices_3d[:, 1:], dim=0
    )  # all (item, store) pairs in the test set
    rmsse_list = []

    for key in unique_pairs:
        item_index = key[0].item()
        store_index = key[1].item()

        mask = (indices_3d[:, 1] == key[0]) & (indices_3d[:, 2] == key[1])
        if mask.sum() == 0:
            continue

        y_true_series = y_true[mask]
        y_pred_series = y_pred[mask]
        mse_forecast = torch.mean((y_true_series - y_pred_series) ** 2)
        rmse_forecast = torch.sqrt(mse_forecast)

        series_train = train_tensor_T1[:, item_index, store_index]  # [T_train]
        # Find first nonzero to avoid long leading zeros
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
        return torch.tensor(0.0, device=train_tensor_T1.device)
    return sum(rmsse_list) / len(rmsse_list)


# ----------------------------
# Data prep: tensors -> long CSV
# ----------------------------


def build_timesfm_dataframe_from_T1(
    tensor_T1: torch.Tensor, start_date_str: str = "2000-01-01"
) -> pd.DataFrame:
    """
    tensor_T1: FloatTensor [T, I, S] = sales units
    start_date_str: str, base date used to generate daily timestamps

    Returns a DataFrame with columns:
      unique_id: str, "item_{i}_store_{j}"
      ds       : pd.Timestamp (daily)
      y        : float, sales units
    This is the exact format TimesFM expects. (Other columns would be ignored by baseline.)
    """
    # Ensure on CPU & numpy for pandas
    T, I, S = tensor_T1.shape  # T = total time, I = num items, S = num stores
    date_index = pd.date_range(
        start=start_date_str, periods=T, freq="D"
    )  # daily timestamps increasing

    rows = []
    # Simple nested loops for clarity. You can vectorize later if needed.
    for i in range(I):
        for j in range(S):
            unique_id_str = f"item_{i}_store_{j}"  # human-readable series id
            y_np = tensor_T1[:, i, j].detach().cpu().numpy().astype(np.float32)  # [T]
            df_series = pd.DataFrame(
                {"unique_id": unique_id_str, "ds": date_index, "y": y_np}
            )
            rows.append(df_series)
    df_all = pd.concat(rows, ignore_index=True)
    return df_all


# ----------------------------
# Rolling forecasting with TimesFM (zero-shot)
# ----------------------------


def rolling_timesfm_zero_shot(
    dataframe_all: pd.DataFrame,
    horizon_len: int,
    context_len: int,
    stride_len: int,
    times_total: int,
    time_test_start_idx: int,
    number_items: int,
    number_stores: int,
    per_core_batch_size: int,  # NEW
    series_batch_size: int,  # NEW
    device_backend: str = "gpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    dataframe_all: long DataFrame with columns [unique_id, ds, y], sorted by ds within each unique_id
    horizon_len  : forecast horizon for each window (28)
    context_len  : max context length to feed into TimesFM (365)
    stride_len   : stride between rolling windows (28)
    times_total  : T, total timesteps
    time_test_start_idx: T_test start index (int)
    number_items : I
    number_stores: S
    device_backend: "gpu" or "cpu" for TimesFM backend

    Returns:
      preds_test: FloatTensor [T_test_len, I, S] with predictions; zeros where not predicted
      counts    : FloatTensor [T_test_len, I, S] accumulation counts for averaging (if overlapping windows)
    """

    # 1) Build Python lists (one array per series) in a fixed order to pass into TimesFM
    #    We'll keep a mapping unique_id -> (item_idx, store_idx) so we can place forecasts back correctly.
    #    Shapes and types:
    #       - series_dict[unique_id] : np.ndarray [T] float32, full history of that series
    #       - series_order           : List[str], the order we feed series into TimesFM
    df = dataframe_all.copy()
    df.sort_values(["unique_id", "ds"], inplace=True)

    # Make grouping fast & memory-friendly
    df["unique_id"] = df["unique_id"].astype("category")

    # Preserve category order to keep a stable series ordering
    series_order: List[str] = df["unique_id"].cat.categories.tolist()

    # groupby(..., observed=False) silences the pandas FutureWarning about the default.
    series_dict: Dict[str, np.ndarray] = {
        uid: grp["y"].to_numpy(dtype=np.float32)  # np.ndarray [T] per series, float32
        for uid, grp in tqdm(
            df.groupby("unique_id", sort=False, observed=False),
            desc="Building series_dict",
            total=len(series_order),
        )
    }

    # Helper to decode unique_id back to (i, j) integer indices
    def parse_uid(uid: str) -> Tuple[int, int]:
        # uid is "item_{i}_store_{j}"
        left = uid.split("item_")[1]
        i_str, store_part = left.split("_store_")
        return int(i_str), int(store_part)

    # 2) Prepare output tensors (on CPU; move to GPU if you want later)
    test_len = times_total - time_test_start_idx  # length of test period
    preds_test = torch.zeros(
        (test_len, number_items, number_stores), dtype=torch.float32
    )
    counts = torch.zeros_like(preds_test)

    # 3) Initialize TimesFM model ONCE (reuse across windows)
    #    Important: We're using the 2.0 (500M, PyTorch) checkpoint with the v1/v2 API.
    #    Hparams fields (uncommon names explained):
    #      - backend: "gpu" or "cpu"
    #      - per_core_batch_size: microbatch size per device core for inference
    #      - horizon_len: the max horizon the model is configured for (28 here)
    #      - context_len: max history length model will use (<= 2048; we use 365)
    #      - input_patch_len, output_patch_len, num_layers, model_dims: model internals (defaults from HF card)
    #    Checkpoint wrapper tells it which HF model weights to load.
    # TimesFM v1 (200M) — same API, smaller context (≤512)
    effective_context_len = min(int(context_len), 512)  # v1 cap
    model = TimesFm(
        hparams=TimesFmHparams(
            backend=(
                "gpu"
                if (device_backend == "gpu" and torch.cuda.is_available())
                else "cpu"
            ),
            per_core_batch_size=int(per_core_batch_size),
            horizon_len=int(horizon_len),  # keep your horizon (e.g., 28)
            context_len=effective_context_len,  # clamp for v1
            # v1-style architecture (public card)
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            use_positional_embedding=False,
        ),
        checkpoint=TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"  # <- v1 weights
        ),
    )
    hp = getattr(model, "hparams", None)
    if hp is not None:
        print(
            f"[Check] num_layers={getattr(hp,'num_layers','?')}, context_len={getattr(hp,'context_len','?')}"
        )

    # The HF card demonstrates this exact initialization and confirms DataFrame format and usage. :contentReference[oaicite:4]{index=4}

    # 4) Rolling windows over the test range
    #    At each 'start_idx', we:
    #      - build per-series context arrays of length <= context_len
    #      - call model.forecast(inputs=[...], freq=[...]) to get H-step forecasts for all series
    #      - place forecasts into preds_test for the test times covered by this window
    #    Notes:
    #      - TimesFM also offers forecast_on_df(DataFrame, freq="daily"); but for rolling slicing,
    #        arrays give us very explicit control per window (also shown on the HF card). :contentReference[oaicite:5]{index=5}
    freq_code_daily = (
        0  # 0 = high-frequency (daily/hourly); see model card usage.  # type: int
    )

    # Build list of rolling window start indices *before* referencing it.
    window_starts: List[int] = list(range(time_test_start_idx, times_total, stride_len))
    print(
        f"[Info] #series={len(series_order)}, #windows={len(window_starts)}, "
        f"context={context_len}, horizon={horizon_len}, stride={stride_len}",
        flush=True,
    )

    for start_idx in tqdm(window_starts, desc="Rolling windows", leave=True):
        # clip horizon on the last window
        window_h = min(horizon_len, times_total - start_idx)
        if window_h <= 0:
            break

        # Build inputs for this window
        window_inputs: List[np.ndarray] = []
        window_freqs: List[int] = []
        ctx_len_effective = 365  # for fair comparison with your other models

        for uid in series_order:
            full_y = series_dict[uid]  # np.ndarray [T], float32
            ctx_start = max(0, start_idx - ctx_len_effective)
            ctx_end = start_idx
            # last 365 *real* points
            context_slice = full_y[ctx_start:ctx_end].astype(np.float32)
            # pad to multiple of 32 so the decoder can view(..., 32)
            context_slice = left_pad_to_multiple(context_slice, base=32)  # 365 -> 384
            window_inputs.append(context_slice)
            window_freqs.append(0)  # 0 = high frequency (<= daily) per model card

        # (optional) debug after we HAVE inputs
        # print(f"[debug] first window context len = {len(window_inputs[0])}", flush=True)

        # Batch series for a single forecast() call
        num_series = len(series_order)
        batch_size = int(series_batch_size)
        window_point_forecast_list: List[np.ndarray] = [None] * num_series

        for b_start in tqdm(
            range(0, num_series, batch_size), desc="Series batches", leave=False
        ):
            b_end = min(b_start + batch_size, num_series)
            inputs_batch = window_inputs[b_start:b_end]
            freqs_batch = window_freqs[b_start:b_end]
            point_forecast_array, _ = model.forecast(
                inputs=inputs_batch, freq=freqs_batch
            )
            # scatter back to full list
            for local_idx, global_idx in enumerate(range(b_start, b_end)):
                window_point_forecast_list[global_idx] = point_forecast_array[local_idx]

        # safety check
        assert all(
            pf is not None for pf in window_point_forecast_list
        ), "Missing forecasts."

        # write into [T_test, I, S]
        for uid_index, uid in enumerate(series_order):
            i_idx, j_idx = parse_uid(uid)
            pf = window_point_forecast_list[uid_index]  # shape [window_h]
            for h in range(window_h):
                global_t = start_idx + h
                test_rel_t = global_t - time_test_start_idx
                preds_test[test_rel_t, i_idx, j_idx] += float(pf[h])
                counts[test_rel_t, i_idx, j_idx] += 1.0

    return preds_test, counts


# ----------------------------
# Main
# ----------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", type=int, default=1, help="Use CUDA if available (1=yes, 0=no)"
    )
    parser.add_argument(
        "--context_len", type=int, default=512, help="TimesFM context length"
    )
    parser.add_argument(
        "--horizon_len", type=int, default=28, help="Forecast horizon per window"
    )
    parser.add_argument(
        "--stride_len", type=int, default=28, help="Stride between windows"
    )
    parser.add_argument(
        "--start_date", type=str, default="2011-01-29", help="Start date for 'ds'"
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default="examples/baseline/walmart_timesfm_v1_baseline_dataset.csv",
        help="Output CSV",
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        default="examples/baseline/walmart_timesfm_v1_baseline_results.csv",
        help="Metrics CSV",
    )
    parser.add_argument(
        "--save_df_format",
        type=str,
        default="none",
        choices=["none", "csv", "parquet"],
        help="If 'csv' or 'parquet', write the long dataset; otherwise skip.",
    )
    parser.add_argument(
        "--per_core_batch_size",
        type=int,
        default=32,
        help="TimesFM micro-batch size per device core (inference).",
    )
    parser.add_argument(
        "--series_batch_size",
        type=int,
        default=131072,
        help="How many series to forecast at once per window.",
    )

    args = parser.parse_args()

    device_backend = "gpu" if (args.cuda and torch.cuda.is_available()) else "cpu"
    print(f"Using TimesFM backend: {device_backend}")
    if device_backend == "gpu" and torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}", flush=True)

    # ---- Load tensors (we only NEED T1 for y) ----
    def safe_load_tensor(path: str) -> torch.Tensor:
        if not os.path.exists(path):
            print(f"[Error] Missing file: {path}")
            sys.exit(1)
        return torch.load(path, map_location="cpu").float()

    tensor_T1 = safe_load_tensor("./tensor_1_units.pt")  # [T, I, S] sales units
    # tensor_T2 = safe_load_tensor("./tensor_2_price.pt")  # [T, I, S] price       (unused in baseline)
    # tensor_T3 = safe_load_tensor("./tensor_3_event.pt")  # [T, F]   events      (unused in baseline)

    times_total, number_items, number_stores = tensor_T1.shape
    time_test_start_idx = math.ceil(0.8 * times_total)  # last 20% as test

    # ---- Build the TimesFM DataFrame (unique_id, ds, y) and save CSV ----
    dataframe_all = build_timesfm_dataframe_from_T1(tensor_T1, args.start_date)
    if args.save_df_format != "none":
        if args.save_df_format == "csv":
            dataframe_all.to_csv(args.save_csv, index=False)
            print(f"[OK] Saved dataset CSV: {args.save_csv}")
        else:
            parquet_path = os.path.splitext(args.save_csv)[0] + ".parquet"
            dataframe_all.to_parquet(parquet_path, index=False)
            print(f"[OK] Saved dataset Parquet: {parquet_path}")
    else:
        print("[OK] Skipping file write (using in-memory DataFrame).")

    # ---- Run rolling zero-shot forecasts with TimesFM (no training) ----
    preds_test, counts = rolling_timesfm_zero_shot(
        dataframe_all=dataframe_all,
        horizon_len=int(args.horizon_len),
        context_len=int(args.context_len),
        stride_len=int(args.stride_len),
        times_total=times_total,
        time_test_start_idx=time_test_start_idx,
        number_items=number_items,
        number_stores=number_stores,
        per_core_batch_size=int(args.per_core_batch_size),  # NEW
        series_batch_size=int(args.series_batch_size),  # NEW
        device_backend=device_backend,
    )

    # ---- Average overlapping predictions (if any) ----
    counts_clamped = torch.clamp(counts, min=1.0)
    preds_test_avg = preds_test / counts_clamped  # [T_test, I, S], Float32

    # ---- Gather ground truth for test region ----
    gt_test = tensor_T1[time_test_start_idx:, :, :].to(
        dtype=torch.float32
    )  # [T_test, I, S]

    # ---- Compute metrics ----
    mse = torch.mean((preds_test_avg - gt_test) ** 2)
    rmse = rmse_torch(gt_test, preds_test_avg)
    r2 = r2_torch(gt_test, preds_test_avg)
    r2k = r2_kelly_torch(gt_test, preds_test_avg)

    # Build flat indices [N,3] so WRMSSE matches series grouping
    T_test_len = gt_test.shape[0]
    I = gt_test.shape[1]
    S = gt_test.shape[2]
    t_idx = torch.arange(T_test_len).repeat_interleave(I * S)
    i_idx = torch.arange(I).repeat(T_test_len * S)
    s_idx = torch.arange(S).repeat(T_test_len * I)
    indices_3d = torch.stack([t_idx, i_idx, s_idx], dim=1).long()  # [N, 3]
    wrmsse = wrmsse_torch(
        indices_3d,
        gt_test.flatten(),
        preds_test_avg.flatten(),
        tensor_T1[:time_test_start_idx],
    )

    print("\n=== TimesFM Baseline (Zero-shot, Rolling Windows) ===")
    print(
        f"T_total={times_total}, test_start={time_test_start_idx}, test_len={T_test_len}"
    )
    print(
        f"Context={args.context_len}, Horizon={args.horizon_len}, Stride={args.stride_len}"
    )
    print(f"RMSE     : {float(rmse):.6f}")
    print(f"R^2      : {float(r2):.6f}")
    print(f"Kelly R^2: {float(r2k):.6f}")
    print(f"WRMSSE   : {float(wrmsse):.6f}")

    # Save metrics
    pd.DataFrame(
        [
            {
                "RMSE": float(rmse),
                "R2": float(r2),
                "Kelly_R2": float(r2k),
                "WRMSSE": float(wrmsse),
                "T_total": times_total,
                "test_start_idx": time_test_start_idx,
                "test_len": T_test_len,
                "context_len": int(args.context_len),
                "horizon_len": int(args.horizon_len),
                "stride_len": int(args.stride_len),
            }
        ]
    ).to_csv(args.results_csv, index=False)
    print(f"[OK] Metrics saved to: {args.results_csv}")


if __name__ == "__main__":
    main()


# tell me what this model (https://huggingface.co/google/timesfm-2.0-500m-pytorch) do in the program above?
