#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TimesFM-only baseline for EPS target (feature index 13) with growing-context windows.

What it does:
- Load EPS tensor .npy with shape [T, FIRM, FEATURE].
- Extract target feature at index 13 -> y_all of shape [T, FIRM].
- Split last 20% as test.
- For each test time p, feed context [0..p-1] to TimesFM (zero-shot), horizon_len=1.
- Compute RMSE / R^2 / Kelly R^2.
- Save metrics CSV.
- NEW: Save final table CSV with columns [t, firm, Ticker, GroundTruth, Forecasted].

TimesFM notes:
- Univariate; list-of-arrays API with a frequency code in {0,1,2}.  # HF README
- Context up to ~2048 trained; inference handles longer/shorter by trunc/pad.     # HF README (JAX notes)
"""

import argparse
import math
import os
import sys
from typing import List

import numpy as np  # type: module; arrays
import pandas as pd  # type: module; CSV tables
import torch  # type: module; metrics math
from tqdm.auto import tqdm  # type: module; progress bars

# TimesFM v1/v2 API (pip install timesfm==1.3.0 recommended)
import timesfm
from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint

torch.set_float32_matmul_precision("high")  # (no shape) speed hint for matmuls


# ----------------------------
# Metrics (vanilla, torch)
# ----------------------------
def rmse_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # y_true/y_pred: FloatTensor same shape
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


def r2_torch(
    y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    # Standard R^2 with clamp on denominator
    ss_res = torch.sum((y_true - y_pred) ** 2)
    mean_y = torch.mean(y_true)
    ss_tot = torch.sum((y_true - mean_y) ** 2)
    ss_tot = torch.clamp(ss_tot, min=eps)  # avoid divide-by-zero
    return 1.0 - ss_res / ss_tot


def r2_kelly_torch(
    y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    # Kelly R^2 = 1 - sum((y - yhat)^2) / sum(y^2)
    denom = torch.sum(y_true**2)
    denom = torch.clamp(denom, min=eps)
    return 1.0 - torch.sum((y_true - y_pred) ** 2) / denom


# ----------------------------
# EPS loader
# ----------------------------
def load_eps_target(npy_path: str, target_feature_index: int = 13) -> torch.Tensor:
    """
    npy_path: str; file path to .npy of shape [T, FIRM, FEATURE]
    target_feature_index: int; which column is EPS (default 13)

    return: FloatTensor [T, FIRM]; EPS per firm over time
    """
    if not os.path.exists(npy_path):
        print(f"[Error] Missing file: {npy_path}")
        sys.exit(1)
    arr = np.load(npy_path)  # np.ndarray [T,F,D]
    if arr.ndim != 3:
        print(f"[Error] Expected 3D array [T,FIRM,FEATURE], got {arr.shape}")
        sys.exit(1)
    T, F, D = arr.shape
    if not (0 <= target_feature_index < D):
        print(
            f"[Error] target_feature_index {target_feature_index} out of range [0, {D-1}]"
        )
        sys.exit(1)
    y_np = arr[:, :, target_feature_index].astype(np.float32)  # np.ndarray [T,F]
    return torch.from_numpy(y_np)  # FloatTensor [T,F]


def sample_stochastic_one_step_from_full_forecast(
    full_forecast_h1: np.ndarray,  # np.ndarray [1 + Q] or compatible; index 0 is mean
    quantile_levels: np.ndarray,  # np.ndarray [Q], e.g., [0.1..0.9]
    rng: np.random.Generator,  # RNG for reproducible stochastic draws
    num_samples: int,  # number of MC draws per forecast
) -> float:
    """
    Draw stochastic one-step predictions from TimesFM's quantile output via
    inverse-CDF interpolation, then average those draws.

    TimesFM full forecast layout (v1 API): [mean, q1, q2, ...].
    """
    full_arr = np.asarray(full_forecast_h1, dtype=np.float32).reshape(-1)
    if full_arr.size == 0:
        return float("nan")
    mean_fallback = (
        float(full_arr[0]) if np.isfinite(full_arr[0]) else float("nan")
    )  # scalar
    if full_arr.size < 2:
        return mean_fallback

    q_levels = np.asarray(quantile_levels, dtype=np.float32).reshape(-1)  # [Q]
    q_values = full_arr[1:]  # [Q']
    n_match = int(min(q_levels.size, q_values.size))
    if n_match <= 0:
        return mean_fallback
    q_levels = q_levels[:n_match]
    q_values = q_values[:n_match]

    # Keep valid pairs only, sort by quantile level, and enforce monotone values.
    finite_mask = np.isfinite(q_levels) & np.isfinite(q_values)
    if not finite_mask.any():
        return mean_fallback
    q_levels = q_levels[finite_mask]
    q_values = q_values[finite_mask]
    order = np.argsort(q_levels)
    q_levels = q_levels[order]
    q_values = q_values[order]
    q_values = np.maximum.accumulate(q_values)

    # Remove duplicate quantile levels (keep first), required by np.interp.
    q_levels_unique, unique_idx = np.unique(q_levels, return_index=True)
    q_values_unique = q_values[unique_idx]
    if q_levels_unique.size == 0:
        return mean_fallback
    if q_levels_unique.size == 1:
        return float(q_values_unique[0])

    n_draws = max(1, int(num_samples))
    uniforms = rng.random(n_draws)  # [n_draws], Uniform(0,1)
    draws = np.interp(
        uniforms,
        q_levels_unique,
        q_values_unique,
        left=float(q_values_unique[0]),
        right=float(q_values_unique[-1]),
    ).astype(np.float32)
    return float(draws.mean())


# ----------------------------
# Rolling zero-shot (growing context)
# ----------------------------
def rolling_timesfm_zero_shot_eps(
    y_all: torch.Tensor,  # FloatTensor [T,F]; target per firm
    time_test_start_idx: int,  # int; start of test
    freq_code: int,  # int in {0,1,2}; TimesFM freq category
    per_core_batch_size: int,  # int; model micro-batch
    series_batch_size: int,  # int; how many series to forecast at once
    context_len: int,  # int; model context cap (<=2048 typical)
    device_backend: str = "gpu",  # str; "gpu" or "cpu"
    use_stochastic_prediction: bool = True,  # bool; sample from quantile output
    stochastic_num_samples: int = 1,  # int; MC draws per series/time
    stochastic_seed: int = 0,  # int; RNG seed for reproducible stochasticity
) -> torch.Tensor:
    """
    return: FloatTensor [T_test, F]; predictions aligned to test times
    """
    T_total, F_total = y_all.shape  # ints
    T_test = T_total - time_test_start_idx  # int
    preds_test = torch.zeros((T_test, F_total), dtype=torch.float32)  # [T_test,F]

    # Build TimesFM once; reuse across windows
    model = TimesFm(
        hparams=TimesFmHparams(
            backend=(
                "gpu"
                if (device_backend == "gpu" and torch.cuda.is_available())
                else "cpu"
            ),
            per_core_batch_size=int(per_core_batch_size),
            horizon_len=1,  # 1-step ahead
            context_len=int(
                context_len
            ),  # cap; inference will handle trunc/pad if needed
            # fixed for 2.0 500M checkpoint:
            input_patch_len=32,
            output_patch_len=128,
            num_layers=50,
            model_dims=1280,
            use_positional_embedding=False,
        ),
        checkpoint=TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
        ),
    )

    quantile_levels = np.asarray(
        getattr(getattr(model, "hparams", None), "quantiles", []), dtype=np.float32
    )  # [Q] or empty
    rng = np.random.default_rng(int(stochastic_seed))  # deterministic RNG
    warned_quantile_fallback = (
        False  # one-time warning when stochastic path cannot be used
    )

    print(
        "[Info] "
        f"#series={F_total}, test_len={T_test}, context_mode=growing, horizon=1, "
        f"stochastic={int(use_stochastic_prediction)}, "
        f"num_samples={int(stochastic_num_samples)}, seed={int(stochastic_seed)}",
        flush=True,
    )
    if use_stochastic_prediction and quantile_levels.size == 0:
        print(
            "[WARN] TimesFM quantile levels are unavailable; using deterministic point forecasts."
        )

    # For each test time p (absolute), feed history [0..p-1], predict value at p
    for p in tqdm(
        range(time_test_start_idx, T_total), desc="Rolling windows (growing context)"
    ):
        inputs_for_p: List[np.ndarray] = (
            []
        )  # list of np.ndarray; each one 1-D float32 history
        freqs_for_p: List[int] = (
            []
        )  # list[int]; freq code per series (same for all here)

        # Build per-firm histories; TimesFM handles truncation/padding internally if longer/shorter.
        for f in range(F_total):
            # np.ndarray [p] float32; history up to p-1
            series_np = y_all[:p, f].detach().cpu().numpy().astype(np.float32)
            inputs_for_p.append(series_np)
            freqs_for_p.append(int(freq_code))

        # Batch across firms to limit memory
        write_row = p - time_test_start_idx  # int; row index inside preds_test
        for s0 in range(0, F_total, int(series_batch_size)):
            s1 = min(s0 + int(series_batch_size), F_total)
            batch_inputs = inputs_for_p[s0:s1]  # list length (s1-s0), each [p]
            batch_freqs = freqs_for_p[s0:s1]  # same length

            # TimesFM forecast: returns (point_forecast, experimental_quantiles)
            point_forecast, quantile_forecast = model.forecast(
                inputs=batch_inputs, freq=batch_freqs
            )
            # point_forecast: array/list shape [(s1-s0), 1]
            point_forecast_np = np.asarray(point_forecast, dtype=np.float32).reshape(-1)
            quantile_forecast_np = np.asarray(quantile_forecast)
            use_quantile_sampling_this_batch = (
                use_stochastic_prediction
                and quantile_levels.size > 0
                and quantile_forecast_np.ndim == 3
                and quantile_forecast_np.shape[1] >= 1
                and quantile_forecast_np.shape[2] >= 2
            )
            if (
                use_stochastic_prediction
                and (not use_quantile_sampling_this_batch)
                and (not warned_quantile_fallback)
            ):
                print(
                    "[WARN] Quantile forecast output not available in expected shape; "
                    "falling back to deterministic point forecasts."
                )
                warned_quantile_fallback = True

            for local_idx, firm_idx in enumerate(range(s0, s1)):
                pred_value = float(point_forecast_np[local_idx])  # deterministic fallback
                if use_quantile_sampling_this_batch:
                    pred_value = sample_stochastic_one_step_from_full_forecast(
                        full_forecast_h1=quantile_forecast_np[local_idx, 0, :],
                        quantile_levels=quantile_levels,
                        rng=rng,
                        num_samples=int(stochastic_num_samples),
                    )
                preds_test[write_row, firm_idx] = pred_value

    return preds_test  # [T_test,F]


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eps_path",
        type=str,
        default="./fundamentals_analyst_forecast_EPS_regression.npy",
        help="Path to EPS npy with shape [T, FIRM, FEATURE].",
    )
    ap.add_argument(
        "--target_feature_index",
        type=int,
        default=13,
        help="Feature index used as target y (default 13).",
    )
    ap.add_argument(
        "--cuda", type=int, default=1, help="Use CUDA if available (1=yes, 0=no)."
    )
    ap.add_argument(
        "--freq_code",
        type=int,
        default=2,
        help="TimesFM frequency: 0=high(≤daily), 1=weekly/monthly, 2=quarterly/yearly.",
    )
    ap.add_argument(
        "--context_len",
        type=int,
        default=4096,
        help="Model max context cap. Inference will truncate/pad as needed.",
    )
    ap.add_argument(
        "--per_core_batch_size",
        type=int,
        default=32,
        help="TimesFM micro-batch per device core.",
    )
    ap.add_argument(
        "--series_batch_size",
        type=int,
        default=131072,
        help="How many firms to forecast at once per window.",
    )
    ap.add_argument(
        "--timesfm_stochastic",
        type=int,
        default=1,
        help="Use stochastic one-step prediction via quantile sampling (1=yes, 0=no).",
    )
    ap.add_argument(
        "--timesfm_num_samples",
        type=int,
        default=1,
        help="Monte Carlo draws per one-step forecast when stochastic mode is on.",
    )
    ap.add_argument(
        "--timesfm_seed",
        type=int,
        default=0,
        help="Random seed for TimesFM stochastic sampling.",
    )
    ap.add_argument(
        "--results_csv",
        type=str,
        default="examples/ablation_studies/EPS/eps_timesfm_metrics.csv",
        help="Where to save metrics CSV.",
    )
    # NEW ↓↓↓
    ap.add_argument(
        "--tickers_path",
        type=str,
        default="./fundamentals_analyst_forecast_EPS_regression_ticker.npy",
        help="Optional .npy shape [FIRM] of ticker strings; if empty, auto-generate 0000..",
    )
    ap.add_argument(
        "--final_csv",
        type=str,
        default="examples/ablation_studies/EPS/eps_timesfm_final_table.csv",
        help="Where to save the final table (t, firm, Ticker, GroundTruth, Forecasted).",
    )
    # NEW ↑↑↑
    args = ap.parse_args()

    device_backend = "gpu" if (args.cuda and torch.cuda.is_available()) else "cpu"
    print(f"Using TimesFM backend: {device_backend}")
    if device_backend == "gpu" and torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}", flush=True)

    # Load target matrix y_all [T,F]
    y_all = load_eps_target(args.eps_path, args.target_feature_index).to(
        dtype=torch.float32
    )  # [T,F]
    T_total, F_total = y_all.shape

    # Load tickers sidecar or fallback (NEW)
    if args.tickers_path and os.path.exists(args.tickers_path):
        tickers_array = np.load(args.tickers_path, allow_pickle=True).astype(
            str
        )  # np.ndarray [FIRM]; firm→ticker
        if tickers_array.ndim != 1 or tickers_array.shape[0] != F_total:
            sys.exit(
                f"Ticker sidecar shape mismatch: got {tickers_array.shape}, expected ({F_total},)"
            )
    else:
        # fallback labels: "0000","0001",...
        tickers_array = np.array([f"{i:04d}" for i in range(F_total)], dtype=object)

    # Split 80/20 in time
    time_test_start_idx = math.ceil(0.8 * T_total)  # int; first test index
    T_test = T_total - time_test_start_idx  # int; length of test span

    # Forecast using growing context
    preds_test = rolling_timesfm_zero_shot_eps(
        y_all=y_all,
        time_test_start_idx=time_test_start_idx,
        freq_code=int(args.freq_code),
        per_core_batch_size=int(args.per_core_batch_size),
        series_batch_size=int(args.series_batch_size),
        context_len=int(args.context_len),
        device_backend=device_backend,
        use_stochastic_prediction=bool(args.timesfm_stochastic),
        stochastic_num_samples=int(args.timesfm_num_samples),
        stochastic_seed=int(args.timesfm_seed),
    )  # FloatTensor [T_test,F]

    # Ground truth for test span
    gt_test = y_all[time_test_start_idx:, :]  # FloatTensor [T_test,F]
    # ---- Align predictions with labels (evaluation mask) ----
    # gt_test: FloatTensor [T_test, FIRM]
    # preds_test: FloatTensor [T_test, FIRM]
    # torch.isnan → BoolTensor mask of same shape as input (docs). :contentReference[oaicite:0]{index=0}
    nan_mask = torch.isnan(gt_test)  # [T_test, FIRM] bool; True where GT missing
    preds_test = preds_test.clone()  # keep original tensor
    preds_test[nan_mask] = float("nan")  # set preds to NaN wherever GT is NaN

    # ---- Masked metrics: only where both GT and Forecast exist ----
    # Build a validity mask on the 2D arrays, then index to 1D vectors.
    valid_mask = (~torch.isnan(gt_test)) & (
        ~torch.isnan(preds_test)
    )  # [T_test, FIRM] bool
    num_valid = int(valid_mask.sum().item())  # scalar int
    print(
        f"[Info] Valid (non-missing) test entries: {num_valid} / {T_test*F_total} total"
    )

    if num_valid == 0:
        # Nothing to score; print a friendly note and emit NaNs
        print("[WARN] No overlapping non-missing GT/Forecast entries in test span.")
        rmse_val = torch.tensor(float("nan"))
        r2_val = torch.tensor(float("nan"))
        r2k_val = torch.tensor(float("nan"))
    else:
        # Indexing with a boolean mask flattens to 1-D of valid elements (row-major).
        y_true = gt_test[valid_mask]  # FloatTensor [num_valid]
        y_pred = preds_test[valid_mask]  # FloatTensor [num_valid]
        rmse_val = rmse_torch(y_true, y_pred)  # scalar FloatTensor
        r2_val = r2_torch(y_true, y_pred)  # scalar FloatTensor
        r2k_val = r2_kelly_torch(y_true, y_pred)  # scalar FloatTensor

    print("\n=== TimesFM EPS Baseline (Zero-shot, Growing Context, Horizon=1) ===")
    print(f"T_total={T_total}, test_start={time_test_start_idx}, test_len={T_test}")
    print(
        f"freq_code={args.freq_code}, context_len={args.context_len}, "
        f"stochastic={int(args.timesfm_stochastic)}, "
        f"num_samples={int(args.timesfm_num_samples)}, seed={int(args.timesfm_seed)}"
    )
    print(f"RMSE     : {float(rmse_val):.6f}")
    print(f"R^2      : {float(r2_val):.6f}")
    print(f"Kelly R^2: {float(r2k_val):.6f}")

    # Save metrics CSV
    pd.DataFrame(
        [
            {
                "RMSE": float(rmse_val),
                "R2": float(r2_val),
                "Kelly_R2": float(r2k_val),
                "T_total": int(T_total),
                "test_start_idx": int(time_test_start_idx),
                "test_len": int(T_test),
                "freq_code": int(args.freq_code),
                "context_len": int(args.context_len),
                "timesfm_stochastic": int(bool(args.timesfm_stochastic)),
                "timesfm_num_samples": int(args.timesfm_num_samples),
                "timesfm_seed": int(args.timesfm_seed),
                "horizon_len": 1,
                "windowing": "growing_context",
            }
        ]
    ).to_csv(args.results_csv, index=False)
    print(f"[OK] Metrics saved to: {args.results_csv}")

    # ----- NEW: Final comparison table with Ticker -----
    # gt_test_np / pred_test_np: np.ndarray [T_test,F] float
    gt_test_np = gt_test.detach().cpu().numpy()
    pred_test_np = preds_test.detach().cpu().numpy()

    # absolute times for test: [time_test_start_idx .. T_total-1]
    absolute_time_index = np.arange(
        time_test_start_idx, T_total, dtype=np.int64
    )  # [T_test]
    # Build columns of length T_test*F
    t_col = np.repeat(absolute_time_index, F_total)  # [T_test*F]
    firm_col = np.tile(np.arange(F_total, dtype=np.int64), T_test)  # [T_test*F]
    ticker_col = np.tile(tickers_array, T_test)  # [T_test*F]
    y_true_col = gt_test_np.reshape(-1).astype(np.float32)  # [T_test*F]
    y_pred_col = pred_test_np.reshape(-1).astype(np.float32)  # [T_test*F]

    final_df = pd.DataFrame(
        {
            "t": t_col,  # int
            "firm": firm_col,  # int
            "Ticker": ticker_col,  # str
            "GroundTruth": y_true_col,  # float
            "Forecasted": y_pred_col,  # float
        }
    )
    final_df = final_df.dropna(subset=["GroundTruth", "Forecasted"])
    final_df.sort_values(["t", "firm"], inplace=True, kind="mergesort")
    final_df.to_csv(args.final_csv, index=False)
    print(f"[OK] Final table saved to: {args.final_csv}")


if __name__ == "__main__":
    main()
