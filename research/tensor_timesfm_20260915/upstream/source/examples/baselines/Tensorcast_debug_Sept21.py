#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rolling CP + Holt/Winters one-step EPS forecasting on a 3D tensor (time x firms x features).

Inputs
------
- --tensor_path: path to a NumPy .npy file with shape (T, N, D)
- --eps_index: index of the EPS/target feature (default -1 == last column)

Method
------
For each cutoff t over the last 20% of time:
  1) Fit CP on X[:t+1]  (TensorLy parafac)
  2) Forecast each time-factor column one step ahead (statsmodels ExponentialSmoothing)
  3) Reconstruct EPS at (t+1) for all firms WITHOUT dense (time,firm,feature) reconstruction

Outputs
-------
- CSV with columns: t_pred_for, firm, y_true, y_pred
- Printed metrics: RMSE, standard R^2, Kelly R^2 (torch)

Usage
-----
python eps_cp_holt.py \
  --tensor_path ./your_tensor.npy \
  --rank 64 \
  --eps_index -1 \
  --start_ratio 0.80 \
  --n_iter_max 150 \
  --trend add \
  --seasonal None \
  --output_csv eps_preds.csv
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorly as tl
from tensorly.decomposition import parafac
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import torch


def _first_n(x, n=3):
    try:
        return [x[i] for i in range(min(len(x), n))]
    except Exception:
        return []


def _key(d, options):
    for k in options:
        if k in d:
            return k
    return None


def materialize_to_tensor(obj):
    """
    Try to turn a loaded object into a numeric (T,N,D) ndarray.
    Handles these common cases:
      A) 1-D object array of 2-D arrays -> stack along time axis
      B) 1-D object array of 4-tuples (t, n, d, value) -> densify (COO->dense)
      C) 1-D object array of 3-tuples (t, n, featvec) -> densify
      D) 1-D object array of dicts with keys like time/firm/feature/value -> densify
    Returns ndarray (T,N,D) of dtype object (we’ll coerce to float later).
    Raises ValueError with a helpful summary if unknown.
    """
    # Already a 3-D ndarray? just return
    if isinstance(obj, np.ndarray) and obj.ndim == 3:
        return obj

    # Unwrap 0-D object containers (sometimes .npy saves a single object)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
        obj = obj.item()

    # Case: 1-D object array
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 1:
        if obj.size == 0:
            raise ValueError("Loaded array is empty.")
        e0 = obj[0]

        # A) Sequence of 2-D arrays (same shape) -> stack as time
        if isinstance(e0, np.ndarray) and e0.ndim == 2:
            shapes = [getattr(x, "shape", None) for x in _first_n(obj, 10)]
            if all(
                (isinstance(x, np.ndarray) and x.ndim == 2 and x.shape == e0.shape)
                for x in obj
            ):
                X3 = np.stack(obj, axis=0)  # (T, N, D) guess
                print(
                    f"[materialize] Stacked {len(obj)} slices of shape {e0.shape} -> {X3.shape}"
                )
                return X3

        # B) Sequence of tuples (t, n, d, value)
        if isinstance(e0, (tuple, list)) and len(e0) == 4:
            try:
                T = int(max(int(rec[0]) for rec in obj)) + 1
                N = int(max(int(rec[1]) for rec in obj)) + 1
                D = int(max(int(rec[2]) for rec in obj)) + 1
                X3 = np.full((T, N, D), np.nan, dtype=object)
                for t, n, d, v in obj:
                    X3[int(t), int(n), int(d)] = v
                print(f"[materialize] Densified from 4-tuples to shape {X3.shape}")
                return X3
            except Exception:
                pass

        # C) Sequence of tuples (t, n, featvec)
        if isinstance(e0, (tuple, list)) and len(e0) == 3 and hasattr(e0[2], "__len__"):
            try:
                D = int(len(e0[2]))
                T = int(max(int(rec[0]) for rec in obj)) + 1
                N = int(max(int(rec[1]) for rec in obj)) + 1
                X3 = np.full((T, N, D), np.nan, dtype=object)
                for t, n, vec in obj:
                    vec = np.asarray(vec, dtype=object)
                    X3[int(t), int(n), : min(D, len(vec))] = vec[:D]
                print(f"[materialize] Densified from (t,n,featvec) to shape {X3.shape}")
                return X3
            except Exception:
                pass

        # D) Sequence of dicts with keys ~ {time/firm/feature/value}
        if isinstance(e0, dict):
            tkey = _key(e0, ("t", "time", "T", "i"))
            nkey = _key(e0, ("n", "firm", "j", "idx", "company"))
            dkey = _key(e0, ("d", "f", "feature", "k"))
            vkey = _key(e0, ("v", "value", "val", "x"))
            # (t,n,d,v) dicts
            if tkey and nkey and dkey and vkey:
                T = int(max(int(rec[tkey]) for rec in obj)) + 1
                N = int(max(int(rec[nkey]) for rec in obj)) + 1
                D = int(max(int(rec[dkey]) for rec in obj)) + 1
                X3 = np.full((T, N, D), np.nan, dtype=object)
                for rec in obj:
                    X3[int(rec[tkey]), int(rec[nkey]), int(rec[dkey])] = rec[vkey]
                print(f"[materialize] Densified from dict (t,n,d,v) -> {X3.shape}")
                return X3

    raise ValueError(
        f"Don’t know how to materialize loaded object: type={type(obj)}, "
        f"ndim={getattr(obj,'ndim',None)}, dtype={getattr(obj,'dtype',None)}, "
        f"example0={type(_first_n(obj,1)[0]) if hasattr(obj,'__len__') and len(obj)>0 else type(obj)}"
    )


def _is_numeric_array(a: np.ndarray) -> bool:
    """Return True if we can safely view/cast a as float64."""
    try:
        # Quick path: already numeric
        if np.issubdtype(a.dtype, np.number):
            return True
        # Try casting a small sample first to avoid huge failures
        # (sample ~min(10000 elements) for speed; fall back to full cast)
        flat = a.ravel()
        k = min(flat.size, 10000)
        if k > 0:
            _ = flat[:k].astype(np.float64)
        # Full cast check
        _ = a.astype(np.float64)
        return True
    except Exception:
        return False


def coerce_numeric_features(X: np.ndarray, eps_index: int):
    """
    Take X with shape (T,N,D) possibly object+strings.
    Returns:
      X_num: (T,N,D_keep) float64
      new_eps_index: int (position of original eps_index after dropping cols)
      kept_cols: list[int] original feature indices kept
      dropped_cols: list[int] original feature indices dropped (non-numeric)
    """
    if X.ndim != 3:
        raise ValueError(
            f"Expected 3D array; got {X.ndim}D with shape {getattr(X, 'shape', None)}"
        )

    T, N, D = X.shape
    # Resolve possibly-negative eps index to absolute
    orig_eps = eps_index if eps_index >= 0 else D + eps_index
    if orig_eps < 0 or orig_eps >= D:
        raise IndexError(f"eps_index out of range for D={D}: got {eps_index}")

    kept_cols, dropped_cols = [], []
    numeric_slices = []

    for d in range(D):
        slice_d = X[:, :, d]
        if _is_numeric_array(slice_d):
            numeric_slices.append(slice_d.astype(np.float64)[..., None])
            kept_cols.append(d)
        else:
            dropped_cols.append(d)

    if not kept_cols:
        raise ValueError(
            "No numeric feature columns found. Your tensor’s last axis seems entirely non-numeric."
        )

    X_num = np.concatenate(numeric_slices, axis=2)  # (T,N,D_keep)

    if orig_eps not in kept_cols:
        raise ValueError(
            f"Your EPS column (feature index {orig_eps}) is not numeric or got dropped. "
            f"Confirm EPS is actually numeric in your source tensor."
        )
    new_eps_index = kept_cols.index(orig_eps)

    return X_num, new_eps_index, kept_cols, dropped_cols


@dataclass
class ForecastConfig:
    rank: int = 32
    start_ratio: float = 0.80  # start rolling at ~80% of time
    eps_index: int = -1  # EPS feature index
    n_iter_max: int = 100  # CP-ALS iterations
    random_state: int = 0
    seasonal_periods: Optional[int] = None  # e.g., 4 for quarterly
    trend: Optional[str] = "add"  # 'add' | 'mul' | None
    seasonal: Optional[str] = None  # 'add' | 'mul' | None
    verbose_cp: bool = False
    mmap: bool = False  # np.load(..., mmap_mode='r') if True


def check_tensor(X: np.ndarray) -> Tuple[int, int, int]:
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    if X.ndim != 3:
        raise ValueError(f"X must be 3D (T, N, D); got ndim={X.ndim}.")
    T, N, D = X.shape
    if T < 3 or N < 2 or D < 2:
        raise ValueError(f"X too small: got {X.shape}, need (T>=3, N>=2, D>=2).")
    return T, N, D


def cp_fit_time_firm_feat(
    X_hist: np.ndarray,
    rank: int,
    n_iter_max: int,
    random_state: int,
    verbose: bool = False,
):
    """CP-ALS on (time, firm, feature) -> (weights, U_time, U_firm, U_feat)."""
    tl.set_backend("numpy")  # use NumPy backend for robustness
    weights, factors = parafac(
        X_hist,
        rank=rank,
        n_iter_max=n_iter_max,
        init="svd",
        random_state=random_state,
        normalize_factors=False,
        return_errors=False,
        verbose=1 if verbose else 0,
    )
    U_time, U_firm, U_feat = factors  # aligned with (time, firm, feature)
    return np.asarray(weights).reshape(-1), U_time, U_firm, U_feat


def forecast_next(
    series: np.ndarray,
    trend: Optional[str],
    seasonal: Optional[str],
    seasonal_periods: Optional[int],
) -> float:
    """One-step Holt/Winters forecast with sane fallbacks."""
    ser = np.asarray(series, dtype=float)
    if not np.all(np.isfinite(ser)):
        mask = np.isfinite(ser)
        if not mask.any():
            return float("nan")
        ser = ser.copy()
        ser[~mask] = ser[mask][-1]
    try:
        model = ExponentialSmoothing(
            ser,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)
        return float(fit.forecast(1)[0])
    except Exception:
        return float(ser[-1])


def rolling_eps_forecast(X: np.ndarray, cfg: ForecastConfig) -> pd.DataFrame:
    """Return DataFrame[t_pred_for, firm, y_true, y_pred] over rolling windows."""
    T, N, D = check_tensor(X)
    eps_idx = cfg.eps_index if cfg.eps_index >= 0 else D + cfg.eps_index
    if eps_idx < 0 or eps_idx >= D:
        raise IndexError(f"eps_index out of range for D={D}: got {cfg.eps_index}")

    start_t = max(1, int(math.floor(cfg.start_ratio * T)) - 1)
    rows = []

    for t in tqdm(range(start_t, T - 1), desc="Rolling", unit="win"):
        X_hist = X[: t + 1, :, :]  # (t+1, N, D)

        # 1) CP on history
        weights, U_time, U_firm, U_feat = cp_fit_time_firm_feat(
            X_hist,
            rank=cfg.rank,
            n_iter_max=cfg.n_iter_max,
            random_state=cfg.random_state,
            verbose=cfg.verbose_cp,
        )

        # 2) Forecast each time-factor one step
        R = weights.shape[0]
        u_next = np.empty(R, dtype=float)
        for r in range(R):
            u_next[r] = forecast_next(
                U_time[:, r],
                trend=cfg.trend,
                seasonal=cfg.seasonal,
                seasonal_periods=cfg.seasonal_periods,
            )

        # 3) Reconstruct EPS for all firms at t+1
        # y_hat = U_firm @ (weights * u_next * U_feat[eps_idx, :])
        w = weights * u_next * U_feat[eps_idx, :]  # (R,)
        y_hat = U_firm @ w  # (N,)

        y_true = X[t + 1, :, eps_idx]  # (N,)
        for i in range(N):
            rows.append((t + 1, i, float(y_true[i]), float(y_hat[i])))

    return pd.DataFrame(rows, columns=["t_pred_for", "firm", "y_true", "y_pred"])


def r2_standard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def r2_kelly_torch(gt: np.ndarray, pred: np.ndarray) -> float:
    """Kelly R^2 using *exactly* the torch form you provided."""
    gt_t = torch.as_tensor(gt, dtype=torch.float32)
    pr_t = torch.as_tensor(pred, dtype=torch.float32)
    mask = torch.isfinite(gt_t) & torch.isfinite(pr_t)
    gt_t = gt_t[mask]
    pr_t = pr_t[mask]
    if gt_t.numel() == 0:
        return float("nan")
    ss_res = torch.sum((gt_t - pr_t) ** 2)
    ss_tot = torch.sum(gt_t**2)
    return (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0


def compute_metrics(df: pd.DataFrame) -> dict:
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    r2 = r2_standard(y_true, y_pred)
    r2_k = r2_kelly_torch(y_true, y_pred)
    return {"rmse": rmse, "r2": r2, "r2_kelly": r2_k}


def main():
    ap = argparse.ArgumentParser(
        description="Rolling CP + Holt/Winters EPS one-step forecasting."
    )
    ap.add_argument(
        "--tensor_path", type=str, required=True, help="Path to .npy file (T,N,D)."
    )
    ap.add_argument("--rank", type=int, default=32, help="CP rank.")
    ap.add_argument(
        "--eps_index", type=int, default=-1, help="EPS feature index (default -1)."
    )
    ap.add_argument(
        "--start_ratio",
        type=float,
        default=0.80,
        help="Start rolling at this fraction of T.",
    )
    ap.add_argument(
        "--n_iter_max", type=int, default=100, help="Max CP-ALS iterations."
    )
    ap.add_argument(
        "--seasonal_periods",
        type=int,
        default=None,
        help="Season length (e.g., 4 for quarterly).",
    )
    ap.add_argument(
        "--trend",
        type=str,
        default="add",
        choices=["add", "mul", "None"],
        help="Trend component.",
    )
    ap.add_argument(
        "--seasonal",
        type=str,
        default="None",
        choices=["add", "mul", "None"],
        help="Seasonal component.",
    )
    ap.add_argument(
        "--random_state", type=int, default=0, help="Random seed for CP init."
    )
    ap.add_argument(
        "--output_csv",
        type=str,
        default="eps_predictions.csv",
        help="Save predictions here.",
    )
    ap.add_argument(
        "--verbose_cp", action="store_true", help="Print CP-ALS progress per window."
    )
    ap.add_argument("--mmap", action="store_true", help="Load .npy with mmap_mode='r'.")

    args = ap.parse_args()

    load_kwargs = {"mmap_mode": "r"} if args.mmap else {}
    raw = np.load(args.tensor_path, allow_pickle=True, **load_kwargs)
    X_obj = raw

    # Try to turn whatever we loaded into a (T,N,**D_all**) tensor
    X_obj = materialize_to_tensor(X_obj)  # returns dtype=object if any non-numerics
    print(f"[materialize] result shape={X_obj.shape}, dtype={X_obj.dtype}")

    # Keep only numeric features (drop strings like ticker), and remap EPS index
    X, mapped_eps_index, kept_cols, dropped_cols = coerce_numeric_features(
        X_obj, args.eps_index
    )
    print(
        f"[preprocess] kept {len(kept_cols)} numeric feature(s), dropped {len(dropped_cols)} non-numeric: {dropped_cols}"
    )
    print(
        f"[preprocess] EPS feature {args.eps_index} -> new index {mapped_eps_index} on cleaned feature axis"
    )
    args.eps_index = mapped_eps_index

    cfg = ForecastConfig(
        rank=args.rank,
        start_ratio=args.start_ratio,
        eps_index=args.eps_index,
        n_iter_max=args.n_iter_max,
        random_state=args.random_state,
        seasonal_periods=(
            args.seasonal_periods
            if args.seasonal_periods and args.seasonal_periods > 1
            else None
        ),
        trend=None if args.trend == "None" else args.trend,
        seasonal=None if args.seasonal == "None" else args.seasonal,
        verbose_cp=args.verbose_cp,
        mmap=args.mmap,
    )

    df = rolling_eps_forecast(X, cfg)
    metrics = compute_metrics(df)
    print(
        f"[Summary] rows={len(df)}  RMSE={metrics['rmse']:.6f}  "
        f"R^2={metrics['r2']:.6f}  R^2_Kelly={metrics['r2_kelly']:.6f}"
    )
    df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
