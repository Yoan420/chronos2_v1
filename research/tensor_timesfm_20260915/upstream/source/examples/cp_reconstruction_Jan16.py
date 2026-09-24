# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# CP decomposition (Option B): RAW in-sample reconstruction on first 80% time slice,
# and APPEND results to a CSV file.

# Ground truth:
#     X_train = X[0:T_train, :, :],  T_train = floor(train_ratio * T)

# Reconstruction:
#     Fit CP(parafac) on X_train, reconstruct X_hat_train, then compute R2/MSE on X_train.

# Reproducibility:
#     - np.random.seed(seed)
#     - parafac(..., random_state=seed) (unless you override)

# Output:
#     Append one row per rank into --output_csv
# """

# import argparse
# import csv
# import os
# import time
# import datetime
# from typing import Optional, List, Tuple

# import numpy as np
# import tensorly as tl
# from tensorly.decomposition import parafac


# def r2_from_sse(y_true_flat: np.ndarray, y_pred_flat: np.ndarray) -> float:
#     """R^2 = 1 - SSE/SST on 1D vectors."""
#     residual = y_true_flat - y_pred_flat
#     sum_squared_error = float(np.sum(residual * residual))
#     mean_true = float(np.mean(y_true_flat))
#     total_sum_squares = float(np.sum((y_true_flat - mean_true) ** 2))
#     return 1.0 - (sum_squared_error / total_sum_squares) if total_sum_squares > 0 else float("nan")


# def cp_recon_r2_raw_first_time_slice(
#     tensor_path: str,
#     cp_rank: int,
#     train_ratio: float,
#     n_iter_max: int,
#     tol: float,
#     seed: int,
#     random_state_override: Optional[int],
# ) -> Tuple[float, float, int, Tuple[int, int, int]]:
#     """Returns (r2_raw, mse_raw, train_time_count, train_shape)."""
#     np.random.seed(int(seed))

#     full_raw_tensor = np.load(tensor_path).astype(np.float32)  # (T,N,F)
#     time_count, firm_count, feature_count = full_raw_tensor.shape

#     train_time_count = int(train_ratio * time_count)
#     if train_time_count <= 0:
#         raise ValueError("train_time_count <= 0. Check train_ratio and T.")

#     train_raw_tensor = full_raw_tensor[:train_time_count]  # (T_train,N,F)

#     if np.isnan(train_raw_tensor).any():
#         raise ValueError(
#             "Found NaN in train_raw_tensor. Plain parafac does not handle NaN directly.\n"
#             "For synthetic verification, please generate without NaN."
#         )

#     effective_random_state = int(seed) if random_state_override is None else int(random_state_override)

#     cp_tensor = parafac(
#         train_raw_tensor,
#         rank=int(cp_rank),
#         n_iter_max=int(n_iter_max),
#         init="random",
#         tol=float(tol),
#         random_state=effective_random_state,
#     )

#     reconstructed_train_raw_tensor = tl.cp_to_tensor(cp_tensor)  # (T_train,N,F)

#     y_true = train_raw_tensor.reshape(-1)
#     y_pred = reconstructed_train_raw_tensor.reshape(-1)

#     r2_value = r2_from_sse(y_true, y_pred)
#     mse_value = float(np.mean((y_true - y_pred) ** 2))
#     return r2_value, mse_value, train_time_count, tuple(train_raw_tensor.shape)


# def append_row_to_csv(csv_path: str, header: List[str], row: dict) -> None:
#     os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
#     file_exists = os.path.exists(csv_path)
#     with open(csv_path, "a", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=header)
#         if not file_exists:
#             writer.writeheader()
#         writer.writerow(row)


# def parse_int_list(comma_list: str) -> List[int]:
#     if comma_list is None:
#         return []
#     s = str(comma_list).strip()
#     if s == "":
#         return []
#     return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


# def main() -> None:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--tensor_path", type=str, required=True)

#     # ranks: either one or many
#     parser.add_argument("--cp_rank", type=int, default=None, help="Single CP rank.")
#     parser.add_argument("--cp_ranks", type=str, default="", help="Comma-separated ranks, e.g. 30,50,100.")

#     parser.add_argument("--train_ratio", type=float, default=0.8)
#     parser.add_argument("--n_iter_max", type=int, default=100)
#     parser.add_argument("--tol", type=float, default=1e-6)

#     parser.add_argument("--seed", type=int, default=42, help="Seed for numpy + default parafac random_state.")
#     parser.add_argument(
#         "--random_state",
#         type=int,
#         default=None,
#         help="Optional: override tensorly parafac random_state. If None, uses --seed.",
#     )

#     parser.add_argument("--output_csv", type=str, default="./results/cp_optionB_results.csv")
#     parser.add_argument("--run_tag", type=str, default="")

#     args = parser.parse_args()

#     rank_list: List[int] = parse_int_list(args.cp_ranks)
#     if len(rank_list) == 0:
#         if args.cp_rank is None:
#             raise ValueError("You must provide --cp_rank or --cp_ranks")
#         rank_list = [int(args.cp_rank)]

#     header = [
#         "timestamp_utc",
#         "method",
#         "run_tag",
#         "tensor_path",
#         "train_ratio",
#         "train_T",
#         "train_shape",
#         "cp_rank",
#         "n_iter_max",
#         "tol",
#         "seed",
#         "random_state",
#         "r2_raw",
#         "mse_raw",
#         "elapsed_sec",
#     ]

#     for cp_rank_value in rank_list:
#         start_wall_time = time.time()
#         r2_raw, mse_raw, train_time_count, train_shape = cp_recon_r2_raw_first_time_slice(
#             tensor_path=args.tensor_path,
#             cp_rank=int(cp_rank_value),
#             train_ratio=float(args.train_ratio),
#             n_iter_max=int(args.n_iter_max),
#             tol=float(args.tol),
#             seed=int(args.seed),
#             random_state_override=args.random_state,
#         )
#         elapsed_seconds = time.time() - start_wall_time

#         print("CP RAW first-time-slice recon (Option B, same ground truth):")
#         print(f"  tensor_path   = {args.tensor_path}")
#         print(f"  train_ratio   = {args.train_ratio}")
#         print(f"  train_T       = {train_time_count}")
#         print(f"  train_shape   = {train_shape}")
#         print(f"  cp_rank       = {cp_rank_value}")
#         print(f"  seed          = {args.seed}")
#         print(f"  random_state  = {args.seed if args.random_state is None else args.random_state}")
#         print(f"  R2_raw        = {r2_raw}")
#         print(f"  MSE_raw       = {mse_raw}")
#         print(f"  elapsed_sec   = {elapsed_seconds:.3f}")

#         row = {
#             "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
#             "method": "CP(parafac)",
#             "run_tag": args.run_tag,
#             "tensor_path": args.tensor_path,
#             "train_ratio": float(args.train_ratio),
#             "train_T": int(train_time_count),
#             "train_shape": str(train_shape),
#             "cp_rank": int(cp_rank_value),
#             "n_iter_max": int(args.n_iter_max),
#             "tol": float(args.tol),
#             "seed": int(args.seed),
#             "random_state": int(args.seed if args.random_state is None else args.random_state),
#             "r2_raw": float(r2_raw),
#             "mse_raw": float(mse_raw),
#             "elapsed_sec": float(elapsed_seconds),
#         }
#         append_row_to_csv(args.output_csv, header, row)
#         print(f"[INFO] Appended 1 row to CSV: {args.output_csv}")
#         print("-" * 30)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CP decomposition (Option B): RAW in-sample reconstruction on first time slice,
and APPEND results to a CSV file.

Adds:
- Optional histogram plots for:
  (1) X_train (true) distribution
  (2) X_hat_train (CP reconstruction) distribution
  (3) residual = X_hat_train - X_train distribution
  (4) overlay plot: true vs recon on same axes (optional)

Notes:
- This script assumes NO NaN in the training slice.
- Uses random sampling for histograms (tensor can be large).
"""

import argparse
import csv
import os
import time
import datetime
from typing import Optional, List, Tuple

import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

import matplotlib.pyplot as plt


# ------------------------ metrics ------------------------
def r2_from_sse(y_true_flat: np.ndarray, y_pred_flat: np.ndarray) -> float:
    """
    R^2 = 1 - SSE/SST on 1D vectors.

    y_true_flat: (M,) float
    y_pred_flat: (M,) float
    returns: float
    """
    residual = y_true_flat - y_pred_flat
    sum_squared_error = float(np.sum(residual * residual))
    mean_true = float(np.mean(y_true_flat))
    total_sum_squares = float(np.sum((y_true_flat - mean_true) ** 2))
    return (
        1.0 - (sum_squared_error / total_sum_squares)
        if total_sum_squares > 0
        else float("nan")
    )


# ------------------------ plotting helpers ------------------------
def _sample_finite_flat_values(
    array_any_shape: np.ndarray,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    """
    Flatten any-shaped array and return a random sample of finite values.

    array_any_shape: np.ndarray, any shape
    sample_size: int, number of points to sample
    seed: int, RNG seed

    returns: (K,) float64, K <= sample_size
    """
    flat = array_any_shape.reshape(-1)

    finite_mask = np.isfinite(flat)
    flat_finite = flat[finite_mask]

    if flat_finite.size == 0:
        return flat_finite.astype(np.float64)

    rng = np.random.default_rng(int(seed))
    k = int(min(sample_size, flat_finite.size))
    idx = rng.integers(low=0, high=flat_finite.size, size=k)
    return flat_finite[idx].astype(np.float64)


def _summarize_1d(values_1d: np.ndarray, name: str) -> None:
    """
    Print basic stats for 1D values (assumes finite).

    values_1d: (K,) float
    """
    if values_1d.size == 0:
        print(f"[{name}] empty.")
        return

    mean_value = float(np.mean(values_1d))
    std_value = float(np.std(values_1d))
    p01 = float(np.percentile(values_1d, 1))
    p50 = float(np.percentile(values_1d, 50))
    p99 = float(np.percentile(values_1d, 99))
    min_value = float(np.min(values_1d))
    max_value = float(np.max(values_1d))

    print(f"\n[{name}]")
    print(f"count = {values_1d.size}")
    print(f"mean  = {mean_value:.6f}")
    print(f"std   = {std_value:.6f}")
    print(f"min   = {min_value:.6f}")
    print(f"p01   = {p01:.6f}")
    print(f"p50   = {p50:.6f}")
    print(f"p99   = {p99:.6f}")
    print(f"max   = {max_value:.6f}")


def plot_reconstruction_distributions(
    *,
    x_true_train: np.ndarray,  # (T_train, N, F)
    x_hat_train: np.ndarray,  # (T_train, N, F)
    out_dir: str,
    out_prefix: str,
    sample_size: int,
    bins: int,
    seed: int,
    overlay: bool,
) -> None:
    """
    Save histogram PNGs for distributions of true, recon, residual.
    Optionally also save an overlay histogram plot (true vs recon).

    Saves:
      - {out_prefix}_hist_true.png
      - {out_prefix}_hist_recon.png
      - {out_prefix}_hist_residual.png
      - {out_prefix}_hist_overlay_true_vs_recon.png (if overlay=True)
    """
    os.makedirs(out_dir, exist_ok=True)

    # sample
    x_true_sample = _sample_finite_flat_values(
        x_true_train, sample_size=sample_size, seed=seed
    )
    x_hat_sample = _sample_finite_flat_values(
        x_hat_train, sample_size=sample_size, seed=seed
    )

    residual_train = x_hat_train - x_true_train
    residual_sample = _sample_finite_flat_values(
        residual_train, sample_size=sample_size, seed=seed
    )

    if x_true_sample.size == 0 or x_hat_sample.size == 0:
        print("[plot] No finite values to plot.")
        return

    _summarize_1d(x_true_sample, "X_train (true) sample")
    _summarize_1d(x_hat_sample, "X_hat_train (recon) sample")
    _summarize_1d(residual_sample, "Residual (hat - true) sample")

    # 1) True histogram
    plt.figure()
    plt.hist(x_true_sample, bins=bins)
    plt.title("Distribution of X_train values (sample)")
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    p_true = os.path.join(out_dir, f"{out_prefix}_hist_true.png")
    plt.savefig(p_true, dpi=200)
    plt.close()

    # 2) Recon histogram
    plt.figure()
    plt.hist(x_hat_sample, bins=bins)
    plt.title("Distribution of CP reconstruction X_hat_train values (sample)")
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    p_recon = os.path.join(out_dir, f"{out_prefix}_hist_recon.png")
    plt.savefig(p_recon, dpi=200)
    plt.close()

    # 3) Residual histogram
    plt.figure()
    plt.hist(residual_sample, bins=bins)
    plt.title("Distribution of residuals (X_hat_train - X_train) (sample)")
    plt.xlabel("residual")
    plt.ylabel("count")
    plt.tight_layout()
    p_resid = os.path.join(out_dir, f"{out_prefix}_hist_residual.png")
    plt.savefig(p_resid, dpi=200)
    plt.close()

    # 4) Overlay (True vs Recon) on same plot
    if overlay:
        plt.figure()
        plt.hist(x_true_sample, bins=bins, alpha=0.5, label="X_train (true)")
        plt.hist(x_hat_sample, bins=bins, alpha=0.5, label="X_hat_train (recon)")
        plt.title("Overlay: X_train vs X_hat_train distributions (sample)")
        plt.xlabel("value")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        p_overlay = os.path.join(
            out_dir, f"{out_prefix}_hist_overlay_true_vs_recon.png"
        )
        plt.savefig(p_overlay, dpi=200)
        plt.close()
    else:
        p_overlay = None

    print("[INFO] Saved distribution plots:")
    print(f"  {p_true}")
    print(f"  {p_recon}")
    print(f"  {p_resid}")
    if p_overlay is not None:
        print(f"  {p_overlay}")


# ------------------------ CP recon core ------------------------
def cp_recon_r2_raw_first_time_slice(
    tensor_path: str,
    cp_rank: int,
    train_ratio: float,
    n_iter_max: int,
    tol: float,
    seed: int,
    random_state_override: Optional[int],
) -> Tuple[float, float, int, Tuple[int, int, int], np.ndarray, np.ndarray]:
    """
    Returns:
      (r2_raw, mse_raw, train_time_count, train_shape, train_raw_tensor, reconstructed_train_raw_tensor)
    """
    np.random.seed(int(seed))

    full_raw_tensor = np.load(tensor_path).astype(np.float32)  # (T,N,F)
    if full_raw_tensor.ndim != 3:
        raise ValueError(
            f"Expected a 3D tensor [T,N,F], got shape={full_raw_tensor.shape}"
        )

    time_count, firm_count, feature_count = full_raw_tensor.shape

    train_time_count = int(train_ratio * time_count)
    if train_time_count <= 0:
        raise ValueError("train_time_count <= 0. Check train_ratio and T.")

    train_raw_tensor = full_raw_tensor[:train_time_count]  # (T_train,N,F)

    if np.isnan(train_raw_tensor).any():
        raise ValueError(
            "Found NaN in train_raw_tensor. Plain parafac does not handle NaN directly.\n"
            "For synthetic verification, please generate without NaN."
        )

    effective_random_state = (
        int(seed) if random_state_override is None else int(random_state_override)
    )

    cp_tensor = parafac(
        train_raw_tensor,
        rank=int(cp_rank),
        n_iter_max=int(n_iter_max),
        init="random",
        tol=float(tol),
        random_state=effective_random_state,
    )

    reconstructed_train_raw_tensor = tl.cp_to_tensor(cp_tensor)  # (T_train,N,F)

    y_true = train_raw_tensor.reshape(-1)
    y_pred = reconstructed_train_raw_tensor.reshape(-1)

    r2_value = r2_from_sse(y_true, y_pred)
    mse_value = float(np.mean((y_true - y_pred) ** 2))
    return (
        r2_value,
        mse_value,
        train_time_count,
        tuple(train_raw_tensor.shape),
        train_raw_tensor,
        reconstructed_train_raw_tensor,
    )


# ------------------------ CSV helpers ------------------------
def append_row_to_csv(csv_path: str, header: List[str], row: dict) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_int_list(comma_list: str) -> List[int]:
    if comma_list is None:
        return []
    s = str(comma_list).strip()
    if s == "":
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


# ------------------------ main ------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_path", type=str, required=True)

    # ranks: either one or many
    parser.add_argument("--cp_rank", type=int, default=None, help="Single CP rank.")
    parser.add_argument(
        "--cp_ranks",
        type=str,
        default="",
        help="Comma-separated ranks, e.g. 30,50,100.",
    )

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--n_iter_max", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for numpy + default parafac random_state.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=None,
        help="Optional: override tensorly parafac random_state. If None, uses --seed.",
    )

    parser.add_argument(
        "--output_csv", type=str, default="./results/cp_optionB_results.csv"
    )
    parser.add_argument("--run_tag", type=str, default="")

    # ---- NEW: plotting controls ----
    parser.add_argument(
        "--plot_hist", action="store_true", help="If set, save distribution histograms."
    )
    parser.add_argument("--hist_bins", type=int, default=200)
    parser.add_argument("--hist_sample_size", type=int, default=1_000_000)
    parser.add_argument("--hist_out_dir", type=str, default="./results")
    parser.add_argument(
        "--hist_overlay",
        action="store_true",
        help="If set, also save overlay histogram (true vs recon).",
    )

    args = parser.parse_args()

    rank_list: List[int] = parse_int_list(args.cp_ranks)
    if len(rank_list) == 0:
        if args.cp_rank is None:
            raise ValueError("You must provide --cp_rank or --cp_ranks")
        rank_list = [int(args.cp_rank)]

    header = [
        "timestamp_utc",
        "method",
        "run_tag",
        "tensor_path",
        "train_ratio",
        "train_T",
        "train_shape",
        "cp_rank",
        "n_iter_max",
        "tol",
        "seed",
        "random_state",
        "r2_raw",
        "mse_raw",
        "elapsed_sec",
    ]

    for cp_rank_value in rank_list:
        start_wall_time = time.time()
        (
            r2_raw,
            mse_raw,
            train_time_count,
            train_shape,
            train_raw_tensor,
            reconstructed_train_raw_tensor,
        ) = cp_recon_r2_raw_first_time_slice(
            tensor_path=args.tensor_path,
            cp_rank=int(cp_rank_value),
            train_ratio=float(args.train_ratio),
            n_iter_max=int(args.n_iter_max),
            tol=float(args.tol),
            seed=int(args.seed),
            random_state_override=args.random_state,
        )
        elapsed_seconds = time.time() - start_wall_time

        print("CP RAW first-time-slice recon (Option B, same ground truth):")
        print(f"  tensor_path   = {args.tensor_path}")
        print(f"  train_ratio   = {args.train_ratio}")
        print(f"  train_T       = {train_time_count}")
        print(f"  train_shape   = {train_shape}")
        print(f"  cp_rank       = {cp_rank_value}")
        print(f"  seed          = {args.seed}")
        print(
            f"  random_state  = {args.seed if args.random_state is None else args.random_state}"
        )
        print(f"  R2_raw        = {r2_raw}")
        print(f"  MSE_raw       = {mse_raw}")
        print(f"  elapsed_sec   = {elapsed_seconds:.3f}")

        row = {
            "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
            "method": "CP(parafac)",
            "run_tag": args.run_tag,
            "tensor_path": args.tensor_path,
            "train_ratio": float(args.train_ratio),
            "train_T": int(train_time_count),
            "train_shape": str(train_shape),
            "cp_rank": int(cp_rank_value),
            "n_iter_max": int(args.n_iter_max),
            "tol": float(args.tol),
            "seed": int(args.seed),
            "random_state": int(
                args.seed if args.random_state is None else args.random_state
            ),
            "r2_raw": float(r2_raw),
            "mse_raw": float(mse_raw),
            "elapsed_sec": float(elapsed_seconds),
        }
        append_row_to_csv(args.output_csv, header, row)
        print(f"[INFO] Appended 1 row to CSV: {args.output_csv}")

        # ---- NEW: plot distributions ----
        if args.plot_hist:
            tensor_base = os.path.splitext(os.path.basename(args.tensor_path))[0]
            out_prefix = f"{tensor_base}_rank{cp_rank_value}_train{int(args.train_ratio * 100)}pct"
            plot_reconstruction_distributions(
                x_true_train=train_raw_tensor,
                x_hat_train=reconstructed_train_raw_tensor,
                out_dir=str(args.hist_out_dir),
                out_prefix=out_prefix,
                sample_size=int(args.hist_sample_size),
                bins=int(args.hist_bins),
                seed=int(args.seed),
                overlay=bool(args.hist_overlay),
            )

        print("-" * 30)


if __name__ == "__main__":
    main()
