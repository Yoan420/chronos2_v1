# run_timesfm_tensor.py  — tensor route + metrics
import os, json, math, argparse, time
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import timesfm
from timesfm import TimesFmHparams, TimesFmCheckpoint

# ------------------ config ------------------

RIDE_FEATURES = [
    "price_min",
    "price_mean",
    "price_max",
    "distance_min",
    "distance_mean",
    "distance_max",
    "surge_min",
    "surge_mean",
    "surge_max",
    "api_calls",
]
RIDE_FEATURE_SET = set(RIDE_FEATURES)
import numpy as np


def apply_random_mask_in_context(
    tensor: np.ndarray,  # [T, N_trip, N_feat], float (NaNs allowed)
    feature_names: list[str],  # from your meta json
    features_to_mask: list[str],  # the 10 ride features
    context_len: int = 210,
    horizon_len: int = 168,
    p_point: float = 0.15,  # Bernoulli per-time-step mask rate (MCAR)
    block_prob: float = 0.25,  # probability to add a block gap (per series)
    block_min: int = 4,  # min length of a gap
    block_max: int = 24,  # max length of a gap
    seed: int = 0,
) -> np.ndarray:
    """
    Returns a *copy* of tensor where only the training context window
    [T - horizon_len - context_len : T - horizon_len) is masked.
    Test horizon [T - horizon_len : T) is untouched.
    """
    T, N_trip, N_feat = tensor.shape
    label_start = T - horizon_len
    ctx_start = label_start - context_len
    masked = tensor.copy()

    name2idx = {n: i for i, n in enumerate(feature_names)}
    feat_idcs = [name2idx[n] for n in features_to_mask if n in name2idx]

    rng = np.random.default_rng(seed)

    for i in range(N_trip):
        for j in feat_idcs:
            ctx = masked[ctx_start:label_start, i, j]  # view into training context
            if ctx.size == 0:
                continue

            # --- pointwise MCAR mask ---
            if p_point > 0:
                m = rng.random(ctx.shape) < p_point
                ctx[m] = np.nan

            # --- optional block/gap masks (simulate outages) ---
            if block_prob > 0:
                L = ctx.shape[0]
                # sample a geometric number of blocks
                while rng.random() < block_prob:
                    start = int(rng.integers(0, L))
                    blen = int(rng.integers(block_min, block_max + 1))
                    end = min(L, start + blen)
                    ctx[start:end] = np.nan

            # ensure *some* finite info remains in the context (so your
            # forward-fill won’t be forced to all zeros)
            if not np.isfinite(ctx).any():
                # pick any finite from the original (pre-mask) tensor if possible
                orig = tensor[ctx_start:label_start, i, j]
                idx = np.where(np.isfinite(orig))[0]
                if idx.size > 0:
                    ctx[idx[0]] = orig[idx[0]]
                else:
                    ctx[0] = 0.0  # last resort

            # write back (ctx is a view already)
            masked[ctx_start:label_start, i, j] = ctx

    return masked


def pad_to_multiple_of_32_and_clip_1d(x: np.ndarray, ctx_len: int) -> torch.Tensor:
    """Keep last ctx_len, then left-pad to multiple of 32 by repeating first value."""
    s = np.asarray(x, dtype=np.float32)[-ctx_len:]
    pad = (-len(s)) % 32
    if pad:
        s = np.concatenate([np.repeat(s[:1], pad), s], axis=0)
    return torch.tensor(s)


# ---- metrics helpers (NaN-safe, aggregated over all series×horizon) ----


def compute_metrics_all(y_true_2d: np.ndarray, y_pred_2d: np.ndarray):
    """
    y_true_2d, y_pred_2d: shape [S, H], may contain NaN.
    Returns dict with rmse, mae, r2 (centered), r2_kelly (uncentered).
    """
    yt = y_true_2d.ravel().astype(np.float64, copy=False)
    yp = y_pred_2d.ravel().astype(np.float64, copy=False)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(mask):
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "r2_kelly": np.nan}

    yt = yt[mask]
    yp = yp[mask]
    err = yp - yt
    sse = float(np.sum(err * err))
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(sse / err.size))

    # centered R^2: 1 - SSE / SST, SST = sum((y - mean(y))^2)
    ybar = float(np.mean(yt))
    sst = float(np.sum((yt - ybar) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else np.nan

    # Kelly (uncentered) R^2: 1 - SSE / sum(y^2)
    sst0 = float(np.sum(yt**2))
    r2_kelly = float(1.0 - sse / sst0) if sst0 > 0 else np.nan

    return {"rmse": rmse, "mae": mae, "r2": r2, "r2_kelly": r2_kelly}


@torch.no_grad()
def compute_mase_denominator_raw_fibers(
    train_raw_tensor: torch.Tensor,  # shape: [T_train, N_trip, F_total], dtype: float32, NaNs allowed
    target_feature_indices_list: list[
        int
    ],  # which feature columns (global indices) to include
    seasonal_period: int = 24,  # m in MASE(m); e.g., 24 for hourly data
) -> float:
    """
    Returns the scalar denominator for MASE:
        mean over training of |y_t - y_{t-m}|
    averaged across all valid (trip, feature) fibers and time indices.
    """
    # --- basic shape/type checks (vanilla, easy to read) ---
    if not isinstance(train_raw_tensor, torch.Tensor):
        raise TypeError(
            "train_raw_tensor must be a torch.Tensor [T_train, N_trip, F_total]."
        )
    if train_raw_tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {tuple(train_raw_tensor.shape)}.")

    T_train, N_trip, F_total = train_raw_tensor.shape
    m: int = int(seasonal_period)  # ensure plain int

    # Not enough history to form lag-m pairs
    if T_train <= m:
        return float("nan")

    # Work on CPU for simple indexing; keep dtype float32
    series_3d_cpu: torch.Tensor = train_raw_tensor.detach().cpu()

    # Python accumulators (simple and safe for NaN filtering)
    total_abs_diff_sum: float = 0.0  # sum of |y_t - y_{t-m}|
    total_valid_pairs: int = 0  # count of valid pairs in denominator

    # -------- iterate fibers: each (trip_index, feature_index) is a 1-D time series --------
    for trip_index in range(N_trip):
        # Slice once per trip to reduce indexing overhead; shape [T_train, F_total]
        trip_matrix_tf: torch.Tensor = series_3d_cpu[:, trip_index, :]

        for feature_index in target_feature_indices_list:
            if feature_index < 0 or feature_index >= F_total:
                continue  # ignore bad indices

            # 1-D series for this fiber
            y_t: torch.Tensor = trip_matrix_tf[:, feature_index]  # shape [T_train]

            # Build lagged pairs via slicing
            cur_vals: torch.Tensor = y_t[m:]  # t = m..T_train-1
            prev_vals: torch.Tensor = y_t[:-m]  # t = 0..T_train-m-1

            # Valid where both are finite (avoid NaNs)
            valid_mask: torch.Tensor = (~torch.isnan(cur_vals)) & (
                ~torch.isnan(prev_vals)
            )

            # Skip if no valid pairs
            if not torch.any(valid_mask):
                continue

            # Absolute differences on valid pairs
            diffs: torch.Tensor = torch.abs(
                cur_vals[valid_mask] - prev_vals[valid_mask]
            )  # [n_valid]

            # Accumulate into Python scalars
            total_abs_diff_sum += float(
                diffs.sum().item()
            )  # .item(): Tensor -> Python float
            total_valid_pairs += int(diffs.numel())  # .numel(): #elements in tensor

    # Mean across all valid pairs; guard against zero division
    if total_valid_pairs == 0:
        return float("nan")

    mase_denominator_value: float = total_abs_diff_sum / float(total_valid_pairs)
    return mase_denominator_value


# ------------------ main ------------------


def main(args):
    # === Load dense tensor + metadata ===
    ten = np.load(args.tensor_path)  # [T, N_trip, N_feat] float64 (with NaNs)
    with open(args.meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    T, N_trip, N_feat = ten.shape
    assert (
        T >= args.context_len + args.horizon_len
    ), f"Need at least context({args.context_len}) + horizon({args.horizon_len}); got T={T}"
    # make a *masked* copy that only touches the training context window
    masked_ten = apply_random_mask_in_context(
        tensor=ten,
        feature_names=meta["feature_names"],
        features_to_mask=RIDE_FEATURES,  # only the 10 ride targets
        context_len=args.context_len,
        horizon_len=args.horizon_len,
        p_point=args.mask_p,
        block_prob=args.mask_block_prob,
        block_min=args.mask_block_min,
        block_max=args.mask_block_max,
        seed=args.mask_seed,
    )

    # optional: save for debugging
    if args.save_masked_tensor:
        masked_path = (
            os.path.splitext(args.tensor_path)[0] + f"_masked_{args.mask_p}.npy"
        )
        np.save(masked_path, masked_ten)
        print(f"[info] saved masked tensor to {masked_path}")

    # quick sanity: report NaN rate in context before/after
    label_start = T - args.horizon_len
    ctx_start = label_start - args.context_len
    orig_ctx = ten[ctx_start:label_start, :, :][
        :,
        :,
        [
            i
            for i, _ in enumerate(meta["feature_names"])
            if meta["feature_names"][i] in RIDE_FEATURES
        ],
    ]
    mask_ctx = masked_ten[ctx_start:label_start, :, :][
        :,
        :,
        [
            i
            for i, _ in enumerate(meta["feature_names"])
            if meta["feature_names"][i] in RIDE_FEATURES
        ],
    ]
    orig_rate = np.isnan(orig_ctx).mean()
    mask_rate = np.isnan(mask_ctx).mean()
    print(
        f"[info] context NaN rate: original={orig_rate:.3%}  after-mask={mask_rate:.3%}"
    )

    # map feature name -> index in dense tensor
    feat_to_idx = {name: i for i, name in enumerate(meta["feature_names"])}
    ride_feat_idxs = [feat_to_idx[n] for n in RIDE_FEATURES if n in feat_to_idx]
    if len(ride_feat_idxs) != 10:
        missing = [n for n in RIDE_FEATURES if n not in feat_to_idx]
        raise ValueError(f"Expected 10 ride features; missing {missing}")

    # Absolute label window (last H steps)
    H = args.horizon_len
    ctx = args.context_len
    label_start = T - H  # absolute t index for the first forecasted step
    # === MASE denominator from training data (up to label_start) ===
    # Shapes:
    #   train_slice_ndarray: np.ndarray [T_train, N_trip, N_feat]
    #   training_raw_tensor_for_mase: torch.Tensor [T_train, N_trip, N_feat], float32
    train_slice_ndarray: np.ndarray = ten[:label_start, :, :]  # training part only
    training_raw_tensor_for_mase: torch.Tensor = torch.tensor(
        train_slice_ndarray, dtype=torch.float32
    )  # convert to torch for the denominator helper

    # Use the same 10 ride features as your forecast target space
    target_feature_indices_list: list[int] = ride_feat_idxs

    # Seasonal period (m). Default 24 for hourly series; configurable by flag.
    mase_denominator_value: float = compute_mase_denominator_raw_fibers(
        train_raw_tensor=training_raw_tensor_for_mase,
        target_feature_indices_list=target_feature_indices_list,
        seasonal_period=int(args.mase_seasonal_period),
    )

    # === Build list of series (trip x ride_feature) ===
    # trip index corresponds to meta["trip_ids"] order; feature is local idx 0..9
    series_index = []
    for ti in range(N_trip):
        for lf, tf in enumerate(ride_feat_idxs):
            series_index.append((ti, lf, tf))

    total_series = len(series_index)  # 156 trips * 10 feats = 1560
    if total_series == 0:
        raise RuntimeError("No ride series found.")

    # === Init TimesFM 2.0 (torch) ===
    tfm = timesfm.TimesFm(
        hparams=TimesFmHparams(
            backend="gpu",
            per_core_batch_size=args.batch_size,
            horizon_len=H,
            # keep model context_len large; we feed exactly ctx points
            context_len=2048,
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

    # === Forecast in batches ===
    all_preds = np.empty((total_series, H), dtype=np.float32)
    all_gts = np.empty((total_series, H), dtype=np.float32)

    start_time = time.time()
    n_batches = math.ceil(total_series / args.batch_size)
    for b in tqdm(range(n_batches), desc="Forecast", unit="batch"):
        a = b * args.batch_size
        z = min(a + args.batch_size, total_series)
        batch = series_index[a:z]

        contexts, labels = [], []
        for trip_i, _, tensor_feat_i in batch:
            full = ten[:, trip_i, tensor_feat_i]  # [T] float64 with NaNs

            # context slice (last ctx points BEFORE label win   dow)
            ctx_series = masked_ten[
                label_start - ctx : label_start, trip_i, tensor_feat_i
            ]

            # simple NaN handling in context: forward-fill; if all-NaN -> zeros
            if np.isnan(ctx_series).any():
                c = ctx_series.astype(np.float32)
                mask = np.isnan(c)
                if mask.any():
                    idx = np.where(~mask, np.arange(len(c)), 0)  # nearest prev
                    np.maximum.accumulate(idx, out=idx)
                    c = c[idx]
                    if np.isnan(c[0]):  # leading all-NaN
                        c[:] = 0.0
                ctx_series = c

            contexts.append(pad_to_multiple_of_32_and_clip_1d(ctx_series, ctx_len=ctx))
            labels.append(full[label_start:])  # may include NaNs

        freq = [0] * len(contexts)  # hourly => category 0 (TimesFM v2)
        point_forecast, _ = tfm.forecast(contexts, freq=freq)  # (B, H)

        all_preds[a:z, :] = np.asarray(point_forecast, dtype=np.float32)
        all_gts[a:z, :] = np.stack(labels, axis=0).astype(np.float32)

    total_time = time.time() - start_time

    # === Write requested rows: t,trip,feature,GroundTruth,Forecasted ===
    out_rows = []
    for s, (trip_i, local_feat_i, _) in tqdm(
        enumerate(series_index),
        total=total_series,
        desc="Writing CSV rows",
        unit="series",
    ):
        gt_s = all_gts[s]
        pr_s = all_preds[s]
        for t_off in range(H):
            t_abs = label_start + t_off
            gt = gt_s[t_off]
            pr = pr_s[t_off]
            if np.isfinite(gt) and np.isfinite(pr):
                out_rows.append((t_abs, trip_i, local_feat_i, float(gt), float(pr)))

    df = pd.DataFrame(
        out_rows, columns=["t", "trip", "feature", "GroundTruth", "Forecasted"]
    )
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    # === Global metrics over all finite pairs ===
    metrics = compute_metrics_all(all_gts, all_preds)  # returns rmse, mae, r2, r2_kelly

    # MASE = MAE / denominator (seasonal naive MAE on training).
    if (
        np.isfinite(metrics["mae"])
        and np.isfinite(mase_denominator_value)
        and (mase_denominator_value > 0)
    ):
        mase_value: float = float(metrics["mae"] / mase_denominator_value)
    else:
        mase_value: float = float("nan")

    metrics["mase"] = mase_value  # add to dict
    metrics_df = pd.DataFrame([metrics])  # will include 'mase' column

    metrics_path = os.path.splitext(args.out_csv)[0] + f"_{args.mask_p}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"\nSaved pairs : {args.out_csv}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Series (trip x ride_feature): {total_series}  |  rows written: {len(df)}")
    print(f"Eval time: {total_time:.2f}s  |  trips: {N_trip}  |  ride features: 10")
    print("\n== METRICS ==")
    for k, v in metrics.items():
        print(f"{k:>10s}: {v:.6f}" if np.isfinite(v) else f"{k:>10s}: nan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor_path", type=str, default="rideshare_tensor.npy")
    ap.add_argument("--meta_path", type=str, default="rideshare_tensor_dims.json")
    ap.add_argument(
        "--out_csv",
        type=str,
        default="./examples/ablation_studies/Rideshare/rideshare_gt_pred_last168.csv",
    )

    ap.add_argument("--context_len", type=int, default=210)
    ap.add_argument("--horizon_len", type=int, default=168)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument(
        "--mask_p",
        type=float,
        default=0.90,
        help="pointwise MCAR mask prob in context (0 disables)",
    )
    ap.add_argument(
        "--mask_block_prob",
        type=float,
        default=0,
        help="probability to add a block gap (repeat until fail)",
    )
    ap.add_argument("--mask_block_min", type=int, default=4, help="min length of a gap")
    ap.add_argument(
        "--mask_block_max", type=int, default=24, help="max length of a gap"
    )
    ap.add_argument("--mask_seed", type=int, default=0, help="random seed for masking")
    ap.add_argument(
        "--save_masked_tensor",
        action="store_true",
        help="if set, saves a copy of the masked tensor for inspection",
    )

    ap.add_argument(
        "--mase_seasonal_period",
        type=int,
        default=24,
        help="Seasonal period m for MASE denominator (e.g., 24 for hourly).",
    )

    args = ap.parse_args()
    main(args)
