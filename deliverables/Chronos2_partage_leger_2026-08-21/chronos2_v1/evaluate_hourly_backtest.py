#!/usr/bin/env python
"""Evaluate two hourly forecasts with a paired delivery-day bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chronos2_hourly.hourly_contract import local_delivery_day_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("backtest_file", type=Path)
    parser.add_argument("--baseline", default="ensemble__q50")
    parser.add_argument("--candidate", default="residual_corrected__q50")
    parser.add_argument("--actual", default="actual")
    parser.add_argument("--timezone", default="Europe/Paris")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _validate_complete_days(
    index: pd.DatetimeIndex,
    *,
    timezone: str,
) -> list[object]:
    local_dates = pd.Index(index.tz_convert(timezone).date)
    days = local_dates.unique().tolist()
    expected_days = pd.date_range(days[0], days[-1], freq="D").date.tolist()
    if days != expected_days:
        raise ValueError("La période évaluée contient des jours manquants.")
    for day in days:
        observed = index[local_dates == day]
        expected = local_delivery_day_index(day, timezone=timezone)
        if not observed.equals(expected):
            raise ValueError(f"Journée locale incomplète: {day}.")
    return days


def evaluate(
    frame: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
    actual: str,
    timezone: str,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = ["delivery_start_utc", actual, baseline, candidate]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Colonnes absentes: {missing}.")
    work = frame.loc[:, required].copy()
    work["delivery_start_utc"] = pd.to_datetime(
        work["delivery_start_utc"], utc=True, errors="raise"
    )
    for column in (actual, baseline, candidate):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=[actual, baseline, candidate]).sort_values(
        "delivery_start_utc"
    )
    if work.empty or work["delivery_start_utc"].duplicated().any():
        raise ValueError("Période évaluée vide ou timestamps dupliqués.")
    work = work.set_index("delivery_start_utc")
    days = _validate_complete_days(work.index, timezone=timezone)
    local = work.index.tz_convert(timezone)
    work["local_date"] = local.date
    work["local_hour"] = local.hour
    work["local_month"] = local.strftime("%Y-%m")
    work["baseline_abs_error"] = (work[actual] - work[baseline]).abs()
    work["candidate_abs_error"] = (work[actual] - work[candidate]).abs()
    work["paired_delta_abs_error"] = (
        work["candidate_abs_error"] - work["baseline_abs_error"]
    )

    daily = work.groupby("local_date", sort=True).agg(
        n_hours=(actual, "size"),
        baseline_abs_error_sum=("baseline_abs_error", "sum"),
        candidate_abs_error_sum=("candidate_abs_error", "sum"),
    )
    daily["baseline_mae"] = (
        daily["baseline_abs_error_sum"] / daily["n_hours"]
    )
    daily["candidate_mae"] = (
        daily["candidate_abs_error_sum"] / daily["n_hours"]
    )
    daily["delta_mae"] = daily["candidate_mae"] - daily["baseline_mae"]

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples doit être >= 1.")
    rng = np.random.default_rng(seed)
    samples = rng.integers(
        0,
        len(daily),
        size=(bootstrap_samples, len(daily)),
    )
    counts = daily["n_hours"].to_numpy(dtype=float)
    baseline_sums = daily["baseline_abs_error_sum"].to_numpy(dtype=float)
    candidate_sums = daily["candidate_abs_error_sum"].to_numpy(dtype=float)
    sampled_counts = counts[samples].sum(axis=1)
    bootstrap_delta = (
        candidate_sums[samples].sum(axis=1) / sampled_counts
        - baseline_sums[samples].sum(axis=1) / sampled_counts
    )

    baseline_mae = float(work["baseline_abs_error"].mean())
    candidate_mae = float(work["candidate_abs_error"].mean())
    delta = candidate_mae - baseline_mae
    summary: dict[str, object] = {
        "baseline": baseline,
        "candidate": candidate,
        "n_hours": int(len(work)),
        "n_local_days": int(len(days)),
        "start_utc": str(work.index[0]),
        "end_utc": str(work.index[-1]),
        "baseline_mae": baseline_mae,
        "candidate_mae": candidate_mae,
        "delta_mae": delta,
        "relative_improvement": float(-delta / baseline_mae),
        "paired_day_bootstrap": {
            "samples": int(bootstrap_samples),
            "seed": int(seed),
            "delta_ci95": [
                float(np.quantile(bootstrap_delta, 0.025)),
                float(np.quantile(bootstrap_delta, 0.975)),
            ],
            "probability_candidate_better": float(
                np.mean(bootstrap_delta < 0.0)
            ),
        },
    }

    def grouped(column: str) -> pd.DataFrame:
        result = work.groupby(column, sort=True).agg(
            n_hours=(actual, "size"),
            baseline_mae=("baseline_abs_error", "mean"),
            candidate_mae=("candidate_abs_error", "mean"),
        )
        result["delta_mae"] = result["candidate_mae"] - result["baseline_mae"]
        return result.reset_index()

    return summary, daily.reset_index(), grouped("local_month"), grouped("local_hour")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.backtest_file.resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, daily, monthly, hourly = evaluate(
        _load(args.backtest_file),
        baseline=args.baseline,
        candidate=args.candidate,
        actual=args.actual,
        timezone=args.timezone,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    daily.to_csv(output_dir / "evaluation_by_day.csv", index=False)
    monthly.to_csv(output_dir / "evaluation_by_month.csv", index=False)
    hourly.to_csv(output_dir / "evaluation_by_hour.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
