
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from chronos2_modular.metrics import compute_metrics
from chronos2_zero_plateau.gate import apply_soft_zero_gate
from chronos2_zero_plateau.labels import (
    daily_primary_events,
    mark_near_zero_plateaus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="chronos2_selected_core_zero_plateau.yaml",
    )
    parser.add_argument("--zone", default="FR")
    parser.add_argument(
        "--predictions-file",
        default="data/derived/zero_plateau_predictions.csv.gz",
    )
    parser.add_argument(
        "--metadata-file",
        default="data/derived/zero_plateau_model_metadata.json",
    )
    return parser.parse_args()


def project_root(config_path: Path, config: dict) -> Path:
    root = Path(config.get("data", {}).get("project_root", "."))
    if not root.is_absolute():
        root = config_path.parent / root
    return root.resolve()


def output_root(project: Path, config: dict) -> Path:
    path = Path(config.get("output", {}).get("directory", "runs/chronos2"))
    if not path.is_absolute():
        path = project / path
    return path.resolve()


def classification_metrics(actual, predicted) -> dict[str, float]:
    a = np.asarray(actual, dtype=bool)
    p = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(a & p))
    fp = int(np.sum(~a & p))
    fn = int(np.sum(a & ~p))
    tn = int(np.sum(~a & ~p))

    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision)
        and np.isfinite(recall)
        and precision + recall
        else float("nan")
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def predicted_daily_events(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    days = frame["timestamp_local"].dt.normalize()

    for day in pd.Index(days.unique()):
        block = frame.loc[days == day].sort_values("timestamp_local")
        selected = block.loc[block["zero_plateau_block_flag"] >= 0.5]
        if selected.empty:
            rows.append(
                {
                    "day": day,
                    "has_plateau": 0,
                    "start": pd.NaT,
                    "end": pd.NaT,
                    "duration_hours": 0,
                }
            )
        else:
            rows.append(
                {
                    "day": day,
                    "has_plateau": 1,
                    "start": selected["timestamp_local"].min(),
                    "end": selected["timestamp_local"].max(),
                    "duration_hours": int(len(selected)),
                }
            )
    return pd.DataFrame(rows)


def event_metrics(
    actual_events: pd.DataFrame,
    predicted_events: pd.DataFrame,
) -> dict[str, float]:
    merged = actual_events.merge(
        predicted_events,
        on="day",
        how="outer",
        suffixes=("_actual", "_predicted"),
    )
    for column in (
        "has_plateau_actual",
        "has_plateau_predicted",
        "duration_hours_actual",
        "duration_hours_predicted",
    ):
        merged[column] = merged[column].fillna(0)

    result = classification_metrics(
        merged["has_plateau_actual"],
        merged["has_plateau_predicted"],
    )

    both = merged.loc[
        (merged["has_plateau_actual"] == 1)
        & (merged["has_plateau_predicted"] == 1)
    ].copy()
    if both.empty:
        result.update(
            {
                "start_mae_hours": float("nan"),
                "end_mae_hours": float("nan"),
                "duration_mae_hours": float("nan"),
            }
        )
        return result

    start_error = (
        (
            pd.to_datetime(both["start_predicted"], utc=True)
            - pd.to_datetime(both["start_actual"], utc=True)
        )
        .dt.total_seconds()
        .abs()
        / 3600
    )
    end_error = (
        (
            pd.to_datetime(both["end_predicted"], utc=True)
            - pd.to_datetime(both["end_actual"], utc=True)
        )
        .dt.total_seconds()
        .abs()
        / 3600
    )
    duration_error = (
        both["duration_hours_predicted"] - both["duration_hours_actual"]
    ).abs()

    result.update(
        {
            "start_mae_hours": float(start_error.mean()),
            "end_mae_hours": float(end_error.mean()),
            "duration_mae_hours": float(duration_error.mean()),
        }
    )
    return result


def price_slice_metrics(
    frame: pd.DataFrame,
    actual_plateau,
    *,
    low: float,
    high: float,
    solar_start: int,
    solar_end: int,
) -> dict[str, float]:
    actual = pd.to_numeric(frame["actual"], errors="coerce").to_numpy(dtype=float)
    predicted = pd.to_numeric(frame["q50"], errors="coerce").to_numpy(dtype=float)
    error = np.abs(predicted - actual)
    plateau = np.asarray(actual_plateau, dtype=bool)
    hours = frame["timestamp_local"].dt.hour.to_numpy()
    midday = (hours >= solar_start) & (hours <= solar_end)

    predicted_near_zero = (
        np.isfinite(predicted)
        & (predicted >= low)
        & (predicted <= high)
    )
    price_detection = classification_metrics(plateau, predicted_near_zero)

    return {
        "overall_mae_q50": float(np.nanmean(error)),
        "midday_mae_q50": float(np.nanmean(error[midday])),
        "actual_plateau_mae_q50": (
            float(np.nanmean(error[plateau])) if plateau.any() else float("nan")
        ),
        "actual_plateau_hours": int(plateau.sum()),
        "predicted_price_near_zero_hours": int(predicted_near_zero.sum()),
        "price_zero_precision": price_detection["precision"],
        "price_zero_recall": price_detection["recall"],
        "price_zero_f1": price_detection["f1"],
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    project = project_root(config_path, config)
    root = output_root(project, config)
    zone_dir = root / args.zone.lower()

    backtest_path = zone_dir / "backtest_native_covariates.csv"
    live_path = zone_dir / "day_ahead_forecast_native.csv"
    probability_path = (project / args.predictions_file).resolve()
    metadata_path = (project / args.metadata_file).resolve()

    backtest = pd.read_csv(backtest_path)
    probabilities = pd.read_csv(probability_path, compression="infer")

    backtest["timestamp_utc"] = pd.to_datetime(
        backtest["timestamp"], errors="coerce", utc=True
    )
    probabilities["timestamp_utc"] = pd.to_datetime(
        probabilities["timestamp"], errors="coerce", utc=True
    )

    probability_columns = [
        "timestamp_utc",
        "zero_plateau_probability",
        "zero_plateau_block_flag",
        "zero_plateau_block_probability",
        "zero_plateau_event_probability",
        "zero_plateau_block_position",
    ]
    merged = (
        backtest.merge(
            probabilities[probability_columns],
            on="timestamp_utc",
            how="left",
            validate="many_to_one",
        )
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    timezone = config["zones"][args.zone].get("timezone", "Europe/Paris")
    merged["timestamp_local"] = merged["timestamp_utc"].dt.tz_convert(timezone)

    settings = config.get("zero_plateau", {}) or {}
    low = float(settings.get("low", -3.0))
    high = float(settings.get("high", 3.0))
    min_hours = int(settings.get("min_consecutive_hours", 3))
    solar_start = int(settings.get("solar_start_hour", 8))
    solar_end = int(settings.get("solar_end_hour", 19))

    actual_series = pd.Series(
        pd.to_numeric(merged["actual"], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(merged["timestamp_local"]),
    )
    labelled = mark_near_zero_plateaus(
        actual_series,
        low=low,
        high=high,
        min_consecutive_hours=min_hours,
        solar_start_hour=solar_start,
        solar_end_hour=solar_end,
    )
    actual_plateau = labelled["plateau_label"].to_numpy(dtype=bool)
    predicted_plateau = (
        merged["zero_plateau_block_flag"].fillna(0).to_numpy() >= 0.5
    )

    hour_detection = classification_metrics(
        actual_plateau,
        predicted_plateau,
    )
    actual_events = daily_primary_events(labelled)
    predicted_events = predicted_daily_events(merged)
    event_detection = event_metrics(actual_events, predicted_events)

    c6a_slices = price_slice_metrics(
        merged,
        actual_plateau,
        low=low,
        high=high,
        solar_start=solar_start,
        solar_end=solar_end,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expert_quantiles = metadata["zero_expert_quantiles"]
    gate = settings.get("soft_gate", {}) or {}
    threshold = float(gate.get("probability_threshold", 0.55))
    max_weight = float(gate.get("max_weight", 0.65))

    gated = apply_soft_zero_gate(
        merged,
        expert_quantiles,
        threshold=threshold,
        max_weight=max_weight,
    )
    gated["variant"] = "zero_plateau_soft_gate"

    extreme_threshold = float(
        config.get("metrics", {}).get("extreme_threshold", 150.0)
    )
    c6a_overall = compute_metrics(merged, extreme_threshold)
    c6b_overall = compute_metrics(gated, extreme_threshold)
    c6b_slices = price_slice_metrics(
        gated,
        actual_plateau,
        low=low,
        high=high,
        solar_start=solar_start,
        solar_end=solar_end,
    )

    gated.to_csv(
        zone_dir / "backtest_zero_plateau_soft_gate.csv",
        index=False,
    )

    if live_path.exists():
        live = pd.read_csv(live_path)
        live["timestamp_utc"] = pd.to_datetime(
            live["timestamp"], errors="coerce", utc=True
        )
        live = live.merge(
            probabilities[probability_columns],
            on="timestamp_utc",
            how="left",
            validate="many_to_one",
        )
        live_gated = apply_soft_zero_gate(
            live,
            expert_quantiles,
            threshold=threshold,
            max_weight=max_weight,
        )
        live_gated.to_csv(
            zone_dir / "day_ahead_forecast_zero_plateau_soft_gate.csv",
            index=False,
        )

    summary = {
        "definition": {
            "low": low,
            "high": high,
            "min_consecutive_hours": min_hours,
            "solar_start_hour": solar_start,
            "solar_end_hour": solar_end,
        },
        "hour_detection": hour_detection,
        "event_detection": event_detection,
        "C6A_probabilities": {
            "overall": c6a_overall,
            "plateau_slices": c6a_slices,
        },
        "C6B_soft_gate": {
            "gate_threshold": threshold,
            "max_weight": max_weight,
            "overall": c6b_overall,
            "plateau_slices": c6b_slices,
        },
    }
    (zone_dir / "zero_plateau_evaluation.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
            default=str,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {"model": "C6A_probabilities", **c6a_slices},
            {"model": "C6B_soft_gate", **c6b_slices},
        ]
    ).to_csv(
        zone_dir / "zero_plateau_price_slices.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {"scope": "hour", **hour_detection},
            {"scope": "event", **event_detection},
        ]
    ).to_csv(
        zone_dir / "zero_plateau_detection_metrics.csv",
        index=False,
    )

    print("=" * 96)
    print("C6 — ÉVALUATION ZERO PLATEAU")
    print("=" * 96)
    print("\nDétection heure")
    print(pd.Series(hour_detection).to_string())
    print("\nDétection événement")
    print(pd.Series(event_detection).to_string())
    print("\nPrix — slices dédiées")
    print(
        pd.DataFrame(
            [
                {"model": "C6A_probabilities", **c6a_slices},
                {"model": "C6B_soft_gate", **c6b_slices},
            ]
        ).to_string(index=False)
    )
    print(f"\nSoft gate : threshold={threshold:.2f}, max_weight={max_weight:.2f}")
    print(f"Évaluation : {zone_dir / 'zero_plateau_evaluation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
