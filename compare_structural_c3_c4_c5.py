#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PRIMARY_METRICS = {
    "bias_q50",
    "correlation_q50",
    "coverage_q10_q90",
    "crps_quantile_approx",
    "extreme_precision",
    "extreme_recall",
    "interval_width_q10_q90",
    "mae_point",
    "mae_q50",
    "mean_pinball_9q",
    "median_ae_q50",
    "negative_precision",
    "negative_recall",
    "ramp_bias",
    "ramp_mae",
    "ramp_rmse",
    "rmse_q50",
}

HIGHER_IS_BETTER = {
    "correlation_q50",
    "coverage_q10_q90",
    "extreme_precision",
    "extreme_recall",
    "negative_precision",
    "negative_recall",
}


def config_output_dir(
    config_path: Path,
) -> Path:
    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    root = Path(
        config.get(
            "data",
            {},
        ).get(
            "project_root",
            ".",
        )
    )
    if not root.is_absolute():
        root = (
            config_path.parent / root
        ).resolve()

    output = Path(
        config.get(
            "output",
            {},
        ).get(
            "directory",
            "runs/chronos2",
        )
    )
    if not output.is_absolute():
        output = root / output
    return output.resolve()


def load_metrics(
    config_path: Path,
    zone: str,
) -> dict[str, float]:
    path = (
        config_output_dir(
            config_path
        )
        / "metrics_all_zones.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Métriques absentes : {path}"
        )

    frame = pd.read_csv(path)
    if "zone" in frame.columns:
        frame = frame.loc[
            frame["zone"]
            .astype(str)
            .str.upper()
            .eq(zone.upper())
        ]
    if frame.empty:
        raise ValueError(
            f"Aucune métrique {zone} dans {path}"
        )

    row = frame.iloc[0]
    result = {}
    for metric in PRIMARY_METRICS:
        if metric not in row.index:
            continue
        value = pd.to_numeric(
            pd.Series(
                [row[metric]]
            ),
            errors="coerce",
        ).iloc[0]
        if np.isfinite(value):
            result[metric] = float(
                value
            )
    return result


def improvement(
    baseline: float,
    candidate: float,
    metric: str,
) -> float:
    if baseline == 0:
        return float("nan")
    if metric in HIGHER_IS_BETTER:
        return float(
            100.0
            * (candidate - baseline)
            / abs(baseline)
        )
    if metric in {
        "bias_q50",
        "ramp_bias",
    }:
        return float(
            100.0
            * (
                abs(baseline)
                - abs(candidate)
            )
            / max(abs(baseline), 1e-12)
        )
    return float(
        100.0
        * (baseline - candidate)
        / abs(baseline)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--c3-config",
        default=(
            "chronos2_selected_core_"
            "structural_residual.yaml"
        ),
    )
    parser.add_argument(
        "--c4-config",
        default=(
            "chronos2_selected_core_"
            "structural_scarcity_residual.yaml"
        ),
    )
    parser.add_argument(
        "--c5-config",
        default=(
            "chronos2_selected_core_"
            "structural_scarcity_features.yaml"
        ),
    )
    parser.add_argument(
        "--zone",
        default="FR",
    )
    parser.add_argument(
        "--output",
        default=(
            "runs/"
            "structural_c3_c4_c5_comparison.csv"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    configs = {
        "C3_structural_residual": Path(
            args.c3_config
        ).resolve(),
        "C4_scarcity_price": Path(
            args.c4_config
        ).resolve(),
        "C5_scarcity_features": Path(
            args.c5_config
        ).resolve(),
    }

    metrics_by_model = {}
    for name, path in configs.items():
        try:
            metrics_by_model[name] = (
                load_metrics(
                    path,
                    args.zone,
                )
            )
        except FileNotFoundError:
            if name == "C4_scarcity_price":
                continue
            raise

    c3 = metrics_by_model[
        "C3_structural_residual"
    ]
    rows = []

    all_metrics = sorted(
        set().union(
            *[
                values.keys()
                for values in metrics_by_model.values()
            ]
        )
    )

    for metric in all_metrics:
        row = {
            "zone": args.zone,
            "metric": metric,
        }
        for model, values in metrics_by_model.items():
            row[model] = values.get(
                metric,
                np.nan,
            )

        c3_value = c3.get(
            metric,
            np.nan,
        )
        c5_value = metrics_by_model[
            "C5_scarcity_features"
        ].get(
            metric,
            np.nan,
        )
        if (
            np.isfinite(c3_value)
            and np.isfinite(c5_value)
        ):
            row[
                "gain_C5_vs_C3_percent"
            ] = improvement(
                c3_value,
                c5_value,
                metric,
            )
        else:
            row[
                "gain_C5_vs_C3_percent"
            ] = np.nan

        rows.append(row)

    result = pd.DataFrame(rows)
    output = Path(
        args.output
    ).resolve()
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    result.to_csv(
        output,
        index=False,
    )

    print("=" * 100)
    print("C3 vs C4 vs C5 — MÉTRIQUES PRIMAIRES UNIQUEMENT")
    print("=" * 100)

    priority = [
        "mae_q50",
        "rmse_q50",
        "crps_quantile_approx",
        "ramp_mae",
        "ramp_rmse",
        "bias_q50",
        "correlation_q50",
        "extreme_precision",
        "extreme_recall",
        "negative_precision",
        "negative_recall",
        "coverage_q10_q90",
    ]
    display = result.set_index(
        "metric"
    ).reindex(
        priority
    ).dropna(
        how="all"
    ).reset_index()

    print(
        display.to_string(
            index=False
        )
    )
    print(
        f"\nComparaison écrite : {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
