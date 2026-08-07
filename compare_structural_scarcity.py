#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def output_dir(config_path: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    root = Path(
        config.get("data", {}).get(
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


def metrics(
    config_path: Path,
    zone: str,
) -> pd.Series:
    path = (
        output_dir(config_path)
        / "metrics_all_zones.csv"
    )
    frame = pd.read_csv(path)
    if "zone" in frame.columns:
        selected = frame.loc[
            frame["zone"]
            .astype(str)
            .str.upper()
            .eq(zone.upper())
        ]
        if selected.empty:
            raise KeyError(
                f"Zone {zone} absente de {path}"
            )
        return selected.iloc[0]
    return frame.iloc[0]


def higher_is_better(name: str) -> bool:
    text = name.lower()
    return any(
        token in text
        for token in (
            "correlation",
            "coverage",
            "precision",
            "recall",
            "specificity",
            "r2",
        )
    )


def parse_args():
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
    parser.add_argument("--zone", default="FR")
    parser.add_argument(
        "--output",
        default=(
            "runs/"
            "structural_scarcity_comparison.csv"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    c3 = metrics(
        Path(args.c3_config).resolve(),
        args.zone,
    )
    c4 = metrics(
        Path(args.c4_config).resolve(),
        args.zone,
    )

    common = set(c3.index) & set(c4.index)
    rows = []

    for metric in sorted(common):
        if metric == "zone":
            continue

        v3 = pd.to_numeric(
            pd.Series([c3[metric]]),
            errors="coerce",
        ).iloc[0]
        v4 = pd.to_numeric(
            pd.Series([c4[metric]]),
            errors="coerce",
        ).iloc[0]

        if not np.isfinite(v3) or not np.isfinite(v4):
            continue

        hib = higher_is_better(metric)

        if v3 == 0:
            gain = np.nan
        elif hib:
            gain = 100.0 * (
                v4 - v3
            ) / abs(v3)
        else:
            gain = 100.0 * (
                v3 - v4
            ) / abs(v3)

        rows.append(
            {
                "zone": args.zone,
                "metric": metric,
                "C3_structural_residual": float(v3),
                "C4_scarcity_residual": float(v4),
                "gain_C4_vs_C3_percent": float(gain),
                "higher_is_better": hib,
            }
        )

    result = pd.DataFrame(rows)
    path = Path(args.output).resolve()
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    result.to_csv(
        path,
        index=False,
    )

    print(f"Comparaison écrite : {path}")

    key = result.loc[
        result["metric"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "mae",
                "rmse",
                "crps",
                "bias",
                "ramp_mae",
                "correlation",
            ]
        )
    ]
    print(
        (
            key
            if not key.empty
            else result.head(20)
        ).to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
