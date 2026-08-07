#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def resolve_config_output(config_path: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    project_root = Path(config.get("data", {}).get("project_root", "."))
    if not project_root.is_absolute():
        project_root = (config_path.parent / project_root).resolve()
    output = Path(config.get("output", {}).get("directory", "runs/chronos2"))
    if not output.is_absolute():
        output = project_root / output
    return output.resolve()


def read_metrics(config_path: Path, label: str, zone: str) -> pd.Series:
    output = resolve_config_output(config_path)
    path = output / "metrics_all_zones.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Métriques absentes pour {label}: {path}"
        )
    frame = pd.read_csv(path)
    if "zone" in frame:
        selected = frame.loc[frame["zone"].astype(str).str.upper().eq(zone.upper())]
        if selected.empty:
            raise KeyError(f"Zone {zone} absente de {path}")
        return selected.iloc[0]
    return frame.iloc[0]


def higher_is_better(metric: str) -> bool:
    text = metric.lower()
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--covariates-config", required=True)
    parser.add_argument("--residual-config", required=True)
    parser.add_argument("--zone", default="FR")
    parser.add_argument(
        "--output",
        default="runs/structural_model_comparison.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs = {
        "baseline": Path(args.baseline_config).expanduser().resolve(),
        "structural_covariates": Path(args.covariates_config).expanduser().resolve(),
        "structural_residual": Path(args.residual_config).expanduser().resolve(),
    }
    metrics = {
        label: read_metrics(path, label, args.zone)
        for label, path in configs.items()
    }

    common = set(metrics["baseline"].index)
    for series in metrics.values():
        common &= set(series.index)

    rows: list[dict[str, Any]] = []
    for metric in sorted(common):
        if metric == "zone":
            continue
        baseline = pd.to_numeric(
            pd.Series([metrics["baseline"][metric]]), errors="coerce"
        ).iloc[0]
        if not np.isfinite(baseline):
            continue
        for label in ("structural_covariates", "structural_residual"):
            candidate = pd.to_numeric(
                pd.Series([metrics[label][metric]]), errors="coerce"
            ).iloc[0]
            if not np.isfinite(candidate):
                continue
            if baseline == 0:
                gain = np.nan
            elif higher_is_better(metric):
                gain = 100.0 * (candidate - baseline) / abs(baseline)
            else:
                gain = 100.0 * (baseline - candidate) / abs(baseline)
            rows.append(
                {
                    "zone": args.zone,
                    "metric": metric,
                    "candidate": label,
                    "baseline_value": float(baseline),
                    "candidate_value": float(candidate),
                    "gain_percent": float(gain),
                    "higher_is_better": higher_is_better(metric),
                }
            )

    result = pd.DataFrame(rows)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Comparaison écrite : {output}")
    if not result.empty:
        key = result.loc[
            result["metric"].astype(str).str.lower().isin(
                ["mae", "rmse", "crps", "bias", "ramp_mae"]
            )
        ]
        print((key if not key.empty else result.head(20)).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
