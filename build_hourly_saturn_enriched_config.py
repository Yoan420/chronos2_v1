#!/usr/bin/env python
"""Build the reproducible hourly FR configuration enriched from Saturn."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Sequence

import yaml

from sync_saturn_hourly_candidates import SATURN_HOURLY_CANDIDATES


DESCRIPTIONS = {
    "fr_load_fcst": "Prévision PIT de consommation France",
    "fr_wind_generation_fcst": "Prévision PIT de production éolienne France",
    "fr_solar_generation_fcst": "Prévision PIT de production solaire France",
    "fr_hydro_ror_generation_fcst": "Prévision PIT hydro fil de l'eau France",
    "fr_nuclear_generation_fcst_long": (
        "Prévision PIT nucléaire France à historique long"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="chronos2_hourly_fr_residual_v1.yaml",
    )
    parser.add_argument(
        "--output",
        default="chronos2_hourly_fr_residual_saturn_v2.yaml",
    )
    parser.add_argument("--stdout", action="store_true")
    parser.add_argument(
        "--aliases",
        nargs="+",
        choices=tuple(SATURN_HOURLY_CANDIDATES),
        default=None,
    )
    parser.add_argument(
        "--run-name",
        default="chronos2_hourly_fr_residual_saturn_v2",
    )
    return parser.parse_args()


def build_config(
    base: dict[str, Any],
    *,
    aliases: Sequence[str] | None = None,
    run_name: str = "chronos2_hourly_fr_residual_saturn_v2",
) -> dict[str, Any]:
    config = dict(base)
    selected_aliases = tuple(aliases or SATURN_HOURLY_CANDIDATES)
    data = config.setdefault("data", {})
    pit_files = data.setdefault("pit_files", {})
    for alias in selected_aliases:
        pit_files[alias] = f"{alias}.parquet"

    covariates = config["zones"]["FR"].setdefault("covariates", {})
    for alias in selected_aliases:
        series = SATURN_HOURLY_CANDIDATES[alias]
        covariates[alias] = {
            "enabled": True,
            "source": "pit_parquet",
            "series": series,
            "naive_timezone": "UTC",
            "description": DESCRIPTIONS[alias],
            "fill_method": "none",
            "minimum_coverage": 0.20,
            "future": {
                "known_future": True,
                "strategies": ["oracle"],
            },
        }

    correction = config["hourly"]["residual_correction"]
    correction.update(
        {
            # Keep the residual-v1 hyperparameters: the nested calibration
            # screen rejected the shallower 500/depth-5 alternative.
            "iterations": 700,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 15.0,
        }
    )
    correction.setdefault("feature_builder", {}).update(
        {
            "include_fundamental_interactions": True,
            "include_missing_indicators": True,
        }
    )
    config["output"]["directory"] = f"runs/{run_name}"
    config["report"].update(
        {
            "enabled": True,
            "filename": f"{run_name}.html",
            "title": "Chronos-2 horaire FR — fondamentaux Saturn v2",
            "native_model": "residual_corrected",
            "baseline_model": "ensemble",
        }
    )
    return config


def main() -> int:
    args = parse_args()
    source = Path(args.base_config).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle) or {}
    rendered = yaml.safe_dump(
        build_config(base, aliases=args.aliases, run_name=args.run_name),
        allow_unicode=True,
        sort_keys=False,
        width=100,
    )
    if args.stdout:
        sys.stdout.write(rendered)
        return 0
    output = Path(args.output).expanduser().resolve()
    output.write_text(rendered, encoding="utf-8", newline="\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
