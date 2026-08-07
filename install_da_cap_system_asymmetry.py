#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ALIAS = "da_cap_system_asymmetry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crée une copie du YAML Chronos-2 intégrant "
            "da_cap_system_asymmetry."
        )
    )
    parser.add_argument(
        "--config",
        default="chronos2_selected_core.yaml",
    )
    parser.add_argument("--destination", default=None)
    parser.add_argument(
        "--minimum-coverage",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--preserve-output",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.config).expanduser().resolve()
    destination = (
        Path(args.destination).expanduser().resolve()
        if args.destination
        else source.with_name(f"{source.stem}_da_cap.yaml")
    )

    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config.setdefault("data", {})
    config["data"].setdefault("pit_files", {})
    config["data"]["pit_files"][ALIAS] = (
        f"{ALIAS}.parquet"
    )

    zone = config["zones"]["FR"]
    zone.setdefault("covariates", {})
    zone["covariates"][ALIAS] = {
        "enabled": True,
        "source": "pit_parquet",
        "description": (
            "Asymétrie point-in-time des capacités Day-Ahead "
            "d'import/export de la France (GW)"
        ),
        "fill_method": "none",
        "minimum_coverage": float(args.minimum_coverage),
        "future": {
            "known_future": True,
            "strategies": ["oracle"],
        },
    }

    config.setdefault("derived_features", {})
    config["derived_features"][ALIAS] = {
        "unit": "GW",
        "formula": (
            "ES->FR + IT_NORTH->FR + UK->FR "
            "- FR->IT_NORTH - FR->UK"
        ),
        "point_in_time": True,
        "cutoff": config["data"].get(
            "forecast_origin_local_time", "08:00"
        ),
        "builder": "build_da_cap_system_asymmetry.py",
    }

    if not args.preserve_output:
        output = config.setdefault("output", {})
        current_dir = str(
            output.get("directory", "runs/chronos2")
        ).rstrip("/\\")
        if not current_dir.endswith("_da_cap"):
            output["directory"] = f"{current_dir}_da_cap"

        report = config.setdefault("report", {})
        report["filename"] = (
            "chronos2_selected_core_da_cap_system_asymmetry.html"
        )
        title = str(report.get("title", "Chronos-2"))
        if "asymétrie" not in title.lower():
            report["title"] = (
                f"{title} + asymétrie capacité Day-Ahead"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        yaml.safe_dump(
            config,
            handle,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        )

    print(f"Configuration créée : {destination}")
    print(f"Covariable ajoutée   : {ALIAS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
