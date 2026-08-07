#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ALIAS = "da_cap_system_asymmetry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="chronos2_selected_core.yaml",
    )
    parser.add_argument(
        "--destination",
        default="chronos2_selected_core_da_cap_cutoff_persistence.yaml",
    )
    parser.add_argument("--cutoff", default="08:00")
    parser.add_argument("--max-age-hours", type=float, default=72.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.config).expanduser().resolve()
    destination = Path(args.destination).expanduser().resolve()

    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    pit_files = config.setdefault("data", {}).setdefault("pit_files", {})
    pit_files.pop(ALIAS, None)

    covariates = config["zones"]["FR"].setdefault("covariates", {})
    covariates[ALIAS] = {
        "enabled": True,
        "source": "file",
        "file": (
            "data/derived/"
            "da_cap_system_asymmetry_history.csv.gz"
        ),
        "timestamp_col": "timestamp",
        "value_col": "value",
        "description": (
            "Asymétrie finale historique des capacités Day-Ahead FR, "
            "utilisée uniquement par persistance au cutoff D-1."
        ),
        "fill_method": "none",
        "minimum_coverage": 0.50,
        "future": {
            "known_future": False,
            "strategies": [],
        },
    }

    extensions = config.setdefault("exogenous_extensions", {})
    extensions["cutoff_persistence"] = {
        "enabled": True,
        "aliases": [ALIAS],
        "cutoff_local_time": args.cutoff,
        "max_age_hours": float(args.max_age_hours),
        "include_age_hours": True,
    }

    output = config.setdefault("output", {})
    base_dir = str(
        output.get("directory", "runs/chronos2_selected_core")
    ).rstrip("/\\")
    output["directory"] = f"{base_dir}_da_cap_cutoff_persistence"

    report = config.setdefault("report", {})
    report["filename"] = "chronos2_da_cap_cutoff_persistence.html"
    report["title"] = (
        "Chronos-2 — asymétrie capacité persistée au cutoff"
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(
            config,
            handle,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        )

    print(f"Configuration créée : {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
