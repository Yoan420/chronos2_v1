#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="chronos2_m1_calendar.yaml",
    )
    parser.add_argument(
        "--output",
        default="chronos2_m1_calendar_interactions.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/ablation_m1_calendar_interactions",
    )
    args = parser.parse_args()

    source = Path(args.base_config).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(
            f"Configuration M1 introuvable : {source}"
        )

    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    covariates = config["zones"]["FR"]["covariates"]
    required_aliases = (
        "fr_residual_load_fcst",
        "fr_nuclear_generation_fcst",
    )
    for alias in required_aliases:
        if alias not in covariates:
            raise KeyError(
                f"Covariable requise absente : {alias}"
            )
        covariates[alias]["enabled"] = True

        future = covariates[alias].setdefault("future", {})
        future["known_future"] = True
        strategies = list(future.get("strategies", []))
        if "oracle" not in strategies:
            strategies.append("oracle")
        future["strategies"] = strategies

    extensions = config.setdefault(
        "exogenous_extensions",
        {},
    )
    extensions["enabled"] = True
    extensions.setdefault(
        "rich_calendar",
        {},
    )["enabled"] = True
    extensions.setdefault(
        "neighbour_price_sources",
        {},
    )["enabled"] = False

    neighbour_prices = extensions.setdefault(
        "neighbour_prices",
        {},
    )
    neighbour_prices["enabled"] = False
    neighbour_prices["include_aggregates"] = False

    extensions.setdefault(
        "forecast_uncertainty",
        {},
    )["enabled"] = False

    extensions["calendar_interactions"] = {
        "enabled": True,
        "residual_load_alias": "fr_residual_load_fcst",
        "nuclear_alias": "fr_nuclear_generation_fcst",
        "primary_country": "FR",
        "neighbour_countries": ["DE", "BE", "NL", "ES"],
        "require_complete_future": True,
    }

    output_dir = str(args.output_dir)
    config.setdefault("output", {})["directory"] = output_dir

    report = config.setdefault("report", {})
    report["filename"] = "chronos2_m1_calendar_interactions.html"
    report["title"] = (
        "M1 - calendrier riche avec interactions PIT"
    )

    config.setdefault("order_signals", {})["output_dir"] = (
        "runs/order_signals_m1_calendar_interactions"
    )

    destination = Path(args.output).expanduser().resolve()
    with destination.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
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
