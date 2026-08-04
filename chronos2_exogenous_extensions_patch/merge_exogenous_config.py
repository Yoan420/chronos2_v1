#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


PRICE_COUNTRIES = {
    "DE": ("de_price_da", "power.price.everyday.de.hourly.eurmwh"),
    "BE": ("be_price_da", "power.price.everyday.be.hourly.eurmwh"),
    "NL": ("nl_price_da", "power.price.everyday.nl.hourly.eurmwh"),
    "ES": ("es_price_da", "power.price.everyday.es.hourly.eurmwh"),
}

UNCERTAINTY_FILES = {
    "fr_residual_load_revision_std": "fr_residual_load_revision_std.parquet",
    "fr_residual_load_revision_abs_delta": (
        "fr_residual_load_revision_abs_delta.parquet"
    ),
    "fr_residual_load_revision_age_hours": (
        "fr_residual_load_revision_age_hours.parquet"
    ),
    "fr_nuclear_revision_std": "fr_nuclear_revision_std.parquet",
    "fr_nuclear_revision_abs_delta": (
        "fr_nuclear_revision_abs_delta.parquet"
    ),
    "neighbour_residual_revision_std_mean": (
        "neighbour_residual_revision_std_mean.parquet"
    ),
    "neighbour_residual_revision_std_max": (
        "neighbour_residual_revision_std_max.parquet"
    ),
    "neighbour_residual_revision_abs_delta_mean": (
        "neighbour_residual_revision_abs_delta_mean.parquet"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="chronos2_inputs_asof_jplus1_regime_order_signals.yaml",
    )
    parser.add_argument(
        "--output",
        default="chronos2_inputs_extended_exogenous.yaml",
    )
    args = parser.parse_args()

    source = Path(args.input).expanduser().resolve()
    with source.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data = config.setdefault("data", {})
    pit_files = data.setdefault("pit_files", {})
    pit_files.update(UNCERTAINTY_FILES)

    config["exogenous_extensions"] = {
        "enabled": True,
        "neighbour_price_sources": {
            "enabled": True,
            "output_dir": "data/derived/neighbour_prices",
            "countries": {
                country: {"alias": alias, "series": series}
                for country, (alias, series) in PRICE_COUNTRIES.items()
            },
        },
        "neighbour_prices": {
            "enabled": True,
            "aliases": {
                country: alias
                for country, (alias, _) in PRICE_COUNTRIES.items()
            },
            "include_aggregates": True,
        },
        "rich_calendar": {
            "enabled": True,
            "primary_country": "FR",
            "countries": ["FR", "DE", "BE", "NL", "ES"],
        },
        "forecast_uncertainty": {
            "enabled": True,
            "max_revisions": 6,
            "metrics": [
                "revision_std",
                "revision_abs_delta",
                "revision_age_hours",
            ],
            "sources": {
                "fr_residual_load_fcst": {
                    "output_prefix": "fr_residual_load",
                    "metrics": [
                        "revision_std",
                        "revision_abs_delta",
                        "revision_age_hours",
                    ],
                },
                "fr_nuclear_generation_fcst": {
                    "output_prefix": "fr_nuclear",
                    "metrics": [
                        "revision_std",
                        "revision_abs_delta",
                    ],
                },
                "de_residual_load_fcst": {
                    "output_prefix": "de_residual_load"
                },
                "be_residual_load_fcst": {
                    "output_prefix": "be_residual_load"
                },
                "nl_residual_load_fcst": {
                    "output_prefix": "nl_residual_load"
                },
                "es_residual_load_fcst": {
                    "output_prefix": "es_residual_load"
                },
            },
            "aggregate_groups": {
                "neighbour_residual": {
                    "sources": [
                        "de_residual_load_fcst",
                        "be_residual_load_fcst",
                        "nl_residual_load_fcst",
                        "es_residual_load_fcst",
                    ],
                    "metrics": [
                        "revision_std_mean",
                        "revision_std_max",
                        "revision_abs_delta_mean",
                    ],
                }
            },
        },
    }

    covariates = config["zones"]["FR"].setdefault("covariates", {})
    for country, (alias, _) in PRICE_COUNTRIES.items():
        covariates[alias] = {
            "enabled": True,
            "source": "file",
            "file": f"data/derived/neighbour_prices/{alias}.csv.gz",
            "timestamp_col": "timestamp",
            "value_col": "value",
            "description": f"Prix Day-Ahead {country}",
            "fill_method": "none",
            "minimum_coverage": 0.95,
            "future": {
                "known_future": False,
                "strategies": ["lag24", "lag168"],
            },
        }

    descriptions = {
        "fr_residual_load_revision_std": (
            "Dispersion des dernières révisions de charge résiduelle FR"
        ),
        "fr_residual_load_revision_abs_delta": (
            "Amplitude de la dernière révision de charge résiduelle FR"
        ),
        "fr_residual_load_revision_age_hours": (
            "Ancienneté du dernier forecast de charge résiduelle FR"
        ),
        "fr_nuclear_revision_std": (
            "Dispersion des révisions du forecast nucléaire FR"
        ),
        "fr_nuclear_revision_abs_delta": (
            "Amplitude de la dernière révision nucléaire FR"
        ),
        "neighbour_residual_revision_std_mean": (
            "Incertitude moyenne des charges résiduelles voisines"
        ),
        "neighbour_residual_revision_std_max": (
            "Incertitude maximale des charges résiduelles voisines"
        ),
        "neighbour_residual_revision_abs_delta_mean": (
            "Révision absolue moyenne des fondamentaux voisins"
        ),
    }
    for alias, description in descriptions.items():
        covariates[alias] = {
            "enabled": True,
            "source": "pit_parquet",
            "description": description,
            "fill_method": "none",
            "minimum_coverage": (
                0.05 if alias.startswith("fr_nuclear") else 0.20
            ),
            "future": {
                "known_future": True,
                "strategies": ["oracle"],
            },
        }

    output = Path(args.output).expanduser().resolve()
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(
            config,
            handle,
            sort_keys=False,
            allow_unicode=True,
            width=100,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
