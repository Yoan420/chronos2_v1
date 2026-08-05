from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


RAW_NEIGHBOUR_PRICE_ALIASES = [
    "de_price_da",
    "be_price_da",
    "nl_price_da",
    "es_price_da",
]

UNCERTAINTY_ALIASES = [
    "fr_residual_load_revision_std",
    "fr_residual_load_revision_abs_delta",
    "fr_residual_load_revision_age_hours",
    "fr_nuclear_revision_std",
    "fr_nuclear_revision_abs_delta",
    "neighbour_residual_revision_std_mean",
    "neighbour_residual_revision_std_max",
    "neighbour_residual_revision_abs_delta_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="chronos2_m1_calendar.yaml",
    )
    parser.add_argument(
        "--interconnection-map",
        default="interconnection_series.yaml",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return value or {}


def save_yaml(path: Path, config: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config,
            handle,
            allow_unicode=True,
            sort_keys=False,
        )


def disable_non_m1_extensions(config: dict[str, Any]) -> None:
    extensions = config.setdefault("exogenous_extensions", {})
    extensions["enabled"] = True
    extensions.setdefault("rich_calendar", {})["enabled"] = True
    extensions.setdefault(
        "neighbour_price_sources", {}
    )["enabled"] = False

    neighbour = extensions.setdefault("neighbour_prices", {})
    neighbour["enabled"] = False
    neighbour["include_aggregates"] = False

    covariates = config["zones"]["FR"]["covariates"]
    for alias in RAW_NEIGHBOUR_PRICE_ALIASES:
        if alias in covariates:
            covariates[alias]["enabled"] = False


def configure_uncertainty(base: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(base)
    disable_non_m1_extensions(config)

    extensions = config["exogenous_extensions"]
    extensions.setdefault(
        "forecast_uncertainty", {}
    )["enabled"] = True
    extensions.setdefault(
        "interconnection_capacities", {}
    )["enabled"] = False

    covariates = config["zones"]["FR"]["covariates"]
    for alias in UNCERTAINTY_ALIASES:
        if alias not in covariates:
            raise KeyError(
                f"Covariable d'incertitude absente de la base : {alias}"
            )
        covariates[alias]["enabled"] = True

    config.setdefault("output", {})["directory"] = (
        "runs/ablation_m1_uncertainty_only"
    )
    config.setdefault("report", {})["filename"] = (
        "chronos2_m1_uncertainty_only.html"
    )
    config["report"]["title"] = (
        "M1 + incertitude des forecasts uniquement"
    )
    config.setdefault("order_signals", {})["output_dir"] = (
        "runs/order_signals_m1_uncertainty_only"
    )
    return config


def valid_series_mapping(mapping: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in mapping.get("series", []) or []:
        raw = raw or {}
        alias = str(raw.get("alias", "")).strip()
        series = str(raw.get("series", "")).strip()
        direction = str(raw.get("direction", "")).strip().lower()

        if (
            not alias
            or not series
            or "REPLACE" in series.upper()
            or direction not in {"import", "export"}
        ):
            continue

        rows.append(
            {
                "alias": alias,
                "series": series,
                "direction": direction,
                "description": str(
                    raw.get("description", alias)
                ),
            }
        )
    return rows


def configure_interconnections(
    base: dict[str, Any],
    mapping: dict[str, Any],
) -> dict[str, Any]:
    rows = valid_series_mapping(mapping)
    if not rows:
        raise ValueError(
            "Aucun identifiant Saturn valide dans "
            "interconnection_series.yaml."
        )

    config = copy.deepcopy(base)
    disable_non_m1_extensions(config)

    extensions = config["exogenous_extensions"]
    extensions.setdefault(
        "forecast_uncertainty", {}
    )["enabled"] = False

    covariates = config["zones"]["FR"]["covariates"]
    for alias in UNCERTAINTY_ALIASES:
        if alias in covariates:
            covariates[alias]["enabled"] = False

    import_aliases: list[str] = []
    export_aliases: list[str] = []

    for row in rows:
        alias = row["alias"]
        direction = row["direction"]
        covariates[alias] = {
            "enabled": True,
            "source": "saturn",
            "series": row["series"],
            "description": row["description"],
            "fill_method": "none",
            "minimum_coverage": float(
                mapping.get("minimum_coverage", 0.50)
            ),
            "future": {
                "known_future": False,
                "strategies": ["lag24", "lag168"],
            },
        }
        if direction == "import":
            import_aliases.append(alias)
        else:
            export_aliases.append(alias)

    extensions["interconnection_capacities"] = {
        "enabled": True,
        "aliases": {
            "import": import_aliases,
            "export": export_aliases,
        },
        "derived_lags": [24, 168],
    }

    config.setdefault("output", {})["directory"] = (
        "runs/ablation_m1_interconnection_only"
    )
    config.setdefault("report", {})["filename"] = (
        "chronos2_m1_interconnection_only.html"
    )
    config["report"]["title"] = (
        "M1 + capacités d'interconnexion uniquement"
    )
    config.setdefault("order_signals", {})["output_dir"] = (
        "runs/order_signals_m1_interconnection_only"
    )
    return config


def main() -> int:
    args = parse_args()
    base_path = Path(args.base_config)
    mapping_path = Path(args.interconnection_map)

    if not base_path.exists():
        raise FileNotFoundError(
            f"Configuration M1 introuvable : {base_path.resolve()}"
        )
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Mapping interconnexion introuvable : "
            f"{mapping_path.resolve()}"
        )

    base = load_yaml(base_path)
    mapping = load_yaml(mapping_path)

    uncertainty = configure_uncertainty(base)
    interconnection = configure_interconnections(base, mapping)

    save_yaml(
        Path("chronos2_m1_uncertainty_only.yaml"),
        uncertainty,
    )
    save_yaml(
        Path("chronos2_m1_interconnection_only.yaml"),
        interconnection,
    )

    print("Créé : chronos2_m1_uncertainty_only.yaml")
    print("Créé : chronos2_m1_interconnection_only.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
