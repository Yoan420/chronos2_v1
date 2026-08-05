from __future__ import annotations

from pathlib import Path
import copy
import yaml


RAW_NEIGHBOUR_PRICES = [
    "de_price_da",
    "be_price_da",
    "nl_price_da",
    "es_price_da",
]

UNCERTAINTY_COVARIATES = [
    "fr_residual_load_revision_std",
    "fr_residual_load_revision_abs_delta",
    "fr_residual_load_revision_age_hours",
    "fr_nuclear_revision_std",
    "fr_nuclear_revision_abs_delta",
    "neighbour_residual_revision_std_mean",
    "neighbour_residual_revision_std_max",
    "neighbour_residual_revision_abs_delta_mean",
]


def create_config(
    base: dict,
    *,
    name: str,
    title: str,
    calendar: bool,
    neighbour_prices: bool,
    spreads: bool,
    aggregates: bool,
    uncertainty: bool,
) -> None:
    config = copy.deepcopy(base)

    extensions = config.setdefault("exogenous_extensions", {})
    extensions["enabled"] = True
    extensions.setdefault("rich_calendar", {})["enabled"] = calendar
    extensions.setdefault(
        "neighbour_price_sources", {}
    )["enabled"] = neighbour_prices

    neighbour = extensions.setdefault("neighbour_prices", {})
    neighbour["enabled"] = spreads
    neighbour["include_aggregates"] = aggregates

    extensions.setdefault(
        "forecast_uncertainty", {}
    )["enabled"] = uncertainty

    covariates = config["zones"]["FR"]["covariates"]

    for alias in RAW_NEIGHBOUR_PRICES:
        if alias in covariates:
            covariates[alias]["enabled"] = neighbour_prices

    for alias in UNCERTAINTY_COVARIATES:
        if alias in covariates:
            covariates[alias]["enabled"] = uncertainty

    config.setdefault("output", {})["directory"] = f"runs/ablation_{name}"

    report = config.setdefault("report", {})
    report["filename"] = f"chronos2_{name}.html"
    report["title"] = title

    config.setdefault("order_signals", {})["output_dir"] = (
        f"runs/order_signals_{name}"
    )

    output = Path(f"chronos2_{name}.yaml")
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config,
            handle,
            allow_unicode=True,
            sort_keys=False,
        )
    print(f"Configuration créée : {output}")


def main() -> int:
    base_path = Path("chronos2_inputs_extended_exogenous.yaml")
    if not base_path.exists():
        raise FileNotFoundError(
            f"Configuration de base introuvable : {base_path.resolve()}"
        )

    with base_path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)

    create_config(
        base,
        name="m1_calendar",
        title="M1 - fondamentaux et calendrier riche",
        calendar=True,
        neighbour_prices=False,
        spreads=False,
        aggregates=False,
        uncertainty=False,
    )
    create_config(
        base,
        name="m2_neighbour_prices",
        title="M2 - calendrier riche et prix voisins",
        calendar=True,
        neighbour_prices=True,
        spreads=False,
        aggregates=False,
        uncertainty=False,
    )
    create_config(
        base,
        name="m3_neighbour_spreads",
        title="M3 - prix, spreads et agrégats voisins",
        calendar=True,
        neighbour_prices=True,
        spreads=True,
        aggregates=True,
        uncertainty=False,
    )
    create_config(
        base,
        name="m4_forecast_uncertainty",
        title="M4 - prix voisins, spreads et incertitude PIT",
        calendar=True,
        neighbour_prices=True,
        spreads=True,
        aggregates=True,
        uncertainty=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
