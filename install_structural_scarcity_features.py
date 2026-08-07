#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from chronos2_structural_market.scarcity_features import (
    C5_FEATURE_COLUMNS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crée C5 : prix LP brut comme ancre du résidu, "
            "scarcity uniquement en covariables."
        )
    )
    parser.add_argument(
        "--config",
        default=(
            "chronos2_selected_core_"
            "structural_residual.yaml"
        ),
    )
    parser.add_argument(
        "--destination",
        default=(
            "chronos2_selected_core_"
            "structural_scarcity_features.yaml"
        ),
    )
    parser.add_argument(
        "--features-file",
        default=(
            "data/derived/"
            "structural_scarcity_features_c5.csv.gz"
        ),
    )
    parser.add_argument(
        "--zone",
        default="FR",
    )
    return parser.parse_args()


def description(alias: str) -> str:
    labels = {
        "milp_reserve_pressure": (
            "Pression de rareté liée à la marge de réserve."
        ),
        "milp_ramp_pressure": (
            "Pression normalisée des contraintes de rampe."
        ),
        "milp_startup_pressure": (
            "Pression normalisée des démarrages."
        ),
        "milp_scarcity_probability_feature": (
            "Score probabiliste de régime de rareté."
        ),
        "milp_scarcity_adder_feature": (
            "Amplitude indicative de scarcity, utilisée comme feature seulement."
        ),
        "milp_scarcity_probability_ewm": (
            "Probabilité de scarcity lissée causalement."
        ),
        "milp_scarcity_adder_ewm": (
            "Amplitude scarcity lissée causalement."
        ),
        "milp_scarcity_delta": (
            "Variation horaire du score de scarcity."
        ),
        "milp_reserve_pressure_delta": (
            "Variation horaire de la pression de réserve."
        ),
    }
    return labels[alias]


def main() -> int:
    args = parse_args()
    source = Path(
        args.config
    ).expanduser().resolve()
    destination = Path(
        args.destination
    ).expanduser().resolve()

    with source.open(
        "r",
        encoding="utf-8",
    ) as handle:
        base = yaml.safe_load(handle)

    result = copy.deepcopy(base)
    block = result.setdefault(
        "structural_model",
        {},
    )

    # Point central de C5 : revenir à l'ancre qui gagnait dans C3.
    block["residual_price_alias"] = (
        "milp_structural_price"
    )
    block.setdefault(
        "residual_mode",
        {},
    )["enabled"] = True

    experiment = block.setdefault(
        "scarcity_feature_experiment",
        {},
    )
    experiment.update(
        {
            "enabled": True,
            "reference_price": (
                "milp_structural_price"
            ),
            "scarcity_role": (
                "covariates_only"
            ),
            "features_file": (
                args.features_file
            ),
            "features": list(
                C5_FEATURE_COLUMNS
            ),
        }
    )

    zone = result["zones"][args.zone]
    covariates = zone.setdefault(
        "covariates",
        {},
    )

    for alias in C5_FEATURE_COLUMNS:
        covariates[alias] = {
            "enabled": True,
            "source": "file",
            "file": args.features_file,
            "timestamp_col": "timestamp",
            "value_col": alias,
            "description": description(alias),
            "fill_method": "none",
            "minimum_coverage": 0.0,
            "include_base_context": True,
            "future": {
                "known_future": True,
                "strategies": ["oracle"],
            },
        }

    # L'alignement doit contrôler aussi les features ajoutées.
    alignment = block.setdefault(
        "alignment",
        {},
    )
    alignment["enabled"] = True
    required = list(
        alignment.get(
            "required_aliases",
            [],
        )
        or []
    )
    for alias in C5_FEATURE_COLUMNS:
        if alias not in required:
            required.append(alias)
    if (
        "milp_structural_price"
        not in required
    ):
        required.insert(
            0,
            "milp_structural_price",
        )
    alignment[
        "required_aliases"
    ] = required

    output = result.setdefault(
        "output",
        {},
    )
    current = str(
        output.get(
            "directory",
            "runs/chronos2",
        )
    ).rstrip("/\\")
    for suffix in (
        "_structural_residual",
        "_structural_scarcity_residual",
        "_structural_scarcity_features",
    ):
        if current.endswith(suffix):
            current = current[
                : -len(suffix)
            ]
    output["directory"] = (
        f"{current}_structural_scarcity_features"
    )

    report = result.setdefault(
        "report",
        {},
    )
    report["filename"] = (
        "chronos2_structural_scarcity_features.html"
    )
    report["title"] = (
        "Chronos-2 — C5 scarcity as features"
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with destination.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        yaml.safe_dump(
            result,
            handle,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        )

    print(f"Configuration C5 créée : {destination}")
    print(
        "Ancre résiduelle        : "
        "milp_structural_price"
    )
    print(
        "Scarcity                : "
        "covariables uniquement"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
