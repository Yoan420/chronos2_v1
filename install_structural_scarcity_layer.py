#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


NEW_ALIASES = (
    "milp_structural_price_raw",
    "milp_scarcity_probability",
    "milp_scarcity_adder",
    "milp_structural_price_adjusted",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crée le YAML C4 = prix structurel ajusté scarcity "
            "+ résidu Chronos."
        )
    )
    parser.add_argument(
        "--config",
        default="chronos2_selected_core_structural_residual.yaml",
    )
    parser.add_argument(
        "--destination",
        default=(
            "chronos2_selected_core_"
            "structural_scarcity_residual.yaml"
        ),
    )
    parser.add_argument("--zone", default="FR")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.config).expanduser().resolve()
    destination = Path(
        args.destination
    ).expanduser().resolve()

    with source.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    result = copy.deepcopy(config)
    block = result.setdefault(
        "structural_model",
        {},
    )
    output_file = str(
        block.get(
            "output_file",
            "data/derived/structural_market_features.csv.gz",
        )
    )

    block["residual_price_alias"] = (
        "milp_structural_price_adjusted"
    )
    block.setdefault(
        "residual_mode",
        {},
    )["enabled"] = True

    feature_aliases = list(
        block.get(
            "feature_aliases",
            [],
        )
        or []
    )
    for alias in NEW_ALIASES:
        if alias not in feature_aliases:
            feature_aliases.append(alias)
    block["feature_aliases"] = feature_aliases

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
    if "milp_structural_price_adjusted" not in required:
        required.append(
            "milp_structural_price_adjusted"
        )
    alignment["required_aliases"] = required

    scarcity = block.setdefault(
        "scarcity_layer",
        {},
    )
    scarcity.update(
        {
            "enabled": True,
            "method": "scarcity_adder_v1",
            "calibration_file": (
                "data/derived/scarcity_layer_calibration.json"
            ),
            "raw_price_alias": "milp_structural_price",
            "adjusted_price_alias": (
                "milp_structural_price_adjusted"
            ),
        }
    )

    zone = result["zones"][args.zone]
    covariates = zone.setdefault(
        "covariates",
        {},
    )

    descriptions = {
        "milp_structural_price_raw": (
            "Prix LP structurel brut, avant scarcity adder."
        ),
        "milp_scarcity_probability": (
            "Probabilité/proxy normalisé de régime de rareté."
        ),
        "milp_scarcity_adder": (
            "Prime de rareté ajoutée au prix LP structurel."
        ),
        "milp_structural_price_adjusted": (
            "Prix structurel ajusté = prix LP + scarcity adder."
        ),
    }

    for alias in NEW_ALIASES:
        covariates[alias] = {
            "enabled": True,
            "source": "file",
            "file": output_file,
            "timestamp_col": "timestamp",
            "value_col": alias,
            "description": descriptions[alias],
            "fill_method": "none",
            "minimum_coverage": 0.0,
            "include_base_context": True,
            "future": {
                "known_future": True,
                "strategies": ["oracle"],
            },
        }

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
    ):
        if current.endswith(suffix):
            current = current[: -len(suffix)]
    output["directory"] = (
        f"{current}_structural_scarcity_residual"
    )

    report = result.setdefault(
        "report",
        {},
    )
    report["filename"] = (
        "chronos2_structural_scarcity_residual.html"
    )
    report["title"] = (
        "Chronos-2 — prix MILP + scarcity adder + résidu"
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

    print(f"Configuration C4 créée : {destination}")
    print(
        "Prix résiduel de référence : "
        "milp_structural_price_adjusted"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
