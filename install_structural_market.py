#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

from chronos2_structural_market.config import (
    DEFAULT_FEATURE_ALIASES,
    default_structural_model_block,
)


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def feature_description(alias: str) -> str:
    descriptions = {
        "milp_structural_price": (
            "Prix structurel €/MWh issu du LP après fixation du commitment MILP."
        ),
        "milp_marginal_technology_code": (
            "Code numérique de la technologie marginale du modèle structurel."
        ),
        "milp_reserve_margin_gw": "Marge de capacité disponible du MILP en GW.",
        "milp_committed_thermal_gw": "Capacité thermique engagée par le MILP en GW.",
        "milp_online_units": "Nombre agrégé d'unités thermiques en ligne.",
        "milp_startups": "Nombre agrégé de démarrages sur l'heure.",
        "milp_startup_cost_eur": "Coût de démarrage engagé sur l'heure en euros.",
        "milp_scarcity_gw": "Variable de pénurie / load shedding du MILP en GW.",
        "milp_ramp_shadow_eur_mwh": "Valeur duale maximale des contraintes de rampe.",
        "milp_reserve_shadow_eur_mwh": "Valeur duale de la contrainte de réserve.",
    }
    return descriptions.get(alias, "Covariable issue du modèle structurel MILP.")


def add_structural_covariates(
    config: dict[str, Any],
    *,
    zone: str,
    residual_mode: bool,
    output_suffix: str,
) -> dict[str, Any]:
    result = copy.deepcopy(config)
    existing_block = result.get("structural_model") or {}
    block = deep_merge(default_structural_model_block(), existing_block)
    block.setdefault("residual_mode", {})
    block["residual_mode"]["enabled"] = bool(residual_mode)

    # IMPORTANT : les features structurelles n'existent pas avant la première
    # période couverte par les fondamentaux PIT. On les charge d'abord, puis
    # alignment.py tronque l'échantillon à la période continue valide.
    block.setdefault("alignment", {})
    block["alignment"].setdefault("enabled", True)
    block["alignment"].setdefault("required_aliases", list(
        block.get("feature_aliases", DEFAULT_FEATURE_ALIASES)
    ))
    result["structural_model"] = block

    zone_config = result["zones"][zone]
    covariates = zone_config.setdefault("covariates", {})
    output_file = str(block["output_file"])

    for alias in block.get("feature_aliases", DEFAULT_FEATURE_ALIASES):
        covariates[str(alias)] = {
            "enabled": True,
            "source": "file",
            "file": output_file,
            "timestamp_col": "timestamp",
            "value_col": str(alias),
            "description": feature_description(str(alias)),
            "fill_method": "none",
            # Ne pas jeter la colonne avant que l'alignement ne puisse
            # déterminer la vraie fenêtre commune.
            "minimum_coverage": 0.0,
            "include_base_context": True,
            "future": {
                "known_future": True,
                "strategies": ["oracle"],
            },
        }

    output = result.setdefault("output", {})
    current = str(output.get("directory", "runs/chronos2")).rstrip("/\\")
    for suffix in (
        "_structural_covariates",
        "_structural_residual",
    ):
        if current.endswith(suffix):
            current = current[: -len(suffix)]
    output["directory"] = f"{current}_{output_suffix}"

    report = result.setdefault("report", {})
    report["filename"] = f"chronos2_{output_suffix}.html"
    report["title"] = (
        "Chronos-2 — prix structurel MILP + prévision résiduelle"
        if residual_mode
        else "Chronos-2 — covariables structurelles MILP"
    )
    return result


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(
            payload,
            handle,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="chronos2_selected_core.yaml")
    parser.add_argument("--zone", default="FR")
    parser.add_argument(
        "--covariates-output",
        default="chronos2_selected_core_structural_covariates.yaml",
    )
    parser.add_argument(
        "--residual-output",
        default="chronos2_selected_core_structural_residual.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.config).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    if args.zone not in base.get("zones", {}):
        raise KeyError(f"Zone absente : {args.zone}")

    covariates = add_structural_covariates(
        base,
        zone=args.zone,
        residual_mode=False,
        output_suffix="structural_covariates",
    )
    residual = add_structural_covariates(
        base,
        zone=args.zone,
        residual_mode=True,
        output_suffix="structural_residual",
    )

    covariates_path = Path(args.covariates_output).expanduser().resolve()
    residual_path = Path(args.residual_output).expanduser().resolve()
    write_yaml(covariates_path, covariates)
    write_yaml(residual_path, residual)

    print(f"Configuration covariables : {covariates_path}")
    print(f"Configuration résiduelle  : {residual_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
