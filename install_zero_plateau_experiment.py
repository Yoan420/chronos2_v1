
#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


PREDICTION_COLUMNS = (
    "zero_plateau_probability",
    "zero_plateau_block_flag",
    "zero_plateau_block_probability",
    "zero_plateau_event_probability",
    "zero_plateau_block_position",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="chronos2_selected_core_structural_residual.yaml",
    )
    parser.add_argument(
        "--destination",
        default="chronos2_selected_core_zero_plateau.yaml",
    )
    parser.add_argument(
        "--predictions-file",
        default="data/derived/zero_plateau_predictions.csv.gz",
    )
    parser.add_argument("--zone", default="FR")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.config).expanduser().resolve()
    destination = Path(args.destination).expanduser().resolve()

    with source.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)

    result = copy.deepcopy(base)
    current_zero = result.get("zero_plateau", {}) or {}
    result["zero_plateau"] = {
        **current_zero,
        "enabled": True,
        "low": -3.0,
        "high": 3.0,
        "min_consecutive_hours": 3,
        "solar_start_hour": 8,
        "solar_end_hour": 19,
        "valley_band_gw": 3.0,
        "classifier_validation_days": 90,
        "decoder_max_length": 9,
        "decoder_min_mean_probability": 0.45,
        "decoder_minimum_score": 0.0,
        "soft_gate": {
            "enabled_for_postprocess": True,
            "probability_threshold": 0.55,
            "max_weight": 0.65,
        },
    }

    structural = result.setdefault("structural_model", {})
    structural["residual_price_alias"] = "milp_structural_price"
    structural.setdefault("residual_mode", {})["enabled"] = True

    covariates = result["zones"][args.zone].setdefault("covariates", {})
    descriptions = {
        "zero_plateau_probability": (
            "Probabilité CatBoost point-in-time qu'une heure appartienne "
            "à un plateau near-zero."
        ),
        "zero_plateau_block_flag": (
            "Indicateur du bloc start/end cohérent décodé."
        ),
        "zero_plateau_block_probability": (
            "Probabilité moyenne du bloc plateau décodé."
        ),
        "zero_plateau_event_probability": (
            "Probabilité maximale de plateau sur la journée."
        ),
        "zero_plateau_block_position": (
            "Position normalisée de l'heure dans le bloc plateau."
        ),
    }

    for alias in PREDICTION_COLUMNS:
        covariates[alias] = {
            "enabled": True,
            "source": "file",
            "file": args.predictions_file,
            "timestamp_col": "timestamp",
            "value_col": alias,
            "description": descriptions[alias],
            "fill_method": "none",
            "minimum_coverage": 0.90,
            "include_base_context": True,
            "future": {
                "known_future": True,
                "strategies": ["oracle"],
            },
        }

    output = result.setdefault("output", {})
    current = str(output.get("directory", "runs/chronos2")).rstrip("/\\")
    for suffix in (
        "_structural_residual",
        "_structural_scarcity_features",
        "_zero_plateau",
    ):
        if current.endswith(suffix):
            current = current[: -len(suffix)]
    output["directory"] = f"{current}_zero_plateau"

    report = result.setdefault("report", {})
    report["filename"] = "chronos2_zero_plateau.html"
    report["title"] = "Chronos-2 — C6 Zero Plateau"

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(
            result,
            handle,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        )

    print(f"Configuration C6 créée : {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
