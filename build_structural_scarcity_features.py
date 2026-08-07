#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from chronos2_structural_market.scarcity_features import (
    C5_FEATURE_COLUMNS,
    ScarcityFeatureParams,
    build_scarcity_feature_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construit les features scarcity C5 à partir du MILP existant. "
            "Aucun recalcul Saturn/MILP."
        )
    )
    parser.add_argument(
        "--config",
        default=(
            "chronos2_selected_core_"
            "structural_covariates.yaml"
        ),
    )
    parser.add_argument(
        "--calibration",
        default=(
            "data/derived/"
            "scarcity_layer_calibration.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "data/derived/"
            "structural_scarcity_features_c5.csv.gz"
        ),
    )
    parser.add_argument(
        "--ewm-alpha",
        type=float,
        default=0.65,
    )
    return parser.parse_args()


def resolve_project_root(
    config_path: Path,
    config: dict,
) -> Path:
    root = Path(
        config.get("data", {}).get(
            "project_root",
            ".",
        )
    ).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    return root.resolve()


def main() -> int:
    args = parse_args()
    config_path = Path(
        args.config
    ).expanduser().resolve()

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    project_root = resolve_project_root(
        config_path,
        config,
    )
    structural_block = config.get(
        "structural_model",
        {},
    )
    structural_path = Path(
        structural_block.get(
            "output_file",
            "data/derived/"
            "structural_market_features.csv.gz",
        )
    )
    if not structural_path.is_absolute():
        structural_path = (
            project_root / structural_path
        ).resolve()

    calibration_path = Path(
        args.calibration
    ).expanduser()
    if not calibration_path.is_absolute():
        calibration_path = (
            project_root / calibration_path
        ).resolve()

    output_path = Path(
        args.output
    ).expanduser()
    if not output_path.is_absolute():
        output_path = (
            project_root / output_path
        ).resolve()

    if not structural_path.exists():
        raise FileNotFoundError(
            f"Features MILP absentes : {structural_path}"
        )
    if not calibration_path.exists():
        raise FileNotFoundError(
            "Calibration C4 absente : "
            f"{calibration_path}. "
            "Le run C4 doit avoir été calibré une fois."
        )

    structural = pd.read_csv(
        structural_path,
        compression="infer",
    )
    structural["timestamp"] = pd.to_datetime(
        structural["timestamp"],
        errors="coerce",
        utc=True,
    )
    structural = (
        structural
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(
            "timestamp",
            keep="last",
        )
        .reset_index(drop=True)
    )

    metadata = json.loads(
        calibration_path.read_text(
            encoding="utf-8"
        )
    )
    params = ScarcityFeatureParams.from_mapping(
        metadata["params"]
    )

    features = build_scarcity_feature_frame(
        structural,
        params,
        ewm_alpha=float(args.ewm_alpha),
    )

    result = pd.concat(
        [
            structural[["timestamp"]].reset_index(
                drop=True
            ),
            features.reset_index(drop=True),
        ],
        axis=1,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    result.to_csv(
        output_path,
        index=False,
        compression="gzip",
    )

    diagnostics = {
        "source_structural_file": str(
            structural_path
        ),
        "calibration_file": str(
            calibration_path
        ),
        "output_file": str(
            output_path
        ),
        "rows": int(len(result)),
        "first_timestamp": str(
            result["timestamp"].min()
        ),
        "last_timestamp": str(
            result["timestamp"].max()
        ),
        "ewm_alpha": float(
            args.ewm_alpha
        ),
        "feature_columns": list(
            C5_FEATURE_COLUMNS
        ),
        "coverage": {
            column: float(
                result[column].notna().mean()
            )
            for column in C5_FEATURE_COLUMNS
        },
    }

    diagnostics_path = output_path.with_name(
        "structural_scarcity_features_c5_metadata.json"
    )
    diagnostics_path.write_text(
        json.dumps(
            diagnostics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print("C5 — SCARCITY AS FEATURES")
    print("=" * 80)
    print(f"Source MILP : {structural_path}")
    print(f"Calibration : {calibration_path}")
    print(f"Sortie      : {output_path}")
    print(f"Lignes      : {len(result):,}")
    print(
        "Période     : "
        f"{result['timestamp'].min()} -> "
        f"{result['timestamp'].max()}"
    )
    print("\nCouverture")
    for column in C5_FEATURE_COLUMNS:
        print(
            f"  {column:<42} "
            f"{result[column].notna().mean():8.2%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
