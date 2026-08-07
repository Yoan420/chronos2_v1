#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="chronos2_selected_core_structural_covariates.yaml",
    )
    parser.add_argument("--zone", default="FR")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    project_root = Path(
        config.get("data", {}).get("project_root", ".")
    )
    if not project_root.is_absolute():
        project_root = (config_path.parent / project_root).resolve()

    block = config["structural_model"]
    features_path = (
        project_root / block["output_file"]
    ).resolve()
    diagnostics_path = (
        project_root / block["diagnostics_file"]
    ).resolve()

    if not features_path.exists():
        raise FileNotFoundError(features_path)

    features = pd.read_csv(
        features_path,
        compression="infer",
    )
    features["timestamp"] = pd.to_datetime(
        features["timestamp"],
        errors="coerce",
        utc=True,
    )
    features = features.dropna(subset=["timestamp"])
    feature_aliases = [
        str(value)
        for value in block["feature_aliases"]
    ]
    missing_columns = [
        col for col in feature_aliases
        if col not in features.columns
    ]
    if missing_columns:
        raise KeyError(
            "Colonnes MILP absentes : "
            + ", ".join(missing_columns)
        )

    complete = features[
        feature_aliases
    ].notna().all(axis=1)

    print("=" * 80)
    print("VALIDATION STRUCTURAL MARKET FEATURES")
    print("=" * 80)
    print(f"Fichier             : {features_path}")
    print(f"Lignes              : {len(features):,}")
    print(
        "Période             : "
        f"{features['timestamp'].min()} -> "
        f"{features['timestamp'].max()}"
    )
    print(
        "Lignes complètes    : "
        f"{int(complete.sum()):,} / {len(features):,}"
    )

    if diagnostics_path.exists():
        diagnostics = pd.read_csv(diagnostics_path)
        print("\nDiagnostics MILP")
        print(f"Journées totales    : {len(diagnostics):,}")
        if "success" in diagnostics:
            success = (
                diagnostics["success"]
                .astype(str)
                .str.lower()
                .isin(["true", "1"])
            )
            print(f"Journées réussies   : {int(success.sum()):,}")
            print(f"Journées échouées   : {int((~success).sum()):,}")

        if "error_type" in diagnostics:
            errors = (
                diagnostics["error_type"]
                .fillna("SUCCESS")
                .value_counts(dropna=False)
            )
            print("\nRépartition error_type :")
            print(errors.to_string())

        if "residual_load_imputed_hours" in diagnostics:
            repaired = pd.to_numeric(
                diagnostics["residual_load_imputed_hours"],
                errors="coerce",
            ).fillna(0)
            print(
                "\nHeures residual_load_gw imputées : "
                f"{int(repaired.sum()):,}"
            )
            print(
                "Journées avec imputation          : "
                f"{int((repaired > 0).sum()):,}"
            )

        failed = diagnostics.loc[
            diagnostics.get(
                "success",
                pd.Series(False, index=diagnostics.index),
            ).astype(str).str.lower().isin(
                ["false", "0"]
            )
        ]
        if not failed.empty:
            columns = [
                col
                for col in ("day", "error_type", "error")
                if col in failed.columns
            ]
            print("\n10 derniers échecs :")
            print(
                failed[columns]
                .tail(10)
                .to_string(index=False)
            )

    build_inputs = (
        project_root
        / "runs"
        / "structural_feature_build"
        / args.zone.lower()
        / "aligned_inputs.csv.gz"
    )
    if build_inputs.exists():
        target = pd.read_csv(
            build_inputs,
            compression="infer",
            usecols=["timestamp", "target"],
        )
        target["timestamp"] = pd.to_datetime(
            target["timestamp"],
            errors="coerce",
            utc=True,
        )
        target = target.dropna(
            subset=["timestamp", "target"]
        )
        merged = target[["timestamp"]].merge(
            features[["timestamp", *feature_aliases]],
            on="timestamp",
            how="left",
        )
        coverage = float(
            merged[feature_aliases]
            .notna()
            .all(axis=1)
            .mean()
        )
        print(
            "\nCouverture vs historique cible du build : "
            f"{coverage:.2%}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
