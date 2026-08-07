#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

from chronos2_structural_market.scarcity import (
    apply_scarcity_layer,
    fit_scarcity_layer,
    metric_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibre une couche de scarcity pricing sur le prix LP structurel "
            "et ajoute les colonnes ajustées au fichier de features existant."
        )
    )
    parser.add_argument(
        "--config",
        default="chronos2_selected_core_structural_covariates.yaml",
    )
    parser.add_argument("--zone", default="FR")
    parser.add_argument(
        "--features",
        default=None,
        help="Chemin optionnel vers structural_market_features.csv.gz.",
    )
    parser.add_argument(
        "--target-file",
        default=None,
        help="aligned_inputs.csv.gz produit par le build structurel.",
    )
    parser.add_argument("--holdout-days", type=int, default=None)
    parser.add_argument("--extreme-threshold", type=float, default=150.0)
    parser.add_argument("--extreme-weight", type=float, default=2.0)
    parser.add_argument("--bias-penalty", type=float, default=0.20)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
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
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    project_root = resolve_project_root(
        config_path,
        config,
    )
    structural = config.get(
        "structural_model",
        {},
    )

    features_path = (
        Path(args.features).expanduser().resolve()
        if args.features
        else (
            project_root
            / structural.get(
                "output_file",
                "data/derived/structural_market_features.csv.gz",
            )
        ).resolve()
    )
    target_path = (
        Path(args.target_file).expanduser().resolve()
        if args.target_file
        else (
            project_root
            / "runs"
            / "structural_feature_build"
            / args.zone.lower()
            / "aligned_inputs.csv.gz"
        ).resolve()
    )

    if not features_path.exists():
        raise FileNotFoundError(
            f"Features structurelles absentes : {features_path}"
        )
    if not target_path.exists():
        raise FileNotFoundError(
            f"Cible du build structurel absente : {target_path}"
        )

    features = pd.read_csv(
        features_path,
        compression="infer",
    )
    target = pd.read_csv(
        target_path,
        compression="infer",
        usecols=["timestamp", "target"],
    )

    features["timestamp"] = pd.to_datetime(
        features["timestamp"],
        errors="coerce",
        utc=True,
    )
    target["timestamp"] = pd.to_datetime(
        target["timestamp"],
        errors="coerce",
        utc=True,
    )
    target["target"] = pd.to_numeric(
        target["target"],
        errors="coerce",
    )

    features = features.dropna(
        subset=["timestamp"]
    ).sort_values("timestamp")
    target = target.dropna(
        subset=["timestamp", "target"]
    ).sort_values("timestamp")

    merged = features.merge(
        target.rename(
            columns={"target": "actual_price"}
        ),
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(
            "Aucune intersection entre features structurelles et prix réels."
        )

    holdout_days = (
        int(args.holdout_days)
        if args.holdout_days is not None
        else int(
            config.get(
                "backtest",
                {},
            ).get(
                "windows",
                60,
            )
        )
    )
    holdout_days = max(1, holdout_days)

    last_actual = merged["timestamp"].max()
    holdout_start = (
        last_actual.normalize()
        - pd.Timedelta(days=holdout_days - 1)
    )

    train = merged.loc[
        merged["timestamp"] < holdout_start
    ].copy()
    holdout = merged.loc[
        merged["timestamp"] >= holdout_start
    ].copy()

    if train.empty or holdout.empty:
        raise ValueError(
            "Split train/holdout impossible : "
            f"train={len(train)}, holdout={len(holdout)}."
        )

    params = fit_scarcity_layer(
        train,
        extreme_threshold=args.extreme_threshold,
        extreme_weight=args.extreme_weight,
        bias_penalty=args.bias_penalty,
        seed=args.seed,
        maxiter=args.maxiter,
    )

    # Application sur TOUT le fichier structurel, y compris l'horizon live.
    enriched = apply_scarcity_layer(
        features,
        params,
    )

    # Sauvegarde une fois le fichier brut avant scarcity.
    backup = features_path.with_name(
        "structural_market_features_pre_scarcity.csv.gz"
    )
    if not backup.exists():
        shutil.copy2(
            features_path,
            backup,
        )

    enriched.to_csv(
        features_path,
        index=False,
        compression="gzip",
    )

    train_eval = apply_scarcity_layer(
        train,
        params,
    )
    holdout_eval = apply_scarcity_layer(
        holdout,
        params,
    )

    metadata = {
        "method": "scarcity_adder_v1",
        "formula": (
            "P_adjusted = P_LP + probability * "
            "max(alpha*exp(-max(reserve_margin,0)/tau) "
            "+ beta_ramp*ramp_pressure "
            "+ gamma_startup*startup_pressure, 0)"
        ),
        "config": {
            "holdout_days": holdout_days,
            "extreme_threshold": args.extreme_threshold,
            "extreme_weight": args.extreme_weight,
            "bias_penalty": args.bias_penalty,
            "maxiter": args.maxiter,
            "seed": args.seed,
        },
        "train": {
            "start": str(train["timestamp"].min()),
            "end": str(train["timestamp"].max()),
            "n": int(len(train)),
            "raw": metric_summary(
                train,
                "milp_structural_price",
                extreme_threshold=args.extreme_threshold,
            ),
            "adjusted": metric_summary(
                train_eval,
                "milp_structural_price_adjusted",
                extreme_threshold=args.extreme_threshold,
            ),
        },
        "holdout": {
            "start": str(holdout["timestamp"].min()),
            "end": str(holdout["timestamp"].max()),
            "n": int(len(holdout)),
            "raw": metric_summary(
                holdout,
                "milp_structural_price",
                extreme_threshold=args.extreme_threshold,
            ),
            "adjusted": metric_summary(
                holdout_eval,
                "milp_structural_price_adjusted",
                extreme_threshold=args.extreme_threshold,
            ),
        },
        "params": params.to_dict(),
        "features_file": str(features_path),
        "backup_file": str(backup),
    }

    metadata_path = features_path.with_name(
        "scarcity_layer_calibration.json"
    )
    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    holdout_output = features_path.with_name(
        "scarcity_layer_holdout_predictions.csv.gz"
    )
    holdout_eval[
        [
            "timestamp",
            "actual_price",
            "milp_structural_price",
            "milp_scarcity_probability",
            "milp_scarcity_adder",
            "milp_structural_price_adjusted",
        ]
    ].to_csv(
        holdout_output,
        index=False,
        compression="gzip",
    )

    print("=" * 80)
    print("SCARCITY PRICING LAYER")
    print("=" * 80)
    print(f"Train   : {train['timestamp'].min()} -> {train['timestamp'].max()}")
    print(f"Holdout : {holdout['timestamp'].min()} -> {holdout['timestamp'].max()}")
    print("\nParamètres calibrés")
    for key, value in params.to_dict().items():
        print(f"  {key:<20} {value:12.6f}")

    print("\nHoldout")
    raw = metadata["holdout"]["raw"]
    adjusted = metadata["holdout"]["adjusted"]
    for metric in ("mae", "rmse", "bias", "extreme_mae"):
        a = raw.get(metric)
        b = adjusted.get(metric)
        if a is None or b is None:
            continue
        gain = (
            100.0 * (a - b) / abs(a)
            if metric != "bias" and a != 0
            else float("nan")
        )
        print(
            f"  {metric:<14} raw={a:9.4f} "
            f"adjusted={b:9.4f} "
            f"gain={gain:8.3f}%"
        )

    print(f"\nFeatures enrichies : {features_path}")
    print(f"Calibration        : {metadata_path}")
    print(f"Holdout            : {holdout_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
