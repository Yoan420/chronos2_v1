
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import run_chronos2_extended_exogenous as extended

from chronos2_modular.common import (
    build_zone_configs,
    deep_get,
    load_yaml,
    resolve_path,
)
from chronos2_structural_market.alignment import (
    make_prepare_zone_data_with_structural_alignment,
)
from chronos2_zero_plateau.decoder import decode_probability_days
from chronos2_zero_plateau.features import build_plateau_features
from chronos2_zero_plateau.labels import mark_near_zero_plateaus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="chronos2_selected_core_structural_residual.yaml",
    )
    parser.add_argument("--zone", default="FR")
    parser.add_argument(
        "--output",
        default="data/derived/zero_plateau_predictions.csv.gz",
    )
    parser.add_argument(
        "--model-output",
        default="data/models/zero_plateau_catboost.cbm",
    )
    parser.add_argument(
        "--metadata-output",
        default="data/derived/zero_plateau_model_metadata.json",
    )
    return parser.parse_args()


def load_structural_raw(
    config: dict,
    config_path: Path,
    timezone: str,
) -> pd.DataFrame:
    project_root = resolve_path(
        deep_get(config, "data.project_root", "."),
        config_path.parent,
    )
    structural_path = resolve_path(
        deep_get(
            config,
            "structural_model.output_file",
            "data/derived/structural_market_features.csv.gz",
        ),
        project_root,
    )
    frame = pd.read_csv(structural_path, compression="infer")
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        errors="coerce",
        utc=True,
    )
    frame = frame.dropna(subset=["timestamp"])
    frame["timestamp"] = frame["timestamp"].dt.tz_convert(timezone)
    return (
        frame.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .set_index("timestamp")
    )


def classifier_metrics(actual, probability) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    y = np.asarray(actual, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    result = {
        "n": int(len(y)),
        "positive_hours": int(y.sum()),
        "positive_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
    }
    if len(np.unique(y)) == 2:
        result["roc_auc"] = float(roc_auc_score(y, p))
        result["average_precision"] = float(
            average_precision_score(y, p)
        )
    return result


def main() -> int:
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError(
            "CatBoost n'est pas installé. "
            "Exécute : python -m pip install catboost"
        ) from exc

    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)

    zone_configs = build_zone_configs(
        config,
        [args.zone],
        None,
        None,
    )
    if len(zone_configs) != 1:
        raise ValueError("Une seule zone est attendue.")
    zone = zone_configs[0]

    project_root = resolve_path(
        deep_get(config, "data.project_root", "."),
        config_path.parent,
    )
    build_output = (
        project_root
        / "runs"
        / "zero_plateau_feature_build"
        / args.zone.lower()
    )

    prepare = make_prepare_zone_data_with_structural_alignment(
        extended.runner.prepare_zone_data
    )
    data = prepare(
        zone,
        config,
        config_path.parent,
        False,
        build_output,
    )

    full_index = pd.DatetimeIndex(data.model_context_covariates.index)
    history_index = pd.DatetimeIndex(data.target.index)
    last_history = history_index[-1]

    settings = deep_get(config, "zero_plateau", {}) or {}
    low = float(settings.get("low", -3.0))
    high = float(settings.get("high", 3.0))
    min_hours = int(settings.get("min_consecutive_hours", 3))
    solar_start = int(settings.get("solar_start_hour", 8))
    solar_end = int(settings.get("solar_end_hour", 19))
    valley_band = float(settings.get("valley_band_gw", 3.0))

    backtest_days = int(deep_get(config, "backtest.windows", 60))
    context_length = int(deep_get(config, "model.context_length", 512))
    context_buffer_days = int(
        settings.get(
            "context_buffer_days",
            math.ceil(context_length / 24) + 7,
        )
    )
    validation_days = int(
        settings.get("classifier_validation_days", 90)
    )

    last_day = last_history.normalize()
    evaluation_start = last_day - pd.Timedelta(days=backtest_days - 1)
    oos_start = evaluation_start - pd.Timedelta(days=context_buffer_days)

    structural_raw = load_structural_raw(
        config,
        config_path,
        zone.timezone,
    )
    train_mask_full = full_index < oos_start
    features, aliases, scales = build_plateau_features(
        data.model_context_covariates,
        structural_raw,
        train_mask=train_mask_full,
        solar_start_hour=solar_start,
        solar_end_hour=solar_end,
        valley_band_gw=valley_band,
    )

    labelled = mark_near_zero_plateaus(
        data.target,
        low=low,
        high=high,
        min_consecutive_hours=min_hours,
        solar_start_hour=solar_start,
        solar_end_hour=solar_end,
    )
    y = labelled["plateau_label"].reindex(history_index).astype(int)
    historical_features = features.reindex(history_index)

    initial_train_mask = history_index < oos_start
    X_pre = historical_features.loc[initial_train_mask]
    y_pre = y.loc[initial_train_mask]

    # Écarte les colonnes presque entièrement manquantes ou constantes
    # sur le train. CatBoost voit ainsi uniquement les signaux disponibles.
    feature_columns = []
    for column in X_pre.columns:
        numeric = pd.to_numeric(X_pre[column], errors="coerce")
        coverage = float(numeric.notna().mean())
        unique = int(numeric.nunique(dropna=True))
        if coverage >= 0.10 and unique >= 2:
            feature_columns.append(column)

    if not feature_columns:
        raise ValueError("Aucune feature plateau exploitable.")

    row_coverage = (
        historical_features[feature_columns].notna().mean(axis=1)
    )
    train_mask = (
        (history_index < oos_start)
        & (row_coverage.to_numpy() >= 0.65)
    )
    train_index = history_index[train_mask]

    if len(train_index) < 24 * 180:
        raise ValueError(
            "Historique d'entraînement insuffisant : "
            f"{len(train_index)} heures."
        )

    validation_start = (
        train_index[-1].normalize()
        - pd.Timedelta(days=validation_days - 1)
    )
    X_train_all = historical_features.loc[
        train_index, feature_columns
    ]
    y_train_all = y.loc[train_index]

    fit_mask = train_index < validation_start
    valid_mask = ~fit_mask

    X_fit = X_train_all.loc[fit_mask]
    y_fit = y_train_all.loc[fit_mask]
    X_valid = X_train_all.loc[valid_mask]
    y_valid = y_train_all.loc[valid_mask]

    if int(y_fit.sum()) < 10:
        raise ValueError(
            "Trop peu d'heures plateau dans le train CatBoost."
        )

    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": int(settings.get("catboost_iterations", 700)),
        "depth": int(settings.get("catboost_depth", 6)),
        "learning_rate": float(
            settings.get("catboost_learning_rate", 0.035)
        ),
        "l2_leaf_reg": float(
            settings.get("catboost_l2_leaf_reg", 6.0)
        ),
        "random_seed": int(deep_get(config, "model.seed", 42)),
        "auto_class_weights": "SqrtBalanced",
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
    }

    pilot = CatBoostClassifier(**params)
    pilot.fit(
        X_fit,
        y_fit,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=80,
        verbose=False,
    )
    best_iteration = int(pilot.get_best_iteration())
    if best_iteration < 20:
        best_iteration = min(199, params["iterations"] - 1)

    final_params = {
        **params,
        "iterations": best_iteration + 1,
    }
    model = CatBoostClassifier(**final_params)
    model.fit(X_train_all, y_train_all, verbose=False)

    model_path = resolve_path(args.model_output, project_root)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))

    # Toute la fenêtre est matérialisée ; la période oos_start -> J+1 est
    # garantie hors-échantillon vis-à-vis des labels de prix.
    probability = model.predict_proba(
        features[feature_columns]
    )[:, 1]
    probability_series = pd.Series(
        probability,
        index=features.index,
        name="zero_plateau_probability",
    )
    decoded = decode_probability_days(
        probability_series,
        min_length=min_hours,
        max_length=int(settings.get("decoder_max_length", 9)),
        solar_start_hour=solar_start,
        solar_end_hour=solar_end,
        min_mean_probability=float(
            settings.get("decoder_min_mean_probability", 0.45)
        ),
        minimum_score=float(
            settings.get("decoder_minimum_score", 0.0)
        ),
    )
    decoded["zero_plateau_is_oos"] = (
        decoded.index >= oos_start
    ).astype(np.float32)

    output_path = resolve_path(args.output, project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decoded.reset_index(names="timestamp").to_csv(
        output_path,
        index=False,
        compression="gzip",
    )

    eval_index = history_index[history_index >= evaluation_start]
    eval_metrics = classifier_metrics(
        y.reindex(eval_index).to_numpy(),
        probability_series.reindex(eval_index).to_numpy(),
    )

    # Expert zero appris uniquement sur le train.
    train_plateau_prices = data.target.reindex(train_index).loc[
        y_train_all.astype(bool)
    ].dropna()
    if train_plateau_prices.empty:
        raise ValueError("Aucun plateau dans le train pour l'expert zero.")

    expert_quantiles = {
        f"q{int(q * 100):02d}": float(train_plateau_prices.quantile(q))
        for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    }

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(
        output_path.with_name("zero_plateau_feature_importance.csv"),
        index=False,
    )

    metadata = {
        "definition": {
            "low": low,
            "high": high,
            "min_consecutive_hours": min_hours,
            "solar_start_hour": solar_start,
            "solar_end_hour": solar_end,
        },
        "aliases": {
            "residual_load": aliases.residual_load,
            "solar": aliases.solar,
            "nuclear": aliases.nuclear,
            "export": aliases.export,
        },
        "solar_direct_available": aliases.solar is not None,
        "scales": scales,
        "feature_columns": feature_columns,
        "train_start": str(train_index.min()),
        "train_end": str(train_index.max()),
        "oos_start": str(oos_start),
        "evaluation_start": str(evaluation_start),
        "last_history": str(last_history),
        "backtest_days": backtest_days,
        "context_buffer_days": context_buffer_days,
        "classifier_validation_days": validation_days,
        "best_iteration": best_iteration,
        "catboost_params": final_params,
        "evaluation_classifier_metrics": eval_metrics,
        "zero_expert_quantiles": expert_quantiles,
        "train_plateau_hours": int(len(train_plateau_prices)),
        "output_file": str(output_path),
        "model_file": str(model_path),
    }

    metadata_path = resolve_path(args.metadata_output, project_root)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print("=" * 88)
    print("C6 — ZERO PLATEAU CATBOOST")
    print("=" * 88)
    print(
        f"Définition : [{low:+.1f}, {high:+.1f}] €/MWh, "
        f">= {min_hours}h, fenêtre {solar_start:02d}–{solar_end:02d}"
    )
    print(f"Train      : {train_index.min()} -> {train_index.max()}")
    print(f"OOS start  : {oos_start}")
    print(f"Évaluation : {evaluation_start} -> {last_history}")
    print(
        "Solar direct : "
        + (aliases.solar or "ABSENT -> proxy horaire seulement")
    )
    print("\nMétriques classifier OOS")
    for key, value in eval_metrics.items():
        print(f"  {key:<22} {value}")
    print("\nTop features")
    print(importance.head(15).to_string(index=False))
    print(f"\nProbabilités : {output_path}")
    print(f"Modèle        : {model_path}")
    print(f"Métadonnées   : {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
