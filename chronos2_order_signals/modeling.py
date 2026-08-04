from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .features import feature_columns
from .labels import LABEL_COLUMNS


@dataclass(frozen=True)
class ModelSettings:
    iterations: int = 350
    depth: int = 7
    learning_rate: float = 0.05
    l2_leaf_reg: float = 8.0
    random_strength: float = 0.5
    validation_days: int = 60
    early_stopping_rounds: int = 50
    random_seed: int = 42
    thread_count: int = -1


def model_settings(config: Mapping[str, Any]) -> ModelSettings:
    raw = config.get("order_signals", {}).get("model", {})
    defaults = ModelSettings()
    kwargs = {
        field: raw.get(field, getattr(defaults, field))
        for field in ModelSettings.__dataclass_fields__
    }
    return ModelSettings(**kwargs)


def _new_model(settings: ModelSettings) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MultiRMSE",
        eval_metric="MultiRMSE",
        iterations=settings.iterations,
        depth=settings.depth,
        learning_rate=settings.learning_rate,
        l2_leaf_reg=settings.l2_leaf_reg,
        random_strength=settings.random_strength,
        random_seed=settings.random_seed,
        thread_count=settings.thread_count,
        allow_writing_files=False,
        verbose=False,
        has_time=True,
        nan_mode="Min",
    )


def fit_multioutput_model(
    training_frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[CatBoostRegressor, list[str], dict[str, Any]]:
    settings = model_settings(config)
    features = feature_columns(training_frame)
    leaked_targets = sorted(set(features).intersection(LABEL_COLUMNS))
    if leaked_targets:
        raise RuntimeError(
            "Fuite de cible détectée dans les features du modèle auxiliaire : "
            f"{leaked_targets}"
        )
    if not features:
        raise ValueError("Aucune feature numérique pour le modèle auxiliaire.")

    ordered = training_frame.sort_values("timestamp")
    valid_mask = ordered[list(LABEL_COLUMNS)].notna().all(axis=1)
    ordered = ordered.loc[valid_mask]
    if ordered.empty:
        raise ValueError("Aucun label complet pour entraîner le modèle auxiliaire.")

    last_day = pd.Timestamp(ordered["delivery_day"].max())
    validation_start = last_day - pd.DateOffset(
        days=max(1, settings.validation_days - 1)
    )
    train_part = ordered.loc[ordered["delivery_day"] < validation_start]
    validation_part = ordered.loc[ordered["delivery_day"] >= validation_start]

    if train_part["delivery_day"].nunique() < 30 or validation_part.empty:
        train_part = ordered
        validation_part = pd.DataFrame(columns=ordered.columns)

    model = _new_model(settings)
    fit_kwargs: dict[str, Any] = {}
    if not validation_part.empty:
        fit_kwargs.update(
            {
                "eval_set": (
                    validation_part[features],
                    validation_part[list(LABEL_COLUMNS)],
                ),
                "early_stopping_rounds": settings.early_stopping_rounds,
                "use_best_model": True,
            }
        )

    model.fit(
        train_part[features],
        train_part[list(LABEL_COLUMNS)],
        **fit_kwargs,
    )

    importance = model.get_feature_importance()
    ranked_importance = sorted(
        (
            {"feature": feature, "importance": float(value)}
            for feature, value in zip(features, importance, strict=True)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    metadata = {
        "settings": asdict(settings),
        "feature_columns": features,
        "training_rows": int(len(train_part)),
        "validation_rows": int(len(validation_part)),
        "training_first_timestamp": str(train_part["timestamp"].min()),
        "training_last_timestamp": str(train_part["timestamp"].max()),
        "best_iteration": int(model.get_best_iteration()),
        "feature_importance": ranked_importance,
    }
    return model, features, metadata


def predict_scores(
    model: CatBoostRegressor,
    frame: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    prediction = np.asarray(model.predict(frame[features]), dtype=np.float64)
    if prediction.ndim == 1:
        prediction = prediction.reshape(-1, 1)
    if prediction.shape[1] != len(LABEL_COLUMNS):
        raise RuntimeError(
            "Dimension de sortie CatBoost inattendue : "
            f"{prediction.shape}."
        )
    result = pd.DataFrame(
        np.clip(prediction, 0.0, 1.0),
        columns=LABEL_COLUMNS,
        index=frame.index,
    )
    return result.astype(np.float32)


def save_model_bundle(
    model: CatBoostRegressor,
    metadata: Mapping[str, Any],
    directory: Path,
    stem: str,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / f"{stem}.cbm"
    metadata_path = directory / f"{stem}.json"
    model.save_model(str(model_path))
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return model_path, metadata_path
