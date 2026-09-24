"""Deterministic residual correction from a sparse topology context.

The model is intentionally lightweight: one native-missing-value
``HistGradientBoostingRegressor`` learns ``actual - base_q50`` under an MAE
objective, and the resulting correction is applied as one common shift to
q10/q50/q90.  Consequently, the base interval widths and quantile order are
preserved exactly.

Storm and MKOnline are never accepted as model features.  A base forecast may
nevertheless be either autonomous or an already-produced MKOnline blend: the
blend is the object being corrected, not an explanatory input.  Model
selection and an optional identity fallback remain explicit orchestration
decisions outside this class.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import re
from typing import Final

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from chronos2_hourly.features import (
    HourlyFeatureContractError,
    validate_utc_hourly_index,
)
from chronos2_hourly.topology_context import (
    TOPOLOGY_CONTEXT_COLUMNS,
    TOPOLOGY_CONTEXT_SCHEMA_VERSION,
    TOPOLOGY_ZONES,
    zones_within_radius,
)


TOPOLOGY_RESIDUAL_SCHEMA_VERSION: Final[str] = "topology-residual-corrector-v1"
QUANTILE_COLUMNS: Final[tuple[str, str, str]] = ("q10", "q50", "q90")
FORBIDDEN_FEATURE_SOURCES: Final[tuple[str, str]] = ("storm", "mkonline")
BASE_FEATURE_COLUMNS: Final[tuple[str, str, str]] = (
    "base__q50",
    "base__lower_width",
    "base__upper_width",
)
_FORBIDDEN_SOURCE = re.compile(r"(?i)(?:storm|mk[\s_-]*online)")


class TopologyResidualCorrectionError(ValueError):
    """Raised when topology correction cannot preserve its causal contract."""


def _normalise_clip(
    value: float | Sequence[float] | None,
) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("correction_clip doit être numérique, pas booléen.")
    if isinstance(value, (int, float, np.integer, np.floating)):
        maximum = float(value)
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("correction_clip scalaire doit être fini et positif.")
        return (-maximum, maximum)
    bounds = tuple(float(item) for item in value)
    if len(bounds) != 2:
        raise ValueError("correction_clip doit contenir exactement deux bornes.")
    lower, upper = bounds
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError("Bornes correction_clip invalides.")
    return (lower, upper)


def _validate_context_metadata(frame: pd.DataFrame) -> dict[str, object]:
    raw = frame.attrs.get("topology_context")
    if not isinstance(raw, Mapping):
        raise TopologyResidualCorrectionError(
            "Métadonnées topology_context absentes; utilisez build_topology_context()."
        )
    metadata = dict(raw)
    required = {
        "schema_version",
        "target_zone",
        "radius",
        "neighbours",
        "included_zones",
        "timezone",
        "residual_load_source",
        "price_lag_hours",
        "missing_policy",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise TopologyResidualCorrectionError(
            f"Métadonnées topology_context incomplètes: {missing}."
        )
    if metadata["schema_version"] != TOPOLOGY_CONTEXT_SCHEMA_VERSION:
        raise TopologyResidualCorrectionError(
            "Version de schéma topology_context incompatible: "
            f"{metadata['schema_version']!r}."
        )
    if metadata["radius"] not in (0, 1):
        raise TopologyResidualCorrectionError("Le rayon topologique doit valoir 0 ou 1.")
    target_zone = metadata["target_zone"]
    if target_zone not in TOPOLOGY_ZONES:
        raise TopologyResidualCorrectionError(
            f"Zone de contexte invalide: {target_zone!r}."
        )
    expected_zones = zones_within_radius(str(target_zone), int(metadata["radius"]))
    if tuple(metadata["included_zones"]) != expected_zones:
        raise TopologyResidualCorrectionError(
            "included_zones ne correspond pas au masque topologique attendu."
        )
    if tuple(metadata["neighbours"]) != expected_zones[1:]:
        raise TopologyResidualCorrectionError(
            "neighbours ne correspond pas au masque topologique attendu."
        )
    if metadata["residual_load_source"] != "sealed_pit":
        raise TopologyResidualCorrectionError(
            "Le contexte doit déclarer residual_load_source='sealed_pit'."
        )
    if metadata["price_lag_hours"] != 24:
        raise TopologyResidualCorrectionError("Le lag prix physique doit être de 24 h.")
    if metadata["missing_policy"] != "native_nan_no_interpolation":
        raise TopologyResidualCorrectionError(
            "La politique de valeurs manquantes doit rester native sans interpolation."
        )
    return metadata


def _context_signature(metadata: Mapping[str, object]) -> tuple[object, ...]:
    return (
        metadata["schema_version"],
        metadata["target_zone"],
        int(metadata["radius"]),
        tuple(metadata["neighbours"]),
        tuple(metadata["included_zones"]),
        metadata["timezone"],
        metadata["residual_load_source"],
        int(metadata["price_lag_hours"]),
        metadata["missing_policy"],
    )


def _validate_context(
    frame: pd.DataFrame,
    *,
    fitted_signature: tuple[object, ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("X doit être un pandas.DataFrame.")
    if frame.empty:
        raise TopologyResidualCorrectionError("X est vide.")
    if not frame.columns.is_unique:
        raise TopologyResidualCorrectionError("X contient des colonnes dupliquées.")
    forbidden = [
        str(column) for column in frame.columns if _FORBIDDEN_SOURCE.search(str(column))
    ]
    if forbidden:
        raise TopologyResidualCorrectionError(
            f"Features Storm/MKOnline interdites: {forbidden}."
        )
    received_columns = tuple(frame.columns)
    if received_columns != TOPOLOGY_CONTEXT_COLUMNS:
        missing = [column for column in TOPOLOGY_CONTEXT_COLUMNS if column not in frame]
        unexpected = [column for column in frame if column not in TOPOLOGY_CONTEXT_COLUMNS]
        raise TopologyResidualCorrectionError(
            "Schéma de contexte différent du schéma scellé; "
            f"missing={missing}, unexpected={unexpected}, ordre_exact_requis=true."
        )
    try:
        validate_utc_hourly_index(
            frame.index,
            name="X.index",
            require_contiguous=False,
        )
    except HourlyFeatureContractError as exc:
        raise TopologyResidualCorrectionError(str(exc)) from exc
    invalid_types = [
        column
        for column in frame
        if not pd.api.types.is_numeric_dtype(frame[column])
        and not pd.api.types.is_bool_dtype(frame[column])
    ]
    if invalid_types:
        raise TypeError(f"Features non numériques dans X: {invalid_types}.")
    numeric = frame.astype(float).copy(deep=True)
    if bool(np.isinf(numeric.to_numpy(dtype=float, copy=False)).any()):
        raise TopologyResidualCorrectionError(
            "X contient des valeurs infinies; aucun remplacement implicite."
        )
    metadata = _validate_context_metadata(frame)
    signature = _context_signature(metadata)
    if fitted_signature is not None and signature != fitted_signature:
        raise TopologyResidualCorrectionError(
            "Le contexte de prédiction diffère du contexte d'entraînement "
            "(zone, rayon, voisinage, fuseau ou politique causale)."
        )
    return numeric, metadata


def _validate_base_predictions(
    base_predictions: pd.DataFrame,
    *,
    expected_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    if not isinstance(base_predictions, pd.DataFrame):
        raise TypeError("base_predictions doit être un pandas.DataFrame.")
    if not base_predictions.index.equals(expected_index):
        raise TopologyResidualCorrectionError(
            "base_predictions.index doit être exactement égal à X.index."
        )
    missing = [column for column in QUANTILE_COLUMNS if column not in base_predictions]
    if missing:
        raise TopologyResidualCorrectionError(
            f"Quantiles absents de base_predictions: {missing}."
        )
    base = base_predictions.loc[:, list(QUANTILE_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    values = base.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise TopologyResidualCorrectionError(
            "Les quantiles de base_predictions doivent être finis."
        )
    crossed = (base["q10"] > base["q50"]) | (base["q50"] > base["q90"])
    if bool(crossed.any()):
        raise TopologyResidualCorrectionError(
            "Les quantiles de base_predictions se croisent."
        )
    return base.astype(float)


def _validate_target(
    y: pd.Series | Sequence[float] | np.ndarray,
    *,
    expected_index: pd.DatetimeIndex,
) -> pd.Series:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1 or not y.index.equals(expected_index):
            raise TopologyResidualCorrectionError(
                "y doit avoir une colonne et un index exactement égal à X.index."
            )
        raw = y.iloc[:, 0].to_numpy(copy=False)
    elif isinstance(y, pd.Series):
        if not y.index.equals(expected_index):
            raise TopologyResidualCorrectionError(
                "y.index doit être exactement égal à X.index."
            )
        raw = y.to_numpy(copy=False)
    else:
        raw = np.asarray(y)
    if raw.ndim != 1 or len(raw) != len(expected_index):
        raise TopologyResidualCorrectionError(
            "y doit être unidimensionnel et de même longueur que X."
        )
    target = pd.to_numeric(
        pd.Series(raw, index=expected_index),
        errors="coerce",
    ).astype(float)
    if bool(np.isinf(target.to_numpy(dtype=float)).any()):
        raise TopologyResidualCorrectionError("y contient des valeurs infinies.")
    return target


def _estimator_features(
    context: pd.DataFrame,
    base: pd.DataFrame,
) -> pd.DataFrame:
    """Add level/width information from the forecast being corrected."""

    base_features = pd.DataFrame(
        {
            "base__q50": base["q50"],
            "base__lower_width": base["q50"] - base["q10"],
            "base__upper_width": base["q90"] - base["q50"],
        },
        index=context.index.copy(),
    )
    return pd.concat([context, base_features], axis=1, copy=False)


class TopologyResidualCorrector:
    """MAE-trained HGB correction with a frozen topology-context schema."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.04,
        max_iter: int = 240,
        max_leaf_nodes: int = 15,
        min_samples_leaf: int = 48,
        l2_regularization: float = 8.0,
        correction_scale: float = 1.0,
        correction_clip: float | Sequence[float] | None = 30.0,
        min_training_rows: int = 168,
    ) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate doit être fini et positif.")
        if isinstance(max_iter, bool) or max_iter < 1:
            raise ValueError("max_iter doit être un entier >= 1.")
        if isinstance(max_leaf_nodes, bool) or max_leaf_nodes < 2:
            raise ValueError("max_leaf_nodes doit être un entier >= 2.")
        if isinstance(min_samples_leaf, bool) or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf doit être un entier >= 1.")
        if not np.isfinite(l2_regularization) or l2_regularization < 0.0:
            raise ValueError("l2_regularization doit être fini et positif ou nul.")
        if not np.isfinite(correction_scale) or correction_scale < 0.0:
            raise ValueError("correction_scale doit être fini et positif ou nul.")
        if isinstance(min_training_rows, bool) or min_training_rows < 2:
            raise ValueError("min_training_rows doit être un entier >= 2.")

        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_regularization = float(l2_regularization)
        self.correction_scale = float(correction_scale)
        self.correction_clip = correction_clip
        self.correction_bounds_ = _normalise_clip(correction_clip)
        self.min_training_rows = int(min_training_rows)

    def hyperparameters(self) -> dict[str, object]:
        """Return the global block that must be identical for radius 0 and 1."""

        return {
            "estimator": "HistGradientBoostingRegressor",
            "loss": "absolute_error",
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "early_stopping": False,
            "random_state": 120,
            "thread_limit": 1,
            "correction_scale": self.correction_scale,
            "correction_clip": None
            if self.correction_bounds_ is None
            else list(self.correction_bounds_),
            "min_training_rows": self.min_training_rows,
        }

    def hyperparameter_sha256(self) -> str:
        encoded = json.dumps(
            self.hyperparameters(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _new_estimator(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=False,
            random_state=120,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | Sequence[float] | np.ndarray,
        base_predictions: pd.DataFrame,
    ) -> "TopologyResidualCorrector":
        features, metadata = _validate_context(X)
        base = _validate_base_predictions(
            base_predictions,
            expected_index=features.index,
        )
        target = _validate_target(y, expected_index=features.index)
        residual = target - base["q50"]
        valid = residual.notna()
        n_valid = int(valid.sum())
        if n_valid < self.min_training_rows:
            raise TopologyResidualCorrectionError(
                "Pas assez de résidus observés pour entraîner le correcteur: "
                f"{n_valid} < {self.min_training_rows}; aucun fallback implicite."
            )
        estimator_features = _estimator_features(features, base)
        training_features = estimator_features.loc[valid]
        if not bool(training_features.notna().any(axis=None)):
            raise TopologyResidualCorrectionError(
                "Toutes les features d'entraînement sont manquantes."
            )

        estimator = self._new_estimator()
        # A single native thread keeps the two radius arms exactly comparable,
        # avoids oversubscription beside Chronos, and remains ample for this
        # small 25-feature model.
        with threadpool_limits(limits=1):
            estimator.fit(
                training_features,
                residual.loc[valid].to_numpy(dtype=float),
            )
            raw_in_sample = np.asarray(
                estimator.predict(training_features),
                dtype=float,
            )
        if raw_in_sample.shape != (n_valid,) or not np.isfinite(raw_in_sample).all():
            raise RuntimeError("HGB a produit une correction d'entraînement invalide.")
        applied_in_sample = raw_in_sample * self.correction_scale
        if self.correction_bounds_ is not None:
            applied_in_sample = np.clip(
                applied_in_sample,
                self.correction_bounds_[0],
                self.correction_bounds_[1],
            )

        self.estimator_ = estimator
        self.context_feature_names_in_ = tuple(features.columns)
        self.feature_names_in_ = tuple(estimator_features.columns)
        self.context_metadata_ = dict(metadata)
        self.context_signature_ = _context_signature(metadata)
        self.n_training_rows_ = n_valid
        self.n_dropped_target_rows_ = int((~valid).sum())
        self.fit_missing_values_by_feature_ = {
            column: int(training_features[column].isna().sum())
            for column in training_features
        }
        self.fit_all_missing_feature_rows_ = int(
            training_features.isna().all(axis=1).sum()
        )
        observed_residual = residual.loc[valid].to_numpy(dtype=float)
        self.residual_training_mae_before_ = float(np.mean(np.abs(observed_residual)))
        self.residual_training_mae_after_ = float(
            np.mean(np.abs(observed_residual - applied_in_sample))
        )
        self.is_fitted_ = True
        return self

    def _prediction_inputs(
        self,
        X: pd.DataFrame,
        base_predictions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError(
                "TopologyResidualCorrector doit être entraîné avant predict()."
            )
        features, _ = _validate_context(
            X,
            fitted_signature=self.context_signature_,
        )
        base = _validate_base_predictions(
            base_predictions,
            expected_index=features.index,
        )
        estimator_features = _estimator_features(features, base)
        self.last_prediction_missing_values_by_feature_ = {
            column: int(estimator_features[column].isna().sum())
            for column in estimator_features
        }
        self.last_prediction_all_missing_feature_rows_ = int(
            estimator_features.isna().all(axis=1).sum()
        )
        return estimator_features, base

    def predict_correction(
        self,
        X: pd.DataFrame,
        base_predictions: pd.DataFrame,
    ) -> pd.Series:
        """Return the exact common shift, after scale and clipping."""

        estimator_features, _ = self._prediction_inputs(X, base_predictions)
        with threadpool_limits(limits=1):
            raw = np.asarray(
                self.estimator_.predict(estimator_features),
                dtype=float,
            )
        if raw.shape != (len(estimator_features),) or not np.isfinite(raw).all():
            raise RuntimeError("HGB a produit une correction invalide.")
        applied = raw * self.correction_scale
        if self.correction_bounds_ is not None:
            applied = np.clip(applied, self.correction_bounds_[0], self.correction_bounds_[1])
        result = pd.Series(
            applied,
            index=estimator_features.index,
            name="topology_correction",
        )
        result.attrs["raw_correction"] = pd.Series(
            raw,
            index=estimator_features.index,
            name="raw_topology_correction",
        )
        result.attrs["correction_clip"] = self.correction_bounds_
        return result

    def predict(
        self,
        X: pd.DataFrame,
        base_predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply one topology shift to all three base quantiles."""

        estimator_features, base = self._prediction_inputs(X, base_predictions)
        with threadpool_limits(limits=1):
            raw = np.asarray(
                self.estimator_.predict(estimator_features),
                dtype=float,
            )
        if raw.shape != (len(estimator_features),) or not np.isfinite(raw).all():
            raise RuntimeError("HGB a produit une correction invalide.")
        applied = raw * self.correction_scale
        if self.correction_bounds_ is not None:
            applied = np.clip(applied, self.correction_bounds_[0], self.correction_bounds_[1])
        result = base.add(applied, axis=0)
        crossed = (result["q10"] > result["q50"]) | (result["q50"] > result["q90"])
        if bool(crossed.any()):
            raise RuntimeError("La correction commune a violé l'ordre des quantiles.")
        result.attrs["topology_correction"] = pd.Series(
            applied,
            index=estimator_features.index,
            name="topology_correction",
        )
        result.attrs["topology_context"] = dict(self.context_metadata_)
        result.attrs["hyperparameter_sha256"] = self.hyperparameter_sha256()
        return result

    def audit_metadata(self) -> dict[str, object]:
        """Return model, causality, and native-NaN facts for report sidecars."""

        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Le correcteur doit être entraîné avant audit_metadata().")
        payload: dict[str, object] = {
            "schema_version": TOPOLOGY_RESIDUAL_SCHEMA_VERSION,
            "component": "topology_residual_corrector",
            "backend": "sklearn_hist_gradient_boosting",
            "loss": "absolute_error",
            "hyperparameters": self.hyperparameters(),
            "hyperparameter_sha256": self.hyperparameter_sha256(),
            "n_training_rows": self.n_training_rows_,
            "n_dropped_target_rows": self.n_dropped_target_rows_,
            "feature_names": list(self.feature_names_in_),
            "missing_policy": "native_nan_no_interpolation",
            "fit_missing_values_by_feature": dict(
                self.fit_missing_values_by_feature_
            ),
            "fit_all_missing_feature_rows": self.fit_all_missing_feature_rows_,
            "residual_training_mae_before": self.residual_training_mae_before_,
            "residual_training_mae_after": self.residual_training_mae_after_,
            "forbidden_feature_sources": list(FORBIDDEN_FEATURE_SOURCES),
            "topology_context": dict(self.context_metadata_),
        }
        if hasattr(self, "last_prediction_missing_values_by_feature_"):
            payload["last_prediction_missing_values_by_feature"] = dict(
                self.last_prediction_missing_values_by_feature_
            )
            payload["last_prediction_all_missing_feature_rows"] = (
                self.last_prediction_all_missing_feature_rows_
            )
        return payload


__all__ = [
    "BASE_FEATURE_COLUMNS",
    "FORBIDDEN_FEATURE_SOURCES",
    "QUANTILE_COLUMNS",
    "TOPOLOGY_RESIDUAL_SCHEMA_VERSION",
    "TopologyResidualCorrectionError",
    "TopologyResidualCorrector",
]
