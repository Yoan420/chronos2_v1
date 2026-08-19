"""Leakage-aware non-negative stacking of external OOF forecasts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import LeakageRiskError, make_prediction_frame, require_frame


_QUANTILE_LABELS = ("q10", "q50", "q90")


def purged_expanding_splits(
    n_samples: int,
    *,
    n_splits: int = 5,
    min_train_size: int | None = None,
    test_size: int | None = None,
    gap: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create strictly chronological expanding-window OOF splits.

    Rows must already be sorted by forecast origin.  ``gap`` is expressed in
    rows and can purge overlapping lookbacks/labels between train and test.
    No row can occur in both sides of a split.
    """

    if n_samples < 3:
        raise ValueError("Au moins trois observations sont nécessaires.")
    if n_splits < 1 or gap < 0:
        raise ValueError("n_splits doit être >= 1 et gap >= 0.")
    if test_size is None:
        test_size = max(1, n_samples // (n_splits + 1))
    if min_train_size is None:
        min_train_size = n_samples - n_splits * test_size - gap
    if min_train_size < 1 or test_size < 1:
        raise ValueError("min_train_size et test_size doivent être positifs.")

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        train_end = min_train_size + fold * test_size
        test_start = train_end + gap
        test_end = min(test_start + test_size, n_samples)
        if train_end <= 0 or test_start >= test_end:
            raise ValueError(
                "Paramètres de split incompatibles avec le nombre de lignes."
            )
        train = np.arange(0, train_end, dtype=int)
        test = np.arange(test_start, test_end, dtype=int)
        splits.append((train, test))
    return splits


def _parse_prediction_columns(frame: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Map model -> quantile -> source column.

    Supported flat formats are ``chronos2__q50`` (all quantiles supported) and
    a bare ``chronos2`` column, interpreted as its median forecast.  Thus an
    externally generated Chronos OOF prediction can be included without any
    dependency on the Chronos runtime.
    """

    mapping: dict[str, dict[str, str]] = {}
    for raw_column in frame.columns:
        if not isinstance(raw_column, str):
            raise TypeError("Les colonnes de prédiction doivent être des chaînes.")
        if "__" in raw_column:
            model, label = raw_column.rsplit("__", 1)
            if not model or label not in _QUANTILE_LABELS:
                raise ValueError(
                    f"Colonne de prédiction invalide: {raw_column!r}; utilisez "
                    "<modèle>__q10, <modèle>__q50 ou <modèle>__q90."
                )
        else:
            model, label = raw_column, "q50"
        if model in mapping and label in mapping[model]:
            raise ValueError(f"Prédiction dupliquée pour {model}/{label}.")
        mapping.setdefault(model, {})[label] = raw_column
    without_median = [model for model, values in mapping.items() if "q50" not in values]
    if without_median:
        raise ValueError(f"Prévision q50 absente pour les modèles {without_median}.")
    return mapping


class NonNegativeOOFEnsemble:
    """MAE-optimal convex combination fitted only on declared OOF forecasts."""

    def __init__(
        self,
        *,
        minimum_rows: int = 48,
        weight_l2: float = 1e-6,
        optimization_maxiter: int = 2_000,
    ) -> None:
        if minimum_rows < 2:
            raise ValueError("minimum_rows doit être >= 2.")
        if weight_l2 < 0:
            raise ValueError("weight_l2 doit être positif ou nul.")
        self.minimum_rows = int(minimum_rows)
        self.weight_l2 = float(weight_l2)
        self.optimization_maxiter = int(optimization_maxiter)

    def fit(
        self,
        oof_predictions: pd.DataFrame,
        y: pd.Series | Sequence[float] | np.ndarray,
        *,
        is_oof: bool = False,
        prediction_origin: Sequence[object] | pd.Series | None = None,
        delivery_start: Sequence[object] | pd.Series | None = None,
    ) -> "NonNegativeOOFEnsemble":
        """Fit convex weights on external OOF predictions.

        The caller must explicitly set ``is_oof=True``.  If origin and delivery
        timestamps are supplied, every prediction must also have been produced
        strictly before its delivery start.  This does not replace proper
        fold construction; use :func:`purged_expanding_splits` upstream.
        """

        require_frame(oof_predictions, name="oof_predictions")
        if not is_oof:
            raise LeakageRiskError(
                "L'ensemble refuse des prédictions non certifiées OOF. "
                "Générez-les avec des folds expanding-window puis passez "
                "is_oof=True."
            )
        if (prediction_origin is None) != (delivery_start is None):
            raise ValueError(
                "prediction_origin et delivery_start doivent être fournis ensemble."
            )
        if prediction_origin is not None and delivery_start is not None:
            origins = pd.to_datetime(list(prediction_origin), utc=True, errors="coerce")
            deliveries = pd.to_datetime(list(delivery_start), utc=True, errors="coerce")
            if len(origins) != len(oof_predictions) or len(deliveries) != len(
                oof_predictions
            ):
                raise ValueError("Longueur incohérente des timestamps de causalité.")
            if origins.isna().any() or deliveries.isna().any():
                raise ValueError("Timestamps de causalité invalides ou manquants.")
            if np.any(origins >= deliveries):
                raise LeakageRiskError(
                    "Au moins une prédiction a été créée après ou au début de "
                    "sa période de livraison."
                )

        mapping = _parse_prediction_columns(oof_predictions)
        model_names = list(mapping)
        matrix = np.column_stack(
            [
                pd.to_numeric(oof_predictions[mapping[name]["q50"]], errors="coerce")
                .to_numpy(dtype=float)
                for name in model_names
            ]
        )
        target = pd.to_numeric(pd.Series(np.asarray(y)), errors="coerce").to_numpy(
            dtype=float
        )
        if len(target) != len(oof_predictions):
            raise ValueError("oof_predictions et y n'ont pas la même longueur.")
        valid = np.isfinite(target) & np.isfinite(matrix).all(axis=1)
        if int(valid.sum()) < self.minimum_rows:
            raise ValueError(
                "Pas assez de lignes OOF complètes: "
                f"{int(valid.sum())} < {self.minimum_rows}."
            )
        x_train = matrix[valid]
        y_train = target[valid]
        n_models = len(model_names)
        initial = np.full(n_models, 1.0 / n_models)

        def objective(weights: np.ndarray) -> float:
            mae = np.mean(np.abs(y_train - x_train @ weights))
            return float(mae + self.weight_l2 * np.dot(weights, weights))

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_models,
            constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
            options={"maxiter": self.optimization_maxiter, "ftol": 1e-10},
        )
        if not result.success or not np.isfinite(result.x).all():
            # Safe deterministic fallback: choose the best OOF base model.  It
            # cannot perform worse in-sample than an equal blend.
            losses = np.mean(np.abs(y_train[:, None] - x_train), axis=0)
            weights = np.zeros(n_models)
            weights[int(np.argmin(losses))] = 1.0
            self.optimization_message_ = f"fallback_best_model: {result.message}"
        else:
            weights = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
            weights /= weights.sum()
            self.optimization_message_ = str(result.message)

        self.model_names_ = model_names
        self.weights_ = pd.Series(weights, index=model_names, name="weight")
        self.training_mae_ = float(np.mean(np.abs(y_train - x_train @ weights)))
        self.n_oof_rows_ = int(valid.sum())
        self.is_fitted_ = True
        return self

    def predict(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Blend q10/q50/q90, using a model's q50 when an interval is absent."""

        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("NonNegativeOOFEnsemble doit être entraîné.")
        require_frame(predictions, name="predictions")
        mapping = _parse_prediction_columns(predictions)
        missing = [name for name in self.model_names_ if name not in mapping]
        if missing:
            raise ValueError(f"Modèles absents à l'inférence: {missing}.")
        values = np.empty((len(predictions), 3), dtype=float)
        for q_index, label in enumerate(_QUANTILE_LABELS):
            matrix = np.column_stack(
                [
                    pd.to_numeric(
                        predictions[
                            mapping[name].get(label, mapping[name]["q50"])
                        ],
                        errors="coerce",
                    ).to_numpy(dtype=float)
                    for name in self.model_names_
                ]
            )
            if not np.isfinite(matrix).all():
                raise ValueError("Prédictions manquantes ou infinies à l'inférence.")
            values[:, q_index] = matrix @ self.weights_.to_numpy()
        return make_prediction_frame(values, predictions.index, (0.1, 0.5, 0.9))

