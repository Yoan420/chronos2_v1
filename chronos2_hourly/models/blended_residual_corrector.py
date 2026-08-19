"""Deterministic weighted blends of fitted residual correctors.

The blend is deliberately applied to the *raw component corrections* and is
clipped only once afterwards.  This matches the frozen extended-OOF recipe
used by the French hourly experiment and avoids silently changing it by
clipping each component separately.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .residual_corrector import ResidualCorrector, apply_residual_correction


class BlendedResidualCorrector:
    """Fit several :class:`ResidualCorrector` objects and blend their shifts."""

    def __init__(
        self,
        components: Mapping[str, ResidualCorrector],
        weights: Mapping[str, float],
        *,
        max_abs_correction: float | None = 40.0,
    ) -> None:
        if not components:
            raise ValueError("components ne doit pas etre vide.")
        if set(components) != set(weights):
            raise ValueError("components et weights doivent avoir les memes noms.")
        if any(not isinstance(model, ResidualCorrector) for model in components.values()):
            raise TypeError("Chaque composant doit etre un ResidualCorrector.")
        raw_weights = np.asarray([weights[name] for name in components], dtype=float)
        if not np.isfinite(raw_weights).all() or (raw_weights < 0.0).any():
            raise ValueError("Les poids doivent etre finis et positifs ou nuls.")
        if not np.isclose(float(raw_weights.sum()), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("La somme des poids doit etre exactement egale a 1.")
        if max_abs_correction is not None:
            bound = float(max_abs_correction)
            if not np.isfinite(bound) or bound <= 0.0:
                raise ValueError("max_abs_correction doit etre fini et strictement positif.")
            max_abs_correction = bound

        self.components = dict(components)
        self.weights = {name: float(weights[name]) for name in components}
        self.max_abs_correction = max_abs_correction

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | Sequence[float] | np.ndarray,
        base_predictions: pd.DataFrame,
        expert_predictions: pd.DataFrame | None = None,
    ) -> "BlendedResidualCorrector":
        fitted: dict[str, ResidualCorrector] = {}
        schemas: set[tuple[str, ...]] = set()
        for name, model in self.components.items():
            fitted[name] = model.fit(X, y, base_predictions, expert_predictions)
            schemas.add(tuple(fitted[name].feature_columns_))
        if len(schemas) != 1:
            raise RuntimeError("Les composants n'utilisent pas les memes meta-features.")

        self.components_ = fitted
        self.feature_columns_ = next(iter(schemas))
        self.n_training_rows_ = int(
            min(model.n_training_rows_ for model in fitted.values())
        )
        self.backend_ = "blend(" + ",".join(
            f"{name}:{self.weights[name]:.6g}" for name in fitted
        ) + ")"
        self.is_fitted_ = True
        return self

    def predict_correction(
        self,
        X: pd.DataFrame,
        base_predictions: pd.DataFrame,
        expert_predictions: pd.DataFrame | None = None,
    ) -> pd.Series:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError(
                "BlendedResidualCorrector doit etre entraine avant predict()."
            )
        correction: np.ndarray | None = None
        for name, model in self.components_.items():
            values = model.predict_correction(
                X,
                base_predictions,
                expert_predictions,
            ).to_numpy(dtype=float)
            weighted = self.weights[name] * values
            correction = weighted if correction is None else correction + weighted
        assert correction is not None
        if self.max_abs_correction is not None:
            correction = np.clip(
                correction,
                -self.max_abs_correction,
                self.max_abs_correction,
            )
        return pd.Series(
            correction,
            index=X.index,
            name="residual_correction",
        )

    def predict(
        self,
        X: pd.DataFrame,
        base_predictions: pd.DataFrame,
        expert_predictions: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        correction = self.predict_correction(
            X,
            base_predictions,
            expert_predictions,
        )
        result = apply_residual_correction(
            base_predictions,
            correction,
            max_abs_correction=None,
        )
        result.attrs.update(
            {
                "backend": self.backend_,
                "n_meta_features": len(self.feature_columns_),
                "weights": dict(self.weights),
            }
        )
        return result

    def diagnostics(self) -> dict[str, Any]:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Le correcteur composite n'est pas entraine.")
        return {
            "backend": self.backend_,
            "weights": dict(self.weights),
            "max_abs_correction": self.max_abs_correction,
            "n_training_rows": self.n_training_rows_,
            "n_meta_features": len(self.feature_columns_),
            "components": {
                name: {
                    "backend": model.backend_,
                    "n_training_rows": model.n_training_rows_,
                    "training_mae_before": model.residual_training_mae_before_,
                    "training_mae_after": model.residual_training_mae_after_,
                }
                for name, model in self.components_.items()
            },
        }


__all__ = ["BlendedResidualCorrector"]
