"""Signed residual means, with no detector fitting or quantile relabelling.

The forest branch evaluates the expectation of the *same* two empirical CDFs
as nyx_coherent_p50. A tree's leaf mean is exactly its uniform empirical-CDF
expectation; averaging tree predictions therefore avoids support construction
and CDF inversion without changing that functional.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

from nyx_coherent_p50.distribution import FOREST_PARAMETERS


DEFAULT_OPTIONS = {
    "threads": 2,
    "learner": {"max_iter": 80, "max_leaf_nodes": 15, "min_samples_leaf": 30,
                "learning_rate": .06, "l2_regularization": 10., "random_state": 1729},
    "governance": {"lookback_days": 90, "minimum_days": 28,
        "minimum_changed_days": 5, "weights": [0., .25, .5, 1.],
        "uncertainty_z": 1., "minimum_mse_gain": 0.,
        "maximum_mae_degradation": 0., "maximum_ordinary_mae_degradation": 0.},
    "correction_clip_eur_mwh": 400.,
    "require_full_training_history": False,
}


def validate_options(options=None):
    """Small explicit search space: no arbitrary estimator or code arguments."""
    if options is None:
        options = {}
    if not isinstance(options, dict) or set(options)-set(DEFAULT_OPTIONS):
        raise ValueError("Unknown NYX RMSE options.")
    result = deepcopy(DEFAULT_OPTIONS)
    for name, value in options.items():
        if name in ("learner", "governance"):
            if not isinstance(value, dict) or set(value)-set(result[name]):
                raise ValueError(f"Unknown {name} parameters.")
            result[name].update(value)
        else:
            result[name] = value
    def integer(value, low, high):
        return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high
    def number(value, low, high):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and np.isfinite(value) and low <= value <= high)
    if not integer(result["threads"], 1, 2):
        raise ValueError("Use one or two CPU threads.")
    learner, governance = result["learner"], result["governance"]
    for name, low, high in (("max_iter", 1, 1000), ("max_leaf_nodes", 2, 128),
                            ("min_samples_leaf", 2, 1000), ("random_state", 0, 2**31-1)):
        if not integer(learner[name], low, high):
            raise ValueError(f"Invalid learner {name}.")
    for name, low, high in (("learning_rate", .0001, 1.), ("l2_regularization", 0., 1e6)):
        if not number(learner[name], low, high):
            raise ValueError(f"Invalid learner {name}.")
    for name, low, high in (("lookback_days", 7, 365), ("minimum_days", 2, 365),
                            ("minimum_changed_days", 1, 365)):
        if not integer(governance[name], low, high):
            raise ValueError(f"Invalid governance {name}.")
    if not (governance["minimum_changed_days"] <= governance["minimum_days"] <= governance["lookback_days"]):
        raise ValueError("Changed days <= minimum days <= lookback days is required.")
    for name in ("uncertainty_z", "minimum_mse_gain", "maximum_mae_degradation", "maximum_ordinary_mae_degradation"):
        if not number(governance[name], 0., 1e6):
            raise ValueError(f"Invalid governance {name}.")
    weights = governance["weights"]
    if (not isinstance(weights, list) or not 2 <= len(weights) <= 5
            or not all(number(w, 0., 1.) for w in weights)
            or len(set(weights)) != len(weights) or 0. not in weights or 1. not in weights):
        raise ValueError("Two to five unique weights in [0,1], including 0 and 1, required.")
    governance["weights"] = sorted(float(w) for w in weights)
    if not number(result["correction_clip_eur_mwh"], .01, 10000.):
        raise ValueError("A finite positive signed correction cap is required.")
    if not isinstance(result["require_full_training_history"], bool):
        raise ValueError("require_full_training_history must be an explicit boolean.")
    return result


def _matrix(X, columns=None):
    X = np.asarray(X, dtype=float)
    if (X.ndim != 2 or not X.shape[1] or np.isinf(X).any()
            or (columns is not None and X.shape[1] != columns)):
        raise ValueError("A two-dimensional finite-or-missing feature matrix is required.")
    return X


def fit_means(X, errors, thresholds, options=None):
    """Fit one shared imputer, a direct MSE model and two regime-mean forests."""
    options = validate_options(options)
    X = _matrix(X)
    errors, thresholds = np.asarray(errors, float), np.asarray(thresholds, float)
    if (not len(X) or errors.shape != (len(X),) or thresholds.shape != errors.shape
            or not np.isfinite(errors).all() or not np.isfinite(thresholds).all() or (thresholds <= 0).any()):
        raise ValueError("Aligned finite signed errors and strictly positive thresholds required.")
    normalized = errors / thresholds
    normal = normalized < 1
    if not normal.any() or normal.all():
        raise ValueError("Both historical residual regimes must be non-empty.")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    transformed = imputer.fit_transform(X)
    threads = options["threads"]
    with threadpool_limits(limits=threads):
        direct = HistGradientBoostingRegressor(loss="squared_error", early_stopping=False,
            **options["learner"]).fit(transformed, errors)
        forests = {}
        for name, mask in (("normal", normal), ("spike", ~normal)):
            forests[name] = RandomForestRegressor(n_jobs=threads, **FOREST_PARAMETERS).fit(
                transformed[mask], normalized[mask])
    return {"imputer": imputer, "direct": direct, "forests": forests,
        "feature_count": X.shape[1], "core_rows": len(X), "normal_rows": int(normal.sum()),
        "spike_rows": int((~normal).sum()), "negative_rows": int((errors < 0).sum()),
        "options": options, "forest_parameters": dict(FOREST_PARAMETERS)}


def predict_means(state, X, probabilities, thresholds):
    """Return unbounded signed means, not P50s or quantile-band centres."""
    X = _matrix(X, state["feature_count"])
    probability, threshold = np.asarray(probabilities, float), np.asarray(thresholds, float)
    if (probability.shape != (len(X),) or threshold.shape != probability.shape
            or not np.isfinite(probability).all() or ((probability < 0)|(probability > 1)).any()
            or not np.isfinite(threshold).all() or (threshold <= 0).any()):
        raise ValueError("Aligned probabilities in [0,1] and positive thresholds required.")
    if not len(X):
        return {"residual_mse": np.empty(0), "mixture_mean": np.empty(0)}
    transformed = state["imputer"].transform(X)
    with threadpool_limits(limits=state["options"]["threads"]):
        direct = state["direct"].predict(transformed)
        normal = state["forests"]["normal"].predict(transformed)
        spike = state["forests"]["spike"].predict(transformed)
    mixture = threshold * ((1-probability)*normal + probability*spike)
    if not np.isfinite(direct).all() or not np.isfinite(mixture).all():
        raise ValueError("Residual mean inference returned a non-finite value.")
    return {"residual_mse": direct, "mixture_mean": mixture}
