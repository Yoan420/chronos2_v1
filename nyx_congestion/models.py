"""Two calibrated hurdles. Expected congestion is a feature, never a P50.

No current labels, final PTDFs, market prices or constraint names enter predict.
Poisson fits the conditional *mean* of positive hourly shadow-price / CGC;
the current positive CGC restores units. This is a hypothesis, not dispatch.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

PARAMETERS = dict(max_iter=100, learning_rate=.06, max_leaf_nodes=15,
                  min_samples_leaf=40, l2_regularization=5., early_stopping=False,
                  random_state=1729)


def matrix(value, columns=None):
    x = np.asarray(value, dtype=np.float32)
    if x.ndim != 2 or not x.shape[1] or np.isinf(x).any() or (columns is not None and x.shape[1] != columns):
        raise ValueError("Aligned finite-or-missing feature matrix required.")
    return x


def vector(value, n, *, probability=False, positive=False):
    x = np.asarray(value, dtype=float)
    if (x.shape != (n,) or not np.isfinite(x).all() or (positive and (x <= 0).any())
            or (probability and ((x < 0)|(x > 1)).any())):
        raise ValueError("Aligned finite vector outside its permitted range.")
    return x


def logit(p):
    p = np.clip(np.asarray(p), 1e-6, 1-1e-6)
    return np.log(p/(1-p)).reshape(-1, 1)


def fit_probability(X, y, V, cy, *, threads=2):
    X, V = matrix(X), matrix(V)
    y, cy = vector(y, len(X), probability=True), vector(cy, len(V), probability=True)
    if (threads not in (1, 2) or X.shape[1] != V.shape[1]
            or not np.isin(y, [0, 1]).all() or not np.isin(cy, [0, 1]).all()
            or min(np.sum(y), np.sum(1-y)) < 20 or min(np.sum(cy), np.sum(1-cy)) < 5):
        raise ValueError("Two chronological blocks with sufficient binary classes required.")
    with threadpool_limits(limits=threads):
        model = HistGradientBoostingClassifier(**PARAMETERS).fit(X, y)
        calibration = LogisticRegression(C=1., random_state=1729).fit(logit(model.predict_proba(V)[:, 1]), cy)
    return dict(model=model, calibration=calibration, columns=X.shape[1], threads=threads,
                prevalence=float(y.mean()), training_rows=len(y), calibration_rows=len(cy))


def predict_probability(state, X):
    X = matrix(X, state['columns'])
    if not len(X):
        return np.empty(0)
    with threadpool_limits(limits=state['threads']):
        p = state['calibration'].predict_proba(logit(state['model'].predict_proba(X)[:, 1]))[:, 1]
    return vector(p, len(X), probability=True)


def fit_activation_intensity(X, active, shadow, fuel, V, cal_active, *, threads=2):
    X = matrix(X)
    active = vector(active, len(X), probability=True)
    shadow, fuel = vector(shadow, len(X)), vector(fuel, len(X), positive=True)
    if (shadow < 0).any() or ((active == 0) & (shadow != 0)).any() or ((active == 1) & (shadow <= 0)).any():
        raise ValueError("Hourly activity and nonnegative shadow intensity are inconsistent.")
    probability = fit_probability(X, active, V, cal_active, threads=threads)
    positive = active == 1
    with threadpool_limits(limits=threads):
        intensity = HistGradientBoostingRegressor(loss='poisson', **PARAMETERS).fit(
            X[positive], shadow[positive]/fuel[positive])
    return dict(probability=probability, intensity=intensity, columns=X.shape[1], threads=threads,
                core_fuel_min=float(fuel.min()), core_fuel_max=float(fuel.max()),
                intensity_target='conditional_mean_hourly_shadow_over_cgc')


def predict_activation_intensity(state, X, fuel):
    X = matrix(X, state['columns'])
    fuel = vector(fuel, len(X), positive=True)
    p = predict_probability(state['probability'], X)
    with threadpool_limits(limits=state['threads']):
        conditional = state['intensity'].predict(X)*fuel if len(X) else np.empty(0)
    if not np.isfinite(conditional).all() or (conditional < 0).any():
        raise ValueError("Nonfinite or negative conditional intensity.")
    return dict(activation_probability=p, intensity_if_active=conditional,
                expected_shadow_price=p*conditional,
                climatology_probability=np.full(len(X), state['probability']['prevalence']))
