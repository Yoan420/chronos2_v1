"""Conditional two-regime empirical CDFs; no forecast gates or price clipping.

The caller supplies exogenous-only X and core-only normalized errors e/u_zone.
The normal regime contains ALL errors below one, including negatives, while the
spike regime contains errors at or above one. A forest represents a CDF through
training-observation weights, not through its usual mean ``predict`` method.
The pre-existing calibrated event probability provides the two regime masses.

This is a quantile-regression-forest-style estimator with deterministic,
non-bootstrap trees and all training observations retained in their leaves.
Quantiles are left empirical inverses, never interpolated or averaged across
regimes. Their empirical support is preserved, without claiming calibration.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits


FOREST_PARAMETERS = {"n_estimators": 48, "max_depth": 6,
    "min_samples_leaf": 20, "max_features": .7, "bootstrap": False,
    "random_state": 1729}
MINIMUM_EMPIRICAL_ZONE_ROWS = 10


def _matrix(X, *, columns=None):
    values = np.asarray(X, dtype=float)
    if (values.ndim != 2 or not values.shape[1]
            or (columns is not None and values.shape[1] != columns)
            or np.isinf(values).any()):
        raise ValueError("X must be a two-dimensional finite-or-NaN matrix with the fitted columns.")
    return values


def _zones(values, rows):
    zones = np.asarray(values, dtype=object)
    if (zones.shape != (rows,) or any(not isinstance(z, str) or not z.strip() for z in zones)):
        raise ValueError("One non-empty country string per row is required.")
    return zones


def weighted_quantiles(support, weights, levels):
    """Inverse of a finite weighted CDF, with explicit q=0 and q=1 endpoints.

    Zero-mass observations are removed, so they cannot affect endpoint results.
    Tied values are valid atoms; no interpolation between observations occurs.
    """
    support, weights = np.asarray(support, float), np.asarray(weights, float)
    levels = np.asarray(levels, float)
    if (support.ndim != 1 or support.shape != weights.shape or not len(support)
            or not np.isfinite(support).all() or not np.isfinite(weights).all()
            or (weights < 0).any() or levels.ndim != 1
            or not np.isfinite(levels).all() or ((levels < 0)|(levels > 1)).any()):
        raise ValueError("Finite aligned support, nonnegative mass and levels in [0,1] required.")
    positive = weights > 0
    if not positive.any():
        raise ValueError("An empirical distribution must have positive mass.")
    values, mass = support[positive], weights[positive]
    if (np.diff(values) < 0).any():
        order = np.argsort(values, kind="stable")
        values, mass = values[order], mass[order]
    # Long-double accumulation limits a false crossing at exact empirical atoms.
    cumulative = np.cumsum(mass, dtype=np.longdouble)
    if not np.isfinite(cumulative[-1]) or cumulative[-1] <= 0:
        raise ValueError("Finite positive total CDF mass required.")
    cumulative /= cumulative[-1]
    cumulative[-1] = 1.
    positions = np.searchsorted(cumulative, levels.astype(np.longdouble), side="left")
    answer = values[np.minimum(positions, len(values)-1)]
    answer[levels == 0] = values[0]
    answer[levels == 1] = values[-1]
    return answer


def mixture_quantiles(normal_support, normal_weights, spike_support, spike_weights,
                      probability, levels=(.1, .5, .9)):
    """Exact left quantiles of disjoint normalized supports r<1 and r>=1."""
    normal_support, spike_support = np.asarray(normal_support, float), np.asarray(spike_support, float)
    levels = np.asarray(levels, float)
    probability = float(probability)
    if (not np.isfinite(probability) or not 0 <= probability <= 1
            or levels.ndim != 1 or not np.isfinite(levels).all()
            or ((levels <= 0)|(levels >= 1)).any()
            or (normal_support >= 1).any() or (spike_support < 1).any()):
        raise ValueError("Disjoint supports r<1/r>=1, p in [0,1] and interior quantile levels required.")
    # Validate both distributions even when p is exactly zero or one, without
    # sorting/accumulating unused supports at every inference hour.
    for support, weights in ((normal_support, normal_weights), (spike_support, spike_weights)):
        weights = np.asarray(weights, float)
        if (support.ndim != 1 or not len(support) or support.shape != weights.shape
                or not np.isfinite(support).all() or not np.isfinite(weights).all()
                or (weights < 0).any() or not (weights > 0).any()):
            raise ValueError("Both regimes must have finite aligned support and positive nonnegative mass.")
    result = np.empty(len(levels), float)
    normal = levels <= 1.-probability
    if normal.any():
        result[normal] = weighted_quantiles(normal_support, normal_weights,
            np.clip(levels[normal]/(1.-probability), 0., 1.))
    if (~normal).any():
        result[~normal] = weighted_quantiles(spike_support, spike_weights,
            np.clip((levels[~normal]-(1.-probability))/probability, 0., 1.))
    return result


def fit_distributions(X, normalized_errors, zones, kind="forest", threads=2):
    """Fit two distributions on the supplied core; never split or add labels.

    X must already include any desired country one-hot columns. The policy owns
    feature allowlists, the causal core selection and the fitted u_zone values.
    For the empirical control a country/regime distribution requires ten rows;
    otherwise the normalized same-regime pool is used, never another regime.
    """
    if kind not in ("forest", "empirical") or isinstance(threads, bool) or threads not in (1, 2):
        raise ValueError("Use forest/empirical with one or two threads.")
    X = _matrix(X)
    errors, zones = np.asarray(normalized_errors, float), _zones(zones, len(X))
    if (not len(X) or errors.shape != (len(X),) or not np.isfinite(errors).all()
            or not (errors < 1).any() or not (errors >= 1).any()):
        raise ValueError("Finite core errors with non-empty normal and spike regimes are required.")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    transformed = imputer.fit_transform(X)
    known_zones = tuple(sorted(set(zones)))
    state = {"kind": kind, "imputer": imputer, "feature_count": X.shape[1],
        "known_zones": known_zones, "threads": int(threads), "core_rows": len(X),
        "regimes": {}, "forest_parameters": dict(FOREST_PARAMETERS),
        "minimum_empirical_zone_rows": MINIMUM_EMPIRICAL_ZONE_ROWS}
    with threadpool_limits(limits=int(threads)):
        for name, mask in (("normal", errors < 1), ("spike", errors >= 1)):
            values, countries = errors[mask], zones[mask]
            order = np.argsort(values, kind="stable")
            distribution = {"support": values[order], "core_rows": int(mask.sum()),
                "core_rows_by_zone": {z: int(np.count_nonzero(countries == z)) for z in known_zones}}
            if kind == "forest":
                forest = RandomForestRegressor(n_jobs=int(threads), **FOREST_PARAMETERS)
                forest.fit(transformed[mask], values)
                leaves = forest.apply(transformed[mask])
                rank = np.empty(len(order), dtype=np.int64)
                rank[order] = np.arange(len(order))
                members = []
                for tree in range(leaves.shape[1]):
                    members.append({int(leaf): np.sort(rank[np.flatnonzero(leaves[:, tree] == leaf)])
                        for leaf in np.unique(leaves[:, tree])})
                distribution.update(forest=forest, leaf_members=members)
            else:
                distribution["zone_supports"] = {z: np.sort(values[countries == z], kind="stable")
                    for z in known_zones if np.count_nonzero(countries == z) >= MINIMUM_EMPIRICAL_ZONE_ROWS}
            state["regimes"][name] = distribution
    return state


def leaf_weights(distribution, leaves):
    """Observation weights sum to one: average uniform CDF within each leaf."""
    members = distribution["leaf_members"]
    leaves = np.asarray(leaves)
    if leaves.shape != (len(members),) or not len(members):
        raise ValueError("One fitted leaf identifier per tree is required.")
    weights = np.zeros(len(distribution["support"]), float)
    for tree, leaf in enumerate(leaves):
        indices = members[tree].get(int(leaf))
        if indices is None or not len(indices):
            raise ValueError("A prediction reached an unknown or empty fitted leaf.")
        weights[indices] += 1./(len(members)*len(indices))
    total = weights.sum()
    if not np.isfinite(total) or not np.isclose(total, 1., rtol=1e-12, atol=1e-12):
        raise ValueError("Forest observation weights must sum to one.")
    return weights/total


def predict_quantiles(state, X, zones, probabilities, thresholds, levels=(.1, .5, .9)):
    """Return EUR/MWh error quantiles and row-aligned distribution diagnostics."""
    X = _matrix(X, columns=state["feature_count"])
    zones = _zones(zones, len(X))
    probabilities, thresholds, levels = (np.asarray(v, float) for v in (probabilities, thresholds, levels))
    if (probabilities.shape != (len(X),) or thresholds.shape != (len(X),)
            or not np.isfinite(probabilities).all() or ((probabilities < 0)|(probabilities > 1)).any()
            or not np.isfinite(thresholds).all() or (thresholds <= 0).any()
            or levels.ndim != 1 or not len(levels) or not np.isfinite(levels).all()
            or ((levels <= 0)|(levels >= 1)).any() or (np.diff(levels) <= 0).any()
            or not set(zones).issubset(state["known_zones"])):
        raise ValueError("Known countries, aligned p/seuil, positive finite thresholds and increasing interior levels required.")
    output = np.empty((len(X), len(levels)), float)
    diagnostics = {"mixture_spike_probability": probabilities.copy(),
        "mixture_spike_threshold_eur_mwh": thresholds.copy()}
    for name in ("normal", "spike"):
        diagnostics[name+"_effective_sample_size"] = np.empty(len(X), float)
        diagnostics[name+"_support_count"] = np.empty(len(X), int)
        diagnostics[name+"_cdf_source"] = np.empty(len(X), object)
    if not len(X):
        return output, diagnostics
    with threadpool_limits(limits=state["threads"]):
        transformed = state["imputer"].transform(X)
        leaves = ({name: distribution["forest"].apply(transformed)
            for name, distribution in state["regimes"].items()} if state["kind"] == "forest" else {})
        for row, zone in enumerate(zones):
            cdfs = {}
            for name in ("normal", "spike"):
                distribution = state["regimes"][name]
                if state["kind"] == "forest":
                    support, weights = distribution["support"], leaf_weights(distribution, leaves[name][row])
                    source = "forest_leaf_conditional"
                else:
                    local = distribution["zone_supports"].get(zone)
                    support = distribution["support"] if local is None else local
                    weights = np.full(len(support), 1./len(support))
                    source = "empirical_pooled_normalized" if local is None else "empirical_country"
                cdfs[name] = support, weights
                diagnostics[name+"_effective_sample_size"][row] = 1./np.dot(weights, weights)
                diagnostics[name+"_support_count"][row] = np.count_nonzero(weights > 0)
                diagnostics[name+"_cdf_source"][row] = source
            output[row] = thresholds[row]*mixture_quantiles(*cdfs["normal"], *cdfs["spike"], probabilities[row], levels)
    if not np.isfinite(output).all() or (np.diff(output, axis=1) < 0).any():
        raise ValueError("Conditional mixture quantiles must remain finite and ordered.")
    return output, diagnostics


__all__ = ["fit_distributions", "predict_quantiles", "mixture_quantiles", "weighted_quantiles", "leaf_weights"]
