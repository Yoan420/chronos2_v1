"""Fuel-transported tail CDF, with a genuine mixture median.

The classifier's event remains error >= u_zone. Its probability and the entire
normal-regime distribution are unchanged. Only the excess *above* this fixed
threshold is transported: u_current + CGC_current * (error_train-u_train)/CGC_train.
This is an explicit location/scale hypothesis, not a dispatch model or a claim
that gas explains scarcity. Forests condition observation weights; they are
never asked to extrapolate their ordinary leaf-mean prediction.
"""
from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from nyx_coherent_p50.distribution import (
    fit_distributions, leaf_weights, mixture_quantiles, _matrix, _zones,
)


def validate_options(options=None):
    options = {} if options is None else options
    if not isinstance(options, dict) or set(options)-{"threads"}:
        raise ValueError("Only the explicit threads option is supported.")
    threads = options.get("threads", 2)
    if type(threads) is not int or threads not in (1, 2):
        raise ValueError("Use one or two CPU threads.")
    return {"threads": threads}


def _aligned(values, rows, name, *, positive=False):
    values = np.asarray(values, float)
    if (values.shape != (rows,) or not np.isfinite(values).all()
            or (positive and (values <= 0).any())):
        raise ValueError(f"Aligned finite {name}{' > 0' if positive else ''} required.")
    return values


def fit_physical_cdf(X, errors, thresholds, fuel, zones, *, threads=2):
    X = _matrix(X)
    errors = _aligned(errors, len(X), "signed errors")
    thresholds = _aligned(thresholds, len(X), "thresholds", positive=True)
    fuel = _aligned(fuel, len(X), "as-of clean gas cost", positive=True)
    zones = _zones(zones, len(X))
    threads = validate_options({"threads": threads})["threads"]
    normalized = errors/thresholds
    state = fit_distributions(X, normalized, zones, kind="forest", threads=threads)
    spike = normalized >= 1
    order = np.argsort(normalized[spike], kind="stable")
    tail = state["regimes"]["spike"]
    # Same stable ordering as fit_distributions: leaf-member positions remain
    # observation identities even though transported values can change order.
    tail["excess_per_fuel"] = ((errors[spike]-thresholds[spike])/fuel[spike])[order]
    tail["training_fuel"] = fuel[spike][order]
    state["fuel_training_min"] = float(fuel.min())
    state["fuel_training_max"] = float(fuel.max())
    state["transport"] = "threshold_plus_current_cgc_times_core_excess_per_core_cgc"
    return state


def predict_physical_cdf(state, X, zones, probabilities, thresholds, fuel, levels=(.1, .5, .9)):
    X = _matrix(X, columns=state["feature_count"])
    zones = _zones(zones, len(X))
    p = _aligned(probabilities, len(X), "probabilities")
    u = _aligned(thresholds, len(X), "thresholds", positive=True)
    g = _aligned(fuel, len(X), "as-of clean gas cost", positive=True)
    levels = np.asarray(levels, float)
    if ((p < 0).any() or (p > 1).any() or not set(zones).issubset(state["known_zones"])
            or levels.ndim != 1 or not len(levels) or not np.isfinite(levels).all()
            or (levels <= 0).any() or (levels >= 1).any() or (np.diff(levels) <= 0).any()):
        raise ValueError("Known countries, probabilities in [0,1], increasing interior quantiles required.")
    quantiles = np.empty((len(X), len(levels)), float)
    diagnostics = {
        "fuel_outside_core_support": (g < state["fuel_training_min"]) | (g > state["fuel_training_max"]),
        "fuel_ratio_to_core_max": g/state["fuel_training_max"],
        "normal_effective_sample_size": np.empty(len(X)),
        "spike_effective_sample_size": np.empty(len(X)),
    }
    if not len(X):
        return quantiles, diagnostics
    with threadpool_limits(limits=state["threads"]):
        transformed = state["imputer"].transform(X)
        leaves = {name: distribution["forest"].apply(transformed)
                  for name, distribution in state["regimes"].items()}
        normal, spike = (state["regimes"][name] for name in ("normal", "spike"))
        for row in range(len(X)):
            weights = {name: leaf_weights(distribution, leaves[name][row])
                       for name, distribution in state["regimes"].items()}
            # Convert to normalized coordinates only for the existing exact
            # CDF inverter; transporting excess never moves a tail atom below u.
            tail_support = 1.+g[row]/u[row]*spike["excess_per_fuel"]
            quantiles[row] = u[row]*mixture_quantiles(normal["support"], weights["normal"],
                tail_support, weights["spike"], p[row], levels)
            for name in weights:
                diagnostics[name+"_effective_sample_size"][row] = 1./np.dot(weights[name], weights[name])
    if not np.isfinite(quantiles).all() or (np.diff(quantiles, axis=1) < 0).any():
        raise ValueError("Transported CDF returned invalid ordered quantiles.")
    return quantiles, diagnostics
