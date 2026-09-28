"""Exact final paired Test2 reference selector from the annual CWE study.

This function needs six *already produced* forecast signals. It does not
produce NYX, Test2, the equal ensemble, or the prior-90-day decision, and is
therefore not a standalone future forecast pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


PARTNER = {"FR": "BE", "BE": "FR", "DE": "NL", "NL": "DE"}
SIGNALS = (
    "ensemble__q50",
    "nyx__q50",
    "test2__q50",
    "prior90_active",
    "spike_probability",
    "partner_nyx__q50",
)


def build_confirmed_pair(signals: pd.DataFrame, *, zone: str, partner_zone: str) -> pd.DataFrame:
    """Apply the frozen `scarcity_confirmed_pair` rule to aligned UTC hours.

    `prior90_active` must come from the original weekly, past-only policy.
    Input provenance and D-1 publication times must be checked by its future
    producers before this final selector is used in production.
    """
    if zone not in PARTNER or partner_zone != PARTNER[zone]:
        raise ValueError("The historical country partner must be supplied explicitly")
    if not isinstance(signals, pd.DataFrame) or tuple(signals.columns) != SIGNALS:
        raise ValueError("All six declared forecast signals are required in historical order")
    index = signals.index
    if (not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC"
            or not len(index) or index.hasnans or not index.is_unique
            or not index.is_monotonic_increasing or not index.equals(index.floor("h"))):
        raise ValueError("Unique ordered physical UTC delivery hours are required")
    if signals.prior90_active.dtype != bool or signals.prior90_active.isna().any():
        raise ValueError("An explicit boolean prior-90-day decision is required")
    numeric = signals.drop(columns="prior90_active")
    if (not np.isfinite(numeric.to_numpy(dtype=float)).all()
            or not signals.spike_probability.between(0., 1.).all()):
        raise ValueError("Finite forecasts and valid probabilities are required")

    base, nyx, test2, probability, neighbor = (
        signals[name].to_numpy(dtype=float)
        for name in ("ensemble__q50", "nyx__q50", "test2__q50",
                     "spike_probability", "partner_nyx__q50")
    )
    inherited = signals.prior90_active.to_numpy(dtype=bool)
    if np.any(inherited & ((test2 <= base) | (test2 <= nyx))):
        raise ValueError("The inherited upward Test2 decision is inconsistent")

    selected = inherited & (probability >= .95) & (np.maximum(nyx, neighbor) >= 300.)
    return pd.DataFrame({
        "scarcity_confirmed_pair": np.where(selected, test2, base),
        "selected_confirmed_pair": selected,
    }, index=index)
