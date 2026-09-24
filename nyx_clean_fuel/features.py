"""Day-ahead-known clean fuel indices enter the residual learner, not Chronos.

Native Saturn indices already include carbon. Never add EUA a second time.
These market benchmarks are neither plant-specific offers nor a price floor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ALIASES = ("cgc_fr", "cgc_de", "cgc_be", "cgc_nl", "ccc")
PREFIX = "known_clean_fuel_"


def build_features(bank: pd.DataFrame, index: pd.DatetimeIndex, *, zone: str,
                   timezone: str, base: pd.DataFrame) -> pd.DataFrame:
    """Broadcast audited daily costs onto physical hours, with exact alignment.

    ``bank`` must first pass sources.load_bank. Recheck causal fields here so
    direct callers cannot accidentally use a future observation or wrong day.
    No backward fill, interpolation, missing-source substitution or labels.
    """
    zone = zone.upper()
    if zone not in {"FR", "DE", "BE", "NL"}:
        raise ValueError("Clean fuel correction currently supports FR, DE, BE and NL.")
    if (not isinstance(index, pd.DatetimeIndex) or index.tz is None or index.hasnans
            or not index.is_unique or not index.is_monotonic_increasing or index.empty
            or not index.equals(index.floor("h"))):
        raise ValueError("A sorted, unique, timezone-aware physical hourly index is required.")
    if not base.index.equals(index) or "q50" not in base:
        raise ValueError("Raw Chronos predictions must exactly match the requested feature index.")
    if not np.isfinite(base.q50.to_numpy(float)).all():
        raise ValueError("Raw Chronos P50 must be finite.")
    days = index.tz_convert(timezone).strftime("%Y-%m-%d")
    if bank.delivery_day.duplicated().any():
        raise ValueError("Duplicate clean fuel delivery days.")
    daily = bank.set_index("delivery_day")
    missing = sorted(set(days) - set(daily.index))
    if missing:
        raise ValueError(f"Clean fuel history incomplete: {missing[:5]}.")
    selected = daily.loc[days].copy()
    selected.index = index
    expected = pd.DatetimeIndex([
        (pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=8))
        .tz_localize(timezone).tz_convert("UTC") for day in days
    ])
    actual = pd.DatetimeIndex(pd.to_datetime(selected.cutoff_time_utc, utc=True))
    if not actual.equals(expected):
        raise ValueError("Clean fuel cutoff must equal civil D-1 08:00.")
    costs = selected.loc[:, ALIASES].astype(float)
    if not np.isfinite(costs.to_numpy()).all():
        raise ValueError("Clean fuel costs are incomplete or nonfinite; no substitution allowed.")
    output = costs.rename(columns=lambda col: PREFIX + col)
    for alias in ALIASES:
        stamp = pd.DatetimeIndex(pd.to_datetime(selected[alias + "__value_time_utc"], utc=True))
        if stamp.hasnans or (stamp.tz_convert(timezone).date >= expected.tz_convert(timezone).date).any():
            raise ValueError("Same-day or future market closes cannot enter a forecast made at 08:00.")
        age = (expected - stamp).total_seconds() / 3600
        if (age < 0).any() or (age > 176).any():
            raise ValueError("Clean fuel source exceeds the maximum 176-hour age.")
        if not np.allclose(age, selected[alias + "__age_hours"].to_numpy(float), rtol=0, atol=1e-9):
            raise ValueError("Clean fuel source age/audit mismatch.")
        output[PREFIX + alias + "_age_hours"] = age
    local = costs["cgc_" + zone.lower()]
    output[PREFIX + "local_gas_minus_coal"] = local - costs.ccc
    output[PREFIX + "local_minus_ttf_cgc"] = local - costs.cgc_nl
    output[PREFIX + "regional_cgc_spread"] = costs.iloc[:, :4].max(axis=1) - costs.iloc[:, :4].min(axis=1)
    output[PREFIX + "chronos_minus_local_cgc"] = base.q50 - local
    output[PREFIX + "chronos_minus_ccc"] = base.q50 - costs.ccc
    return output
