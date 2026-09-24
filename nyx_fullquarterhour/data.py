"""Matched physical inputs for the complete experimental NYX chain.

The hourly schema mirrors ``prepare_zone_data``/``build_origin_frames``:
six past-only base aliases, seven known calendar columns and six known oracle
aliases. "Oracle" is the incumbent feature name for archived forecast drivers,
not a realized future-price input. At 15 minutes the six hourly fundamentals
remain explicitly constant within each physical hour; only the target is native.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from chronos2_hourly.features import build_calendar_features
from chronos2_modular.common import CALENDAR_COLUMNS
from nyx_intrahour.data import HOURLY_ALIASES, ZONES
from nyx_quarterhour.data import MatchedInputs, day_index, utc_index


ZONE_TIMEZONES = {"BE": "Europe/Brussels", "DE": "Europe/Berlin",
                  "FR": "Europe/Paris", "NL": "Europe/Amsterdam"}
ORACLE_COLUMNS = tuple(f"known_{alias}_oracle" for alias in HOURLY_ALIASES)
FUTURE_COLUMNS = (*CALENDAR_COLUMNS, *ORACLE_COLUMNS)
CONTEXT_COLUMNS = (*HOURLY_ALIASES, *FUTURE_COLUMNS)


def _index(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    if frequency not in {"h", "15min"}:
        raise ValueError("Only h and 15min physical cadences are supported.")
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        raise ValueError("A nonempty DatetimeIndex is required.")
    result = utc_index(index, frequency=frequency)
    if not result.equals(pd.date_range(result[0], result[-1], freq=frequency)):
        raise ValueError("A complete contiguous physical grid is required; no filling.")
    return result


def calendar(index: pd.DatetimeIndex, *, timezone: str = "Europe/Paris") -> pd.DataFrame:
    """The production seven float32 known features, with fractional local hour.

    This intentionally retains dayofyear/365.25, as in the Chronos production
    calendar; the residual calendar uses its separate (dayofyear-1)/365.2425.
    No eighth UTC-offset feature is added to the Chronos input schema.
    """
    if index.empty or index.tz is None or index.hasnans:
        raise ValueError("A nonempty aware calendar index is required.")
    local = index.tz_convert(timezone)
    hour = local.hour.to_numpy(dtype=float) + local.minute.to_numpy(dtype=float)/60
    weekday = local.dayofweek.to_numpy(dtype=float)
    dayofyear = local.dayofyear.to_numpy(dtype=float)
    return pd.DataFrame({
        "known_hour_sin": np.sin(2*np.pi*hour/24),
        "known_hour_cos": np.cos(2*np.pi*hour/24),
        "known_dow_sin": np.sin(2*np.pi*weekday/7),
        "known_dow_cos": np.cos(2*np.pi*weekday/7),
        "known_doy_sin": np.sin(2*np.pi*dayofyear/365.25),
        "known_doy_cos": np.cos(2*np.pi*dayofyear/365.25),
        "known_is_weekend": (weekday >= 5).astype(np.float32),
    }, index=index, dtype=np.float32)


def residual_calendar(index: pd.DatetimeIndex, *, frequency: str,
                      timezone: str = "Europe/Paris") -> pd.DataFrame:
    """Production residual calendar including fold/offset, at either cadence."""
    index = _index(index, frequency)
    hours = index.floor("h")
    result = build_calendar_features(hours.unique(), timezone=timezone).reindex(hours)
    result.index = index
    if frequency == "15min":
        local = index.tz_convert(timezone)
        hour = local.hour.to_numpy(dtype=float) + local.minute.to_numpy(dtype=float)/60
        result["calendar_local_hour"] = hour
        result["calendar_hour_sin"] = np.sin(2*np.pi*hour/24)
        result["calendar_hour_cos"] = np.cos(2*np.pi*hour/24)
    return result


class FullChainInputs(MatchedInputs):
    """Strict native targets and production-schema, matched-cadence drivers.

    Inputs are retrospective, not certified publication vintages. D-1 day-ahead
    prices through D-1's end are allowed at D-1 08:00, the same convention as
    NYX. Delivery-D prices never enter any context, future or residual feature.
    """

    def __init__(self, native_prices: pd.DataFrame, baseline: pd.DataFrame):
        required = {"timestamp_utc", "zone", *(
            f"feature_hourly_{alias}" for alias in HOURLY_ALIASES)}
        if (native_prices.empty or baseline.empty or not required.issubset(baseline)
                or set(baseline.zone) != set(ZONES)):
            raise ValueError("Complete four-zone native prices and archived fundamentals required.")
        super().__init__(native_prices, baseline)

    def _known(self, index: pd.DatetimeIndex, zone: str) -> pd.DataFrame:
        if zone not in ZONE_TIMEZONES:
            raise ValueError("Unknown NYX zone.")
        covariates = self.covariates[zone].reindex(index.floor("h")).copy()
        covariates.index = index
        if not np.isfinite(covariates.to_numpy(dtype=float)).all():
            raise ValueError("Archived hourly fundamentals do not cover this origin.")
        covariates.columns = list(ORACLE_COLUMNS)
        known = pd.concat([calendar(index, timezone=ZONE_TIMEZONES[zone]), covariates], axis=1)
        return known.astype(np.float32)

    def build_origin(self, delivery_day: str, frequency: str, *, context_hours: int = 2048):
        if not isinstance(delivery_day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", delivery_day):
            raise ValueError("Delivery day must be an ISO civil date YYYY-MM-DD.")
        contexts, futures, audits = super().build_origin(
            delivery_day, frequency, context_hours=context_hours)
        for context, future, audit in zip(contexts, futures, audits):
            context["target"] = context.target.astype(np.float32)
            for position, alias in enumerate(HOURLY_ALIASES, start=3):
                # prepare_zone_data constructs both names from the same base;
                # the prepared 16/09 archive confirms exact equality in all zones.
                context.insert(position, alias, context[f"known_{alias}_oracle"].to_numpy(copy=True))
            audit.update({
                "input_schema": "nyx_production_19_context_13_future",
                "context_covariate_columns": list(CONTEXT_COLUMNS),
                "future_covariate_columns": list(FUTURE_COLUMNS),
                "past_only_covariate_columns": list(HOURLY_ALIASES),
                "calendar_policy": "production_seven_known_fractional_local_hour",
                "fundamental_native_resolution": "hourly",
                "native_quarter_hour_target": frequency == "15min",
                "model_numeric_dtype": "float32",
            })
        return contexts, futures, audits

    def residual_features(self, index: pd.DatetimeIndex, zone: str,
                          frequency: str) -> pd.DataFrame:
        """Return production X's 25 non-price columns on a physical UTC grid.

        The incumbent first constructs 41 columns, then excludes all 16 price
        lags/rolling statistics and four day-of-year columns in its residual
        builder. The 16 excluded price features are deliberately never read or
        constructed here. All 25 non-price columns (including day-of-year) are
        supplied, so that the builder applies the identical retained schema.
        """
        index = _index(index, frequency)
        known = self._known(index, zone)
        deterministic = residual_calendar(index, frequency=frequency,
                                           timezone=ZONE_TIMEZONES[zone])
        return pd.concat([known, deterministic], axis=1)

