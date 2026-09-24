"""Construct matched Chronos inputs without exposing delivery-day prices."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nyx_intrahour.data import HOURLY_ALIASES, ZONES


CALENDAR_COLUMNS = ("known_hour_sin", "known_hour_cos", "known_dow_sin",
                    "known_dow_cos", "known_doy_sin", "known_doy_cos",
                    "known_is_weekend", "known_utc_offset_hours")


def utc_index(values, *, frequency: str) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(values)
    if index.tz is None or index.hasnans or index.has_duplicates:
        raise ValueError("Unique timezone-aware physical timestamps required.")
    index = index.tz_convert("UTC")
    if not index.is_monotonic_increasing or not index.equals(index.floor(frequency)):
        raise ValueError("Timestamps must be ordered and aligned to the declared cadence.")
    return index


def day_index(day: str, frequency: str) -> pd.DatetimeIndex:
    date = pd.Timestamp(day)
    if date.tzinfo is not None or date != date.normalize():
        raise ValueError("A naive civil date is required.")
    start = date.tz_localize("Europe/Paris")
    end = (date+pd.Timedelta(days=1)).tz_localize("Europe/Paris")
    return pd.date_range(start, end, freq=frequency, inclusive="left").tz_convert("UTC")


def calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_convert("Europe/Paris")
    hour = local.hour.to_numpy(float)+local.minute.to_numpy(float)/60
    dow, doy = local.dayofweek.to_numpy(float), local.dayofyear.to_numpy(float)
    return pd.DataFrame({
        "known_hour_sin": np.sin(2*np.pi*hour/24),
        "known_hour_cos": np.cos(2*np.pi*hour/24),
        "known_dow_sin": np.sin(2*np.pi*dow/7),
        "known_dow_cos": np.cos(2*np.pi*dow/7),
        "known_doy_sin": np.sin(2*np.pi*doy/365.25),
        "known_doy_cos": np.cos(2*np.pi*doy/365.25),
        "known_is_weekend": (dow >= 5).astype(float),
        "known_utc_offset_hours": [t.utcoffset().total_seconds()/3600 for t in local],
    }, index=index, dtype=float)


class MatchedInputs:
    """One immutable numerical panel; each origin slices only prior targets.

    Day-ahead D-1 prices are admitted through the end of D-1, consistently for
    both cadences. They are retrospective observations, not certified historical
    publication vintages. No delivery-D target enters a model context or future.
    """
    def __init__(self, native_prices: pd.DataFrame, baseline: pd.DataFrame):
        required = {"timestamp_utc", "zone", "actual_15m"}
        if not required.issubset(native_prices) or set(native_prices.zone) != set(ZONES):
            raise ValueError("Four-zone native price panel required.")
        if native_prices.duplicated(["timestamp_utc", "zone"]).any():
            raise ValueError("Duplicate native price key.")
        self.quarters, self.hours, self.covariates = {}, {}, {}
        for zone in ZONES:
            frame = native_prices.loc[native_prices.zone.eq(zone)].sort_values("timestamp_utc")
            index = utc_index(frame.timestamp_utc, frequency="15min")
            values = pd.to_numeric(frame.actual_15m, errors="raise").to_numpy(float)
            expected = pd.date_range(index[0], index[-1], freq="15min")
            if not index.equals(expected) or not np.isfinite(values).all():
                raise ValueError(f"{zone}: incomplete native quarter-hour target history.")
            target = pd.Series(values, index=index, name="target")
            grouped = target.groupby(target.index.floor("h"))
            if not grouped.size().eq(4).all():
                raise ValueError("Partial physical hour in source history.")
            self.quarters[zone], self.hours[zone] = target, grouped.mean()
            panel = baseline.loc[baseline.zone.eq(zone)].sort_values("timestamp_utc")
            cov_index = utc_index(panel.timestamp_utc, frequency="h")
            columns = [f"feature_hourly_{alias}" for alias in HOURLY_ALIASES]
            cov = panel[columns].copy()
            cov.columns = [f"known_{alias}" for alias in HOURLY_ALIASES]
            cov.index = cov_index
            if not np.isfinite(cov.to_numpy(float)).all():
                raise ValueError("All archived hourly fundamentals must be finite.")
            self.covariates[zone] = cov
        if any(not self.quarters[z].index.equals(self.quarters[ZONES[0]].index) for z in ZONES):
            raise ValueError("Native price histories must have identical four-zone grids.")

    def _known(self, index: pd.DatetimeIndex, zone: str) -> pd.DataFrame:
        # Hourly forecast fundamentals are explicitly constant inside each hour.
        # This is not a claim that these covariates have native 15-minute detail.
        cov = self.covariates[zone].reindex(index.floor("h")).copy()
        cov.index = index
        if not np.isfinite(cov.to_numpy(float)).all():
            raise ValueError("Archived hourly fundamentals do not cover this origin.")
        return pd.concat([cov, calendar(index)], axis=1)

    def build_origin(self, delivery_day: str, frequency: str, *, context_hours: int = 2048):
        if frequency not in {"h", "15min"}:
            raise ValueError("Only the paired hourly and native-quarter-hour cadences are supported.")
        if type(context_hours) is not int or context_hours < 1:
            raise ValueError("context_hours must be positive.")
        future_index = day_index(delivery_day, frequency)
        count = context_hours*(4 if frequency == "15min" else 1)
        offset = pd.tseries.frequencies.to_offset(frequency)
        history_index = pd.date_range(end=future_index[0]-offset, periods=count, freq=frequency)
        contexts, futures, audit = [], [], []
        for zone in ZONES:
            source = self.quarters[zone] if frequency == "15min" else self.hours[zone]
            target = source.reindex(history_index)
            if not np.isfinite(target.to_numpy(float)).all():
                raise ValueError(f"{zone}: insufficient genuine history for {delivery_day}/{frequency}.")
            item = f"{zone}_{delivery_day}_{frequency}"
            context = self._known(history_index, zone).reset_index(drop=True)
            context.insert(0, "target", target.to_numpy(float))
            context.insert(0, "timestamp", history_index.tz_localize(None))
            context.insert(0, "item_id", item)
            future = self._known(future_index, zone).reset_index(drop=True)
            future.insert(0, "timestamp", future_index.tz_localize(None))
            future.insert(0, "item_id", item)
            contexts.append(context)
            futures.append(future)
            cutoff = ((pd.Timestamp(delivery_day)-pd.Timedelta(days=1))+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
            audit.append({"zone": zone, "delivery_day": delivery_day, "frequency": frequency,
                          "forecast_origin_utc": cutoff.isoformat(),
                          "context_first_utc": history_index[0].isoformat(),
                          "context_last_utc": history_index[-1].isoformat(),
                          "context_points": count, "context_hours": context_hours,
                          "horizon_points": len(future_index),
                          "last_price_delivery_day": (pd.Timestamp(delivery_day)-pd.Timedelta(days=1)).date().isoformat(),
                          "target_policy": "previous_day_day_ahead_prices_through_day_end",
                          "future_realized_target_used": False,
                          "publication_vintage_verified": False,
                          "hourly_fundamentals_repeated_within_hour": frequency == "15min"})
        return contexts, futures, audit

    def evaluation_baseline(self, baseline: pd.DataFrame, start_day: str, end_day: str):
        lower = day_index(start_day, "h")[0]
        upper = day_index(end_day, "h")[-1]
        frame = baseline.loc[baseline.timestamp_utc.between(lower, upper),
                             ["timestamp_utc", "zone", "actual", "nyx_q50"]].copy()
        frame.rename(columns={"actual": "archived_hourly_actual"}, inplace=True)
        frame["actual"] = [self.hours[z].loc[t] for z, t in zip(frame.zone, frame.timestamp_utc)]
        delta = frame.actual-frame.archived_hourly_actual
        audit = {"scoring_target": "arithmetic_mean_of_four_native_quarter_hour_prices",
                 "compared_rows": len(frame), "max_abs_difference_from_archived_hourly": float(delta.abs().max()),
                 "mean_abs_difference_from_archived_hourly": float(delta.abs().mean()),
                 "differences_over_001_eur_mwh": int(delta.abs().gt(.01).sum())}
        return frame, audit
