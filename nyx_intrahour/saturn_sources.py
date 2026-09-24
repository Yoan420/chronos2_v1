"""Read-only acquisition of the audited, native 15-minute Belgian solar forecast.

This isolated research adapter never resamples, fills or writes Saturn data.
The revision field records the requested as-of cutoff, not a provider issue time.
"""
from __future__ import annotations

from datetime import date, timedelta
import re
from typing import Any

import numpy as np
import pandas as pd
import requests

SATURN_URL = "https://saturn-energyscan.gem.myengie.com//api"
SOURCE = {
    "alias": "be_solar_elia_fcst", "zone": "BE", "driver": "solar", "unit": "MW",
    "series": "23259", "native_resolution_minutes": 15, "is_forecast": True,
    "interpolation": "none",
    "native_resolution_evidence": (
        "Saturn primary 23259, UTC-aware metadata and null formula; underlying source of "
        "power.stp.da_solar_production.be.mw.qh.fcst.elia. Raw as-of D-1 08:00 Europe/Paris "
        "samples on 2025-10-15, 2026-01-15, 2026-04-15, 2026-07-15 and 2026-09-15 "
        "each contain 96 consecutive 900-second values and within-hour variation. "
        "Read raw primary directly, never the wrapper's resample/ffill."
    ),
}


def civil_day(value: str) -> date:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise ValueError("An ISO calendar date YYYY-MM-DD is required.")
    return date.fromisoformat(value)


def day_grid(value: str) -> tuple[pd.DatetimeIndex, pd.Timestamp]:
    day = civil_day(value)
    start = pd.Timestamp(day).tz_localize("Europe/Paris").tz_convert("UTC")
    end = pd.Timestamp(day+timedelta(days=1)).tz_localize("Europe/Paris").tz_convert("UTC")
    cutoff = (pd.Timestamp(day-timedelta(days=1))+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    return pd.date_range(start, end, freq="15min", inclusive="left"), cutoff


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "source_alias": pd.Series(dtype="str"),
        "value_time_utc": pd.Series(dtype="datetime64[ns, UTC]"),
        "snapshot_time_utc": pd.Series(dtype="datetime64[ns, UTC]"),
        "revision_time_utc": pd.Series(dtype="datetime64[ns, UTC]"),
        "value": pd.Series(dtype="float64"),
    })


class ReadOnlySession(requests.Session):
    """Honor the inherited environment; reject every HTTP mutation."""
    def __init__(self, timeout_seconds: float = 45):
        super().__init__()
        if timeout_seconds <= 0:
            raise ValueError("A positive request timeout is required.")
        self.timeout_seconds = timeout_seconds

    def request(self, method: str, url: str, *args: Any, **kwargs: Any):
        if method.upper() != "GET":
            raise ValueError("Native source collection allows GET only.")
        kwargs.setdefault("timeout", (10, self.timeout_seconds))
        return super().request(method, url, *args, **kwargs)


def make_client(timeout_seconds: float = 45):
    from tshistory_lite import Client
    client = Client(SATURN_URL, author="nyx-intrahour-readonly")
    client.session.close()
    client.session = ReadOnlySession(timeout_seconds)
    return client


def fetch_day(client: Any, delivery_day: str) -> tuple[pd.DataFrame, dict]:
    """Read one historical forecast at the exact cutoff; retain real gaps."""
    grid, cutoff = day_grid(delivery_day)
    audit = {
        "delivery_day": delivery_day, "source_alias": SOURCE["alias"], "series": SOURCE["series"],
        "cutoff_time_utc": cutoff.isoformat(), "expected_quarters": len(grid),
        "query_from_utc": grid[0].isoformat(), "query_to_utc": grid[-1].isoformat(),
        "temporal_evidence": "retrospective_asof", "provider_revision_timestamp_available": False,
        "timestamp_semantics": "snapshot_time_utc and revision_time_utc equal the requested as-of cutoff, not actual publication timestamps",
        "downloaded_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    try:
        raw = client.get(SOURCE["series"], revision_date=cutoff,
                         from_value_date=grid[0], to_value_date=grid[-1], _keep_nans=True)
        if raw is None or len(raw) == 0:
            return empty_frame(), {**audit, "status": "empty", "returned_quarters": 0,
                                   "missing_quarters": len(grid)}
        if not isinstance(raw, pd.Series):
            raise ValueError("Expected a raw pandas series.")
        index = pd.DatetimeIndex(raw.index)
        if index.tz is None or index.hasnans or index.has_duplicates:
            raise ValueError("Native source requires unique aware physical timestamps.")
        index = index.tz_convert("UTC")
        if not index.equals(index.floor("15min")):
            raise ValueError("Off-grid source timestamps cannot be repaired.")
        numeric = pd.to_numeric(raw, errors="raise")
        if np.iscomplexobj(numeric):
            raise ValueError("Complex source values are not supported.")
        series = pd.Series(numeric.to_numpy(float), index=index).sort_index()
        outside = int((~series.index.isin(grid)).sum())
        series = series.loc[series.index.isin(grid)]
        if len(series) and set(series.index.minute) == {0}:
            raise ValueError("Only hourly timestamps returned; no native quarters can be inferred.")
        # Retain observed NaNs, and retain absences as absences. Feature extraction
        # must invalidate those hours instead of selecting or imputing old values.
        frame = pd.DataFrame({"source_alias": SOURCE["alias"], "value_time_utc": series.index,
                              "snapshot_time_utc": cutoff, "revision_time_utc": cutoff,
                              "value": series.to_numpy(float)})
        valid = int(np.isfinite(series.to_numpy()).sum())
        complete = len(series) == len(grid) and valid == len(grid)
        varying = sum(len(group) == 4 and np.isfinite(group.to_numpy()).all() and group.nunique() > 1
                      for _, group in series.groupby(series.index.floor("h")))
        return frame, {**audit, "status": "complete" if complete else "incomplete",
                       "returned_quarters": len(series), "finite_quarters": valid,
                       "missing_quarters": len(grid)-len(series), "nonfinite_quarters": len(series)-valid,
                       "outside_quarters_excluded": outside, "varying_hours": int(varying)}
    except Exception as exc:
        # Provider response/error text may contain credential-bearing proxy URLs.
        # Persist only a safe type, never exception text or HTTP response bodies.
        return empty_frame(), {**audit, "status": "error", "error_type": type(exc).__name__,
                               "returned_quarters": 0, "missing_quarters": len(grid)}
