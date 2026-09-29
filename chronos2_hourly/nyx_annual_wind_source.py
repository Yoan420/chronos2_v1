"""The two audited native NL wind substitutions used by the historical inputs.

Only a missing 02:00 UTC product on the two approved spring dates may use
the ECMWF component, queried at the identical own delivery origin. This is
source substitution, not interpolation or certified provider publication.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from materialize_saturn_kalman_weather import SeriesPlan
from . import solar_wind_sources as historical


ALIAS = "nl_wind_generation_fcst"
POLICY = historical.NL_SPRING_GAP_POLICY
COMPONENT = historical.NL_ECMWF_COMPONENT


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _origin(day):
    return (pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize(
        "Europe/Amsterdam").tz_convert("UTC")


def _native_digest(series):
    digest = hashlib.sha256(series.index.asi8.astype("<i8").tobytes())
    digest.update(series.to_numpy(dtype="<f8").tobytes())
    return digest.hexdigest()


def verify_nl_wind_substitutions(series, substitutions, day, query_revision):
    """Recheck the exact scope, same origin, unit conversion and native values."""
    _require(isinstance(substitutions, list), "NL wind substitution audit must be a list")
    if not substitutions:
        return
    hour = historical.NL_SPRING_GAP_HOURS.get(day)
    _require(hour is not None and len(substitutions) == 1 and pd.Timestamp(query_revision) == _origin(day),
             "NL wind substitution requires the approved hour and its own historical origin")
    expected = pd.date_range(pd.Timestamp(day, tz="Europe/Amsterdam"),
        pd.Timestamp(day, tz="Europe/Amsterdam") + pd.DateOffset(days=1), freq="h", inclusive="left").tz_convert("UTC")
    _require(isinstance(series.index, pd.DatetimeIndex) and series.index.tz is not None
             and series.index.tz_convert("UTC").equals(expected) and series.index.is_unique
             and np.isfinite(series.to_numpy(float)).all() and (series.to_numpy(float) >= 0).all(),
             "NL wind substitution has an invalid completed physical-hour grid")
    record = substitutions[0]
    _require(record.get("native_gap_kind") in ("absent", "nan"), "NL wind native gap evidence differs")
    _require(record.get("native_values_sha256") == _native_digest(series.drop(hour)),
             "NL wind substitution changed the native forecast products")
    plan = SeriesPlan("NL", "wind", ALIAS, historical.WIND_SERIES[ALIAS],
        "Europe/Amsterdam", "Europe/Amsterdam", Path("unused"), unit="GW")
    frame = pd.DataFrame({"value_time_utc": series.index, "value": series.to_numpy(float),
        "snapshot_time_utc": _origin(day), "revision_time_utc": _origin(day),
        "downloaded_at_utc": pd.Timestamp(record["fallback_downloaded_at_utc"])})
    frame.attrs["source_substitutions"] = substitutions
    historical._validate_substitutions(plan, frame, {"source_substitutions": substitutions,
        "source_substitution_count": 1, "wind_gap_policy": POLICY}, wind_gap_policy=POLICY)


def normalize_nl_wind_profile(client, series, day, expected, query_revision):
    """Return native values unless the exact previously approved gap is present."""
    hour = historical.NL_SPRING_GAP_HOURS.get(day)
    if hour is None or pd.Timestamp(query_revision) != _origin(day):
        return series, []
    _require(isinstance(series.index, pd.DatetimeIndex) and series.index.tz is not None
             and series.index.is_unique, "NL wind native timestamps must be explicit and unique")
    native = series.copy()
    native.index = native.index.tz_convert("UTC")
    native = native.loc[(native.index >= expected[0]) & (native.index <= expected[-1])].sort_index()
    # The historical daily downloader drops NaN products before checking the
    # grid. Match that exact authorized-hour case, never infinity or another gap.
    gap_kind = "nan" if hour in native.index and pd.isna(native.loc[hour]) else "absent"
    if gap_kind == "nan":
        native = native.drop(hour)
    _require(native.index.isin(expected).all() and np.isfinite(native.to_numpy(float)).all()
             and (native.to_numpy(float) >= 0).all(), "NL wind native profile has invalid values or timestamps")
    missing = expected.difference(native.index)
    if len(missing) == 0:
        return series, []  # A complete native profile always wins.
    _require(missing.equals(pd.DatetimeIndex([hour])), "NL wind gaps differ from the single approved spring hour")
    raw = client.get(COMPONENT, from_value_date=hour, to_value_date=hour + pd.Timedelta(hours=1),
                     revision_date=pd.Timestamp(query_revision), nocache=True)
    _require(isinstance(raw, pd.Series) and isinstance(raw.index, pd.DatetimeIndex)
             and raw.index.tz is not None and raw.index.is_unique,
             "NL wind ECMWF component needs unique explicit UTC-aware timestamps")
    selected = raw.loc[raw.index.tz_convert("UTC") == hour]
    _require(len(selected) == 1, "NL wind ECMWF component lacks the approved physical hour")
    value = float(pd.to_numeric(selected, errors="raise").iloc[0])
    _require(np.isfinite(value) and value >= 0, "NL wind ECMWF component must be finite nonnegative MW")
    scaled, downloaded = value * .001, pd.Timestamp.now(tz="UTC")
    record = {"policy": POLICY, "alias": ALIAS, "native_series": historical.WIND_SERIES[ALIAS],
        "fallback_series": COMPONENT, "delivery_day": day, "value_time_utc": hour.isoformat(),
        "local_time": hour.tz_convert("Europe/Amsterdam").isoformat(),
        "query_cutoff_utc": pd.Timestamp(query_revision).isoformat(),
        "query_cutoff_local": pd.Timestamp(query_revision).tz_convert("Europe/Amsterdam").isoformat(),
        "raw_value_mw": value, "value_scale": .001, "scaled_value_gw": scaled, "native_missing": True,
        "fallback_downloaded_at_utc": downloaded.isoformat(), "provenance": historical.SUBSTITUTION_PROVENANCE,
        "provider_revision_timestamp_available": False, "production_pit_evidence": False,
        "native_values_sha256": _native_digest(native), "native_gap_kind": gap_kind}
    completed = pd.concat([native, pd.Series([scaled], index=pd.DatetimeIndex([hour]), name=series.name)]).sort_index()
    completed.index.name = series.index.name
    completed.attrs = dict(series.attrs)
    verify_nl_wind_substitutions(completed, [record], day, query_revision)
    return completed, [record]
