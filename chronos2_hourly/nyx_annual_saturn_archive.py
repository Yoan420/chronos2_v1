"""Pinned NL forecast archive recovery for historical training days only.

The tracked archive's alias is bound by the NL configuration to
``power.nl.residual.load.hourly.gw.fcst``, with UTC timestamps and no filling.
A complete civil day must come from one common recorded snapshot. This is
current-fit recovery of an archived forecast, never proof that the snapshot
existed at the historical delivery day's own forecast origin.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .nyx_annual_live_preflight import ROOT


PROTOCOL = "nyx_annual_nl_repository_profile_v1"
ALIAS = "nl_residual_load_fcst"
SERIES = "power.nl.residual.load.hourly.gw.fcst"
ARCHIVE_RELATIVE_PATH = "data/pit/vintages/nl_residual_load_fcst.parquet"
DEFAULT_ARCHIVE = ROOT / ARCHIVE_RELATIVE_PATH
CONFIG_RELATIVE_PATH = "chronos2_hourly_nl_residual_candidate_v1.yaml"
CONFIG_PATH = ROOT / CONFIG_RELATIVE_PATH
# Checked against the tracked file (commit 4dabd12f). Updating the archive is
# a reviewed source-contract change, not automatic admission of another file.
PINNED_ARCHIVE_SHA256 = "7c705137eb1b96a5e9a13a0ec92f403ceed7e79824e79fb479d38f2a5ac3d8ba"
SELECTION_POLICY = "latest_complete_common_snapshot_at_or_before_outer_cutoff_v1"
COLUMNS = ("value_time_utc", "snapshot_time_utc", "revision_time_utc", "value", "downloaded_at_utc")
TIME_COLUMNS = tuple(name for name in COLUMNS if name != "value")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _day(value):
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError("NL archive dates must use YYYY-MM-DD") from error
    _require(parsed.isoformat() == value, "NL archive dates must use YYYY-MM-DD")
    return parsed


def _grid(day):
    return pd.date_range(pd.Timestamp(day, tz="Europe/Paris"),
        pd.Timestamp(day + timedelta(days=1), tz="Europe/Paris"),
        freq="h", inclusive="left").tz_convert("UTC").rename("timestamp_utc")


def _cutoff(day):
    return (pd.Timestamp(day - timedelta(days=1)) + pd.Timedelta(hours=8)).tz_localize(
        "Europe/Paris").tz_convert("UTC")


def _canonical_contract():
    try:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        specification = config["zones"]["NL"]["covariates"][ALIAS]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as error:
        raise ValueError("NL archive canonical configuration is missing or invalid") from error
    _require(specification.get("series") == SERIES
             and specification.get("naive_timezone") == "UTC"
             and specification.get("source") == "pit_parquet"
             and specification.get("fill_method") == "none",
             "NL archive canonical series, UTC or no-fill contract changed")


def _read_archive(path):
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise ValueError("Pinned NL archive is missing or unreadable") from error
    _require(hashlib.sha256(raw).hexdigest() == PINNED_ARCHIVE_SHA256,
             "Pinned NL archive SHA-256 differs")
    # Parse the exact bytes just verified, rather than reopening a mutable path.
    try:
        frame = pd.read_parquet(BytesIO(raw))
    except Exception as error:
        raise ValueError("Pinned NL archive parquet cannot be read") from error
    _require(tuple(frame.columns) == COLUMNS, "Pinned NL archive schema differs")
    for column in TIME_COLUMNS:
        _require(isinstance(frame[column].dtype, pd.DatetimeTZDtype)
                 and str(frame[column].dtype.tz) == "UTC" and frame[column].notna().all(),
                 f"Pinned NL archive {column} must contain explicit UTC timestamps")
    return frame


def recover_nl_profile(day, outer_day, *, archive_path=DEFAULT_ARCHIVE):
    """Return one complete archived NL day and its portable source evidence.

    Only snapshots whose selected rows were both revised and downloaded by
    the outer cutoff are eligible. Incomplete snapshots are never combined.
    """
    inner, outer = _day(day), _day(outer_day)
    _require(inner < outer, "NL archive recovery is forbidden for the live or future delivery day")
    _canonical_contract()
    frame = _read_archive(archive_path)
    expected, ceiling = _grid(inner), _cutoff(outer)
    stop = expected[-1] + pd.Timedelta(hours=1)
    selected_day = frame.loc[(frame.value_time_utc >= expected[0]) & (frame.value_time_utc < stop)]
    snapshots = selected_day.snapshot_time_utc.loc[selected_day.snapshot_time_utc <= ceiling].unique()
    for snapshot in sorted(snapshots, reverse=True):
        block = selected_day.loc[selected_day.snapshot_time_utc == snapshot].sort_values("value_time_utc")
        index = pd.DatetimeIndex(block.value_time_utc)
        if not index.is_unique or not index.equals(expected):
            continue
        if not ((block.revision_time_utc <= snapshot).all()
                and (block.downloaded_at_utc <= ceiling).all()
                and (block.downloaded_at_utc >= block.snapshot_time_utc).all()):
            continue
        values = pd.to_numeric(block.value, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            continue
        result = pd.Series(values, index=expected, name=ALIAS)
        evidence = {"protocol": PROTOCOL, "profile_day": day, "outer_delivery_day": outer_day,
            "archive_sha256": PINNED_ARCHIVE_SHA256, "archive_contract_path": ARCHIVE_RELATIVE_PATH,
            "config_contract_path": CONFIG_RELATIVE_PATH, "alias": ALIAS, "series": SERIES,
            "naive_timezone": "UTC", "selection_policy": SELECTION_POLICY,
            "outer_cutoff_utc": ceiling.isoformat(), "snapshot_time_utc": pd.Timestamp(snapshot).isoformat(),
            "revision_time_max_utc": block.revision_time_utc.max().isoformat(),
            "downloaded_at_max_utc": block.downloaded_at_utc.max().isoformat(),
            "first_value_time_utc": expected[0].isoformat(), "last_value_time_utc": expected[-1].isoformat(),
            "hours": len(expected), "imputation": False, "mosaic": False,
            "historical_origin_snapshot_verified": False}
        return result, evidence
    raise ValueError(f"NL archive {day}: no complete common snapshot available by outer cutoff {ceiling.isoformat()}")


def verify_nl_profile(series, evidence, day, outer_day, archive_path=DEFAULT_ARCHIVE):
    """Recompute the pinned selection and reject changed values or provenance."""
    expected, expected_evidence = recover_nl_profile(day, outer_day, archive_path=archive_path)
    _require(isinstance(evidence, dict) and evidence == expected_evidence
             and evidence.get("imputation") is False and evidence.get("mosaic") is False
             and evidence.get("historical_origin_snapshot_verified") is False,
             "NL archive recovery evidence differs from the pinned selection")
    _require(isinstance(series, pd.Series) and isinstance(series.index, pd.DatetimeIndex)
             and str(series.index.tz) == "UTC" and series.index.is_unique
             and series.index.equals(expected.index), "NL archive recovered physical-hour grid differs")
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    _require(np.isfinite(values).all() and np.array_equal(values, expected.to_numpy(float)),
             "NL archive recovered values differ from the pinned snapshot")
    return expected_evidence
