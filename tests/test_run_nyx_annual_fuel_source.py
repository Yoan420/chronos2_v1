"""Prospective fuel adapter tests with synthetic as-of Saturn materialization."""
from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import materialize_saturn_kalman_fuel as fuel
from run_nyx_annual_fuel_source import (
    ASSUMPTIONS, publish_fuel_receipt, verify_fuel_artifact,
)
from chronos2_hourly.nyx_annual_live_preflight import delivery_grid


START = "2025-09-24"
DAY = "2026-09-24"


def _synthetic_cache(cache: Path) -> None:
    start, end = pd.Timestamp(START), pd.Timestamp(DAY)
    query_start = start - pd.Timedelta(days=fuel.WARMUP_DAYS)
    days = pd.date_range(query_start, end, freq="D")

    raw = _synthetic_raw(days)
    daily = fuel._derive_daily_features(raw, **ASSUMPTIONS)
    frame = fuel._broadcast_daily_features(daily, start_day=start, end_day=end)
    audit = fuel._audit_payload(frame, start_day=start, end_day=end,
                                requested_start=start, requested_end=end,
                                query_start=query_start, **ASSUMPTIONS)
    output = cache / fuel.OUTPUT_NAME
    fuel._write_bundle(frame, output,
                       cache / (fuel.OUTPUT_NAME + fuel.AUDIT_SUFFIX), audit)


def _synthetic_raw(days: pd.DatetimeIndex) -> pd.DataFrame:
    cutoff = pd.DatetimeIndex([fuel._civil_cutoff(day).tz_convert("UTC")
                               for day in days])
    source = cutoff - pd.Timedelta(days=1)
    absolute_days = np.asarray((days - pd.Timestamp("2025-01-01")) / pd.Timedelta(days=1),
                               dtype=float)
    return pd.DataFrame({
        "ttf_m1_eur_mwh_th": 35.0 + absolute_days / 100.0,
        "eua_first_dec_eur_tco2": 70.0 + absolute_days / 200.0,
        "ttf_m1_eur_mwh_th__source_value_time_utc": source,
        "eua_first_dec_eur_tco2__source_value_time_utc": source,
        "cutoff_time_utc": cutoff,
    }, index=days)


def test_verified_fuel_source_publishes_bound_receipt(tmp_path: Path):
    cache, bundle = tmp_path / "cache", tmp_path / "bundle"
    _synthetic_cache(cache)
    evidence = verify_fuel_artifact(cache, delivery_day=DAY,
                                    history_start_day=START)
    assert evidence["rows_in_training_and_delivery"] == len(delivery_grid(DAY)[0])
    path = publish_fuel_receipt(bundle, delivery_day=DAY, evidence=evidence)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["source_group"] == "fuel"
    assert receipt["asof_cutoff_verified"] is True
    assert receipt["asof_state_utc"] == receipt["cutoff_utc"]
    assert receipt["latest_source_value_time_utc"] != receipt["cutoff_utc"]
    assert receipt["availability_basis"] == "Saturn revision_date query as-of D-1 08:00"
    assert receipt["provider_revision_timestamp_available"] is False
    assert receipt["provider_publication_timestamp_verified"] is False
    assert receipt["model_inputs_complete"] is False
    assert publish_fuel_receipt(bundle, delivery_day=DAY, evidence=evidence) == path
    copied = bundle / ("source_artifacts/fuel/" + fuel.OUTPUT_NAME)
    copied.write_bytes(b"changed")
    with pytest.raises(ValueError, match="existant différent"):
        publish_fuel_receipt(bundle, delivery_day=DAY, evidence=evidence)


def test_fuel_source_blocks_early_and_stale_delivery(tmp_path: Path):
    _, _, cutoff = delivery_grid(DAY)
    with pytest.raises(ValueError, match="pas encore atteinte"):
        verify_fuel_artifact(tmp_path, delivery_day=DAY,
                             history_start_day=START,
                             now_utc=cutoff - pd.Timedelta(seconds=1))
    _synthetic_cache(tmp_path)
    with pytest.raises(ValueError, match="autre jour"):
        verify_fuel_artifact(tmp_path, delivery_day="2026-09-25",
                             history_start_day=START)


def test_tracked_materializer_builds_and_extends_past_historical_boundary(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_fetch(days, **kwargs):
        calls.append((days[0], days[-1]))
        return _synthetic_raw(pd.DatetimeIndex(days))

    monkeypatch.setattr(fuel, "_fetch_daily_market", fake_fetch)
    cache = tmp_path / "fresh_clone_cache"
    common = ["--start-day", START, "--output-dir", str(cache),
              "--skip-residual-load"]
    assert fuel.main([*common, "--end-day", DAY]) == 0
    later = "2026-09-25"
    assert fuel.main([*common, "--end-day", later]) == 0
    assert len(calls) == 2
    assert calls[0][1] == pd.Timestamp(DAY)
    assert calls[1][1] == pd.Timestamp(later)
    evidence = verify_fuel_artifact(cache, delivery_day=later,
                                    history_start_day=START)
    assert evidence["audit"]["end_day"] == later
