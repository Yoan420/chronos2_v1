"""Outer-cutoff target reconstruction preserves historical forecast profiles."""
import json
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_annual_saturn_source as m


def fake_fetch(client, series, start, end, timezone, **kwargs):
    index = pd.date_range(start, end, freq="h", tz="UTC")
    return pd.Series(np.arange(len(index), dtype=float), index=index)


def test_new_profile_packet_matches_legacy_profiles_but_queries_no_prices(monkeypatch, tmp_path):
    day = "2026-03-29"
    calls = []
    def fetch(*args, **kwargs):
        calls.append((args[1], kwargs["revision_date"]))
        return fake_fetch(*args, **kwargs)
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    m.capture_day(object(), day, tmp_path / "old", now_utc=m.cutoff(day))
    old, _, _ = m.verify_day(tmp_path / "old" / day, day)
    calls.clear()
    receipt = m.capture_profile_day(object(), day, tmp_path / "new", now_utc=m.cutoff(day))
    new, _ = m.verify_profile_day(tmp_path / "new/profiles_v2" / day, day)
    pd.testing.assert_frame_equal(new, old)
    assert len(calls) == 14 and all("price.da" not in name and vintage == m.cutoff(day) for name, vintage in calls)
    assert set(receipt["artifact_sha256"]) == {"covariates.parquet"}
    assert not list((tmp_path / "new").rglob("prices.parquet"))


def test_targets_use_only_outer_cutoff_and_resume_verified_snapshot(monkeypatch, tmp_path):
    calls = []
    def fetch(*args, **kwargs):
        calls.append((args, kwargs))
        return fake_fetch(*args, **kwargs)
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    outer, first = "2026-09-29", "2024-06-18"
    receipt = m.capture_target_snapshot(object(), outer, first, tmp_path, now_utc=m.cutoff(outer))
    frame, verified = m.verify_target_snapshot(tmp_path / "targets_current_fit_v1" / outer, outer, first)
    assert receipt == verified and len(calls) == 4
    assert all(k["revision_date"] == m.cutoff(outer) and k["naive_timezone"] == "UTC"
               and k["incomplete_dst_policy"] == "raise" for _, k in calls)
    assert frame.index.equals(m.target_grid(first, outer)) and frame.index[-1] < m.grid(outer)[0]
    assert receipt["target_origin_snapshot_verified"] is False
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", lambda *a, **k: pytest.fail("cache contacted Saturn"))
    assert m.capture_target_snapshot(object(), outer, first, tmp_path) == receipt


def test_sync_targets_fail_before_any_profile_download(monkeypatch, tmp_path):
    import run_nyx_annual_auction_prices_source as auction
    monkeypatch.setattr(auction, "load_plan", lambda: {})
    closed = []
    client = SimpleNamespace(session=SimpleNamespace(close=lambda: closed.append(True)))
    def fail(*args, **kwargs):
        raise ValueError("Missing canonical current-cutoff price")
    monkeypatch.setattr(m, "capture_target_snapshot", fail)
    monkeypatch.setattr(m, "capture_profile_day", lambda *a, **k: pytest.fail("prices failed but profiles downloaded"))
    with pytest.raises(m.SaturnSourceError, match="current_fit_targets"):
        m.sync("2026-09-29", cache=tmp_path, client_factory=lambda: client)
    assert closed == [True]


@pytest.fixture
def current_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fake_fetch)
    outer = "2026-09-29"
    first = (pd.Timestamp(outer).date() - timedelta(days=365)).isoformat()
    cache, bundle = tmp_path / "cache", tmp_path / "bundle"
    m.capture_target_snapshot(object(), outer, first, cache, now_utc=m.cutoff(outer))
    for stamp in pd.date_range(first, outer, freq="D"):
        m.capture_profile_day(object(), stamp.date().isoformat(), cache, now_utc=m.cutoff(outer))
    receipt_path = m.publish(bundle, outer, first_day=first, cache=cache)
    return bundle, outer, first, json.loads(receipt_path.read_text(encoding="utf-8"))


def test_portable_source_binding_inner_windows_and_raw_validator(current_bundle):
    from chronos2_hourly import nyx_annual_source_validation as validator
    bundle, outer, first, receipt = current_bundle
    contract = m.target_history_contract(bundle)
    assert contract == m._target_contract(outer)
    load = m.load_target_snapshots(bundle)
    for inner in (first, "2026-03-29", outer):
        values = load(inner)
        assert set(values) == set(m.ZONES)
        assert all(series.index.equals(m.target_grid(inner, inner)) and series.index[-1] < m.grid(inner)[0]
                   and series.attrs == contract for series in values.values())
    with pytest.raises(ValueError, match="outside"):
        load((pd.Timestamp(first) - pd.Timedelta(days=1)).date().isoformat())
    with pytest.raises(ValueError, match="outside"):
        load((pd.Timestamp(outer) + pd.Timedelta(days=1)).date().isoformat())
    verdict, latest = validator._saturn(bundle, outer, receipt)
    assert verdict["daily_states_verified"] == 366 and verdict["asof_cutoff_verified"] is True
    assert latest.index.equals(m.target_grid(outer, outer))
    assert all("prices_sha256" not in item for item in receipt["daily_vintages"].values())


def test_revision_and_artifact_tampering_refused(current_bundle):
    bundle, outer, first, receipt = current_bundle
    path = bundle / "source_receipts/saturn.json"
    receipt["target_revision_utc"] = m.cutoff(first).isoformat()
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        m.target_history_contract(bundle)
    receipt["target_revision_utc"] = m.cutoff(outer).isoformat()
    path.write_text(json.dumps(receipt), encoding="utf-8")
    prices = bundle / "source_artifacts/saturn/targets/prices.parquet"
    frame = pd.read_parquet(prices)
    frame.iloc[0, 0] += 1
    frame.to_parquet(prices)
    with pytest.raises(ValueError, match="changed"):
        m.load_target_snapshots(bundle)


def test_legacy_loader_and_identity_remain_readable(monkeypatch, tmp_path):
    day = "2026-09-29"
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fake_fetch)
    folder = tmp_path / "source_artifacts/saturn/days"
    receipt = m.capture_day(object(), day, folder, now_utc=m.cutoff(day))
    source = {"daily_vintages": {day: {"prices_sha256": receipt["artifact_sha256"]["prices.parquet"]}}}
    (tmp_path / "source_receipts").mkdir()
    (tmp_path / "source_receipts/saturn.json").write_text(json.dumps(source), encoding="utf-8")
    assert m.target_history_contract(tmp_path) == {}
    values = m.load_target_snapshots(tmp_path)(day)
    assert values["FR"].index.equals(m.target_grid(day, day))


@pytest.mark.parametrize("day", ["2025-03-30", "2026-03-29"])
def test_new_nl_spring_profiles_match_historical_repair_with_bound_ledger(monkeypatch, tmp_path, day):
    from materialize_saturn_kalman_fuel import RESIDUAL_LOAD_SERIES, _repair_nl_spring_dst_hour
    expected = m.grid(day)
    missing = expected[expected.tz_convert("Europe/Amsterdam").strftime("%H:%M").isin(["04:00", "06:00"])]
    supplied = {}
    def fetch(client, series, start, end, timezone, **kwargs):
        assert kwargs["revision_date"] == m.cutoff(day)
        index = pd.date_range(start, end, freq="h", tz="UTC")
        values = pd.Series(np.arange(len(index), dtype=float) ** 2, index=index)
        if series == "power.nl.residual.load.hourly.gw.fcst":
            values = values.drop(missing)
            supplied["nl"] = values.loc[(values.index >= expected[0]) & (values.index <= expected[-1])]
        return values
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    receipt = m.capture_profile_day(object(), day, tmp_path, now_utc=m.cutoff(day))
    frame, _ = m.verify_profile_day(tmp_path / "profiles_v2" / day, day)
    raw = supplied["nl"].copy()
    raw.index = raw.index.tz_convert("Europe/Amsterdam")
    spec = next(item for item in RESIDUAL_LOAD_SERIES if item.alias == "nl_residual_load_fcst")
    historical, repaired = _repair_nl_spring_dst_hour(raw, spec=spec, day=pd.Timestamp(day), expected=expected)
    np.testing.assert_array_equal(frame["nl_residual_load_fcst"], historical.to_numpy(float))
    entries = receipt["spring_dst_repair"]["entries"]
    assert len(entries) == 2 and tuple(pd.Timestamp(item["value_time_utc"]) for item in entries) == repaired
    assert all(item["forecast_origin_utc"] == m.cutoff(day).isoformat() for item in entries)
    with pytest.raises(m.SaturnSourceError, match="missing/nonfinite"):
        m.capture_day(object(), day, tmp_path / "legacy", now_utc=m.cutoff(day))
    entries[0]["donor_values_gw"][0] += 1
    (tmp_path / "profiles_v2" / day / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="ledger"):
        m.verify_profile_day(tmp_path / "profiles_v2" / day, day)


@pytest.mark.parametrize("alias,times,nan", [
    ("nl", ["04:00"], False), ("nl", ["04:00", "07:00"], False),
    ("nl", ["04:00", "06:00", "08:00"], False),
    ("nl", ["04:00", "06:00"], True), ("fr", ["04:00", "06:00"], False),
])
def test_new_profile_repair_never_fills_other_holes_or_nan(monkeypatch, tmp_path, alias, times, nan):
    day = "2025-03-30"
    expected = m.grid(day)
    missing = expected[expected.tz_convert("Europe/Amsterdam").strftime("%H:%M").isin(times)]
    def fetch(client, series, start, end, timezone, **kwargs):
        values = fake_fetch(client, series, start, end, timezone, **kwargs)
        if series == f"power.{alias}.residual.load.hourly.gw.fcst":
            if nan:
                values.loc[missing] = np.nan
            else:
                values = values.drop(missing)
        return values
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    with pytest.raises(m.SaturnSourceError, match="missing/nonfinite"):
        m.capture_profile_day(object(), day, tmp_path, now_utc=m.cutoff(day))
    assert not list(tmp_path.rglob("receipt.json"))
