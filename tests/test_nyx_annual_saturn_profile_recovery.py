"""Bounded own-origin retries and explicit outer-cutoff profile reconstruction."""
import json
import shutil
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_annual_saturn_source as m


@pytest.fixture
def clients(monkeypatch):
    created, closed, sleeps = [], [], []
    def factory():
        identity = len(created) + 1
        client = SimpleNamespace(identity=identity,
            session=SimpleNamespace(close=lambda: closed.append(identity)))
        created.append(client)
        return client
    monkeypatch.setattr(m.time, "sleep", sleeps.append)
    return factory, created, closed, sleeps


def finite(client, series, start, end, timezone, **kwargs):
    index = pd.date_range(start, end, freq="h", tz="UTC")
    return pd.Series(np.arange(len(index), dtype=float) ** 2, index=index)


def test_three_fresh_retries_then_complete_historical_profile_at_outer_cutoff(monkeypatch, tmp_path, clients, capsys):
    inner, outer = "2024-08-16", "2026-09-28"
    calls = []
    def fetch(client, series, start, end, timezone, **kwargs):
        calls.append((client.identity, series, start, end, kwargs.copy()))
        if kwargs["revision_date"] == m.cutoff(inner):
            raise RuntimeError("Historic origin returned an empty response")
        return finite(client, series, start, end, timezone, **kwargs)
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    receipt = m._sync_profile_day(inner, outer, tmp_path, clients[0])
    assert len(clients[1]) == 4 and clients[2] == [1, 2, 3, 4] and clients[3] == [1, 2]
    assert len(calls) == 17 and all(call[2:] == calls[0][2:] for call in calls[:3])
    assert len(set(call[1] for call in calls[3:])) == 14
    assert all(call[4]["revision_date"] == m.cutoff(outer) for call in calls[3:])
    assert receipt["forecast_origin_utc"] == receipt["profile_revision_utc"] == m.cutoff(outer).isoformat()
    assert receipt["logical_forecast_origin_utc"] == m.cutoff(inner).isoformat()
    assert receipt["profile_origin_snapshot_verified"] is False
    assert not (tmp_path / "profiles_v2" / inner / "receipt.json").exists()
    directory = tmp_path / "profiles_current_fit_v1" / outer / inner
    assert m.verify_profile_day(directory, inner, outer_day=outer)[1] == receipt
    output = capsys.readouterr().out
    assert '"event": "profile_current_fit_recovery"' in output and m.cutoff(outer).isoformat() in output
    assert m._sync_profile_day(inner, outer, tmp_path, lambda: pytest.fail("recovery cache contacted Saturn")) == receipt


def test_successful_retry_keeps_original_vintage_and_never_recovers(monkeypatch, tmp_path, clients):
    inner, outer = "2024-08-16", "2026-09-28"
    def fetch(client, series, start, end, timezone, **kwargs):
        assert kwargs["revision_date"] == m.cutoff(inner)
        if client.identity == 1:
            raise RuntimeError("Transient proxy response")
        return finite(client, series, start, end, timezone, **kwargs)
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    receipt = m._sync_profile_day(inner, outer, tmp_path, clients[0])
    assert receipt["protocol"] == m.PROFILE_PROTOCOL
    assert receipt["forecast_origin_utc"] == m.cutoff(inner).isoformat()
    assert len(clients[1]) == 2 and clients[3] == [1]
    assert not (tmp_path / "profiles_current_fit_v1").exists()


def test_live_day_failure_never_uses_recovery(monkeypatch, tmp_path, clients, capsys):
    day = "2026-09-28"
    def fetch(*args, **kwargs):
        assert kwargs["revision_date"] == m.cutoff(day)
        raise RuntimeError("Live curve unavailable")
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    with pytest.raises(m.SaturnSourceError, match="3 attempts"):
        m._sync_profile_day(day, day, tmp_path, clients[0])
    assert len(clients[1]) == 3 and clients[2] == [1, 2, 3]
    assert "profile_current_fit_recovery" not in capsys.readouterr().out
    with pytest.raises(ValueError, match="forbidden"):
        m.capture_profile_day(object(), day, tmp_path, outer_day=day)
    assert not list(tmp_path.rglob("receipt.json"))


def test_outer_recovery_also_fails_closed_when_source_remains_empty(monkeypatch, tmp_path, clients):
    def fetch(*args, **kwargs):
        raise RuntimeError("Still unavailable password=example-secret")
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    with pytest.raises(m.SaturnSourceError) as caught:
        m._sync_profile_day("2024-08-16", "2026-09-28", tmp_path, clients[0])
    assert caught.value.phase == "current_fit_profile_recovery"
    assert "3 attempts" in str(caught.value) and "example-secret" not in str(caught.value)
    assert len(clients[1]) == 6 and clients[2] == list(range(1, 7))
    assert clients[3] == [1, 2, 1, 2] and not list(tmp_path.rglob("receipt.json"))


def test_corrupt_strict_cache_is_not_hidden_by_valid_recovery(monkeypatch, tmp_path, clients):
    inner, outer = "2024-08-16", "2026-09-28"
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", finite)
    m.capture_profile_day(object(), inner, tmp_path, now_utc=m.cutoff(outer))
    m.capture_profile_day(object(), inner, tmp_path, outer_day=outer, now_utc=m.cutoff(outer))
    path = tmp_path / "profiles_v2" / inner / "covariates.parquet"
    frame = pd.read_parquet(path)
    frame.iloc[0, 0] += 1
    frame.to_parquet(path)
    with pytest.raises(ValueError, match="modified"):
        m._sync_profile_day(inner, outer, tmp_path, lambda: pytest.fail("corrupt cache contacted Saturn"))
    assert not clients[1]


def test_recovery_cache_is_bound_to_outer_day_and_revision(monkeypatch, tmp_path, clients):
    inner, outer, next_outer = "2024-08-16", "2026-09-28", "2026-09-29"
    def fetch(client, series, start, end, timezone, **kwargs):
        if kwargs["revision_date"] == m.cutoff(inner):
            raise RuntimeError("Historical profile empty")
        return finite(client, series, start, end, timezone, **kwargs)
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    m._sync_profile_day(inner, outer, tmp_path, clients[0])
    receipt = m._sync_profile_day(inner, next_outer, tmp_path, clients[0])
    assert len(clients[1]) == 8 and receipt["profile_revision_utc"] == m.cutoff(next_outer).isoformat()
    directory = tmp_path / "profiles_current_fit_v1" / next_outer / inner
    with pytest.raises(ValueError, match="revision"):
        m.verify_profile_day(directory, inner, outer_day=outer)
    receipt["forecast_origin_utc"] = m.cutoff(inner).isoformat()
    (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        m.verify_profile_day(directory, inner, outer_day=next_outer)


def test_nl_spring_recovery_ledger_records_real_outer_revision(monkeypatch, tmp_path):
    inner, outer = "2025-03-30", "2026-09-28"
    expected = m.grid(inner)
    absent = expected[expected.tz_convert("Europe/Amsterdam").strftime("%H:%M").isin(["04:00", "06:00"])]
    def fetch(client, series, start, end, timezone, **kwargs):
        assert kwargs["revision_date"] == m.cutoff(outer)
        values = finite(client, series, start, end, timezone, **kwargs)
        return values.drop(absent) if series == "power.nl.residual.load.hourly.gw.fcst" else values
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    receipt = m.capture_profile_day(object(), inner, tmp_path, outer_day=outer, now_utc=m.cutoff(outer))
    entries = receipt["spring_dst_repair"]["entries"]
    assert len(entries) == 2 and all(item["forecast_origin_utc"] == m.cutoff(outer).isoformat() for item in entries)
    directory = tmp_path / "profiles_current_fit_v1" / outer / inner
    assert len(m.verify_profile_day(directory, inner, outer_day=outer)[0]) == 23
    entries[0]["forecast_origin_utc"] = m.cutoff(inner).isoformat()
    (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="ledger"):
        m.verify_profile_day(directory, inner, outer_day=outer)


@pytest.mark.parametrize("bad", ["nan", "duplicate"])
def test_recovery_never_accepts_nonfinite_or_ambiguous_source(monkeypatch, tmp_path, bad):
    def fetch(client, series, start, end, timezone, **kwargs):
        values = finite(client, series, start, end, timezone, **kwargs)
        if bad == "nan":
            values.iloc[8] = np.nan
        else:
            values = pd.concat([values, values.iloc[[8]]])
        return values
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    with pytest.raises(m.SaturnSourceError):
        m.capture_profile_day(object(), "2024-08-16", tmp_path, outer_day="2026-09-28")
    assert not list(tmp_path.rglob("receipt.json"))


def test_portable_mixed_source_binds_actual_revisions_and_rejects_downgrade(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", finite)
    outer, recovered = "2026-09-28", "2026-03-29"
    first = (pd.Timestamp(outer).date() - timedelta(days=365)).isoformat()
    cache, bundle = tmp_path / "cache", tmp_path / "bundle"
    m.capture_target_snapshot(object(), outer, first, cache, now_utc=m.cutoff(outer))
    for stamp in pd.date_range(first, outer, freq="D"):
        day = stamp.date().isoformat()
        m.capture_profile_day(object(), day, cache, outer_day=outer if day == recovered else None, now_utc=m.cutoff(outer))
    path = m.publish(bundle, outer, first_day=first, cache=cache)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert m.profile_history_contract(bundle) == m._profile_contract(outer)
    origin = pd.read_parquet(bundle / "source_artifacts/saturn/origins.parquet")
    revision_path = bundle / "source_artifacts/saturn/profile_revisions.parquet"
    revision = pd.read_parquet(revision_path)
    assert list(revision) == list(m.ALIASES)
    assert (origin.loc[m.grid(recovered), "forecast_origin_utc"] == m.cutoff(recovered)).all()
    assert (revision.loc[m.grid(recovered)] == m.cutoff(outer)).all().all()
    _, _, verdict = m.verify_current_fit_source(bundle, outer, receipt)
    assert verdict["profile_revision_ceiling_utc"] == m.cutoff(outer).isoformat()
    original = path.read_bytes()
    assert m.publish(bundle, outer, cache=tmp_path / "absent-cache") == path
    assert path.read_bytes() == original
    # Re-hash the altered artifact: validation must recompute the real revisions,
    # not only trust the updated checksum in a manually edited packet.
    raw_revision = revision_path.read_bytes()
    revision.iloc[0, 0] = m.cutoff(outer) + pd.Timedelta(hours=1)
    revision.to_parquet(revision_path)
    receipt["artifact_sha256"]["source_artifacts/saturn/profile_revisions.parquet"] = m.sha256(revision_path)
    with pytest.raises(ValueError, match="actual profile revisions"):
        m.verify_current_fit_source(bundle, outer, receipt)
    revision_path.write_bytes(raw_revision)
    receipt = json.loads(original)
    for key in m.PROFILE_HISTORY_FIELDS:
        receipt.pop(key)
    with pytest.raises(ValueError, match="downgraded"):
        m.verify_current_fit_source(bundle, outer, receipt)
    receipt.pop("logical_forecast_origins")
    receipt["artifact_sha256"].pop("source_artifacts/saturn/profile_revisions.parquet")
    revision_path.unlink()
    with pytest.raises(ValueError, match="recovery"):
        m.verify_current_fit_source(bundle, outer, receipt)
    # A real old packet containing only strict profiles remains readable and
    # reusable. Replace the recovered day with independently captured strict
    # evidence before converting the envelope to the old schema.
    strict = m.capture_profile_day(object(), recovered, cache, now_utc=m.cutoff(outer))
    relative = f"source_artifacts/saturn/days/{recovered}/receipt.json"
    portable = bundle / relative
    portable.unlink()  # Break the test packet's hard link, preserving the cache.
    shutil.copyfile(cache / "profiles_v2" / recovered / "receipt.json", portable)
    receipt["artifact_sha256"][relative] = m.sha256(portable)
    receipt["daily_vintages"][recovered]["forecast_origin_utc"] = strict["forecast_origin_utc"]
    for declared in receipt["daily_vintages"].values():
        for field in ("logical_forecast_origin_utc", "profile_revision_utc", "profile_origin_snapshot_verified"):
            declared.pop(field)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert m.profile_history_contract(bundle) == {}
    original = path.read_bytes()
    assert m.publish(bundle, outer, cache=tmp_path / "absent-cache") == path
    assert path.read_bytes() == original
