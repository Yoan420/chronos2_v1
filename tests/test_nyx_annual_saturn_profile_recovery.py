"""Per-series origin retries, resumable recovery and honest portable provenance."""
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


def test_only_missing_series_uses_outer_cutoff_after_three_fresh_retries(monkeypatch, tmp_path, clients, capsys):
    inner, outer = "2024-08-16", "2026-09-28"
    calls = []
    def fetch(client, series, start, end, timezone, **kwargs):
        calls.append((client.identity, series, start, end, kwargs.copy()))
        if series == m.specs()["fr_residual_load_fcst"]["series"] and kwargs["revision_date"] == m.cutoff(inner):
            raise RuntimeError("Historic origin returned an empty response")
        if series == m.specs()["nl_residual_load_fcst"]["series"]:
            assert kwargs["revision_date"] == m.cutoff(inner), "Valid NL origin must not be replaced"
        return finite(client, series, start, end, timezone, **kwargs)
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    receipt = m._sync_profile_day(inner, outer, tmp_path, clients[0])
    assert len(clients[1]) == 17 and clients[2] == list(range(1, 18)) and clients[3] == [1, 2]
    assert len(calls) == 17 and all(call[2:] == calls[0][2:] for call in calls[:3])
    assert len(set(call[1] for call in calls)) == 14
    assert all(call[4]["revision_date"] == m.cutoff(inner) for call in calls[:-1])
    assert calls[-1][4]["revision_date"] == m.cutoff(outer)
    assert receipt["alias_revisions_utc"]["fr_residual_load_fcst"] == m.cutoff(outer).isoformat()
    assert receipt["alias_revisions_utc"]["nl_residual_load_fcst"] == m.cutoff(inner).isoformat()
    assert "forecast_origin_utc" not in receipt
    assert receipt["logical_forecast_origin_utc"] == m.cutoff(inner).isoformat()
    assert receipt["profile_origin_snapshot_verified"] is False
    assert not (tmp_path / "profiles_v2" / inner / "receipt.json").exists()
    directory = tmp_path / "profiles_per_series_v2" / outer / inner
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
    assert len(clients[1]) == 15 and clients[3] == [1]
    assert not (tmp_path / "profiles_current_fit_v1").exists()
    assert m._sync_profile_day(inner, "2026-09-29", tmp_path,
        lambda: pytest.fail("complete original profiles must survive changing outer day")) == receipt


def test_live_day_failure_never_uses_recovery(monkeypatch, tmp_path, clients, capsys):
    day = "2026-09-28"
    def fetch(*args, **kwargs):
        assert kwargs["revision_date"] == m.cutoff(day)
        if args[1] == m.specs()["fr_residual_load_fcst"]["series"]:
            raise RuntimeError("Live curve unavailable")
        return finite(*args, **kwargs)
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    with pytest.raises(m.SaturnSourceError, match="3 attempts"):
        m._sync_profile_day(day, day, tmp_path, clients[0])
    assert len(clients[1]) == 16 and clients[2] == list(range(1, 17))
    assert "profile_current_fit_recovery" not in capsys.readouterr().out
    with pytest.raises(ValueError, match="forbidden"):
        m.capture_profile_day(object(), day, tmp_path, outer_day=day)
    assert not list(tmp_path.rglob("receipt.json"))


def test_outer_recovery_also_fails_closed_when_source_remains_empty(monkeypatch, tmp_path, clients):
    def fetch(*args, **kwargs):
        if args[1] == m.specs()["fr_residual_load_fcst"]["series"]:
            raise RuntimeError("Still unavailable password=example-secret")
        return finite(*args, **kwargs)
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    with pytest.raises(m.SaturnSourceError) as caught:
        m._sync_profile_day("2024-08-16", "2026-09-28", tmp_path, clients[0])
    assert caught.value.phase == "current_fit_profile_recovery"
    assert "3 attempts" in str(caught.value) and "example-secret" not in str(caught.value)
    assert len(clients[1]) == 19 and clients[2] == list(range(1, 20))
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
        if series == m.specs()["fr_residual_load_fcst"]["series"] and kwargs["revision_date"] == m.cutoff(inner):
            raise RuntimeError("Historical profile empty")
        return finite(client, series, start, end, timezone, **kwargs)
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    m._sync_profile_day(inner, outer, tmp_path, clients[0])
    receipt = m._sync_profile_day(inner, next_outer, tmp_path, clients[0])
    assert len(clients[1]) == 34 and receipt["alias_revisions_utc"]["fr_residual_load_fcst"] == m.cutoff(next_outer).isoformat()
    directory = tmp_path / "profiles_per_series_v2" / next_outer / inner
    with pytest.raises(ValueError, match="outer delivery"):
        m.verify_profile_day(directory, inner, outer_day=outer)
    receipt["alias_revisions_utc"]["fr_residual_load_fcst"] = m.cutoff(inner).isoformat()
    (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        m.verify_profile_day(directory, inner, outer_day=next_outer)


def test_partial_day_resumes_only_missing_alias_and_rejects_corrupt_partial(monkeypatch, tmp_path, clients):
    inner, outer = "2024-08-17", "2026-09-28"
    missing = m.specs()["fr_residual_load_fcst"]["series"]
    def fetch(*args, **kwargs):
        if args[1] == missing:
            raise RuntimeError("Missing FR at both revisions")
        return finite(*args, **kwargs)
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    with pytest.raises(m.SaturnSourceError):
        m._sync_profile_day(inner, outer, tmp_path, clients[0])
    directory = tmp_path / "profiles_per_series_v2" / outer / inner
    assert len(list((directory / "series").glob("*.json"))) == 13
    partial = directory / "series" / "de_residual_load_fcst.json"
    original = partial.read_bytes()
    damaged = json.loads(original)
    damaged["values"][0] += 1
    partial.write_text(json.dumps(damaged), encoding="utf-8")
    def restored(*args, **kwargs):
        assert args[1] == missing, "Existing valid alias queried again"
        return finite(*args, **kwargs)
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", restored)
    with pytest.raises(ValueError, match="partial series values"):
        m._sync_profile_day(inner, outer, tmp_path, clients[0])
    partial.write_bytes(original)
    before = len(clients[1])
    receipt = m._sync_profile_day(inner, outer, tmp_path,
        lambda: pytest.fail("all fourteen successful partial series should already exist"))
    assert receipt["protocol"] == m.PROFILE_PROTOCOL and len(clients[1]) == before


def test_legacy_whole_cache_only_supplies_alias_after_own_origin_fails(monkeypatch, tmp_path, clients):
    inner, outer = "2024-08-17", "2026-09-28"
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", finite)
    old = m.capture_profile_day(object(), inner, tmp_path, outer_day=outer)
    assert old["profile_history_policy"] == m.WHOLE_PROFILE_HISTORY_POLICY
    calls = []
    def fetch(*args, **kwargs):
        calls.append(args[1])
        assert kwargs["revision_date"] == m.cutoff(inner)
        if args[1] == m.specs()["fr_residual_load_fcst"]["series"]:
            raise RuntimeError("Own FR missing")
        return finite(*args, **kwargs)
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    receipt = m._sync_profile_day(inner, outer, tmp_path, clients[0])
    assert len(calls) == 16 and len(set(calls)) == 14
    assert receipt["alias_sources"]["fr_residual_load_fcst"] == "outer_revision"
    assert receipt["alias_sources"]["nl_residual_load_fcst"] == "own_origin"
    assert receipt["alias_evidence"]["fr_residual_load_fcst"]["reused_whole_profile_receipt_sha256"] == m.sha256(
        tmp_path / "profiles_current_fit_v1" / outer / inner / "receipt.json")


def test_repository_cache_is_a_verified_copy_independent_of_mutable_input(monkeypatch, tmp_path):
    import hashlib
    from chronos2_hourly import nyx_annual_saturn_archive as archive
    raw = b"test immutable repository source"
    tracked = tmp_path / "tracked.parquet"
    tracked.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(archive, "DEFAULT_ARCHIVE", tracked)
    monkeypatch.setattr(archive, "PINNED_ARCHIVE_SHA256", digest)
    path = m._repository_archive_path(tmp_path / "cache")
    assert path.read_bytes() == raw and not path.samefile(tracked)
    tracked.write_bytes(b"later NYX source refresh")
    assert m._repository_archive_path(tmp_path / "cache") == path and path.read_bytes() == raw
    path.write_bytes(b"corrupt immutable proof")
    with pytest.raises(ValueError, match="repository cache changed"):
        m._repository_archive_path(tmp_path / "cache")


@pytest.mark.parametrize("nl_recovered", [False, True])
def test_mixed_nl_spring_ledger_uses_nl_actual_revision(monkeypatch, tmp_path, clients, nl_recovered):
    inner, outer = "2025-03-30", "2026-09-28"
    expected = m.grid(inner)
    missing = expected[expected.tz_convert("Europe/Amsterdam").strftime("%H:%M").isin(["04:00", "06:00"])]
    failed = "nl_residual_load_fcst" if nl_recovered else "fr_residual_load_fcst"
    def fetch(*args, **kwargs):
        if args[1] == m.specs()[failed]["series"] and kwargs["revision_date"] == m.cutoff(inner):
            raise RuntimeError("Missing own profile")
        values = finite(*args, **kwargs)
        return values.drop(missing) if args[1] == m.specs()["nl_residual_load_fcst"]["series"] else values
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    receipt = m._sync_profile_day(inner, outer, tmp_path, clients[0])
    actual = m.cutoff(outer if nl_recovered else inner).isoformat()
    assert all(e["forecast_origin_utc"] == actual for e in receipt["spring_dst_repair"]["entries"])
    assert receipt["alias_revisions_utc"]["nl_residual_load_fcst"] == actual
    directory = tmp_path / "profiles_per_series_v2" / outer / inner
    receipt["alias_revisions_utc"].pop("be_wind_generation_fcst")
    (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="revision aliases"):
        m.verify_profile_day(directory, inner, outer_day=outer)


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
    outer, recovered, archived = "2026-09-28", "2026-03-29", "2025-09-30"
    def fetch(*args, **kwargs):
        if args[1] == m.specs()["fr_residual_load_fcst"]["series"] and kwargs["revision_date"] == m.cutoff(recovered):
            raise RuntimeError("Missing original FR profile")
        if args[1] == m.specs()["nl_residual_load_fcst"]["series"] and args[2] == m.grid(archived)[0] - pd.Timedelta(hours=8):
            raise RuntimeError("NL unavailable at both queried revisions")
        values = finite(*args, **kwargs)
        if args[1] == m.specs()["nl_wind_generation_fcst"]["series"] and args[2] == m.grid(recovered)[0] - pd.Timedelta(hours=8):
            return values.drop(pd.Timestamp(recovered + "T02:00:00Z"))
        return values
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    first = (pd.Timestamp(outer).date() - timedelta(days=365)).isoformat()
    cache, bundle = tmp_path / "cache", tmp_path / "bundle"
    m.capture_target_snapshot(object(), outer, first, cache, now_utc=m.cutoff(outer))
    for stamp in pd.date_range(first, outer, freq="D"):
        day = stamp.date().isoformat()
        if day in (recovered, archived):
            def component(name, **kwargs):
                from chronos2_hourly.nyx_annual_wind_source import COMPONENT
                assert name == COMPONENT and kwargs["revision_date"] == m.cutoff(recovered)
                return pd.Series([4321.], index=pd.DatetimeIndex([kwargs["from_value_date"]]))
            m._sync_profile_day(day, outer, cache, lambda: SimpleNamespace(get=component))
        else:
            m.capture_profile_day(object(), day, cache, now_utc=m.cutoff(outer))
    path = m.publish(bundle, outer, first_day=first, cache=cache)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert m.profile_history_contract(bundle) == m._profile_contract(outer)
    origin = pd.read_parquet(bundle / "source_artifacts/saturn/origins.parquet")
    revision_path = bundle / "source_artifacts/saturn/profile_revisions.parquet"
    revision = pd.read_parquet(revision_path)
    assert list(revision) == list(m.ALIASES)
    assert (origin.loc[m.grid(recovered), "forecast_origin_utc"] == m.cutoff(recovered)).all()
    assert (revision.loc[m.grid(recovered), "fr_residual_load_fcst"] == m.cutoff(outer)).all()
    assert (revision.loc[m.grid(recovered), "de_residual_load_fcst"] == m.cutoff(recovered)).all()
    raw_wind_day = json.loads((bundle / f"source_artifacts/saturn/days/{recovered}/receipt.json").read_text())
    assert raw_wind_day["source_substitution_count"] == 1
    assert raw_wind_day["alias_evidence"]["nl_wind_generation_fcst"]["source_substitutions"] == raw_wind_day["source_substitutions"]
    archive_relative = f"source_artifacts/saturn/days/{archived}/nl_repository_vintages.parquet"
    assert (bundle / archive_relative).is_file()
    assert receipt["artifact_sha256"][archive_relative] == m.sha256(bundle / archive_relative)
    raw_archive_day = json.loads((bundle / f"source_artifacts/saturn/days/{archived}/receipt.json").read_text())
    archive_revision = pd.Timestamp(raw_archive_day["alias_revisions_utc"]["nl_residual_load_fcst"])
    assert (revision.loc[m.grid(archived), "nl_residual_load_fcst"] == archive_revision).all()
    unbound = json.loads(json.dumps(receipt))
    unbound["artifact_sha256"].pop(archive_relative)
    with pytest.raises(ValueError, match="source artifact changed"):
        m.verify_current_fit_source(bundle, outer, unbound)
    downgraded = json.loads(json.dumps(receipt))
    downgraded["profile_history_policy"] = m.WHOLE_PROFILE_HISTORY_POLICY
    for day, declaration in downgraded["daily_vintages"].items():
        declaration.update(forecast_origin_utc=declaration["profile_revision_max_utc"],
            profile_revision_utc=declaration["profile_revision_max_utc"],
            profile_origin_snapshot_verified=all(declaration["alias_origins_verified"].values()))
    with pytest.raises(ValueError, match="old whole-day policy"):
        m.verify_current_fit_source(bundle, outer, downgraded)
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
    for day, declared in receipt["daily_vintages"].items():
        declared["forecast_origin_utc"] = m.cutoff(day).isoformat()
    with pytest.raises(ValueError, match="per-series logical origin"):
        m.verify_current_fit_source(bundle, outer, receipt)
    # A real old packet containing only strict profiles remains readable and
    # reusable. Replace the recovered day with independently captured strict
    # evidence before converting the envelope to the old schema.
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", finite)
    for day in (recovered, archived):
        strict = m.capture_profile_day(object(), day, cache, now_utc=m.cutoff(outer))
        for name in ("receipt.json", "covariates.parquet"):
            relative = f"source_artifacts/saturn/days/{day}/{name}"
            portable = bundle / relative
            portable.unlink()  # Break the test packet's hard link, preserving the cache.
            shutil.copyfile(cache / "profiles_v2" / day / name, portable)
            receipt["artifact_sha256"][relative] = m.sha256(portable)
        receipt["daily_vintages"][day]["covariates_sha256"] = strict["artifact_sha256"]["covariates.parquet"]
    (bundle / archive_relative).unlink()
    receipt["artifact_sha256"].pop(archive_relative)
    combined = pd.concat(pd.read_parquet(bundle / f"source_artifacts/saturn/days/{d}/covariates.parquet")
                         for d in receipt["daily_vintages"])
    combined_path = "source_artifacts/saturn/covariates.parquet"
    combined.to_parquet(bundle / combined_path)
    receipt["artifact_sha256"][combined_path] = m.sha256(bundle / combined_path)
    for day, declared in receipt["daily_vintages"].items():
        declared["forecast_origin_utc"] = m.cutoff(day).isoformat()
        for field in ("logical_forecast_origin_utc", "alias_revisions_utc", "alias_sources", "alias_origins_verified", "profile_revision_max_utc"):
            declared.pop(field)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert m.profile_history_contract(bundle) == {}
    original = path.read_bytes()
    assert m.publish(bundle, outer, cache=tmp_path / "absent-cache") == path
    assert path.read_bytes() == original
