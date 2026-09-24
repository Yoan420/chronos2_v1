import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from materialize_saturn_kalman_weather import build_plan
from chronos2_hourly import solar_cwe_sources as sources


def artifact(plan, first="2025-10-24", last="2025-10-27"):
    plan.output.parent.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(first, last)
    stamps = pd.date_range(days[0].tz_localize(plan.timezone),
                           (days[-1]+pd.Timedelta(days=1)).tz_localize(plan.timezone),
                           freq="h", inclusive="left").tz_convert("UTC")
    civil = stamps.tz_convert(plan.timezone).tz_localize(None).normalize()
    origins = (civil-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize(plan.timezone).tz_convert("UTC")
    pd.DataFrame({"value_time_utc": stamps, "snapshot_time_utc": origins, "revision_time_utc": origins,
                  "value": 0., "downloaded_at_utc": origins+pd.Timedelta(hours=1)}).to_parquet(plan.output, index=False)
    policy = "duplicate_zero_only" if plan.zone == "NL" else "duplicate"
    metadata = {"schema_version": 1, "alias": plan.alias, "series": plan.series,
                "timezone": plan.timezone, "naive_timezone": plan.naive_timezone,
                "cutoff_timezone": plan.timezone, "cutoff_time": "08:00", "daily_broadcast": False,
                "value_scale": 1., "unit": "GW", "start_day": first, "end_day": last,
                "days": len(days), "rows": len(stamps), "sha256": hashlib.sha256(plan.output.read_bytes()).hexdigest(),
                "first_delivery_utc": stamps[0].isoformat(), "last_delivery_utc": stamps[-1].isoformat(),
                "incomplete_dst_policy": policy, "dst_zero_duplicate_repairs": [], "dst_zero_duplicate_repair_count": 0,
                "fill_or_interpolation": "none_except_verified_zero_duplicate_for_singleton_autumn_fold" if policy == "duplicate_zero_only" else "none_except_duplicate_missing_autumn_fold_if_source_is_civil_naive",
                "causal_contract": "Saturn state queried as-of D-1 civil cutoff",
                "snapshot_time_semantics": "query_asof_cutoff",
                "revision_time_semantics": "query_asof_cutoff; provider insertion timestamp is not returned by Client.get(revision_date=...)",
                "provider_revision_timestamp_available": False}
    plan.audit_path.write_text(json.dumps(metadata), encoding="utf8")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    root = tmp_path/"project"
    root.mkdir()
    monkeypatch.setattr(sources, "ROOT", root)
    output = root/"data/pit/solar_cwe/test"
    return root, output


def seed(root, first="2025-10-24", last="2025-10-27"):
    directory = root/"runs/experiments/nyx_scarcity_v1/source_refresh/latest_verified"
    plans = [p for p in build_plan(sources.ZONES, directory) if p.kind == "solar"]
    for plan in plans:
        artifact(plan, first, last)
    return plans


def test_readonly_missing_cache_fails_without_copy_or_download(sandbox, monkeypatch):
    root, output = sandbox
    seed(root)
    monkeypatch.setattr(sources.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected process"))
    with pytest.raises(ValueError, match="missing/invalid"):
        sources.ensure_solar_sources(output_root=output, start_day="2025-10-25", end_day="2025-10-27")
    assert not output.exists()


def test_sync_copies_deeper_seed_without_download_then_readonly_reuses(sandbox, monkeypatch):
    root, output = sandbox
    plans = seed(root)
    before = {p.output: p.output.read_bytes() for p in plans}
    monkeypatch.setattr(sources.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected process"))
    result = sources.ensure_solar_sources(output_root=output, start_day="2025-10-25", end_day="2025-10-27", sync=True)
    assert set(result) == set(sources.SOLAR_SERIES)
    for name, record in result.items():
        assert Path(record["path"]).is_relative_to(output)
        assert record["start_day"] == "2025-10-24" and record["complete"]
        assert not record["production_pit_evidence"] and not record["provider_revision_timestamp_available"]
        assert record["downloaded_suffix_start_day"] is None
        assert record["series"] == sources.SOLAR_SERIES[name]
    assert before == {p.output: p.output.read_bytes() for p in plans}
    reread = sources.ensure_solar_sources(output_root=output, start_day="2025-10-25", end_day="2025-10-26")
    assert all(r["end_day"] == "2025-10-27" for r in reread.values())


def test_sync_only_missing_suffix_uses_four_solar_plans_and_shell_false(sandbox, monkeypatch):
    root, output = sandbox
    seeds = seed(root)
    before = {p.output: p.output.read_bytes() for p in seeds}
    calls = []
    def execute(command, **kwargs):
        calls.append((command, kwargs))
        get = lambda flag: command[command.index(flag)+1]
        assert get("--start-day") == "2025-10-28" and get("--end-day") == "2025-10-29"
        assert "--merge-existing" in command and kwargs["shell"] is False and kwargs["check"] is True
        plan = next(p for p in build_plan(sources.ZONES, output) if p.alias == get("--alias"))
        assert plan.kind == "solar"
        artifact(plan, last="2025-10-29")
    monkeypatch.setattr(sources.subprocess, "run", execute)
    result = sources.ensure_solar_sources(output_root=output, start_day="2025-10-25", end_day="2025-10-29", sync=True)
    assert len(calls) == 4
    assert all(r["end_day"] == "2025-10-29" for r in result.values())
    assert before == {p.output: p.output.read_bytes() for p in seeds}


@pytest.mark.parametrize("problem", ["prefix", "long_suffix", "checksum", "nl_timezone", "negative", "naive", "partial"])
def test_invalid_cache_or_unbounded_collection_fails_closed(sandbox, monkeypatch, problem):
    root, output = sandbox
    plans = seed(root)
    first, last = "2025-10-24", "2025-10-27"
    if problem == "prefix":
        first = "2025-10-23"
    elif problem == "long_suffix":
        last = "2025-12-01"
    else:
        output.mkdir(parents=True)
        plan = next(p for p in build_plan(sources.ZONES, output) if p.zone == "NL" and p.kind == "solar")
        artifact(plan)
        metadata = json.loads(plan.audit_path.read_text())
        if problem == "partial":
            plan.audit_path.unlink()
        elif problem == "checksum":
            metadata["sha256"] = "wrong"
            plan.audit_path.write_text(json.dumps(metadata))
        elif problem == "nl_timezone":
            metadata["naive_timezone"] = "UTC"
            plan.audit_path.write_text(json.dumps(metadata))
        else:
            frame = pd.read_parquet(plan.output)
            if problem == "negative":
                frame.loc[0, "value"] = -1.
            else:
                frame["downloaded_at_utc"] = frame.downloaded_at_utc.dt.tz_localize(None)
            frame.to_parquet(plan.output, index=False)
            metadata["sha256"] = hashlib.sha256(plan.output.read_bytes()).hexdigest()
            plan.audit_path.write_text(json.dumps(metadata))
    monkeypatch.setattr(sources.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected process"))
    with pytest.raises(ValueError):
        sources.ensure_solar_sources(output_root=output, start_day=first, end_day=last, sync=True)


def test_readonly_incomplete_suffix_does_not_fetch(sandbox, monkeypatch):
    _, output = sandbox
    for p in build_plan(sources.ZONES, output):
        if p.kind == "solar":
            artifact(p)
    monkeypatch.setattr(sources.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected process"))
    with pytest.raises(ValueError, match="missing suffix"):
        sources.ensure_solar_sources(output_root=output, start_day="2025-10-24", end_day="2025-10-28")


def test_namespace_and_invalid_workers_are_rejected(sandbox):
    root, output = sandbox
    with pytest.raises(ValueError, match="data/pit/solar_cwe"):
        sources.ensure_solar_sources(output_root=root/"data/pit/kalman_weather", start_day="2025-10-24", end_day="2025-10-27", sync=True)
    with pytest.raises(ValueError, match="workers"):
        sources.ensure_solar_sources(output_root=output, start_day="2025-10-24", end_day="2025-10-27", workers=True)
