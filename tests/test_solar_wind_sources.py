import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from chronos2_hourly import solar_wind_sources as sources


def artifact(plan, first="2025-10-24", last="2025-10-27", *, enriched=False, value=4.):
    plan.output.parent.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(first, last)
    stamps = pd.date_range(days[0].tz_localize(plan.timezone),
                           (days[-1] + pd.Timedelta(days=1)).tz_localize(plan.timezone),
                           freq="h", inclusive="left").tz_convert("UTC")
    civil = stamps.tz_convert(plan.timezone).tz_localize(None).normalize()
    origins = (civil - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize(plan.timezone).tz_convert("UTC")
    frame = pd.DataFrame({"value_time_utc": stamps, "snapshot_time_utc": origins, "revision_time_utc": origins,
                          "value": value if plan.kind == "wind" else 0.,
                          "downloaded_at_utc": origins + pd.Timedelta(hours=1)})
    frame.to_parquet(plan.output, index=False)
    policy = "duplicate_zero_only" if plan.alias == "nl_solar_generation_fcst" else "duplicate"
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
                "revision_time_semantics": "query_asof_cutoff; provider insertion timestamp unavailable",
                "provider_revision_timestamp_available": False}
    if enriched and plan.kind == "wind":
        repairs = sources._repair_rows(plan, frame, evidence="test_raw_singleton")
        metadata.update(wind_dst_policy="duplicate", fill_or_interpolation=sources.WIND_FILL_POLICY,
                        dst_duplicate_repairs=repairs, dst_duplicate_repair_count=len(repairs))
    plan.audit_path.write_text(json.dumps(metadata), encoding="utf8")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(sources, "ROOT", root)
    return root, root / "data/pit/solar_wind_v1/test"


def seed(root, *, native_nl=True):
    plans = sources._plans(root / "data/pit/solar_cwe")
    for plan in plans:
        if plan.kind == "solar":
            artifact(plan)
    windplans = sources._plans(root / "data/pit/kalman_weather")
    for plan in windplans:
        if plan.kind == "wind":
            artifact(plan)
            if plan.zone == "NL" and not native_nl:
                metadata = json.loads(plan.audit_path.read_text())
                metadata.update(series="power.nl.prod.total.wind.mw.ecmwf_avg.pointconnect.6h.cache",
                                naive_timezone="UTC", value_scale=.001)
                plan.audit_path.write_text(json.dumps(metadata))
    return [p for p in plans if p.kind == "solar"] + [p for p in windplans if p.kind == "wind"]


def test_exact_allowlist_never_substitutes_nl_ecmwf(sandbox):
    _, output = sandbox
    plans = sources._plans(output)
    assert len(plans) == 6
    assert {p.alias: p.series for p in plans} == sources.GENERATION_SERIES
    nl = next(p for p in plans if p.alias == "nl_wind_generation_fcst")
    assert nl.series == "power.nl.generation.wind.hourly.gw.fcst"
    assert nl.naive_timezone == "Europe/Amsterdam" and nl.value_scale == 1.


def test_readonly_never_copies_seeds(sandbox, monkeypatch):
    root, output = sandbox
    seed(root)
    monkeypatch.setattr(sources, "_materialize_wind", lambda *a, **k: pytest.fail("Unexpected download"))
    with pytest.raises(sources.SolarWindSourceError, match="missing/invalid"):
        sources.ensure_solar_wind_sources(output_root=output, start_day="2025-10-24", end_day="2025-10-27")
    assert not output.exists()


def test_seed_reuse_is_immutable_and_audits_nonzero_wind_folds(sandbox, monkeypatch):
    root, output = sandbox
    seeds = seed(root)
    before = {path: path.read_bytes() for p in seeds for path in (p.output, p.audit_path)}
    monkeypatch.setattr(sources, "_materialize_wind", lambda *a, **k: pytest.fail("Unexpected download"))
    result = sources.ensure_solar_wind_sources(output_root=output, start_day="2025-10-25", end_day="2025-10-27", sync=True)
    assert set(result) == set(sources.GENERATION_SERIES)
    for alias in sources.WIND_SERIES:
        row = result[alias]
        assert row["dst_duplicate_repair_count"] == 1
        repair = row["dst_duplicate_repairs"][0]
        assert repair["duplicated_value"] == 4.
        assert repair["physical_hours_utc"] == ["2025-10-26T00:00:00+00:00", "2025-10-26T01:00:00+00:00"]
        assert "raw_singleton_not_requeried" in repair["evidence"]
        assert row["fill_or_interpolation"] == sources.WIND_FILL_POLICY
        assert not row["production_pit_evidence"]
    assert before == {path: path.read_bytes() for path in before}
    assert set(sources.ensure_solar_wind_sources(output_root=output, start_day="2025-10-25", end_day="2025-10-27")) == set(result)


def test_native_nl_initial_history_download_ignores_old_ecmwf_seed(sandbox, monkeypatch):
    root, output = sandbox
    seed(root, native_nl=False)
    calls = []
    def materialize(plan, first, last, workers, *, merge_existing, wind_dst_policy, wind_gap_policy):
        calls.append(plan)
        assert plan.alias == "nl_wind_generation_fcst"
        assert plan.series == sources.WIND_SERIES[plan.alias]
        assert str(first.date()) == "2025-10-24" and not merge_existing and workers == 2
        artifact(plan, enriched=True)
    monkeypatch.setattr(sources, "_materialize_wind", materialize)
    result = sources.ensure_solar_wind_sources(output_root=output, start_day="2025-10-24", end_day="2025-10-27", sync=True)
    assert len(calls) == 1
    assert result["nl_wind_generation_fcst"]["initial_native_wind_backfill"]
    assert result["nl_wind_generation_fcst"]["seed_path"] is None


def test_extension_uses_only_missing_suffix(sandbox, monkeypatch):
    root, output = sandbox
    seed(root)
    wind_calls, solar_calls = [], []
    def materialize(plan, first, last, workers, *, merge_existing, wind_dst_policy, wind_gap_policy):
        wind_calls.append(plan.alias)
        assert str(first.date()) == "2025-10-28" and str(last.date()) == "2025-10-29" and merge_existing
        artifact(plan, last="2025-10-29", enriched=True)
    def run(command, **kwargs):
        get = lambda flag: command[command.index(flag) + 1]
        solar_calls.append(get("--alias"))
        assert get("--start-day") == "2025-10-28" and "--merge-existing" in command
        assert kwargs["check"] and kwargs["shell"] is False
        plan = next(p for p in sources._plans(output) if p.alias == get("--alias"))
        artifact(plan, last="2025-10-29")
    monkeypatch.setattr(sources, "_materialize_wind", materialize)
    monkeypatch.setattr(sources.subprocess, "run", run)
    sources.ensure_solar_wind_sources(output_root=output, start_day="2025-10-24", end_day="2025-10-29", sync=True)
    assert set(wind_calls) == set(sources.WIND_SERIES) and set(solar_calls) == set(sources.SOLAR_SERIES)


@pytest.mark.parametrize("problem", ["missing_audit", "missing_repairs", "wrong_hours", "wrong_value", "negative", "partial_cache"])
def test_invalid_isolated_wind_fails_closed(sandbox, monkeypatch, problem):
    _, output = sandbox
    for plan in sources._plans(output):
        artifact(plan, enriched=True)
    plan = next(p for p in sources._plans(output) if p.alias == "nl_wind_generation_fcst")
    metadata = json.loads(plan.audit_path.read_text())
    if problem == "partial_cache":
        plan.audit_path.unlink()
    else:
        if problem == "missing_audit":
            metadata.pop("wind_dst_policy")
        elif problem == "missing_repairs":
            metadata.pop("dst_duplicate_repairs")
        elif problem == "wrong_hours":
            metadata["dst_duplicate_repairs"][0]["physical_hours_utc"].reverse()
        elif problem == "wrong_value":
            metadata["dst_duplicate_repairs"][0]["duplicated_value"] = 0.
        else:
            frame = pd.read_parquet(plan.output)
            frame.loc[0, "value"] = -1.
            frame.to_parquet(plan.output, index=False)
            metadata["sha256"] = hashlib.sha256(plan.output.read_bytes()).hexdigest()
        plan.audit_path.write_text(json.dumps(metadata))
    monkeypatch.setattr(sources, "_materialize_wind", lambda *a, **k: pytest.fail("Unexpected download"))
    with pytest.raises(sources.SolarWindSourceError):
        sources.ensure_solar_wind_sources(output_root=output, start_day="2025-10-24", end_day="2025-10-27", sync=True)


def test_namespace_worker_and_policy_guards(sandbox):
    root, output = sandbox
    base = dict(output_root=output, start_day="2025-10-24", end_day="2025-10-27")
    with pytest.raises(sources.SolarWindSourceError, match="solar_wind_v1"):
        sources.ensure_solar_wind_sources(**{**base, "output_root": root / "data/pit/kalman_weather"})
    with pytest.raises(sources.SolarWindSourceError, match="workers"):
        sources.ensure_solar_wind_sources(**base, workers=True)
    with pytest.raises(sources.SolarWindSourceError, match="no wind zero-fill"):
        sources.ensure_solar_wind_sources(**base, wind_dst_policy="duplicate_zero_only")


def test_initial_wind_materializer_preserves_repair_provenance(sandbox, monkeypatch):
    import materialize_saturn_daily_asof as materializer
    _, output = sandbox
    output.mkdir(parents=True)
    plan = next(p for p in sources._plans(output) if p.alias == "nl_wind_generation_fcst")
    def one_day(day, args):
        assert args.series == plan.series and args.incomplete_dst_policy == "duplicate"
        stamps = materializer._physical_utc_index(day, timezone=plan.timezone)
        origin = materializer._civil_cutoff(day, timezone=plan.timezone, cutoff_time="08:00").tz_convert("UTC")
        frame = pd.DataFrame({"value_time_utc": stamps, "snapshot_time_utc": origin, "revision_time_utc": origin,
                              "value": 6., "downloaded_at_utc": pd.Timestamp.now(tz="UTC")})
        frame.attrs["dst_repairs"] = sources._repair_rows(plan, frame, evidence="raw")
        return frame
    monkeypatch.setattr(materializer, "_one_day", one_day)
    sources._materialize_wind(plan, pd.Timestamp("2025-10-26"), pd.Timestamp("2025-10-26"), 2, merge_existing=False)
    record = sources._inspect(plan, pd.Timestamp("2025-10-26"), pd.Timestamp("2025-10-26"))
    assert record["audit"]["rows"] == 25
    assert record["dst_duplicate_repair_count"] == 1
    assert record["dst_duplicate_repairs"][0]["duplicated_value"] == 6.
    assert record["dst_duplicate_repairs"][0]["evidence"] == "raw_Saturn_singleton_at_query_asof"


def test_strict_wind_policy_rejects_legacy_duplicate_caches(sandbox):
    root, output = sandbox
    seed(root)
    sources.ensure_solar_wind_sources(output_root=output, start_day="2025-10-24", end_day="2025-10-27", sync=True)
    with pytest.raises(sources.SolarWindSourceError, match="DST provenance"):
        sources.ensure_solar_wind_sources(output_root=output, start_day="2025-10-24", end_day="2025-10-27",
                                          wind_dst_policy="raise")


def test_strict_materialization_accepts_true_physical_grid_without_repair(sandbox, monkeypatch):
    import materialize_saturn_daily_asof as materializer
    _, output = sandbox
    output.mkdir(parents=True)
    plan = next(p for p in sources._plans(output) if p.alias == "nl_wind_generation_fcst")
    def one_day(day, args):
        assert args.incomplete_dst_policy == "raise"
        stamps = materializer._physical_utc_index(day, timezone=plan.timezone)
        origin = materializer._civil_cutoff(day, timezone=plan.timezone, cutoff_time="08:00").tz_convert("UTC")
        return pd.DataFrame({"value_time_utc": stamps, "snapshot_time_utc": origin, "revision_time_utc": origin,
                             "value": [float(i) for i in range(len(stamps))],
                             "downloaded_at_utc": pd.Timestamp.now(tz="UTC")})
    monkeypatch.setattr(materializer, "_one_day", one_day)
    sources._materialize_wind(plan, pd.Timestamp("2025-10-26"), pd.Timestamp("2025-10-26"), 2,
                              merge_existing=False, wind_dst_policy="raise")
    record = sources._inspect(plan, pd.Timestamp("2025-10-26"), pd.Timestamp("2025-10-26"), wind_dst_policy="raise")
    assert record["audit"]["rows"] == 25 and record["fill_or_interpolation"] == "none"
    assert record["dst_duplicate_repair_count"] == 0


def test_wind_backfill_resume_reuses_verified_day_checkpoints(sandbox, monkeypatch):
    import materialize_saturn_daily_asof as materializer
    _, output = sandbox
    output.mkdir(parents=True)
    plan = next(p for p in sources._plans(output) if p.alias == "nl_wind_generation_fcst")
    calls = []
    def one_day(day, args):
        calls.append(str(day.date()))
        stamps = materializer._physical_utc_index(day, timezone=plan.timezone)
        origin = materializer._civil_cutoff(day, timezone=plan.timezone, cutoff_time="08:00").tz_convert("UTC")
        return pd.DataFrame({"value_time_utc": stamps, "snapshot_time_utc": origin, "revision_time_utc": origin,
                             "value": 8., "downloaded_at_utc": pd.Timestamp.now(tz="UTC")})
    monkeypatch.setattr(materializer, "_one_day", one_day)
    sources._materialize_wind(plan, pd.Timestamp("2025-10-24"), pd.Timestamp("2025-10-25"), 1, merge_existing=False)
    before = {p: p.read_bytes() for p in output.glob("_wind_days/**/*.parquet*")}
    assert len(before) == 4 and calls == ["2025-10-24", "2025-10-25"]
    sources._materialize_wind(plan, pd.Timestamp("2025-10-24"), pd.Timestamp("2025-10-25"), 1, merge_existing=False)
    assert calls == ["2025-10-24", "2025-10-25"]
    assert before == {p: p.read_bytes() for p in before}
    checkpoint = next(output.glob("_wind_days/**/*.audit.json"))
    metadata = json.loads(checkpoint.read_text())
    metadata["sha256"] = "wrong"
    checkpoint.write_text(json.dumps(metadata))
    with pytest.raises(sources.SolarWindSourceError, match="checksum"):
        sources._materialize_wind(plan, pd.Timestamp("2025-10-24"), pd.Timestamp("2025-10-25"), 1, merge_existing=False)
    assert calls == ["2025-10-24", "2025-10-25"]


def gap_args(plan, *, policy=sources.NL_SPRING_GAP_POLICY):
    from argparse import Namespace
    return Namespace(series=plan.series, alias=plan.alias, timezone=plan.timezone,
                     naive_timezone=plan.naive_timezone, cutoff_timezone=plan.timezone, cutoff_time="08:00",
                     incomplete_dst_policy="duplicate", allow_incomplete_days=False, request_padding_hours=8,
                     wind_gap_policy=policy)


def day_frame(plan, day, missing=()):
    from materialize_saturn_daily_asof import _civil_cutoff, _physical_utc_index
    stamps = _physical_utc_index(day, timezone=plan.timezone)
    stamps = stamps[~stamps.isin(pd.DatetimeIndex(missing))]
    cutoff = _civil_cutoff(day, timezone=plan.timezone, cutoff_time="08:00").tz_convert("UTC")
    return pd.DataFrame({"value_time_utc": stamps, "snapshot_time_utc": cutoff, "revision_time_utc": cutoff,
                         "value": 11., "downloaded_at_utc": pd.Timestamp.now(tz="UTC")})


@pytest.mark.parametrize("daytext,raw_mw", [("2025-03-30", 7767.66), ("2026-03-29", 4049.499)])
def test_approved_spring_hour_substitution_same_cutoff_mw_to_gw(sandbox, monkeypatch, daytext, raw_mw):
    import materialize_saturn_daily_asof as materializer
    _, output = sandbox
    output.mkdir(parents=True)
    plan = next(p for p in sources._plans(output) if p.alias == "nl_wind_generation_fcst")
    day, hour = pd.Timestamp(daytext), sources.NL_SPRING_GAP_HOURS[daytext]
    expected_cutoff = (day - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize(plan.timezone).tz_convert("UTC")
    calls = []
    def native(request_day, args):
        assert request_day == day and args.allow_incomplete_days is True
        assert args.series == plan.series
        return day_frame(plan, day, [hour])
    class Client:
        def get(self, name, **kwargs):
            calls.append((name, kwargs))
            assert name == sources.NL_ECMWF_COMPONENT
            assert kwargs["revision_date"] == expected_cutoff and kwargs["nocache"] is True
            assert kwargs["from_value_date"] == hour
            return pd.Series([raw_mw, raw_mw + 100.], index=pd.DatetimeIndex([hour, hour + pd.Timedelta(hours=1)]))
    monkeypatch.setattr(materializer, "_one_day", native)
    monkeypatch.setattr(materializer, "_client", lambda args: Client())
    sources._materialize_wind(plan, day, day, 1, merge_existing=False, wind_gap_policy=sources.NL_SPRING_GAP_POLICY)
    frame = pd.read_parquet(plan.output)
    assert len(frame) == 23 and len(calls) == 1
    assert frame.loc[frame.value_time_utc.eq(hour), "value"].item() == raw_mw * .001
    assert frame.loc[frame.value_time_utc.ne(hour), "value"].eq(11.).all()
    record = sources._inspect(plan, day, day, wind_gap_policy=sources.NL_SPRING_GAP_POLICY)
    assert record["source_substitution_count"] == 1
    substitution = record["source_substitutions"][0]
    assert substitution["raw_value_mw"] == raw_mw and substitution["scaled_value_gw"] == raw_mw * .001
    assert substitution["query_cutoff_utc"] == expected_cutoff.isoformat()
    assert substitution["local_time"] == f"{daytext}T04:00:00+02:00"
    assert not substitution["production_pit_evidence"] and not substitution["provider_revision_timestamp_available"]
    with pytest.raises(sources.SolarWindSourceError, match="strict wind gap policy"):
        sources._inspect(plan, day, day)
    daily_audit = next(output.glob("_wind_days/**/*.audit.json"))
    assert json.loads(daily_audit.read_text())["source_substitutions"] == record["source_substitutions"]
    # A valid persisted affected-day checkpoint needs neither native nor fallback refetch.
    monkeypatch.setattr(materializer, "_one_day", lambda *a: pytest.fail("Unexpected native refetch"))
    monkeypatch.setattr(materializer, "_client", lambda *a: pytest.fail("Unexpected component refetch"))
    cached = sources._checkpoint_day(plan, day, gap_args(plan))
    assert cached.attrs["source_substitutions"] == record["source_substitutions"]


def test_native_priority_never_queries_ecmwf_for_finite_authorized_hour(sandbox, monkeypatch):
    import materialize_saturn_daily_asof as materializer
    _, output = sandbox
    plan = next(p for p in sources._plans(output) if p.alias == "nl_wind_generation_fcst")
    day = pd.Timestamp("2025-03-30")
    monkeypatch.setattr(materializer, "_one_day", lambda *a: day_frame(plan, day))
    monkeypatch.setattr(sources, "_fetch_ecmwf_hour", lambda *a: pytest.fail("Finite native must win"))
    result = sources._download_wind_day(plan, day, gap_args(plan))
    assert len(result) == 23 and result.value.eq(11.).all()
    assert not result.attrs.get("source_substitutions")


@pytest.mark.parametrize("case", ["different_hour", "extra_gap", "other_day", "de", "no_policy"])
def test_substitution_scope_is_exact_and_all_other_gaps_remain_strict(sandbox, monkeypatch, case):
    import materialize_saturn_daily_asof as materializer
    _, output = sandbox
    alias = "de_wind_generation_fcst" if case == "de" else "nl_wind_generation_fcst"
    plan = next(p for p in sources._plans(output) if p.alias == alias)
    day = pd.Timestamp("2025-03-31" if case == "other_day" else "2025-03-30")
    hour = pd.Timestamp("2025-03-30T02:00:00Z")
    def native(request_day, args):
        if case in {"other_day", "de", "no_policy"}:
            assert not args.allow_incomplete_days
            raise RuntimeError("native strict missing hour")
        missing = [hour + pd.Timedelta(hours=1)] if case == "different_hour" else [hour, hour + pd.Timedelta(hours=1)]
        return day_frame(plan, day, missing)
    monkeypatch.setattr(materializer, "_one_day", native)
    monkeypatch.setattr(sources, "_fetch_ecmwf_hour", lambda *a: pytest.fail("Out-of-scope fallback"))
    with pytest.raises((sources.SolarWindSourceError, RuntimeError)):
        sources._download_wind_day(plan, day, gap_args(plan, policy=None if case == "no_policy" else sources.NL_SPRING_GAP_POLICY))


@pytest.mark.parametrize("tamper", ["remove", "hour", "cutoff", "value", "component", "count"])
def test_corrupted_substitution_audit_is_rejected(sandbox, monkeypatch, tamper):
    import materialize_saturn_daily_asof as materializer
    _, output = sandbox
    output.mkdir(parents=True)
    plan = next(p for p in sources._plans(output) if p.alias == "nl_wind_generation_fcst")
    day, hour = pd.Timestamp("2025-03-30"), pd.Timestamp("2025-03-30T02:00:00Z")
    monkeypatch.setattr(materializer, "_one_day", lambda *a: day_frame(plan, day, [hour]))
    monkeypatch.setattr(sources, "_fetch_ecmwf_hour", lambda *a: (5000., pd.Timestamp.now(tz="UTC")))
    sources._materialize_wind(plan, day, day, 1, merge_existing=False, wind_gap_policy=sources.NL_SPRING_GAP_POLICY)
    audit = json.loads(plan.audit_path.read_text())
    if tamper == "remove":
        audit.pop("source_substitutions")
        audit.pop("source_substitution_count")
    elif tamper == "count":
        audit["source_substitution_count"] = 0
    else:
        key, value = {"hour": ("value_time_utc", "2025-03-30T03:00:00+00:00"),
                      "cutoff": ("query_cutoff_utc", "2025-03-29T08:00:00+00:00"),
                      "value": ("scaled_value_gw", 9.), "component": ("fallback_series", "other")}[tamper]
        audit["source_substitutions"][0][key] = value
    plan.audit_path.write_text(json.dumps(audit))
    with pytest.raises(sources.SolarWindSourceError, match="substitution"):
        sources._inspect(plan, day, day, wind_gap_policy=sources.NL_SPRING_GAP_POLICY)


def test_approved_policy_reuses_legacy_healthy_checkpoint_namespace_and_bytes(sandbox, monkeypatch):
    import materialize_saturn_daily_asof as materializer
    _, output = sandbox
    output.mkdir(parents=True)
    plan = next(p for p in sources._plans(output) if p.alias == "nl_wind_generation_fcst")
    day = pd.Timestamp("2025-03-28")
    contract = {"schema_version": 1, "series": plan.series, "alias": plan.alias,
                "timezone": plan.timezone, "naive_timezone": plan.naive_timezone,
                "cutoff_time": "08:00", "wind_dst_policy": "duplicate", "request_padding_hours": 8, "value_scale": 1.}
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:20]
    from dataclasses import replace
    checkpoint = replace(plan, output=output / "_wind_days" / plan.alias / identity / "2025-03-28.parquet")
    artifact(checkpoint, first="2025-03-28", last="2025-03-28", enriched=True)
    before = {p: p.read_bytes() for p in (checkpoint.output, checkpoint.audit_path)}
    monkeypatch.setattr(materializer, "_one_day", lambda *a: pytest.fail("Healthy checkpoint must not refetch"))
    result = sources._checkpoint_day(plan, day, gap_args(plan))
    assert len(result) == 24 and result.attrs["source_substitutions"] == []
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize("problem", ["naive", "duplicate", "negative", "nan", "missing"])
def test_fallback_component_must_be_unique_aware_finite_nonnegative(sandbox, monkeypatch, problem):
    import materialize_saturn_daily_asof as materializer
    _, output = sandbox
    plan = next(p for p in sources._plans(output) if p.alias == "nl_wind_generation_fcst")
    hour, cutoff = pd.Timestamp("2025-03-30T02:00:00Z"), pd.Timestamp("2025-03-29T07:00:00Z")
    raw = pd.Series([7000.], index=pd.DatetimeIndex([hour]))
    if problem == "naive":
        raw.index = raw.index.tz_localize(None)
    elif problem == "duplicate":
        raw = pd.concat([raw, raw])
    elif problem == "negative":
        raw.iloc[0] = -1.
    elif problem == "nan":
        raw.iloc[0] = float("nan")
    else:
        raw.index += pd.Timedelta(hours=1)
    class Client:
        def get(self, *a, **k):
            return raw
    monkeypatch.setattr(materializer, "_client", lambda args: Client())
    with pytest.raises(sources.SolarWindSourceError):
        sources._fetch_ecmwf_hour(gap_args(plan), hour, cutoff)
