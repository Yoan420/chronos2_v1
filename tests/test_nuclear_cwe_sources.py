from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nuclear_cwe_sources as sources


def _seed(root, alias, first, last):
    spec = sources.CWE_CAPACITY_SOURCES[alias]
    days = pd.date_range(first, last, freq="D")
    blocks = []
    for day in days:
        blocks.append(pd.DataFrame({"value_time_utc": sources._physical(day, day),
            "snapshot_time_utc": sources._cutoff(day), "revision_time_utc": sources._cutoff(day),
            "value": 2. if spec["zone"] == "BE" else .492,
            "downloaded_at_utc": pd.Timestamp("2026-09-10", tz="UTC")}))
    frame = pd.concat(blocks, ignore_index=True)
    path = root / spec["seed_relative_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    audit = {"schema_version": 1, "alias": alias, "series": spec["series"], "unit": "GW",
             "information_type": "capacity_forecast", "daily_broadcast": True,
             "cutoff_time": "08:00", "cutoff_timezone": "Europe/Paris", "complete": True,
             "provider_revision_timestamp_available": False, "production_pit_evidence": False,
             "revision_time_semantics": "query_asof_cutoff", "fill_or_interpolation": sources.FILL_POLICY,
             "start_day": first, "end_day": last, "rows": len(frame), "days": len(days),
             "sha256": sources._sha(path)}
    sources._json(path.with_name(path.name + ".audit.json"), audit)
    return path


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "ROOT", tmp_path)
    originals = [_seed(tmp_path, alias, "2025-10-25", "2025-10-26") for alias in sources.CWE_CAPACITY_SOURCES]
    calls = []
    def fake(spec, day):
        calls.append((spec["alias"], day.date().isoformat()))
        return {"schema_version": 1, "alias": spec["alias"], "series": spec["series"],
                "day": day.date().isoformat(), "cutoff_utc": sources._cutoff(day).isoformat(),
                "value_gw": 0., "downloaded_at_utc": pd.Timestamp("2026-09-10", tz="UTC").isoformat(),
                "forecast_type": sources.FORECAST_TYPE, "provider_revision_timestamp_available": False}
    monkeypatch.setattr(sources, "_fetch_day", fake)
    return tmp_path / "data/pit/nuclear_cwe", originals, calls, fake


def test_reuse_old_hours_query_only_missing_and_preserve_incumbent(setup):
    output, originals, calls, _ = setup
    before = {p: sources._sha(p) for p in originals}
    result = sources.materialize_cwe_sources(output, "2025-10-24", "2025-10-27", 2)
    assert set(result) == set(sources.CWE_CAPACITY_SOURCES)
    assert len(calls) == 4
    assert {day for _, day in calls} == {"2025-10-24", "2025-10-27"}
    for alias, value in result.items():
        frame = pd.read_parquet(value["path"])
        assert len(frame) == 97  # 24 + 24 + 25 + 24 physical hours.
        assert value["audit"]["covered_hours"] == 97
        assert value["audit"]["seed_hours_reused"] == 49
        assert value["audit"]["forecast_type"] == sources.FORECAST_TYPE
        assert value["audit"]["production_pit_evidence"] is False
        assert value["audit"]["national_fleet_attested"] is False
        selected = frame.set_index("value_time_utc").reindex(pd.read_parquet(originals[list(result).index(alias)]).value_time_utc)
        original = pd.read_parquet(originals[list(result).index(alias)]).set_index("value_time_utc")
        pd.testing.assert_frame_equal(selected, original, check_dtype=False)
        assert frame.value.eq(0).sum() == 48  # Valid forecast zero, no filling.
    assert {p: sources._sha(p) for p in originals} == before


def test_completed_range_reused_without_network_or_seed_refresh(setup, monkeypatch):
    output, originals, _, _ = setup
    first = sources.materialize_cwe_sources(output, "2025-10-25", "2025-10-27", 1)
    before = {alias: sources._sha(Path(value["path"])) for alias, value in first.items()}
    monkeypatch.setattr(sources, "_fetch_day", lambda *args: pytest.fail("No network on completed range"))
    for path in originals:
        path.write_bytes(b"changed upstream cache after snapshot")
    again = sources.materialize_cwe_sources(output, "2025-10-25", "2025-10-27", 1)
    assert {alias: sources._sha(Path(value["path"])) for alias, value in again.items()} == before


def test_incomplete_fetch_preserves_valid_day_checkpoints_then_resumes(setup, monkeypatch):
    output, _, calls, fake = setup
    def unavailable(spec, day):
        if day == pd.Timestamp("2025-10-28"):
            raise ValueError("No vintage")
        return fake(spec, day)
    monkeypatch.setattr(sources, "_fetch_day", unavailable)
    with pytest.raises(ValueError, match="No vintage"):
        sources.materialize_cwe_sources(output, "2025-10-25", "2025-10-28", 1)
    for alias in sources.CWE_CAPACITY_SOURCES:
        assert (output / alias / "days/2025-10-27.json").is_file()
        assert not (output / alias / "bundles/2025-10-25_2025-10-28").exists()
    calls.clear()
    monkeypatch.setattr(sources, "_fetch_day", fake)
    result = sources.materialize_cwe_sources(output, "2025-10-25", "2025-10-28", 1)
    assert len(result) == 2 and len(calls) == 2
    assert all(day == "2025-10-28" for _, day in calls)


@pytest.mark.parametrize("key,value", [("series", "power.be.generation.nuclear.gw.fcst"),
                                     ("unit", "MW"), ("daily_broadcast", False),
                                     ("production_pit_evidence", True), ("complete", False)])
def test_seed_semantics_must_not_be_relabelled(setup, key, value):
    output, originals, calls, _ = setup
    sidecar = originals[0].with_name(originals[0].name + ".audit.json")
    audit = json.loads(sidecar.read_text())
    audit[key] = value
    sidecar.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="contract mismatch"):
        sources.materialize_cwe_sources(output, "2025-10-25", "2025-10-26", 1)
    assert not any(alias.startswith("be") for alias, _ in calls)


@pytest.mark.parametrize("mutation", ["sha", "naive", "cutoff", "missing_hour", "varying_capacity", "missing_capacity"])
def test_invalid_seed_bytes_or_timing_rejected(setup, mutation):
    output, originals, _, _ = setup
    path = originals[0]
    frame = pd.read_parquet(path)
    if mutation == "naive":
        frame["snapshot_time_utc"] = frame.snapshot_time_utc.dt.tz_localize(None)
    elif mutation == "cutoff":
        frame.loc[0, "revision_time_utc"] += pd.Timedelta(hours=1)
    elif mutation == "missing_hour":
        frame = frame.iloc[1:]
    elif mutation == "varying_capacity":
        frame.loc[0, "value"] += .2
    elif mutation == "missing_capacity":
        frame.loc[0, "value"] = np.nan
    else:
        frame.loc[0, "value"] = 1.
    frame.to_parquet(path, index=False)
    if mutation != "sha":
        sidecar = path.with_name(path.name + ".audit.json")
        audit = json.loads(sidecar.read_text())
        audit["sha256"] = sources._sha(path)
        sidecar.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError):
        sources.materialize_cwe_sources(output, "2025-10-25", "2025-10-26", 1)


def test_day_checkpoint_tamper_is_not_silently_replaced(setup, monkeypatch):
    output, _, _, fake = setup
    def unavailable(spec, day):
        if day == pd.Timestamp("2025-10-28"):
            raise ValueError("temporarily unavailable")
        return fake(spec, day)
    monkeypatch.setattr(sources, "_fetch_day", unavailable)
    with pytest.raises(ValueError):
        sources.materialize_cwe_sources(output, "2025-10-25", "2025-10-28", 1)
    path = output / "be_nuclear_available_gw/days/2025-10-27.json"
    envelope = json.loads(path.read_text())
    envelope["record"]["value_gw"] = 4.
    path.write_text(json.dumps(envelope), encoding="utf-8")
    before = sources._sha(path)
    monkeypatch.setattr(sources, "_fetch_day", fake)
    with pytest.raises(ValueError, match="checkpoint SHA256 mismatch"):
        sources.materialize_cwe_sources(output, "2025-10-25", "2025-10-28", 1)
    assert sources._sha(path) == before


def test_published_bundle_tamper_rejected_without_rewrite(setup):
    output, _, _, _ = setup
    result = sources.materialize_cwe_sources(output, "2025-10-25", "2025-10-26", 1)
    path = Path(result["be_nuclear_available_gw"]["path"])
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        sources.materialize_cwe_sources(output, "2025-10-25", "2025-10-26", 1)
    assert path.read_bytes() == b"corrupt"


def test_spring_dst_daily_capacity_has_23_physical_hours(setup):
    output, _, _, _ = setup
    result = sources.materialize_cwe_sources(output, "2026-03-29", "2026-03-30", 1)
    for value in result.values():
        frame = pd.read_parquet(value["path"])
        assert len(frame) == 47
        monday = frame.value_time_utc.dt.tz_convert("Europe/Paris").dt.date.eq(pd.Timestamp("2026-03-30").date())
        assert frame.loc[monday, "snapshot_time_utc"].eq(pd.Timestamp("2026-03-29T06:00:00Z")).all()


@pytest.mark.parametrize("workers", [0, 3, True])
def test_worker_bounds(setup, workers):
    with pytest.raises(ValueError, match="workers"):
        sources.materialize_cwe_sources(setup[0], "2025-10-25", "2025-10-26", workers)


def test_output_isolated_and_structural_zeros_are_not_forecast_channels(setup):
    output, _, _, _ = setup
    with pytest.raises(ValueError, match="data/pit/nuclear_cwe"):
        sources.materialize_cwe_sources(output.parent / "nuclear_forecast", "2025-10-25", "2025-10-26", 1)
    assert set(sources.STRUCTURAL_ZERO_COUNTRIES) == {"DE", "AT", "LU"}
    assert all(not record["model_channel"] for record in sources.STRUCTURAL_ZERO_COUNTRIES.values())
    assert set(sources.CWE_CAPACITY_SOURCES) == {"be_nuclear_available_gw", "nl_nuclear_available_gw"}


def test_specification_cannot_switch_capacity_to_generation(setup):
    spec = deepcopy(sources.CWE_CAPACITY_SOURCES["be_nuclear_available_gw"])
    spec["forecast_type"] = "generation_forecast"
    with pytest.raises(ValueError, match="catalogued"):
        sources.audit_cwe_source(Path("unused"), spec, "2025-10-25", "2025-10-26")
