"""Frozen-input orchestration boundaries for the full CGC/CCC variant."""
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from nyx_clean_fuel import full_runner as r
from nyx_clean_fuel import sources as s


@pytest.fixture
def scoped(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "ROOT", tmp_path)
    monkeypatch.setattr(r, "OUTPUT", tmp_path / "runs/experiments/nyx_clean_fuel_full_v1")
    monkeypatch.setattr(r.residual_lab, "ROOT", tmp_path)
    monkeypatch.setattr(r.residual_lab, "NAMESPACE", tmp_path / "runs/experiments/nyx_clean_fuel_v1")
    monkeypatch.setattr(s, "ROOT", tmp_path)
    evidence = {alias: {"series": s.SERIES[alias], "formula": s.FORMULAS[alias],
        "formula_sha256": s._sha(s.FORMULAS[alias].encode()), "unit": s.UNITS[alias], "tzaware": False}
        for alias in s.SERIES}
    monkeypatch.setattr(s, "_source_evidence", lambda settings: deepcopy(evidence))
    def one_day(day, settings):
        cutoff = s.civil_cutoff(day)
        stamp = (cutoff.tz_convert(s.TIMEZONE).normalize() - pd.DateOffset(days=1)).tz_convert("UTC")
        result = {"delivery_day": day, "cutoff_time_utc": cutoff.isoformat()}
        for number, alias in enumerate(s.SERIES):
            result.update({alias: 70. + number, alias + "__value_time_utc": stamp.isoformat(),
                alias + "__age_hours": (cutoff-stamp).total_seconds()/3600})
        return result
    monkeypatch.setattr(s, "_one_day", one_day)
    return tmp_path


def bank(scoped, start, end):
    return s.materialize({}, start, end, r.residual_lab.NAMESPACE / "inputs" / end)


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2026-10-25", 25), ("2026-09-18", 24)])
def test_native_close_bank_broadcasts_to_exact_physical_delivery_hours(scoped, day, hours):
    source = bank(scoped, day, day)
    frozen_hashes = {path: r.sha256(path) for path in (source, Path(str(source) + ".audit.json"))}
    destination = r.OUTPUT / day / "fr" / "test" / "snapshot"
    paths, audit = r.write_pit_costs(source, destination)
    assert set(paths) == set(s.SERIES)
    for number, (alias, path) in enumerate(paths.items()):
        frame = pd.read_parquet(path)
        assert list(frame) == ["value_time_utc", "snapshot_time_utc", "revision_time_utc", "value"]
        assert len(frame) == hours
        assert frame.value_time_utc.is_unique
        assert frame.value_time_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d").eq(day).all()
        assert frame.snapshot_time_utc.eq(s.civil_cutoff(day)).all()
        assert frame.revision_time_utc.eq(s.civil_cutoff(day)).all()
        assert frame.value.eq(70. + number).all()
    assert audit["production_pit_evidence"] is False
    assert {path: r.sha256(path) for path in frozen_hashes} == frozen_hashes
    initial = {alias: r.sha256(path) for alias, path in paths.items()}
    r.write_pit_costs(source, destination)
    assert {alias: r.sha256(path) for alias, path in paths.items()} == initial


def test_pit_conversion_rejects_existing_changed_artifact_and_operational_destination(scoped):
    source = bank(scoped, "2026-09-17", "2026-09-18")
    destination = r.OUTPUT / "test/snapshot"
    paths, _ = r.write_pit_costs(source, destination)
    frame = pd.read_parquet(paths["ccc"])
    frame.loc[0, "value"] += 10
    frame.to_parquet(paths["ccc"], index=False)
    with pytest.raises(ValueError, match="Frozen clean-fuel PIT values/schema changed for ccc"):
        r.write_pit_costs(source, destination)
    pd.testing.assert_frame_equal(pd.read_parquet(paths["ccc"]), frame)
    with pytest.raises(ValueError, match="isolated"):
        r.write_pit_costs(source, scoped / "runs/exports")
    assert not (scoped / "runs/exports").exists()


def settings():
    return {"schema_version": 1, "delivery_day": "2026-09-18", "zones": ["FR"],
        "output_root": "runs/experiments/nyx_clean_fuel_full_v1", "source_root": "runs/experiments/nuclear_forecast_v1",
        "fuel_sources": {"maximum_age_hours": 176}, "include_attribution": True,
        "diagnostic_only": True, "production_modified": False}


@pytest.mark.parametrize("key,value", [("output_root", "runs/exports"), ("diagnostic_only", False),
    ("production_modified", True), ("include_attribution", "yes"), ("zones", ["FR", "FR"]),
    ("source_root", "runs/live"), ("delivery_day", "2026-09-18T00:00:00")])
def test_full_settings_refuse_scope_and_date_errors(scoped, key, value):
    cfg = settings()
    cfg[key] = value
    path = scoped / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        r.load_settings(path)
    assert not r.OUTPUT.exists()


@pytest.fixture
def baseline(scoped, monkeypatch):
    import chronos2_hourly.nuclear_run_archive as archive
    import chronos2_hourly.nuclear_incremental as incremental
    cfg = settings()
    directory = r.residual_lab.source_dir(cfg, "FR")
    snapshot = directory / "snapshot"
    snapshot.mkdir(parents=True)
    paths = {}
    for alias in ("target", "fr_residual_load_fcst", "fr_nuclear_generation_fcst_gw"):
        source = snapshot / (alias + ".parquet")
        pd.DataFrame({"fixture": [1]}).to_parquet(source)
        paths[alias] = str(source)
    covariates = {alias: {"enabled": True, "source": "pit_parquet", "series": alias, "pit_file": path,
                          "future": {"known_future": True, "strategies": ["oracle"]}}
                  for alias, path in paths.items() if alias != "target"}
    original = {"data": {"project_root": str(scoped), "pit_files": {a: p for a, p in paths.items() if a != "target"}},
        "zones": {"FR": {"timezone": "Europe/Paris", "target": {"file": paths["target"]}, "covariates": covariates}},
        "output": {"directory": str(directory)}, "report": {"title": "untouched operational report"},
        "hourly": {"feature_engineering": {"covariate_columns": ["known_fr_nuclear_generation_fcst_gw_oracle"]},
                   "residual_correction": {"backend": "catboost", "max_depth": 4}},
        "nuclear_experiment": {"mode": "incremental", "history_anchor_day": "2024-09-09",
             "raw_history_start_day": "2024-09-09", "incremental_namespace": "incumbent", "filter_parameters": {"q_over_r": .001}}}
    (directory / "resolved_config.yaml").write_text(yaml.safe_dump(original), encoding="utf8")
    (directory / "input_snapshot.json").write_text(json.dumps({"files": [{"snapshot": p, "sha256": r.sha256(Path(p))} for p in paths.values()]}))
    (directory / "run_result.json").write_text(json.dumps({"reporting_sources": {"status": "complete"}}))
    frozen = directory / "report_only/frozen_result"
    frozen.mkdir(parents=True)
    (frozen / "manifest.json").write_text(json.dumps({"fixture": "immutable baseline bundle identity"}))
    frame = pd.DataFrame({"example": [1.]})
    incumbent = SimpleNamespace(audit={"history_anchor_day": "2024-09-09"}, residual_statistics=frame,
                               source_forecast=frame, kalman_view=SimpleNamespace(backtest=frame, forecast=frame))
    monkeypatch.setattr(archive, "load_nuclear_result_bundle", lambda **kwargs: incumbent)
    def prepare_epoch(config, day):
        config["nuclear_experiment"].update(history_anchor_day="2024-09-09", raw_history_start_day="2024-09-09")
        return Path(config["nuclear_experiment"]["incremental_cache_dir"]), date(2024, 9, 9), date(2024, 9, 9)
    monkeypatch.setattr(incremental, "prepare_incremental_settings", prepare_epoch)
    def freeze_reporting(baseline, work, receipt):
        (work / "reference/reporting").mkdir(parents=True, exist_ok=True)
        return {}, []
    monkeypatch.setattr(r, "_freeze_reporting", freeze_reporting)
    monkeypatch.setattr(r, "_verify_reporting", lambda *args, **kwargs: (None, {}))
    return cfg, directory, original


def test_prepare_pins_isolated_sources_and_keeps_incumbent_byte_identical(scoped, baseline):
    cfg, directory, original = baseline
    source = bank(scoped, "2026-09-17", "2026-09-18")
    before = {p: r.sha256(p) for p in directory.rglob("*") if p.is_file()}
    work = r.prepare(cfg, "FR", source)
    actual = r.verify_snapshot(work, r.identity(cfg, "FR", source))
    manifest = r.read(work / "input_snapshot.json")
    assert work.is_relative_to(r.OUTPUT)
    assert actual["hourly"]["residual_correction"] == original["hourly"]["residual_correction"]
    assert actual["nuclear_experiment"]["filter_parameters"] == original["nuclear_experiment"]["filter_parameters"]
    assert actual["nuclear_experiment"]["history_anchor_day"] == original["nuclear_experiment"]["history_anchor_day"]
    snapshot_paths = [item["snapshot"] for item in manifest["files"]]
    assert len(snapshot_paths) == len(set(snapshot_paths))
    for alias in s.SERIES:
        spec = actual["zones"]["FR"]["covariates"][alias]
        assert spec["pit_file"] in snapshot_paths
        assert spec["unit"] == "EUR/MWh_e"
        assert spec["carbon_included"] is True
        assert spec["fill_method"] == "none"
        assert f"known_{alias}_oracle" in actual["hourly"]["feature_engineering"]["covariate_columns"]
    for path in actual["data"]["pit_files"].values():
        assert Path(path).is_relative_to(work / "snapshot")
    assert Path(actual["zones"]["FR"]["target"]["file"]).is_relative_to(work / "snapshot")
    assert {p: r.sha256(p) for p in before} == before
    assert r.prepare(cfg, "FR", source) == work
    assert not (scoped / "runs/live").exists()
    assert not (scoped / "runs/exports").exists()


@pytest.mark.parametrize("mutation", ["checksum", "configuration", "identity", "external_path", "duplicate", "empty_files", "missing_pinned_input"])
def test_snapshot_rejects_changed_or_unpinned_inputs(scoped, baseline, mutation):
    cfg, _, _ = baseline
    source = bank(scoped, "2026-09-17", "2026-09-18")
    work = r.prepare(cfg, "FR", source)
    path = work / "input_snapshot.json"
    manifest = r.read(path)
    expected = r.identity(cfg, "FR", source)
    if mutation == "checksum": manifest["files"][0]["sha256"] = "bad"
    elif mutation == "configuration": manifest["resolved_config_sha256"] = "bad"
    elif mutation == "identity": manifest["identity"] = {}
    elif mutation == "external_path": manifest["files"][0]["snapshot"] = str(source)
    elif mutation == "duplicate": manifest["files"].append(deepcopy(manifest["files"][0]))
    elif mutation == "empty_files": manifest["files"] = []
    elif mutation == "missing_pinned_input": manifest["files"] = [item for item in manifest["files"] if not item["snapshot"].endswith("ccc.parquet")]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        r.verify_snapshot(work, expected)


def test_full_run_workdir_key_is_short_fixed_and_not_old_correction_namespace(scoped, baseline):
    cfg, _, _ = baseline
    source = bank(scoped, "2026-09-17", "2026-09-18")
    work = r.workdir_for(cfg, "FR", source)
    assert len(work.name) == 16
    assert work.parent == r.OUTPUT / "2026-09-18/fr"
    assert not work.is_relative_to(r.residual_lab.NAMESPACE)
    assert work == r.workdir_for(deepcopy(cfg), "FR", source)


@pytest.mark.parametrize("source_name", ["run_result.json", "input_snapshot.json", "report_only/frozen_result/manifest.json"])
def test_new_baseline_or_reporting_evidence_creates_new_candidate_snapshot(scoped, baseline, source_name):
    cfg, directory, _ = baseline
    source = bank(scoped, "2026-09-17", "2026-09-18")
    before = r.workdir_for(cfg, "FR", source)
    path = directory / source_name
    record = r.read(path)
    record["changed_fixture_evidence"] = True
    path.write_text(json.dumps(record))
    assert r.workdir_for(cfg, "FR", source) != before


def test_finished_run_reuses_frozen_forecast_without_any_neural_execution(scoped, baseline, monkeypatch):
    import chronos2_hourly.nuclear_preparation as preparation
    import chronos2_hourly.nuclear_run_archive as archive
    import chronos2_modular.common as common
    from nyx_clean_fuel import standard_report
    cfg, _, _ = baseline
    source = bank(scoped, "2026-09-17", "2026-09-18")
    work = r.prepare(cfg, "FR", source)
    (work / "report_only/frozen_result").mkdir(parents=True)
    @dataclass
    class Data:
        target: object
    @contextmanager
    def progress(*args):
        yield lambda *args, **kwargs: None
    calls = []
    monkeypatch.setattr(r, "run_progress", progress)
    monkeypatch.setattr(common, "build_zone_configs", lambda *args: [SimpleNamespace(timezone="Europe/Paris")])
    monkeypatch.setattr(preparation, "prepare_nuclear_zone_data", lambda *args: Data(target=None))
    def forbidden(*args, **kwargs):
        pytest.fail("A completed full clean-fuel run attempted neural replay or replaced its frozen model result")
    monkeypatch.setattr(r, "run_forecast_with_storage_retry", forbidden)
    monkeypatch.setattr(archive, "save_nuclear_result_bundle", forbidden)
    monkeypatch.setattr(standard_report, "render_clean_fuel_reports", lambda *args, **kwargs: calls.append("report") or {})
    # Attribution cache behavior is separately exercised by the real
    # prepare_nuclear_attribution cache regression tests (not mocked there).
    for _ in range(2):
        result = r.run_zone(cfg, "FR", work, action="run", device="cpu", threads=1, workers=1, skip_attribution=True)
        assert result["status"] == "complete"
        assert result["result_reused"] is True
    assert calls == ["report", "report"]
