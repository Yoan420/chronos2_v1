"""Launcher orchestration in temporary namespaces; no model/network calls."""
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

import run_solar_cwe_forecast as runner
from chronos2_hourly import nuclear_incremental, nuclear_run_archive, nuclear_preparation, solar_cwe_forecast, solar_cwe_sources


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "OUTPUT", tmp_path/"runs/experiments/solar_cwe_v1")
    monkeypatch.setattr(runner, "SOURCE", tmp_path/"data/pit/solar_cwe")
    monkeypatch.setattr(runner, "BASELINE", tmp_path/"runs/experiments/nuclear_forecast_v1")
    cfg = {"schema_version": 1, "delivery_day": "2026-09-19", "zones": ["FR"],
           "output_root": "runs/experiments/solar_cwe_v1", "source_root": "data/pit/solar_cwe",
           "include_attribution": True, "diagnostic_only": True, "production_modified": False}
    path = tmp_path/"config/solar_cwe.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(cfg), encoding="utf8")
    return cfg, path


def blocked(*args, **kwargs):
    pytest.fail("Unexpected fitting, fetching or source mutation")


def hashes(folder):
    return {p.relative_to(folder).as_posix(): runner.sha256(p) for p in folder.rglob("*") if p.is_file()}


@pytest.fixture
def synthetic_baseline(sandbox, monkeypatch):
    cfg, cfg_path = sandbox
    base = runner.baseline_for(cfg, "FR")
    source = base/"snapshot"
    source.mkdir(parents=True)
    residuals = [z+"_residual_load_fcst" for z in ("fr", "de", "be", "nl", "es")]
    aliases = residuals+["fr_nuclear_generation_fcst_gw"]
    files, paths = [], {}
    for alias in ["target"]+aliases:
        path = source/(alias+(".csv" if alias == "target" else ".parquet"))
        path.write_bytes(("frozen baseline "+alias).encode())
        paths[alias] = str(path)
        files.append({"snapshot": str(path), "sha256": runner.sha256(path)})
    bank_audit = source/(residuals[0]+".parquet.audit.json")
    bank_audit.write_text("{}", encoding="utf8")
    files.append({"snapshot": str(bank_audit), "sha256": runner.sha256(bank_audit)})
    config = {"model": {"model_id": "local/Chronos", "context_length": 2048, "local_files_only": True},
              "backtest": {"windows": 730}, "output": {"directory": str(base)}, "report": {"title": "Incumbent"},
              "hourly": {"feature_engineering": {"target_lags": [24, 48], "covariate_columns": ["known_fr_nuclear_generation_fcst_gw_oracle"]},
                         "residual_corrector": {"max_iterations": 70, "minimum_training_rows": 168}},
              "data": {"pit_files": {a: paths[a] for a in aliases}, "pit_vintage_dir": str(source), "cache_dir": str(base/"cache"),
                       "runtime_as_of": "2026-09-18T08:00:00+02:00"},
              "zones": {"FR": {"timezone": "Europe/Paris", "include_calendar": True,
                                 "target": {"file": paths["target"], "source": "file"},
                                 "covariates": {a: {"enabled": True, "source": "pit_parquet", "series": "test."+a,
                                                    "pit_file": paths[a], "fill_method": "none",
                                                    "future": {"known_future": True, "strategies": ["oracle"]}} for a in aliases}}},
              "nuclear_experiment": {"history_anchor_day": "2024-09-09", "raw_history_start_day": "2024-09-09",
                                     "incremental_namespace": str(base/"epoch"), "input_protocol": "civil_pit_v2",
                                     "mode": "incremental", "residual_bank_audit": str(bank_audit),
                                     "filter_parameters": {"q_over_r": .001, "candidate_kinds": ["linear_market"]}}}
    resolved = base/"resolved_config.yaml"
    resolved.write_text(yaml.safe_dump(config), encoding="utf8")
    runner.write_json(base/"input_snapshot.json", {"files": files, "resolved_config_sha256": runner.sha256(resolved)})
    runner.write_json(base/"run_result.json", {"status": "complete"})
    bundle = base/"report_only/frozen_result"
    bundle.mkdir(parents=True)
    runner.write_json(bundle/"manifest.json", {"status": "complete"})
    audit = {"history_anchor_day": "2024-09-09", "raw_history_start_day": "2024-09-09"}
    runner.write_json(bundle/"audits.json", {"result": audit})
    frame = pd.DataFrame({"value": [1., 2.]}, index=pd.date_range("2026-09-17", periods=2, tz="UTC", freq="h"))
    incumbent = SimpleNamespace(audit=audit, kalman_view=SimpleNamespace(backtest=frame, forecast=frame),
                                residual_statistics=frame, source_forecast=frame)
    monkeypatch.setattr(nuclear_run_archive, "load_nuclear_result_bundle", lambda **kw: incumbent)
    def incremental(cloned, day):
        anchor = pd.Timestamp("2024-09-09").date()
        cloned["nuclear_experiment"].update(history_anchor_day=str(anchor), raw_history_start_day=str(anchor),
                                          incremental_namespace=str(runner.OUTPUT/"_daily_cache/fr/test"))
        return runner.OUTPUT/"_daily_cache/fr/test", anchor, anchor
    monkeypatch.setattr(nuclear_incremental, "prepare_incremental_settings", incremental)
    def freeze(baseline, work, receipt):
        path = work/"reference/reporting/frozen.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf8")
        return {}, [{"snapshot_path": str(path), "sha256": runner.sha256(path)}]
    monkeypatch.setattr(runner, "_freeze_reporting", freeze)
    monkeypatch.setattr(runner, "_verify_reporting", lambda *a, **kw: (pd.Series(dtype=float), {}))
    sources = {}
    runner.SOURCE.mkdir(parents=True)
    for alias, series in solar_cwe_forecast.SOLAR_SERIES.items():
        path = runner.SOURCE/(alias+".parquet")
        path.write_bytes(("isolated solar "+alias).encode())
        Path(str(path)+".audit.json").write_text(json.dumps({"series": series, "unit": "GW"}), encoding="utf8")
        sources[alias] = {"path": str(path), "sha256": runner.sha256(path), "series": series}
    return cfg, cfg_path, config, sources, base, incumbent


def test_valid_config_and_confined_safe_paths(sandbox):
    cfg, path = sandbox
    assert runner.load_settings(path) == cfg
    assert runner.safe(runner.OUTPUT/"one/two") == runner.OUTPUT/"one/two"
    for invalid in (runner.OUTPUT, runner.ROOT/"runs/exports/a", runner.OUTPUT/"../escape"):
        with pytest.raises(ValueError):
            runner.safe(invalid)


@pytest.mark.parametrize("field,value", [("schema_version", True), ("output_root", "runs/exports"), ("source_root", "data/pit/kalman_weather"),
    ("production_modified", True), ("diagnostic_only", False), ("include_attribution", "true"),
    ("zones", ["FR", "FR"]), ("zones", ["ES"]), ("zones", []), ("delivery_day", "../2026-09-19")])
def test_invalid_configuration_rejected(sandbox, field, value):
    cfg, path = sandbox
    cfg[field] = value
    path.write_text(yaml.safe_dump(cfg), encoding="utf8")
    with pytest.raises((ValueError, TypeError)):
        runner.load_settings(path)


def test_audit_reads_existing_baseline_without_model_or_network(synthetic_baseline, monkeypatch):
    cfg, _, _, _, base, _ = synthetic_baseline
    monkeypatch.setattr(runner, "run_forecast_with_storage_retry", blocked)
    monkeypatch.setattr(solar_cwe_sources, "ensure_solar_sources", blocked)
    before = hashes(base)
    result = runner.audit_inputs(cfg)
    assert result["start_day"] == "2024-09-09" and result["end_day"] == "2026-09-19"
    assert not result["production_modified"] and before == hashes(base)


def test_prepare_adds_exact_four_inputs_preserves_recipe_and_original_bytes(synthetic_baseline):
    cfg, _, original, sources, base, _ = synthetic_baseline
    before, solar_before = hashes(base), hashes(runner.SOURCE)
    work = runner.prepare(cfg, "FR", sources)
    candidate = runner.verify_snapshot(work)
    assert before == hashes(base) and solar_before == hashes(runner.SOURCE)
    assert candidate["model"] == original["model"] and candidate["backtest"] == original["backtest"]
    assert candidate["hourly"]["residual_corrector"] == original["hourly"]["residual_corrector"]
    assert candidate["nuclear_experiment"]["filter_parameters"] == original["nuclear_experiment"]["filter_parameters"]
    assert candidate["nuclear_experiment"]["history_anchor_day"] == original["nuclear_experiment"]["history_anchor_day"]
    assert candidate["data"]["runtime_as_of"] == original["data"]["runtime_as_of"]
    old = set(original["zones"]["FR"]["covariates"])
    new = candidate["zones"]["FR"]["covariates"]
    assert set(new)-old == set(solar_cwe_forecast.SOLAR_SERIES)
    assert set(candidate["data"]["pit_files"]) == old | set(solar_cwe_forecast.SOLAR_SERIES)
    for alias, series in solar_cwe_forecast.SOLAR_SERIES.items():
        assert new[alias]["series"] == series and new[alias]["unit"] == "GW"
        assert new[alias]["fill_method"] == "none" and new[alias]["include_base_context"] is True
        assert new[alias]["future"] == {"known_future": True, "strategies": ["oracle"]}
    for alias, prior in original["data"]["pit_files"].items():
        assert runner.sha256(Path(candidate["data"]["pit_files"][alias])) == runner.sha256(Path(prior))
    assert runner.prepare(cfg, "FR", sources) == work


def test_prepare_rejects_extra_or_missing_solar_channels_before_copy(synthetic_baseline):
    cfg, _, _, sources, _, _ = synthetic_baseline
    for changed in ({**sources, "wind": next(iter(sources.values()))}, dict(list(sources.items())[1:])):
        with pytest.raises(ValueError, match="four"):
            runner.prepare(cfg, "FR", changed)
    assert not runner.OUTPUT.exists()


@pytest.mark.parametrize("corruption", ["input", "reference", "resolved", "files_empty", "references_empty", "missing_seal"])
def test_changed_or_incomplete_snapshot_fails_closed(synthetic_baseline, corruption):
    cfg, _, _, sources, _, _ = synthetic_baseline
    work = runner.prepare(cfg, "FR", sources)
    path = work/"input_snapshot.json"
    manifest = runner.read(path)
    if corruption in ("input", "reference"):
        field = "files" if corruption == "input" else "reference_files"
        Path(manifest[field][0]["snapshot_path"]).write_bytes(b"corrupted")
    elif corruption == "resolved":
        (work/"resolved_config.yaml").write_text("model: {}", encoding="utf8")
    elif corruption == "files_empty":
        manifest["files"] = []
    elif corruption == "references_empty":
        manifest["reference_files"] = []
    else:
        candidate = yaml.safe_load((work/"resolved_config.yaml").read_text())
        target = candidate["zones"]["FR"]["target"]["file"]
        manifest["files"] = [r for r in manifest["files"] if r["snapshot_path"] != target]
    runner.write_json(path, manifest)
    with pytest.raises((ValueError, OSError, KeyError)):
        runner.verify_snapshot(work)


def test_status_neither_fetches_prepares_fits_nor_writes(sandbox, monkeypatch, capsys):
    cfg, path = sandbox
    monkeypatch.setattr(runner, "audit_inputs", blocked)
    monkeypatch.setattr(runner, "prepare", blocked)
    monkeypatch.setattr(runner, "run_zone", blocked)
    monkeypatch.setattr(solar_cwe_sources, "ensure_solar_sources", blocked)
    before = hashes(runner.ROOT)
    assert runner.main(["--config", str(path), "--action", "status"]) == 0
    assert before == hashes(runner.ROOT) and not runner.OUTPUT.exists()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "not_prepared"


@pytest.mark.parametrize("field,value", [("zone", "DE"), ("delivery_day", "2026-09-18"), ("input_protocol", "civil_pit_v2")])
def test_snapshot_identity_mismatch_is_rejected(synthetic_baseline, field, value):
    cfg, _, _, sources, _, _ = synthetic_baseline
    work = runner.prepare(cfg, "FR", sources)
    manifest = runner.read(work/"input_snapshot.json")
    manifest["identity"][field] = value
    runner.write_json(work/"input_snapshot.json", manifest)
    with pytest.raises(ValueError, match="mismatch"):
        runner.verify_snapshot(work, zone="FR", delivery_day=cfg["delivery_day"])


@pytest.mark.parametrize("suffix", ["runs/exports/foreign", "runs/experiments/solar_cwe_v1/2026-09-18/fr/old"])
def test_status_rejects_escaped_or_wrong_day_workdir_without_writing(sandbox, suffix):
    cfg, path = sandbox
    pointer = runner.OUTPUT/cfg["delivery_day"]/"latest_FR.json"
    runner.write_json(pointer, {"workdirs": {"FR": str(runner.ROOT/suffix)}})
    before = hashes(runner.ROOT)
    with pytest.raises(ValueError):
        runner.main(["--config", str(path), "--action", "status"])
    assert before == hashes(runner.ROOT)


@pytest.mark.parametrize("fail_second", [False, True])
def test_run_pointer_tracks_active_zone_and_preserves_completed_results(sandbox, monkeypatch, fail_second):
    cfg, path = sandbox
    cfg["zones"] = ["FR", "DE"]
    path.write_text(yaml.safe_dump(cfg), encoding="utf8")
    workdirs = {z: str(runner.OUTPUT/cfg["delivery_day"]/z.lower()/"test") for z in cfg["zones"]}
    pointer = runner.OUTPUT/cfg["delivery_day"]/"latest_FR_DE.json"
    runner.write_json(pointer, {"status": "complete", "workdirs": workdirs})
    for name in ("audit_inputs", "prepare", "run_forecast_with_storage_retry"):
        monkeypatch.setattr(runner, name, blocked)
    monkeypatch.setattr(solar_cwe_sources, "ensure_solar_sources", blocked)
    visited = []
    def report(settings, zone, directory, args):
        state = runner.read(pointer)
        assert state["status"] == "running" and state["active_zone"] == zone
        assert state["workdirs"] == workdirs and set(state["results"]) == set(visited)
        assert state["production_modified"] is False
        visited.append(zone)
        if fail_second and zone == "DE":
            raise RuntimeError("Synthetic rendering failure")
        return {"reports": {}, "production_modified": False, "zone": zone}
    monkeypatch.setattr(runner, "run_zone", report)
    args = ["--config", str(path), "--action", "report"]
    if fail_second:
        with pytest.raises(RuntimeError, match="Synthetic rendering failure"):
            runner.main(args)
    else:
        assert runner.main(args) == 0
    state = runner.read(pointer)
    assert visited == ["FR", "DE"] and state["production_modified"] is False
    assert "active_zone" not in state
    if fail_second:
        assert state["status"] == "failed" and state["failed_zone"] == "DE"
        assert state["error"] == "Synthetic rendering failure" and set(state["results"]) == {"FR"}
        assert not (pointer.parent/"solar_cwe_FR_DE_index.html").exists()
    else:
        assert state["status"] == "complete" and set(state["results"]) == {"FR", "DE"}


def test_report_dispatch_skips_audit_sync_and_prepare(sandbox, monkeypatch):
    cfg, path = sandbox
    work = runner.OUTPUT/cfg["delivery_day"]/"fr/test"
    pointer = runner.OUTPUT/cfg["delivery_day"]/"latest_FR.json"
    runner.write_json(pointer, {"workdirs": {"FR": str(work)}})
    for name in ("audit_inputs", "prepare", "run_forecast_with_storage_retry"):
        monkeypatch.setattr(runner, name, blocked)
    monkeypatch.setattr(solar_cwe_sources, "ensure_solar_sources", blocked)
    def report(settings, zone, directory, args):
        assert args.action == "report" and zone == "FR" and directory == work
        return {"reports": {}, "production_modified": False}
    monkeypatch.setattr(runner, "run_zone", report)
    assert runner.main(["--config", str(path), "--action", "report"]) == 0


def test_report_zone_loads_frozen_result_without_fit_save_or_attribution(synthetic_baseline, monkeypatch):
    from chronos2_modular import common
    from chronos2_hourly import nuclear_attribution, solar_cwe_reporting
    cfg, _, _, sources, _, incumbent = synthetic_baseline
    work = runner.prepare(cfg, "FR", sources)
    (work/"report_only/frozen_result").mkdir(parents=True)
    @dataclass
    class Data:
        target: object
    data = Data(pd.Series(dtype=float))
    monkeypatch.setattr(common, "build_zone_configs", lambda *a, **kw: [SimpleNamespace(timezone="Europe/Paris")])
    monkeypatch.setattr(nuclear_preparation, "prepare_nuclear_zone_data", lambda *a, **kw: data)
    monkeypatch.setattr(runner, "run_forecast_with_storage_retry", blocked)
    monkeypatch.setattr(solar_cwe_forecast, "run_solar_cwe_forecast", blocked)
    monkeypatch.setattr(nuclear_run_archive, "save_nuclear_result_bundle", blocked)
    monkeypatch.setattr(nuclear_attribution, "prepare_nuclear_attribution", blocked)
    monkeypatch.setattr(solar_cwe_sources, "ensure_solar_sources", blocked)
    @contextmanager
    def progress(*args):
        yield lambda *a, **kw: None
    monkeypatch.setattr(runner, "run_progress", progress)
    rendered = []
    def render(result, **kwargs):
        assert result is incumbent and kwargs["attribution_directory"] is None
        rendered.append(kwargs)
        return {}
    monkeypatch.setattr(solar_cwe_reporting, "render_solar_cwe_reports", render)
    args = SimpleNamespace(action="report", device="cpu", threads=1, workers=1, skip_attribution=False)
    result = runner.run_zone(cfg, "FR", work, args)
    assert rendered and result["result_reused"] is True
    assert result["production_modified"] is False and result["activation_performed"] is False
