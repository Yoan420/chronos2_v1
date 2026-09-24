"""Launcher orchestration in temporary namespaces; no model/network calls."""
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

import run_solar_wind_forecast as runner
from chronos2_hourly import nuclear_incremental, nuclear_run_archive, nuclear_preparation, solar_wind_forecast, solar_wind_sources


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "OUTPUT", tmp_path/"runs/experiments/solar_wind_v1")
    monkeypatch.setattr(runner, "SOURCE", tmp_path/"data/pit/solar_wind_v1")
    monkeypatch.setattr(runner, "BASELINE", tmp_path/"runs/experiments/nuclear_forecast_v1")
    cfg = {"schema_version": 1, "delivery_day": "2026-09-22", "zones": ["DE"],
           "output_root": "runs/experiments/solar_wind_v1", "source_root": "data/pit/solar_wind_v1",
           "include_attribution": True, "wind_dst_policy": "duplicate", "wind_gap_policy": None,
           "diagnostic_only": True, "production_modified": False}
    path = tmp_path/"config/solar_wind.yaml"
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
    base = runner.baseline_for(cfg, "DE")
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
                       "runtime_as_of": "2026-09-21T08:00:00+02:00"},
              "zones": {"DE": {"timezone": "Europe/Berlin", "include_calendar": True,
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
                                          incremental_namespace=str(runner.OUTPUT/"_daily_cache/de/test"))
        return runner.OUTPUT/"_daily_cache/de/test", anchor, anchor
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
    for alias, series in solar_wind_forecast.GENERATION_SERIES.items():
        path = runner.SOURCE/(alias+".parquet")
        path.write_bytes(("isolated solar "+alias).encode())
        Path(str(path)+".audit.json").write_text(json.dumps({"series": series, "unit": "GW"}), encoding="utf8")
        sources[alias] = {"path": str(path), "sha256": runner.sha256(path), "series": series,
                          "audit_sha256": runner.sha256(Path(str(path)+".audit.json"))}
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
    ("zones", ["DE", "DE"]), ("zones", ["ES"]), ("zones", ["FR"]), ("zones", []),
    ("wind_dst_policy", "zero"), ("wind_gap_policy", "all_gaps"), ("delivery_day", "../2026-09-22")])
def test_invalid_configuration_rejected(sandbox, field, value):
    cfg, path = sandbox
    cfg[field] = value
    path.write_text(yaml.safe_dump(cfg), encoding="utf8")
    with pytest.raises((ValueError, TypeError)):
        runner.load_settings(path)


def test_audit_reads_existing_baseline_without_model_or_network(synthetic_baseline, monkeypatch):
    cfg, _, _, _, base, _ = synthetic_baseline
    monkeypatch.setattr(runner, "run_forecast_with_storage_retry", blocked)
    monkeypatch.setattr(solar_wind_sources, "ensure_solar_wind_sources", blocked)
    before = hashes(base)
    result = runner.audit_inputs(cfg)
    assert result["start_day"] == "2024-09-09" and result["end_day"] == "2026-09-22"
    assert not result["production_modified"] and before == hashes(base)


def test_prepare_adds_exact_six_inputs_preserves_recipe_and_original_bytes(synthetic_baseline):
    cfg, _, original, sources, base, _ = synthetic_baseline
    before, solar_before = hashes(base), hashes(runner.SOURCE)
    work = runner.prepare(cfg, "DE", sources)
    candidate = runner.verify_snapshot(work)
    assert before == hashes(base) and solar_before == hashes(runner.SOURCE)
    assert candidate["model"] == original["model"] and candidate["backtest"] == original["backtest"]
    assert candidate["hourly"]["residual_corrector"] == original["hourly"]["residual_corrector"]
    assert candidate["nuclear_experiment"]["filter_parameters"] == original["nuclear_experiment"]["filter_parameters"]
    assert candidate["nuclear_experiment"]["history_anchor_day"] == original["nuclear_experiment"]["history_anchor_day"]
    assert candidate["data"]["runtime_as_of"] == original["data"]["runtime_as_of"]
    old = set(original["zones"]["DE"]["covariates"])
    new = candidate["zones"]["DE"]["covariates"]
    assert set(new)-old == set(solar_wind_forecast.GENERATION_SERIES)
    assert set(candidate["data"]["pit_files"]) == old | set(solar_wind_forecast.GENERATION_SERIES)
    for alias, series in solar_wind_forecast.GENERATION_SERIES.items():
        assert new[alias]["series"] == series and new[alias]["unit"] == "GW"
        assert new[alias]["fill_method"] == "none" and new[alias]["include_base_context"] is True
        assert new[alias]["future"] == {"known_future": True, "strategies": ["oracle"]}
    for alias, prior in original["data"]["pit_files"].items():
        assert runner.sha256(Path(candidate["data"]["pit_files"][alias])) == runner.sha256(Path(prior))
    assert runner.prepare(cfg, "DE", sources) == work


def test_prepare_rejects_extra_or_missing_solar_channels_before_copy(synthetic_baseline):
    cfg, _, _, sources, _, _ = synthetic_baseline
    for changed in ({**sources, "wind": next(iter(sources.values()))}, dict(list(sources.items())[1:])):
        with pytest.raises(ValueError, match="four"):
            runner.prepare(cfg, "DE", changed)
    assert not runner.OUTPUT.exists()


@pytest.mark.parametrize("audit", [False, True])
def test_changed_audited_source_rejected_before_snapshot(synthetic_baseline, audit):
    cfg, _, _, sources, _, _ = synthetic_baseline
    source = Path(next(iter(sources.values()))["path"])
    if audit:
        source = Path(str(source)+".audit.json")
    source.write_bytes(b"changed after source verification")
    with pytest.raises(ValueError, match="audited source changed"):
        runner.prepare(cfg, "DE", sources)
    assert not runner.OUTPUT.exists()


def test_changed_scientific_code_rejects_frozen_reuse(synthetic_baseline, monkeypatch):
    cfg, _, _, sources, _, _ = synthetic_baseline
    work = runner.prepare(cfg, "DE", sources)
    monkeypatch.setattr(runner, "scientific_identity", lambda: {"files": {}, "dependencies": {}})
    with pytest.raises(ValueError, match="snapshot"):
        runner.verify_snapshot(work)


@pytest.mark.parametrize("corruption", ["input", "reference", "resolved", "files_empty", "references_empty", "missing_seal"])
def test_changed_or_incomplete_snapshot_fails_closed(synthetic_baseline, corruption):
    cfg, _, _, sources, _, _ = synthetic_baseline
    work = runner.prepare(cfg, "DE", sources)
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
        target = candidate["zones"]["DE"]["target"]["file"]
        manifest["files"] = [r for r in manifest["files"] if r["snapshot_path"] != target]
    runner.write_json(path, manifest)
    with pytest.raises((ValueError, OSError, KeyError)):
        runner.verify_snapshot(work)


def test_status_neither_fetches_prepares_fits_nor_writes(sandbox, monkeypatch, capsys):
    cfg, path = sandbox
    monkeypatch.setattr(runner, "audit_inputs", blocked)
    monkeypatch.setattr(runner, "prepare", blocked)
    monkeypatch.setattr(runner, "run_zone", blocked)
    monkeypatch.setattr(solar_wind_sources, "ensure_solar_wind_sources", blocked)
    before = hashes(runner.ROOT)
    assert runner.main(["--config", str(path), "--action", "status"]) == 0
    assert before == hashes(runner.ROOT) and not runner.OUTPUT.exists()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "not_prepared"


def test_audit_dispatch_remains_read_only(sandbox, monkeypatch, capsys):
    cfg, path = sandbox
    monkeypatch.setattr(runner, "audit_inputs", lambda cfg: {"production_modified": False})
    monkeypatch.setattr(runner, "exclusive_process_lock", blocked)
    monkeypatch.setattr(solar_wind_sources, "ensure_solar_wind_sources", blocked)
    before = hashes(runner.ROOT)
    assert runner.main(["--config", str(path), "--action", "audit"]) == 0
    assert hashes(runner.ROOT) == before
    assert json.loads(capsys.readouterr().out)["production_modified"] is False


@pytest.mark.parametrize("gap_policy", [None, "nl_ecmwf_spring_2025_2026"])
def test_sync_failure_is_visible_without_forecast(sandbox, monkeypatch, gap_policy):
    cfg, path = sandbox
    cfg["wind_gap_policy"] = gap_policy
    path.write_text(yaml.safe_dump(cfg), encoding="utf8")
    monkeypatch.setattr(runner, "audit_inputs", lambda cfg: {"start_day": "2024-09-09", "end_day": cfg["delivery_day"]})
    monkeypatch.setattr(runner, "prepare", blocked)
    monkeypatch.setattr(runner, "run_zone", blocked)
    def fail(**kwargs):
        assert kwargs["wind_dst_policy"] == "duplicate"
        assert kwargs["wind_gap_policy"] == gap_policy
        pointer = runner.OUTPUT/cfg["delivery_day"]/"latest_DE.json"
        assert runner.read(pointer)["status"] == "syncing_sources"
        raise ValueError("synthetic source gap")
    monkeypatch.setattr(solar_wind_sources, "ensure_solar_wind_sources", fail)
    with pytest.raises(ValueError, match="synthetic source gap"):
        runner.main(["--config", str(path), "--action", "run"])
    state = runner.read(runner.OUTPUT/cfg["delivery_day"]/"latest_DE.json")
    assert state["status"] == "failed" and state["stage"] == "sync_sources"
    assert state["workdirs"] == {} and state["production_modified"] is False


@pytest.mark.parametrize("field,value", [("zone", "NL"), ("delivery_day", "2026-09-21"), ("input_protocol", "civil_pit_v2")])
def test_snapshot_identity_mismatch_is_rejected(synthetic_baseline, field, value):
    cfg, _, _, sources, _, _ = synthetic_baseline
    work = runner.prepare(cfg, "DE", sources)
    manifest = runner.read(work/"input_snapshot.json")
    manifest["identity"][field] = value
    runner.write_json(work/"input_snapshot.json", manifest)
    with pytest.raises(ValueError, match="mismatch"):
        runner.verify_snapshot(work, zone="DE", delivery_day=cfg["delivery_day"])


@pytest.mark.parametrize("suffix", ["runs/exports/foreign", "runs/experiments/solar_wind_v1/2026-09-21/fr/old"])
def test_status_rejects_escaped_or_wrong_day_workdir_without_writing(sandbox, suffix):
    cfg, path = sandbox
    pointer = runner.OUTPUT/cfg["delivery_day"]/"latest_DE.json"
    runner.write_json(pointer, {"workdirs": {"DE": str(runner.ROOT/suffix)}})
    before = hashes(runner.ROOT)
    with pytest.raises(ValueError):
        runner.main(["--config", str(path), "--action", "status"])
    assert before == hashes(runner.ROOT)


@pytest.mark.parametrize("fail_second", [False, True])
def test_run_pointer_tracks_active_zone_and_preserves_completed_results(sandbox, monkeypatch, fail_second):
    cfg, path = sandbox
    cfg["zones"] = ["DE", "NL"]
    path.write_text(yaml.safe_dump(cfg), encoding="utf8")
    workdirs = {z: str(runner.OUTPUT/cfg["delivery_day"]/z.lower()/"test") for z in cfg["zones"]}
    pointer = runner.OUTPUT/cfg["delivery_day"]/"latest_DE_NL.json"
    runner.write_json(pointer, {"status": "complete", "workdirs": workdirs})
    for name in ("audit_inputs", "prepare", "run_forecast_with_storage_retry"):
        monkeypatch.setattr(runner, name, blocked)
    monkeypatch.setattr(solar_wind_sources, "ensure_solar_wind_sources", blocked)
    visited = []
    def report(settings, zone, directory, args):
        state = runner.read(pointer)
        assert state["status"] == "running" and state["active_zone"] == zone
        assert state["workdirs"] == workdirs and set(state["results"]) == set(visited)
        assert state["production_modified"] is False
        visited.append(zone)
        if fail_second and zone == "NL":
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
    assert visited == ["DE", "NL"] and state["production_modified"] is False
    assert "active_zone" not in state
    if fail_second:
        assert state["status"] == "failed" and state["failed_zone"] == "NL"
        assert state["error"] == "Synthetic rendering failure" and set(state["results"]) == {"DE"}
        assert not (pointer.parent/"solar_wind_DE_NL_index.html").exists()
    else:
        assert state["status"] == "complete" and set(state["results"]) == {"DE", "NL"}


def test_report_dispatch_skips_audit_sync_and_prepare(sandbox, monkeypatch):
    cfg, path = sandbox
    work = runner.OUTPUT/cfg["delivery_day"]/"de/test"
    pointer = runner.OUTPUT/cfg["delivery_day"]/"latest_DE.json"
    runner.write_json(pointer, {"workdirs": {"DE": str(work)}})
    for name in ("audit_inputs", "prepare", "run_forecast_with_storage_retry"):
        monkeypatch.setattr(runner, name, blocked)
    monkeypatch.setattr(solar_wind_sources, "ensure_solar_wind_sources", blocked)
    def report(settings, zone, directory, args):
        assert args.action == "report" and zone == "DE" and directory == work
        return {"reports": {}, "production_modified": False}
    monkeypatch.setattr(runner, "run_zone", report)
    assert runner.main(["--config", str(path), "--action", "report"]) == 0


def test_report_zone_loads_frozen_result_without_fit_save_or_attribution(synthetic_baseline, monkeypatch):
    from chronos2_modular import common
    from chronos2_hourly import nuclear_attribution, solar_wind_reporting
    cfg, _, _, sources, _, incumbent = synthetic_baseline
    work = runner.prepare(cfg, "DE", sources)
    (work/"report_only/frozen_result").mkdir(parents=True)
    @dataclass
    class Data:
        target: object
    data = Data(pd.Series(dtype=float))
    monkeypatch.setattr(common, "build_zone_configs", lambda *a, **kw: [SimpleNamespace(timezone="Europe/Berlin")])
    monkeypatch.setattr(nuclear_preparation, "prepare_nuclear_zone_data", lambda *a, **kw: data)
    monkeypatch.setattr(runner, "run_forecast_with_storage_retry", blocked)
    monkeypatch.setattr(solar_wind_forecast, "run_solar_wind_forecast", blocked)
    monkeypatch.setattr(nuclear_run_archive, "save_nuclear_result_bundle", blocked)
    monkeypatch.setattr(nuclear_attribution, "prepare_nuclear_attribution", blocked)
    monkeypatch.setattr(solar_wind_sources, "ensure_solar_wind_sources", blocked)
    @contextmanager
    def progress(*args):
        yield lambda *a, **kw: None
    monkeypatch.setattr(runner, "run_progress", progress)
    rendered = []
    def render(result, **kwargs):
        assert result is incumbent and kwargs["attribution_directory"] is None
        rendered.append(kwargs)
        return {}
    monkeypatch.setattr(solar_wind_reporting, "render_solar_wind_reports", render)
    args = SimpleNamespace(action="report", device="cpu", threads=1, workers=1, skip_attribution=False)
    result = runner.run_zone(cfg, "DE", work, args)
    assert rendered and result["result_reused"] is True
    assert result["production_modified"] is False and result["activation_performed"] is False
