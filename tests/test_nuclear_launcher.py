from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

import run_nuclear_forecast as runner
from chronos2_hourly.nuclear_sources import NUCLEAR_ALIAS, NUCLEAR_SERIES


def _settings(tmp_path: Path, **overrides) -> tuple[Path, dict]:
    project = tmp_path / "project with spaces"
    config = project / "config" / "nuclear.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "schema_version": 1, "project_root": "..",
        "output_root": "runs/experiments/nuclear_test",
        "nuclear_store": "data/pit/nuclear_forecast/nuclear.parquet",
        "lora_activation_config": "config/activation.yaml",
        "kalman_config": "config/kalman.yaml",
        "incomplete_dst_policy": "duplicate", "sync_chunk_days": 2,
        "zone_configs": {"FR": "base.yaml"},
    }
    raw.update(overrides)
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    (config.parent / "kalman.yaml").write_text(
        yaml.safe_dump({"training_lookback_days": 365, "filter_parameters": {"observation_variance": 9.0}}),
        encoding="utf-8",
    )
    return config, raw


def _audit(complete: bool = True, missing=()) -> dict:
    return {"complete": complete, "missing_days": list(missing),
            "blockers": [] if complete else ["Missing source hours"]}


def test_settings_resolve_to_isolated_canonical_paths(tmp_path: Path) -> None:
    config, _ = _settings(tmp_path)
    before = config.read_bytes()
    settings = runner.load_settings(config)
    project = config.parent.parent.resolve()
    assert settings["project_root"] == project
    assert settings["output_root"] == project / "runs/experiments/nuclear_test"
    assert settings["nuclear_store"] == project / "data/pit/nuclear_forecast/nuclear.parquet"
    assert config.read_bytes() == before
    assert not settings["output_root"].exists()
    assert settings["computation_mode"] == "incremental"


@pytest.mark.parametrize("field,value", [
    ("output_root", "runs/experiments/../operational"),
    ("output_root", "runs/experiments"),
    ("output_root", "runs/experiments_sibling/nuclear"),
    ("nuclear_store", "data/pit/vintages/fr_nuclear_generation_fcst_long.parquet"),
    ("nuclear_store", "data/pit/nuclear_forecast/../kalman_weather/fr.parquet"),
    ("incomplete_dst_policy", "ffill"), ("sync_chunk_days", 32),
    ("computation_mode", "unexpected"),
])
def test_settings_reject_path_escape_and_invalid_contracts(tmp_path: Path, field: str, value) -> None:
    config, _ = _settings(tmp_path, **{field: value})
    with pytest.raises(ValueError):
        runner.load_settings(config)


def test_sync_chunks_new_source_and_merges_only_isolated_artifact(tmp_path: Path, monkeypatch) -> None:
    config, _ = _settings(tmp_path)
    settings = runner.load_settings(config)
    path = settings["nuclear_store"]
    monkeypatch.setattr(runner, "source_bounds", lambda day: ("2026-01-01", "2026-01-03"))
    audits = iter([_audit(False, ["2026-01-01", "2026-01-02", "2026-01-03"]), _audit()])
    monkeypatch.setattr(runner, "audit_nuclear_store", lambda *args: next(audits))
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        path.write_bytes(b"isolated materializer output")

    monkeypatch.setattr(runner.subprocess, "run", execute)
    assert runner.sync_source(settings, pd.Timestamp("2026-01-03"), 80)["complete"]
    assert len(calls) == 2
    first, second = calls[0][0], calls[1][0]
    assert "--merge-existing" not in first
    assert "--merge-existing" in second
    for command, kwargs in calls:
        assert command[command.index("--series") + 1] == NUCLEAR_SERIES
        assert command[command.index("--output") + 1] == str(path)
        assert command[command.index("--workers") + 1] == "32"
        assert command[command.index("--naive-timezone") + 1] == "Europe/Paris"
        assert kwargs == {"check": True, "cwd": runner.ROOT}
    assert first[first.index("--end-day") + 1] == "2026-01-02"
    assert second[second.index("--start-day") + 1] == "2026-01-03"
    assert not path.with_suffix(".sync.lock").exists()


def _existing_source(settings: dict) -> Path:
    path = settings["nuclear_store"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"verified existing artifact")
    path.with_name(path.name + ".audit.json").write_text(
        json.dumps({"start_day": "2026-01-01", "end_day": "2026-01-03"}), encoding="utf-8",
    )
    return path


def test_sync_existing_source_fetches_only_missing_day(tmp_path: Path, monkeypatch) -> None:
    config, _ = _settings(tmp_path)
    settings = runner.load_settings(config)
    path = _existing_source(settings)
    monkeypatch.setattr(runner, "source_bounds", lambda day: ("2026-01-01", "2026-01-04"))
    audits = iter([_audit(), _audit(False, ["2026-01-04"]), _audit()])
    monkeypatch.setattr(runner, "audit_nuclear_store", lambda *args: next(audits))
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: calls.append(command))
    runner.sync_source(settings, pd.Timestamp("2026-01-04"), 2)
    assert len(calls) == 1
    assert calls[0][calls[0].index("--start-day") + 1] == "2026-01-04"
    assert calls[0][calls[0].index("--end-day") + 1] == "2026-01-04"
    assert "--merge-existing" in calls[0]
    assert path.read_bytes() == b"verified existing artifact"


def test_sync_rejects_invalid_prior_source_without_subprocess(tmp_path: Path, monkeypatch) -> None:
    config, _ = _settings(tmp_path)
    settings = runner.load_settings(config)
    path = _existing_source(settings)
    monkeypatch.setattr(runner, "audit_nuclear_store", lambda *args: _audit(False))
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("No invalid source merge permitted"))
    with pytest.raises(ValueError, match="non valide"):
        runner.sync_source(settings, pd.Timestamp("2026-01-04"), 2)
    assert path.read_bytes() == b"verified existing artifact"
    assert not path.with_suffix(".sync.lock").exists()


def test_sync_subprocess_failure_propagates_and_releases_own_lock(tmp_path: Path, monkeypatch) -> None:
    config, _ = _settings(tmp_path)
    settings = runner.load_settings(config)
    path = settings["nuclear_store"]
    monkeypatch.setattr(runner, "source_bounds", lambda day: ("2026-01-01", "2026-01-01"))
    monkeypatch.setattr(runner, "audit_nuclear_store", lambda *args: _audit(False, ["2026-01-01"]))

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(3, command)

    monkeypatch.setattr(runner.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        runner.sync_source(settings, pd.Timestamp("2026-01-01"), 1)
    assert not path.with_suffix(".sync.lock").exists()
    assert not path.exists()


def test_sync_never_removes_another_workers_lock(tmp_path: Path) -> None:
    config, _ = _settings(tmp_path)
    settings = runner.load_settings(config)
    lock = settings["nuclear_store"].with_suffix(".sync.lock")
    lock.parent.mkdir(parents=True)
    lock.write_text("other worker", encoding="utf-8")
    with pytest.raises(ValueError, match="Verrou"):
        runner.sync_source(settings, pd.Timestamp("2026-01-01"), 1)
    assert lock.read_text(encoding="utf-8") == "other worker"


def _snapshot_fixture(tmp_path: Path, monkeypatch):
    config_path, _ = _settings(tmp_path)
    settings = runner.load_settings(config_path)
    source = settings["project_root"] / "base.yaml"
    config = {
        "model": {"model_id": "amazon/chronos-2", "local_files_only": False},
        "data": {"source": "auto", "cache_dir": "data/cache", "runtime_as_of": "original"},
        "backtest": {"windows": 10}, "output": {"directory": "runs/live"},
        "hourly": {"feature_engineering": {"covariate_columns": ["known_load_oracle"]}},
        "zones": {
            "FR": {"target": {"series": "price.fr"}, "covariates": {
                "load": {"enabled": True, "source": "pit_parquet", "series": "load.fr"},
                "disabled": {"enabled": False, "source": "auto"},
            }},
            "DE": {"target": {"series": "price.de"}, "covariates": {}},
        },
    }
    source.write_text(yaml.safe_dump(config), encoding="utf-8")
    live = settings["project_root"] / "live_inputs"
    live.mkdir()
    inputs = {"target": live / "target.csv.gz", "load": live / "load.parquet", NUCLEAR_ALIAS: settings["nuclear_store"]}
    for alias, path in inputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((alias + " original bytes").encode())
    audit_source = settings["nuclear_store"].with_name(settings["nuclear_store"].name + ".audit.json")
    audit_source.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "zone_inputs", lambda *args: (config, source, inputs))
    monkeypatch.setattr(runner, "audit_nuclear_store", lambda *args: _audit())
    monkeypatch.setattr(runner, "resolve_local_model_revision", lambda *args: "pinned-local-commit")
    # This fixture isolates orchestration using opaque copied bytes; the real
    # target-cache parser has its own CSV/parquet data-contract tests.
    monkeypatch.setattr(runner, "validate_target_cache", lambda path: None)
    return settings, config, source, inputs


def test_snapshot_config_pins_inputs_without_live_mutations(tmp_path: Path, monkeypatch) -> None:
    settings, source_config, source, inputs = _snapshot_fixture(tmp_path, monkeypatch)
    validated_targets = []
    monkeypatch.setattr(runner, "validate_target_cache", lambda path: validated_targets.append(path))
    original = deepcopy(source_config)
    original_bytes = {path: path.read_bytes() for path in [source, *inputs.values()]}
    workdir = settings["output_root"] / "2026-01-04" / "fr"
    result = runner.snapshot_config(settings, "FR", pd.Timestamp("2026-01-04"), workdir)
    assert source_config == original
    assert {path: path.read_bytes() for path in original_bytes} == original_bytes
    assert list(result["zones"]) == ["FR"]
    assert result["model"]["local_files_only"] is True
    assert result["model"]["revision"] == "pinned-local-commit"
    assert result["nuclear_experiment"]["filter_parameters"] == {"observation_variance": 9.0}
    assert result["nuclear_experiment"]["mode"] == "incremental"
    assert Path(result["nuclear_experiment"]["incremental_cache_dir"]) == settings["output_root"] / "_daily_cache/fr/civil_pit_v2"
    assert result["data"]["source"] == "cache"
    assert result["output"]["directory"] == str(workdir)
    assert result["backtest"]["windows"] == 730
    target = Path(result["zones"]["FR"]["target"]["file"])
    assert workdir in target.parents
    assert validated_targets == [target]
    covariates = result["zones"]["FR"]["covariates"]
    assert set(covariates) == {"load", NUCLEAR_ALIAS}
    assert covariates[NUCLEAR_ALIAS]["series"] == NUCLEAR_SERIES
    assert covariates[NUCLEAR_ALIAS]["fill_method"] == "none"
    assert covariates[NUCLEAR_ALIAS]["future"] == {"known_future": True, "strategies": ["oracle"]}
    assert all(workdir in Path(item["pit_file"]).parents for item in covariates.values())
    assert f"known_{NUCLEAR_ALIAS}_oracle" in result["hourly"]["feature_engineering"]["covariate_columns"]


@pytest.mark.parametrize("day", ["2025-03-31", "2025-10-27"])
def test_snapshot_runtime_cutoff_remains_0800_after_dst(tmp_path: Path, monkeypatch, day: str) -> None:
    settings, _, _, _ = _snapshot_fixture(tmp_path, monkeypatch)
    result = runner.snapshot_config(settings, "FR", pd.Timestamp(day), settings["output_root"] / day / "fr")
    cutoff = pd.Timestamp(result["data"]["runtime_as_of"]).tz_convert("Europe/Paris")
    assert cutoff.hour == 8
    assert cutoff.date() == (pd.Timestamp(day) - pd.Timedelta(days=1)).date()


@pytest.mark.parametrize("day,now,allowed", [
    ("2025-03-31", "2025-03-30T08:30:00+02:00", True),
    ("2025-10-27", "2025-10-26T07:30:00+01:00", False),
])
def test_delivery_date_rejects_only_before_civil_cutoff(monkeypatch, day: str, now: str, allowed: bool) -> None:
    def timestamp(*args, **kwargs):
        return pd.Timestamp(*args, **kwargs)

    timestamp.now = lambda tz: pd.Timestamp(now).tz_convert(tz)
    monkeypatch.setattr(runner, "pd", SimpleNamespace(Timestamp=timestamp, Timedelta=pd.Timedelta, isna=pd.isna))
    if allowed:
        assert runner.delivery_date(day) == pd.Timestamp(day)
    else:
        with pytest.raises(ValueError, match="cutoff"):
            runner.delivery_date(day)


def test_resume_uses_pinned_bytes_after_live_data_changes(tmp_path: Path, monkeypatch) -> None:
    settings, _, _, inputs = _snapshot_fixture(tmp_path, monkeypatch)
    day = pd.Timestamp("2026-01-04")
    workdir = settings["output_root"] / "2026-01-04" / "fr"
    original = runner.snapshot_config(settings, "FR", day, workdir)
    manifest = (workdir / "input_snapshot.json").read_bytes()
    pinned = Path(original["data"]["pit_files"][NUCLEAR_ALIAS])
    before = pinned.read_bytes()
    for source in inputs.values():
        source.write_bytes(b"new live values")
    monkeypatch.setattr(runner, "audit_nuclear_store", lambda *args: pytest.fail("Do not rebuild an existing snapshot"))
    assert runner.snapshot_config(settings, "FR", day, workdir) == original
    assert (workdir / "input_snapshot.json").read_bytes() == manifest
    assert pinned.read_bytes() == before


@pytest.mark.parametrize("tamper", ["input", "resolved", "base_config"])
def test_resume_rejects_pinned_input_or_configuration_tampering(tmp_path: Path, monkeypatch, tamper: str) -> None:
    settings, _, source, _ = _snapshot_fixture(tmp_path, monkeypatch)
    day = pd.Timestamp("2026-01-04")
    workdir = settings["output_root"] / "2026-01-04" / "fr"
    config = runner.snapshot_config(settings, "FR", day, workdir)
    changed = {"input": Path(config["data"]["pit_files"][NUCLEAR_ALIAS]),
               "resolved": workdir / "resolved_config.yaml", "base_config": source}[tamper]
    changed.write_bytes(b"tampered")
    with pytest.raises(ValueError):
        runner.snapshot_config(settings, "FR", day, workdir)


def test_resume_rejects_changed_filter_parameters(tmp_path: Path, monkeypatch) -> None:
    settings, _, _, _ = _snapshot_fixture(tmp_path, monkeypatch)
    day = pd.Timestamp("2026-01-04")
    workdir = settings["output_root"] / "2026-01-04" / "fr"
    runner.snapshot_config(settings, "FR", day, workdir)
    settings["kalman_config"].write_text(yaml.safe_dump({
        "training_lookback_days": 365, "filter_parameters": {"observation_variance": 10.0},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration a change"):
        runner.snapshot_config(settings, "FR", day, workdir)


def test_snapshot_mode_is_sealed_and_legacy_mode_remains_full(tmp_path: Path, monkeypatch) -> None:
    settings, _, _, _ = _snapshot_fixture(tmp_path, monkeypatch)
    day = pd.Timestamp("2026-01-04")
    workdir = settings["output_root"] / "2026-01-04" / "fr"
    result = runner.snapshot_config(settings, "FR", day, workdir, computation_mode="full")
    assert result["nuclear_experiment"]["mode"] == "full"
    before = (workdir / "input_snapshot.json").read_bytes(), (workdir / "resolved_config.yaml").read_bytes()
    assert runner.snapshot_config(settings, "FR", day, workdir) == result
    assert ((workdir / "input_snapshot.json").read_bytes(), (workdir / "resolved_config.yaml").read_bytes()) == before
    with pytest.raises(ValueError, match="configuration a change"):
        runner.snapshot_config(settings, "FR", day, workdir, computation_mode="incremental")
    # Construct the exact earlier archive contract, predating the mode field.
    manifest_path, resolved_path = workdir / "input_snapshot.json", workdir / "resolved_config.yaml"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"].pop("computation_mode")
    result["nuclear_experiment"].pop("mode")
    result["nuclear_experiment"].pop("incremental_cache_dir")
    resolved_path.write_text(yaml.safe_dump(result), encoding="utf-8")
    manifest["resolved_config_sha256"] = runner.sha256(resolved_path)
    runner.write_json(manifest_path, manifest)
    before = manifest_path.read_bytes(), resolved_path.read_bytes()
    resumed = runner.snapshot_config(settings, "FR", day, workdir)
    assert "mode" not in resumed["nuclear_experiment"]
    assert (manifest_path.read_bytes(), resolved_path.read_bytes()) == before
    with pytest.raises(ValueError, match="configuration a change"):
        runner.snapshot_config(settings, "FR", day, workdir, computation_mode="incremental")


def test_local_model_revision_resolution_forbids_network(tmp_path: Path, monkeypatch) -> None:
    import huggingface_hub

    pinned = tmp_path / "models--amazon--chronos-2" / "snapshots" / ("a" * 40) / "config.json"
    calls = []

    def local_download(**kwargs):
        calls.append(kwargs)
        return str(pinned)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", local_download)
    revision = runner.resolve_local_model_revision({"model": {"model_id": "amazon/chronos-2", "revision": "main"}})
    assert revision == "a" * 40
    assert calls == [{"repo_id": "amazon/chronos-2", "filename": "config.json", "revision": "main", "local_files_only": True}]
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(tmp_path / "config.json"))
    with pytest.raises(ValueError, match="commit"):
        runner.resolve_local_model_revision({"model": {"model_id": "amazon/chronos-2"}})


@pytest.mark.parametrize("mode", ["autonomous", "kalman"])
def test_lora_activation_rejected_before_source_work(tmp_path: Path, monkeypatch, mode: str) -> None:
    config, _ = _settings(tmp_path)
    settings = runner.load_settings(config)
    settings["lora_activation_config"].write_text(yaml.safe_dump({"zones": {"FR": {"enabled_modes": [mode]}}}), encoding="utf-8")
    monkeypatch.setattr(runner, "load_settings", lambda *args: settings)
    monkeypatch.setattr(runner, "delivery_date", lambda *args: pd.Timestamp("2026-01-04"))
    monkeypatch.setattr(runner, "sync_source", lambda *args: pytest.fail("LoRA preflight must precede source writes"))
    with pytest.raises(ValueError, match="LoRA"):
        runner.main(["--config", str(config), "--stage", "sync", "--zones", "FR"])


def _ps(arguments: str) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    if executable is None:
        pytest.skip("Windows PowerShell unavailable")
    launcher = str(runner.ROOT / "Forecast.ps1").replace("'", "''")
    return subprocess.run(
        [executable, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", f"& '{launcher}' {arguments} -DryRun"],
        cwd=runner.ROOT, check=False, capture_output=True, text=True, encoding="utf-8",
    )


def _ps_argv(arguments: str) -> list[str]:
    completed = _ps(arguments)
    assert completed.returncode == 0, completed.stderr
    prefix = "Commande (argv, shell=False): "
    line = next(line for line in completed.stdout.splitlines() if line.startswith(prefix))
    return json.loads(line[len(prefix):])


def test_powershell_nuclear_opt_in_builds_separate_argv() -> None:
    argv = _ps_argv("-Action Run -Mode Both -WithNuclear -NuclearStage Audit -Countries NL,FR -DeliveryDay 2026-01-04 -Device cpu -Threads 3 -Workers 2")
    assert argv[1] == str(runner.ROOT / "run_nuclear_forecast.py")
    assert argv[argv.index("--stage") + 1].lower() == "audit"
    assert argv[argv.index("--zones") + 1:argv.index("--device")] == ["NL", "FR"]
    assert argv[argv.index("--config") + 1] == str(runner.ROOT / "config/nuclear_forecast.yaml")
    assert "--lora-activation-config" not in argv
    assert "--kalman-config" not in argv
    assert "--allow-model-download" not in argv


def test_powershell_regular_both_is_unchanged() -> None:
    argv = _ps_argv("-Action Run -Mode Both -Countries FR -DeliveryDay 2026-01-04 -Device cpu -Threads 3 -Workers 2")
    assert argv[1:] == [str(runner.ROOT / "run_multicountry_forecast.py"), "--zones", "FR", "--mode", "Both",
                        "--device", "cpu", "--threads", "3", "--workers", "2", "--delivery-day", "2026-01-04"]


def test_corrected_protocol_preserves_legacy_failed_snapshot(tmp_path: Path, monkeypatch, capsys) -> None:
    import chronos2_hourly.nuclear_preparation as preparation
    import chronos2_hourly.nuclear_residual_inputs as residual

    settings, _, _, _ = _snapshot_fixture(tmp_path, monkeypatch)
    day = pd.Timestamp("2026-01-04")
    old = settings["output_root"] / str(day.date()) / "fr"
    old.mkdir(parents=True)
    old_manifest = old / "input_snapshot.json"
    old_manifest.write_text("legacy sealed evidence; never rewrite", encoding="utf-8")
    original = old_manifest.read_bytes()
    monkeypatch.setattr(runner, "load_settings", lambda path: settings)
    monkeypatch.setattr(runner, "delivery_date", lambda value: day)
    monkeypatch.setattr(runner, "check_lora_inactive", lambda *args: None)
    ensured, prepared = [], []
    monkeypatch.setattr(runner, "ensure_residual_inputs", lambda *args: ensured.append(args))
    monkeypatch.setattr(residual, "audit_residual_bank", lambda *args: _audit())
    monkeypatch.setattr(preparation, "prepare_nuclear_zone_data", lambda *args: prepared.append(args))
    assert runner.main(["--stage", "prepare", "--zones", "FR", "--delivery-day", str(day.date())]) == 0
    corrected = old / runner.INPUT_PROTOCOL
    assert len(ensured) == len(prepared) == 1
    assert prepared[0][2] == corrected
    assert (corrected / "input_snapshot.json").is_file()
    assert old_manifest.read_bytes() == original
    assert not (old / "reports").exists()
    assert not (corrected / "run.lock").exists()
    assert '"forecast_started": false' in capsys.readouterr().out


def test_real_zone_definitions_select_verified_wide_bank_without_yaml_mutation() -> None:
    settings = runner.load_settings(runner.ROOT / "config/nuclear_forecast.yaml")
    for zone in ("FR", "DE", "BE", "NL", "ES"):
        path = runner.ROOT / settings["zone_configs"][zone]
        before = path.read_bytes()
        config, _, inputs = runner.zone_inputs(settings, zone)
        assert all(inputs[alias] == settings["residual_bank"] for alias in runner.RESIDUAL_ALIASES)
        assert config["zones"][zone]["target"]
        assert path.read_bytes() == before


def test_powershell_prepare_is_explicit_no_model_stage() -> None:
    argv = _ps_argv("-Action Run -Mode Both -Countries FR -WithNuclear -NuclearStage Prepare")
    assert argv[argv.index("--stage") + 1].lower() == "prepare"


def test_powershell_report_updates_reports_without_training_stage() -> None:
    argv = _ps_argv("-Action Run -Mode Both -Countries FR -WithNuclear -NuclearStage Report -SkipObservedSync")
    assert argv[argv.index("--stage") + 1].lower() == "report"
    assert "--skip-observed-sync" in argv


def test_offline_storm_archive_is_resolved_per_country() -> None:
    settings = runner.load_settings(runner.ROOT / "config/nuclear_forecast.yaml")
    paths = [runner.local_storm_archive(settings, zone, pd.Timestamp("2026-09-09"))
             for zone in ("FR", "DE", "BE", "NL")]
    assert len(set(paths)) == 4
    for zone, path in zip(("fr", "de", "be", "nl"), paths):
        assert path.name == f"{zone}_day_ahead_2026-09-09"
        assert path.parent.is_relative_to(runner.ROOT / "runs/live")


def test_target_refresh_extends_only_unissued_snapshot_and_never_uses_D(tmp_path, monkeypatch):
    import chronos2_modular.saturn as saturn
    settings, config, source, inputs = _snapshot_fixture(tmp_path, monkeypatch)
    config["zones"]["FR"]["timezone"] = "Europe/Paris"
    inputs["target"] = tmp_path / "canonical.parquet"
    index = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
    pd.DataFrame({"timestamp": index, "value": 50.}).to_parquet(inputs["target"])
    original = inputs["target"].read_bytes()
    new_index = pd.date_range("2025-01-02", "2026-01-04", inclusive="left", freq="h", tz="UTC")
    observed = pd.Series(60., index=new_index)
    calls = []
    def fetch(client, series, start, end, timezone, **options):
        calls.append((series, end, options))
        assert series == config["zones"]["FR"]["target"]["series"]
        return observed.copy()
    monkeypatch.setattr(saturn, "fetch_saturn_series_from_client", fetch)
    work = settings["output_root"] / "new_run"
    target = runner.refreshed_target_snapshot(settings, "FR", pd.Timestamp("2026-01-03"), work, client=object())
    values = pd.read_parquet(target)
    assert len(calls) == 1 and calls[0][1] < pd.Timestamp("2026-01-03", tz="Europe/Paris")
    assert calls[0][2]["nocache"] is True and calls[0][2]["live"] is True
    assert inputs["target"].read_bytes() == original
    assert values.timestamp.max() < pd.Timestamp("2026-01-03", tz="Europe/Paris")
    assert values.value.eq(60).all()
    audit = json.loads(target.with_suffix(".audit.json").read_text())
    assert audit["report_observations_used"] is False and audit["used_for_prediction"] is True
    assert audit["series"] == config["zones"]["FR"]["target"]["series"]


@pytest.mark.parametrize("fault", ["missing", "infinite", "duplicate", "naive", "off_hour", "sealed"])
def test_training_target_refresh_rejects_invalid_canonical_data_without_epex_fallback(tmp_path, monkeypatch, fault):
    import numpy as np
    import chronos2_modular.saturn as saturn
    settings, config, _, inputs = _snapshot_fixture(tmp_path, monkeypatch)
    config["zones"]["FR"]["timezone"] = "Europe/Paris"
    inputs["target"] = tmp_path / "canonical.parquet"
    index = pd.date_range("2025-01-03", "2026-01-03", inclusive="left", freq="h", tz="Europe/Paris")
    pd.DataFrame({"timestamp": index, "value": 50.}).to_parquet(inputs["target"])
    before = inputs["target"].read_bytes()
    values = pd.Series(60., index=index)
    if fault == "missing": values.iloc[-1] = np.nan
    if fault == "infinite": values.iloc[5] = np.inf
    if fault == "duplicate": values = pd.concat([values.iloc[:1], values])
    if fault == "naive": values.index = values.index.tz_localize(None)
    if fault == "off_hour": values.index = values.index + pd.Timedelta(minutes=15)
    calls = []
    def fetch(client, series, *args, **kwargs):
        calls.append(series)
        assert series == config["zones"]["FR"]["target"]["series"]
        return values
    monkeypatch.setattr(saturn, "fetch_saturn_series_from_client", fetch)
    work = settings["output_root"] / "new_run"
    if fault == "sealed":
        work.mkdir(parents=True)
        (work / "input_snapshot.json").write_text("{}")
    with pytest.raises(ValueError):
        runner.refreshed_target_snapshot(settings, "FR", pd.Timestamp("2026-01-03"), work, client=object())
    assert len(calls) == (0 if fault == "sealed" else 1)
    assert inputs["target"].read_bytes() == before
    assert not (work / "report_only/target_for_new_snapshot.parquet").exists()


@pytest.mark.parametrize("stage", ["report", "run"])
@pytest.mark.parametrize("kalman_only", [False, True])
def test_report_and_existing_run_load_bundle_without_model_calls(tmp_path, monkeypatch, capsys, stage, kalman_only):
    from dataclasses import dataclass
    import chronos2_hourly.nuclear_preparation as preparation
    import chronos2_hourly.nuclear_residual_inputs as residual
    import chronos2_hourly.nuclear_forecast as engine
    import chronos2_hourly.nuclear_run_archive as archive
    import chronos2_hourly.nuclear_attribution as attribution
    import chronos2_hourly.nuclear_reporting as reporting
    import chronos2_hourly.nuclear_exports as exports
    import chronos2_modular.data as readers

    settings, _, _, _ = _snapshot_fixture(tmp_path, monkeypatch)
    day = pd.Timestamp("2026-01-04")
    work = runner.experiment_workdir(settings, day, "FR")
    runner.snapshot_config(settings, "FR", day, work)
    (work / "report_only/frozen_result").mkdir(parents=True)
    runner.write_json(work / "run_result.json", {"variable_attribution": {"status": "unavailable"}})
    monkeypatch.setattr(runner, "load_settings", lambda path: settings)
    monkeypatch.setattr(runner, "delivery_date", lambda value: day)
    monkeypatch.setattr(runner, "check_lora_inactive", lambda *args: None)
    def sync(*args):
        if kalman_only:
            pytest.fail("Shared source sync must not run a second time")
    monkeypatch.setattr(runner, "sync_source", sync)
    monkeypatch.setattr(runner, "ensure_residual_inputs", sync)
    monkeypatch.setattr(residual, "audit_residual_bank", lambda *args: _audit())
    observed = pd.Series([50.], index=pd.date_range("2026-01-03", periods=1, tz="Europe/Paris"))

    @dataclass
    class Data:
        target: pd.Series

    monkeypatch.setattr(preparation, "prepare_nuclear_zone_data", lambda *args: Data(observed))
    frozen = SimpleNamespace(audit={"computation_mode": "incremental", "daily_chronos_cache": {"history_misses": 731}},
                             kalman_view=SimpleNamespace(replay=SimpleNamespace(audit={})))
    calls = []
    monkeypatch.setattr(archive, "load_nuclear_result_bundle", lambda **kwargs: calls.append("load") or frozen)
    monkeypatch.setattr(engine, "run_nuclear_forecast", lambda **kwargs: pytest.fail("No model/replay allowed"))
    monkeypatch.setattr(archive, "save_nuclear_result_bundle", lambda *args, **kwargs: pytest.fail("No bundle rewrite"))
    def explain(**kwargs):
        if kalman_only:
            pytest.fail("Explicitly skipped attribution must not start a model")
        return work / "existing_attribution"
    monkeypatch.setattr(attribution, "prepare_nuclear_attribution", explain)
    monkeypatch.setattr(readers, "read_series_file", lambda *args: observed)
    monkeypatch.setattr(runner, "local_storm_archive", lambda *args: None)
    renders = []
    monkeypatch.setattr(reporting, "render_nuclear_reports", lambda *args, **kwargs: renders.append(kwargs) or {})
    monkeypatch.setattr(exports, "publish_nuclear_exports", lambda *args, **kwargs: {"index": "test index"})
    arguments = ["--stage", stage, "--zones", "FR", "--skip-observed-sync"]
    if kalman_only:
        arguments += ["--skip-source-sync", "--skip-attribution", "--report-variants", "kalman"]
    assert runner.main(arguments) == 0
    assert calls == ["load"]
    assert renders[0]["operational_layout"] is True
    assert renders[0]["report_variants"] == (("kalman",) if kalman_only else ("autonomous", "kalman"))
    status = json.loads((work / "run_status.json").read_text())
    assert status["status"] == "complete" and status["model_result_saved"] is True
    assert status["forecast_result_reused"] is True
    output = capsys.readouterr().out
    assert "0 nouveau calcul Chronos, correcteur ou Kalman" in output
    assert "nouveaux calculs Chronos=731" not in output
    assert json.loads((work / "run_result.json").read_text())["execution"]["forecast_result_reused"] is True
    assert json.loads((work / "run_result.json").read_text())["exports"]["index"] == "test index"


@pytest.mark.parametrize("arguments", [
    "-Action Run -Mode Autonomous -WithNuclear", "-Action Run -Mode All -WithNuclear",
    "-Action Run -Mode Both -NuclearStage Audit",
    "-Action Run -Mode Both -WithNuclear -AllowModelDownload",
    "-Action Run -Mode Both -WithNuclear -LoraActivationConfig config/chronos2_exogenous_activation_v1.yaml",
    "-Action Run -Mode Both -WithNuclear -KalmanConfig config/kalman_operational.yaml",
])
def test_powershell_rejects_incompatible_nuclear_overrides(arguments: str) -> None:
    completed = _ps(arguments)
    assert completed.returncode != 0
    assert "Nuclear" in completed.stdout + completed.stderr
