"""The fixtures here only validate adapter planning; they do not train models."""
from __future__ import annotations

import gzip
from pathlib import Path
import sys

import pytest
import yaml

from experiment_console.adapters import AdapterRegistry, SCRIPTS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def registry(tmp_path):
    project = tmp_path / "project with spaces"
    project.mkdir()
    for script in SCRIPTS.values():
        (project / script).write_text("# Adapter planning fixture: never executed.\n", encoding="utf-8")
    original = PROJECT_ROOT / "chronos2_hourly_fr_residual_v1.yaml"
    (project / original.name).write_bytes(original.read_bytes())
    source = project / "runs" / "chronos2_hourly_fixture"
    source.mkdir(parents=True)
    with gzip.open(source / "backtest_hourly_oof.csv.gz", "wt", encoding="utf-8") as stream:
        stream.write("delivery_start_utc,actual,ensemble__q50,residual_corrected__q50\n")
        stream.write("2026-01-01T00:00:00Z,12,11,13\n")
    return AdapterRegistry(project, Path(sys.executable).resolve())


def test_catalog_contains_real_clis_and_real_config_models(registry):
    catalog = {item["id"]: item for item in registry.catalog()}
    assert set(catalog) == set(SCRIPTS)
    forecast = catalog["hourly_forecast"]
    assert forecast["models"] == ["residual_corrected"]
    assert forecast["configs"][0]["zone"] == "FR"
    assert forecast["configs"][0]["id"] == "chronos2_hourly_fr_residual_v1.yaml"


def test_preview_is_read_only_and_snapshot_is_exclusive(registry):
    destination = registry.project_root / "console-data" / "run-1"
    request = {"adapter_id": "model_storm_report", "parameters": {"delivery_day": "2026-09-11"}}
    plan = registry.prepare(request, destination)
    assert not destination.exists()
    assert plan["command"] == [str(Path(sys.executable).resolve()), "-u",
                               str(registry.project_root / "run_model_storm_report.py"),
                               "--delivery-day", "2026-09-11", "--output",
                               str(destination / "outputs" / "model_storm.html")]
    written = registry.prepare(request, destination, write=True)
    assert yaml.safe_load(Path(written["config_path"]).read_text(encoding="utf-8")) == plan["config"]
    assert not Path(written["output_dir"]).exists()
    with pytest.raises(FileExistsError):
        registry.prepare(request, destination, write=True)


def test_hourly_snapshot_retains_science_and_resolves_original_paths(registry):
    original_path = registry.project_root / "chronos2_hourly_fr_residual_v1.yaml"
    original_bytes = original_path.read_bytes()
    original = yaml.safe_load(original_bytes)
    destination = registry.project_root / "console-data" / "forecast-1"
    plan = registry.prepare({"adapter_id": "hourly_forecast", "model": "residual_corrected"}, destination, write=True)
    effective = plan["config"]
    assert original_path.read_bytes() == original_bytes
    assert effective["hourly"] == original["hourly"]
    assert effective["backtest"] == original["backtest"]
    assert effective["data"]["runtime_as_of"] == original["data"]["runtime_as_of"]
    assert effective["data"]["source"] == "cache"
    assert effective["data"]["project_root"] == str(registry.project_root)
    assert effective["data"]["pit_vintage_dir"] == str(registry.project_root / "data" / "pit" / "vintages")
    assert effective["zones"]["FR"]["target"]["source"] == "cache"
    assert effective["zones"]["FR"]["covariates"]["fr_residual_load_fcst"]["source"] == "pit_parquet"
    assert Path(effective["report"]["filename"]).is_relative_to(destination)
    assert plan["resource_keys"] == ["scientific-cache"]
    assert "--local-files-only" in plan["command"]
    assert not any(flag in plan["command"] for flag in ("--refresh-data", "--data-as-of"))


def test_report_source_is_never_used_as_output(registry):
    source = registry.project_root / "runs" / "chronos2_hourly_fixture"
    before = {path: path.read_bytes() for path in source.iterdir()}
    plan = registry.prepare({"adapter_id": "hourly_report", "model": "residual_corrected",
                             "parameters": {"title": "Example; $(never execute)", "source_run": str(source)}},
                            registry.project_root / "console-data" / "report-1", write=True)
    assert {path: path.read_bytes() for path in source.iterdir()} == before
    title_index = plan["command"].index("--title")
    assert plan["command"][title_index + 1] == "Example; $(never execute)"
    assert not Path(plan["output_dir"]).is_relative_to(source)


def test_evaluation_requires_actual_columns_and_validates_integer_bounds(registry):
    destination = registry.project_root / "console-data" / "eval-1"
    plan = registry.prepare({"adapter_id": "hourly_evaluation", "parameters": {"bootstrap_samples": 20}}, destination)
    assert "20" in plan["command"]
    for params in ({"candidate": "missing"}, {"bootstrap_samples": 0}, {"bootstrap_samples": True},
                   {"baseline": "ensemble__q50", "candidate": "ensemble__q50"}):
        with pytest.raises(ValueError):
            registry.prepare({"adapter_id": "hourly_evaluation", "parameters": params}, destination)


@pytest.mark.parametrize("run_request", [
    {"adapter_id": "arbitrary_command"},
    {"adapter_id": "model_storm_report", "config_id": "../../secret.yaml"},
    {"adapter_id": "model_storm_report", "model": "invented"},
    {"adapter_id": "model_storm_report", "parameters": {"delivery_day": "2026-99-99"}},
    {"adapter_id": "model_storm_report", "parameters": {"shell": "powershell"}},
    {"adapter_id": "hourly_report", "parameters": {"timezone": "invented/timezone"}},
])
def test_unknown_requests_are_rejected(registry, run_request):
    with pytest.raises(ValueError):
        registry.prepare(run_request, registry.project_root / "console-data" / "invalid")


def test_missing_inputs_and_paths_outside_project_are_rejected(registry, tmp_path):
    destination = registry.project_root / "console-data" / "missing"
    with pytest.raises(ValueError, match="introuvable"):
        registry.prepare({"adapter_id": "hourly_report", "parameters": {"source_run": "missing"}}, destination)
    with pytest.raises(ValueError, match="dépôt"):
        registry.prepare({"adapter_id": "hourly_report", "parameters": {"source_run": str(tmp_path)}}, destination)
    with pytest.raises(ValueError, match="dépôt"):
        registry.prepare({"adapter_id": "model_storm_report"}, tmp_path / "outside")


def test_embedded_secrets_are_rejected_and_never_written(registry):
    destination = registry.project_root / "console-data" / "secret"
    with pytest.raises(ValueError, match="secret"):
        registry.prepare({"adapter_id": "hourly_report", "parameters": {"title": "api_key=private-value"}}, destination, write=True)
    assert not destination.exists()
    original_path = registry.project_root / "chronos2_hourly_fr_residual_v1.yaml"
    payload = yaml.safe_load(original_path.read_text(encoding="utf-8"))
    payload["data"]["password"] = "private-value"
    original_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert "hourly_forecast" not in {item["id"] for item in registry.catalog()}


def test_interpreter_must_be_explicit_absolute_existing_path(registry):
    with pytest.raises(ValueError):
        AdapterRegistry(registry.project_root, "python")
    with pytest.raises(ValueError):
        AdapterRegistry(registry.project_root, registry.project_root / "missing-python.exe")


def test_duplicate_uses_saved_science_after_original_configuration_is_removed(registry):
    original = registry.prepare({"adapter_id": "hourly_forecast"}, registry.project_root / "console-data" / "original")
    (registry.project_root / "chronos2_hourly_fr_residual_v1.yaml").unlink()
    duplicate = registry.prepare(original["request"], registry.project_root / "console-data" / "duplicate",
                                 snapshot_config=original["config"])
    assert duplicate["config"]["hourly"] == original["config"]["hourly"]
    assert duplicate["config"]["zones"] == original["config"]["zones"]
    assert duplicate["config"]["data"] == original["config"]["data"]
    assert duplicate["output_dir"] != original["output_dir"]
    assert duplicate["config"]["report"]["filename"] != original["config"]["report"]["filename"]
