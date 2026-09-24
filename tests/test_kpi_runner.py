"""Independent orchestration checks; no production data, model or network."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from kpi_report import data, runner


@pytest.fixture
def source(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    economic_config = Path(__file__).resolve().parents[1] / "config/economic_value.yaml"
    (tmp_path / "config/economic_value.yaml").write_bytes(economic_config.read_bytes())
    hours = pd.date_range("2026-09-12", "2026-09-15", freq="h", inclusive="left", tz="Europe/Paris").tz_convert("UTC")
    frame = pd.DataFrame([
        {"zone": zone, "timestamp_utc": hour, "model_id": model,
         "forecast": 0. if model == "nyx" else -2., "actual": -1., "storm": 3.}
        for zone in ("BE", "FR") for model in ("nyx", "trial") for hour in hours
    ])
    catalog = [
        {"id": "nyx", "label": "NYX", "family": "production", "kind": "production", "source_path": "old/protected.parquet"},
        {"id": "trial", "label": "Trial", "family": "experiment", "kind": "experiment", "source_path": "lab/sealed.parquet"},
    ]
    audit = {"zones": ["BE", "FR"], "recommended_end_day": "2026-09-14", "source_delivery_day": "2026-09-15"}
    verified = []
    monkeypatch.setattr(data, "load_recent_models", lambda root: (frame.copy(), catalog.copy(), audit.copy()))
    monkeypatch.setattr(data, "verify_sources", lambda item: verified.append(item))
    return tmp_path, frame, catalog, audit, verified


def test_report_has_four_paired_periods_sealed_json_and_status(source):
    root, frame, _, _, verified = source
    original = frame.copy(deep=True)
    manifest = runner.produce_report(root=root)
    directory = Path(manifest["snapshot"])
    assert directory.is_relative_to(root / "runs/reports/kpi/snapshots")
    assert manifest["production_modified"] is False
    assert manifest["models_fitted"] is False
    assert manifest["forecasts_generated"] is False
    payload = json.loads((directory / "kpi_metrics.json").read_text(encoding="utf-8"))
    assert list(payload["periods"]) == ["365", "90", "30", "7"]
    for period in payload["periods"].values():
        assert {row["n_hours"] for row in period["rows"] if row["zone"] == "ALL"} == {144}
        assert {row["n_days"] for row in period["rows"] if row["zone"] == "ALL"} == {6}
        assert period["daily_rows"]
        economic = period["economic"]
        assert {row["n_country_hours"] for row in economic["rows"] if row["zone"] == "ALL"} == {96}
        assert {row["potential_energy_mwh"] for row in economic["rows"] if row["zone"] == "ALL"} == {2400.}
        assert economic["audit"]["executable_reference"] is False
        assert economic["audit"]["parameters_fitted_on_evaluation"] is False
    audit = json.loads((directory / "source_audit.json").read_text(encoding="utf-8"))
    assert audit["economic_config"]["sha256"] == runner.digest(root / "config/economic_value.yaml")
    assert all(runner.digest(directory / name) == sha for name, sha in manifest["files"].items())
    status = runner.status(root=root)
    assert status["report_hashes_verified"] is True
    assert status["models"] == 2
    assert len(verified) == 2
    pd.testing.assert_frame_equal(frame, original)


def test_repeat_generation_uses_unique_snapshot_never_overwrites(source):
    root, *_ = source
    first = runner.produce_report(root=root)
    first_bytes = Path(first["report"]).read_bytes()
    second = runner.produce_report(root=root)
    assert first["snapshot"] != second["snapshot"]
    assert Path(first["report"]).read_bytes() == first_bytes
    assert runner.status(root=root)["report"] == second["report"]


@pytest.mark.parametrize("kwargs,match", [
    ({"models": []}, "selection"), ({"models": ["unknown"]}, "selection"),
    ({"models": ["nyx", "nyx"]}, "selection"), ({"models": ["trial"]}, "production comparator"),
    ({"zones": []}, "country"), ({"zones": ["US"]}, "country"),
    ({"zones": ["FR", "FR"]}, "country"),
    ({"end_day": "2026-09-15"}, "last common"),
    ({"end_day": "20260914"}, "last common"),
])
def test_invalid_selection_never_publishes(source, kwargs, match):
    root, *_ = source
    with pytest.raises(ValueError, match=match):
        runner.produce_report(root=root, **kwargs)
    assert not (root / runner.NAMESPACE).exists()


def test_filter_once_preserves_explicit_production_and_country(source):
    root, *_ = source
    manifest = runner.produce_report(root=root, models=["nyx"], zones=["FR"])
    payload = json.loads((Path(manifest["snapshot"]) / "kpi_metrics.json").read_text(encoding="utf-8"))
    assert manifest["models"] == ["nyx"]
    assert manifest["zones"] == ["FR"]
    assert {row["zone"] for row in payload["periods"]["365"]["rows"]} == {"FR", "ALL"}


def test_all_missing_forecasts_refuses_empty_publication(source):
    root, frame, *_ = source
    frame["forecast"] = float("nan")
    with pytest.raises(ValueError, match="no common"):
        runner.produce_report(root=root)
    assert not (root / runner.NAMESPACE).exists()


def test_source_change_during_generation_does_not_publish_latest(source, monkeypatch):
    root, *_ = source
    original = runner.produce_report(root=root)
    counter = [0]
    def verify(_):
        counter[0] += 1
        if counter[0] == 2:
            raise ValueError("source changed")
    monkeypatch.setattr(data, "verify_sources", verify)
    with pytest.raises(ValueError, match="source changed"):
        runner.produce_report(root=root)
    assert runner.status(root=root)["report"] == original["report"]


def test_economic_assumptions_changed_during_computation_refuse_publication(source, monkeypatch):
    root, *_ = source
    original = runner.compute_economic_kpis
    called = [False]
    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        if not called[0]:
            called[0] = True
            path = root / "config/economic_value.yaml"
            path.write_bytes(path.read_bytes() + b"\n# Changed during report generation\n")
        return result
    monkeypatch.setattr(runner, "compute_economic_kpis", changing)
    with pytest.raises(ValueError, match="economic assumptions changed"):
        runner.produce_report(root=root)
    assert not (root / runner.NAMESPACE / "latest.json").exists()


def test_economic_assumptions_changed_during_render_refuse_latest(source, monkeypatch):
    root, *_ = source
    original = runner.render_kpi
    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        path = root / "config/economic_value.yaml"
        path.write_bytes(path.read_bytes() + b"\n# Changed during rendering\n")
        return result
    monkeypatch.setattr(runner, "render_kpi", changing)
    with pytest.raises(ValueError, match="economic assumptions changed"):
        runner.produce_report(root=root)
    assert not (root / runner.NAMESPACE / "latest.json").exists()


@pytest.mark.parametrize("name", ["KPI.html", "kpi_metrics.json", "source_audit.json"])
def test_status_detects_modified_artifacts(source, name):
    root, *_ = source
    manifest = runner.produce_report(root=root)
    path = Path(manifest["snapshot"]) / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="incomplete or changed"):
        runner.status(root=root)


def test_status_detects_manifest_tampering(source):
    root, *_ = source
    manifest = runner.produce_report(root=root)
    path = Path(manifest["snapshot"]) / "report_manifest.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest checksum"):
        runner.status(root=root)


@pytest.mark.parametrize("relative", ["runs/exports/new", "runs/reports/kpi", "runs/reports/kpi/../escaped", "outside"])
def test_safe_output_rejects_outside_namespace(tmp_path, relative):
    with pytest.raises(ValueError):
        runner.safe_output(tmp_path, relative)


def test_status_pointer_cannot_escape_namespace(source):
    root, *_ = source
    runner.produce_report(root=root)
    latest = root / runner.NAMESPACE / "latest.json"
    latest.write_text(json.dumps({"snapshot": str(root / "runs/exports"), "manifest_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="stay in"):
        runner.status(root=root)


def test_cli_status_success_and_failure_are_explicit(source, monkeypatch, capsys):
    import run_kpi_report as cli
    root, *_ = source
    monkeypatch.setattr(cli, "ROOT", root)
    assert cli.main(["--action", "status"]) == 2
    runner.produce_report(root=root)
    assert cli.main(["--action", "status"]) == 0
    assert json.loads(capsys.readouterr().out)["report_hashes_verified"] is True


def test_powershell_dryrun_does_not_execute_python(tmp_path):
    if sys.platform != "win32":
        pytest.skip("PowerShell launcher check is Windows-specific")
    script = Path(__file__).resolve().parents[1] / "KPI.ps1"
    command = "& '" + str(script).replace("'", "''") + "' -Action Report -Countries FR,DE -Models nyx,trial -EndDay 2026-09-14 -PythonExecutable '" + sys.executable.replace("'", "''") + "' -DryRun"
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command], capture_output=True, text=True, timeout=30, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "DryRun" in result.stdout
    assert '"--zones","FR","DE"' in result.stdout
    assert '"--models","nyx","trial"' in result.stdout
    assert not list(tmp_path.iterdir())


def test_powershell_fallback_python_discovery_is_valid(tmp_path):
    if sys.platform != "win32":
        pytest.skip("PowerShell launcher check is Windows-specific")
    script = Path(__file__).resolve().parents[1] / "KPI.ps1"
    copied = tmp_path / "KPI.ps1"
    copied.write_bytes(script.read_bytes())
    # No sibling pricefm virtualenv exists for this copied launcher: discovery
    # must take the installed PATH Python, still without running it.
    command = "& '" + str(copied).replace("'", "''") + "' -Action Status -DryRun"
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command], capture_output=True, text=True, timeout=30, cwd=tmp_path)
    if shutil.which("python.exe") is None and shutil.which("python") is None:
        assert result.returncode != 0
        assert "Python introuvable" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        line = next(line for line in result.stdout.splitlines() if line.startswith("Commande"))
        arguments = json.loads(line.split(": ", 1)[1])
        assert Path(arguments[0]).is_file()
        assert arguments[-2:] == ["--action", "status"]
    assert list(tmp_path.iterdir()) == [copied]
