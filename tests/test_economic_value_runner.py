"""Isolated annual snapshot/launcher tests; no live files or network changes."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from economic_value import runner


ROOT = Path(__file__).resolve().parents[1]
PS = shutil.which("powershell.exe") or shutil.which("pwsh")


@pytest.fixture
def config():
    cfg = runner.load_config(ROOT / "config/economic_value.yaml")
    cfg["zones"], cfg["models"] = ["FR"], ["kalman"]
    return cfg


def _panel(config):
    tz = "Europe/Paris"
    times = pd.date_range(pd.Timestamp("2025-09-11", tz=tz), pd.Timestamp("2026-09-11", tz=tz),
                          freq="h", inclusive="left").tz_convert("UTC")
    local = times.tz_convert(tz).tz_localize(None).normalize()
    origins = (local-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize(tz).tz_convert("UTC")
    return pd.DataFrame({"timestamp_utc": times, "zone": "FR", "model": "kalman", "forecast_origin_utc": origins,
                         "forecast": 70., "q10": 55., "q90": 85., "actual": np.where(np.arange(len(times)) % 7, 75., 35.),
                         "benchmark_forecast": 40., "reference_price": 50., "reference_available_at_utc": origins-pd.Timedelta(hours=14),
                         "duration_hours": 1., "forecast_eligible": True, "reference_eligible": True, "sample": "evaluation"})


@pytest.mark.parametrize("section,key,value", [(None,"cutoff_time","10:30"), (None,"evaluation_days",364),
                                               ("reference","executable",True), (None,"order_execution_enabled",True),
                                               ("strategy","transaction_cost_eur_mwh",-1), ("portfolio","capacity_mw",0),
                                               ("strategy","confidence_filter","high")])
def test_invalid_recipe(config, section, key, value):
    (config if section is None else config[section])[key] = value
    with pytest.raises(ValueError):
        runner.validate_config(config)


def test_portfolio_total_not_per_country(config):
    config["zones"] = ["FR", "DE", "BE", "NL"]
    config["portfolio"]["capacity_mw"] = 200.
    policy = runner.engine_config(config)
    assert policy["portfolio_capacity_mw"] == 200.
    assert policy["zone_capacity_mw"] == {z: 50. for z in config["zones"]}


def test_malformed_yaml_is_actionable(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("strategy: [unclosed", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML illisible"):
        runner.load_config(path)


def test_same_model_comparison_labels_required(config):
    a = _panel(config)
    b = a.assign(model="autonomous", actual=a.actual+1)
    config["models"].append("autonomous")
    with pytest.raises(ValueError, match="shared actual"):
        runner._validate_panel(pd.concat([a,b]), config)


def test_late_forecast_origin_refused(config):
    panel = _panel(config)
    panel["forecast_origin_utc"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="08:00"):
        runner._validate_panel(panel, config)


def test_shortened_calendar_refused(config):
    with pytest.raises(ValueError, match="support"):
        runner._validate_panel(_panel(config).iloc[1:], config)


def test_snapshot_engine_report_and_idempotent_replay(tmp_path, monkeypatch, config):
    panel = _panel(config)
    data = {**runner._validate_panel(panel, config), "delivery_day": "2026-09-10"}
    monkeypatch.setattr(runner, "audit_inputs", lambda *a, **k: (panel, data, {"kind":"lagged_day_ahead_proxy"}))
    preserved = tmp_path / "Forecast.ps1"
    preserved.write_text("user production script", encoding="utf-8")
    snapshot = runner.prepare(config, root=tmp_path)
    path = runner.evaluate(snapshot, root=tmp_path)
    assert path.is_file() and path.stat().st_size > 10000
    result_hash = runner.digest(snapshot / "rows.parquet")
    metrics = pd.read_parquet(snapshot / "metrics.parquet")
    assert set(metrics.strategy) == {"model", "benchmark", "no_forecast"}
    assert metrics.annual_fully_observed.all()
    assert metrics.loc[metrics.strategy.eq("model"), "economic_value_added_eur"].gt(0).all()
    # An already computed snapshot must not simulate again, even after sources disappear.
    import economic_value.engine as engine
    monkeypatch.setattr(engine, "simulate", lambda *a, **k: pytest.fail("Completed snapshot must not be resimulated"))
    runner.evaluate(snapshot, root=tmp_path)
    assert runner.digest(snapshot / "rows.parquet") == result_hash
    assert preserved.read_text(encoding="utf-8") == "user production script"
    assert json.loads((snapshot / "status.json").read_text())["status"] == "completed"
    audit_path = snapshot / "results_manifest.json"
    saved_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    transplanted = deepcopy(saved_audit)
    transplanted["snapshot_files"]["config.json"] = "other-portfolio-configuration"
    runner._json(audit_path, transplanted)
    with pytest.raises(ValueError, match="frozen input manifest"):
        runner.report(snapshot, root=tmp_path)
    runner._json(audit_path, saved_audit)
    (snapshot / "rows.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        runner.report(snapshot, root=tmp_path)


def test_failed_evaluation_preserves_latest(tmp_path, monkeypatch, config):
    panel = _panel(config)
    data = runner._validate_panel(panel, config)
    monkeypatch.setattr(runner, "audit_inputs", lambda *a, **k: (panel, data, {}))
    snapshot = runner.prepare(config, root=tmp_path)
    pointer = tmp_path / config["output_root"] / "latest.json"
    pointer.write_text('{"report":"previous validated report"}', encoding="utf-8")
    import economic_value.engine as engine
    def broken(*args, **kwargs):
        raise ValueError("Synthetic failure")
    monkeypatch.setattr(engine, "simulate", broken)
    with pytest.raises(ValueError, match="Synthetic failure"):
        runner.evaluate(snapshot, root=tmp_path)
    assert json.loads(pointer.read_text())["report"] == "previous validated report"
    assert json.loads((snapshot / "status.json").read_text())["status"] == "failed"


def test_manifest_dates_cannot_change_scores(tmp_path, monkeypatch, config):
    panel = _panel(config)
    monkeypatch.setattr(runner, "audit_inputs", lambda *a, **k: (panel, runner._validate_panel(panel,config), {}))
    snapshot = runner.prepare(config, root=tmp_path)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    manifest["evaluation_end"] = "2026-09-09"
    runner._json(snapshot / "manifest.json", manifest)
    with pytest.raises(ValueError, match="calendar manifest"):
        runner.read_snapshot(snapshot, root=tmp_path)


def _quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def _ps(script, *, cwd):
    if PS is None:
        pytest.skip("PowerShell is unavailable")
    return subprocess.run([PS, "-NoProfile", "-Command", script], cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=False)


@pytest.fixture
def launcher(tmp_path):
    root = tmp_path / "project with spaces"
    (root / "config").mkdir(parents=True)
    shutil.copyfile(ROOT / "EconomicValue.ps1", root / "EconomicValue.ps1")
    shutil.copyfile(ROOT / "config/economic_value.yaml", root / "config/economic_value.yaml")
    (root / "run_economic_value.py").write_text("import sys,json\nprint(json.dumps(sys.argv[1:]))\n", encoding="utf-8")
    return root


def test_launcher_dryrun_from_other_directory(launcher, tmp_path):
    before = set(tmp_path.rglob("*"))
    completed = _ps(f"& {_quote(launcher/'EconomicValue.ps1')} -PythonExecutable {_quote(sys.executable)} "
                    "-Action Run -Countries FR,DE -Models kalman -PortfolioMW 12.5 -DryRun", cwd=tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    line = next(s for s in completed.stdout.splitlines() if s.startswith("Commande (argv, shell=False): "))
    argv = json.loads(line.split(": ",1)[1])
    assert argv[1] == str(launcher / "run_economic_value.py")
    assert argv[argv.index("--portfolio-mw")+1] == "12.5"
    assert argv[argv.index("--config")+1] == str(launcher / "config/economic_value.yaml")
    assert set(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("selection", ["", "-Countries fr", "-Models Kalman", "-Countries fr -Models Kalman"])
def test_launcher_defaults_and_case(launcher, tmp_path, selection):
    completed = _ps(f"& {_quote(launcher/'EconomicValue.ps1')} -PythonExecutable {_quote(sys.executable)} "
                    f"{selection} -DryRun", cwd=tmp_path)
    assert completed.returncode == 0, completed.stdout+completed.stderr
    if "-Countries" in selection:
        assert '"FR"' in completed.stdout and '"fr"' not in completed.stdout
    if "-Models" in selection:
        assert '"kalman"' in completed.stdout and '"Kalman"' not in completed.stdout


@pytest.mark.parametrize("options", ["-DeliveryDay 2026-02-30", "-Countries FR,FR", "-PortfolioMW 0", "-Models invalid"])
def test_launcher_rejects_invalid_input(launcher, tmp_path, options):
    completed = _ps(f"& {_quote(launcher/'EconomicValue.ps1')} -PythonExecutable {_quote(sys.executable)} {options} -DryRun", cwd=tmp_path)
    assert completed.returncode != 0


def test_launcher_propagates_python_failure(launcher, tmp_path):
    (launcher / "run_economic_value.py").write_text("raise SystemExit(7)\n", encoding="utf-8")
    completed = _ps(f"& {_quote(launcher/'EconomicValue.ps1')} -PythonExecutable {_quote(sys.executable)} -Action Run", cwd=tmp_path)
    assert completed.returncode != 0 and "code 7" in completed.stderr
