"""V2 must preserve the incumbent and exact365-day denominators when unqualified."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from marginal_cost_expert import runner_v2 as runner


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return runner.load_config(ROOT / "config/marginal_cost_expert_v2.yaml")


@pytest.mark.parametrize("field,value", [("cutoff_time", "10:30"), ("training_days", 364),
                                         ("end_day", "2026-09-08"), ("timezone", "UTC")])
def test_strict_protocol(config, field, value):
    config["evaluation"][field] = value
    with pytest.raises(ValueError):
        runner.validate_config(config)


@pytest.mark.parametrize("field", ["domain_reference_qualified", "boundary_qualified"])
def test_no_coupling_by_boolean(config, field):
    config["network"][field] = True
    with pytest.raises(ValueError, match="configuration flag"):
        runner.validate_config(config)


def test_no_activation(config):
    config["activate"] = True
    with pytest.raises(ValueError, match="activate"):
        runner.validate_config(config)


def test_raw_diagnostics_never_intervene_even_if_perfect():
    index = runner.physical_index("2026-06-24", "2026-06-24")
    reference = pd.DataFrame({"timestamp": index, "zone": "DE", "actual": 250., "base": 40., "storm": 230.})
    raw = pd.DataFrame({"timestamp": index, "zone": "DE", "expert": 250., "expert_oof": True})
    result = runner.assemble_predictions(reference, raw)
    assert result.diagnostic_expert.eq(250).all()
    assert result.expert.isna().all() and not result.expert_available.any()
    assert result.weight.eq(0).all() and result.guarded.equals(reference.base)


def _fixture(tmp_path, monkeypatch, config):
    root = tmp_path
    config = deepcopy(config)
    old = root / config["reference_snapshot"]
    old.mkdir(parents=True)
    support = runner.physical_index("2024-09-10", "2026-09-09")
    evaluated = runner.physical_index("2025-09-10", "2026-09-09")
    features = pd.DataFrame({"delivery_start_utc": support, "zone": "FR", "demand_basis": "residual",
                             "demand_mw": 50., "inputs_complete": True, "inputs_qualified": False})
    targets = pd.DataFrame({"timestamp": support, "zone": "FR", "actual": 50.})
    references = pd.DataFrame({"timestamp": evaluated, "zone": "FR", "actual": 50., "base": 52., "storm": 55.})
    old_config = {"schema_version": 1, "activate": False, "evaluation": config["evaluation"]}
    for name, frame in (("features", features), ("targets", targets), ("references", references)):
        frame.to_parquet(old / f"{name}.parquet", index=False)
    for name, value in (("config", old_config), ("sources_config", {})):
        runner._json(old / f"{name}.json", value)
    old_audit = {"evaluation_start": "2025-09-10", "evaluation_end": "2026-09-09", "zones": ["FR"],
                 "snapshot_files": {name: runner.digest_file(old / name) for name in
                                    ("features.parquet", "targets.parquet", "references.parquet", "config.json", "sources_config.json")}}
    runner._json(old / "audit.json", old_audit)
    sources = {"timezone": "Europe/Paris", "zones": {"FR": {"demand_basis": "residual",
               "residual_netting": config["expert"]["residual_netting"]["FR"]}}}
    source_path = root / config["sources_config"]
    source_path.parent.mkdir(parents=True)
    source_path.write_text(yaml.safe_dump(sources), encoding="utf-8")
    network = {"first_day": "2024-09-10", "last_day": "2026-09-09", "calendar_days": 730,
               "cutoff_time": "08:00", "timezone": "Europe/Paris", "domain_reference_qualified": False,
               "boundary_qualified": False, "complete_research_days": 0,
               "network_code_sha256": runner.digest_file(ROOT / "marginal_cost_expert/network.py"),
               "daily": [{"delivery_day": str(d.date()), "expected_hours": len(runner.physical_index(str(d.date()), str(d.date()))),
                          "available_hours": 0, "inputs_qualified": False} for d in pd.date_range("2024-09-10", "2026-09-09")]}
    network_path = root / config["network"]["audit_path"]
    network_path.parent.mkdir(parents=True)
    runner._json(network_path, network)
    def inputs(*args, **kwargs):
        return features, {"zones": {"FR": {"qualification": {"omitted_segments": ["hydro", "storage"]}}}}
    def physical(*args, **kwargs):
        return pd.DataFrame({"delivery_start_utc": support, "zone": "FR", "candidate_id": "central",
                             "price_eur_mwh": np.nan, "raw_price_eur_mwh": 80., "eligible": False,
                             "shortage_mw": 0., "capacity_margin_mw": 25., "curtailment_mw": 0.})
    monkeypatch.setattr(runner, "load_supply_inputs", inputs)
    monkeypatch.setattr(runner, "_physical_candidates", physical)
    return config, network_path, source_path


def test_full_365_day_sealed_diagnostic_and_offline_report(tmp_path, monkeypatch, config):
    config, _, _ = _fixture(tmp_path, monkeypatch, config)
    snapshot = runner.prepare(config, project_root=tmp_path)
    runner.backtest(snapshot, project_root=tmp_path)
    predictions = pd.read_parquet(snapshot / "predictions.parquet")
    assert len(predictions) == 8760
    assert predictions.weight.eq(0).all() and predictions.expert.isna().all()
    assert predictions.base.equals(predictions.guarded)
    metrics = pd.read_parquet(snapshot / "metrics.parquet")
    assert metrics.days.eq(365).all() and metrics.hours.eq(8760).all()
    assert metrics.loc[metrics.model.eq("base"), "mae"].iloc[0] == 2
    audit = json.loads((snapshot / "result_audit.json").read_text())
    assert not audit["annual_improvement_demonstrated"] and audit["intervention_hours"] == 0
    report = (snapshot / "marginal_cost_v2_report.html").read_text(encoding="utf-8")
    assert "10 h 30 est exclue" in report and "Simulation zonale NON QUALIFIÉE" in report
    assert "__PAYLOAD__" not in report and "cdn.plot.ly" not in report.split("<script>")[0]
    with pytest.raises(ValueError, match="already sealed"):
        runner.backtest(snapshot, project_root=tmp_path)
    # Local snapshot is sufficient: report doesn't reload mutable source banks.
    monkeypatch.setattr(runner, "load_supply_inputs", lambda *a, **k: pytest.fail("No source requests during Report"))
    runner.report(snapshot, project_root=tmp_path)
    (snapshot / "predictions.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        runner.report(snapshot, project_root=tmp_path)


def test_audit_dates_cannot_shorten_evaluation(tmp_path, monkeypatch, config):
    config, _, _ = _fixture(tmp_path, monkeypatch, config)
    snapshot = runner.prepare(config, project_root=tmp_path)
    audit = json.loads((snapshot / "audit.json").read_text())
    audit["evaluation_end"] = "2026-09-08"
    runner._json(snapshot / "audit.json", audit)
    with pytest.raises(ValueError, match="disagrees"):
        runner._read_snapshot(snapshot, tmp_path)


def test_netting_mismatch_is_not_silent(tmp_path, monkeypatch, config):
    config, _, sources = _fixture(tmp_path, monkeypatch, config)
    value = yaml.safe_load(sources.read_text())
    value["zones"]["FR"]["residual_netting"] = ["wind", "solar"]
    sources.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ValueError, match="netting disagree"):
        runner.prepare(config, project_root=tmp_path)


def test_late_network_audit_refused(tmp_path, monkeypatch, config):
    config, path, _ = _fixture(tmp_path, monkeypatch, config)
    audit = json.loads(path.read_text())
    audit["cutoff_time"] = "10:30"
    runner._json(path, audit)
    with pytest.raises(ValueError, match="08:00"):
        runner.prepare(config, project_root=tmp_path)


def test_input_seal_refuses_corruption(tmp_path, monkeypatch, config):
    config, _, _ = _fixture(tmp_path, monkeypatch, config)
    snapshot = runner.prepare(config, project_root=tmp_path)
    (snapshot / "features.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        runner.backtest(snapshot, project_root=tmp_path)
