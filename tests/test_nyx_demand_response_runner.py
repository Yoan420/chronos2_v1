"""Sealed unavailable-expert replay tests; no real experiments or APIs touched."""
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pytest

from nyx_demand_response import runner


def config():
    return dict(schema_version=1, source_suite="source", output_root=runner.NAMESPACE.as_posix(),
                bundle_path=None, diagnostic_only=True, production_modified=False)


@pytest.mark.parametrize("path", ["runs/exports/x", "../outside", "runs/experiments/nyx_congestion_v1/x"])
def test_namespace_rejects_operational_or_other_experiment_paths(tmp_path, path):
    with pytest.raises(ValueError, match="isolated demand-response"):
        runner.safe_path(tmp_path, path)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path.parent/("dr_"+uuid.uuid4().hex[:8])
    root.mkdir()
    source = root/"source"
    source.mkdir()
    wide = pd.DataFrame([dict(zone=z, timestamp_utc=pd.Timestamp(f"2026-09-14T{h:02d}:00Z"),
        forecast_origin_utc=pd.Timestamp("2026-09-13T06:00Z"), forecast=100., q10=80., q90=120.,
        actual=110., storm=105., sample="evaluation") for z in ("FR", "DE", "BE", "NL") for h in (17, 18)])
    wide.to_parquet(source/"predictions.parquet", index=False)
    for name in runner.source_runner.RESULTS.difference({"predictions.parquet"}):
        (source/name).write_text("fixture", encoding="utf8")
    manifest = {"source_dir": str(root/"ancestor"), "source_files": {"activation_manifest.json": "frozen-stage-one"}}
    (source/"manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    (source/"results_manifest.json").write_text(json.dumps(dict(status="completed",
        suite_manifest_sha256=runner.digest(source/"manifest.json"), source_activation_manifest_sha256="frozen-stage-one",
        result_files={name: runner.digest(source/name) for name in runner.source_runner.RESULTS})), encoding="utf8")
    (root/"config").mkdir()
    for name in runner.EVIDENCE_CONFIGS:
        path = root/name
        if name == runner.SOURCE_AUDIT:
            content = {"sources": []}
        elif "sources" in name:
            content = {"data": "no available real curves"}
        else:
            content = {"expert": {"stack_scope_qualified_by_zone": {z: False for z in ("FR", "DE", "BE", "NL")}},
                       "network": {"domain_reference_qualified": False, "boundary_qualified": False}}
        path.write_text(json.dumps(content), encoding="utf8")  # JSON is valid YAML for these fixture mappings.
    monkeypatch.setattr(runner.source_runner, "read_snapshot", lambda path, **kw: (path.resolve(), {}, manifest))
    monkeypatch.setattr(runner, "protected_state", lambda root: {"Forecast.ps1": "unchanged"})
    monkeypatch.setattr(runner, "code_identity", lambda root: {"new_runner.py": "fixed"})
    monkeypatch.setattr(runner, "runtime_identity", lambda: {"python": "fixture"})
    long = wide.assign(model_id="nuclear_kalman")
    monkeypatch.setattr(runner, "assemble_panel", lambda predictions, ancestor: (
        long.copy(deep=True), wide.copy(deep=True), {"source_config": {"delivery_day": "2026-09-15", "end_day": "2026-09-14"}},
        {str(source/"predictions.parquet"): runner.digest(source/"predictions.parquet")}))
    before = {name: runner.digest(source/name) for name in ("manifest.json", "results_manifest.json", *runner.source_runner.RESULTS)}
    directory = runner.prepare(config(), root)
    return root, source, directory, before, wide


def test_prepare_preserves_source_and_copies_exact_frozen_predictions(prepared):
    root, source, directory, before, wide = prepared
    assert (source/"predictions.parquet").read_bytes() == (directory/"source_predictions.parquet").read_bytes()
    _, manifest = runner.read_snapshot(directory, root)
    assert manifest["config"] == config()
    assert runner.resolve(config(), root) == directory
    assert {name: runner.digest(source/name) for name in before} == before


@pytest.mark.parametrize("target", ["input", "source", "code", "runtime"])
def test_changed_input_source_or_runtime_cannot_be_resumed(prepared, monkeypatch, target):
    root, source, directory, _, _ = prepared
    if target == "input":
        (directory/"source_predictions.parquet").write_bytes(b"tampered")
    elif target == "source":
        (source/"predictions.parquet").write_bytes(b"tampered")
    elif target == "code":
        monkeypatch.setattr(runner, "code_identity", lambda root: {"new_runner.py": "changed"})
    else:
        monkeypatch.setattr(runner, "runtime_identity", lambda: {"python": "changed"})
    with pytest.raises(ValueError):
        runner.read_snapshot(directory, root)


def test_unsealed_bundle_added_after_prepare_is_rejected_before_reading_curves(prepared, monkeypatch):
    root, _, directory, _, _ = prepared
    (directory/"bundle.json").write_text("{}", encoding="utf8")
    monkeypatch.setattr(runner, "scenario_forecasts", lambda *a: pytest.fail("Unsealed scenario inputs were consumed"))
    with pytest.raises(ValueError):
        runner.evaluate(directory, root)
    assert not (directory/"results_manifest.json").exists()


def test_unavailable_expert_is_all_nan_not_silently_filled_with_nyx(prepared, monkeypatch):
    root, source, directory, before, wide = prepared
    monkeypatch.setattr(runner, "scenario_forecasts", lambda *a: pytest.fail("Missing curves triggered scenario fitting"))
    assert runner.evaluate(directory, root) == directory
    result = pd.read_parquet(directory/"predictions.parquet")
    assert result[["expert_price", "expert_q10", "expert_q90"]].isna().all().all()
    assert result.expert_status.eq("unavailable_no_evidenced_demand_curve").all()
    pd.testing.assert_series_equal(result.nyx_unchanged, wide.forecast, check_names=False)
    payload = json.loads((directory/"evaluation.json").read_text(encoding="utf8"))
    assert payload["decision"]["qualified_country_hours"] == 0
    assert payload["decision"]["interventions"] == 0
    assert payload["decision"]["empirical_gain_demonstrated"] is False
    assert payload["paired_expert_evaluation"] is None
    assert payload["decision"]["integration_ready"] is False
    assert payload["audit"]["production_modified"] is False
    assert {name: runner.digest(source/name) for name in before} == before
    runner.verify_results(directory)
    assert runner.evaluate(directory, root) == directory  # Reuse complete immutable result, no new calculation.


def test_result_file_and_suite_seals_are_enforced(prepared):
    root, _, directory, _, _ = prepared
    runner.evaluate(directory, root)
    runner.verify_results(directory)
    (directory/"scenario_solutions.json").write_text("[42]", encoding="utf8")
    with pytest.raises(ValueError):
        runner.verify_results(directory)


def test_concurrent_production_change_prevents_result_seal(prepared, monkeypatch):
    root, _, directory, _, _ = prepared
    count = []
    def changed(root):
        count.append(1)
        return {"Forecast.ps1": "unchanged" if len(count) == 1 else "different"}
    monkeypatch.setattr(runner, "protected_state", changed)
    with pytest.raises(ValueError, match="Concurrent mutation"):
        runner.evaluate(directory, root)
    assert not (directory/"results_manifest.json").exists()


def test_source_assembly_retains_forecast_column_needed_for_explicit_unchanged_baseline():
    # Integration smoke on the real sealed ancestor, read-only. If absent in
    # another checkout, unit tests above still enforce the expected interface.
    root = Path(__file__).resolve().parents[1]
    source = root/"runs/experiments/nyx_congestion_calibration_v1/snapshots/20260915T160622Z_2de43144"
    if not (source/"results_manifest.json").exists():
        pytest.skip("Read-only ancestor fixture not present in this checkout")
    manifest = json.loads((source/"manifest.json").read_text(encoding="utf8"))
    raw = pd.read_parquet(source/"predictions.parquet")
    _, wide, _, _ = runner.assemble_panel(raw, Path(manifest["source_dir"]))
    assert {"forecast", "q10", "q90", "actual", "storm", *runner.KEYS}.issubset(wide)
    assert np.isfinite(wide.forecast).all()


def test_result_provenance_includes_metric_assembly_code_and_scipy_runtime():
    from importlib.metadata import version
    root = Path(__file__).resolve().parents[1]
    identity = runner.code_identity(root)
    required = ("kpi_report/metrics.py", "kpi_report/economic.py", "nyx_congestion_calibration/report.py",
                "nyx_congestion/report.py", "nyx_physical_p50/report.py")
    for name in required:
        assert identity[name] == runner.digest(root/name)
    runtime = runner.runtime_identity()
    assert runtime["versions"]["scipy"] == version("scipy")
