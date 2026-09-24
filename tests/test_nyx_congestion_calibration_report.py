"""Sealed comparison, chronological reports and immutable-source regressions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from nyx_congestion_calibration.report import (CATALOG, MODELS, assemble_panel, build_report,
    diagnostics, fold_summary, read_folds, render_report, probability_diagnostics)
from nyx_congestion import report as old_report

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf8")


@pytest.fixture
def source(tmp_path):
    ancestor, physical, old = (tmp_path/name for name in ("ancestor", "physical", "old"))
    for path in (ancestor, physical, old):
        path.mkdir()
    timestamps = pd.date_range("2026-06-24", periods=72, freq="h", tz="Europe/Paris").append(
        pd.date_range("2026-09-13", periods=72, freq="h", tz="Europe/Paris")).tz_convert("UTC")
    local = timestamps.tz_convert("Europe/Paris").tz_localize(None)
    origins = (local.normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    base = pd.DataFrame({"zone": "FR", "timestamp_utc": timestamps, "forecast_origin_utc": origins,
        "actual": 100., "forecast": 90., "benchmark_forecast": 80., "q10": 70., "q90": 110.,
        "feature_fixed": np.arange(len(timestamps)), "sample": np.where(local.day == 15, "live", "evaluation")})
    audit = {"source_config": {"delivery_day": "2026-09-15", "end_day": "2026-09-14", "zones": ["FR"],
        "evaluation_days": 365, "timezone": "Europe/Paris", "cutoff_time": "08:00"}}
    dump(ancestor/"source_audit.json", audit)
    base.to_parquet(physical/"panel.parquet", index=False)
    physical_manifest = {"source_dir": str(ancestor),
        "source_files": {"source_audit.json": sha(ancestor/"source_audit.json")},
        "input_files": {"panel.parquet": sha(physical/"panel.parquet")}}
    dump(physical/"manifest.json", physical_manifest)
    base.assign(network_fuel_direct=92., nyx_physical_p50=91.).to_parquet(physical/"predictions.parquet", index=False)
    dump(physical/"results_manifest.json", {"status": "completed", "suite_manifest_sha256": sha(physical/"manifest.json"),
        "result_files": {"predictions.parquet": sha(physical/"predictions.parquet")}})
    base.to_parquet(old/"panel.parquet", index=False)
    old_predictions = base.assign(**{model: 93. for model in old_report.MODELS},
        **{model+q: 70. if q == "_q10" else 115. for model in old_report.MODELS for q in ("_q10", "_q90")})
    old_predictions.to_parquet(old/"predictions.parquet", index=False)
    dump(old/"manifest.json", {"source_dir": str(physical),
        "source_files": {"manifest.json": sha(physical/"manifest.json")},
        "input_files": {"panel.parquet": sha(old/"panel.parquet")}})
    dump(old/"results_manifest.json", {"status": "completed", "suite_manifest_sha256": sha(old/"manifest.json"),
        "result_files": {"predictions.parquet": sha(old/"predictions.parquet")}})
    new = base.assign(**{model: 95. for model in MODELS},
        **{model+q: 70. if q == "_q10" else 115. for model in MODELS for q in ("_q10", "_q90")})
    for strategy in ("control", "congestion"):
        for name, value in {"expert_ready": True, "raw_probability": .2, "spike_probability": .25,
                "calibration_status": "regularized_sparse_support", "calibration_window_days": 90,
                "threshold_eur_mwh": 50., "expert_fit_day": "2026-09-14", "proposal_reason": "test"}.items():
            new[strategy+"_"+name] = value
    return old, new


def test_assemble_twelve_models_preserves_source_and_old_constants(source):
    folder, predictions = source
    before = predictions.copy(deep=True)
    old_catalog = json.dumps(old_report.CATALOG)
    long, wide, _, seals = assemble_panel(predictions, folder)
    assert len(CATALOG) == 12 and long.model_id.nunique() == 11
    assert len(long) == len(predictions)*11
    assert wide.nyx_congestion.eq(93).all() and wide.network_fuel_direct.eq(92).all()
    assert wide.nuclear_kalman.eq(90).all() and wide.nyx_congestion_calibrated.eq(95).all()
    assert len(seals) == 9
    assert json.dumps(old_report.CATALOG) == old_catalog
    pd.testing.assert_frame_equal(predictions, before)


@pytest.mark.parametrize("name", ["actual", "forecast", "benchmark_forecast", "q10", "feature_fixed"])
def test_changed_original_panel_rejected(source, name):
    folder, predictions = source
    predictions.loc[0, name] = -99.
    with pytest.raises(ValueError, match="Changed"):
        assemble_panel(predictions, folder)


@pytest.mark.parametrize("damage", ["duplicate", "naive", "infinite", "probability", "unordered", "missing", "comparator"])
def test_invalid_comparison_rejected(source, damage):
    folder, predictions = source
    if damage == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[[0]]])
    elif damage == "naive":
        predictions["forecast_origin_utc"] = predictions.forecast_origin_utc.dt.tz_localize(None)
    elif damage == "infinite":
        predictions.loc[0, MODELS[0]] = np.inf
    elif damage == "probability":
        predictions.loc[0, "control_raw_probability"] = 1.01
    elif damage == "unordered":
        predictions.loc[0, MODELS[0]+"_q10"] = 120.
    elif damage == "missing":
        predictions = predictions.drop(columns="feature_fixed")
    else:
        predictions["nyx_congestion"] = 999.
    with pytest.raises(ValueError):
        assemble_panel(predictions, folder)


@pytest.mark.parametrize("name", ["predictions.parquet", "panel.parquet"])
def test_old_source_checksum_required(source, name):
    folder, predictions = source
    with (folder/name).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        assemble_panel(predictions, folder)


def test_common_support_readiness_and_live_exclusion(source):
    folder, predictions = source
    predictions.loc[0, MODELS[0]] = np.nan
    _, wide, _, _ = assemble_panel(predictions, folder)
    annual = diagnostics(wide, end_day="2026-09-14", days=365, zones=["FR"])
    assert {r["n_hours"] for r in annual["interventions"]} == {119}
    assert {r["n_common_country_hours"] for r in annual["readiness"]} == {119}
    assert {r["ready_country_hours"] for r in annual["readiness"]} == {119}
    recent = diagnostics(wide, end_day="2026-09-14", days=7, zones=["FR"])
    assert {r["n_hours"] for r in recent["interventions"]} == {48}


def test_oos_probabilities_common_support_and_scores(source):
    folder, predictions = source
    _, wide, _, _ = assemble_panel(predictions, folder)
    wide.loc[0, "control_expert_ready"] = False
    wide.loc[1, "congestion_spike_probability"] = np.nan
    result = probability_diagnostics(wide, end_day="2026-09-14", days=365, zones=["FR"])
    assert result["available"] and {r["n_country_hours"] for r in result["rows"]} == {118}
    for row in result["rows"]:
        assert row["n_events"] == 0
        assert row["raw_brier"] == pytest.approx(.04)
        assert row["calibrated_brier"] == pytest.approx(.0625)
        assert row["raw_log_loss"] == pytest.approx(-np.log(.8))
        assert row["calibrated_log_loss"] == pytest.approx(-np.log(.75))
        assert row["sparse_calibration_hours"] == 118
    for s in ("control", "congestion"):
        for v in ("raw", "calibrated"):
            bins = [r for r in result["reliability"] if r["zone"] == "FR" and r["strategy"] == s and r["variant"] == v]
            assert len(bins) == 10 and sum(r["n_country_hours"] for r in bins) == 118
            assert bins[2]["observed_frequency"] == 0
            assert bins[0]["mean_probability"] is None
    recent = probability_diagnostics(wide, end_day="2026-09-14", days=7, zones=["FR"])
    assert {r["n_country_hours"] for r in recent["rows"]} == {48}


def test_oos_event_definition_bin_endpoints_and_threshold_contract(source):
    folder, predictions = source
    _, wide, _, _ = assemble_panel(predictions, folder)
    wide.loc[0, "actual"] = 140.  # exactly NYX90 + CORE threshold50
    wide.loc[0, "control_raw_probability"] = 1.
    result = probability_diagnostics(wide, end_day="2026-09-14", days=365, zones=["FR"])
    assert {r["n_events"] for r in result["rows"]} == {1}
    last = next(r for r in result["reliability"] if r["zone"] == "FR" and r["strategy"] == "control"
                and r["variant"] == "raw" and r["bin_upper"] == 1)
    assert last["n_country_hours"] == last["n_events"] == 1
    assert last["mean_probability"] == last["observed_frequency"] == 1
    wide.loc[1, "congestion_threshold_eur_mwh"] = 51.
    with pytest.raises(ValueError, match="identical"):
        probability_diagnostics(wide, end_day="2026-09-14", days=365, zones=["FR"])


def test_oos_probabilities_no_ready_support_is_unknown(source):
    folder, predictions = source
    _, wide, _, _ = assemble_panel(predictions, folder)
    wide["congestion_expert_ready"] = False
    result = probability_diagnostics(wide, end_day="2026-09-14", days=365, zones=["FR"])
    assert all(r["n_country_hours"] == 0 and r["calibrated_brier"] is None for r in result["rows"])


@pytest.fixture
def folds(tmp_path):
    directory = tmp_path/"runs/experiments/nyx_congestion_calibration_v1/snapshots/test"
    directory.mkdir(parents=True)
    dump(directory/"manifest.json", {"version": "test"})
    frame = pd.DataFrame([{"strategy": strategy, "fit_day": day, "status": status, "reason": reason}
        for strategy in ("control", "congestion")
        for day, status, reason in (("2026-06-01", "trained", ""), ("2026-09-14", "fallback", "sparse_test"),
                                   ("2026-09-21", "trained", ""))])
    frame.to_parquet(directory/"folds.parquet", index=False)
    dump(directory/"results_manifest.json", {"status": "completed", "suite_manifest_sha256": sha(directory/"manifest.json"),
        "result_files": {"folds.parquet": sha(directory/"folds.parquet")}})
    return directory, {"snapshot": str(directory), "suite_manifest_sha256": sha(directory/"manifest.json")}


def test_fold_latest_attempt_not_latest_success(folds, tmp_path):
    _, audit = folds
    seals = {}
    frame = read_folds(audit, root=tmp_path, seals=seals)
    result = fold_summary(frame, end_day="2026-09-14", days=365)
    assert len(seals) == 3
    assert all(r["fit_day"] == "2026-09-14" and r["reason"] == "sparse_test" for r in result["case"])
    assert all(r["n_attempts"] == 2 and r["n_trained"] == 1 for r in result["rows"])
    assert all(r["last_trained_fit"] == "2026-06-01" for r in result["rows"])
    recent = fold_summary(frame, end_day="2026-09-14", days=7)
    assert all(r["n_attempts"] == 1 and r["n_trained"] == 0 for r in recent["rows"])


@pytest.mark.parametrize("tamper", ["folds", "suite", "result"])
def test_fold_seals_reject_tampering(folds, tmp_path, tamper):
    directory, audit = folds
    if tamper == "folds":
        with (directory/"folds.parquet").open("ab") as handle:
            handle.write(b"tamper")
    elif tamper == "suite":
        audit["suite_manifest_sha256"] = "bad"
    else:
        result = json.loads((directory/"results_manifest.json").read_text(encoding="utf8"))
        result["suite_manifest_sha256"] = "bad"
        dump(directory/"results_manifest.json", result)
    with pytest.raises(ValueError):
        read_folds(audit, root=tmp_path, seals={})


def test_html_is_offline_safe_and_explicit_about_split(tmp_path):
    destination = tmp_path/"test.html"
    render_report({"unsafe": "</script><script>alert(1)</script>", "nan": np.nan}, destination)
    text = destination.read_text(encoding="utf8")
    assert "</script><script>alert" not in text and "\\u003c/script\\u003e" in text
    assert "<script src=" not in text and "cdn." not in text
    assert "Mode nuit" in text and "pas de tous les composants" in text
    assert "scikit-learn.org/stable/modules/calibration.html" in text
    assert "douze modèles" in text and "non exécutable" in text
    with pytest.raises(FileExistsError):
        render_report({}, destination)


def test_full_report_kpi_eva_cases_and_fixed_methods(source, tmp_path):
    source_dir, predictions = source
    root = tmp_path/"root"
    (root/"config").mkdir(parents=True)
    shutil.copyfile(ROOT/"config/economic_value.yaml", root/"config/economic_value.yaml")
    dest = root/"runs/experiments/nyx_congestion_calibration_v1/reports/test"
    result = build_report(predictions, source_dir, dest, root=root, audit={"stage1_retrained": False})
    payload = json.loads(Path(result["metrics_path"]).read_text(encoding="utf8"))
    assert len(payload["catalog"]) == 12
    assert {r["n_hours"] for r in payload["periods"]["365"]["rows"]} == {120}
    assert {r["n_hours"] for r in payload["periods"]["7"]["rows"]} == {48}
    assert {r["n_hours"] for r in payload["june"]["rows"]} == {72}
    assert len(payload["case"]) == 96
    assert set(r["day"] for r in payload["case"]) == {"2026-06-24", "2026-06-25", "2026-06-26", "2026-09-14"}
    assert payload["periods"]["365"]["economic"]["audit"]["zone_capacity_mw"]["FR"] == 25
    assert payload["method"]["calibration_calendar_days"] == 90
    assert payload["method"]["severity_exclusion_days"] == 28
    assert {r["n_country_hours"] for r in payload["periods"]["365"]["probability"]["rows"]} == {120}
    assert {r["n_country_hours"] for r in payload["periods"]["7"]["probability"]["rows"]} == {48}
    assert payload["production_modified"] is False and payload["independent_validation"] is False
    with pytest.raises(FileExistsError):
        build_report(predictions, source_dir, dest, root=root, audit={})


def test_output_isolation(source, tmp_path):
    source_dir, predictions = source
    with pytest.raises(ValueError):
        build_report(predictions, source_dir, tmp_path/"prod", root=tmp_path, audit={})
