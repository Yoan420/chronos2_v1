from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
import numpy as np
import pandas as pd
import pytest

from nyx_stress_guard import reporting
from test_nyx_scarcity_zonal_reporting import fixture as source_fixture


def fixtures(tmp_path):
    frame, audit = source_fixture(tmp_path)
    frame["feature_fundamental_test_pressure"] = 1.2
    frame["feature_nyx_price_must_not_be_an_input"] = 999999.
    frame["candidate_forecast"] = frame.forecast+20.
    frame["applied_correction"] = 20.
    frame["selected_weight"] = 1.
    frame["bounded_correction"] = 20.
    frame["candidate_q10"] = frame.q10-5.
    frame["candidate_q90"] = frame.q90+25.
    frame["interval_calibration_status"] = "calibrated"
    frame["intervention_active"] = True
    predictions = {key: frame.copy(deep=True) for key in reporting.LABELS if key not in {"nuclear_kalman", "storm"}}
    directory = tmp_path/"runs/experiments/nyx_scarcity_v1/stress_guard/snapshots/test"
    directory.mkdir(parents=True)
    return predictions, audit, directory


def test_common_interval_scores_exclude_missing_storm_and_live(tmp_path):
    predictions, _, _ = fixtures(tmp_path)
    frame = predictions["physics_governed"]
    before = frame.copy(deep=True)
    summary = reporting.calibration_summary(frame)
    assert summary["all"]["all"]["hours"] == 8735
    assert summary["FR"]["active"]["hours"] == 8735
    assert summary["FR"]["inactive"] == {"hours": 0}
    assert summary["all"]["all"]["coverage"] >= summary["all"]["all"]["nyx_same_hours_coverage"]
    assert summary["all"]["all"]["mean_width"] > summary["all"]["all"]["nyx_mean_width"]
    pd.testing.assert_frame_equal(frame, before, check_exact=True)


def test_interval_summary_uses_explicit_common_identities(tmp_path):
    predictions, _, _ = fixtures(tmp_path)
    frame = predictions["physics_governed"]
    full = reporting.calibration_summary(frame)["all"]["all"]["hours"]
    prepared = reporting._prepare(frame)
    eligible = prepared.in_evaluation_window & prepared.benchmark_forecast.notna()
    identities = pd.MultiIndex.from_frame(prepared.loc[eligible, ["zone", "timestamp_utc"]].iloc[1:])
    summary = reporting.calibration_summary(frame, common_keys=identities)
    assert summary["all"]["all"]["hours"] == full-1


def test_real_production_format_preserves_comparison_and_discloses_calibration(tmp_path, monkeypatch):
    predictions, audit, directory = fixtures(tmp_path)
    before = {key: value.copy(deep=True) for key,value in predictions.items()}
    source_before = deepcopy(audit)
    def comparison(values):
        return {"overall": {"annual": {key: {"mae_eur_mwh": 20.} for key in reporting.LABELS}}}
    def render(values, summary, provenance, path):
        path.write_text("<html><main></main></html>", encoding="utf-8")
    monkeypatch.setattr(reporting, "build_comparison", comparison)
    monkeypatch.setattr(reporting, "render_comparison", render)
    paths = reporting.render_reports(predictions, source_audit=audit, model_audit={}, output_directory=directory, root=tmp_path)
    proof = json.loads(paths["audit"].read_text(encoding="utf-8"))
    intervals = json.loads(paths["interval_comparison"].read_text(encoding="utf-8"))
    nyx = intervals["nuclear_kalman"]["all"]["all"]
    assert nyx["hours"] == 8735
    assert nyx["coverage"] == nyx["nyx_same_hours_coverage"]
    assert nyx["mean_width"] == nyx["nyx_mean_width"]
    for variant in ("physics_direct", "physics_governed"):
        data = proof[variant+"_FR"]
        assert data["paired_hours"] == 8735
        assert data["evaluation_days"] == 365 and data["live_excluded_from_statistics"]
        assert data["strict_governor_enforced"] == (variant == "physics_governed")
        document = paths[variant+"_FR"].read_text(encoding="utf-8")
        for marker in ("STRESS GUARD", "INTERVALLES — CALIBRATION CHRONOLOGIQUE", "STATISTICS — PRIX MOYENS",
            "storm-comparison", "hourly-comparison", "AUCUNE ACTIVATION", "sans porte rigide", "repli régional",
            "80 % n’est pas garanti", "Score intervalle80", "feature_fundamental_test_pressure"):
            assert marker in document
        assert "feature_nyx_price_must_not_be_an_input" not in document
        node = Path(r"C:\Users\BQ6757\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
        if node.is_file():
            scripts = re.findall(r'<script(?:\s+type="text/javascript")?>(.*?)</script>', document, re.S)
            checked = subprocess.run([str(node), "--check"], input="\n".join(scripts), text=True, encoding="utf-8", capture_output=True)
            assert checked.returncode == 0, checked.stderr
    assert "StressGuard — gouverné (principal)" in paths["index"].read_text(encoding="utf-8")
    for key in predictions:
        pd.testing.assert_frame_equal(predictions[key], before[key], check_exact=True)
    assert audit == source_before


@pytest.mark.parametrize("column,value", [("candidate_q10", 101.), ("candidate_q90", 110.), ("candidate_forecast", 121.)])
def test_invalid_saved_envelope_refused_before_reports(tmp_path, column, value):
    predictions, audit, directory = fixtures(tmp_path)
    predictions["physics_direct"].loc[0, column] = value
    with pytest.raises(ValueError):
        reporting.render_reports(predictions, source_audit=audit, model_audit={}, output_directory=directory, root=tmp_path)
    assert not list(directory.rglob("*.html"))


def test_interval_ablation_point_identity_enforced(tmp_path):
    predictions, audit, directory = fixtures(tmp_path)
    predictions["p50_calibrated"].loc[0, "candidate_forecast"] += 1.
    predictions["p50_calibrated"].loc[0, "applied_correction"] += 1.
    predictions["p50_calibrated"].loc[0, "bounded_correction"] += 1.
    with pytest.raises(ValueError, match="Interval-only"):
        reporting.render_reports(predictions, source_audit=audit, model_audit={}, output_directory=directory, root=tmp_path)


def test_wrong_output_namespace_rejected(tmp_path):
    predictions, audit, _ = fixtures(tmp_path)
    with pytest.raises(ValueError):
        reporting.render_reports(predictions, source_audit=audit, model_audit={}, output_directory=tmp_path/"runs/exports", root=tmp_path)
