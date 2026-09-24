from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
import pytest

from nyx_coherent_p50 import reporting as report
from test_nyx_scarcity_zonal_reporting import fixture as archive_fixture


def fixture(tmp_path, monkeypatch, policy="direct"):
    frame, audit = archive_fixture(tmp_path)
    frame["feature_fundamental_residual_load_gw"] = 40.
    frame["feature_fundamental_temperature_c"] = 23.
    frame["feature_nyx_price_lag_must_not_be_displayed"] = 999999.
    frame["mixture_error_q10"] = -10.
    frame["mixture_error_q50"] = 80.
    frame["mixture_error_q90"] = 800.
    frame["mixture_raw_p50_eur_mwh"] = 180.
    frame["threshold_eur_mwh"] = 50.
    frame["risk_probability_gate"] = .06
    frame["physical_gate_passed"] = True
    frame["proposal_reason"] = "coherent_positive_median_proposal"
    frame["raw_correction"] = 80.
    frame["bounded_correction"] = 80.
    weight = 1. if policy == "direct" else .25
    frame["selected_weight"] = weight
    frame["applied_correction"] = weight*80.
    frame["candidate_forecast"] = 100.+weight*80.
    frame["candidate_q10"] = 90.
    frame["candidate_q90"] = (1.-weight)*120.+weight*500.
    namespace = tmp_path/"coherent_p50_namespace"
    monkeypatch.setattr(report, "_NAMESPACE", namespace)
    return frame, audit, namespace/"snapshot/reports"


@pytest.mark.parametrize("policy", ["direct", "governed"])
def test_real_production_report_cdf_contract_inputs_and_frozen_window(tmp_path, monkeypatch, policy):
    frame, audit, output = fixture(tmp_path, monkeypatch, policy)
    before, source_before = frame.copy(deep=True), deepcopy(audit)
    paths = report.render_p50_reports(frame, source_audit=audit, output_directory=output, decision_policy=policy)
    document = paths["FR"].read_text(encoding="utf-8")
    for marker in ("coherent-p50-experiment", "P50 COHÉRENT", "AUCUNE ACTIVATION", "400 EUR/MWh",
        "2026-09-15 visible séparément", "STATISTICS — PRIX MOYENS", "storm-comparison", "hourly-comparison",
        "feature_fundamental_residual_load_gw", "Erreur Q50 brute", "température lacunaire", "réseau JAO exclu",
        "Qfinal(a)", "PAS un mélange des CDF", "ne décrit pas automatiquement", "14 septembre à 19 h",
        "si p &gt; 50 %", "ni p × sévérité", "masse artificielle à zéro"):
        assert marker in document
    assert "feature_nyx_price_lag_must_not_be_displayed" not in document
    assert "Modèle de médiane HGB" not in document
    assert "<td>50.000</td><td>-10.000</td><td>80.000</td><td>800.000</td><td>180.000</td>" in document
    assert "p &gt; 0,6" not in document
    assert ("n’est PAS appliqué" in document) == (policy == "direct")
    proof = json.loads(paths["audit"].read_text(encoding="utf-8"))["reports"]["FR"]
    assert proof["paired_hours"] == 8735
    assert proof["evaluation_end_day"] == "2026-09-14" and proof["live_excluded_from_statistics"]
    assert proof["strict_governor_enforced"] == (policy == "governed")
    assert proof["model_feature_column_count"] == proof["displayed_feature_count"] == 2
    assert proof["decision_checks"]["raw_median_spike_support_checked"]
    assert proof["decision_checks"]["quantile_function_interpolation_checked"]
    assert proof["decision_checks"]["classifier_probability_is_final_decision_distribution_probability"] is False
    assert "../coherent_p50_comparison.html" in paths["index"].read_text(encoding="utf-8")
    pd.testing.assert_frame_equal(frame, before); assert source_before == audit
    node = Path(r"C:\Users\BQ6757\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
    if node.is_file():
        scripts = re.findall(r'<script(?:\s+type="text/javascript")?>(.*?)</script>', document, re.S)
        checked = subprocess.run([str(node), "--check"], input="\n".join(scripts), text=True, encoding="utf-8", capture_output=True)
        assert checked.returncode == 0, checked.stderr


@pytest.mark.parametrize("column,value", [("selected_weight", .75), ("applied_correction", -1.),
    ("physical_gate_passed", False), ("risk_probability_gate", .7), ("risk_probability_gate", np.nan),
    ("bounded_correction", 401.), ("mixture_error_q50", 20.), ("mixture_error_q10", 90.),
    ("mixture_error_q90", np.nan), ("mixture_raw_p50_eur_mwh", 181.), ("candidate_q90", 501.),
    ("raw_correction", 79.), ("candidate_q10", 91.)])
def test_invalid_saved_distribution_or_decision_fails_before_writing(tmp_path, monkeypatch, column, value):
    frame, audit, output = fixture(tmp_path, monkeypatch)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        report.render_p50_reports(frame, source_audit=audit, output_directory=output)
    assert not output.exists()


def test_raw_distribution_above_half_enforces_spike_support(tmp_path, monkeypatch):
    frame, _, _ = fixture(tmp_path, monkeypatch)
    frame["mixture_error_q50"] = 49.
    frame["mixture_raw_p50_eur_mwh"] = 149.
    with pytest.raises(ValueError, match="spike support"):
        report._validate_decisions(report._prepare(frame), "direct")


def test_positive_ordinary_median_can_intervene_below_half_without_threshold_tuning(tmp_path, monkeypatch):
    frame, _, _ = fixture(tmp_path, monkeypatch)
    frame["spike_probability"] = .2
    frame["mixture_error_q50"] = 20.
    frame["mixture_raw_p50_eur_mwh"] = 120.
    frame["raw_correction"] = frame["bounded_correction"] = frame["applied_correction"] = 20.
    frame["candidate_forecast"] = 120.
    checked = report._validate_decisions(report._prepare(frame), "direct")
    assert checked["active_hours_including_live"] == len(frame)
    assert checked["probability_above_half_hours"] == 0


def test_zero_weight_preserves_baseline_quantiles_even_when_raw_median_is_high(tmp_path, monkeypatch):
    frame, _, _ = fixture(tmp_path, monkeypatch, "governed")
    frame["selected_weight"] = frame["applied_correction"] = 0.
    frame["candidate_forecast"] = frame.forecast
    frame["candidate_q10"] = frame.q10
    frame["candidate_q90"] = frame.q90
    checked = report._validate_decisions(report._prepare(frame), "governed")
    assert checked["active_hours_including_live"] == 0
    frame.loc[0, "candidate_q90"] += 1.
    with pytest.raises(ValueError, match="zero weight"):
        report._validate_decisions(report._prepare(frame), "governed")


def test_warmup_distribution_can_remain_missing_without_inventing_quantiles(tmp_path, monkeypatch):
    frame, _, _ = fixture(tmp_path, monkeypatch)
    frame["expert_ready"] = False
    frame[["mixture_error_q10", "mixture_error_q50", "mixture_error_q90", "mixture_raw_p50_eur_mwh"]] = np.nan
    frame[["raw_correction", "bounded_correction", "selected_weight", "applied_correction"]] = 0.
    frame["candidate_forecast"], frame["candidate_q10"], frame["candidate_q90"] = frame.forecast, frame.q10, frame.q90
    checked = report._validate_decisions(report._prepare(frame), "direct")
    assert checked["complete_raw_distribution_hours"] == 0
    frame.loc[0, "expert_ready"] = True
    with pytest.raises(ValueError, match="ready CDF expert"):
        report._validate_decisions(report._prepare(frame), "direct")


def test_no_fundamental_inputs_or_wrong_namespace_rejected(tmp_path, monkeypatch):
    frame, audit, output = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="namespace"):
        report.render_p50_reports(frame, source_audit=audit, output_directory=tmp_path/"not_p50")
    with pytest.raises(ValueError, match="parent"):
        report.render_p50_reports(frame, source_audit=audit, output_directory=output/".."/"other")
    frame = frame.drop(columns=[c for c in frame if c.startswith("feature_fundamental_")])
    with pytest.raises(ValueError, match="fundamental input"):
        report.render_p50_reports(frame, source_audit=audit, output_directory=output)
    assert not output.exists()


@pytest.mark.parametrize("policy", ["fixed25", "bad", "both"])
def test_unknown_policy_rejected(policy, tmp_path):
    with pytest.raises(ValueError, match="decision_policy"):
        report.render_p50_reports(pd.DataFrame(), source_audit={}, output_directory=tmp_path, decision_policy=policy)


def test_index_distinguishes_models_and_policies_without_inventing_harm_counts():
    score = {"hours": 8735, "mae_eur_mwh": 10.3, "rmse_eur_mwh": 18.4, "bias_eur_mwh": -.2,
        "daily_mean_mae_eur_mwh": 4.5, "mean_forecast_eur_mwh": 91., "mean_observed_eur_mwh": 90.}
    models = ("nuclear_kalman", "storm", "regional_25", "fundamental_old", "forest", "forest_governed", "empirical", "empirical_governed")
    value = {"by_zone": {"FR": {"annual": {key: dict(score) for key in models},
        "interventions": {"forest": {"active_hours": 20, "worsened_absolute_error_hours": 3},
                          "forest_governed": {"active_hours": 2, "worsened_absolute_error_hours": 1}}}}}
    before = deepcopy(value)
    text = report._index_comparison(value)
    for marker in ("Fondamental précédent", "forêt conditionnelle — direct principal", "forêt conditionnelle — gouverné",
        "empirique — direct témoin", "empirique — gouverné témoin", "Storm figé", "10.300", "Corrections aggravantes", "sans promotion"):
        assert marker in text
    assert before == value
