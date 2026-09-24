from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
import pytest

from nyx_fundamental_stress import reporting as report
from test_nyx_scarcity_zonal_reporting import fixture as frozen_archive_fixture


def fixture(tmp_path, monkeypatch):
    frame, audit = frozen_archive_fixture(tmp_path)
    frame["feature_fundamental_residual_load_gw"] = 40.
    frame["feature_fundamental_temperature_c"] = 23.
    frame["feature_nyx_price_lag_must_not_be_displayed"] = 999999.
    frame["predicted_signed_residual_median"] = 4.
    frame["physical_gate_passed"] = True
    frame["risk_probability_gate"] = .06
    frame["proposal_reason"] = "fundamental_proposal_ready"
    namespace = tmp_path/"fundamental_namespace"
    monkeypatch.setattr(report, "_NAMESPACE", namespace)
    return frame, audit, namespace/"snapshot/reports"


def test_real_production_report_fundamental_inputs_only_and_governed_policy(tmp_path, monkeypatch):
    frame, audit, output = fixture(tmp_path, monkeypatch)
    before, source_before = frame.copy(deep=True), deepcopy(audit)
    paths = report.render_fundamental_reports(frame, source_audit=audit, output_directory=output)
    text = paths["FR"].read_text(encoding="utf-8")
    for marker in ("fundamental-experiment", "EXPERT FONDAMENTAL", "Version principale gouvernée", "90 jours",
        "prévalence réelle", "porte physique", "28 jours de calibration", "TOUTES les observations", "400 EUR/MWh",
        "2026-09-15 visible", "STATISTICS — PRIX MOYENS", "storm-comparison", "hourly-comparison",
        "feature_fundamental_residual_load_gw", "Médiane d’erreur signée", "température lacunaire",
        "réseau JAO exclu", "cible supervisée", "pas le produit p × sévérité"):
        assert marker in text
    assert "feature_nyx_price_lag_must_not_be_displayed" not in text
    assert "gouverneur annuel strict" not in text
    assert "<td>4.000</td><td>0.700</td><td>0.060</td><td>Oui</td><td>fundamental_proposal_ready</td>" in text
    assert "p &gt; 0,6" not in text and "correction fixe de 25 %" not in text
    payload = json.loads(paths["audit"].read_text(encoding="utf-8"))["reports"]["FR"]
    assert payload["paired_hours"] == 8735
    assert payload["evaluation_end_day"] == "2026-09-14" and payload["live_excluded_from_statistics"]
    assert payload["strict_governor_enforced"] is True
    assert payload["model_feature_column_count"] == payload["displayed_feature_count"] == 2
    assert set(payload["allowed_input_features"]) == {"feature_fundamental_residual_load_gw", "feature_fundamental_temperature_c"}
    assert payload["decision_checks"]["saved_probability_gate_checked"]
    assert payload["decision_checks"]["saved_physical_gate_checked"]
    assert "../fundamental_comparison.html" in paths["index"].read_text(encoding="utf-8")
    pd.testing.assert_frame_equal(frame, before); assert audit == source_before
    node = Path(r"C:\Users\BQ6757\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
    if node.is_file():
        scripts = re.findall(r'<script(?:\s+type="text/javascript")?>(.*?)</script>', text, re.S)
        checked = subprocess.run([str(node), "--check"], input="\n".join(scripts), text=True, encoding="utf-8", capture_output=True)
        assert checked.returncode == 0, checked.stderr


def test_fixed25_banner_is_distinct_and_optional_gate_columns_remain_missing(tmp_path, monkeypatch):
    frame, audit, output = fixture(tmp_path, monkeypatch)
    frame = frame.drop(columns=["predicted_signed_residual_median", "physical_gate_passed", "proposal_reason", "risk_probability_gate"])
    paths = report.render_fundamental_reports(frame, source_audit=audit, output_directory=output,
        decision_policy="fixed25", model_name="fundamental_stress_25")
    text = paths["FR"].read_text(encoding="utf-8")
    assert "poids fixe de 25 %" in text and "n’est PAS appliqué" in text
    assert "p &gt; 0,6" not in text
    assert "gouverneur annuel strict" not in text
    saved = json.loads(paths["audit"].read_text(encoding="utf-8"))["reports"]["FR"]
    assert saved["strict_governor_enforced"] is False
    assert saved["decision_checks"]["saved_probability_gate_checked"] is False
    assert saved["decision_checks"]["saved_physical_gate_checked"] is False


@pytest.mark.parametrize("column,value", [("selected_weight", .75), ("applied_correction", -1.),
    ("physical_gate_passed", False), ("risk_probability_gate", .7), ("risk_probability_gate", np.nan),
    ("bounded_correction", 401.)])
def test_inconsistent_decision_contract_rejected_before_output(tmp_path, monkeypatch, column, value):
    frame, audit, output = fixture(tmp_path, monkeypatch)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        report.render_fundamental_reports(frame, source_audit=audit, output_directory=output)
    assert not output.exists()


def test_probability_gate_is_prevalence_not_fixed_point6(tmp_path, monkeypatch):
    frame, _, _ = fixture(tmp_path, monkeypatch)
    frame["spike_probability"] = .08
    checked = report._validate_decisions(report._prepare(frame), "governed")
    assert checked["active_hours_including_live"] == len(frame)
    frame["spike_probability"] = .06
    with pytest.raises(ValueError, match="prevalence"):
        report._validate_decisions(report._prepare(frame), "governed")


def test_invalid_namespace_or_missing_fundamental_inputs_rejected(tmp_path, monkeypatch):
    frame, audit, output = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="namespace"):
        report.render_fundamental_reports(frame, source_audit=audit, output_directory=tmp_path/"not_fundamental")
    frame = frame.drop(columns=[c for c in frame if c.startswith("feature_fundamental_")])
    with pytest.raises(ValueError, match="fundamental input"):
        report.render_fundamental_reports(frame, source_audit=audit, output_directory=output)
    assert not output.exists()


@pytest.mark.parametrize("policy", ["bad", "fixed26", "both"])
def test_unknown_policy_rejected(policy, tmp_path):
    with pytest.raises(ValueError):
        report.render_fundamental_reports(pd.DataFrame(), source_audit={}, output_directory=tmp_path, decision_policy=policy)


def test_index_keeps_fundamental_and_fixed25_separate_without_inventing_harm_counts():
    score = {"hours": 8735, "mae_eur_mwh": 10.3, "rmse_eur_mwh": 18.4, "bias_eur_mwh": -.2,
             "daily_mean_mae_eur_mwh": 4.5, "mean_forecast_eur_mwh": 91., "mean_observed_eur_mwh": 90.}
    values = {"by_zone": {"FR": {"annual": {key: dict(score) for key in
        ("nuclear_kalman", "storm", "regional_25", "fundamental", "fundamental_25")},
        "interventions": {"fundamental": {"active_hours": 20, "worsened_absolute_error_hours": 3},
                          "fundamental_25": {"active_hours": 30, "worsened_absolute_error_hours": 5}}}}}
    before = deepcopy(values)
    text = report._index_comparison(values)
    for marker in ("Fondamental gouverné", "25 % — diagnostic sans gouverneur strict", "Storm figé", "régional précédent",
                   "10.300", "91.000", "90.000", "Corrections aggravantes", "sans promotion"):
        assert marker in text
    assert values == before
