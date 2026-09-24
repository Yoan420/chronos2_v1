"""Frozen common masks, distinct native targets and compact variant HTML."""
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from nyx_scarcity import variant_reporting as report


def fixture(days=3, end="2026-09-14", zones=("FR",), live=True):
    blocks = []
    for zone in zones:
        day = pd.Timestamp(end)
        tz = report.TIMEZONES[zone]
        index = pd.date_range((day-pd.Timedelta(days=days-1)).tz_localize(tz),
                              (day+pd.Timedelta(days=1+int(live))).tz_localize(tz),
                              freq="h", inclusive="left").tz_convert("UTC")
        labels = np.array(["evaluation" if t.tz_convert(tz).date() <= day.date() else "live" for t in index])
        actual = 100.+np.arange(len(index)) % 10
        actual[labels == "live"] = np.nan
        blocks.append(pd.DataFrame({"zone": zone, "timestamp_utc": index,
            "forecast_origin_utc": index.normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=6),
            "actual": actual, "forecast": 90., "benchmark_forecast": 95., "candidate_forecast": 98.,
            "q10": 80., "q90": 200., "candidate_q10": 88., "candidate_q90": 208.,
            "spike_probability": .2, "threshold_eur_mwh": 50., "probability_gate": .6,
            "raw_correction": 20., "bounded_correction": 15., "applied_correction": 8.,
            "selected_weight": .5, "expert_ready": True, "gate_reason": "accepted",
            "interval_status": "calibrated", "sample": labels}))
    hgb = pd.concat(blocks, ignore_index=True)
    new = hgb.copy(deep=True)
    new["candidate_forecast"] = 100.
    new["applied_correction"] = 10.
    return {"hgb_v1": hgb, "xgb_unweighted_fixed": new}


def test_all_models_use_same_annual_support_and_frozen_sources():
    predictions = fixture(days=365, zones=("FR", "DE"))
    original = {key: frame.copy(deep=True) for key, frame in predictions.items()}
    summary = report.build_comparison(predictions)
    for zone in ("FR", "DE"):
        assert summary["windows"][zone]["complete_365_common_support"] is True
        annual = summary["by_zone"][zone]["annual"]
        assert set(annual) == {"nuclear_kalman", "storm", *predictions}
        assert {score["hours"] for score in annual.values()} == {8760}
        assert annual["xgb_unweighted_fixed"]["gain_vs_hgb_mae_eur_mwh"] == pytest.approx(2.)
        assert annual["xgb_unweighted_fixed"]["mean_observed_eur_mwh"] == annual["storm"]["mean_observed_eur_mwh"]
        assert annual["nuclear_kalman"]["mean_forecast_eur_mwh"] == 90.
    assert summary["live_rows_excluded"] == 48
    assert len(summary["daily"]) == 365*2*4
    for key in predictions:
        pd.testing.assert_frame_equal(predictions[key], original[key])


@pytest.mark.parametrize("defect", ["missing_candidate", "missing_storm", "missing_actual", "missing_row"])
def test_missing_one_variant_excludes_hour_from_every_comparison(defect):
    predictions = fixture(days=1, live=False)
    key = "xgb_unweighted_fixed"
    if defect == "missing_candidate": predictions[key].loc[0, "candidate_forecast"] = np.nan
    elif defect == "missing_storm": predictions[key].loc[0, "benchmark_forecast"] = np.nan
    elif defect == "missing_actual": predictions[key].loc[0, "actual"] = np.nan
    else: predictions[key] = predictions[key].iloc[1:].copy()
    summary = report.build_comparison(predictions)
    assert summary["overall"]["paired_hours"] == 23
    assert summary["overall"]["unpaired_window_hours"] == 1
    assert {r["hours"] for r in summary["overall"]["annual"].values()} == {23}
    assert {r["paired_hours"] for r in summary["daily"]} == {23}


@pytest.mark.parametrize("field", ["actual", "forecast", "benchmark_forecast", "q10", "q90", "forecast_origin_utc", "sample"])
def test_source_divergence_is_never_silently_compared(field):
    predictions = fixture(days=1, live=False)
    frame = predictions["xgb_unweighted_fixed"]
    if field == "sample": frame.loc[0, field] = "live"
    elif field == "forecast_origin_utc": frame.loc[0, field] += pd.Timedelta(hours=1)
    else: frame.loc[0, field] += 1.
    with pytest.raises(ValueError, match="frozen source field"):
        report.build_comparison(predictions)


def test_classifier_native_targets_are_separate_from_common_ranking():
    predictions = fixture(days=1, live=False)
    for frame in predictions.values():
        frame["actual"] = 90. + np.arange(24)*5.
        frame["spike_probability"] = np.linspace(.05, .95, 24)
    predictions["xgb_unweighted_fixed"]["threshold_eur_mwh"] = np.where(np.arange(24)%2, 30., 70.)
    predictions["hgb_v1"].loc[0, "expert_ready"] = False
    predictions["xgb_unweighted_fixed"].loc[1, "expert_ready"] = False
    summary = report.build_comparison(predictions)
    common = summary["overall"]["classification_common_ranking"]
    native = summary["overall"]["classification_native"]
    assert common["hours"] == 22 and common["brier_comparison_provided"] is False
    events = np.arange(24)[2:]*5. >= 50.
    probabilities = np.linspace(.05, .95, 24)[2:]
    for variant in common["variants"].values():
        assert variant["hours"] == 22
        assert variant["average_precision"] == pytest.approx(average_precision_score(events, probabilities))
        assert variant["roc_auc"] == pytest.approx(roc_auc_score(events, probabilities))
        assert "brier" not in variant and "precision" not in variant
    assert common["variants"]["hgb_v1"]["native_target_matches_common_on_all_rows"] is True
    assert common["variants"]["xgb_unweighted_fixed"]["native_target_matches_common_on_all_rows"] is False
    for variant in native.values():
        assert variant["hours"] == 23
        assert variant["cross_variant_calibration_comparable"] is False
        assert variant["brier"] is not None
    assert native["hgb_v1"]["threshold_min"] == 50.
    assert native["xgb_unweighted_fixed"]["threshold_min"] == 30.
    assert native["xgb_unweighted_fixed"]["threshold_max"] == 70.


def test_native_gate_strict_and_false_positive_rate():
    predictions = fixture(days=1, live=False)
    for frame in predictions.values():
        frame["actual"] = 90.
        frame.loc[0:1, "actual"] = 140.
        frame["spike_probability"] = .1
        frame.loc[0, "spike_probability"] = .6  # Strictly greater, so FN.
        frame.loc[1:2, "spike_probability"] = .8  # TP, then FP.
    result = report.build_comparison(predictions)["overall"]["classification_native"]["hgb_v1"]
    assert result["true_positive"] == result["false_positive"] == result["false_negative"] == 1
    assert result["true_negative"] == 21
    assert result["false_positive_rate"] == pytest.approx(1/22)
    assert result["precision"] == result["recall"] == .5


def test_no_positive_events_does_not_invent_ap_roc_or_recall():
    summary = report.build_comparison(fixture(days=1, live=False))
    common = summary["overall"]["classification_common_ranking"]["variants"]["hgb_v1"]
    assert common["average_precision"] is None and common["roc_auc"] is None
    native = summary["overall"]["classification_native"]["hgb_v1"]
    assert native["recall"] is None and native["brier"] is not None


def test_fallback_and_false_interventions_are_not_confused_with_harm():
    predictions = fixture(days=1, live=False)
    new = predictions["xgb_unweighted_fixed"]
    new.loc[:11, "candidate_forecast"] = 90.
    new.loc[:11, "applied_correction"] = 0.
    new.loc[:11, "expert_ready"] = False
    summary = report.build_comparison(predictions)["overall"]
    coverage = summary["trained_coverage"]["xgb_unweighted_fixed"]
    assert coverage["baseline_fallback_hours_in_common_support"] == 12
    assert coverage["ready_hours_on_common_point_support"] == 12
    active = summary["interventions"]["xgb_unweighted_fixed"]
    assert active["active_hours"] == 12
    assert active["non_event_interventions_common_error50"] == 12
    assert active["worsened_absolute_error_hours"] == 0
    assert summary["annual"]["xgb_unweighted_fixed"]["hours"] == 24
    assert active["all_models_on_this_variant_active_subset"]["hgb_v1"]["hours"] == 12


def test_tails_are_identical_ex_post_cohorts_for_all_models():
    predictions = fixture(days=5, live=False)
    for frame in predictions.values(): frame["actual"] = np.arange(120, dtype=float)
    summary = report.build_comparison(predictions)["overall"]
    tail = summary["tails"]["top_1_percent"]
    assert tail["used_for_training_or_governance"] is False
    assert tail["threshold_eur_mwh"] == pytest.approx(np.quantile(np.arange(120), .99))
    assert {r["hours"] for r in tail["scores"].values()} == {2}


def test_live_never_scored_and_dst_physical_hours_preserved():
    predictions = fixture(days=370, end="2026-10-25")
    for frame in predictions.values(): frame.loc[frame["sample"].eq("live"), "actual"] = 1e6
    summary = report.build_comparison(predictions)
    assert summary["windows"]["FR"]["represented_days"] == 365
    assert summary["windows"]["FR"]["complete_365_common_support"] is True
    assert summary["live_rows_excluded"] == 24
    rows = [r for r in summary["daily"] if r["local_day"] == "2026-10-25"]
    assert {r["paired_hours"] for r in rows} == {25}
    assert summary["by_zone"]["FR"]["annual"]["storm"]["mean_observed_eur_mwh"] < 200.


def test_hgb_frozen_control_required():
    with pytest.raises(ValueError, match="hgb_v1"):
        report.build_comparison({"only_new": fixture()["hgb_v1"]})


@pytest.mark.parametrize("status,verified,shown", [("complete", True, True), ("partial", True, True),
                                                  ("complete", False, False), ("unavailable", True, False)])
def test_shap_numbers_require_verified_reconstruction(status, verified, shown):
    value = {"status": status, "method": "TreeSHAP exact", "explained_output": "calibrated log-odds",
             "audit": {"reconstruction_verified": verified}, "global_importance": [{"feature": "wind", "mean_abs_shap_calibrated": .2}],
             "local_cases": [{"zone": "FR"}], "raw_feature_matrix": ["never embed this"]}
    compact = report._compact_explanations({"xgb_unweighted_fixed": value})["xgb_unweighted_fixed"]
    assert compact["numeric_display_verified"] == shown
    assert ("global_importance" in compact) == shown
    assert "raw_feature_matrix" not in compact


def test_runner_shap_wrapper_never_embeds_long_tables():
    wrapped = {"xgb_unweighted_fixed": {"summary": {"status": "partial", "method": "TreeSHAP exact",
        "explained_output": "calibrated log-odds", "audit": {"reconstruction_verified": True},
        "global_importance": [{"feature": "wind", "mean_abs_shap_calibrated": .2}]},
        "values": pd.DataFrame({"raw_long_shap_table": [1., 2.]}),
        "observations": pd.DataFrame({"raw_long_observations": [1., 2.]})}}
    compact = report._compact_explanations(wrapped)["xgb_unweighted_fixed"]
    assert compact["numeric_display_verified"] is True
    assert compact["global_importance"][0]["feature"] == "wind"
    assert "values" not in compact and "observations" not in compact


def test_local_waterfall_preserves_signed_identity_and_calibrated_probability():
    values = np.array([(-1 if i % 2 else 1) * (.3 + i/20) for i in range(15)])
    base = -.75
    margin = base + values.sum()
    case = {"zone": "FR", "timestamp_utc": "2026-09-14T17:00:00+00:00",
            "base_value_calibrated": base, "calibrated_margin": margin,
            "prob_calibrated": 1/(1+np.exp(-margin)),
            "contributions": [{"feature": f"feature_{i}", "feature_value": i,
                               "shap_value_calibrated": float(value)} for i, value in enumerate(values)]}
    waterfall = report._local_waterfall(case)
    assert len(waterfall["displayed_contributions"]) == 11
    assert waterfall["displayed_contributions"][-1]["feature"] == "Autres (5 variables)"
    assert waterfall["reconstructed_log_odds"] == pytest.approx(margin)
    assert waterfall["probability_calibrated"] == pytest.approx(case["prob_calibrated"])
    assert waterfall["local_label"] == "2026-09-14 19:00 +0200"
    assert waterfall["all_contributions_retained_in_sum"] is True
    for key in ("calibrated_margin", "prob_calibrated"):
        broken = deepcopy(case)
        broken[key] += .1
        with pytest.raises(ValueError, match="reconstruct"):
            report._local_waterfall(broken)


def test_global_shap_layout_reserves_space_for_long_feature_names():
    assert "automargin:true" in report._SCRIPT
    assert "height:id==='shap'?680" in report._SCRIPT


def test_html_compact_safe_and_offline(tmp_path):
    predictions = fixture(days=2)
    for key, frame in predictions.items():
        extra = pd.DataFrame({f"training_feature_{i:03}": np.arange(len(frame)) + i for i in range(250)})
        predictions[key] = pd.concat([frame, extra], axis=1)
    summary = report.build_comparison(predictions)
    before = deepcopy(summary)
    explanations = {"xgb_unweighted_fixed": {"status": "complete", "method": "TreeSHAP exact",
        "explained_output": "calibrated log-odds", "audit": {"reconstruction_verified": True},
        "global_importance": [{"feature": '</script><script>bad()</script>', "mean_abs_shap_calibrated": .2}]}}
    path = report.render_comparison(predictions, summary, {"note": "</pre><script>bad()</script>"},
                                     tmp_path/"comparison.html", explanations=explanations)
    document = path.read_text(encoding="utf-8")
    payload_text = re.search(r'<script id="variant-data" type="application/json">(.*?)</script>', document, re.S).group(1)
    payload = json.loads(payload_text)
    assert "training_feature_" not in document
    assert set(payload["base"][0]) == set(report._BASE_PAYLOAD)
    assert set(payload["variant_values"]["hgb_v1"]) == set(report._VARIANT_PAYLOAD)
    assert len(payload["base"]) == len(predictions["hgb_v1"])
    assert all(len(v) == len(payload["base"]) for v in payload["variant_values"]["hgb_v1"].values())
    assert payload["base"][-1]["actual"] is None
    assert "NaN" not in payload_text and "Infinity" not in payload_text
    assert '<script src=' not in document and '<script>bad()' not in document
    assert '&lt;script&gt;bad()' in document and '\\u003c/script\\u003e' in payload_text
    for marker in ('data-report-section="variant-statistics"', 'data-report-section="variant-daily"',
                   'data-report-section="variant-hourly"', 'data-report-section="variant-calendar"',
                   'data-report-section="variant-common-classification"', 'data-report-section="variant-native-classification"',
                   'data-report-section="variant-shap"', 'html[data-theme=dark]', 'nyx-variants-theme',
                   'pas un seuil absolu de prix', 'aucun Brier commun', 'pas du prix final'):
        assert marker in document
    assert summary == before


def test_browser_script_syntax():
    node = Path(r"C:\Users\BQ6757\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
    if not node.is_file(): pytest.skip("Bundled Node unavailable")
    done = subprocess.run([str(node), "--check"], input=report._SCRIPT, capture_output=True, encoding="utf-8")
    assert done.returncode == 0, done.stderr


def test_unified_browser_controls_execute_with_compact_arrays(tmp_path):
    node = Path(r"C:\Users\BQ6757\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
    if not node.is_file(): pytest.skip("Bundled Node unavailable")
    predictions = fixture(days=2, zones=("FR", "DE"))
    summary = report.build_comparison(predictions)
    path = report.render_comparison(predictions, summary, {}, tmp_path/"browser.html")
    payload = re.search(r'<script id="variant-data" type="application/json">(.*?)</script>',
                        path.read_text(encoding="utf-8"), re.S).group(1)
    harness = r'''
const vm=require('vm'),assert=require('assert'),elements={},plots=[];
function element(id){if(elements[id])return elements[id];let content='',value='';const e={textContent:'',options:[],querySelectorAll(){return []}};
Object.defineProperty(e,'value',{get(){return value},set(v){value=v}});Object.defineProperty(e,'innerHTML',{get(){return content},set(v){content=v;const opts=[...v.matchAll(/<option(?: value="([^"]*)")?[^>]*>(.*?)<\/option>/g)].map(m=>({value:m[1]||m[2]}));if(opts.length){e.options=opts;if(!opts.some(o=>o.value===value))value=opts[0].value}}});return elements[id]=e;}
const document={documentElement:{dataset:{}},getElementById:element};element('variant-data').textContent=PAYLOAD;
const sandbox={document,console,Plotly:{react(id,traces,layout){plots.push({id,traces,layout})}},getComputedStyle(){return {getPropertyValue(){return document.documentElement.dataset.theme==='dark'?'#123456':'#abcdef'}}},localStorage:{getItem(){return 'dark'},setItem(){}}};vm.runInNewContext(SCRIPT,sandbox);
assert.equal(element('variant').value,'xgb_unweighted_fixed');assert.equal(element('day').value,'2026-09-14');assert(plots.some(p=>p.id==='commonPR'));assert(plots.filter(p=>p.id==='hourlyMean').at(-1).traces.some(t=>t.name==='Observé'));assert(element('dailyMeans').innerHTML.includes('Prix moyen observé'));
element('country').value='FR';element('country').onchange();assert(element('support').textContent.includes('48 heures communes'));
element('variant').value='hgb_v1';element('variant').onchange();assert.equal(plots.filter(p=>p.id==='dailyChart').at(-1).traces.at(-1).name,'HGB original — snapshot figé');
element('theme').onclick();assert.equal(document.documentElement.dataset.theme,'light');assert.equal(plots.filter(p=>p.id==='dailyChart').at(-1).layout.paper_bgcolor,'#abcdef');
element('day').value='2026-09-15';element('day').onchange();assert(element('dayNote').textContent.includes('hors métriques historiques'));assert(element('shapText').textContent.includes('aucune valeur'));
'''.replace("PAYLOAD", json.dumps(payload)).replace("SCRIPT", json.dumps(report._SCRIPT))
    harness = harness.replace(r"<\\/option>", r"<\/option>")
    done = subprocess.run([str(node), "-"], input=harness, capture_output=True, encoding="utf-8")
    assert done.returncode == 0, done.stderr
