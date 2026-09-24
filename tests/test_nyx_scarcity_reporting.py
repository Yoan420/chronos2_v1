"""Common-support metrics and offline HTML for the NYX scarcity experiment."""
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
import pytest

from nyx_scarcity import reporting as report


def predictions(days=365, zone="FR", end="2026-09-14", live=True):
    tz = report.TIMEZONES[zone]
    end = pd.Timestamp(end)
    index = pd.date_range((end-pd.Timedelta(days=days-1)).tz_localize(tz),
                         (end+pd.Timedelta(days=1)).tz_localize(tz), freq="h", inclusive="left").tz_convert("UTC")
    frame = pd.DataFrame({"zone": zone, "timestamp_utc": index,
                         "forecast_origin_utc": index.normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=6),
                         "actual": 100., "forecast": 90., "benchmark_forecast": 95., "candidate_forecast": 98.,
                         "q10": 80., "q90": 105., "candidate_q10": 88., "candidate_q90": 113.,
                         "spike_probability": .2, "threshold_eur_mwh": 50., "raw_correction": 20.,
                         "bounded_correction": 15., "applied_correction": 8., "selected_weight": .5,
                         "expert_ready": True, "gate_reason": "accepted", "interval_status": "mixture",
                         "probability_gate": .6, "sample": "evaluation"})
    if live:
        future = pd.date_range((end+pd.Timedelta(days=1)).tz_localize(tz),
                               (end+pd.Timedelta(days=2)).tz_localize(tz), freq="h", inclusive="left").tz_convert("UTC")
        latest = frame.iloc[:len(future)].copy()
        latest["timestamp_utc"] = future
        latest["forecast_origin_utc"] = future[0]-pd.Timedelta(hours=16)
        latest["actual"] = np.nan
        latest["sample"] = "live"
        frame = pd.concat([frame, latest], ignore_index=True)
    return frame


def test_full_365_common_metrics_and_no_input_mutation():
    frame = predictions()
    original = frame.copy(deep=True)
    metrics, daily, hourly = report.evaluate_predictions(frame)
    m = metrics["by_zone"]["FR"]
    assert metrics["windows"]["FR"]["complete_365_paired_support"] is True
    assert metrics["live_rows_excluded"] == 24
    assert m["paired_hours"] == 8760
    assert m["annual"]["baseline"]["mae_eur_mwh"] == 10.
    assert m["annual"]["candidate"]["mae_eur_mwh"] == 2.
    assert m["annual"]["storm"]["mae_eur_mwh"] == 5.
    assert m["annual"]["gain_baseline_minus_candidate"]["mae_eur_mwh"] == 8.
    assert m["annual"]["candidate"]["mean_observed_eur_mwh"] == 100.
    assert m["annual"]["candidate"]["mean_forecast_eur_mwh"] == 98.
    assert m["annual"]["raw_ungoverned_diagnostic"]["raw_candidate"]["mae_eur_mwh"] == 10.
    assert m["annual"]["bounded_weight_one_diagnostic"]["bounded_candidate"]["mae_eur_mwh"] == 5.
    assert len(daily) == 365 and len(hourly) == 24
    assert daily.local_day.min() == "2025-09-15" and daily.local_day.max() == "2026-09-14"
    assert daily.paired_hours.sum() == hourly.paired_hours.sum() == 8760
    assert daily.actual_mean_eur_mwh.eq(100).all()
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("missing", ["actual", "benchmark_forecast", "forecast", "candidate_forecast"])
def test_one_missing_member_excludes_same_hour_everywhere(missing):
    frame = predictions(days=2, live=False)
    frame.loc[0, missing] = np.nan
    frame.loc[0, [c for c in ("actual", "forecast", "candidate_forecast", "benchmark_forecast") if c != missing]] = 10000.
    metrics, daily, hourly = report.evaluate_predictions(frame)
    m = metrics["overall"]
    assert m["paired_hours"] == 47 and m["unpaired_hours"] == 1
    assert m["annual"]["baseline"]["hours"] == m["annual"]["candidate"]["hours"] == m["annual"]["storm"]["hours"] == 47
    assert m["annual"]["candidate"]["mae_eur_mwh"] == 2.
    assert daily.paired_hours.sum() == hourly.paired_hours.sum() == 47
    assert metrics["windows"]["FR"]["complete_365_paired_support"] is False


def test_missing_storm_entire_day_keeps_empty_scores_not_zero():
    frame = predictions(days=2, live=False)
    frame.loc[:23, "benchmark_forecast"] = np.nan
    metrics, daily, hourly = report.evaluate_predictions(frame)
    assert metrics["overall"]["paired_hours"] == 24
    first = daily.iloc[0]
    assert first.paired_hours == 0
    assert pd.isna(first.candidate_mae_eur_mwh) and pd.isna(first.actual_mean_eur_mwh)
    assert first.actual_mean_available_eur_mwh == 100.
    assert hourly.paired_hours.eq(1).all()


def test_unpublished_actuals_and_live_actuals_are_never_scored():
    frame = predictions(days=2)
    frame.loc[:23, "actual"] = np.nan
    frame.loc[frame["sample"].eq("live"), "actual"] = 1000000.
    metrics, daily, _ = report.evaluate_predictions(frame)
    assert metrics["overall"]["paired_hours"] == 24
    assert metrics["live_rows_excluded"] == 24
    assert metrics["overall"]["annual"]["candidate"]["mae_eur_mwh"] == 2.
    assert daily.local_day.max() == "2026-09-14"


def test_baseline_fallback_rows_stay_in_annual_scores():
    frame = predictions(days=2, live=False)
    frame.loc[:23, "candidate_forecast"] = frame.loc[:23, "forecast"]
    frame.loc[:23, "applied_correction"] = 0.
    frame.loc[:23, "expert_ready"] = False
    frame.loc[:23, "gate_reason"] = "warmup"
    metrics, _, _ = report.evaluate_predictions(frame)
    m = metrics["overall"]
    assert m["paired_hours"] == 48
    assert m["annual"]["candidate"]["mae_eur_mwh"] == 6.
    assert m["exact_baseline_fallback_hours"] == 24
    assert m["active_hours"] == 24 and m["not_ready_hours"] == 24
    assert m["active_only"]["scores"]["candidate"]["mae_eur_mwh"] == 2.
    assert m["gate_reason_counts"]["warmup"] == 24


def test_tail_selection_uses_observed_prices_ex_post_on_common_rows():
    frame = predictions(days=5, live=False)
    frame["actual"] = np.arange(len(frame), dtype=float)
    frame.loc[0, "actual"] = 1e9
    frame.loc[0, "benchmark_forecast"] = np.nan
    metrics, _, _ = report.evaluate_predictions(frame)
    tails = metrics["overall"]["tails"]
    assert tails["top_1_percent"]["threshold_eur_mwh"] == pytest.approx(np.quantile(np.arange(1, 120), .99))
    assert tails["top_1_percent"]["scores"]["candidate"]["hours"] == 2
    assert tails["top_5_percent"]["scores"]["candidate"]["hours"] == 6
    assert tails["top_1_percent"]["used_for_training_or_governance"] is False
    assert "ex_post" in tails["top_1_percent"]["selection"]


def test_classifier_uses_residual_threshold_and_strict_configured_gate():
    frame = predictions(days=1, live=False).iloc[:4].copy()
    frame["actual"] = [140., 150., 160., 140.]
    frame["forecast"] = 100.
    frame["threshold_eur_mwh"] = 50.
    frame["spike_probability"] = [.8, .6, .9, .1]
    frame["probability_gate"] = [.6, .6, .7, .6]
    metrics, _, _ = report.evaluate_predictions(frame)
    c = metrics["overall"]["classifier"]
    assert c["observed_events"] == 2  # Errors 50 and 60, not all four price levels > 50.
    assert c["true_positive"] == 1 and c["false_positive"] == 1
    assert c["false_negative"] == 1 and c["true_negative"] == 1
    assert c["precision"] == c["recall"] == .5
    assert c["brier"] == pytest.approx(np.mean((np.array([.8, .6, .9, .1])-np.array([0, 1, 1, 0]))**2))
    assert c["probability_gates"] == [.6, .7]


def test_missing_classifier_inputs_do_not_change_point_support():
    frame = predictions(days=1, live=False)
    frame.loc[:2, "spike_probability"] = np.nan
    frame.loc[3, "threshold_eur_mwh"] = np.nan
    metrics, _, _ = report.evaluate_predictions(frame)
    assert metrics["overall"]["paired_hours"] == 24
    assert metrics["overall"]["classifier"]["hours"] == 20


def test_quantile_metrics_share_all_four_endpoints_and_never_fill():
    frame = predictions(days=1, live=False)
    frame.loc[0, "candidate_q10"] = np.nan
    frame.loc[1, "q90"] = np.nan
    metrics, _, _ = report.evaluate_predictions(frame)
    q = metrics["overall"]["quantiles"]
    assert metrics["overall"]["paired_hours"] == 24
    assert q["excluded_interval_hours"] == 2
    assert q["scores"]["baseline"]["hours"] == q["scores"]["candidate"]["hours"] == 22
    assert q["scores"]["baseline"]["coverage_p10_p90"] == 1.
    assert q["scores"]["candidate"]["pinball_p10_eur_mwh"] == pytest.approx(1.2)
    assert q["scores"]["candidate"]["pinball_p90_eur_mwh"] == pytest.approx(1.3)


def test_last_365_civil_days_only_and_country_local_dst():
    frames = [predictions(days=370, zone=zone, end="2026-10-25", live=False) for zone in report.TIMEZONES]
    metrics, daily, hourly = report.evaluate_predictions(pd.concat(frames, ignore_index=True))
    assert metrics["older_evaluation_rows_excluded"] > 0
    for zone in report.TIMEZONES:
        w = metrics["windows"][zone]
        assert w["represented_days"] == 365 and w["complete_365_physical_support"]
        assert daily.loc[(daily.zone == zone) & (daily.local_day == "2026-10-25"), "paired_hours"].item() == 25
        assert len(hourly.loc[hourly.zone.eq(zone)]) == 24


@pytest.mark.parametrize("defect", ["naive", "duplicate", "infinite", "probability", "weight", "crossing", "sample", "boolean", "zone"])
def test_invalid_inputs_refused(defect):
    frame = predictions(days=1, live=False)
    if defect == "naive": frame["timestamp_utc"] = frame.timestamp_utc.dt.tz_localize(None)
    elif defect == "duplicate": frame = pd.concat([frame, frame.iloc[:1]])
    elif defect == "infinite": frame.loc[0, "raw_correction"] = np.inf
    elif defect == "probability": frame.loc[0, "spike_probability"] = 1.01
    elif defect == "weight": frame.loc[0, "selected_weight"] = -.1
    elif defect == "crossing": frame.loc[0, "candidate_q10"] = 999.
    elif defect == "sample": frame.loc[0, "sample"] = "training"
    elif defect == "boolean": frame["expert_ready"] = "false"
    elif defect == "zone": frame["zone"] = "XX"
    with pytest.raises(ValueError): report.evaluate_predictions(frame)


def test_missing_optional_columns_and_empty_data_are_explicit():
    frame = predictions(days=1, live=False)[["zone", "timestamp_utc", "sample"]]
    metrics, daily, hourly = report.evaluate_predictions(frame)
    assert metrics["overall"]["paired_hours"] == 0
    assert metrics["overall"]["annual"]["candidate"]["mae_eur_mwh"] is None
    assert daily.candidate_mae_eur_mwh.isna().all()
    empty, d, h = report.evaluate_predictions(frame.iloc[:0])
    assert empty["overall"]["rows"] == 0 and d.empty and h.empty
    assert "local_day" in d and "local_hour" in h


def test_report_is_offline_safe_and_contains_all_sections(tmp_path):
    frame = predictions(days=2)
    frame.loc[0, "gate_reason"] = '</script><img src=x onerror="evil()">'
    metrics, daily, hourly = report.evaluate_predictions(frame)
    before = deepcopy(metrics)
    audit = {"data": {"note": "</pre><script>bad()</script>"}, "config": {"training": 90}, "model": {"active": False}}
    path = report.render_report(frame, metrics, daily, hourly, audit, tmp_path/"report.html")
    document = path.read_text(encoding="utf-8")
    assert '<script src=' not in document
    assert 'plotly.js' in document.lower()
    assert '<script>bad()' not in document and '<img src=x onerror="evil()">' not in document
    assert '&lt;script&gt;bad()' in document
    assert '\\u003c/script\\u003e' in document
    for marker in ('data-report-section="statistics"', 'data-report-section="daily-forecast"',
                   'data-report-section="hourly-performance"', 'data-report-section="daily-calendar"',
                   'data-report-section="tails"', 'data-report-section="active-subset"',
                   'data-report-section="probabilistic"', 'data-theme="light"', 'html[data-theme=dark]',
                   'nyx-scarcity-theme', '90 à 365 jours', 'année déjà examinée', 'Aucune activation',
                   'Ce n’est pas un seuil de niveau de prix'):
        assert marker in document
    embedded = re.search(r'<script id="nyx-data" type="application/json">(.*?)</script>', document, re.S).group(1)
    payload = json.loads(embedded)
    assert payload["predictions"][-1]["actual"] is None
    assert 'NaN' not in embedded and 'Infinity' not in embedded
    assert metrics == before


def test_report_omits_hundreds_of_training_features_without_changing_backtest_frame(tmp_path):
    frame = predictions(days=2)
    extra = pd.DataFrame({f"training_feature_{i:03d}": np.arange(len(frame), dtype=float) + i
                          for i in range(250)}, index=frame.index)
    frame = pd.concat([frame, extra], axis=1)
    original = frame.copy(deep=True)
    metrics, daily, hourly = report.evaluate_predictions(frame)
    audit = {"data": {"feature_count": 250, "coverage": .98},
             "model": {"folds": [{"fit_days": 90, "weight": 0.}]}, "config": {"minimum_training_days": 90}}
    path = report.render_report(frame, metrics, daily, hourly, audit, tmp_path/"wide.html")
    document = path.read_text(encoding="utf-8")
    embedded = re.search(r'<script id="nyx-data" type="application/json">(.*?)</script>', document, re.S).group(1)
    payload = json.loads(embedded)
    assert len(payload["predictions"]) == len(frame)
    assert set(payload["predictions"][0]) == set(report._REPORT_PREDICTION_COLUMNS)
    assert len(payload["predictions"][0]) == 19
    assert "training_feature_" not in embedded
    assert "training_feature_" not in document
    assert "fit_days" in document and "feature_count" in document
    assert len(embedded) < 150_000
    pd.testing.assert_frame_equal(frame, original)


def test_browser_script_syntax_with_node_if_available(tmp_path):
    node = Path(r"C:\Users\BQ6757\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
    if not node.is_file(): pytest.skip("Bundled Node runtime unavailable")
    completed = subprocess.run([str(node), "--check"], input=report._SCRIPT, capture_output=True,
                               text=True, encoding="utf-8")
    assert completed.returncode == 0, completed.stderr


def test_json_encoder_escapes_script_separators_and_nonfinite_values():
    data = {"x": "</script><script>alert(1)</script>\u2028&", "y": np.nan, "z": np.float32(.6)}
    encoded = report._script_json(data)
    assert '<' not in encoded and '&' not in encoded and '\u2028' not in encoded
    assert json.loads(encoded)["y"] is None
    assert json.loads(encoded)["z"] == float(np.float32(.6))


def test_duplicate_pandas_indices_do_not_mix_countries():
    frame = pd.concat([predictions(days=2, zone="FR", live=False), predictions(days=2, zone="DE", live=False)])
    metrics, daily, _ = report.evaluate_predictions(frame)
    assert metrics["by_zone"]["FR"]["paired_hours"] == metrics["by_zone"]["DE"]["paired_hours"] == 48
    assert len(daily) == 4


def test_browser_controls_execute_under_dom_stub():
    node = Path(r"C:\Users\BQ6757\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
    if not node.is_file(): pytest.skip("Bundled Node runtime unavailable")
    frame = pd.concat([predictions(days=2, zone="FR"), predictions(days=2, zone="DE")])
    metrics, daily, hourly = report.evaluate_predictions(frame)
    payload = {"predictions": report._prepare(frame).loc[:, report._REPORT_PREDICTION_COLUMNS].to_dict("records"), "metrics": metrics,
               "daily": daily.to_dict("records"), "hourly": hourly.to_dict("records")}
    harness = r'''
const vm=require('vm'),assert=require('assert');const elements={},plots=[];
function element(id){if(elements[id])return elements[id];let value='',content='';const e={textContent:'',checked:false,options:[],dataset:{},scrollIntoView(){},querySelectorAll(){return []}};
Object.defineProperty(e,'value',{get(){return value},set(v){value=v}});Object.defineProperty(e,'innerHTML',{get(){return content},set(v){content=v;const options=[...v.matchAll(/<option[^>]*>(.*?)<\/option>/g)].map(m=>({value:m[1]}));if(options.length){e.options=options;if(!options.some(o=>o.value===value))value=options[0].value;}}});elements[id]=e;return e;}
const document={documentElement:{dataset:{}},getElementById:element};element('nyx-data').textContent=PAYLOAD;element('calendarMetric').value='gain_baseline_minus_candidate_eur_mwh';
const sandbox={document,console,Plotly:{react(id,traces,layout){plots.push({id,traces,layout})}},localStorage:{getItem(){return 'dark'},setItem(){}},getComputedStyle(){return {getPropertyValue(name){return document.documentElement.dataset.theme==='dark'?'#123456':'#abcdef'}}}};
vm.runInNewContext(SCRIPT,sandbox);assert.equal(document.documentElement.dataset.theme,'dark');assert.equal(element('day').value,'2026-09-14');assert(plots.some(p=>p.id==='dailyChart'));assert(element('stats').innerHTML.includes('NYX + expert gouverné'));
element('theme').onclick();assert.equal(document.documentElement.dataset.theme,'light');assert.equal(plots.filter(p=>p.id==='dailyChart').at(-1).layout.paper_bgcolor,'#abcdef');
element('showRaw').checked=true;element('showRaw').onchange();assert(plots.filter(p=>p.id==='dailyChart').at(-1).traces.some(t=>t.name?.includes('non gouvernée')));
element('country').value='DE';element('country').onchange();assert.equal(element('day').value,'2026-09-14');assert(element('support').textContent.includes('48 heures communes'));
element('day').value='2026-09-15';element('day').onchange();assert(element('dayNote').textContent.includes('exclue des métriques'));
'''.replace("PAYLOAD", json.dumps(report._script_json(payload))).replace("SCRIPT", json.dumps(report._SCRIPT))
    # The raw harness is JavaScript, so its regex uses one escaped slash.
    harness = harness.replace(r"<\\/option>", r"<\/option>")
    completed = subprocess.run([str(node), "-"], input=harness, capture_output=True, text=True, encoding="utf-8")
    assert completed.returncode == 0, completed.stderr
