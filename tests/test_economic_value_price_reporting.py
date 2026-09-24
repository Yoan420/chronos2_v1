from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from economic_value.price_reporting import _pooled, render_price_report


BASE = "nuclear_kalman"
CANDIDATE = "nuclear_kalman_extreme"


def inputs():
    stamps = pd.date_range("2026-09-09T22:00Z", periods=4, freq="h")
    actual = [105., 115., 95., np.nan]
    forecasts = {BASE: [100., 100., 100., 100.], CANDIDATE: [100., 110., 100., 100.]}
    rows, metrics, daily, errors, error_days = [], [], [], [], []
    for model in (BASE, CANDIDATE):
        for strategy in ("no_forecast", "benchmark", "model"):
            for offset, stamp in enumerate(stamps):
                altered = model == CANDIDATE and offset == 1
                forecast = forecasts[model][offset] if strategy == "model" else 90.
                rows.append({"timestamp_utc": stamp, "delivery_day": "2026-09-10", "zone": "FR",
                             "model": model, "strategy": strategy, "forecast": forecast,
                             "actual": actual[offset], "q10": np.nan if altered else 80.,
                             "q90": np.nan if altered else 120., "reference_price": 95.,
                             "edge_eur_mwh": forecast - 95, "sample": "evaluation",
                             "paired_eligible": offset < 3, "portfolio_eligible": offset < 3,
                             "signal": "BUY", "position_mw": 100., "confidence": "unknown" if altered else "low",
                             "pnl_net_eur": 10. if offset < 3 else np.nan,
                             "evaluation_status": "evaluated" if offset < 3 else "pending_observation"})
            for zone in ("FR", "PORTFOLIO"):
                pnl = 80. if model == CANDIDATE else 50.
                metrics.append({"zone": zone, "model": model, "strategy": strategy,
                                "pnl_net_eur": pnl, "economic_value_added_eur": pnl - 40,
                                "eligible_hours": 3, "total_hours": 4, "expected_hours": 4,
                                "annual_fully_observed": False, "annual_window": False,
                                "economic_value_added_per_mw_year": np.nan})
                daily.append({"zone": zone, "model": model, "strategy": strategy,
                              "delivery_day": "2026-09-10", "cumulative_pnl_eur": pnl, "complete_day": False})
        mae = 4. if model == CANDIDATE else 6.
        for subset in ("all", "high", "normal", "negative", "expert_ready"):
            errors.append({"zone": "FR", "model": model, "subset": subset, "hours": 3,
                           "mae_eur_mwh": mae, "rmse_eur_mwh": mae + 1, "bias_eur_mwh": -1.})
        error_days.append({"zone": "FR", "model": model, "delivery_day": "2026-09-10", "hours": 3,
                           "mae_eur_mwh": mae, "rmse_eur_mwh": mae + 1, "bias_eur_mwh": -1.})
    for subset in ("all", "high", "normal", "negative", "expert_ready"):
        errors.append({"zone": "FR", "model": "storm", "subset": subset, "hours": 3,
                       "mae_eur_mwh": 5., "rmse_eur_mwh": 7., "bias_eur_mwh": -2.})
    error_days.append({"zone": "FR", "model": "storm", "delivery_day": "2026-09-10", "hours": 3,
                       "mae_eur_mwh": 5., "rmse_eur_mwh": 7., "bias_eur_mwh": -2.})
    decisions = pd.DataFrame({"timestamp_utc": stamps, "zone": "FR", "baseline_forecast": forecasts[BASE],
                              "candidate_forecast": forecasts[CANDIDATE], "raw_residual_prediction": [np.nan, 20., 10., 2.],
                              "applied_correction": [0., 10., 0., 0.], "selected_weight": [0., .5, 0., 0.],
                              "expert_ready": [False, True, True, True],
                              "reason": ["warmup_baseline", "governed", "baseline_retained", "baseline_retained"],
                              "forecast_origin_utc": pd.Timestamp("2026-09-09T06:00Z")})
    audit = {"reference_kind": "lagged_day_ahead_proxy", "price_expert": {
        "baseline_model": BASE, "candidate_model": CANDIDATE, "baseline_comparison_paired": True,
        "summary": [{"zone": "PORTFOLIO", "annual_non_regression": True, "annual_eva_gain_pass": True}],
        "minimum_training_days": 90, "training_window_days": 365,
        "model_audit": {"trained_folds": 2, "fallback_folds": 3,
                        "actual_training_days_min": 90, "actual_training_days_max": 97},
        "calibration_policy": "progressive_min90_max365",
    }}
    return {"rows": pd.DataFrame(rows), "metrics": pd.DataFrame(metrics), "daily": pd.DataFrame(daily),
            "breakdowns": {}, "audit": audit, "forecast_metrics": pd.DataFrame(errors),
            "forecast_daily": pd.DataFrame(error_days), "decisions": decisions,
            "governance": pd.DataFrame({"zone": ["FR"], "weight": [.5]})}


def price_payload(text):
    match = re.search(r'<script id="price-expert-payload" type="application/json">(.*?)</script>', text, re.S)
    assert match
    return json.loads(match.group(1))


def test_separate_report_reuses_eva_and_describes_progressive_calibration(tmp_path):
    values = inputs()
    values["audit"]["extreme_policy"] = {"not_applicable": True}
    original = values["rows"].copy(deep=True)
    path = render_price_report(**values, destination=tmp_path / "price.html")
    text = path.read_text(encoding="utf-8")
    assert not re.search(r'<script[^>]+src=', text, re.I)
    assert '<section id="priceExpertSection">' in text
    assert '<section id="extremeSection">' not in text
    assert "minimum de 90 jours" in text
    assert "de 90 à 97 jours historiques par bloc" in text
    assert "Le plafond n'est pas une preuve" in text
    assert "Prix non positifs (≤ 0)" in text
    assert "ne constituerait pas une calibration" in text
    assert 'id="equity"' in text and 'id="priceDayForecast"' in text
    pd.testing.assert_frame_equal(values["rows"], original)
    assert not list(tmp_path.glob('.price_render_*'))


def test_columnar_hourly_values_keep_missing_uncertainty(tmp_path):
    text = render_price_report(**inputs(), destination=tmp_path / "report.html").read_text(encoding="utf-8")
    payload = price_payload(text)
    packed = payload["hourly"]
    assert packed["values"][1][packed["columns"].index("candidate_q10")] is None
    assert packed["values"][1][packed["columns"].index("candidate_q90")] is None
    assert packed["values"][3][packed["columns"].index("actual")] is None
    assert payload["governance_rows"] == 1


def test_pooled_error_metrics_use_hour_weights_not_mean_rmse():
    frame = pd.DataFrame({"zone": ["FR", "BE"], "model": ["x", "x"], "subset": ["all", "all"],
                          "hours": [1, 3], "mae_eur_mwh": [2, 4], "rmse_eur_mwh": [2, 6], "bias_eur_mwh": [-1, 1]})
    portfolio = _pooled(frame, ["model", "subset"]).iloc[-1]
    assert portfolio.zone == "PORTFOLIO"
    assert portfolio.hours == 4
    assert portfolio.mae_eur_mwh == 3.5
    assert portfolio.rmse_eur_mwh == pytest.approx(np.sqrt(28))
    assert portfolio.bias_eur_mwh == .5


@pytest.mark.parametrize("field", ["q10", "q90"])
def test_modified_forecast_cannot_keep_uncalibrated_quantiles(tmp_path, field):
    values = inputs()
    mask = values["rows"].model.eq(CANDIDATE) & values["rows"].strategy.eq("model")
    values["rows"].loc[mask, field] = 150.
    with pytest.raises(ValueError, match="uncalibrated quantiles"):
        render_price_report(**values, destination=tmp_path / "bad.html")


def test_rejects_wrong_fallback_quantiles_or_applied_correction(tmp_path):
    values = inputs()
    index = values["rows"].index[values["rows"].model.eq(CANDIDATE) & values["rows"].strategy.eq("model")][0]
    values["rows"].loc[index, "q10"] = 60.
    with pytest.raises(ValueError, match="fallback quantiles"):
        render_price_report(**values, destination=tmp_path / "bad.html")
    values = inputs()
    values["decisions"].loc[1, "applied_correction"] = 11.
    with pytest.raises(ValueError, match="applied correction"):
        render_price_report(**values, destination=tmp_path / "bad.html")


def test_rejects_missing_baseline_or_changed_scored_forecast(tmp_path):
    values = inputs()
    values["rows"] = values["rows"].loc[values["rows"].model.ne(BASE)]
    with pytest.raises(ValueError, match="support differs"):
        render_price_report(**values, destination=tmp_path / "bad.html")
    values = inputs()
    values["decisions"].loc[1, "candidate_forecast"] = 109.
    with pytest.raises(ValueError, match="disagrees with the scored"):
        render_price_report(**values, destination=tmp_path / "bad.html")


def test_requires_price_audit_and_html_destination(tmp_path):
    values = inputs()
    values["audit"] = {}
    with pytest.raises(ValueError, match="price_expert"):
        render_price_report(**values, destination=tmp_path / "bad.html")
    with pytest.raises(ValueError, match=".html"):
        render_price_report(**inputs(), destination=tmp_path / "bad.txt")


def test_strict_365_reports_zero_trained_folds_without_progressive_90_claim(tmp_path):
    values = inputs()
    policy = values["audit"]["price_expert"]
    policy.update(minimum_training_days=365, training_window_days=365,
                  model_audit={"trained_folds": 0, "fallback_folds": 53,
                               "actual_training_days_min": None, "actual_training_days_max": None})
    text = render_price_report(**values, destination=tmp_path / "strict.html").read_text(encoding="utf-8")
    description = price_payload(text)["calibration_text"]
    assert "Calibration stricte : au moins 365 jours" in description
    assert "0 bloc entraîné" in description
    assert "Blocs en repli : 53" in description
    assert "90 jours" not in description
    assert "Calibration progressive" not in description


def test_calibration_uses_actual_training_depth_when_all_folds_are_full(tmp_path):
    values = inputs()
    values["audit"]["price_expert"].update(minimum_training_days=365, training_window_days=365,
        model_audit={"trained_folds": 5, "fallback_folds": 0,
                     "actual_training_days_min": 365, "actual_training_days_max": 365})
    text = render_price_report(**values, destination=tmp_path / "full.html").read_text(encoding="utf-8")
    description = price_payload(text)["calibration_text"]
    assert "5 blocs, de 365 à 365 jours" in description
    assert "90 jours" not in description


def test_rejects_different_observed_labels_between_models(tmp_path):
    values = inputs()
    mask = values["rows"].model.eq(CANDIDATE) & values["rows"].strategy.eq("model")
    values["rows"].loc[mask, "actual"] += 5
    with pytest.raises(ValueError, match="observed labels differ"):
        render_price_report(**values, destination=tmp_path / "bad.html")


def test_payload_escapes_html_and_template_markers(tmp_path):
    values = inputs()
    hostile = '</script><img src=x onerror="alert(1)">__PRICE_SECTION__'
    values["audit"]["price_expert"]["limitations"] = [hostile]
    values["decisions"]["reason"] = hostile
    text = render_price_report(**values, destination=tmp_path / "report.html").read_text(encoding="utf-8")
    assert hostile not in text
    assert price_payload(text)["audit"]["limitations"] == [hostile]
    assert "innerHTML" not in text[text.index('<script id="price-expert-payload"'):]


def test_runtime_price_charts_fields_subset_and_darkmode(tmp_path):
    node = shutil.which("node")
    bundled = Path("C:/Users/BQ6757/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe")
    if node is None and bundled.is_file():
        node = str(bundled)
    if node is None:
        pytest.skip("Node unavailable for static DOM smoke")
    text = render_price_report(**inputs(), destination=tmp_path / "report.html").read_text(encoding="utf-8")
    base_tail = text.split('<script id="economic-payload" type="application/json">', 1)[1]
    base_payload, base_tail = base_tail.split('</script><script>', 1)
    base_script = base_tail.split('</script>', 1)[0]
    price_tail = text.split('<script id="price-expert-payload" type="application/json">', 1)[1]
    price_json, price_tail = price_tail.split('</script><script>', 1)
    price_script = price_tail.split('</script>', 1)[0]
    harness = r'''
const assert=require('node:assert/strict');
class Element{constructor(){this.children=[];this.options=[];this.value='';this.textContent='';this.className='';this.attributes={}}appendChild(x){this.children.push(x);return x}replaceChildren(){this.children=[]}add(x){this.options.push(x);if(this.options.length===1)this.value=x.value}setAttribute(k,v){this.attributes[k]=v}}
const elements=new Map(),classes=new Set(),plots=new Map();
const document={getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id)},createElement(){return new Element()},body:{classList:{contains:x=>classes.has(x),toggle(x){if(classes.has(x))classes.delete(x);else classes.add(x)},add:x=>classes.add(x)}}};
class Option{constructor(label,value){this.label=label;this.value=value}}
const Plotly={react(id,traces,layout){plots.set(id,{traces,layout})}},localStorage={getItem(){return null},setItem(){}};
'''
    harness += "document.getElementById('economic-payload').textContent=" + json.dumps(base_payload) + ";\n"
    harness += "document.getElementById('price-expert-payload').textContent=" + json.dumps(price_json) + ";\n"
    checks = r'''
assert.equal(selected.model.value,'nuclear_kalman_extreme');
assert.equal(plots.size,11);
assert.equal(document.getElementById('priceMaeGain').textContent,'2,000 €/MWh');
assert.equal(document.getElementById('pricePnlGain').textContent,'30 €');
assert.equal(document.getElementById('priceChanged').textContent,'1');
assert.equal(document.getElementById('priceWeight').textContent,'16,7 %');
const priceTable=document.getElementById('priceDayTable').children[0].children[1].children;
assert.equal(priceTable.length,4);
assert.equal(priceTable[1].children[3].textContent,'120,00');
assert.equal(priceTable[1].children[5].textContent,'110,00');
assert.equal(priceTable[1].children[10].textContent,'—');
assert.equal(priceTable[1].children[11].textContent,'—');
assert.equal(priceTable[0].children[10].textContent,'80,00');
assert.equal(priceTable[3].children[7].textContent,'—');
assert.equal(plots.get('priceDayForecast').traces.at(-1).visible,'legendonly');
document.getElementById('priceSubset').value='expert_ready';document.getElementById('priceSubset').onchange();
assert.equal(document.getElementById('priceMetrics').children[0].children[1].children[0].children[1].textContent,'expert_ready');
priceExpert.baseline_comparison_paired=false;drawPrice();
assert.equal(document.getElementById('pricePnlGain').textContent,'—');
assert(plots.get('pricePnlCurve').traces[0].y.every(x=>x===null));
document.getElementById('theme').onclick();
assert.equal(plots.get('priceDayForecast').layout.font.color,'#e6edf8');
assert.equal(plots.get('priceDayForecast').traces[2].line.color,'#5cd8b7');
selected.zone.value='FR';selected.zone.onchange();assert.equal(document.getElementById('priceChanged').textContent,'1');
selected.day.value='2026-09-11';selected.day.onchange();assert.equal(plots.get('priceDayForecast').traces[0].x.length,0);
console.log('Price report DOM smoke passed');
'''
    target = tmp_path / "price_smoke.cjs"
    target.write_text(harness + base_script + price_script + checks, encoding="utf-8")
    result = subprocess.run([node, str(target)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DOM smoke passed" in result.stdout
