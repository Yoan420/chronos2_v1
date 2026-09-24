from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from economic_value.reporting import render_report


def _inputs():
    rows, metrics, daily, grouped = [], [], [], []
    for model in ("autonomous", "kalman"):
        for strategy in ("no_forecast", "benchmark", "model"):
            for offset in (0, 1):
                rows.append({"timestamp_utc": pd.Timestamp("2026-09-08T22:00Z") + pd.Timedelta(hours=offset),
                             "delivery_day": "2026-09-09", "delivery_hour": offset,
                             "zone": "FR", "model": model, "strategy": strategy,
                             "forecast": 120.0, "reference_price": 110.0,
                             "actual": None if offset else 115.0,
                             "q10": 100.0, "q90": 140.0,
                             "edge_eur_mwh": 10.0, "confidence": "low", "signal": "BUY",
                             "position_mw": 10.0, "pnl_net_eur": None if offset else 50.0,
                             "sample": "backtest", "evaluation_status": "pending" if offset else "paired"})
            for zone in ("FR", "PORTFOLIO"):
                metrics.append({"model": model, "zone": zone, "strategy": strategy,
                                "pnl_net_eur": 50.0, "pnl_gross_eur": 50.0,
                                "trading_cost_eur": 0.0, "economic_value_added_eur": 10.0,
                                "economic_value_added_per_mw_year": None,
                                "annual_window": False, "annual_fully_observed": False,
                                "eligible_hours": 1, "total_hours": 2, "active_hours": 1,
                                "max_drawdown_eur": 0.0})
                daily.append({"model": model, "zone": zone, "strategy": strategy,
                              "delivery_day": "2026-09-09", "pnl_net_eur": 50.0,
                              "cumulative_pnl_eur": 50.0, "complete_day": False})
                grouped.append({"model": model, "zone": zone, "strategy": strategy,
                                "group": "low", "pnl_net_eur": 50.0,
                                "eligible_hours": 1, "active_hours": 1})
    return (pd.DataFrame(rows), pd.DataFrame(metrics), pd.DataFrame(daily),
            {key: pd.DataFrame(grouped) for key in ("hour", "season", "spike", "confidence", "monthly")},
            {"reference_mode": "lagged_da_proxy", "cutoff_time": "08:00", "result": "diagnostic"})


def _extreme_inputs():
    rows, metrics, daily, breakdowns, audit = _inputs()
    names = {"autonomous": "nuclear_kalman", "kalman": "nuclear_kalman_extreme_governed"}
    for frame in (rows, metrics, daily, *breakdowns.values()):
        frame["model"] = frame["model"].map(names)
    rows["sample"] = "evaluation"
    rows["paired_eligible"] = rows["actual"].notna()
    rows["portfolio_eligible"] = rows["actual"].notna()
    mask = rows["model"].eq(names["kalman"]) & rows["strategy"].eq("model")
    rows.loc[mask, "baseline_position_fraction"] = 1.0
    rows.loc[mask, "policy_position_fraction"] = [1.0, 0.5]
    rows.loc[mask, "governance_weight"] = [0.0, 0.5]
    rows.loc[mask, "extreme_probability_up"] = 0.2
    rows.loc[mask, "extreme_probability_down"] = 0.3
    rows.loc[mask, "extreme_expected_edge"] = -5.0
    rows.loc[mask, "policy_reason"] = pd.Series(["warmup_baseline", "governed_position"], index=rows.index[mask])
    rows.loc[mask, "position_mw"] = [10.0, 5.0]
    metric_mask = metrics["model"].eq(names["kalman"]) & metrics["strategy"].eq("model")
    metrics.loc[metric_mask, "pnl_net_eur"] = 70.0
    metrics.loc[metric_mask, "economic_value_added_eur"] = 30.0
    audit["extreme_policy"] = {
        "enabled": True, "baseline_model": names["autonomous"], "candidate_model": names["kalman"],
        "training_days": 365, "refit_every_days": 7, "governance_minimum_days": 28,
        "governance_lookback_days": 60, "baseline_comparison_paired": True,
        "summary": [{"zone": "PORTFOLIO", "intervention_hours": 0, "mean_governance_weight": 0.0}],
    }
    return rows, metrics, daily, breakdowns, audit


def _payload(text, *, decode=True):
    match = re.search(r'<script id="economic-payload" type="application/json">(.*?)</script>', text, re.S)
    assert match is not None
    payload = json.loads(match.group(1))
    if decode:
        for key in ("rows", "daily"):
            packed = payload[key]
            dictionaries = packed["dictionaries"]
            records = []
            for values in packed["values"]:
                record = {"strategy": "model"} if key == "rows" else {}
                for index, column in enumerate(packed["columns"]):
                    value = values[index]
                    record[column] = (dictionaries[str(index)][value]
                                      if value is not None and str(index) in dictionaries else value)
                records.append(record)
            payload[key] = records
    return payload


def test_self_contained_report_contains_all_views_and_disclaimers(tmp_path):
    output = render_report(*_inputs(), tmp_path / "report.html")
    text = output.read_text(encoding="utf-8")
    assert not re.search(r'<script[^>]+src=', text, re.I)
    assert "Plotly.react" in text
    for element in ("equity", "drawdown", "hour", "season", "spike", "confidence", "monthly", "traderTable"):
        assert f'id="{element}"' in text
    for phrase in ("référence proxy non négociable", "Strategy 0", "Strategy 1", "Strategy 2",
                   "08 h", "pas des recommandations de trading", "365 jours entièrement appariés",
                   "ce n'est pas une probabilité", "position nulle", "sans extrapolation annuelle"):
        assert phrase in text
    assert "body.night" in text
    assert "aria-pressed" in text
    assert "color:dark()" in text


def test_pending_values_stay_null_and_source_is_not_mutated(tmp_path):
    inputs = _inputs()
    before = inputs[0].copy(deep=True)
    text = render_report(*inputs, tmp_path / "report.html").read_text(encoding="utf-8")
    payload = _payload(text)
    assert payload["rows"][1]["actual"] is None
    assert payload["rows"][1]["pnl_net_eur"] is None
    assert payload["metrics"][0]["economic_value_added_per_mw_year"] is None
    pd.testing.assert_frame_equal(inputs[0], before)


def test_html_keeps_only_hourly_candidate_rows(tmp_path):
    inputs = _inputs()
    payload = _payload(render_report(*inputs, tmp_path / "report.html").read_text(encoding="utf-8"))
    assert {row["strategy"] for row in payload["rows"]} == {"model"}
    assert len(payload["rows"]) == 4
    assert payload["report_payload"]["archive_rows"] == 12
    assert {row["strategy"] for row in payload["metrics"]} == {"no_forecast", "benchmark", "model"}


def test_payload_is_columnar_and_daily_is_restricted(tmp_path):
    inputs = list(_inputs())
    inputs[2]["unused_archived_detail"] = "not needed in charts"
    raw = _payload(render_report(*inputs, tmp_path / "report.html").read_text(encoding="utf-8"), decode=False)
    assert set(raw["rows"]) == {"columns", "values", "dictionaries"}
    assert "strategy" not in raw["rows"]["columns"]
    assert raw["rows"]["dictionaries"]
    assert "unused_archived_detail" not in raw["daily"]["columns"]
    assert raw["report_payload"]["encoding"] == "columnar_dictionary_v1"


def test_payload_cannot_close_script_or_inject_html(tmp_path):
    inputs = list(_inputs())
    hostile = '</script><img src=x onerror="alert(1)">\u2028'
    inputs[4] = {"source_label": hostile, "nan": float("nan"), "path": Path("some/path")}
    text = render_report(*inputs, tmp_path / "report.html").read_text(encoding="utf-8")
    assert hostile not in text
    payload = _payload(text, decode=False)
    assert payload["audit"]["source_label"] == hostile
    assert payload["audit"]["nan"] is None
    assert "innerHTML" not in text[text.index('<script id="economic-payload"'):]


def test_rejects_non_html_destination(tmp_path):
    with pytest.raises(ValueError, match=".html"):
        render_report(*_inputs(), tmp_path / "report.txt")


def test_rejects_missing_columns(tmp_path):
    inputs = list(_inputs())
    inputs[0] = inputs[0].drop(columns="timestamp_utc")
    with pytest.raises(ValueError, match="timestamp_utc"):
        render_report(*inputs, tmp_path / "report.html")


def test_rejects_unknown_strategy(tmp_path):
    inputs = list(_inputs())
    inputs[0].loc[0, "strategy"] = "surprise"
    with pytest.raises(ValueError, match="unknown strategy"):
        render_report(*inputs, tmp_path / "report.html")


def test_empty_breakdowns_still_render(tmp_path):
    inputs = list(_inputs())
    inputs[3] = {}
    text = render_report(*inputs, tmp_path / "report.html").read_text(encoding="utf-8")
    assert _payload(text)["breakdowns"] == {}


def test_extreme_section_is_optional_and_preserves_new_fields(tmp_path):
    ordinary = render_report(*_inputs(), tmp_path / "ordinary.html").read_text(encoding="utf-8")
    assert '<section id="extremeSection">' not in ordinary
    text = render_report(*_extreme_inputs(), tmp_path / "extreme.html").read_text(encoding="utf-8")
    assert '<section id="extremeSection">' in text
    assert "Ce n'est ni un nouveau forecast électrique" in text
    assert "pas une année indépendante laissée intacte" in text
    payload = _payload(text)
    policy_rows = [row for row in payload["rows"] if row["model"] == "nuclear_kalman_extreme_governed"]
    assert policy_rows[0]["policy_reason"] == "warmup_baseline"
    assert policy_rows[0]["policy_position_fraction"] == policy_rows[0]["baseline_position_fraction"]
    assert policy_rows[0]["forecast"] == 120
    assert policy_rows[1]["governance_weight"] == 0.5
    assert policy_rows[1]["extreme_expected_edge"] == -5


@pytest.mark.parametrize("field", ["forecast", "q10", "q90"])
def test_rejects_extreme_report_that_changes_price_forecasts(tmp_path, field):
    inputs = list(_extreme_inputs())
    mask = inputs[0]["model"].eq("nuclear_kalman_extreme_governed") & inputs[0]["strategy"].eq("model")
    inputs[0].loc[mask, field] += 1
    with pytest.raises(ValueError, match="not a position-only"):
        render_report(*inputs, tmp_path / "bad.html")


def test_rejects_extreme_report_without_matching_baseline(tmp_path):
    inputs = list(_extreme_inputs())
    inputs[0] = inputs[0].loc[inputs[0]["model"].ne("nuclear_kalman")]
    with pytest.raises(ValueError, match="baseline forecasts missing"):
        render_report(*inputs, tmp_path / "bad.html")


def test_extreme_reason_html_is_escaped(tmp_path):
    inputs = list(_extreme_inputs())
    hostile = '</script><img src="x" onerror="alert(1)">'
    inputs[0]["policy_reason"] = hostile
    inputs[4]["extreme_policy"]["summary"][0]["comment"] = hostile
    text = render_report(*inputs, tmp_path / "report.html").read_text(encoding="utf-8")
    assert hostile not in text
    assert _payload(text)["rows"][0]["policy_reason"] == hostile


def test_extreme_payload_cannot_expand_template_placeholders(tmp_path):
    inputs = list(_extreme_inputs())
    inputs[0]["policy_reason"] = "__EXTREME_SECTION__"
    text = render_report(*inputs, tmp_path / "report.html").read_text(encoding="utf-8")
    assert text.count('<section id="extremeSection">') == 1
    assert _payload(text)["rows"][0]["policy_reason"] == "__EXTREME_SECTION__"


def _run_js_smoke(tmp_path, inputs, checks):
    node = shutil.which("node")
    bundled = Path("C:/Users/BQ6757/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe")
    if node is None and bundled.is_file():
        node = str(bundled)
    if node is None:
        pytest.skip("Node runtime unavailable for static DOM smoke test")
    text = render_report(*inputs, tmp_path / "report.html").read_text(encoding="utf-8")
    payload = _payload(text, decode=False)
    script = text.split('<script id="economic-payload"', 1)[1].split('</script><script>', 1)[1].split('</script>', 1)[0]
    harness = r'''
const assert=require('node:assert/strict');
class Element {
 constructor(){this.children=[];this.options=[];this.value='';this.textContent='';this.className='';this.attributes={}}
 appendChild(child){this.children.push(child);return child}
 replaceChildren(){this.children=[]}
 add(item){this.options.push(item);if(this.options.length===1)this.value=item.value}
 setAttribute(key,value){this.attributes[key]=value}
}
const elements=new Map();const classes=new Set();
const document={getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id)},createElement(){return new Element()},body:{classList:{contains(x){return classes.has(x)},toggle(x){if(classes.has(x))classes.delete(x);else classes.add(x)},add(x){classes.add(x)}}}};
class Option{constructor(label,value){this.label=label;this.value=value}}
const localStorage={getItem(){return null},setItem(){}};
const plots=new Map();const Plotly={react(id,traces,layout){plots.set(id,{traces,layout})}};
document.getElementById('economic-payload').textContent=__JSON__;
'''.replace("__JSON__", json.dumps(json.dumps(payload)))
    target = tmp_path / "smoke.cjs"
    target.write_text(harness + script + checks, encoding="utf-8")
    result = subprocess.run([node, str(target)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DOM smoke passed" in result.stdout


def test_javascript_bootstrap_filters_and_theme(tmp_path):
    _run_js_smoke(tmp_path, _inputs(), r'''
assert.equal(selected.zone.value,'PORTFOLIO');
assert.equal(plots.get('equity').traces.length,3);
assert.equal(plots.get('confidence').traces.length,3);
assert.equal(document.getElementById('evaMw').textContent,'—');
data.metrics.find(r=>r.zone==='PORTFOLIO'&&r.model==='autonomous'&&r.strategy==='model').economic_value_added_per_allocated_mw=12.5;
drawMetrics();
assert.equal(document.getElementById('evaMw').textContent,'12,50 €/MW');
assert.match(document.getElementById('evaMwLabel').textContent,/échantillon/);
assert.match(document.getElementById('evaMwSub').textContent,/sans annualisation/);
assert.match(document.getElementById('referenceWarning').textContent,/proxy non négociable/);
assert.match(document.getElementById('dayStatus').textContent,/1 observations en attente/);
selected.model.value='kalman';selected.model.onchange();
assert.equal(plots.get('equity').traces.length,3);
assert.equal(plots.get('equity').traces[2].y.length,1);
selected.zone.value='FR';selected.zone.onchange();
assert.equal(selected.traderZone.value,'FR');
const testTraderTable=document.getElementById('traderTable').children[0];
assert.equal(testTraderTable.children[1].children.length,2);
assert.equal(testTraderTable.children[1].children[1].children[9].textContent,'—');
assert.equal(testTraderTable.children[1].children[1].children[10].textContent,'—');
document.getElementById('theme').onclick();
assert.equal(document.getElementById('theme').attributes['aria-pressed'],'true');
assert.equal(plots.get('equity').layout.font.color,'#e6edf8');
assert.equal(plots.get('equity').traces[2].line.color,'#5cd8b7');
console.log('Economic report DOM smoke passed');
''')


def test_extreme_javascript_policy_only_comparison_and_fallback(tmp_path):
    _run_js_smoke(tmp_path, _extreme_inputs(), r'''
assert.equal(plots.size,10);
assert.equal(document.getElementById('extremeEva').textContent,'30 €');
assert.equal(document.getElementById('extremeBaselineGain').textContent,'20 €');
assert.equal(document.getElementById('extremeInterventions').textContent,'0');
assert.equal(document.getElementById('extremeMeanWeight').textContent,'0,0 %');
assert.equal(plots.get('extremeWeights').traces[0].y[0],0);
assert.equal(plots.get('extremeWeights').traces[1].y[0],0);
assert.match(document.getElementById('extremeProtocol').textContent,/365 jours glissants/);
assert.match(document.getElementById('extremeProtocol').textContent,/28 jours/);
assert.match(document.getElementById('candidateProtocol').textContent,/année déjà étudiée/);
assert.match(selected.model.options.find(x=>x.value==='nuclear_kalman_extreme_governed').label,/positions/);
selected.model.value='nuclear_kalman_extreme_governed';selected.model.onchange();
const extended=document.getElementById('traderTable').children[0];
const headers=extended.children[0].children[0].children.map(x=>x.textContent);
assert.equal(headers[1],'Forecast de base (inchangé)');
assert(headers.includes('Score hausse extrême*'));
assert(headers.includes('Score baisse extrême*'));
assert(headers.includes('Fraction de base'));
assert(headers.includes('Fraction retenue'));
assert(headers.includes('Poids gouvernance'));
assert.equal(extended.children[1].children[0].children.at(-1).textContent,'warmup_baseline');
assert.equal(extended.children[1].children[0].children[1].textContent,'120,00');
assert.equal(extended.children[1].children[0].children[4].textContent,'100,00');
assert.equal(extended.children[1].children[1].children[10].textContent,'—');
extremePolicy.baseline_comparison_paired=false;drawExtreme();
assert.equal(document.getElementById('extremeBaselineGain').textContent,'—');
assert.match(document.getElementById('extremePairedNote').textContent,/non attesté/);
assert(plots.get('extremeBaselineCurve').traces[0].y.every(x=>x===null));
assert(plots.get('extremeSpikes').traces[0].y.every(x=>x===null));
document.getElementById('theme').onclick();
assert.equal(plots.get('extremeWeights').layout.font.color,'#e6edf8');
assert.equal(plots.get('extremeWeights').traces[1].line.color,'#5cd8b7');
console.log('Economic report extreme DOM smoke passed');
''')


def test_extreme_interventions_require_paired_hours_and_full_portfolio(tmp_path):
    inputs = list(_extreme_inputs())
    inputs[4]["extreme_policy"]["summary"] = []
    _run_js_smoke(tmp_path, inputs, r'''
const testedRow=data.rows.find(r=>r.model==='nuclear_kalman_extreme_governed'&&r.paired_eligible);
testedRow.policy_position_fraction=0.5;testedRow.governance_weight=0.5;
testedRow.portfolio_eligible=false;drawExtreme();
assert.equal(document.getElementById('extremeInterventions').textContent,'—');
assert.equal(plots.get('extremeWeights').traces[0].y[0],null);
assert.equal(plots.get('extremeWeights').traces[1].y[0],null);
selected.zone.value='FR';drawExtreme();
assert.equal(document.getElementById('extremeInterventions').textContent,'1');
assert.equal(plots.get('extremeWeights').traces[0].y[0],1);
assert.equal(plots.get('extremeWeights').traces[1].y[0],0.5);
testedRow.sample='live';drawExtreme();
assert.equal(document.getElementById('extremeInterventions').textContent,'—');
assert.equal(plots.get('extremeWeights').traces[0].y[0],null);
console.log('Economic report paired-hours DOM smoke passed');
''')
