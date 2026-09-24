"""The amplitude report displays existing proposals, never refits or selects them."""
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
import pytest

from nyx_scarcity.adjustment_analysis import build_adjustments
from nyx_scarcity import adjustment_reporting as report


NODE = Path(r"C:\Users\BQ6757\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")


def fixture():
    frames = []
    for zone in ("FR", "DE"):
        stamps = pd.date_range("2026-09-12 17:00", periods=4, freq="D", tz="UTC")
        frames.append(pd.DataFrame({
            "zone": zone, "timestamp_utc": stamps,
            "forecast_origin_utc": stamps - pd.Timedelta(days=1, hours=11),
            "sample": ["evaluation"] * 3 + ["live"],
            "actual": [110., 200., 70., np.nan], "forecast": 100., "benchmark_forecast": 120.,
            "candidate_forecast": [100., 100., 150., 100.],
            "raw_correction": [0., 100., 100., 100.], "bounded_correction": [0., 100., 100., 100.],
            "applied_correction": [0., 0., 50., 0.], "selected_weight": [0., 0., .5, 0.],
            "expert_ready": [False, True, True, True], "spike_probability": [.1, .8, .8, .8],
            "probability_gate": .6, "gate_reason": "annual_nonregression_refused"}))
    frame = pd.concat(frames, ignore_index=True)
    return {key: frame.copy(deep=True) for key in ("hgb_v1", "xgb_unweighted_fixed", "xgb_weighted_dwt")}


def payload_from_file(path):
    document = path.read_text(encoding="utf-8")
    text = re.search(r'<script id="adjustment-data" type="application/json">(.*?)</script>', document, re.S).group(1)
    return document, text, json.loads(text)


def test_render_preserves_projections_metrics_and_original_governance(tmp_path):
    long, summary = build_adjustments(fixture())
    before, prior = long.copy(deep=True), deepcopy(summary)
    path = report.render_adjustments(long, summary, {"diagnostic": True}, tmp_path/"report.html")
    _, _, payload = payload_from_file(path)
    assert len(payload["base"]) == 8
    point = next(i for i, row in enumerate(payload["base"]) if row["zone"] == "FR" and row["local_day"] == "2026-09-13")
    model = payload["variant_values"]["xgb_unweighted_fixed"]
    assert model["proposal25"][point] == 125
    assert model["proposal50"][point] == 150
    assert model["proposal100"][point] == 200
    assert model["candidate_forecast"][point] == 100
    assert model["applied_correction"][point] == model["selected_weight"][point] == 0
    assert payload["summary"]["overall"]["paired_hours"] == 6
    assert summary == prior
    pd.testing.assert_frame_equal(long, before)


def test_compact_payload_excludes_hundreds_of_features_and_quantiles(tmp_path):
    long, summary = build_adjustments(fixture())
    extra = pd.DataFrame({f"training_feature_{i:03}": np.arange(len(long)) for i in range(250)})
    long = pd.concat([long, extra], axis=1)
    document, text, payload = payload_from_file(report.render_adjustments(long, summary, {}, tmp_path/"compact.html"))
    assert "training_feature_" not in document
    assert set(payload["base"][0]) == set(report._BASE_COLUMNS)
    assert set(payload["variant_values"]["hgb_v1"]) == set(report._MODEL_COLUMNS)
    assert all(len(values) == len(payload["base"]) for model in payload["variant_values"].values() for values in model.values())
    assert not any("q10" in name or "q90" in name for name in payload["variant_values"]["hgb_v1"])
    assert "NaN" not in text and "Infinity" not in text


def test_missing_observation_remains_empty_and_live_is_not_scored(tmp_path):
    predictions = fixture()
    for frame in predictions.values():
        frame.loc[(frame.zone == "FR") & (frame["sample"] == "evaluation") & (frame.actual == 200), "actual"] = np.nan
        frame.loc[frame["sample"] == "live", "actual"] = 10000.
    rows, summary = build_adjustments(predictions)
    _, _, payload = payload_from_file(report.render_adjustments(rows, summary, {}, tmp_path/"missing.html"))
    assert summary["overall"]["paired_hours"] == 5
    assert summary["by_zone"]["FR"]["annual"]["nyx"]["mean_observed_eur_mwh"] == 90
    assert any(row["actual"] is None for row in payload["base"])
    assert any(row["sample"] == "live" and row["actual"] == 10000 for row in payload["base"])


@pytest.mark.parametrize("column", ["proposal25", "proposal50", "proposal100"])
def test_inconsistent_fixed_amplitude_rejected(column):
    rows, summary = build_adjustments(fixture())
    rows.loc[0, column] += 1.
    with pytest.raises(ValueError, match="declared fraction"):
        report._display_payload(rows, summary)


@pytest.mark.parametrize("corruption", ["fractions", "variant", "missing_column"])
def test_summary_contract_rejected(corruption):
    rows, summary = build_adjustments(fixture())
    if corruption == "fractions": summary["fractions"]["proposal25"] = .3
    elif corruption == "variant": summary["variants"] = ["hgb_v1"]
    else: rows = rows.drop(columns="bounded_correction")
    with pytest.raises(ValueError): report._display_payload(rows, summary)


def test_html_safe_offline_and_honest(tmp_path):
    rows, summary = build_adjustments(fixture())
    rows.loc[0, "gate_reason"] = "</script><script>bad()</script>"
    document, text, _ = payload_from_file(report.render_adjustments(rows, summary,
        {"source": "</pre><script>bad()</script>", "missing": float("nan")}, tmp_path/"safe.html"))
    assert "<script>bad()" not in document
    assert "&lt;script&gt;bad()" in document
    assert "\\u003c/script\\u003e" in text
    assert "<script src=" not in document
    for marker in ("html[data-theme=dark]", "nyx-adjustment-theme", "NON gouvernées", "AUCUNE ACTIVATION",
                   "Aucun alpha gagnant", "ni un P&amp;L ni une EVA", "exclue des scores historiques",
                   'data-report-section="adjustment-hourly"', 'data-report-section="adjustment-statistics"',
                   'data-report-section="adjustment-risk"', 'data-report-section="adjustment-tails"'):
        assert marker in document


def test_real_browser_script_runs_all_amplitude_controls(tmp_path):
    if not NODE.is_file(): pytest.skip("Bundled Node unavailable")
    rows, summary = build_adjustments(fixture())
    _, payload, _ = payload_from_file(report.render_adjustments(rows, summary, {}, tmp_path/"controls.html"))
    harness = r'''
const vm=require('vm'),assert=require('assert'),elements={},plots=[];
function element(id){if(elements[id])return elements[id];let content='',value='';const e={textContent:'',options:[]};
Object.defineProperty(e,'value',{get(){return value},set(v){value=v}});Object.defineProperty(e,'innerHTML',{get(){return content},set(v){content=v;const opts=[...v.matchAll(/<option(?: value="([^"]*)")?[^>]*>(.*?)<\/option>/g)].map(m=>({value:m[1]||m[2]}));if(opts.length){e.options=opts;if(!opts.some(o=>o.value===value))value=opts[0].value}}});return elements[id]=e;}
const document={documentElement:{dataset:{}},getElementById:element};element('adjustment-data').textContent=PAYLOAD;
const sandbox={document,console,Plotly:{react(id,traces,layout){plots.push({id,traces,layout})}},getComputedStyle(){return {getPropertyValue(){return document.documentElement.dataset.theme==='dark'?'#123456':'#abcdef'}}},localStorage:{getItem(){return 'dark'},setItem(){}}};vm.runInNewContext(SCRIPT,sandbox);
assert.equal(element('country').value,'FR');assert.equal(element('variant').value,'xgb_unweighted_fixed');assert.equal(element('alpha').value,'proposal25');assert.equal(element('day').value,'2026-09-14');
assert(element('window').textContent.includes('3 heures communes'));assert(element('window').textContent.includes('Prix moyen observé'));assert(!element('window').textContent.includes('{'));
assert(element('annual').innerHTML.includes('NON gouvernée'));assert(element('annual').innerHTML.includes('Prix moyen prévu'));assert(element('hourly').innerHTML.includes('Prix proposé 100 %'));assert(element('hourly').innerHTML.includes('Motif de gouvernance'));
let chart=plots.filter(p=>p.id==='priceChart').at(-1);assert.equal(chart.traces.at(-1).y[0],125);assert.equal(chart.traces.at(-2).y[0],150);
element('alpha').value='proposal100';element('alpha').onchange();chart=plots.filter(p=>p.id==='priceChart').at(-1);assert.equal(chart.traces.at(-1).y[0],200);assert.equal(chart.traces.at(-2).y[0],150);assert(element('hourly').innerHTML.includes('Prix proposé 100 %'));
element('alpha').value='proposal50';element('alpha').onchange();assert.equal(plots.filter(p=>p.id==='priceChart').at(-1).traces.at(-1).y[0],150);
element('variant').value='hgb_v1';element('variant').onchange();assert(plots.filter(p=>p.id==='priceChart').at(-1).layout.title.text.includes('HGB'));
element('country').value='DE';element('country').onchange();assert(plots.filter(p=>p.id==='priceChart').at(-1).layout.title.text.startsWith('DE'));
element('theme').onclick();assert.equal(document.documentElement.dataset.theme,'light');assert.equal(plots.filter(p=>p.id==='priceChart').at(-1).layout.paper_bgcolor,'#abcdef');
element('day').value='2026-09-15';element('day').onchange();assert(element('dayNote').textContent.includes('exclue des scores historiques'));assert.equal(plots.filter(p=>p.id==='priceChart').at(-1).traces[0].y[0],null);
'''.replace("PAYLOAD", json.dumps(payload)).replace("SCRIPT", json.dumps(report._SCRIPT))
    done = subprocess.run([str(NODE), "-"], input=harness, capture_output=True, encoding="utf-8")
    assert done.returncode == 0, done.stderr
