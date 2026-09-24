"""Temporary read-only-source QA of three completed variants; not a final report."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from nyx_scarcity.variant_reporting import build_comparison, render_comparison, _SCRIPT

SOURCE = ROOT / "runs/experiments/nyx_scarcity_v1/variants/snapshots/20260914T133114Z_8f239442"
OUTPUT = Path(__file__).resolve().parent
VARIANTS = ("xgb_unweighted_fixed", "xgb_weighted_fixed")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
protected = {path: digest(path) for path in (ROOT / "nyx_scarcity" / "reporting.py", ROOT / "nyx_scarcity" / "variant_reporting.py")}
protected[SOURCE / "manifest.json"] = digest(SOURCE / "manifest.json")
for filename, expected in manifest["input_files"].items():
    path = SOURCE / filename
    assert digest(path) == expected, filename
    protected[path] = expected
predictions = {"hgb_v1": pd.read_parquet(SOURCE / "control_predictions.parquet")}
explanations, audits = {}, {}
for variant in VARIANTS:
    folder = SOURCE / variant
    record = json.loads((folder / "results_manifest.json").read_text(encoding="utf-8"))
    assert record["status"] == "completed" and record["variant_id"] == variant
    assert record["suite_manifest_sha256"] == protected[SOURCE / "manifest.json"]
    for filename, expected in record["result_files"].items():
        path = folder / filename
        assert digest(path) == expected, path
        protected[path] = expected
    predictions[variant] = pd.read_parquet(folder / "predictions.parquet")
    explanations[variant] = {"summary": json.loads((folder / "shap_summary.json").read_text(encoding="utf-8"))}
    audits[variant] = json.loads((folder / "model_audit.json").read_text(encoding="utf-8"))

summary = build_comparison(predictions)
report_path = render_comparison(predictions, summary,
    {"preview_only": True, "not_final_suite_report": True, "source_snapshot": str(SOURCE),
     "models": audits, "data": json.loads((SOURCE / "data_audit.json").read_text(encoding="utf-8"))},
    OUTPUT / "PREVIEW_HGB_TWO_FIXED_NOT_FINAL.html", explanations=explanations)
document = report_path.read_text(encoding="utf-8")
payload_text = re.search(r'<script id="variant-data" type="application/json">(.*?)</script>', document, re.S).group(1)
payload = json.loads(payload_text)
assert "training_feature_" not in payload_text
assert set(payload["base"][0]) == {"zone", "timestamp_utc", "forecast_origin_utc", "sample", "local_day", "local_hour",
                                    "local_label", "actual", "forecast", "benchmark_forecast", "q10", "q90"}
assert len(payload["variant_values"]) == 3
for variant, columns in payload["variant_values"].items():
    assert len(columns) == 9
    assert all(len(values) == len(payload["base"]) for values in columns.values())

local_proof = {}
for variant in VARIANTS:
    explanation = payload["explanations"][variant]
    assert explanation["numeric_display_verified"]
    cases = explanation["local_waterfalls"]
    assert {case["zone"] for case in cases} == {"FR", "DE", "BE", "NL"}
    assert {case["local_day"] for case in cases} == {"2026-09-14", "2026-09-15"}
    assert all("19:00" in case["local_label"] for case in cases)
    assert all(np.isclose(case["reconstructed_log_odds"], case["calibrated_log_odds"], atol=2e-5, rtol=2e-5) for case in cases)
    local_proof[variant] = {"status": explanation["status"], "global_features": len(explanation["global_importance"]),
                            "verified_local_cases": len(cases),
                            "maximum_grouped_reconstruction_error": max(abs(c["reconstructed_log_odds"]-c["calibrated_log_odds"]) for c in cases)}
means = [row for row in summary["daily"] if row["local_day"] == "2026-09-14"]
assert len(means) == 4*5
for row in means:
    assert np.isfinite(row["mean_forecast_eur_mwh"]) and np.isfinite(row["mean_observed_eur_mwh"])
    assert row["paired_hours"] == 24

payload_path = OUTPUT / "preview_payload.json"
payload_path.write_text(payload_text, encoding="utf-8")
harness = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert'),elements={},plots=[];
function element(id){if(elements[id])return elements[id];let content='',value='';const e={textContent:'',options:[],querySelectorAll(){return []}};
Object.defineProperty(e,'value',{get(){return value},set(v){value=v}});Object.defineProperty(e,'innerHTML',{get(){return content},set(v){content=v;const opts=[...v.matchAll(/<option(?: value="([^"]*)")?[^>]*>(.*?)<\/option>/g)].map(m=>({value:m[1]||m[2]}));if(opts.length){e.options=opts;if(!opts.some(o=>o.value===value))value=opts[0].value}}});return elements[id]=e;}
const raw=fs.readFileSync(PAYLOAD_PATH,'utf8'),payload=JSON.parse(raw);const document={documentElement:{dataset:{}},getElementById:element};element('variant-data').textContent=raw;
const sandbox={document,console,Plotly:{react(id,traces,layout){plots.push({id,traces,layout})}},getComputedStyle(){return {getPropertyValue(){return document.documentElement.dataset.theme==='dark'?'#123456':'#abcdef'}}},localStorage:{getItem(){return 'dark'},setItem(){}}};vm.runInNewContext(SCRIPT,sandbox);
const latestObserved=payload.base.filter(r=>r.zone==='FR'&&typeof r.actual==='number').map(r=>r.local_day).sort().at(-1);assert.equal(element('country').value,'FR');assert.equal(element('variant').value,'xgb_unweighted_fixed');assert.equal(element('day').value,latestObserved);assert(element('dailyMeans').innerHTML.includes('Prix moyen observé'));assert(plots.filter(p=>p.id==='hourlyMean').at(-1).traces.some(t=>t.name==='Observé'));element('day').value='2026-09-14';element('day').onchange();
function verifyLocal(v,z,day){const chart=plots.filter(p=>p.id==='shapLocal').at(-1),t=chart.traces[0],selected=payload.explanations[v].local_waterfalls[Number(element('localCase').value)];assert.equal(t.type,'waterfall');assert.equal(t.orientation,'h');assert.equal(selected.zone,z);assert.equal(selected.local_day,day);assert.equal(t.measure[0],'absolute');assert.equal(t.measure.at(-1),'total');assert(t.y.some(n=>n.startsWith('Autres (')));assert(t.x.length<=13);const total=t.x.slice(0,-1).reduce((a,b)=>a+b,0);assert(Math.abs(total-selected.calibrated_log_odds)<2e-5+2e-5*Math.abs(selected.calibrated_log_odds));assert(element('localNote').textContent.includes('probabilité calibrée'));assert.equal(chart.layout.yaxis.automargin,true);assert.equal(chart.layout.height,560);}
verifyLocal('xgb_unweighted_fixed','FR','2026-09-14');assert.equal(plots.filter(p=>p.id==='shap').at(-1).layout.height,680);
element('theme').onclick();assert.equal(document.documentElement.dataset.theme,'light');assert.equal(plots.filter(p=>p.id==='shap').at(-1).layout.paper_bgcolor,'#abcdef');
element('variant').value='xgb_weighted_fixed';element('variant').onchange();verifyLocal('xgb_weighted_fixed','FR','2026-09-14');
element('country').value='DE';element('country').onchange();element('day').value='2026-09-14';element('day').onchange();verifyLocal('xgb_weighted_fixed','DE','2026-09-14');
element('day').value='2026-09-15';element('day').onchange();verifyLocal('xgb_weighted_fixed','DE','2026-09-15');assert(element('dayNote').textContent.includes('hors métriques historiques'));
element('month').value='2026-06';element('month').onchange();assert(element('calendar').innerHTML.includes('2026-06-26'));
element('variant').value='hgb_v1';element('variant').onchange();assert.equal(plots.filter(p=>p.id==='dailyChart').at(-1).traces.at(-1).name,'HGB original — snapshot figé');assert(element('shapText').textContent.includes('aucune valeur'));assert.equal(plots.filter(p=>p.id==='shapLocal').at(-1).traces.length,0);
console.log(JSON.stringify({javascript_controls:'passed',initial_observed_day:latestObserved,live_day:'2026-09-15',countries_tested:['FR','DE'],variants_tested:['hgb_v1','xgb_unweighted_fixed','xgb_weighted_fixed'],waterfall_identity:'verified',night_light:'passed',daily_mean_columns:'present',calendar_june_26:'present',plot_calls:plots.length}));
'''.replace("PAYLOAD_PATH", json.dumps(str(payload_path))).replace("SCRIPT", json.dumps(_SCRIPT))
harness = harness.replace(r"<\\/option>", r"<\/option>")
node = Path(r"C:\Users\BQ6757\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
completed = subprocess.run([str(node), "-"], input=harness, text=True, encoding="utf-8", capture_output=True, timeout=90)
assert completed.returncode == 0, completed.stderr[-10000:]
node_qa = json.loads(completed.stdout.strip())
assert all(digest(path) == expected for path, expected in protected.items()), "Protected source/module changed during preview"
result = {"preview_only": True, "not_final_suite_report": True, "report": str(report_path),
          "report_bytes": report_path.stat().st_size, "payload_bytes": payload_path.stat().st_size,
          "shared_payload_rows": len(payload["base"]), "variants": list(predictions),
          "windows": summary["windows"], "local_shap": local_proof,
          "mean_prices_2026_09_14": means, "javascript": node_qa,
          "sealed_modules_and_source_artifacts_unchanged": True}
(OUTPUT / "preview_qa_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k != "mean_prices_2026_09_14"}, ensure_ascii=True, indent=2))
