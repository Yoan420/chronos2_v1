"""Read-only-source real-data amplitude QA, temporary and not a final suite report."""
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
from nyx_scarcity.adjustment_analysis import build_adjustments
from nyx_scarcity.adjustment_reporting import render_adjustments, _SCRIPT

SOURCE = ROOT / "runs/experiments/nyx_scarcity_v1/variants/snapshots/20260914T133114Z_8f239442"
OUTPUT = Path(__file__).resolve().parent
VARIANTS = ("xgb_unweighted_fixed", "xgb_weighted_fixed")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
protected = {path: digest(path) for path in (ROOT / "nyx_scarcity").glob("*.py")}
protected[SOURCE / "manifest.json"] = digest(SOURCE / "manifest.json")
for filename, expected in manifest["input_files"].items():
    path = SOURCE / filename
    assert digest(path) == expected, filename
    protected[path] = expected
predictions = {"hgb_v1": pd.read_parquet(SOURCE / "control_predictions.parquet")}
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

rows, summary = build_adjustments(predictions)
for variant in summary["variants"]:
    selected = rows.loc[rows.variant_id.eq(variant) & rows.common_support]
    for name in ("proposal25", "proposal50", "proposal100"):
        score = summary["overall"]["annual"]["variants"][variant][name]
        assert np.isclose((selected[name] - selected.actual).abs().mean(), score["mae_eur_mwh"])
        assert np.isclose(selected.actual.mean(), score["mean_observed_eur_mwh"])
        assert np.isclose(selected[name].mean(), score["mean_forecast_eur_mwh"])
    assert not selected["sample"].eq("live").any()
    for zone in summary["by_zone"]:
        local = selected.loc[selected.zone.eq(zone)]
        assert len(local) == 8735
        assert np.isclose(local.actual.mean(), summary["by_zone"][zone]["annual"]["nyx"]["mean_observed_eur_mwh"])

report = render_adjustments(rows, summary,
    {"preview_only": True, "not_final_suite_report": True, "source": str(SOURCE)},
    OUTPUT / "PREVIEW_ADJUSTMENTS_HGB_TWO_FIXED_NOT_FINAL.html")
document = report.read_text(encoding="utf-8")
payload_text = re.search(r'<script id="adjustment-data" type="application/json">(.*?)</script>', document, re.S).group(1)
payload = json.loads(payload_text)
assert len(payload["base"]) == 35136
assert len(payload["variant_values"]) == 3
assert all(len(columns) == 13 for columns in payload["variant_values"].values())
assert not any("feature_" in key for key in payload["base"][0])
assert not any("feature_" in key for columns in payload["variant_values"].values() for key in columns)
payload_path = OUTPUT / "adjustment_preview_payload.json"
payload_path.write_text(payload_text, encoding="utf-8")

harness = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert'),elements={},plots=[];
function element(id){if(elements[id])return elements[id];let content='',value='';const e={textContent:'',options:[]};
Object.defineProperty(e,'value',{get(){return value},set(v){value=v}});Object.defineProperty(e,'innerHTML',{get(){return content},set(v){content=v;const opts=[...v.matchAll(/<option(?: value="([^"]*)")?[^>]*>(.*?)<\/option>/g)].map(m=>({value:m[1]||m[2]}));if(opts.length){e.options=opts;if(!opts.some(o=>o.value===value))value=opts[0].value}}});return elements[id]=e;}
const raw=fs.readFileSync(PAYLOAD_PATH,'utf8'),payload=JSON.parse(raw),document={documentElement:{dataset:{}},getElementById:element};element('adjustment-data').textContent=raw;
const sandbox={document,console,Plotly:{react(id,traces,layout){plots.push({id,traces,layout})}},getComputedStyle(){return {getPropertyValue(){return document.documentElement.dataset.theme==='dark'?'#123456':'#abcdef'}}},localStorage:{getItem(){return 'dark'},setItem(){}}};vm.runInNewContext(SCRIPT,sandbox);
assert.equal(element('country').value,'FR');assert.equal(element('variant').value,'xgb_unweighted_fixed');assert.equal(element('alpha').value,'proposal25');assert.equal(element('day').value,'2026-09-15');
const tests=[];
for(const country of Object.keys(payload.summary.by_zone)){element('country').value=country;element('country').onchange();assert(element('window').textContent.includes('Prix moyen observé'));assert(element('window').textContent.includes('25 heures présentes non appariées'));assert(!element('window').textContent.includes('{'));
 for(const variant of payload.summary.variants){element('variant').value=variant;element('variant').onchange();element('day').value='2026-09-14';element('day').onchange();
  const dayRows=payload.base.map((r,i)=>({...r,i})).filter(r=>r.zone===country&&r.local_day==='2026-09-14');assert.equal(dayRows.length,24);
  for(const alpha of ['proposal25','proposal50','proposal100']){element('alpha').value=alpha;element('alpha').onchange();let chart=plots.filter(p=>p.id==='priceChart').at(-1);assert.equal(chart.traces.length,5);const columns=payload.variant_values[variant];for(let j=0;j<24;j++){const row=dayRows[j];assert.equal(chart.traces[0].y[j],row.actual);assert.equal(chart.traces[1].y[j],row.forecast);assert.equal(chart.traces[2].y[j],row.benchmark_forecast);assert.equal(chart.traces[3].y[j],columns.candidate_forecast[row.i]);assert.equal(chart.traces[4].y[j],columns[alpha][row.i]);}
   assert(element('hourly').innerHTML.includes('Prix proposé 100 %'));assert(element('hourly').innerHTML.includes('Ajustement appliqué'));assert(element('hourly').innerHTML.includes('Motif de gouvernance'));assert.equal((element('hourly').innerHTML.match(/<tr>/g)||[]).length,25);assert(element('annual').innerHTML.includes('NON gouvernée'));assert(element('tails').innerHTML.includes('ex post'));tests.push({country,variant,alpha});
  }
 }
}
element('theme').onclick();assert.equal(document.documentElement.dataset.theme,'light');assert.equal(plots.filter(p=>p.id==='priceChart').at(-1).layout.paper_bgcolor,'#abcdef');element('theme').onclick();assert.equal(document.documentElement.dataset.theme,'dark');
element('day').value='2026-09-15';element('day').onchange();assert(element('dayNote').textContent.includes('exclue des scores historiques'));assert(plots.filter(p=>p.id==='priceChart').at(-1).traces[0].y.every(Number.isFinite));
process.stdout.write(JSON.stringify({control_combinations:tests.length,all_countries:true,all_fixed_variants:true,all_fractions:true,night_light:true,observed_live_excluded_from_metrics:true,hourly_graph_prices_verified:true,hourly_table_24_rows:true,mean_observed_caption:true,unpaired_hours_caption:true}));
'''.replace("PAYLOAD_PATH", json.dumps(str(payload_path))).replace("SCRIPT", json.dumps(_SCRIPT))
node = Path(r"C:\Users\BQ6757\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
done = subprocess.run([str(node), "-"], input=harness, capture_output=True, encoding="utf-8")
assert done.returncode == 0, done.stderr
for path, expected in protected.items():
    assert digest(path) == expected, f"Source/module changed during preview: {path}"
result = {"preview_only": True, "report": str(report), "report_bytes": report.stat().st_size,
          "payload_bytes": len(payload_text.encode("utf-8")), "javascript": json.loads(done.stdout),
          "overall_annual": summary["overall"]["annual"], "source_and_sealed_modules_unchanged": True}
(OUTPUT / "adjustment_preview_qa_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
print(json.dumps({key: value for key, value in result.items() if key != "overall_annual"}, indent=2))
print("NYX MAE:", summary["overall"]["annual"]["nyx"]["mae_eur_mwh"])
print("Unweighted 100% MAE:", summary["overall"]["annual"]["variants"]["xgb_unweighted_fixed"]["proposal100"]["mae_eur_mwh"])
