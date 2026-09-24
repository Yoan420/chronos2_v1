// Execute the saved comparison's own UI logic with a small DOM/Plotly test double.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const sourcePath = process.argv[2];
const before = fs.readFileSync(sourcePath);
const digest = b => crypto.createHash('sha256').update(b).digest('hex');
const html = before.toString('utf8');
const match = html.match(/<script id="variant-data" type="application\/json">([\s\S]*?)<\/script>/);
assert.ok(match, 'Saved comparison data required.');
const payload = JSON.parse(match[1]);
const comparison = JSON.parse(fs.readFileSync(path.join(path.dirname(sourcePath), 'comparison.json'), 'utf8'));
assert.deepEqual(payload.summary, comparison, 'HTML summary must match frozen comparison JSON.');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const logic = scripts.find(s => s.includes("const P=JSON.parse(document.getElementById('variant-data')"));
assert.ok(logic);
new vm.Script(logic);
const elements = new Map(), plots = new Map();
function element(id) {
  if (!elements.has(id)) {
    const item = {id, value:'', textContent:'', dataset:{}, _html:'', _buttons:null};
    Object.defineProperty(item,'innerHTML',{get(){return this._html;},set(v){this._html=v;this._buttons=null;}});
    Object.defineProperty(item,'options',{get(){return [...this._html.matchAll(/<option(?: value="([^"]*)")?>([^<]*)<\/option>/g)].map(m=>({value:m[1]??m[2]}));}});
    item.querySelectorAll = function(selector) {
      if(selector!=='[data-date]')return [];
      if(this._buttons===null)this._buttons=[...this._html.matchAll(/data-date="([^"]*)"/g)].map(m=>({dataset:{date:m[1]}}));
      return this._buttons;
    };
    elements.set(id,item);
  }
  return elements.get(id);
}
element('variant-data').textContent=match[1];
const root={dataset:{theme:'light'}};
const document={documentElement:root,getElementById:element};
const storage=new Map();
const context=vm.createContext({document,console,localStorage:{getItem(k){return storage.get(k)||null;},setItem(k,v){storage.set(k,v);}},
  getComputedStyle(){return {getPropertyValue(k){return k==='--ink'?(root.dataset.theme==='dark'?'#eeeeee':'#111111'):root.dataset.theme==='dark'?'#151515':'#ffffff';}};},
  Plotly:{react(id,traces,layout){plots.set(id,{traces,layout});}}});
vm.runInContext(logic,context,{timeout:30000});
assert.equal(element('variant').value,'forest');
const results=[];
for(const zone of ['FR','DE','BE','NL']) {
  element('country').value=zone;element('country').onchange();
  assert.match(element('support').textContent,/365\/365/);
  assert.match(element('support').textContent,/8735 heures communes/);
  const controls=[];
  for(const variant of payload.summary.variants) {
    element('variant').value=variant;element('variant').onchange();
    element('day').value='2026-09-14';element('day').onchange();
    const rows=payload.base.map((r,i)=>({...r,i})).filter(r=>r.zone===zone&&r.local_day==='2026-09-14');
    const actual=plots.get('dailyChart').traces.at(-1);
    assert.equal(actual.y.length,24);
    assert.deepEqual(Array.from(actual.y),rows.map(r=>payload.variant_values[variant].candidate_forecast[r.i]));
    const interval=plots.get('dailyChart').traces;
    assert.deepEqual(Array.from(interval[0].y),rows.map(r=>payload.variant_values[variant].candidate_q90[r.i]));
    assert.deepEqual(Array.from(interval[1].y),rows.map(r=>payload.variant_values[variant].candidate_q10[r.i]));
    assert.match(element('dailyMeans').innerHTML,/<table/);
    element('month').value='2026-09';element('month').onchange();
    const button=element('calendar').querySelectorAll('[data-date]').find(b=>b.dataset.date==='2026-09-14');
    assert.ok(button&&button.onclick);button.onclick();
    assert.equal(element('day').value,'2026-09-14');
    element('day').value='2026-09-15';element('day').onchange();
    assert.match(element('dayNote').textContent,/hors métriques historiques/);
    assert.equal(plots.get('dailyChart').traces.at(-1).y.length,24);
    assert.ok(!payload.summary.daily.some(r=>r.zone===zone&&r.model===variant&&r.local_day==='2026-09-15'));
    assert.ok(plots.get('hourlyMae').traces.every(t=>t.x.length===24));
    assert.match(element('shapText').textContent,/indisponible/);
    controls.push(variant);
  }
  root.dataset.theme='light';element('theme').onclick();assert.equal(root.dataset.theme,'dark');
  assert.equal(plots.get('dailyChart').layout.font.color,'#eeeeee');
  element('theme').onclick();assert.equal(root.dataset.theme,'light');
  assert.equal(plots.get('dailyChart').layout.font.color,'#111111');
  results.push({zone,variants:controls,day_and_variant_quantile_traces:true,sep14_calendar_selection:true,
    live15sep_explicitly_outside_statistics:true,hourly_profiles_24_hours:true,light_dark_chart_colors:true});
}
assert.equal(digest(before),digest(fs.readFileSync(sourcePath)));
const result={status:'passed',html_matches_frozen_summary:true,default_variant:'forest',countries:results,
  interactive_logic_test:'saved JavaScript executed in Node VM with DOM/Plotly test doubles; no screenshot or browser pixel QA',
  file_unchanged:true,sha256:digest(before)};
fs.writeFileSync(path.join(path.dirname(sourcePath),'comparison_ui_validation.json'),JSON.stringify(result,null,2),'utf8');
console.log(JSON.stringify({status:'passed',countries:results.length,variant_controls:payload.summary.variants.length,default_variant:'forest'}));
