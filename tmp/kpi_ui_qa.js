// Execute the actual offline report JavaScript against a deterministic DOM double.
// This checks behaviour/data parity, not pixel-level browser rendering.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert/strict');
const input = process.argv[2];
if (!input) throw new Error('Usage: node kpi_ui_qa.js report.html');
const html = fs.readFileSync(input, 'utf8');
assert(!/<script[^>]+src\s*=/i.test(html), 'external script runtime');
assert(!/<link[^>]+href\s*=\s*["']https?:/i.test(html), 'external style runtime');
assert(!/\b(?:fetch|XMLHttpRequest|WebSocket)\s*\(/.test(html), 'unexpected network API');
const dataMatch = html.match(/<script id="kpi-data" type="application\/json">([\s\S]*?)<\/script>/);
assert(dataMatch, 'embedded JSON missing');
const payload = JSON.parse(dataMatch[1]);
const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].filter(m => !m[0].includes('id="kpi-data"'));
assert.equal(scripts.length, 1);
class Element {
  constructor(id) { this.id=id; this.value=''; this.textContent=''; this.innerHTML=''; this.hidden=false; this.dataset={}; this.listeners={}; this.attributes={}; }
  addEventListener(name, fn) { this.listeners[name]=fn; }
  setAttribute(name, value) { this.attributes[name]=value; }
}
const nodes = new Map();
const get = id => { if(!nodes.has(id)) nodes.set(id,new Element(id)); return nodes.get(id); };
get('kpi-data').textContent=dataMatch[1];
get('country').value='ALL'; get('period').value=Object.keys(payload.periods).includes('365')?'365':Object.keys(payload.periods)[0]; get('family').value='all';
const doc={documentElement:{dataset:{}}, getElementById:get,
  querySelectorAll(selector) {
    if(selector==='[data-sort]') return [...get('kpi-table').innerHTML.matchAll(/data-sort="([^"]+)"/g)].map(m=>{const b=new Element(m[1]);b.dataset.sort=m[1];return b;});
    return [];
  }};
const context=vm.createContext({document:doc, localStorage:{getItem(){throw Error('storage denied');},setItem(){throw Error('storage denied');}}, console});
vm.runInContext(scripts[0][1],context,{timeout:10000});
const exec=s=>vm.runInContext(s,context,{timeout:10000});
const fmt=v=>typeof v==='number'&&Number.isFinite(v)?v.toLocaleString('fr-FR',{minimumFractionDigits:2,maximumFractionDigits:2}):'—';
const keys=['mae_eur_mwh','rmse_eur_mwh','win_rate_hour_pct','win_rate_day_mae_pct','mae_day_mean_price_eur_mwh','win_rate_day_mean_price_pct','mean_price_eur_mwh'];
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let checks=0;
function validateCurrent() {
  const period=payload.periods[get('period').value], zone=get('country').value;
  const selected=period.rows.filter(r=>r.zone===zone),storm=selected.find(r=>r.model_id==='__storm__');
  const trs=[...get('kpi-table').innerHTML.matchAll(/<tr[^>]*data-model="([^"]+)"[^>]*>([\s\S]*?)<\/tr>/g)];
  const production=payload.catalog.find(x=>x.kind==='production');
  if(production && selected.some(r=>r.model_id===production.id)) assert.equal(trs[0][1],esc(production.id),'production pinned');
  const [sortKey, ascending]=exec('[sortKey,sortAscending]');
  const family=get('family').value,labels=Object.fromEntries(payload.catalog.map(x=>[x.id,x]));
  const expectedOrder=selected.filter(r=>r.model_id!=='__storm__'&&(family==='all'||labels[r.model_id]?.family===family||labels[r.model_id]?.kind==='production'));
  expectedOrder.sort((a,b)=>{
    const ap=labels[a.model_id]?.kind==='production',bp=labels[b.model_id]?.kind==='production';
    if(ap!==bp)return ap?-1:1;
    const av=a[sortKey],bv=b[sortKey],af=typeof av==='number'&&Number.isFinite(av),bf=typeof bv==='number'&&Number.isFinite(bv);
    if(!af||!bf)return af?-1:bf?1:0;
    return (av-bv)*(ascending?1:-1)||a.model_id.localeCompare(b.model_id);
  });
  assert.deepEqual(trs.filter(r=>r[1]!=='__storm__').map(r=>r[1]),expectedOrder.map(r=>esc(r.model_id)),'selected models and sorting must match JSON values');
  for(const [,modelId,body] of trs) {
    const row=selected.find(r=>esc(r.model_id)===modelId); assert(row,'rendered row without JSON');
    const cells=[...body.matchAll(/<td([^>]*)>([\s\S]*?)<\/td>/g)].slice(1);
    assert.equal(cells.length,keys.length);
    keys.forEach((key,i)=>{
      const suffix=key.startsWith('win_')&&typeof row[key]==='number'&&row.model_id!=='__storm__'?' %':'';
      assert.equal(cells[i][2],row.model_id==='__storm__'&&key.startsWith('win_')?'—':fmt(row[key])+suffix,`${modelId}/${key} JSON/HTML parity`);
      if(row.model_id!=='__storm__' && key.startsWith('win_') && typeof row[key]==='number') {
        const group={win_rate_hour_pct:'hour',win_rate_day_mae_pct:'day_mae',win_rate_day_mean_price_pct:'day_mean_price'}[key];
        const wins=row['wins_'+group],losses=row['losses_'+group];
        const expected=wins>losses?'good':wins<losses?'bad':'';
        assert(cells[i][1].includes(`class="${expected}"`),'win colour uses wins/losses, not 50%');
        assert(cells[i][1].includes(`${wins} victoires`),'win count tooltip');
      }
      if(key==='mean_price_eur_mwh') assert(!/class="(?:good|bad)"/.test(cells[i][1]),'mean price has no lower-is-better colouring');
    });
  }
  if(period.economic) {
    assert.equal(get('economic-section').hidden,false);
    const econRows=period.economic.rows.filter(r=>r.zone===zone);
    const econTrs=[...get('economic-table').innerHTML.matchAll(/<tr[^>]*data-model="([^"]+)"[^>]*>([\s\S]*?)<\/tr>/g)];
    assert.deepEqual(econTrs.map(r=>r[1]),trs.map(r=>r[1]),'economic and price tables keep same model order');
    for(const [,modelId,body] of econTrs) {
      const row=econRows.find(r=>esc(r.model_id)===modelId)||{};
      const cells=[...body.matchAll(/<td([^>]*)>([\s\S]*?)<\/td>/g)].slice(1);
      const fields=['pnl_net_eur','gain_vs_storm_eur','gain_vs_storm_per_potential_mwh'];
      assert.equal(cells.length,fields.length);
      fields.forEach((field,i)=>{
        assert.equal(cells[i][2],fmt(row[field]),`${modelId}/${field} economic JSON/HTML parity`);
        if(i && modelId!=='__storm__') {
          const value=row[field],finite=typeof value==='number'&&Number.isFinite(value);
          const cls=finite?(value>1e-9?'good':value< -1e-9?'bad':''):'';
          assert(cells[i][1].includes(`class="${cls}"`),'economic gain colour follows its sign');
        }
      });
    }
    assert(get('economic-note').textContent.includes('MWh potentiels communs'),'economic denominator disclosed');
    assert(get('economic-note').textContent.includes('Aucun résultat n’est annualisé'),'partial support is not annualised');
  } else assert.equal(get('economic-section').hidden,true);
  checks++;
}
validateCurrent();
for(const period of Object.keys(payload.periods)) {
  get('period').value=period;
  for(const zone of ['ALL',...payload.zones]) {
    get('country').value=zone;exec('render()');validateCurrent();
  }
}
get('country').value='ALL';get('period').value=Object.keys(payload.periods).includes('365')?'365':Object.keys(payload.periods)[0];exec('render()');
const support=[get('hours').textContent,get('days').textContent,get('observed-price').textContent,get('coverage-data').textContent,get('economic-note').textContent,get('economic-audit').textContent];
for(const family of [...new Set(payload.catalog.map(x=>x.family))]) {
  get('family').value=family;exec('render()');validateCurrent();
  assert.deepEqual([get('hours').textContent,get('days').textContent,get('observed-price').textContent,get('coverage-data').textContent,get('economic-note').textContent,get('economic-audit').textContent],support,'family visibility changed common price or economic support');
}
get('family').value='all';
for(const key of keys) {exec(`sortKey=${JSON.stringify(key)};sortAscending=true;render()`);validateCurrent();exec('sortAscending=false;render()');validateCurrent();}
exec("theme('dark')");assert.equal(doc.documentElement.dataset.theme,'dark');assert.equal(get('theme-toggle').attributes['aria-pressed'],'true');
get('theme-toggle').listeners.click();assert.equal(doc.documentElement.dataset.theme,'light');
// Synthetic zero/empty/tied fixtures exercise falsy-safe display regardless of real report values.
const primary=payload.catalog.find(x=>x.kind==='production')||payload.catalog[0];
exec(`P.periods[$('period').value].rows.filter(r=>r.zone==='ALL'&&r.model_id===${JSON.stringify(primary.id)}).forEach(r=>{r.mae_eur_mwh=0;r.mean_price_eur_mwh=0;r.win_rate_hour_pct=0;r.wins_hour=0;r.losses_hour=0;r.ties_hour=r.n_hours;});render();`);
const zeroBody=[...get('kpi-table').innerHTML.matchAll(/<tr[^>]*data-model="([^"]+)"[^>]*>([\s\S]*?)<\/tr>/g)].find(x=>x[1]===esc(primary.id))[2];
assert(zeroBody.includes('>0,00</td>'),'zero rendered as missing');
assert(zeroBody.includes('class="" title="0 victoires · 0 défaites'),'all-ties should be neutral');
assert(zeroBody.includes('>0,00 %</td>'),'zero win rate rendered as missing');
if(exec("Boolean(P.periods[$('period').value].economic)")) {
  exec(`P.periods[$('period').value].economic.rows.filter(r=>r.zone==='ALL'&&r.model_id===${JSON.stringify(primary.id)}).forEach(r=>{r.pnl_net_eur=0;r.gain_vs_storm_eur=null;r.gain_vs_storm_per_potential_mwh=0;});render();`);
  const economicBody=[...get('economic-table').innerHTML.matchAll(/<tr[^>]*data-model="([^"]+)"[^>]*>([\s\S]*?)<\/tr>/g)].find(x=>x[1]===esc(primary.id))[2];
  const cells=[...economicBody.matchAll(/<td([^>]*)>([\s\S]*?)<\/td>/g)].slice(1);
  assert.deepEqual(cells.map(c=>c[2]),['0,00','—','0,00'],'economic zero and unavailable values are distinct');
  assert(cells.every(c=>c[1].includes('class=""')),'economic zero/missing values are neutral');
}
get('country').value='ABSENT';exec('render()');assert.equal(get('empty-state').hidden,false);assert.equal(get('hours').textContent,'—');
console.log(JSON.stringify({status:'passed',checks,periods:Object.keys(payload.periods).length,zones:payload.zones.length,models:payload.catalog.length,pixel_visual_qa:false}));
