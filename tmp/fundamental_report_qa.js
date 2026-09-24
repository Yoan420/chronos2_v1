const vm = require('node:vm');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const bundles = JSON.parse(fs.readFileSync(0, 'utf8'));
const results = [];
for (const B of bundles) {
  const elements = new Map(), events = new Map(), plotUpdates = [];
  function element(id) {
    if (!elements.has(id)) elements.set(id, {id, value:'', innerHTML:'', textContent:'', dataset:{}, options:[], attrs:{}, events:{},
      classList:{toggle(){}}, querySelectorAll(){return [];},
      setAttribute(k,v){this.attrs[k]=v;}, appendChild(e){this.options.push(e);},
      addEventListener(k,f){this.events[k]=f;}});
    return elements.get(id);
  }
  const root = {dataset:{theme:'light'}};
  const plots = [{layout:{xaxis:{},yaxis:{},annotations:[{text:'test'}]}}];
  const document = {documentElement:root, getElementById:element, createElement:()=>element('new'+Math.random()),
    querySelectorAll(selector){return selector==='.js-plotly-plot'?plots:[];}};
  const window = {localStorage:{getItem(){return 'light';},setItem(){}},
    Plotly:{relayout(plot,update){plotUpdates.push(update);}},
    requestAnimationFrame(fn){fn();},setTimeout(fn){fn();},
    addEventListener(k,fn){if(!events.has(k))events.set(k,[]);events.get(k).push(fn);},
    dispatchEvent(event){for(const fn of events.get(event.type)||[])fn(event);}};
  const context=vm.createContext({document,window,console,CustomEvent:class {constructor(type,settings){this.type=type;Object.assign(this,settings);}}});
  vm.runInContext(B.bootstrap,context,{timeout:5000});
  vm.runInContext(B.statistics,context,{timeout:15000});
  vm.runInContext(B.theme,context,{timeout:15000});
  const table=element('statistics-table-container'), price=element('statistics-price-table-container');
  assert.equal(element('statistics-metric-select').value,'mae');
  assert.match(price.innerHTML,/statistics-observed-cell/);
  assert.match(price.innerHTML,/statistics-model-cell/);
  assert.match(price.innerHTML,/statistics-storm-cell/);
  const calendar=element('statistics-price-calendar-container');
  assert.match(calendar.innerHTML,/data-period="2026-09-14"/);
  assert.doesNotMatch(calendar.innerHTML,/data-period="2026-09-15"/);
  for(const sample of ['daily','weekly','monthly']) {
    const select=element('statistics-sample-select');select.value=sample;select.events.change();
    const priceSelect=element('statistics-price-sample-select');priceSelect.value=sample;priceSelect.events.change();
    assert.match(table.innerHTML,/<table/);assert.match(price.innerHTML,/<table/);
  }
  element('statistics-metric-select').value='mean_price';element('statistics-metric-select').events.change();
  assert.match(table.innerHTML,/Prix observé/);
  element('statistics-price-sample-select').value='daily';element('statistics-price-sample-select').events.change();
  const light=price.innerHTML;
  element('theme-toggle').events.click();
  assert.equal(root.dataset.theme,'dark');assert.equal(plotUpdates.at(-1)['font.color'],'#e7edf6');
  assert.notEqual(price.innerHTML,light);
  element('theme-toggle').events.click();
  assert.equal(root.dataset.theme,'light');assert.equal(plotUpdates.at(-1)['font.color'],'#18212b');
  calendar.events.click({target:{closest(){return {dataset:{zone:B.zone,period:'2026-09-14'}};}}});
  assert.ok(element('statistics-price-calendar-detail').innerHTML.length>100);
  results.push({zone:B.zone,policy:B.policy,statistics_samples:['daily','weekly','monthly'],mean_price_metric:true,
    calendar_14sep:true,live_15sep_excluded:true,night_light_and_chart_colors:true,observed_model_storm_columns:true});
}
console.log(JSON.stringify(results));
