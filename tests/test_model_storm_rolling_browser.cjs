// End-to-end offline HTML QA. Set PLAYWRIGHT_MODULE when using the bundled runtime.
const assert=require('node:assert/strict');
const path=require('node:path');
const fs=require('node:fs');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');

(async()=>{
  const input=path.resolve(process.argv[2]||'runs/reports/model_storm/CWE_Model_Storm_2026-09-23.html');
  const out=path.resolve(process.argv[3]||'tmp');
  const edge='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
  const browser=await chromium.launch({headless:true,...(fs.existsSync(edge)?{executablePath:edge}:{})});
  try{
    const context=await browser.newContext({viewport:{width:1870,height:1150},offline:true});
    const page=await context.newPage();
    const errors=[];page.on('pageerror',error=>errors.push(error.message));
    await page.goto(pathToFileURL(input).href,{waitUntil:'load'});
    const root=page.locator('#rolling-performance');
    await root.scrollIntoViewIfNeeded();
    assert.equal(await root.locator('[data-rolling-zone]').count(),4);
    assert.equal(await root.locator('[data-rolling-period]').count(),5);
    assert.deepEqual(await root.locator('[data-rolling-frequency]').evaluateAll(bs=>bs.map(b=>b.dataset.rollingFrequency)),['60min','day']);
    const data=await page.locator('#rolling-performance-data').evaluate(el=>JSON.parse(el.textContent));
    assert.equal(data.schema_version,3);
    assert.deepEqual(await root.locator('[data-rolling-strategy]').evaluateAll(bs=>bs.map(b=>b.dataset.rollingStrategy)),['quantile_based','unlimited_bid']);
    assert.equal(await root.locator('[data-rolling-scope]').count(),0);
    assert.equal(await root.getAttribute('data-active-strategy'),'unlimited_bid');
    assert.ok(!/CACHE ONLY|Daily P&L \(VPS\)|4 MWh/.test(await root.innerText()));
    assert.equal(data.zones.length,4);
    for(const zone of data.zones){
      const card=root.locator('[data-rolling-zone="'+zone.zone+'"]');
      assert.equal(await card.locator('tbody tr').count(),2);
      const win=zone.windows['90'];
      if(zone.anchor_day)assert.equal(await card.locator('.rolling-dates').innerText(),win.start_day+' — '+win.end_day);
      for(const provider of win.frequencies['60min'].providers){
        const cell=card.locator('[data-provider="'+provider.key+'"] td').first();
        assert.equal(await cell.innerText(),provider.mae===null?'—':provider.mae.toFixed(2));
      }
    }
    const beforePriceMetrics=await root.locator('tbody tr').evaluateAll(rows=>rows.map(row=>Array.from(row.querySelectorAll('td')).slice(0,5).map(cell=>cell.textContent)));
    const beforeCoverage=await root.locator('.rolling-coverage').allTextContents();
    await root.locator('[data-rolling-strategy="quantile_based"]').click();
    assert.equal(await root.locator('[data-rolling-strategy="quantile_based"]').getAttribute('aria-pressed'),'true');
    assert.deepEqual(await root.locator('tbody tr').evaluateAll(rows=>rows.map(row=>Array.from(row.querySelectorAll('td')).slice(0,5).map(cell=>cell.textContent))),beforePriceMetrics);
    assert.deepEqual(await root.locator('.rolling-coverage').allTextContents(),beforeCoverage);
    assert.match(await root.locator('.rolling-strategy-note').innerText(),/Storm calibrated from past errors/);
    for(const zone of data.strategy_zones.quantile_based){
      const card=root.locator('[data-rolling-zone="'+zone.zone+'"]');
      const providers=zone.windows['90'].frequencies['60min'].providers;
      for(const row of providers){
        assert.equal(await card.locator('[data-provider="'+row.key+'"] [data-column="daily_pnl"]').innerText(),row.display_values.daily_pnl);
        assert.equal(await card.locator('[data-provider="'+row.key+'"] th').innerText(),row.label);
      }
      if(providers.some(row=>row.pnl_comparison_eligible===false)){
        assert.equal(await card.locator('td[data-column="daily_pnl"].rolling-best').count(),0);
      }
    }
    await root.locator('[data-rolling-strategy="unlimited_bid"]').click();
    await root.screenshot({path:path.join(out,'cwe_rolling_desktop.png')});
    await root.locator('[data-rolling-period="7"]').click();
    const hourlyPnl=await root.locator('tbody tr td:last-child').allTextContents();
    await root.locator('[data-rolling-frequency="day"]').click();
    assert.deepEqual(await root.locator('tbody tr td:last-child').allTextContents(),hourlyPnl);
    assert.equal(await root.locator('.rolling-frequency-note').count(),0);
    for(const zone of data.zones){
      const card=root.locator('[data-rolling-zone="'+zone.zone+'"]');
      const view=zone.windows['7'].frequencies.day;
      if(zone.anchor_day)assert.match(await card.locator('.rolling-coverage').innerText(),new RegExp(view.samples+' complete daily pairs'));
      for(const provider of view.providers){
        assert.equal(await card.locator('[data-provider="'+provider.key+'"] td').first().innerText(),provider.mae===null?'—':provider.mae.toFixed(2));
      }
    }
    const be=root.locator('[data-rolling-zone="BE"]');
    await be.locator('[data-rolling-sort="bias"]').click();
    assert.equal(await be.locator('th[aria-sort="ascending"]').count(),1);
    const biasValues=(await be.locator('tbody tr td:nth-child(3)').allTextContents()).filter(v=>v!=='—').map(v=>Math.abs(Number(v)));
    assert.deepEqual(biasValues,biasValues.slice().sort((a,b)=>a-b));
    await be.locator('[data-rolling-sort="bias"]').click();
    assert.equal(await be.locator('th[aria-sort="descending"]').count(),1);
    await page.locator('#tab-table').click();
    await page.locator('#table-zone').selectOption('DE');
    assert.equal(await root.locator('tbody tr:visible').count(),8);
    assert.equal(await page.locator('#view-table tbody tr:visible').count(),data.report_delivery_day==='2026-09-19'?24:await page.locator('#view-table tr[data-zone="DE"]').count());
    await page.locator('#tab-graph').click();
    await root.locator('[data-rolling-period="90"]').click();
    await root.locator('[data-rolling-frequency="60min"]').click();
    await root.locator('[data-rolling-period="365"]').click();
    assert.equal(await root.locator('[data-rolling-period="365"]').getAttribute('aria-pressed'),'true');
    for(const zone of data.zones){
      const card=root.locator('[data-rolling-zone="'+zone.zone+'"]');
      const win=zone.windows['365'];
      if(zone.anchor_day)assert.equal(await card.locator('.rolling-dates').innerText(),win.start_day+' — '+win.end_day);
      for(const provider of win.frequencies['60min'].providers){
        assert.equal(await card.locator('[data-provider="'+provider.key+'"] td').first().innerText(),provider.mae===null?'—':provider.mae.toFixed(2));
      }
    }
    await root.screenshot({path:path.join(out,'cwe_rolling_annual.png')});
    // A separate context ensures CSS and Plotly start at the mobile width.
    const small=await browser.newContext({viewport:{width:390,height:844},offline:true});
    const phone=await small.newPage();phone.on('pageerror',error=>errors.push(error.message));
    await phone.goto(pathToFileURL(input).href,{waitUntil:'load'});
    await phone.locator('#rolling-performance').scrollIntoViewIfNeeded();
    const widths=await phone.evaluate(()=>({page:document.documentElement.scrollWidth,viewport:innerWidth}));
    assert.ok(widths.page<=widths.viewport+1,JSON.stringify(widths));
    await phone.locator('#rolling-performance [data-rolling-frequency="day"]').click();
    await phone.locator('#rolling-performance [data-rolling-period="365"]').click();
    await phone.locator('#rolling-performance [data-rolling-strategy="quantile_based"]').click();
    assert.equal(await phone.locator('#rolling-performance [data-rolling-period="365"]').getAttribute('aria-pressed'),'true');
    assert.equal(await phone.locator('#rolling-performance [data-rolling-strategy="quantile_based"]').getAttribute('aria-pressed'),'true');
    await phone.locator('#rolling-performance').screenshot({path:path.join(out,'cwe_rolling_mobile.png')});
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({status:'passed',report:input,zones:4,windows:5,frequencies:2,strategies:2,jsErrors:errors,mobile:widths}));
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
