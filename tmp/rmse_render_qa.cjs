// Local artifact QA in a fresh headless context. No user browser profile or network.
const fs = require('fs');
const path = require('path');
const {pathToFileURL} = require('url');
const {chromium} = require('C:/Users/BQ6757/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async () => {
  const report = path.resolve(process.argv[2]), out = path.resolve(process.argv[3]);
  fs.mkdirSync(out, {recursive:true});
  const browser = await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless:true, chromiumSandbox:true});
  try {
    const context = await browser.newContext({viewport:{width:1700,height:1100}, serviceWorkers:'block'});
    await context.route('**/*', r => /^(file:|data:)/.test(r.request().url()) ? r.continue() : r.abort());
    const page = await context.newPage(), errors=[];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(pathToFileURL(report).href);
    await page.locator('#kpi tbody tr').first().waitFor();
    const payload = await page.locator('#rmse-data').textContent();
    const data = JSON.parse(payload);
    const expected = data.catalog.length;
    await page.screenshot({path:path.join(out,'rmse_light.png')});
    await page.locator('#economic').screenshot({path:path.join(out,'economic_light.png')});
    await page.locator('#theme').click();
    await page.evaluate(()=>window.scrollTo(0,0));
    await page.screenshot({path:path.join(out,'rmse_dark.png')});
    await page.locator('.twocol').screenshot({path:path.join(out,'charts_dark.png')});
    const filters=[];
    for (const zone of ['ALL',...data.zones]) {
      await page.locator('#zone').selectOption(zone);
      for (const period of ['365','90','30','7']) {
        await page.locator('#period').selectOption(period);
        if(await page.locator('#kpi tbody tr').count()!==expected) throw Error('KPI model count');
        if(await page.locator('#economic tbody tr').count()!==expected) throw Error('Economic model count');
        if(!(await page.locator('#subtitle').innerText()).includes(data.periods[period].period.start_day)) throw Error('Period dates');
        if(await page.locator('#hour-plot svg path.curve').count()!==3) throw Error('Hourly plot');
        filters.push(zone+'/'+period);
      }
    }
    for (const model of data.catalog.filter(x=>x.kind==='candidate')) {
      await page.locator('#model').selectOption(model.id);
      const text = await page.locator('#rmse').innerText();
      if(text.includes('—')) throw Error('Missing candidate metric '+model.id);
    }
    await page.locator('#zone').selectOption('ALL');
    await page.locator('#period').selectOption('365');
    await page.locator('#model').selectOption('nyx_rmse');
    await page.locator('#hour-metric').selectOption('rmse_eur_mwh');
    await page.locator('#hour-plot').hover();
    if(!(await page.locator('#hour-tooltip').innerText()).includes('€/MWh')) throw Error('Chart interaction');
    const econ = await page.locator('#economic-support').innerText();
    if(econ.includes('—')) throw Error('Economic policy fields unavailable: '+econ);
    if(errors.length) throw Error(errors.join('\n'));
    console.log(JSON.stringify({status:'passed',screenshots:out,model_count:expected,filters,console_errors:errors,economic_policy:econ,network_blocked:true}));
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
