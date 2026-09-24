// Hermetic, fresh browser context; never uses the user's browser profile.
const fs=require('fs'),path=require('path'),{pathToFileURL}=require('url');
const {chromium}=require('C:/Users/BQ6757/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
  const report=path.resolve(process.argv[2]),out=path.resolve(process.argv[3]);
  fs.mkdirSync(out,{recursive:true});
  const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true,chromiumSandbox:true});
  try {
    const context=await browser.newContext({viewport:{width:1700,height:1100},serviceWorkers:'block'});
    await context.route('**/*',r=>/^(file:|data:)/.test(r.request().url())?r.continue():r.abort());
    const page=await context.newPage(),errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto(pathToFileURL(report).href);
    await page.locator('#kpi tbody tr').first().waitFor();
    const data=JSON.parse(await page.locator('#physical-data').textContent());
    await page.screenshot({path:path.join(out,'physical_light.png')});
    await page.locator('#theme').click();
    await page.screenshot({path:path.join(out,'physical_dark.png')});
    await page.locator('#case-charts').screenshot({path:path.join(out,'physical_cases_dark.png')});
    const filters=[];
    for(const zone of ['ALL',...data.zones]) {
      await page.locator('#zone').selectOption(zone);
      for(const period of ['365','7']) {
        await page.locator('#period').selectOption(period);
        if(await page.locator('#kpi tbody tr').count()!==data.catalog.length)throw Error('KPI row count');
        if(await page.locator('#eva tbody tr').count()!==data.catalog.length)throw Error('EVA row count');
        if(!(await page.locator('#subtitle').innerText()).includes(data.periods[period].period.start_day))throw Error('Dates');
        if(await page.locator('#case-charts svg').count()!==(zone==='ALL'?data.zones.length:1))throw Error('Country plots');
        filters.push(zone+'/'+period);
      }
    }
    for(const model of data.catalog.filter(x=>x.kind==='candidate'))await page.locator('#model').selectOption(model.id);
    await page.locator('#zone').selectOption('ALL');await page.locator('#period').selectOption('365');
    await page.locator('#model').selectOption('nyx_physical_p50');
    if((await page.locator('#case tbody').innerText()).includes('—'))throw Error('Missing case values');
    if(!(await page.locator('#forensic').innerText()).includes('398'))throw Error('Forensic spread absent');
    if(await page.locator('#initial tbody tr').count()<5)throw Error('Initial signal absent');
    await page.locator('#hour-plot svg').hover();
    if(!(await page.locator('#hour-tip').innerText()).includes('h'))throw Error('Hourly interaction');
    if(errors.length)throw Error(errors.join('\n'));
    const summary={status:'passed',report,screenshots:out,filters,console_errors:errors,network_blocked:true,
      case_text:await page.locator('#case tbody').innerText(),policy:await page.locator('#policy').innerText()};
    fs.writeFileSync(path.join(out,'qa.json'),JSON.stringify(summary,null,2));console.log(JSON.stringify(summary));
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
