// Fresh isolated browser, local artifact only; no user profile or network.
const fs=require('fs'),path=require('path'),{pathToFileURL}=require('url');
const {chromium}=require('C:/Users/BQ6757/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
 const report=path.resolve(process.argv[2]),out=path.resolve(process.argv[3]);fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true,chromiumSandbox:true});
 try {
  const context=await browser.newContext({viewport:{width:1700,height:1100},serviceWorkers:'block'});
  await context.route('**/*',r=>/^(file:|data:)/.test(r.request().url())?r.continue():r.abort());
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(report).href);await page.locator('#kpi tbody tr').first().waitFor();
  const data=JSON.parse(await page.locator('#congestion-data').textContent());
  await page.screenshot({path:path.join(out,'congestion_light.png')});
  await page.locator('#theme').click();await page.screenshot({path:path.join(out,'congestion_dark.png')});
  await page.locator('#case-charts').screenshot({path:path.join(out,'congestion_cases.png')});
  const filters=[];
  for(const zone of ['ALL',...data.zones]) {
   await page.locator('#zone').selectOption(zone);
   for(const period of ['365','7']) {
    await page.locator('#period').selectOption(period);
    if(await page.locator('#kpi tbody tr').count()!==data.catalog.length)throw Error('KPI rows');
    if(await page.locator('#eva tbody tr').count()!==data.catalog.length)throw Error('EVA rows');
    if(await page.locator('#case-charts svg').count()!==(zone==='ALL'?data.zones.length:1))throw Error('Country charts');
    if(await page.locator('#detection tbody tr').count()!==2)throw Error('Paired classifier scores');
    if(await page.locator('#regional-table tbody tr').count()!==2)throw Error('Regional paired classifier scores');
    if(await page.locator('#regional-case tbody tr').count()!==(zone==='ALL'?data.zones.length:1))throw Error('Regional cases');
    if(await page.locator('#stage2-fits tbody tr').count()!==2)throw Error('Stage 2 fold reasons');
    if(await page.locator('#stage2-ready tbody tr').count()!==2)throw Error('Stage 2 readiness');
    filters.push(zone+'/'+period);
   }
  }
  for(const model of data.catalog.filter(x=>x.kind==='candidate'||x.kind==='control'))await page.locator('#model').selectOption(model.id);
  const choices=await page.locator('#constraint option').evaluateAll(nodes=>nodes.map(n=>n.value));
  for(const choice of choices.slice(0,4))await page.locator('#constraint').selectOption(choice);
  await page.locator('#zone').selectOption('ALL');await page.locator('#period').selectOption('365');
  await page.locator('#model').selectOption('nyx_congestion');
  if(errors.length)throw Error(errors.join('\n'));
  const result={status:'passed',report,filters,errors,network_blocked:true,constraint_choices:choices.length,
    case_text:await page.locator('#case tbody').innerText(),classification:await page.locator('#detection tbody').innerText(),
    regional:await page.locator('#regional-table tbody').innerText(),regional_case:await page.locator('#regional-case tbody').innerText()};
  fs.writeFileSync(path.join(out,'qa.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
 } finally {await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
