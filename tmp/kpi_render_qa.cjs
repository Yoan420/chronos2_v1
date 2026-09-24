// Isolated headless rendering of the generated local artifact. No user browser profile.
const fs = require('fs');
const path = require('path');
const {pathToFileURL} = require('url');
const {chromium} = require('C:/Users/BQ6757/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async () => {
  const report = path.resolve(process.argv[2]);
  const out = path.resolve(process.argv[3]);
  fs.mkdirSync(out, {recursive:true});
  const browser = await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless:true, chromiumSandbox:true});
  try {
    const context = await browser.newContext({viewport:{width:1700,height:1100}, serviceWorkers:'block'});
    await context.route('**/*', route => /^(file:|data:)/.test(route.request().url()) ? route.continue() : route.abort());
    const page = await context.newPage(), errors=[];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(pathToFileURL(report).href);
    await page.locator('#kpi-table tbody tr').first().waitFor();
    await page.screenshot({path:path.join(out,'kpi_light.png')});
    await page.locator('#economic-section').screenshot({path:path.join(out,'economic_light.png')});
    await page.getByRole('button',{name:'Mode nuit',exact:true}).click();
    await page.locator('#economic-section').screenshot({path:path.join(out,'economic_dark.png')});
    await page.evaluate(()=>window.scrollTo(0,0));
    await page.screenshot({path:path.join(out,'kpi_dark.png')});
    await page.locator('#country').selectOption('FR');
    await page.locator('#period').selectOption('7');
    if(!(await page.locator('#window').innerText()).includes('2026-09-08')) throw Error('Period filter failed');
    if(await page.locator('#kpi-table tbody tr[data-model]').count()!==13) throw Error('Model count failed');
    if(await page.locator('#economic-table tbody tr[data-model]').count()!==13) throw Error('Economic model count failed');
    if(errors.length) throw Error(errors.join('\n'));
    console.log(JSON.stringify({status:'passed',headless_browser:'Edge',screenshots:out,console_errors:errors,filters_checked:['FR','7 days'],network_blocked:true}));
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
