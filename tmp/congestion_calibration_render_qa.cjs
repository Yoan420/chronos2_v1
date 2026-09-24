// Local-only visual QA for the NEW calibration report. Never use a user profile.
// Run only after the real report exists:
// node tmp/congestion_calibration_render_qa.cjs <report.html> <new-empty-QA-directory>
'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const {pathToFileURL} = require('url');
const {chromium} = require('C:/Users/BQ6757/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

const root = path.resolve(__dirname, '..');
const namespace = path.join(root, 'runs', 'experiments', 'nyx_congestion_calibration_v1');
const samePath = (a, b) => process.platform === 'win32' ? a.toLowerCase() === b.toLowerCase() : a === b;
const within = (base, child) => {
  const relative = path.relative(base, child);
  return relative === '' || (!relative.startsWith('..'+path.sep) && relative !== '..' && !path.isAbsolute(relative));
};
function confined(value) {
  const result = path.resolve(value);
  if (!within(namespace, result)) throw Error('QA reads and outputs must remain in the NEW calibration namespace.');
  for (let cursor = result; ; cursor = path.dirname(cursor)) {
    if (fs.existsSync(cursor) && !samePath(fs.realpathSync(cursor), cursor)) throw Error('Path aliases/junctions are forbidden: '+cursor);
    if (path.dirname(cursor) === cursor) break;
  }
  return result;
}
const sha = filename => crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const check = (condition, message) => { if (!condition) throw Error(message); };
const normalise = value => String(value).replace(/[\s\u00a0\u202f]+/g, ' ').trim();
const numberText = (value, digits=2) => Number.isFinite(value) ? value.toLocaleString('fr-FR', {minimumFractionDigits:digits, maximumFractionDigits:digits}) : '—';
const percentageText = value => Number.isFinite(value) ? numberText(value*100)+' %' : '—';
function luminance(hex) {
  check(/^#[0-9a-f]{3}([0-9a-f]{3})?$/i.test(hex), 'Unexpected CSS colour: '+hex);
  let digits = hex.slice(1);
  if (digits.length === 3) digits = [...digits].map(v => v+v).join('');
  const rgb = [0,2,4].map(i => parseInt(digits.slice(i,i+2),16)/255)
    .map(v => v <= .04045 ? v/12.92 : ((v+.055)/1.055)**2.4);
  return rgb[0]*.2126 + rgb[1]*.7152 + rgb[2]*.0722;
}
const contrast = (a,b) => (Math.max(luminance(a),luminance(b))+.05)/(Math.min(luminance(a),luminance(b))+.05);

(async () => {
  check(process.argv.length === 4, 'Expected report.html and a new empty QA output directory.');
  const report = confined(process.argv[2]), out = confined(process.argv[3]);
  check(fs.statSync(report).isFile() && path.extname(report).toLowerCase() === '.html', 'A real local HTML report is required.');
  check(!fs.existsSync(out) || (fs.statSync(out).isDirectory() && fs.readdirSync(out).length === 0), 'QA destination must be new or empty; existing artifacts are never overwritten.');
  const reportSha = sha(report);
  fs.mkdirSync(out, {recursive:true});
  const errors = [], blockedRequests = [], screenshots = [], filters = [], models = [], themes = [];
  let browser, page;
  try {
    browser = await chromium.launch({
      executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
      headless:true, chromiumSandbox:true,
      args:['--disable-background-networking','--disable-component-update','--no-first-run']
    });
    const context = await browser.newContext({viewport:{width:1700,height:1100}, serviceWorkers:'block'});
    await context.route('**/*', route => {
      if (/^(file:|data:)/.test(route.request().url())) return route.continue();
      blockedRequests.push(route.request().url());
      return route.abort('blockedbyclient');
    });
    if (typeof context.routeWebSocket === 'function') {
      await context.routeWebSocket('**/*', socket => { blockedRequests.push('websocket:'+socket.url()); socket.close(); });
    }
    page = await context.newPage();
    page.on('pageerror', error => errors.push('pageerror: '+error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push('console: '+message.text()); });
    await page.goto(pathToFileURL(report).href, {waitUntil:'load'});
    await page.locator('#kpi tbody tr').first().waitFor();
    const data = JSON.parse(await page.locator('#calibration-data').textContent());
    check(data.schema_version === 1, 'Unknown report data schema.');
    check(data.catalog.length === 12, 'Expected twelve price comparators.');
    check(['FR','DE','BE','NL'].every(z => data.zones.includes(z)) && data.zones.length === 4, 'Expected exactly four CWE countries.');
    check(data.method.stage1_retrained === false && data.method.calibration_calendar_days === 90, 'Wrong experiment or stage-one contract.');
    check(data.periods['365'] && data.periods['7'], 'Both fixed KPI windows are required.');
    check(data.case_days.includes('2026-09-14'), 'The September 14 event must be present.');
    const catalogue = Object.fromEntries(data.catalog.map(row => [row.id,row.label]));
    const modelChoices = await page.locator('#model option').evaluateAll(nodes => nodes.map(n => n.value));
    const expectedChoices = data.catalog.filter(m => !['__storm__','nuclear_kalman'].includes(m.id)).map(m => m.id);
    check(JSON.stringify([...modelChoices].sort()) === JSON.stringify([...expectedChoices].sort()), 'Model options do not match the report catalogue.');

    async function screenshot(name, locator) {
      const target = path.join(out, name);
      if (locator) await locator.screenshot({path:target});
      else await page.screenshot({path:target});
      screenshots.push(target);
    }
    async function select(id, value) {
      await page.locator('#'+id).selectOption(value);
      check(await page.locator('#'+id).inputValue() === value, 'Filter selection failed: '+id+'/'+value);
    }
    async function verifyRows(zone, period, day) {
      const rowCount = data.catalog.length;
      for (const id of ['kpi','eva','june']) check(await page.locator('#'+id+' tbody tr').count() === rowCount, id+' comparator count: '+zone+'/'+period+'/'+day);
      check(await page.locator('#ready tbody tr').count() === 2, 'Two readiness chains required.');
      check(await page.locator('#changes tbody tr').count() === rowCount-1, 'Intervention comparators differ.');
      const countries = zone === 'ALL' ? data.zones : [zone];
      const expected = data.case.filter(r => r.day === day && r.hour === 19 && countries.includes(r.zone));
      check(expected.length === countries.length, 'Every selected country must have its 19 h case row.');
      const actual = await page.locator('#case tbody tr').evaluateAll(nodes => nodes.map(tr => [...tr.cells].map(c => c.textContent)));
      check(actual.length === expected.length, '19 h table country count differs.');
      const fields = ['actual','storm','nuclear_kalman','nyx_congestion','calibrated_control_direct',
        'calibrated_control_governed','congestion_calibrated_direct','nyx_congestion_calibrated'];
      for (const cells of actual) {
        const source = expected.find(r => r.zone === cells[0]);
        check(Boolean(source), 'Unexpected country in the 19 h table: '+cells[0]);
        fields.forEach((field,i) => check(normalise(cells[i+1]) === normalise(numberText(source[field])), '19 h price mismatch: '+day+'/'+cells[0]+'/'+field));
      }
      check(await page.locator('#case-charts svg').count() === countries.length, 'Hourly country chart count differs.');
      check(await page.locator('#hour-plot svg').count() === 1, 'Hourly error chart missing.');
      check((await page.locator('#case-title').textContent()).includes(day), 'Episode title did not update.');
      const probabilityRows = await page.locator('#probability tbody tr').evaluateAll(nodes => nodes.map(tr => [...tr.cells].map(c => c.textContent)));
      check(probabilityRows.length === countries.length*2, 'Two probability chains per country required.');
      for (const cells of probabilityRows) {
        const country = cells[0].split(' · ')[0];
        const strategy = cells[0].includes('contrôle') ? 'control' : 'congestion';
        const source = expected.find(r => r.zone === country);
        check(normalise(cells[1]) === normalise(percentageText(source[strategy+'_raw_probability'])), 'Raw probability mismatch.');
        check(normalise(cells[2]) === normalise(percentageText(source[strategy+'_spike_probability'])), 'Calibrated probability mismatch.');
      }
      const periodData = data.periods[period];
      const probabilityData = periodData.probability;
      check(probabilityData && probabilityData.available === true, 'OOS probability diagnostics missing.');
      const expectedScores = probabilityData.rows.filter(r => r.zone === zone);
      const scoreRows = await page.locator('#probability-kpi tbody tr').evaluateAll(nodes => nodes.map(tr => [...tr.cells].map(c => c.textContent)));
      check(scoreRows.length === 2 && expectedScores.length === 2, 'Two OOS probability score chains required.');
      for (const cells of scoreRows) {
        const strategy = cells[0].includes('contrôle') ? 'control' : 'congestion';
        const source = expectedScores.find(r => r.strategy === strategy);
        check(Boolean(source), 'Unexpected probability score chain: '+cells[0]);
        const expectedCells = [numberText(source.n_country_hours,0), numberText(source.n_events,0),
          percentageText(source.event_rate), numberText(source.raw_brier,5), numberText(source.calibrated_brier,5),
          numberText(source.raw_log_loss,5), numberText(source.calibrated_log_loss,5), numberText(source.sparse_calibration_hours,0)];
        expectedCells.forEach((value,i) => check(normalise(cells[i+1]) === normalise(value),
          'Probability KPI/payload mismatch: '+zone+'/'+period+'/'+strategy+'/column'+(i+1)));
      }
      const expectedBins = probabilityData.reliability.filter(r => r.zone === zone);
      const binRows = await page.locator('#reliability tbody tr').evaluateAll(nodes => nodes.map(tr => [...tr.cells].map(c => c.textContent)));
      check(binRows.length === 40 && expectedBins.length === 40, 'Expected 2 chains × 2 probability types × 10 bins.');
      expectedBins.forEach((source,i) => {
        const label = (source.strategy === 'control' ? 'Contrôle' : 'Congestion')+' · '+(source.variant === 'raw' ? 'brut' : 'calibré');
        const expectedCells = [label, '['+numberText(source.bin_lower,1)+' ; '+numberText(source.bin_upper,1)+(source.bin_upper === 1 ? ']' : '['),
          numberText(source.n_country_hours,0), numberText(source.n_events,0),
          percentageText(source.mean_probability), percentageText(source.observed_frequency)];
        expectedCells.forEach((value,j) => check(normalise(binRows[i][j]) === normalise(value),
          'Reliability/payload mismatch: '+zone+'/'+period+'/row'+i+'/column'+j));
      });
      const subtitle = await page.locator('#subtitle').textContent();
      check(subtitle.includes(periodData.period.start_day) && subtitle.includes(periodData.period.end_day), 'Displayed KPI window differs from the selected period.');
      const tableLabels = await page.locator('#kpi tbody tr td:first-child').allTextContents();
      check(JSON.stringify(tableLabels.map(normalise).sort()) === JSON.stringify(data.catalog.map(r=>normalise(r.label)).sort()), 'KPI model labels are incomplete.');
      const text = await page.locator('main').innerText();
      check(!/\b(?:undefined|NaN|Infinity)\b/.test(text), 'Raw invalid values are exposed in the report.');
    }
    async function verifyTheme(theme) {
      const colours = await page.evaluate(() => Object.fromEntries(['paper','ink','muted','obs','storm','nyx','new'].map(k =>
        [k,getComputedStyle(document.documentElement).getPropertyValue('--'+k).trim()])));
      check(contrast(colours.ink,colours.paper) >= 4.5, theme+' main text contrast below 4.5:1.');
      check(contrast(colours.muted,colours.paper) >= 4.5, theme+' secondary text contrast below 4.5:1.');
      for (const name of ['obs','storm','nyx','new']) check(contrast(colours[name],colours.paper) >= 3, theme+' chart contrast below 3:1: '+name);
      const strokes = await page.locator('#case-charts svg').first().locator('path').evaluateAll(nodes => nodes.map(n=>n.getAttribute('stroke')));
      check(JSON.stringify(strokes) === JSON.stringify(['obs','storm','nyx','new'].map(k=>colours[k])), theme+' chart colours did not follow the CSS theme.');
      themes.push({theme,colours});
    }

    await select('zone','ALL'); await select('period','365'); await select('day','2026-09-14');
    await select('model','nyx_congestion_calibrated');
    await verifyRows('ALL','365','2026-09-14');
    await verifyTheme('light');
    await screenshot('calibration_light.png');
    await page.locator('#theme').click();
    check(await page.locator('html').getAttribute('data-theme') === 'dark', 'Night mode did not activate.');
    await verifyTheme('dark');
    await screenshot('calibration_dark.png');
    await screenshot('calibration_september14_cases_dark.png', page.locator('#case-title').locator('..'));
    const reliabilityDetails = page.locator('#reliability').locator('xpath=ancestor::details');
    await reliabilityDetails.locator('summary').click();
    check(await page.locator('#reliability').isVisible(), 'Reliability details did not expand.');
    await screenshot('calibration_probability_scores_dark.png', page.locator('#probability-kpi').locator('xpath=ancestor::section'));
    await reliabilityDetails.locator('summary').click();
    check(themes[0].colours.paper !== themes[1].colours.paper && themes[0].colours.new !== themes[1].colours.new, 'Theme did not update both surface and chart colours.');

    for (const zone of ['ALL',...data.zones]) {
      await select('zone',zone);
      for (const period of ['365','7']) {
        await select('period',period);
        for (const day of data.case_days) {
          await select('day',day);
          await verifyRows(zone,period,day);
          filters.push({zone,period,day});
        }
      }
    }
    await select('zone','DE'); await select('period','365'); await select('day','2026-09-14');
    const event = data.case.find(r=>r.zone==='DE' && r.day==='2026-09-14' && r.hour===19);
    for (const model of modelChoices) {
      await select('model',model);
      check((await page.locator('#price-legend').innerText()).includes(catalogue[model]), 'Selected model missing from price legend.');
      const graph = page.locator('#case-DE svg');
      await graph.scrollIntoViewIfNeeded();
      const box = await graph.boundingBox();
      check(Boolean(box) && box.width > 0 && box.height > 0, 'Country chart is not visible.');
      const x19 = 60+19*595/23;
      await page.mouse.move(box.x+box.width*x19/680,box.y+box.height*.45);
      const tooltip = normalise(await page.locator('#tip-DE').textContent());
      check(tooltip.startsWith('19 h'), 'Hourly tooltip does not select 19 h.');
      check(tooltip.includes(normalise(catalogue[model]+': '+numberText(event[model]))), 'Tooltip price differs from selected model at 19 h.');
      models.push(model);
    }
    await screenshot('calibration_germany_19h_dark.png', page.locator('#case-title').locator('..'));
    await select('zone','ALL'); await select('period','365'); await select('day','2026-09-14');
    await select('model','nyx_congestion_calibrated');
    await page.locator('#theme').click();
    check(await page.locator('html').getAttribute('data-theme') === 'light', 'Day mode did not restore.');
    await verifyRows('ALL','365','2026-09-14');
    await page.locator('header').scrollIntoViewIfNeeded();
    await screenshot('calibration_final_light.png');
    check(errors.length === 0, errors.join('\n'));
    check(sha(report) === reportSha, 'The report changed during read-only QA.');
    const result = {status:'passed',report,report_sha256:reportSha,filters,models,themes,
      errors,blocked_requests:blockedRequests,network_blocked:true,user_profile_used:false,screenshots,
      case_text:await page.locator('#case tbody').innerText(),
      probability_text:await page.locator('#probability tbody').innerText(),
      probability_kpi_text:await page.locator('#probability-kpi tbody').innerText(),
      readiness_text:await page.locator('#ready tbody').innerText()};
    fs.writeFileSync(path.join(out,'qa.json'),JSON.stringify(result,null,2));
    console.log(JSON.stringify(result));
  } catch (error) {
    if (page) {
      try { await page.screenshot({path:path.join(out,'failure.png')}); } catch (_) { /* Browser may have failed to launch. */ }
    }
    const result = {status:'failed',report,report_sha256:reportSha,error:String(error.stack||error),errors,
      blocked_requests:blockedRequests,network_blocked:true,user_profile_used:false,filters,models,screenshots};
    fs.writeFileSync(path.join(out,'qa.json'),JSON.stringify(result,null,2));
    throw error;
  } finally {
    if (browser) await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode=1; });
