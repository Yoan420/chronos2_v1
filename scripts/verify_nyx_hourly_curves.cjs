const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require('C:/Users/BQ6757/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const report = path.resolve('runs/exports/nyx_test2_heures_DE_NL_14_et_22_septembre_2026.html');
  const browser = await chromium.launch({headless: true, channel:'msedge', args:['--disable-gpu']});
  try {
    const page = await browser.newPage({viewport:{width:1360,height:1100}, offline:true});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(pathToFileURL(report).href);
    const frame = page.frameLocator('#codex-visualization');
    await frame.locator('.nyx-price-chart').waitFor();
    const count = selector => frame.locator(selector).count();
    assert.equal(await count('.nyx-price-chart'), 1);
    assert.equal(await count('[data-series-path]'), 5);
    assert.equal(await count('#nyx-hourly-rows tr'), 24);
    assert.equal(await count('[data-test2-marker]'), 2);
    await frame.locator('[data-series="storm"]').click();
    assert.equal(await count('[data-series-path="storm"]'),0);
    await frame.locator('[data-series="storm"]').click();
    assert.equal(await count('[data-series-path="storm"]'),1);
    await frame.locator('#nyx-day').selectOption('all');
    await frame.locator('#nyx-country').selectOption('all');
    assert.equal(await count('.nyx-price-chart'),4);
    assert.equal(await count('[data-series-path]'),18);
    assert.equal(await count('[data-test2-marker]'),7);
    assert.equal(await count('#nyx-hourly-rows tr'),96);
    for (const key of ['nyx','hybrid','storm','actual']) assert.equal(await count('[data-series-path="'+key+'"]'),4);
    assert.equal(await count('[data-series-path="fuel"]'),2);
    await frame.locator('#nyx-only-test2').check();
    assert.equal(await count('#nyx-hourly-rows tr'),7);
    assert.equal(await count('[data-series-path]'),18);
    await frame.locator('button[data-key="actual"]').click();
    assert.equal(await count('[data-series-path]'),18);
    await frame.locator('#nyx-only-test2').uncheck();
    await frame.locator('#nyx-day').selectOption('2026-09-22');
    await frame.locator('#nyx-country').selectOption('DE');
    assert.equal(await count('[data-series-path]'),4);
    assert.match(await frame.locator('#nyx-curve-coverage').textContent(), /données indisponibles/);
    const overlay=frame.locator('[data-chart-hit]');
    await overlay.focus();
    for(let h=12;h<19;h++) await overlay.press('ArrowRight');
    const tooltip=await frame.locator('#nyx-curve-tooltip').textContent();
    assert.match(tooltip,/19–20 h/);
    assert.match(tooltip,/571,8/);
    assert.match(tooltip,/596,4/);
    assert.match(tooltip,/Test2 actif/);
    assert.match(tooltip,/N\/D/);
    await overlay.press('Escape');
    assert.equal(await frame.locator('#nyx-curve-tooltip').isHidden(),true);
    const screenshot=path.resolve('runs/exports/nyx_test2_courbes_qa_desktop.png');
    await frame.locator('#nyx-scenario-curves').screenshot({path:screenshot});
    await page.setViewportSize({width:390,height:900});
    await page.waitForTimeout(250);
    const geometry=await frame.locator('#nyx-scenario-curves').evaluate(section=>({
      width:section.getBoundingClientRect().width,
      svgWidth:section.querySelector('svg.nyx-price-chart').getBoundingClientRect().width,
      viewBox:section.querySelector('svg.nyx-price-chart').viewBox.baseVal.width,
      horizontalOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1
    }));
    assert.equal(geometry.horizontalOverflow,false);
    assert.ok(Math.abs(geometry.width-geometry.svgWidth)<2);
    assert.ok(Math.abs(geometry.viewBox-geometry.svgWidth)<2);
    assert.deepEqual(errors,[]);
    await frame.locator('#nyx-scenario-curves').screenshot({path:path.resolve('runs/exports/nyx_test2_courbes_qa_mobile.png')});
    console.log(JSON.stringify({result:'PASS',checks:29,offline:true,charts:4,priceSeries:5,availablePaths:18,routedHours:7,dataRows:96,errors,geometry,screenshot},null,2));
  } finally { await browser.close(); }
})().catch(e=>{console.error(e); process.exitCode=1;});
