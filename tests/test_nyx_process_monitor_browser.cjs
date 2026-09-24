"use strict";

/**
 * Read-only browser checks against an already running NYX monitor.
 * Uses a fresh headless Edge profile. Does not launch/stop a server or calculation.
 * Usage: node tests/test_nyx_process_monitor_browser.cjs [http://127.0.0.1:8766]
 * Optional NYX_MONITOR_PLAYWRIGHT_MODULE and NYX_MONITOR_EDGE override local paths.
 */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const modulePath = process.env.NYX_MONITOR_PLAYWRIGHT_MODULE || path.join(
  process.env.USERPROFILE || "C:\\Users\\BQ6757",
  ".cache", "codex-runtimes", "codex-primary-runtime", "dependencies", "node", "node_modules", "playwright"
);
const edgePath = process.env.NYX_MONITOR_EDGE || "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const { chromium } = require(modulePath);
const origin = process.argv[2] || "http://127.0.0.1:8766";
const target = new URL(origin);
assert(["127.0.0.1", "localhost"].includes(target.hostname), "Tests are loopback-only");
assert.equal(target.protocol, "http:");
const qa = path.join(__dirname, "..", "runs", ".process_monitor", "qa", "v2");
fs.mkdirSync(qa, { recursive: true });

const results = [];
const errors = [];
let browser;
let currentPage;
function passed(name, details) {
  results.push({ name, status: "PASS", details });
  console.log(`PASS ${name}${details ? ` — ${details}` : ""}`);
}
function recordErrors(page, label, allowedNetworkErrors = false) {
  page.on("pageerror", error => errors.push({ label, type: "pageerror", message: error.message }));
  page.on("console", message => {
    if (message.type() !== "error") return;
    if (allowedNetworkErrors && /Failed to load resource.*(?:503|ERR_FAILED)/.test(message.text())) return;
    errors.push({ label, type: "console", message: message.text() });
  });
}
async function waitConnected(page) {
  await page.locator('#connection[data-state="online"]').waitFor({ state: "visible", timeout: 15000 });
}
async function assertNoHorizontalOverflow(page, label) {
  const layout = await page.evaluate(() => ({
    viewport: document.documentElement.clientWidth,
    width: document.documentElement.scrollWidth,
    badCards: [...document.querySelectorAll(".job-card:not([hidden])")].filter(node => {
      const box = node.getBoundingClientRect();
      return box.left < -1 || box.right > window.innerWidth + 1;
    }).length,
  }));
  assert(layout.width <= layout.viewport + 1, `${label}: ${JSON.stringify(layout)}`);
  assert.equal(layout.badCards, 0, label);
  passed(label, `viewport ${layout.viewport}px, document ${layout.width}px`);
}
async function screenshot(page, filename) {
  await page.screenshot({ path: path.join(qa, filename), fullPage: true, animations: "disabled" });
}

async function realServerChecks() {
  const context = await browser.newContext({ viewport: { width: 1180, height: 900 }, locale: "fr-FR", timezoneId: "Europe/Paris" });
  const page = await context.newPage();
  currentPage = page;
  recordErrors(page, "real-server");
  const reads = [];
  page.on("response", response => { if (new URL(response.url()).pathname === "/api/state") reads.push({ at: Date.now(), status: response.status() }); });
  const initialResponse = page.waitForResponse(response => new URL(response.url()).pathname === "/api/state");
  const html = await page.goto(origin, { waitUntil: "domcontentloaded", timeout: 15000 });
  assert.equal(html.status(), 200);
  assert.match(html.headers()["content-security-policy"], /script-src 'self'/);
  const state = await (await initialResponse).json();
  assert.equal(state.app, "nyx-process-monitor");
  await waitConnected(page);
  const activeJobs = state.jobs.filter(job => typeof job.active === "boolean" ? job.active : job.processes.length > 0);
  assert.equal(await page.locator(".job-card").count(), activeJobs.length);
  assert.equal(await page.locator("h1").innerText(), "Calculs en cours.");
  assert.equal(await page.locator(".read-only").innerText(), "Lecture seule");
  assert.equal(await page.locator("[data-filter], #count-complete, #count-attention").count(), 0);
  assert.equal(await page.locator("#count-running").innerText(), String(activeJobs.length));
  passed("Real server, API, CSP and active-only UI", `${activeJobs.length} active jobs rendered from ${state.jobs.length} observed launches`);

  const tracked = activeJobs.find(job => job.kind === "registered") || activeJobs[0];
  assert(tracked, "At least one job must exist for live UI checks");
  const card = page.locator(".job-card").filter({ has: page.locator(".job-title", { hasText: tracked.title }) }).first();
  if (tracked.progress.percent === null) {
    assert.equal(await card.locator(".progress-track").getAttribute("aria-valuenow"), null);
    assert.match(await card.locator(".progress-value").innerText(), /Lot en cours/);
    passed("Indeterminate progress is not a time estimate");
  }
  const etaValues = await page.locator(".eta-value").allTextContents();
  assert(etaValues.every(value => /^≈ \d/.test(value)));
  assert.equal(await page.locator(".eta-range").count(), activeJobs.length);
  assert.equal(await page.locator(".eta-confidence").count(), activeJobs.length);
  passed("Every active calculation has numeric ETA, interval and confidence");
  await screenshot(page, "v2-01-desktop-live.png");
  await assertNoHorizontalOverflow(page, "Desktop layout");

  await card.locator("summary").click();
  await card.locator(".process-tree").waitFor({ state: "visible" });
  assert.equal(await card.locator(".process-row").count(), tracked.processes.length);
  if (tracked.processes.length) {
    assert.match(await card.locator(".process-tree").innerText(), /PID .*Créé le/s);
    assert.match(await card.locator(".process-tree").innerText(), /CPU .*RAM/s);
  }
  await card.locator('[data-log="stderr"]').click();
  assert.equal(await card.locator('[data-log="stderr"]').getAttribute("aria-pressed"), "true");
  assert.equal(await card.locator(".stderr-hint").isVisible(), true);
  await card.locator('[data-log="stdout"]').click();
  assert.equal(await card.locator(".log-output").innerText(), tracked.stdout_tail || "Aucune ligne disponible dans ce journal.");
  await card.locator(".follow-log").uncheck();
  await card.locator(".log-output").focus();
  await card.locator(".log-output").evaluate(node => { node.scrollTop = 0; });
  const refreshResponse = page.waitForResponse(response => new URL(response.url()).pathname === "/api/state", { timeout: 15000 });
  await refreshResponse;
  await page.waitForFunction(() => document.querySelector(".job-details[open]") !== null);
  assert.equal(await card.locator(".job-details").evaluate(node => node.open), true);
  assert.equal(await card.locator(".follow-log").isChecked(), false);
  assert.equal(await card.locator(".log-output").evaluate(node => node === document.activeElement), true);
  assert.equal(await card.locator(".log-output").evaluate(node => node.scrollTop), 0);
  assert(reads.length >= 2);
  const cadence = reads[1].at - reads[0].at;
  assert(cadence >= 650 && cadence < 5000, `Refresh interval was ${cadence} ms`);
  passed("Automatic refresh preserves details, log tab, focus and scroll", `${cadence} ms between API responses`);
  await screenshot(page, "v2-02-desktop-details-live.png");
  await page.setViewportSize({ width: 390, height: 844 });
  await assertNoHorizontalOverflow(page, "Responsive 390px layout with details");
  await screenshot(page, "v2-03-mobile-live.png");

  for (const link of await page.locator(".zone-report").evaluateAll(nodes => nodes.map(node => node.getAttribute("href")))) {
    assert.match(link, /^\/artifact\?id=/);
  }
  const report = state.jobs.flatMap(job => job.zones || []).find(zone => zone.report_url);
  if (report) {
    const response = await context.request.get(origin + report.report_url);
    assert.equal(response.status(), 200);
    assert.match(response.headers()["content-security-policy"], /sandbox/);
    passed("Existing report link resolves through protected artifact endpoint");
  }
  assert.equal(errors.length, 0, JSON.stringify(errors, null, 2));
  await context.close();
  return state;
}

async function simulatedStateChecks(liveState) {
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, locale: "fr-FR", timezoneId: "Europe/Paris" });
  const page = await context.newPage();
  currentPage = page;
  recordErrors(page, "browser-only-fixtures", true);
  const state = structuredClone(liveState);
  state.updated_at = new Date().toISOString();
  state.refresh_seconds = 2;
  state.jobs = [structuredClone(liveState.jobs.find(job => job.active) || liveState.jobs[0])];
  state.jobs[0].active = true;
  state.jobs[0].title = '<img src=x onerror="window.__nyx_xss=true">';
  state.jobs[0].stdout_tail = Array.from({ length: 120 }, (_, n) => `Line ${n + 1}: browser-only test text`).join("\n");
  state.jobs[0].warnings = ["Avertissement de test, uniquement dans ce navigateur éphémère."];
  state.jobs[0].zones = [{ zone: "DE", status: "complete", report_url: "javascript:window.__nyx_xss=true" }];
  state.jobs[0].status = "unknown";
  state.jobs[0].progress.percent = null;
  state.jobs[0].eta = { remaining_seconds: 600, low_seconds: 300, high_seconds: 1200, confidence: "very_low", basis: "Browser-only test estimate", method: "test", as_of: new Date().toISOString() };
  const inactive = structuredClone(state.jobs[0]);
  inactive.id = "inactive-browser-fixture";
  inactive.active = false;
  inactive.status = "complete";
  inactive.processes = [];
  state.jobs.push(inactive);
  let fail = false;
  let stale = false;
  await page.route("**/api/state", route => {
    if (fail) return route.fulfill({ status: 503, contentType: "application/json", body: '{"error":"browser-only simulated outage"}' });
    state.updated_at = new Date(Date.now() - (stale ? 60000 : 0)).toISOString();
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(state) });
  });
  await page.goto(origin, { waitUntil: "domcontentloaded" });
  await waitConnected(page);
  const card = page.locator(".job-card");
  assert.equal(await card.locator(".job-title").innerText(), state.jobs[0].title);
  assert.equal(await card.locator(".job-title img").count(), 0);
  assert.equal(await card.locator(".zone-report").count(), 0);
  assert.equal(await page.evaluate(() => Boolean(window.__nyx_xss)), false);
  assert.equal(await card.count(), 1);
  assert.match(await card.locator(".eta-value").innerText(), /^≈ \d/);
  const countdownBefore = await card.locator(".eta-value").innerText();
  await page.waitForFunction(before => document.querySelector(".eta-value").textContent !== before, countdownBefore, { timeout: 3500 });
  passed("Untrusted text, unsafe URL, active-only filtering and local countdown");

  await card.locator("summary").click();
  await card.locator(".log-output").evaluate(node => { node.scrollTop = 0; });
  await page.waitForFunction(() => !document.querySelector(".follow-log").checked);
  state.jobs[0].stdout_tail += "\nLine 121: new content";
  await page.waitForFunction(() => document.querySelector(".log-output").textContent.includes("Line 121"), null, { timeout: 7000 });
  assert.equal(await card.locator(".log-output").evaluate(node => node.scrollTop), 0);
  passed("Manual log scroll disables follow and survives appended content");

  stale = true;
  await page.locator('#connection[data-state="stale"]').waitFor({ timeout: 7000 });
  assert.equal(await page.locator("#connection-error").isVisible(), true);
  passed("Stale state warning");
  fail = true;
  await page.locator('#connection[data-state="offline"]').waitFor({ timeout: 7000 });
  assert.equal(await card.count(), 1);
  assert.match(await page.locator("#connection-error").innerText(), /Dernier relevé conservé/);
  assert.equal(await card.locator(".status-text").innerText(), "Non confirmé");
  const frozenCountdown = await card.locator(".eta-value").innerText();
  await page.waitForTimeout(1200);
  assert.equal(await card.locator(".eta-value").innerText(), frozenCountdown);
  await screenshot(page, "v2-04-mobile-offline-fixture.png");
  passed("Offline state marks activity unconfirmed and freezes countdown");
  fail = false; stale = false;
  await waitConnected(page);
  assert.equal(await page.locator("#connection-error").isVisible(), false);
  passed("Automatic recovery from outage");

  state.jobs = [];
  await page.waitForFunction(() => document.querySelectorAll(".job-card").length === 0, null, { timeout: 7000 });
  assert.equal(await page.locator("#empty-state").isVisible(), true);
  assert.match(await page.locator("#empty-title").innerText(), /Aucun calcul/);
  passed("Empty monitored state");
  assert.equal(errors.length, 0, JSON.stringify(errors, null, 2));
  await context.close();
}

(async () => {
  try {
    browser = await chromium.launch({ executablePath: edgePath, headless: true, args: ["--no-first-run", "--disable-extensions"] });
    const liveState = await realServerChecks();
    await simulatedStateChecks(liveState);
    passed("No JavaScript errors or CSP violations", "live server and isolated browser-only fixtures");
    fs.writeFileSync(path.join(qa, "browser-test-results-v2.json"), JSON.stringify({ tested_at: new Date().toISOString(), origin, browser: await browser.version(), results, errors }, null, 2));
    console.log(`\n${results.length} checks passed. Screenshots: ${qa}`);
  } catch (error) {
    console.error(error.stack || error);
    if (currentPage && !currentPage.isClosed()) await screenshot(currentPage, "failure.png").catch(() => {});
    fs.writeFileSync(path.join(qa, "browser-test-results-v2.json"), JSON.stringify({ tested_at: new Date().toISOString(), origin, results, errors, failure: error.stack || String(error) }, null, 2));
    process.exitCode = 1;
  } finally {
    if (browser) await browser.close();
  }
})();
