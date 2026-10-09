import { chromium } from '../frontend/node_modules/playwright/index.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const out = path.join(root, 'artifacts');
await fs.mkdir(out, { recursive: true });
const base = process.env.CHRONOLENS_URL || 'http://127.0.0.1:8767';
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 1512, height: 1040 }, locale: 'zh-CN', acceptDownloads: true });
const page = await context.newPage();
const errors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => { if (message.type() === 'error' && !message.text().includes('Failed to load resource')) errors.push(message.text()); });
const report = {};
async function trainPublishedModel(dataset, algorithm) {
  const response = await context.request.post(`${base}/api/trainings`, { data: { dataset_id: dataset.id, name: `Browser ${algorithm} ${Date.now()}`, algorithm,
    fit_start: 0, fit_end: Math.floor(dataset.rows * .2), calibration_end: Math.floor(dataset.rows * .35), threshold_quantile: .99, pca_variance: .9, window: 8 } });
  assert.equal(response.status(), 202, await response.text());
  const submitted = await response.json();
  for (let attempt = 0; attempt < 300; attempt++) {
    const job = await (await context.request.get(`${base}/api/trainings/${submitted.id}`)).json();
    assert.notEqual(job.status, 'failed', job.error || job.message);
    if (job.status === 'completed') {
      const published = await context.request.post(`${base}/api/models/${job.model_id}/publish`);
      assert.equal(published.status(), 200, await published.text()); return published.json();
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  assert.fail('training timed out');
}
try {
  // Real seed request ensures an old dataset is selected before testing a transition.
  const seed = await context.request.post(base + '/api/datasets/demo', { data: { seed: 42 } });
  assert.equal(seed.status(), 200);
  const seededDataset = await seed.json();
  const pcaModel = await trainPublishedModel(seededDataset, 'pca');
  await page.goto(base, { waitUntil: 'networkidle' });
  await page.getByTestId('dataset-select').selectOption(seededDataset.id);
  await page.getByTestId('run-button').waitFor({ state: 'visible' });
  await page.getByTestId('model-select').selectOption(pcaModel.id);
  // Delay the real request, not its result: regression for starting a run on the old dataset.
  await page.route('**/api/datasets/demo', async route => { await new Promise(resolve => setTimeout(resolve, 500)); await route.continue(); });
  const demoResponse = page.waitForResponse(response => response.url().endsWith('/api/datasets/demo') && response.request().method() === 'POST');
  await page.getByTestId('demo-button').first().click();
  assert.equal(await page.getByTestId('run-button').isDisabled(), true, 'run must wait while the dataset changes');
  const demo = await (await demoResponse).json();
  await page.waitForFunction(id => document.querySelector('[data-testid="dataset-select"]')?.value === id, demo.id);
  await page.getByTestId('model-select').selectOption(pcaModel.id);
  await page.waitForFunction(() => !document.querySelector('[data-testid="run-button"]')?.disabled);
  await page.unroute('**/api/datasets/demo');
  report.datasetTransitionGuard = true;
  await page.getByTestId('run-button').click();
  const firstEvent = page.locator('[data-testid^="event-row-"]').first();
  await firstEvent.waitFor({ state: 'visible', timeout: 45000 });
  const eventId = (await firstEvent.getAttribute('data-testid')).replace('event-row-', '');
  const pcaRunId = await page.locator('#run-select').inputValue();
  await firstEvent.click();
  await page.getByTestId('review-note').waitFor({ state: 'visible' });
  const note = '浏览器验收：已核对异常区间与变量贡献。';
  await page.getByTestId('review-note').fill(note);
  await page.getByTestId('review-status').selectOption('confirmed');
  const savedResponse = page.waitForResponse(response => response.request().method() === 'PATCH' && response.url().includes('/events/'));
  await page.getByTestId('review-save').click();
  assert.equal((await savedResponse).status(), 200);
  const explanationResponse = await context.request.get(`${base}/api/runs/${pcaRunId}/events/${eventId}/explanation`);
  const explanation = await explanationResponse.json();
  assert.equal(explanation.event.status, 'confirmed');
  assert.equal(explanation.event.note, note);
  assert.match(await page.locator('.range-heading').innerText(), new RegExp(`#${explanation.context.start}–#${explanation.context.end - 1}`));
  report.eventContextLinked = true;
  await page.waitForTimeout(700);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: path.join(out, 'dashboard-desktop.png'), fullPage: true });
  report.canvasesObserved = await page.locator('canvas').count();
  report.desktopOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2);
  assert.equal(report.desktopOverflow, false, 'desktop must not overflow horizontally');
  for (const [testid, filename] of [['export-csv','browser-analysis.csv'], ['export-report','browser-report.html']]) {
    const waiting = page.waitForEvent('download');
    await page.getByTestId(testid).click();
    const download = await waiting;
    const target = path.join(out, filename);
    await download.saveAs(target);
    assert.ok((await fs.readFile(target, 'utf8')).includes(note), 'download must contain the persisted review');
  }
  report.reviewAndExports = true;
  await page.getByTestId('reset-range').click();
  await page.waitForFunction(() => document.querySelector('.range-heading')?.textContent.includes('#0–#2399'));
  const score = page.locator('.score-card [role="img"]');
  await score.scrollIntoViewIfNeeded();
  const box = await score.boundingBox();
  await page.mouse.move(box.x + box.width * .6, box.y + box.height * .4);
  await page.mouse.wheel(0, -500);
  await page.waitForFunction(() => !document.querySelector('.range-heading')?.textContent.includes('#0–#2399'));
  report.actualWheelZoomLinked = true;
  await page.getByTestId('import-button').first().click();
  await page.getByTestId('file-input').setInputFiles(path.join(root, 'examples/server_metrics.csv'));
  await page.getByTestId('upload-submit').click();
  await page.getByTestId('upload-submit').waitFor({ state: 'hidden', timeout: 30000 });
  const uploadedDatasetId = await page.getByTestId('dataset-select').inputValue();
  const uploadedDataset = await (await context.request.get(`${base}/api/datasets/${uploadedDatasetId}`)).json();
  const temporalModel = await trainPublishedModel(uploadedDataset, 'temporal');
  await page.reload({ waitUntil: 'networkidle' });
  await page.getByTestId('dataset-select').selectOption(uploadedDatasetId);
  await page.getByTestId('model-select').selectOption(temporalModel.id);
  await page.getByTestId('run-button').click();
  await page.locator('[data-testid^="event-row-"]').first().waitFor({ state: 'visible', timeout: 45000 });
  const uploadedId = await page.getByTestId('dataset-select').inputValue();
  const temporalRunId = await page.locator('#run-select').inputValue();
  report.uploadAndTemporalRun = true;
  // Actual rejected upload leaves the dialog open and retains the analysis underneath.
  await page.getByTestId('import-button').first().click();
  await page.getByTestId('file-input').setInputFiles({ name: 'broken.csv', mimeType: 'text/csv', buffer: Buffer.from('timestamp,a,b\nbad,1,2') });
  await page.getByTestId('upload-submit').click();
  await page.locator('.import-dialog [role="alert"]').waitFor({ state: 'visible' });
  assert.equal(await page.getByTestId('dataset-select').inputValue(), uploadedId);
  await page.getByRole('button', { name: '取消', exact: true }).click();
  report.failedUploadRetainsAnalysis = true;
  await page.reload({ waitUntil: 'networkidle' });
  await page.getByTestId('dataset-select').selectOption(uploadedId);
  await page.locator(`#run-select option[value="${temporalRunId}"]`).waitFor({ state: 'attached' });
  await page.locator('#run-select').selectOption(temporalRunId);
  await page.locator('[data-testid^="event-row-"]').first().waitFor({ state: 'visible', timeout: 30000 });
  report.historyAfterReload = true;
  report.sameDatasetPreservesHistory = true;
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(1200);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: path.join(out, 'dashboard-mobile.png'), fullPage: true });
  report.mobileOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2);
  assert.equal(report.mobileOverflow, false, 'mobile must not overflow horizontally');
  assert.deepEqual(errors, [], 'browser must not emit runtime errors');
  report.browserErrors = errors;
  await fs.writeFile(path.join(out, 'verification-browser.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  await page.screenshot({ path: path.join(out, 'browser-failure.png'), fullPage: true });
  await fs.writeFile(path.join(out, 'browser-failure.json'), JSON.stringify({ error: error.stack, errors, body: await page.locator('body').innerText() }, null, 2));
  throw error;
} finally {
  await browser.close();
}
