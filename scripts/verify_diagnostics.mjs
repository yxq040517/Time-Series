import { chromium } from '../frontend/node_modules/playwright/index.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const out = path.join(root, 'artifacts', 'upgrade');
const base = process.env.CHRONOLENS_URL || 'http://127.0.0.1:8766';
await fs.mkdir(out, { recursive: true });
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1512, height: 1040 }, locale: 'zh-CN' });
const errors = [];
page.on('pageerror', (error) => errors.push(error.message));
const evidence = {};
const counts = new Intl.NumberFormat('zh-CN');
const numbers = new Intl.NumberFormat('zh-CN', { maximumSignificantDigits: 6 });
const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
async function json(response, expected = 200) {
  assert.equal(response.status(), expected, await response.text());
  return response.json();
}
async function awaitRun(runId) {
  for (let i = 0; i < 300; i++) {
    const run = await json(await page.request.get(`${base}/api/runs/${runId}`));
    if (run.status === 'completed') return run;
    assert.notEqual(run.status, 'failed', run.error || run.message);
    await sleep(100);
  }
  assert.fail(`检测任务 ${runId} 超时`);
}
async function noOverflow() {
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false, 'page must not overflow');
}
try {
  const lines = ['timestamp,flow_count,temperature,constant,with_gaps'];
  for (let i = 0; i < 160; i++) {
    const minute = i - (i >= 5 ? 1 : 0) + (i >= 20 ? 4 : 0);
    const timestamp = new Date(Date.UTC(2026, 0, 1) + minute * 60000).toISOString();
    lines.push(`${timestamp},${10 + i * 2},${20 + i / 20},7,${i % 7 ? i / 2 : ''}`);
  }
  const primary = await json(await page.request.post(`${base}/api/datasets/upload`, { multipart: {
    name: '浏览器验收 · 无标签统计与对比',
    file: { name: 'diagnostics-fixture.csv', mimeType: 'text/csv', buffer: Buffer.from(lines.join('\n')) },
  } }));
  const indexLines = ['counter,level', ...Array.from({ length: 120 }, (_, i) => `${i},${100 + i}`)];
  const indexDataset = await json(await page.request.post(`${base}/api/datasets/upload`, { multipart: {
    name: '浏览器验收 · 样本序号与空历史',
    file: { name: 'diagnostics-index.csv', mimeType: 'text/csv', buffer: Buffer.from(indexLines.join('\n')) },
  } }));
  const configs = [
    { algorithm: 'pca', threshold_quantile: .98 },
    { algorithm: 'temporal', threshold_quantile: .99, window: 4 },
    { algorithm: 'isolation', threshold_quantile: .99 },
    { algorithm: 'pca', threshold_quantile: .97 },
  ];
  const created = [];
  for (const [index, config] of configs.entries()) {
    const training = await json(await page.request.post(`${base}/api/trainings`, { data: {
      dataset_id: primary.id, name: `Diagnostics ${index} ${Date.now()}`, fit_start: 0, fit_end: 100,
      calibration_end: 140, pca_variance: .9, window: 8, ...config,
    } }), 202);
    let modelId;
    for (let attempt = 0; attempt < 300; attempt++) {
      const job = await json(await page.request.get(`${base}/api/trainings/${training.id}`));
      assert.notEqual(job.status, 'failed', job.error || job.message);
      if (job.status === 'completed') { modelId = job.model_id; break; }
      await sleep(100);
    }
    assert.ok(modelId, 'training must produce a durable model');
    await json(await page.request.post(`${base}/api/models/${modelId}/publish`));
    created.push(await json(await page.request.post(`${base}/api/runs`, { data: { dataset_id: primary.id,
      model_id: modelId, min_event_length: 3, merge_gap: 2 } }), 202));
  }
  const completed = await Promise.all(created.map((run) => awaitRun(run.id)));
  evidence.fixture = { datasetId: primary.id, indexDatasetId: indexDataset.id, runIds: completed.map((run) => run.id) };
  const profile = await json(await page.request.get(`${base}/api/datasets/${primary.id}/profile`));
  assert.equal(profile.features.length, 4);
  assert.equal(profile.sampling.median_interval_seconds, 60);
  assert.equal(profile.sampling.duplicate_intervals, 1);
  assert.equal(profile.sampling.irregular_intervals, 1);
  const expectedFlow = { count: 160, missing: 0, min: 10, max: 328, mean: 169, p25: 89.5, median: 169, p75: 248.5 };
  const flow = profile.features.find((feature) => feature.name === 'flow_count');
  for (const [key, value] of Object.entries(expectedFlow)) {
    if (key === 'count' || key === 'missing') assert.equal(flow[key], value);
    else assert.ok(Math.abs(flow[key] - value) < 1e-9, `${key}: ${flow[key]} vs ${value}`);
  }
  assert.ok(profile.correlation.values[2].every((value) => value === null), 'constant correlation must remain unavailable');
  evidence.fixtureObservedStatistics = true;

  await page.goto(base, { waitUntil: 'networkidle' });
  await page.getByTestId('dataset-select').selectOption(primary.id);
  await page.getByRole('button', { name: '数据诊断', exact: true }).click();
  await page.getByTestId('profile-feature-count').waitFor();
  assert.match(await page.getByTestId('profile-feature-count').innerText(), /4/);
  const table = page.getByTestId('profile-table');
  assert.equal(await table.locator('tbody tr').count(), 4);
  await page.getByTestId('profile-search').fill('FLOW_COUNT');
  assert.equal(await table.locator('tbody tr').count(), 1, 'search must ignore letter case');
  const cells = table.locator('tbody tr').first().locator('th,td');
  for (const [position, value] of [[1, 160], [3, 10], [4, 328], [5, 169], [7, 89.5], [8, 169], [9, 248.5]]) assert.equal((await cells.nth(position).innerText()).trim(), position === 1 ? counts.format(value) : numbers.format(value));
  await page.getByTestId('profile-search').fill('no-matching-variable-xyz');
  await page.getByText('没有匹配的变量', { exact: true }).waitFor();
  await page.getByRole('button', { name: '清空搜索', exact: true }).click();
  assert.equal(await table.locator('tbody tr').count(), 4);
  await page.getByTestId('profile-search').fill('with_gaps');
  const gaps = profile.features.find((feature) => feature.name === 'with_gaps');
  assert.match(await table.locator('tbody tr').innerText(), new RegExp(`${counts.format(gaps.missing)} 格`));
  await page.getByTestId('profile-search').fill('');
  assert.equal(await page.locator('.profile-cadence strong').first().innerText(), '60 秒');
  await noOverflow();
  await page.screenshot({ path: path.join(out, 'diagnostics-desktop.png'), fullPage: true });
  evidence.profileActualTableValuesAndSearch = true;

  await page.getByRole('button', { name: '检测任务对比', exact: true }).click();
  const checkboxes = page.locator('[data-testid^="comparison-run-"]');
  await checkboxes.first().waitFor();
  assert.equal(await checkboxes.count(), 4);
  for (const checkbox of await checkboxes.all()) await checkbox.uncheck();
  await page.getByText('选择任务，查看差异', { exact: true }).waitFor();
  for (const run of completed.slice(0, 3)) await page.getByTestId(`comparison-run-${run.id}`).check();
  assert.equal(await page.getByTestId(`comparison-run-${completed[3].id}`).isDisabled(), true, 'fourth selection must be disabled');
  assert.equal(await page.getByTestId('comparison-table').locator('thead th').count(), 4);
  for (const run of completed.slice(0, 3)) {
    assert.equal(await page.getByTestId(`comparison-events-${run.id}`).innerText(), counts.format(run.summary.event_count));
    assert.equal(await page.getByTestId(`comparison-ratio-${run.id}`).innerText(), `${(run.summary.anomaly_ratio * 100).toFixed(2)}%`);
    assert.equal(await page.getByTestId(`comparison-duration-${run.id}`).innerText(), `${(run.summary.duration_ms / 1000).toFixed(2)} 秒`);
    assert.equal(Number(await page.getByTestId(`comparison-duration-${run.id}`).getAttribute('data-value')), run.summary.duration_ms);
    assert.equal(run.summary.metrics, null);
  }
  await page.getByText('当前数据集未提供真值标签', { exact: false }).waitFor();
  const f1 = page.getByTestId('comparison-table').locator('tbody tr').filter({ has: page.getByRole('rowheader', { name: 'F1', exact: true }) });
  assert.deepEqual(await f1.locator('td').allInnerTexts(), ['—', '—', '—']);
  assert.match(await page.locator('.comparison-layout').innerText(), /不同算法的原始分数不可直接比较/);
  assert.ok(await page.locator('.comparison-difference').count() >= 1, 'configuration differences must be marked');
  await page.getByTestId(`comparison-run-${completed[0].id}`).uncheck();
  assert.equal(await page.getByTestId(`comparison-run-${completed[3].id}`).isDisabled(), false, 'deselecting restores capacity');
  await page.getByTestId(`comparison-run-${completed[3].id}`).check();
  await noOverflow();
  await page.screenshot({ path: path.join(out, 'comparison-desktop.png'), fullPage: true });
  evidence.comparisonRealValuesMaximumThreeAndNoLabels = true;

  await page.getByTestId('dataset-select').selectOption(indexDataset.id);
  await page.getByText('先完成一次检测，再开始比较', { exact: true }).waitFor();
  assert.equal(await checkboxes.count(), 0, 'different dataset must not expose prior dataset tasks');
  await page.getByRole('button', { name: '数据诊断', exact: true }).click();
  await page.getByTestId('profile-feature-count').waitFor();
  assert.match(await page.getByTestId('profile-feature-count').innerText(), /2/);
  assert.match(await page.locator('.profile-summary').innerText(), /样本序号/);
  assert.equal(await page.locator('.profile-cadence strong').first().innerText(), '—');
  evidence.datasetIsolationAndIndexCadence = true;

  const primaryRoute = `**/api/datasets/${primary.id}/profile`;
  await page.route(primaryRoute, async (route) => { await sleep(600); await route.continue().catch(() => {}); });
  const slowRequest = page.waitForRequest((request) => request.url().endsWith(`/datasets/${primary.id}/profile`));
  await page.getByTestId('dataset-select').selectOption(primary.id);
  await slowRequest;
  await page.getByTestId('dataset-select').selectOption(indexDataset.id);
  await page.getByTestId('profile-feature-count').waitFor();
  await sleep(700);
  assert.match(await page.getByTestId('profile-feature-count').innerText(), /2/);
  assert.equal(await table.locator('tbody tr').count(), 2, 'late prior-dataset request must not replace current diagnostics');
  await page.unroute(primaryRoute);
  evidence.abortSafeDatasetTransition = true;

  await page.route(primaryRoute, (route) => route.abort('connectionfailed'));
  await page.getByTestId('dataset-select').selectOption(primary.id);
  await page.getByRole('button', { name: '重试诊断', exact: true }).waitFor();
  await page.unroute(primaryRoute);
  await page.getByRole('button', { name: '重试诊断', exact: true }).click();
  await page.getByTestId('profile-feature-count').waitFor();
  assert.match(await page.getByTestId('profile-feature-count').innerText(), /4/);
  evidence.networkFailureRetry = true;

  for (const width of [390, 768]) {
    await page.setViewportSize({ width, height: 844 });
    await page.waitForTimeout(200);
    await page.getByRole('button', { name: '数据诊断', exact: true }).click();
    await page.getByTestId('profile-feature-count').waitFor();
    await noOverflow();
    if (width === 390) await page.screenshot({ path: path.join(out, 'diagnostics-mobile.png'), fullPage: true });
    await page.getByRole('button', { name: '检测任务对比', exact: true }).click();
    await page.getByTestId('comparison-table').waitFor();
    await noOverflow();
    if (width === 390) await page.screenshot({ path: path.join(out, 'comparison-mobile.png'), fullPage: true });
  }
  assert.deepEqual(errors, [], 'browser runtime must remain error free');
  evidence.mobileAndTabletOverflow = false;
  evidence.browserErrors = errors;
  await fs.writeFile(path.join(out, 'verification-diagnostics.json'), JSON.stringify(evidence, null, 2));
  console.log(JSON.stringify(evidence, null, 2));
} catch (error) {
  await page.screenshot({ path: path.join(out, 'diagnostics-failure.png'), fullPage: true });
  await fs.writeFile(path.join(out, 'diagnostics-failure.json'), JSON.stringify({ error: error.stack, evidence, errors, body: await page.locator('body').innerText() }, null, 2));
  throw error;
} finally {
  await browser.close();
}
