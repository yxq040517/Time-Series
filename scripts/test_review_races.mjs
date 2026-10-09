import assert from 'node:assert/strict';
import { chromium } from '../frontend/node_modules/playwright/index.mjs';

// Only the built frontend is loaded from the server. Every API call is mocked,
// so running this regression never creates or edits the user's stored data.
const base = process.env.CHRONOLENS_URL || 'http://127.0.0.1:8767';
const originalNote = '已持久化的原始备注';
const datasetId = '11111111111111111111111111111111';
const runId = '22222222222222222222222222222222';
const eventId = 'e-0001';
const featureNames = ['signal_a', 'signal_b'];
const timestamp = index => new Date(Date.UTC(2026, 0, 1, 0, index)).toISOString();
const dataset = {
  id: datasetId, name: '复核并发回归样本', source: 'upload', created_at: timestamp(0),
  rows: 120, features: featureNames, start_time: timestamp(0), end_time: timestamp(119),
  has_labels: false, description: '浏览器内模拟数据，不写入数据库。',
  quality: { missing_cells: 0, missing_ratio: 0, duplicate_timestamps: 0,
    constant_features: [], ignored_columns: [], warnings: [] },
};
const modelId = '33333333333333333333333333333333';
const config = { dataset_id: datasetId, model_id: modelId, min_event_length: 3, merge_gap: 2 };
const summary = { points: 120, n_features: 2, scored_points: 120, warmup_points: 0, threshold: 1,
  anomaly_points: 5, anomaly_ratio: 5 / 120, event_count: 1, duration_ms: 25,
  explanation_method: 'PCA 标准化重构残差平方（非因果解释）',
  score_stats: { min: .2, max: 3, median: .2, p95: 3 }, metrics: null };
const run = { id: runId, dataset_id: datasetId, dataset_name: dataset.name, algorithm: 'pca',
  config, model_id: modelId, model_name: '复核竞态模型', model_version: 1,
  status: 'completed', progress: 100, message: '检测完成', created_at: timestamp(0),
  completed_at: timestamp(1), error: null, summary };
const originalEvent = { id: eventId, start: 60, end: 65, start_time: timestamp(60), end_time: timestamp(64),
  length: 5, anomaly_points: 5, peak_score: 3, mean_score: 3, severity: 'high',
  top_feature: featureNames[0], status: 'unreviewed', note: originalNote };
const clone = value => JSON.parse(JSON.stringify(value));
const report = { unsavedDraftRemainsUnsaved: false, delayedExplanationKeepsLatestStatus: false,
  mockedWrites: 0, realWrites: 0 };
const browser = await chromium.launch({ headless: true });

async function scenario({ holdExplanation = false } = {}) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1100 }, locale: 'zh-CN' });
  const page = await context.newPage();
  const state = { event: clone(originalEvent) };
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  let releaseExplanation;
  const explanationGate = new Promise(resolve => { releaseExplanation = resolve; });
  let announceExplanation;
  const explanationStarted = new Promise(resolve => { announceExplanation = resolve; });
  await page.route('**/api/**', async route => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const fulfill = body => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
    if (request.method() === 'POST' && path === `/api/runs/${runId}/events/review`) {
      const payload = request.postDataJSON();
      assert.deepEqual(payload.event_ids, [eventId]);
      assert.equal(payload.note, undefined, 'bulk status changes must not submit the note draft');
      state.event.status = payload.status;
      report.mockedWrites += 1;
      return fulfill({ items: [clone(state.event)] });
    }
    assert.equal(request.method(), 'GET', `Unexpected API mutation: ${request.method()} ${path}`);
    if (path === '/api/datasets') return fulfill({ items: [dataset] });
    if (path === '/api/models') return fulfill({ items: [] });
    if (path === `/api/datasets/${datasetId}/preview`) return fulfill({ columns: ['timestamp', ...featureNames],
      rows: Array.from({ length: 8 }, (_, index) => ({ timestamp: timestamp(index), signal_a: index, signal_b: index * 2 })) });
    if (path === '/api/runs') return fulfill({ items: [run] });
    if (path === `/api/runs/${runId}`) return fulfill(run);
    if (path === `/api/runs/${runId}/events`) return fulfill({ items: [clone(state.event)] });
    if (path === `/api/runs/${runId}/events/${eventId}/explanation`) {
      // Capture the old server snapshot before holding its response. The bulk
      // review completes first, reproducing an actual response-order race.
      const snapshot = clone(state.event);
      announceExplanation();
      if (holdExplanation) await explanationGate;
      return fulfill({ event: snapshot, method: summary.explanation_method,
        top_features: [{ name: featureNames[0], contribution: 3, share: .75, rank: 1 },
          { name: featureNames[1], contribution: 1, share: .25, rank: 2 }],
        context: { start: 30, end: 95 }, notes: ['变量偏离不证明因果关系。'] });
    }
    if (path === `/api/runs/${runId}/insights`) return fulfill({ run_id: runId,
      review: { unreviewed: 1, confirmed: 0, false_positive: 0 }, severity: { high: 1, medium: 0, low: 0 },
      top_features: [{ name: featureNames[0], event_count: 1 }],
      timeline: [{ start: 0, end: 120, label: timestamp(0), anomaly_points: 5, total_points: 120 }] });
    if (path === `/api/runs/${runId}/series`) {
      const start = Number(url.searchParams.get('start') || 0);
      const end = Number(url.searchParams.get('end') || 120);
      const indices = Array.from({ length: end - start }, (_, index) => start + index);
      const features = (url.searchParams.get('features') || featureNames.join(',')).split(',');
      const entries = transform => Object.fromEntries(features.map(name => [name, indices.map(transform)]));
      const flags = indices.map(index => Number(index >= 60 && index < 65));
      return fulfill({ indices, timestamps: indices.map(timestamp), scores: flags.map(value => value ? 3 : .2),
        threshold: 1, train_end: null, scored: indices.map(() => true), feature_names: features, values: entries(index => Math.sin(index / 5)),
        reference: entries(index => Math.sin(index / 5) * .9), contributions: entries(index => index >= 60 && index < 65 ? 3 : .2),
        flags, labels: null, sampling: { total_points: indices.length, returned_points: indices.length } });
    }
    if (path === `/api/runs/${runId}/heatmap`) return fulfill({ features: featureNames,
      timestamps: [timestamp(30), timestamp(60)], indices: [30, 60], values: [[.2, 3], [.1, 1]],
      method: summary.explanation_method, range: { start: 0, end: 120 } });
    throw new Error(`Unexpected API request: ${path}`);
  });
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  await page.getByTestId(`event-row-${eventId}`).waitFor({ state: 'visible' });
  return { context, page, state, pageErrors, explanationStarted, releaseExplanation };
}

try {
  const first = await scenario();
  try {
    await first.page.getByTestId(`event-row-${eventId}`).click();
    await first.page.getByTestId('review-note').waitFor({ state: 'visible' });
    const draft = 'UNSAVED REVIEW DRAFT';
    await first.page.getByTestId('review-note').fill(draft);
    await first.page.getByRole('checkbox', { name: `选择事件 ${eventId}`, exact: true }).check();
    await first.page.getByTestId('batch-false-positive').click();
    await first.page.getByTestId('batch-message').waitFor({ state: 'visible' });
    assert.equal(first.state.event.note, originalNote, 'bulk review must retain the persisted note');
    assert.equal(await first.page.getByTestId('review-note').inputValue(), draft, 'bulk review must preserve the local draft');
    assert.doesNotMatch(await first.page.locator('.review-actions > span').innerText(), /已保存复核记录/,
      'the unsaved note must not be presented as persisted');
    assert.equal(await first.page.getByTestId('review-status').inputValue(), 'false_positive');
    assert.deepEqual(first.pageErrors, []);
    report.unsavedDraftRemainsUnsaved = true;
  } finally {
    first.releaseExplanation();
    await first.context.close();
  }
  const second = await scenario({ holdExplanation: true });
  try {
    await second.page.getByTestId(`event-row-${eventId}`).click();
    await second.explanationStarted;
    await second.page.getByRole('checkbox', { name: `选择事件 ${eventId}`, exact: true }).check();
    await second.page.getByTestId('batch-false-positive').click();
    await second.page.getByTestId('batch-message').waitFor({ state: 'visible' });
    assert.equal(second.state.event.status, 'false_positive');
    second.releaseExplanation();
    await second.page.getByTestId('review-status').waitFor({ state: 'visible' });
    assert.equal(await second.page.getByTestId('review-status').inputValue(), 'false_positive',
      'the delayed old explanation must not overwrite the latest review');
    assert.match(await second.page.getByTestId(`event-row-${eventId}`).innerText(), /误报/);
    assert.equal(await second.page.getByTestId('review-note').inputValue(), originalNote);
    assert.deepEqual(second.pageErrors, []);
    report.delayedExplanationKeepsLatestStatus = true;
  } finally {
    second.releaseExplanation();
    await second.context.close();
  }
  assert.equal(report.mockedWrites, 2);
  console.log(JSON.stringify(report));
} finally {
  await browser.close();
}
