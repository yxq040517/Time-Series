import { chromium } from '../frontend/node_modules/playwright/index.mjs';
import assert from 'node:assert/strict';

const base = process.env.CHRONOLENS_URL || 'http://127.0.0.1:5173';
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ locale: 'zh-CN' });
const errors = [];
page.on('pageerror', error => errors.push(error.message));
try {
  await page.request.post(base + '/api/datasets/demo', { data: { seed: 42 } });
  await page.goto(base, { waitUntil: 'networkidle' });
  await page.getByTestId('run-button').waitFor();
  await page.getByTestId('import-button').first().click();
  const csv = '自定义时间,"cpu,load",memory,自定义标签\n' + Array.from({ length: 120 }, (_, i) => `2026-01-01T00:${String(Math.floor(i / 60)).padStart(2, '0')}:${String(i % 60).padStart(2, '0')}Z,${10 + i / 10},${20 + i / 5},0`).join('\n');
  const transfer = await page.evaluateHandle(text => {
    const data = new DataTransfer();
    data.items.add(new File([text], 'dragged-metrics.csv', { type: 'text/csv' }));
    return data;
  }, csv);
  await page.locator('.import-dropzone').dispatchEvent('drop', { dataTransfer: transfer });
  await page.getByTestId('csv-header-preview').waitFor();
  assert.ok((await page.getByTestId('csv-header-preview').innerText()).includes('cpu,load'));
  await page.getByRole('combobox', { name: '时间列名', exact: true }).selectOption('自定义时间');
  await page.getByRole('combobox', { name: '标签列名', exact: true }).selectOption('自定义标签');
  const uploaded = page.waitForResponse(response => response.url().endsWith('/api/datasets/upload') && response.request().method() === 'POST');
  await page.getByTestId('upload-submit').click();
  const response = await uploaded;
  assert.equal(response.status(), 200, 'drag-and-drop can submit without a populated file input');
  const dataset = await response.json();
  assert.deepEqual(dataset.features, ['cpu,load', 'memory']);
  assert.equal(dataset.has_labels, true);
  await page.getByTestId('upload-submit').waitFor({ state: 'hidden' });
  await page.getByTestId('import-button').first().click();
  await page.getByTestId('file-input').setInputFiles({ name: 'bad.csv', mimeType: 'text/csv', buffer: Buffer.from('timestamp,a,b\nbad,1,2') });
  await page.getByTestId('upload-submit').click();
  await page.locator('.import-dialog [role="alert"]').waitFor();
  assert.equal(await page.getByTestId('dataset-select').inputValue(), dataset.id);
  assert.deepEqual(errors, []);
  console.log('Bounded header preview, mapped drag-and-drop upload and failed upload retention passed.');
} finally { await browser.close(); }
