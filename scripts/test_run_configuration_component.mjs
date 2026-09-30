import { chromium } from '../frontend/node_modules/playwright/index.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const base = process.env.CHRONOLENS_URL || 'http://127.0.0.1:5173';
const name = `.configuration-test-${process.pid}.html`;
const temporary = new URL(`../frontend/${name}`, import.meta.url);
await fs.writeFile(temporary, `<!doctype html><html><body><div id="root"></div><script type="module">
import React from 'react';
import { createRoot } from 'react-dom/client';
import { RunConfiguration } from './src/components/RunConfiguration.tsx';
const root = createRoot(document.getElementById('root'));
const config = { dataset_id:'old-dataset', algorithm:'temporal', train_ratio:0.42, threshold_quantile:0.982, pca_variance:0.85, window:10, min_event_length:4, merge_gap:5 };
window.reconfigure = (initialConfig, busy = false) => root.render(React.createElement(RunConfiguration, { datasetId:'current-dataset', initialConfig, busy, onRun: value => window.lastRun = value }));
window.reconfigure(config);
</script></body></html>`);
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
try {
  await page.goto(`${base}/${name}`);
  await page.getByTestId('run-button').waitFor();
  assert.equal(await page.getByTestId('algorithm-select').inputValue(), 'temporal');
  assert.equal(await page.getByRole('spinbutton', { name: '历史训练比例', exact: true }).inputValue(), '42');
  assert.equal(await page.getByRole('spinbutton', { name: '历史窗口', exact: true }).inputValue(), '10');
  await page.getByTestId('run-button').click();
  const restored = await page.evaluate(() => window.lastRun);
  assert.equal(restored.dataset_id, 'current-dataset');
  assert.equal(restored.threshold_quantile, 0.982);
  await page.evaluate(() => window.reconfigure({ dataset_id:'old', algorithm:'pca', train_ratio:0.35, threshold_quantile:0.99, pca_variance:0.9, window:8, min_event_length:3, merge_gap:2 }));
  await page.getByRole('spinbutton', { name: '目标解释方差', exact: true }).waitFor();
  assert.equal(await page.getByRole('spinbutton', { name: '历史训练比例', exact: true }).inputValue(), '35');
  await page.getByRole('spinbutton', { name: '目标解释方差', exact: true }).fill('');
  await page.getByTestId('algorithm-select').selectOption('temporal');
  await page.getByTestId('run-button').click();
  const switched = await page.evaluate(() => window.lastRun);
  assert.equal(switched.pca_variance, 0.9, 'switching models uses valid defaults for a cleared inactive parameter');
  assert.equal(switched.algorithm, 'temporal');
  await page.evaluate(() => window.reconfigure(undefined, true));
  await page.waitForFunction(() => document.querySelector('[data-testid="run-button"]')?.disabled);
  assert.equal(await page.getByTestId('run-button').isDisabled(), true);
  assert.equal(await page.getByRole('button', { name: '敏感发现', exact: true }).isDisabled(), true);
  console.log('Historical configuration reuse, current dataset targeting, model-switch defaults and busy guards passed.');
} finally {
  await browser.close();
  await fs.unlink(temporary);
}
