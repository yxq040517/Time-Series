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
const dataset = { id:'current-dataset', features:['b','a'] };
const model = { id:'saved-temporal', name:'Saved temporal', version:2, algorithm:'temporal', status:'published', features:['a','b'], dataset_name:'Normal history', fit_start:0, fit_end:100, calibration_end:120, threshold:1.25, window:10, training_config:{threshold_quantile:.982,pca_variance:.85} };
const models = [model, {...model,id:'ready',status:'ready'}, {...model,id:'disabled',status:'disabled'}, {...model,id:'incompatible',features:['a','c']}, {...model,id:'saved-pca',algorithm:'pca'}];
const config = {dataset_id:'old-dataset', model_id:model.id, min_event_length:4, merge_gap:5, stream_id:'stream-a'};
window.reconfigure = (initialConfig, busy = false, items = models) => root.render(React.createElement(RunConfiguration, { dataset, models:items, modelLoading:false, modelError:'', onRefresh:()=>{}, onTrain:()=>window.trainingRequested=true, initialConfig, busy, onRun:value=>window.lastRun=value }));
window.reconfigure(config);
</script></body></html>`);
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
try {
  await page.goto(`${base}/${name}`);
  await page.getByTestId('run-button').waitFor();
  assert.deepEqual(await page.getByTestId('model-select').locator('option').evaluateAll(options => options.map(option => option.value)), ['', 'saved-temporal', 'saved-pca'], 'only published exact feature sets are usable, independent of column order');
  await page.getByTestId('run-button').click();
  assert.deepEqual(await page.evaluate(() => window.lastRun), {dataset_id:'current-dataset',model_id:'saved-temporal',min_event_length:4,merge_gap:5,stream_id:'stream-a'});
  assert.equal(await page.locator('#configuration input[name="threshold_quantile"], #configuration input[name="window"], #configuration input[name="pca_variance"]').count(), 0, 'training parameters are not editable during detection');
  await page.getByTestId('model-select').selectOption('saved-pca');
  await page.getByTestId('run-button').click();
  assert.equal((await page.evaluate(() => window.lastRun)).stream_id, undefined, 'non-temporal detection cannot retain a temporal stream');
  await page.evaluate(() => window.reconfigure(undefined, true));
  await page.waitForFunction(() => document.querySelector('[data-testid="run-button"]')?.disabled);
  assert.equal(await page.getByTestId('model-select').isDisabled(), true);
  await page.evaluate(() => window.reconfigure(undefined, false, []));
  await page.getByTestId('train-empty-action').click();
  assert.equal(await page.getByTestId('run-button').isDisabled(), true, 'missing model never falls back to automatic training');
  assert.equal(await page.evaluate(() => window.trainingRequested), true);
  console.log('Published compatibility, saved-config reuse, fixed parameters, stream isolation and empty-model guards passed.');
} finally {
  await browser.close();
  await fs.unlink(temporary);
}
