import test from 'node:test';
import assert from 'node:assert/strict';

const presets = await import('../frontend/src/components/runPresets.ts');
const preview = await import('../frontend/src/components/csvPreview.ts');

test('balanced preset retains the original calibration defaults', () => {
  assert.equal(typeof presets.getRunPresetConfig, 'function', 'parameter presets are available');
  assert.deepEqual(presets.getRunPresetConfig('balanced', 'dataset-a'), {
    dataset_id: 'dataset-a', algorithm: 'pca', train_ratio: 0.35, threshold_quantile: 0.99,
    pca_variance: 0.9, window: 8, min_event_length: 3, merge_gap: 2,
  });
});

test('sensitive and quiet presets change actual event parameters without changing the chosen model', () => {
  assert.equal(typeof presets.getRunPresetConfig, 'function', 'parameter presets are available');
  const sensitive = presets.getRunPresetConfig('sensitive', 'dataset-b', 'temporal');
  const quiet = presets.getRunPresetConfig('quiet', 'dataset-b', 'temporal');
  assert.equal(sensitive.algorithm, 'temporal');
  assert.equal(quiet.algorithm, 'temporal');
  assert.ok(sensitive.threshold_quantile < 0.99);
  assert.equal(sensitive.min_event_length, 1);
  assert.ok(quiet.threshold_quantile > 0.99);
  assert.ok(quiet.min_event_length > 3);
  assert.equal(sensitive.dataset_id, 'dataset-b');
});

test('UTF-8 BOM CSV header preview keeps commas and escaped quotes inside column names', async () => {
  assert.equal(typeof preview.readCsvPreview, 'function', 'CSV preview is available');
  const file = new Blob(['\uFEFFtimestamp,"cpu,load","sensor ""A""",label\r\n2026-01-01,1,2,0\r\n']);
  const result = await preview.readCsvPreview(file);
  assert.deepEqual(result.columns, ['timestamp', 'cpu,load', 'sensor "A"', 'label']);
  assert.equal(result.encoding, 'UTF-8');
  assert.equal(result.timestampColumn, 'timestamp');
  assert.equal(result.labelColumn, 'label');
});

test('GB18030 header preview identifies Chinese timestamp and label fields', async () => {
  assert.equal(typeof preview.readCsvPreview, 'function', 'CSV preview is available');
  const file = new Blob([Buffer.from('cab1bce42c612c622cb1eac7a90a323032362d30312d30312c312c322c30', 'hex')]);
  const result = await preview.readCsvPreview(file);
  assert.deepEqual(result.columns, ['时间', 'a', 'b', '标签']);
  assert.equal(result.encoding, 'GB18030');
  assert.equal(result.timestampColumn, '时间');
  assert.equal(result.labelColumn, '标签');
});

test('preview reads only a bounded file slice and tolerates an incomplete trailing UTF-8 character', async () => {
  assert.equal(typeof preview.readCsvPreview, 'function', 'CSV preview is available');
  const prefix = Buffer.from('time,a,b\n');
  const bytes = new Uint8Array(128 * 1024);
  bytes.fill(65); bytes.set(prefix); bytes[bytes.length - 1] = 0xe6;
  const requests = [];
  const file = {
    size: 25 * 1024 * 1024,
    slice(start, end) { requests.push([start, end]); return new Blob([bytes]); },
  };
  const result = await preview.readCsvPreview(file);
  assert.deepEqual(result.columns, ['time', 'a', 'b']);
  assert.equal(result.encoding, 'UTF-8');
  assert.deepEqual(requests, [[0, 128 * 1024]]);
});

test('an unfinished quoted header gives a readable preview error', async () => {
  assert.equal(typeof preview.readCsvPreview, 'function', 'CSV preview is available');
  await assert.rejects(preview.readCsvPreview(new Blob(['"timestamp,a,b\n1,2,3'])), /表头|引号/);
});
