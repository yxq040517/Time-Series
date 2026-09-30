import { chromium } from '../frontend/node_modules/playwright/index.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const out=path.join(root,'artifacts','upgrade');
await fs.mkdir(out,{recursive:true});
const base=process.env.CHRONOLENS_URL||'http://127.0.0.1:8767';
const browser=await chromium.launch({headless:true});
const page=await browser.newPage({viewport:{width:1512,height:1040},locale:'zh-CN'});
const errors=[];page.on('pageerror',error=>errors.push(error.message));
const evidence={};
try {
  const seed=await page.request.post(`${base}/api/datasets/demo`,{data:{seed:73}});
  assert.equal(seed.status(),200);
  const seededDataset=await seed.json();
  for(const algorithm of ['pca','temporal','isolation']){
    const creation=await page.request.post(`${base}/api/runs`,{data:{dataset_id:seededDataset.id,algorithm}});
    assert.equal(creation.status(),202);
    const created=await creation.json();
    let status=created.status;
    for(let attempt=0;attempt<100 && status!=='completed';attempt++){
      await new Promise(resolve=>setTimeout(resolve,250));
      const current=await(await page.request.get(`${base}/api/runs/${created.id}`)).json();
      assert.notEqual(current.status,'failed',current.error||'detection failed');status=current.status;
    }
    assert.equal(status,'completed');
  }
  await page.goto(base,{waitUntil:'networkidle'});
  await page.getByTestId('dataset-select').selectOption(seededDataset.id);
  await page.getByRole('button',{name:'数据诊断',exact:true}).click({timeout:5000});
  const id=await page.getByTestId('dataset-select').inputValue();
  const dataset=await (await page.request.get(`${base}/api/datasets/${id}`)).json();
  await page.getByTestId('profile-feature-count').waitFor();
  assert.match(await page.getByTestId('profile-feature-count').innerText(),new RegExp(String(dataset.features.length)));
  await page.getByTestId('profile-search').fill(dataset.features[0]);
  assert.ok(await page.getByTestId('profile-table').locator('tbody tr').count()>=1);
  evidence.profile=true;
  await page.getByRole('button',{name:'模型对比',exact:true}).click();
  const runs=await (await page.request.get(`${base}/api/runs?dataset_id=${id}`)).json();
  const completed=runs.items.filter(r=>r.status==='completed');
  assert.ok(completed.length>=2);
  if(completed.length>=2){
    for(const run of completed.slice(0,2)){const cb=page.getByTestId(`comparison-run-${run.id}`);if(!(await cb.isChecked()))await cb.check();}
    await page.getByTestId('comparison-table').waitFor();
    for(const run of completed.slice(0,2))assert.ok((await page.getByTestId('comparison-table').innerText()).includes(String(run.summary.event_count)));
    evidence.comparison=true;
  }
  await page.getByRole('button',{name:'分析工作台',exact:true}).click();
  const first=page.locator('[data-testid^="event-row-"]').first();
  await first.waitFor({state:'visible'});
  if(await first.count()){
    await page.getByTestId('event-search').fill('no-matching-variable-xyz');
    assert.equal(await page.locator('[data-testid^="event-row-"]').count(),0);
    await page.getByTestId('event-search').fill('');
    await page.getByTestId('event-severity').selectOption('high');
    await page.getByTestId('event-severity').selectOption('all');
    await page.getByTestId('event-sort').selectOption('peak');
    await first.click();
    await page.getByTestId('review-note').waitFor();
    const draft='未提交备注：批量复核只修改结论，不保存这份草稿';
    await page.getByTestId('review-note').fill(draft);
    await page.getByTestId('select-filtered-events').check();
    const runId=await page.locator('#run-select').inputValue();
    const response=page.waitForResponse(r=>r.request().method()==='POST'&&r.url().endsWith('/events/review'));
    await page.getByTestId('batch-confirm').click();
    assert.equal((await response).status(),200);
    await page.getByTestId('batch-message').waitFor();
    const events=await(await page.request.get(`${base}/api/runs/${runId}/events`)).json();
    assert.ok(events.items.length&&events.items.every(e=>e.status==='confirmed'));
    assert.ok(events.items.every(e=>e.note!==draft));
    assert.equal(await page.getByTestId('review-note').inputValue(),draft);
    assert.equal(await page.locator('.saved-label').count(),0,'batch review must not mark an unsaved note as saved');
    evidence.batchReview=true;
  }
  await page.evaluate(()=>window.scrollTo(0,0));
  await page.screenshot({path:path.join(out,'workbench-desktop.png'),fullPage:true});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+2),false);
  for(const width of [390,768]){
    await page.setViewportSize({width,height:844});
    for(const name of ['分析工作台','数据诊断','模型对比','数据集管理','检测历史']){
      await page.getByRole('button',{name,exact:true}).click();
      await page.waitForTimeout(200);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+2),false,`overflow ${name} ${width}`);
    }
  }
  await page.getByRole('button',{name:'分析工作台',exact:true}).click();
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:path.join(out,'workbench-mobile.png'),fullPage:true});
  assert.deepEqual(errors,[]);evidence.browserErrors=errors;
  await fs.writeFile(path.join(out,'verification-upgrade.json'),JSON.stringify(evidence,null,2));
  console.log(JSON.stringify(evidence,null,2));
}finally{await browser.close();}
