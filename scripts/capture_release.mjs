import {chromium} from '../frontend/node_modules/playwright/index.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
const base=process.env.CHRONOLENS_URL || 'http://127.0.0.1:8767';
const out=process.env.CHRONOLENS_OUTPUT;
assert.ok(out);await fs.mkdir(out,{recursive:true});
const browser=await chromium.launch({headless:true});
const page=await browser.newPage({viewport:{width:1512,height:1040},locale:'zh-CN'});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
try{
 const datasets=await(await page.request.get(base+'/api/datasets')).json();
 const runs=await(await page.request.get(base+'/api/runs')).json();
 const baseline=process.env.CHRONOLENS_BASELINE;
 if(baseline){
  const before=JSON.parse(await fs.readFile(baseline,'utf8'));
  assert.deepEqual(datasets.items.map(x=>x.id).sort(),before.dataset_ids.sort());
  assert.deepEqual(runs.items.map(x=>x.id).sort(),before.run_ids.sort());
 }
 const run=runs.items.find(r=>r.status==='completed');assert.ok(run);
 const dataset=datasets.items.find(d=>d.id===run.dataset_id);
 const profile=await page.request.get(`${base}/api/datasets/${dataset.id}/profile`);assert.equal(profile.status(),200);
 const insights=await page.request.get(`${base}/api/runs/${run.id}/insights`);assert.equal(insights.status(),200);
 await page.goto(base,{waitUntil:'networkidle'});
 await page.getByTestId('dataset-select').selectOption(dataset.id);
 await page.locator(`#run-select option[value="${run.id}"]`).waitFor({state:'attached'});
 await page.locator('#run-select').selectOption(run.id);
 await page.locator('.score-card [role="img"]').waitFor();
 await page.waitForTimeout(300);
 await page.screenshot({path:path.join(out,'workbench-first-screen.png')});
 await page.locator('[data-testid^="event-row-"]').first().click();
 await page.getByTestId('review-note').waitFor();
 await page.evaluate(()=>window.scrollTo(0,0));
 await page.screenshot({path:path.join(out,'workbench-desktop.png'),fullPage:true});
 await page.getByRole('button',{name:'数据诊断',exact:true}).click();
 await page.getByTestId('profile-table').waitFor();
 await page.screenshot({path:path.join(out,'data-diagnostics.png'),fullPage:true});
 await page.getByRole('button',{name:'模型对比',exact:true}).click();
 await page.getByTestId('comparison-table').waitFor();
 await page.screenshot({path:path.join(out,'model-comparison.png'),fullPage:true});
 await page.getByRole('button',{name:'分析工作台',exact:true}).click();
 await page.setViewportSize({width:390,height:844});
 await page.evaluate(()=>window.scrollTo(0,0));
 await page.screenshot({path:path.join(out,'workbench-mobile.png'),fullPage:true});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+2),false);
 assert.deepEqual(errors,[]);
 const result={base,datasetCount:datasets.items.length,runCount:runs.items.length,baselineVerified:Boolean(baseline),profile:profile.status(),insights:insights.status(),browserErrors:errors};
 await fs.writeFile(path.join(out,'verification-release.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}finally{await browser.close();}
