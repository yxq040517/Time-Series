import { chromium } from '../frontend/node_modules/playwright/index.mjs';
import assert from 'node:assert/strict';
const browser=await chromium.launch({headless:true});const page=await browser.newPage({viewport:{width:390,height:844}});
await page.goto(process.env.CHRONOLENS_URL || 'http://127.0.0.1:8767',{waitUntil:'networkidle'});
const boxes=await page.evaluate(()=>Object.fromEntries(['.app-shell','.sidebar','.sidebar nav','.main-shell','.topbar'].map(selector=>{const node=document.querySelector(selector);const style=getComputedStyle(node);const rect=node.getBoundingClientRect();return [selector,{display:style.display,height:style.height,minHeight:style.minHeight,position:style.position,flex:style.flex,top:rect.top,bottom:rect.bottom,width:rect.width,overflow:style.overflow}]})));
console.log(JSON.stringify(boxes,null,2));
await browser.close();assert.ok(boxes['.sidebar'].bottom>=boxes['.sidebar nav'].bottom,'sidebar must contain navigation');

