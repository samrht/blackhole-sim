// The accuracy limit in the 1.3 mm hot-flow image, headless on the discrete GPU; writes docs/diagnostics/2026-11-06-accuracy-limit/image/flow-<view>/.
//   npx vite (:5173), then:  node scripts/flow-error-run.mjs <sgra|m87>
import { chromium } from "playwright-core";
import { mkdirSync, writeFileSync } from "node:fs";

const [view = "m87"] = process.argv.slice(2);
const dir = `docs/diagnostics/2026-11-06-accuracy-limit/image/flow-${view}`;
mkdirSync(dir, { recursive: true });
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--force_high_performance_gpu"] });
const page = await browser.newPage({ viewport: { width: 700, height: 700 } });
page.on("console", (m) => { const t = m.text(); if (/^(reference|arm )/.test(t)) console.log(`  ${view}: ${t.slice(0, 200)}`); else if ((m.type() === "error" || m.type() === "warning") && !/404/.test(t)) console.log(`[${m.type()}]`, t.slice(0, 900)); });
page.on("pageerror", (e) => console.log("PAGEERROR", e.message.slice(0, 900)));
const t0 = Date.now();
await page.goto(`http://localhost:5173/?imgflow&view=${view}`);
await page.waitForFunction(() => window.__imgflow || window.__imgflowError, null, { timeout: 560000 });
const err = await page.evaluate(() => window.__imgflowError);
if (err) { console.log("FAILED:", err.slice(0, 1200)); await browser.close(); process.exit(1); }
const report = await page.evaluate(() => window.__imgflow.report);
const url = await page.evaluate(() => window.__imgflow.images.reference);
writeFileSync(`${dir}/reference.png`, Buffer.from(url.split(",")[1], "base64"));
writeFileSync(`${dir}/report.json`, JSON.stringify(report, (_, v) => (typeof v === "number" && !Number.isFinite(v) ? String(v) : v), 1));
await browser.close();
console.log(`${view} done in ${((Date.now() - t0) / 1000).toFixed(0)} s on ${report.adapter}`);
