// Does the accuracy limit reach the image? Headless on the discrete GPU; writes docs/diagnostics/2026-11-06-accuracy-limit/image/<view>/.
//   npx vite (:5173), then:  node scripts/image-error-run.mjs <view>     views: default edge schw zoom-pole zoom-ring
import { chromium } from "playwright-core";
import { mkdirSync, writeFileSync } from "node:fs";

const [view = "default"] = process.argv.slice(2);
const dir = `docs/diagnostics/2026-11-06-accuracy-limit/image/${view}`;
mkdirSync(dir, { recursive: true });
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--force_high_performance_gpu"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
page.on("console", (m) => { const t = m.text(); if (/^(reference|arm )/.test(t)) console.log(`  ${view}: ${t.slice(0, 200)}`); else if ((m.type() === "error" || m.type() === "warning") && !/404/.test(t)) console.log(`[${m.type()}]`, t.slice(0, 700)); });
page.on("pageerror", (e) => console.log("PAGEERROR", e.message.slice(0, 700)));
const t0 = Date.now();
await page.goto(`http://localhost:5173/?imgerr&view=${view}`);
await page.waitForFunction(() => window.__imgerr || window.__imgerrError, null, { timeout: 1500000 });
const err = await page.evaluate(() => window.__imgerrError);
if (err) { console.log("FAILED:", err.slice(0, 900)); await browser.close(); process.exit(1); }
const report = await page.evaluate(() => window.__imgerr.report);
const names = await page.evaluate(() => Object.keys(window.__imgerr.images));
for (const k of names) { const url = await page.evaluate((n) => window.__imgerr.images[n], k); writeFileSync(`${dir}/${k}.png`, Buffer.from(url.split(",")[1], "base64")); }
writeFileSync(`${dir}/report.json`, JSON.stringify(report, (_, v) => (typeof v === "number" && !Number.isFinite(v) ? String(v) : v), 1));
await browser.close();
console.log(`${view} done in ${((Date.now() - t0) / 1000).toFixed(0)} s on ${report.adapter}`);
