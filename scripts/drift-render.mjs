// Renders the ?drift diagnostic headless on the discrete GPU and saves every view plus the report.
//   npx vite (dev server on :5173), then:  node scripts/drift-render.mjs <tag> "<query>"
//   e.g. node scripts/drift-render.mjs disk "a=0.998&i=85"
// Output: docs/diagnostics/2026-11-06-conserved-drift/<tag>/<view>.png, report.json
import { chromium } from "playwright-core";
import { mkdirSync, writeFileSync } from "node:fs";

const [tag = "frame", query = ""] = process.argv.slice(2);
const dir = `docs/diagnostics/2026-11-06-conserved-drift/${tag}`;
mkdirSync(dir, { recursive: true });
const browser = await chromium.launch({
  channel: "chrome", headless: true,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--force_high_performance_gpu"],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
page.on("console", (m) => { if (m.type() !== "log") console.log(`[${m.type()}]`, m.text().slice(0, 800)); });
page.on("pageerror", (e) => console.log("PAGEERROR", e.message));
const t0 = Date.now();
await page.goto(`http://localhost:5173/?drift&${query}`);
await page.waitForFunction(() => window.__drift || window.__driftError, null, { timeout: 570000 });
const err = await page.evaluate(() => window.__driftError);
if (err) { console.log("FAILED:", err); await browser.close(); process.exit(1); }
const report = await page.evaluate(() => window.__drift);
writeFileSync(`${dir}/report.json`, JSON.stringify(report, (_, v) => (typeof v === "number" && !Number.isFinite(v) ? String(v) : v), 1));
for (const view of ["total", "pTotal", "relQ", "pRelQ", "absH", "pH", "pW", "absdQ", "wRes", "hUlps", "qUlps", "phase", "fate"]) {
  const url = await page.evaluate((v) => window.__driftPNG(v), view);
  writeFileSync(`${dir}/${view}.png`, Buffer.from(url.split(",")[1], "base64"));
}
console.log(`done in ${((Date.now() - t0) / 1000).toFixed(0)} s -> ${dir}; adapter ${report.adapter}`);
console.log(JSON.stringify({ audit: report.audit, measureParity: report.measureParity, h0Max: report.h0Max, budget: report.budget, atFloor: report.atFloor, cpuFateMismatch: report.cpu.fateMismatch, cpuBudget: report.cpu.budgetRows }));
await browser.close();
