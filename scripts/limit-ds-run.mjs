// The double-single arms of the accuracy-limit experiment, headless; adds them to results/<config>[-igpu].json.
//   npx vite (:5173), then:  node scripts/limit-ds-run.mjs <config> [igpu]
import { chromium } from "playwright-core";
import { readFileSync, writeFileSync } from "node:fs";

const [config = "edge-998", gpu = "rtx"] = process.argv.slice(2);
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist"];
if (gpu !== "igpu") args.push("--force_high_performance_gpu");
const browser = await chromium.launch({ channel: "chrome", headless: true, args });
const page = await browser.newPage();
page.on("console", (m) => { const t = m.text(); if (/^(ds |cost |double-single)/.test(t)) console.log(`  ${config}: ${t}`); else if (m.type() === "error" && !/404/.test(t)) console.log("[error]", t.slice(0, 700)); });
page.on("pageerror", (e) => console.log("PAGEERROR", e.message.slice(0, 700)));
const t0 = Date.now();
await page.goto(`http://localhost:5173/?limitds&config=${config}`);
await page.waitForFunction(() => window.__limitDs || window.__limitDsError, null, { timeout: 570000 });
const err = await page.evaluate(() => window.__limitDsError);
if (err) { console.log("FAILED:", err.slice(0, 900)); await browser.close(); process.exit(1); }
const res = await page.evaluate(() => window.__limitDs);
await browser.close();
const file = `docs/diagnostics/2026-11-06-accuracy-limit/results/${gpu === "igpu" ? `${config}-igpu` : config}.json`;
const r = JSON.parse(readFileSync(file, "utf8"));
Object.assign(r.arms, res.arms); r.dsProbe = res.probe; r.dsCost = res.cost;
writeFileSync(file, JSON.stringify(r, null, 1));
console.log(`${config} ${gpu} done in ${((Date.now() - t0) / 1000).toFixed(0)} s on ${res.adapter}`);
