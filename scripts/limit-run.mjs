// GPU arms of the accuracy-limit experiment, headless; merges them with the CPU arms into results/<config>[-<tag>].json.
//   npx vite (:5173), scripts/limit-cpu.ts done, then:  node scripts/limit-run.mjs <config> [igpu]
import { chromium } from "playwright-core";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

const [config = "edge-998", gpu = "rtx"] = process.argv.slice(2);
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist"];
if (gpu !== "igpu") args.push("--force_high_performance_gpu");
const browser = await chromium.launch({ channel: "chrome", headless: true, args });
const page = await browser.newPage();
page.on("console", (m) => { const t = m.text(); if (/^(mino|rk4|bs3) f32/.test(t)) console.log(`  ${config}: ${t}`); else if (m.type() === "error" && !/404/.test(t)) console.log("[error]", t.slice(0, 500)); });
page.on("pageerror", (e) => console.log("PAGEERROR", e.message));
const t0 = Date.now();
await page.goto(`http://localhost:5173/?limit&config=${config}`);
await page.waitForFunction(() => window.__limit || window.__limitError, null, { timeout: 570000 });
const err = await page.evaluate(() => window.__limitError);
if (err) { console.log("FAILED:", err); await browser.close(); process.exit(1); }
const gpuRes = await page.evaluate(() => window.__limit);
await browser.close();
const dir = "docs/diagnostics/2026-11-06-accuracy-limit";
const cpu = JSON.parse(readFileSync(`${dir}/data/${config}.json`, "utf8"));
mkdirSync(`${dir}/results`, { recursive: true });
const tag = gpu === "igpu" ? `${config}-igpu` : config;
writeFileSync(`${dir}/results/${tag}.json`, JSON.stringify({ config: cpu.config, rIn: cpu.rIn, rays: cpu.rays.length, noJ: cpu.noJ, adapter: gpuRes.adapter, arms: { ...cpu.arms, ...gpuRes.arms }, worst: gpuRes.worst, axisScan: gpuRes.axisScan }, null, 1));
console.log(`${tag} done in ${((Date.now() - t0) / 1000).toFixed(0)} s on ${gpuRes.adapter}`);
