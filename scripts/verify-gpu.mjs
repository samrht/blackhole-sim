// Headless-WebGPU verification: drives a real Chrome (via playwright-core + the system Chrome
// channel) against the dev/preview server to run the in-browser validation routes and capture a
// reference render. WebGPU can't run under jsdom, so these checks live here rather than in vitest.
//
//   npm run dev            # in one terminal (serves on :4178 by config below, or pass BASE=)
//   node scripts/verify-gpu.mjs
//
// Exits non-zero if either the CPU<->GPU parity route or the Schwarzschild shadow route fails.
import { chromium } from "playwright-core";

const BASE = process.env.BASE || "http://localhost:4178";
const SHOT = process.env.SHOT || "render.png";

const browser = await chromium.launch({
  channel: "chrome", headless: true,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist"],
});
const page = await browser.newPage({ viewport: { width: 1000, height: 680 } });
const errors = [];
page.on("pageerror", (e) => errors.push("PAGEERROR " + e.message));

let failed = false;
async function check(path, expect) {
  await page.goto(BASE + path, { waitUntil: "load", timeout: 20000 });
  await page.waitForFunction((e) => document.body.innerText.includes(e), expect, { timeout: 25000 }).catch(() => {});
  const txt = (await page.innerText("body")).replace(/\s+/g, " ").trim();
  const ok = txt.includes(expect);
  console.log(`${ok ? "✓ PASS" : "✗ FAIL"}  ${path}\n        ${txt.slice(0, 150)}`);
  if (!ok) failed = true;
}

await check("/?parity", "PARITY PASS");
await check("/?shadow", "SHADOW PASS");

// Capture a reference render of the interactive view.
await page.goto(BASE + "/", { waitUntil: "load", timeout: 20000 });
await page.waitForTimeout(4500); // let progressive accumulation converge
await page.screenshot({ path: SHOT, timeout: 15000 });
console.log(`• saved ${SHOT}`);

// Sky panorama should be fetched and served (200) during the interactive render.
const skyResp = await page.request.get(BASE + "/sky/milkyway-4k.jpg");
const skyOk = skyResp.ok();
console.log(`${skyOk ? "✓ PASS" : "✗ FAIL"}  /sky/milkyway-4k.jpg  (${skyResp.status()})`);
if (!skyOk) failed = true;

if (errors.length) console.log("console/page errors:", errors.join(" | "));
await browser.close();
process.exit(failed ? 1 : 0);
