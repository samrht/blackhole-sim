// Headless-WebGPU verification: drives a real Chrome (via playwright-core + the system Chrome
// channel) against the dev/preview server to run the in-browser validation routes and capture a
// reference render. WebGPU can't run under jsdom, so these checks live here rather than in vitest.
//
//   npm run dev            # in one terminal (serves on :5173 by vite.config.ts, or pass BASE=)
//   node scripts/verify-gpu.mjs
//
// Exits non-zero if either the CPU<->GPU parity route or the Schwarzschild shadow route fails.
import { chromium } from "playwright-core";
import { writeFileSync } from "node:fs";

const BASE = process.env.BASE || "http://localhost:5173";
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
  console.log(`${ok ? "✓ PASS" : "✗ FAIL"}  ${path}\n        ${txt.slice(0, 300)}`);
  if (!ok) failed = true;
}

async function checkAny(path, accepts, timeout = 60000) {
  await page.goto(BASE + path, { waitUntil: "load", timeout: 20000 });
  await page.waitForFunction((a) => a.some((e) => document.body.innerText.includes(e)), accepts, { timeout }).catch(() => {});
  const txt = (await page.innerText("body")).replace(/\s+/g, " ").trim();
  const ok = accepts.some((e) => txt.includes(e));
  console.log(`${ok ? "✓ PASS" : "✗ FAIL"}  ${path}\n        ${txt.slice(0, 2000)}`);
  if (!ok) failed = true;
}

const ONLY_APP = process.env.ONLY_APP === "1";
if (!ONLY_APP) {
await check("/?parity", "PARITY PASS");
await check("/?shadow", "SHADOW PASS");
if (process.env.RECORD_GOLDEN === "1") {
  await page.goto(BASE + "/?golden&record", { waitUntil: "load", timeout: 20000 });
  await page.waitForFunction(() => document.body.innerText.includes("GOLDEN RECORD"), null, { timeout: 60000 });
  writeFileSync("src/test/golden.json", (await page.innerText("#json")) + "\n");
  console.log("• recorded src/test/golden.json");
}
await checkAny("/?golden", ["GOLDEN PASS", "GOLDEN SKIP"]);
await checkAny("/?cachecheck", ["CACHECHECK PASS"], 600000);

}

// Capture a reference render of the interactive view.
await page.goto(BASE + "/", { waitUntil: "load", timeout: 20000 });
await page.waitForTimeout(4500); // let progressive accumulation converge
// 30 s: a converged headless frame takes ~15 s after the monitored-step branch (measured 14.8 s).
await page.screenshot({ path: SHOT, timeout: 30000 });
console.log(`• saved ${SHOT}`);

// Sky panorama should be fetched and served (200) during the interactive render.
const skyResp = await page.request.get(BASE + "/sky/milkyway-4k.jpg");
const skyOk = skyResp.ok();
console.log(`${skyOk ? "✓ PASS" : "✗ FAIL"}  /sky/milkyway-4k.jpg  (${skyResp.status()})`);
if (!skyOk) failed = true;

// Geodesic cache in the live app. Every resize reallocates the cache, so after one the mode must
// drop to `live` and come back to `cached` -- also when the size is unchanged (a same-size resize
// once left the cache cleared but still "valid": a black cached view). The mode is polled every
// 16 ms, and the cached view must not be black (mean brightness of a screenshot of the disk, right
// of the control panel, decoded in-page).
await page.goto(BASE + "/", { waitUntil: "load", timeout: 20000 });
const waitMode = (prefix, timeout = 120000) => page.waitForFunction(
  (p) => (document.getElementById("cmode")?.textContent || "").startsWith(p), prefix, { timeout, polling: 16 }).then(() => true, () => false);
const meanBrightness = async () => {
  const b64 = (await page.screenshot({ clip: { x: 440, y: 120, width: 360, height: 300 } })).toString("base64");
  return page.evaluate(async (data) => {
    const img = new Image(); img.src = "data:image/png;base64," + data; await img.decode();
    const c = document.createElement("canvas"); c.width = img.width; c.height = img.height;
    const g = c.getContext("2d"); g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height).data;
    let sum = 0; for (let k = 0; k < d.length; k += 4) sum += d[k] + d[k + 1] + d[k + 2];
    return sum / (d.length / 4) / 3;
  }, b64);
};
const steps = [];
const step = (name, ok) => { steps.push(`${name} ${ok ? "ok" : "FAILED"}`); return ok; };
let cacheOk = step("cached", await waitMode("cached"));
const lit0 = await meanBrightness();
cacheOk = step(`lit ${lit0.toFixed(1)}`, lit0 > 5) && cacheOk;
await page.setViewportSize({ width: 820, height: 560 });
cacheOk = step("resize->live", await waitMode("live", 5000)) && step("->cached", await waitMode("cached")) && cacheOk;
await page.evaluate(() => dispatchEvent(new Event("resize"))); // same size: buffers reallocated
cacheOk = step("same-size resize->live", await waitMode("live", 5000)) && step("->cached", await waitMode("cached")) && cacheOk;
await page.waitForTimeout(1500); // let the EMA settle onto the rebuilt cache
const lit1 = await meanBrightness();
cacheOk = step(`lit ${lit1.toFixed(1)}`, lit1 > 5) && cacheOk;
console.log(`${cacheOk && !errors.length ? "✓ PASS" : "✗ FAIL"}  cache in the app: ${steps.join(", ")}`);
if (!cacheOk || errors.length) failed = true;

if (errors.length) console.log("console/page errors:", errors.join(" | "));
await browser.close();
process.exit(failed ? 1 : 0);
