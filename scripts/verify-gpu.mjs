// Headless-WebGPU verification: drives a real Chrome (via playwright-core + the system Chrome
// channel) against the dev/preview server to run the in-browser validation routes and capture a
// reference render. WebGPU can't run under jsdom, so these checks live here rather than in vitest.
//
//   npm run dev            # in one terminal (serves on :5173 by vite.config.ts, or pass BASE=)
//   node scripts/verify-gpu.mjs
//
// Exits non-zero if any check fails. A check also fails when the page logs a console warning or
// error (the favicon 404 excepted): WebGPU never throws on a broken shader -- Chrome reports a WGSL
// compile failure as a console WARNING, returns an invalid module, and the invalidity cascades into
// zeroed buffers, so without this a broken shader surfaced only as a strange downstream number.
import { chromium } from "playwright-core";
import { writeFileSync } from "node:fs";

const BASE = process.env.BASE || "http://localhost:5173";
const SHOT = process.env.SHOT || "render.png";

const browser = await chromium.launch({
  channel: "chrome", headless: true,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist"],
});
const page = await browser.newPage({ viewport: { width: 1000, height: 680 } });
// Page errors and console warnings/errors (e.g. "Error while parsing WGSL" and its "is invalid due
// to a previous error" cascade). The dev server has no favicon; its 404 is the one expected message.
const diags = [];
let onDiag = () => {};
const addDiag = (msg) => { diags.push(msg); onDiag(); };
page.on("pageerror", (e) => addDiag("PAGEERROR " + e.message));
page.on("console", (m) => {
  if (m.type() !== "warning" && m.type() !== "error") return;
  if (/\/favicon\.ico$/.test(m.location().url)) return;
  addDiag(m.text().split("\n").slice(0, 3).join(" / "));
});
/** Resolves on the next diagnostic, so a broken route fails fast instead of waiting out its
 *  timeout (up to 10 min for ?cachecheck). */
const nextDiag = () => new Promise((res) => { onDiag = res; });
/** Diagnostics logged since index `from`, as a printable suffix; empty when clean. */
function diagSince(from) {
  const d = diags.slice(from);
  if (!d.length) return "";
  return `\n        console: ${d.slice(0, 4).join(" | ").slice(0, 900)}${d.length > 4 ? ` (+${d.length - 4} more)` : ""}`;
}

let failed = false;
async function checkAny(path, accepts, timeout = 60000, show = 2000) {
  const d0 = diags.length;
  await page.goto(BASE + path, { waitUntil: "load", timeout: 20000 });
  await Promise.race([
    page.waitForFunction((a) => a.some((e) => document.body.innerText.includes(e)), accepts, { timeout }).catch(() => {}),
    diags.length > d0 ? Promise.resolve() : nextDiag(),
  ]);
  await page.waitForTimeout(250); // let a cascade of follow-on warnings land in the report
  const txt = (await page.innerText("body")).replace(/\s+/g, " ").trim();
  const extra = diagSince(d0);
  const ok = accepts.some((e) => txt.includes(e)) && !extra;
  console.log(`${ok ? "✓ PASS" : "✗ FAIL"}  ${path}\n        ${txt.slice(0, show)}${extra}`);
  if (!ok) failed = true;
}
const check = (path, expect) => checkAny(path, [expect], 25000, 300);

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
const dApp = diags.length;
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
const appDiag = diagSince(dApp);
console.log(`${cacheOk && !appDiag ? "✓ PASS" : "✗ FAIL"}  cache in the app: ${steps.join(", ")}${appDiag}`);
if (!cacheOk || appDiag) failed = true;
// Presets (spec 2026-10-01): each one must leave the app lit, back in `cached` mode, warning-free.
// Spin/inclination changes rebuild the cache; mass/accretion are shading-only (Review Focus 1).
const dPre = diags.length;
const presetIds = await page.evaluate(() => [...document.getElementById("preset").options].map((o) => o.value).filter((v) => v !== "custom" && v !== "default"));
const preSteps = [];
for (const id of presetIds) {
  await page.selectOption("#preset", id);
  await page.waitForTimeout(300); // let the new geometry key reach the scheduler (else "cached" is the old preset's)
  const cached = await waitMode("cached");
  await page.waitForTimeout(1500);
  const lit = await meanBrightness();
  preSteps.push(`${id} ${cached && lit > 2 ? "ok" : "FAILED"} (lit ${lit.toFixed(1)})`);
  if (!cached || !(lit > 2)) failed = true;
}
// Dragging the canvas tilts the camera: that must leave the preset for Custom (Review Focus 4).
await page.mouse.move(700, 300); await page.mouse.down(); await page.mouse.move(700, 360, { steps: 6 }); await page.mouse.up();
const afterDrag = await page.evaluate(() => document.getElementById("preset").value);
preSteps.push(`drag -> ${afterDrag}`);
if (afterDrag !== "custom") failed = true;
// "Default view" restores the opening view (spin 0.9, 72 deg, 1e8 M_sun, the 30,000 K disk).
await page.selectOption("#preset", "default");
await page.waitForTimeout(300);
const defOk = await page.evaluate(() => ["spinv", "inclv", "massv", "tpk"].map((k) => document.getElementById(k).textContent).join("|"));
const defPass = defOk === "0.900|72|1.00e+8|30,000 K" && await waitMode("cached");
preSteps.push(`default ${defPass ? "ok" : "FAILED"} (${defOk})`);
if (!defPass) failed = true;
// Mass is shading-only: nudging it while cached must re-shade with NO rebuild (mode never leaves
// `cached`) and leave the preset for Custom (Review Focus 1).
let leftCached = false;
await page.evaluate(() => { const m = document.getElementById("mass"); m.value = "8.5"; m.dispatchEvent(new Event("input", { bubbles: true })); });
for (let k = 0; k < 20; k++) { await page.waitForTimeout(75); if (!(await page.evaluate(() => document.getElementById("cmode").textContent)).startsWith("cached")) leftCached = true; }
const afterMass = await page.evaluate(() => document.getElementById("preset").value);
const massPass = !leftCached && afterMass === "custom";
preSteps.push(`mass nudge ${massPass ? "ok" : "FAILED"} (stayed cached ${!leftCached}, selector ${afterMass})`);
if (!massPass) failed = true;
const preDiag = diagSince(dPre);
if (preDiag) failed = true;
console.log(`${preSteps.every((s) => !s.includes("FAILED")) && afterDrag === "custom" && !preDiag ? "✓ PASS" : "✗ FAIL"}  presets: ${preSteps.join(", ")}${preDiag}`);

await browser.close();
process.exit(failed ? 1 : 0);
