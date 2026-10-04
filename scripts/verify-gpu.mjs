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
await checkAny("/?hotflow", ["HOTFLOW PASS"], 300000);

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
// Screenshot export (2026-10-03): pause, click Save PNG, catch the download. The PNG must be the canvas's
// full internal size and show what the canvas shows (mean brightness right of the panel within 10 % of the screen).
{
  const dShot = diags.length;
  await page.click("#playpause"); await page.waitForTimeout(1500);
  const [dl] = await Promise.all([page.waitForEvent("download", { timeout: 20000 }), page.click("#saveshot")]);
  const name = dl.suggestedFilename();
  const png = (await import("node:fs")).readFileSync(await dl.path()).toString("base64");
  // Compare the region right of the control panel (the panel is drawn over the canvas, not into it).
  const clip = await page.evaluate(() => { const p = document.getElementById("panel").getBoundingClientRect();
    const c = document.querySelector("canvas").getBoundingClientRect();
    return { x: Math.ceil(p.right) + 10, y: c.top + 10, width: Math.floor(c.right - p.right) - 20, height: c.height - 20 }; });
  const screenB64 = (await page.screenshot({ clip })).toString("base64");
  const r = await page.evaluate(async ([a, b, cl]) => {
    const load = async (data) => { const img = new Image(); img.src = "data:image/png;base64," + data; await img.decode(); return img; };
    const mean = (img, sx, sy, sw, sh) => { const c = document.createElement("canvas"); c.width = sw; c.height = sh;
      const g = c.getContext("2d"); g.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
      const d = g.getImageData(0, 0, sw, sh).data; let s = 0;
      for (let k = 0; k < d.length; k += 4) s += d[k] + d[k + 1] + d[k + 2];
      return s / (d.length / 4) / 3; };
    const cv = document.querySelector("canvas"), k = cv.width / cv.clientWidth; // CSS px -> canvas px
    const file = await load(a), shot = await load(b);
    return { w: file.width, h: file.height, cw: cv.width, ch: cv.height,
      fileMean: mean(file, cl.x * k, cl.y * k, cl.width * k, cl.height * k), screenMean: mean(shot, 0, 0, shot.width, shot.height) };
  }, [png, screenB64, clip]);
  await page.click("#playpause");
  const sizeOk = r.w === r.cw && r.h === r.ch;
  const litOk = r.fileMean > 5 && Math.abs(r.fileMean / r.screenMean - 1) < 0.1;
  const nameOk = /^blackhole-[a-z0-9]+-a-?\d\.\d\d-i\d+-\d{4}-\d\d-\d\dT\d\d-\d\d-\d\d\.png$/.test(name);
  const shotDiag = diagSince(dShot);
  const ok = sizeOk && litOk && nameOk && !shotDiag;
  console.log(`${ok ? "✓ PASS" : "✗ FAIL"}  screenshot export: ${name} ${r.w}x${r.h} (canvas ${r.cw}x${r.ch}), mean right of the panel ${r.fileMean.toFixed(1)} vs screen ${r.screenMean.toFixed(1)}${shotDiag}`);
  if (!ok) failed = true;
}
// Clip export (2026-10-03): record ~3 s, stop, catch the download. The video must have the canvas's size,
// last about 3 s, and its middle frame must not be black.
{
  const dClip = diags.length;
  const recSupported = await page.evaluate(() => !document.getElementById("record").disabled);
  await page.click("#record"); await page.waitForTimeout(3000);
  const [dl] = await Promise.all([page.waitForEvent("download", { timeout: 20000 }), page.click("#record")]);
  const name = dl.suggestedFilename();
  const vid = (await import("node:fs")).readFileSync(await dl.path()).toString("base64");
  const v = await page.evaluate(async ([b64, type]) => {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const el = document.createElement("video"); el.muted = true; el.src = URL.createObjectURL(new Blob([bytes], { type }));
    await new Promise((res, rej) => { el.onloadedmetadata = res; el.onerror = () => rej(new Error("video did not load")); });
    if (!Number.isFinite(el.duration)) { el.currentTime = 1e6; await new Promise((res) => (el.ondurationchange = res)); } // WebM: no duration header
    const duration = el.duration;
    el.currentTime = duration / 2; await new Promise((res) => (el.onseeked = res));
    const c = document.createElement("canvas"); c.width = el.videoWidth; c.height = el.videoHeight;
    const g = c.getContext("2d"); g.drawImage(el, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height).data; let s = 0;
    for (let k = 0; k < d.length; k += 4) s += d[k] + d[k + 1] + d[k + 2];
    const cv = document.querySelector("canvas");
    return { w: el.videoWidth, h: el.videoHeight, cw: cv.width, ch: cv.height, duration, mean: s / (d.length / 4) / 3, bytes: bytes.length };
  }, [vid, name.endsWith(".mp4") ? "video/mp4" : "video/webm"]);
  const ok = recSupported && v.w === v.cw && v.h === v.ch && v.duration > 2 && v.duration < 5 && v.mean > 5
    && /^blackhole-[a-z0-9]+-a-?\d\.\d\d-i\d+-\d{4}-\d\d-\d\dT\d\d-\d\d-\d\d\.(mp4|webm)$/.test(name) && !diagSince(dClip);
  console.log(`${ok ? "✓ PASS" : "✗ FAIL"}  clip export: ${name} ${v.w}x${v.h} (canvas ${v.cw}x${v.ch}), ${v.duration.toFixed(2)} s, ${(v.bytes / 1e6).toFixed(1)} MB, middle-frame mean ${v.mean.toFixed(1)}${diagSince(dClip)}`);
  if (!ok) failed = true;
}
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
// Light delay is shading-only: toggling it while cached must re-shade without a rebuild (Review Focus 1).
await page.selectOption("#preset", "default");
await waitMode("cached");
await page.waitForTimeout(800);
let delayLeft = false;
await page.evaluate(() => { const c = document.getElementById("ldelay"); c.checked = !c.checked; c.dispatchEvent(new Event("change", { bubbles: true })); });
for (let k = 0; k < 20; k++) { await page.waitForTimeout(75); if (!(await page.evaluate(() => document.getElementById("cmode").textContent)).startsWith("cached")) delayLeft = true; }
await page.evaluate(() => { const c = document.getElementById("ldelay"); c.checked = !c.checked; c.dispatchEvent(new Event("change", { bubbles: true })); });
preSteps.push(`delay toggle ${!delayLeft ? "ok" : "FAILED"} (stayed cached ${!delayLeft})`);
if (delayLeft) failed = true;
// Turbulence (spec 2026-10-03): the flicker slider reads in percent with the observed 2 % default, flares
// start off, and nudging the flicker is shading-only (stays cached).
const panel = await page.evaluate(() => ({
  turb: document.getElementById("turb").value, turbMax: document.getElementById("turb").max,
  turbv: document.getElementById("turbv").textContent, flare: document.getElementById("flare").value,
  label: document.getElementById("turb").closest(".ctrl").textContent,
}));
const panelOk = panel.turb === "2" && panel.turbMax === "10" && panel.turbv === "2.0" && panel.flare === "0"
  && /flicker/i.test(panel.label);
preSteps.push(`flicker panel ${panelOk ? "ok" : "FAILED"} (${JSON.stringify(panel)})`);
if (!panelOk) failed = true;
let turbLeft = false;
await page.evaluate(() => { const t = document.getElementById("turb"); t.value = "5"; t.dispatchEvent(new Event("input", { bubbles: true })); });
for (let k = 0; k < 20; k++) { await page.waitForTimeout(75); if (!(await page.evaluate(() => document.getElementById("cmode").textContent)).startsWith("cached")) turbLeft = true; }
await page.evaluate(() => { const t = document.getElementById("turb"); t.value = "2"; t.dispatchEvent(new Event("input", { bubbles: true })); });
preSteps.push(`flicker nudge ${!turbLeft ? "ok" : "FAILED"} (stayed cached ${!turbLeft})`);
if (turbLeft) failed = true;
// Motion is playback speed: at 0 the scene freezes (Review Focus 2).
await page.evaluate(() => { const m = document.getElementById("ts"); m.value = "0"; m.dispatchEvent(new Event("input", { bubbles: true })); });
await page.waitForTimeout(1500);
const f0 = await meanBrightness(); await page.waitForTimeout(1500); const f1 = await meanBrightness();
const frozen = Math.abs(f1 - f0) < 1;
preSteps.push(`motion 0 ${frozen ? "ok" : "FAILED"} (${f0.toFixed(2)} -> ${f1.toFixed(2)})`);
if (!frozen) failed = true;
await page.evaluate(() => { const m = document.getElementById("ts"); m.value = "1"; m.dispatchEvent(new Event("input", { bubbles: true })); });
const preDiag = diagSince(dPre);
if (preDiag) failed = true;
console.log(`${preSteps.every((s) => !s.includes("FAILED")) && afterDrag === "custom" && !preDiag ? "✓ PASS" : "✗ FAIL"}  presets: ${preSteps.join(", ")}${preDiag}`);

// Every control must be reachable in a laptop-sized window, with the longest preset caption showing: the panel
// scrolls inside itself (it is fixed and the page does not scroll). Was clipped at the window's edge.
await page.setViewportSize({ width: 1280, height: 720 });
await page.selectOption("#preset", "m87");
await page.waitForTimeout(300);
const reach = await page.evaluate(() => {
  const p = document.getElementById("panel"), ctrls = [...p.querySelectorAll("input, select")].filter((c) => c.offsetParent);
  p.scrollTop = 1e6; // the lowest control (by layout, not document order) once scrolled to the end
  const r = { panelBottom: p.getBoundingClientRect().bottom, lastBottom: Math.max(...ctrls.map((c) => c.getBoundingClientRect().bottom)), viewport: innerHeight };
  p.scrollTop = 0;
  return r;
});
const reachOk = reach.panelBottom <= reach.viewport && reach.lastBottom <= reach.viewport;
console.log(`${reachOk ? "✓ PASS" : "✗ FAIL"}  panel reachable at 1280x720 with a caption: panel bottom ${reach.panelBottom.toFixed(0)}, last control ${reach.lastBottom.toFixed(0)}, window ${reach.viewport}`);
if (!reachOk) failed = true;

// Shareable links (2026-10-03): a #hash opens that view; changing a control rewrites the hash; Copy link copies it.
{
  const dShare = diags.length, sSteps = [];
  const sstep = (name, ok) => { sSteps.push(`${name} ${ok ? "ok" : "FAILED"}`); return ok; };
  await page.context().grantPermissions(["clipboard-read", "clipboard-write"], { origin: BASE });
  await page.goto(BASE + "/#p=m87&x=0.5&fv=1.4&play=0", { waitUntil: "load", timeout: 20000 });
  await page.waitForTimeout(1500);
  const opened = await page.evaluate(() => ({ p: document.getElementById("preset").value, x: document.getElementById("exp").value,
    fv: document.getElementById("fv").value, i: document.getElementById("incl").value, play: document.getElementById("playpause").textContent }));
  let ok = sstep(`opened ${JSON.stringify(opened)}`, opened.p === "m87" && opened.x === "0.5" && opened.fv === "1.4" && opened.i === "17" && opened.play === "Play");
  await page.evaluate(() => { const s = document.getElementById("spin"); s.value = "0.5"; s.dispatchEvent(new Event("input", { bubbles: true })); });
  await page.waitForTimeout(800);
  const hash = await page.evaluate(() => location.hash);
  ok = sstep(`hash ${hash}`, (/^#p=custom&/.test(hash) && /a=0\.5(&|$)/.test(hash) && /i=17/.test(hash) && /x=0\.5/.test(hash) && /play=0/.test(hash))) && ok;
  await page.click("#copylink");
  const copied = await page.waitForFunction(() => document.getElementById("copylink").textContent === "Copied", null, { timeout: 5000 }).then(() => true, () => false);
  const clip = await page.evaluate(() => navigator.clipboard.readText()).catch(() => "");
  ok = sstep("copy", copied && clip === BASE + "/" + hash) && ok;
  // Back to a clean page: hashchange to the default view re-applies it.
  await page.evaluate(() => { location.hash = ""; });
  await page.waitForTimeout(300);
  const shareDiag = diagSince(dShare);
  console.log(`${ok && !shareDiag ? "✓ PASS" : "✗ FAIL"}  shareable links: ${sSteps.join(", ")}${shareDiag}`);
  if (!ok || shareDiag) failed = true;
}

// 1.3 mm view (spec 2026-10-04): a link to Sgr A* at 1.3 mm opens the hot flow, reaches `cached`, and shows a bright
// ring around a darker centre (brightness profile through the canvas centre; the panel overlays the left edge).
{
  const dMm = diags.length;
  await page.goto(BASE + "/#p=sgra&b=mm", { waitUntil: "load", timeout: 20000 });
  const band = await page.evaluate(() => document.getElementById("band").value);
  const cached = await waitMode("cached");
  await page.waitForTimeout(1500);
  const vp = page.viewportSize(), cx = Math.round(vp.width / 2), cy = Math.round(vp.height / 2), R = 180;
  const b64 = (await page.screenshot({ clip: { x: cx - R, y: cy - R, width: 2 * R, height: 2 * R } })).toString("base64");
  const prof = await page.evaluate(async ([data, R]) => {
    const img = new Image(); img.src = "data:image/png;base64," + data; await img.decode();
    const c = document.createElement("canvas"); c.width = img.width; c.height = img.height;
    const g = c.getContext("2d"); g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height).data, s = c.width / (2 * R);
    let max = 0, centre = 0, n = 0;
    for (let y = 0; y < c.height; y++) for (let x = 0; x < c.width; x++) {
      const k = (y * c.width + x) * 4, v = (d[k] + d[k + 1] + d[k + 2]) / 3, rr = Math.hypot(x / s - R, y / s - R);
      max = Math.max(max, v); if (rr < 20) { centre += v; n++; }
    }
    return { max, centre: centre / n };
  }, [b64, R]);
  const mmOk = band === "mm" && cached && prof.max > 120 && prof.centre < 0.6 * prof.max;
  const mmDiag = diagSince(dMm);
  console.log(`${mmOk && !mmDiag ? "✓ PASS" : "✗ FAIL"}  1.3 mm view: band ${band}, cached ${cached}, ring max ${prof.max.toFixed(0)}, centre ${prof.centre.toFixed(0)}${mmDiag}`);
  if (!mmOk || mmDiag) failed = true;
}

await browser.close();
process.exit(failed ? 1 : 0);
