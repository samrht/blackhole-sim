// Visual confirmation for the polar-axis fix: renders a = 0 with the sky and jet off at i = 72
// and i = 8 degrees, and checks the band around the pole column is no darker than a band beside
// it, by two measures: the void-class dark fraction (catches the tunnelling wedge) and the mean
// brightness ratio (catches the far-field axis-crossing streak, a ~50 % dimming). The band is
// x in [488, 512] EXCLUDING the two axis columns 499-500 (alpha in ~0.05-0.5 M): the exact axis
// column has xi ~ 0 and no centrifugal barrier, so it is clean on every build and cannot show the
// defect -- the streak lives on either side of it. The axis columns' own dark fraction is printed
// for information only.
// main.ts reads only the `steps` query parameter, so the sliders are driven through the DOM and
// each value is read back to prove it was applied.
//
//   npm run dev                    # in another terminal
//   node scripts/probe-axis.mjs    # writes axis-i72.png / axis-i8.png, exits non-zero on failure
import { chromium } from "playwright-core";

const BASE = process.env.BASE || "http://localhost:5173";
const browser = await chromium.launch({
  channel: "chrome", headless: true,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist"],
});
const page = await browser.newPage({ viewport: { width: 1000, height: 680 } });
const diags = [];
page.on("console", (m) => { if (m.type() === "warning" || m.type() === "error") diags.push(m.text()); });
page.on("pageerror", (e) => diags.push("PAGEERROR " + e.message));
await page.goto(BASE + "/", { waitUntil: "load", timeout: 20000 });
// The slider handlers are attached after the async WebGPU init, just before the render loop
// starts; the first frame writes a number into #spp, so wait for that before touching anything.
await page.waitForFunction(() => /^\d+$/.test(document.getElementById("spp").textContent), null, { timeout: 30000 });

async function setSlider(id, value) {
  // Read back the app's own readout (#spinv etc.), which only the input handler updates: proof
  // the handler ran, not just that the element accepted the value.
  const applied = await page.evaluate(([id, v]) => {
    const el = document.getElementById(id);
    el.value = v; el.dispatchEvent(new Event("input", { bubbles: true }));
    return document.getElementById(id + "v").textContent;
  }, [id, String(value)]);
  if (Number(applied) !== Number(value)) throw new Error(`slider #${id} did not take ${value} (got ${applied})`);
}

/** Mean dark-pixel fraction over the columns `xs` and rows [y0, y1). Decoded in-page from the PNG
 *  so no PNG lib is needed. "Dark" = RGB sum < 80: the shadow reads 0, the procedural void (what
 *  the tunnelled rays render as, stars aside) reads ~55 at the default exposure, and lit disk
 *  reads >= 300. */
async function darkFraction(png, xs, y0, y1) {
  return page.evaluate(async ([b64, xs, y0, y1]) => {
    const img = new Image(); img.src = "data:image/png;base64," + b64; await img.decode();
    const cv = document.createElement("canvas"); cv.width = img.width; cv.height = img.height;
    const g = cv.getContext("2d"); g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, cv.width, cv.height).data;
    let dark = 0;
    for (const x of xs) for (let y = y0; y < y1; y++) { const i = (y * cv.width + x) * 4; if (d[i] + d[i + 1] + d[i + 2] < 80) dark++; }
    return dark / Math.max(1, (y1 - y0) * xs.length);
  }, [png.toString("base64"), xs, y0, y1]);
}
/** Mean RGB sum over the columns `xs` and rows [y0, y1). The pre-cap streak was a ~50 % DIMMING of
 *  columns 496-503 (half the jittered samples escaped), not void-class pixels, so a dark-fraction
 *  count alone does not see it; the band/side brightness ratio does (~0.88 with the streak, ~0.99
 *  without) and is the second assertion below. */
async function meanLum(png, xs, y0, y1) {
  return page.evaluate(async ([b64, xs, y0, y1]) => {
    const img = new Image(); img.src = "data:image/png;base64," + b64; await img.decode();
    const cv = document.createElement("canvas"); cv.width = img.width; cv.height = img.height;
    const g = cv.getContext("2d"); g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, cv.width, cv.height).data;
    let sum = 0;
    for (const x of xs) for (let y = y0; y < y1; y++) { const i = (y * cv.width + x) * 4; sum += d[i] + d[i + 1] + d[i + 2]; }
    return sum / Math.max(1, (y1 - y0) * xs.length);
  }, [png.toString("base64"), xs, y0, y1]);
}
const range = (lo, hi, skip = []) => Array.from({ length: hi - lo + 1 }, (_, k) => lo + k).filter((x) => !skip.includes(x));
/** Top edge of the dark run containing the frame centre around column x (the shadow's upper edge).
 *  Uses the median luminance over x-2..x+2 per row, so an isolated lit pixel on the axis column
 *  inside the shadow (a near-axis residual, see README limitation (a)) does not stop the walk. */
async function shadowTop(png, x) {
  return page.evaluate(async ([b64, x]) => {
    const img = new Image(); img.src = "data:image/png;base64," + b64; await img.decode();
    const cv = document.createElement("canvas"); cv.width = img.width; cv.height = img.height;
    const g = cv.getContext("2d"); g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, cv.width, cv.height).data;
    const lum = (y) => {
      const v = [];
      for (let dx = -2; dx <= 2; dx++) { const i = (y * cv.width + x + dx) * 4; v.push(d[i] + d[i + 1] + d[i + 2]); }
      return v.sort((p, q) => p - q)[2];
    };
    let top = cv.height >> 1; while (top > 0 && lum(top) < 20) top--;
    return top;
  }, [png.toString("base64"), x]);
}

let failed = false;
for (const incl of [72, 8]) {
  await setSlider("spin", 0); await setSlider("sky", 0); await setSlider("jet", 0); await setSlider("incl", incl);
  await page.waitForTimeout(6000); // EMA at ~15 fps converges well inside this
  const png = await page.screenshot({ type: "png" });
  const file = `axis-i${incl}.png`;
  (await import("node:fs")).writeFileSync(file, png);
  const cx = 500, top = await shadowTop(png, cx);
  const y0 = 0, y1 = Math.max(1, top - 4);
  const band = await darkFraction(png, range(488, 512, [499, 500]), y0, y1), side = await darkFraction(png, range(548, 572), y0, y1);
  const axisCols = await darkFraction(png, [499, 500], y0, y1); // informational: xi ~ 0, no barrier
  const bandCols = range(488, 512, [499, 500]), sideCols = range(548, 572);
  const lumRatio = (await meanLum(png, bandCols, y0, y1)) / Math.max(1, await meanLum(png, sideCols, y0, y1));
  // Before the fix the band beside the axis is a solid black wedge above the shadow at i = 8 and
  // carries a dark band at i = 72; the side band is lit disk. Allow 10% for the ISCO edge and AA
  // jitter.
  // Two assertions. The dark-fraction difference catches the tunnelling wedge (main: 0.927 vs 0.498
  // at i = 8) but NOT the far-field axis-crossing streak, which was a ~50 % dimming: it passed on
  // the pre-cap shader 4a98a7f (0.162 vs 0.157). The band/side brightness ratio does discriminate
  // -- measured main 0.203, pre-cap 4a98a7f 0.875, post-cap 0.993 (i = 8; i = 72 reads 0.44 /
  // 0.987 / 0.987) -- so 0.95 sits between the two measured branch states with margin on both
  // sides: a discrimination threshold between known states, not a tuned one.
  const ok = top > 10 && band - side < 0.10 && lumRatio >= 0.95;
  console.log(`${ok ? "✓ PASS" : "✗ FAIL"}  i=${incl}°  shadowTop=${top}px  darkFrac band[488-512 minus 499-500]=${band.toFixed(3)} side[548-572]=${side.toFixed(3)} axisCols[499-500]=${axisCols.toFixed(3)} band/side brightness=${lumRatio.toFixed(3)} (>= 0.95)  -> ${file}`);
  if (!ok) failed = true;
}
if (diags.length) console.log("console diagnostics:", diags.join(" | "));
await browser.close();
process.exit(failed ? 1 : 0);
