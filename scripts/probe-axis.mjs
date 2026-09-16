// Visual confirmation for the polar-axis fix: renders a = 0 with the sky and jet off at i = 72
// and i = 8 degrees, and checks the column through the pole is no darker than a column beside it.
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

/** Dark-pixel fraction of column x over rows [y0, y1). Decoded in-page from the PNG so no PNG lib is needed.
 *  "Dark" = RGB sum < 80: the shadow reads 0, the procedural void (what the tunnelled rays render
 *  as, stars aside) reads ~55 at the default exposure, and lit disk reads >= 300. */
async function darkFraction(png, x, y0, y1) {
  return page.evaluate(async ([b64, x, y0, y1]) => {
    const img = new Image(); img.src = "data:image/png;base64," + b64; await img.decode();
    const cv = document.createElement("canvas"); cv.width = img.width; cv.height = img.height;
    const g = cv.getContext("2d"); g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, cv.width, cv.height).data;
    let dark = 0;
    for (let y = y0; y < y1; y++) { const i = (y * cv.width + x) * 4; if (d[i] + d[i + 1] + d[i + 2] < 80) dark++; }
    return dark / Math.max(1, y1 - y0);
  }, [png.toString("base64"), x, y0, y1]);
}
/** Top edge of the dark run containing the frame centre on column x (the shadow's upper edge). */
async function shadowTop(png, x) {
  return page.evaluate(async ([b64, x]) => {
    const img = new Image(); img.src = "data:image/png;base64," + b64; await img.decode();
    const cv = document.createElement("canvas"); cv.width = img.width; cv.height = img.height;
    const g = cv.getContext("2d"); g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, cv.width, cv.height).data;
    const lum = (y) => { const i = (y * cv.width + x) * 4; return d[i] + d[i + 1] + d[i + 2]; };
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
  const axis = await darkFraction(png, cx, y0, y1), side = await darkFraction(png, cx + 60, y0, y1);
  // Before the fix the axis column is ~100% dark above the shadow at i = 8 and carries a solid
  // band at i = 72; the side column is lit disk. Allow 10% for the ISCO edge and AA jitter.
  const ok = top > 10 && axis - side < 0.10;
  console.log(`${ok ? "✓ PASS" : "✗ FAIL"}  i=${incl}°  shadowTop=${top}px  darkFrac axis=${axis.toFixed(3)} side=${side.toFixed(3)}  -> ${file}`);
  if (!ok) failed = true;
}
if (diags.length) console.log("console diagnostics:", diags.join(" | "));
await browser.close();
process.exit(failed ? 1 : 0);
