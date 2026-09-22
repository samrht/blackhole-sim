// Browser checks for the adaptive render scale: resize while scaled down, a tiny odd viewport, and
// pause restoring full resolution. Exits non-zero on failure. Needs the dev server on :5173.
import { chromium } from "playwright-core";

const BASE = process.env.BASE || "http://localhost:5173";
const browser = await chromium.launch({ channel: "chrome", headless: true,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--force_high_performance_gpu"] });
const page = await browser.newPage({ viewport: { width: 1000, height: 680 } });
const diags = [];
// The page has no favicon; Chrome's 404 for it is not a render diagnostic.
page.on("console", (m) => { if ((m.type() === "warning" || m.type() === "error") && !/favicon\.ico$/.test(m.location().url)) diags.push(m.text()); });
page.on("pageerror", (e) => diags.push("PAGEERROR " + e.message));
let failed = false;
const check = (ok, msg) => { console.log(`${ok ? "✓ PASS" : "✗ FAIL"}  ${msg}`); if (!ok) failed = true; };

/** Mean RGB sum of the canvas region right of the panel (decoded in-page from a screenshot). */
async function meanLum() {
  const png = await page.screenshot({ type: "png" });
  return page.evaluate(async (b64) => {
    const img = new Image(); img.src = "data:image/png;base64," + b64; await img.decode();
    const cv = document.createElement("canvas"); cv.width = img.width; cv.height = img.height;
    const g = cv.getContext("2d"); g.drawImage(img, 0, 0);
    const x0 = Math.min(340, img.width - 1);
    const d = g.getImageData(x0, 0, img.width - x0, img.height).data;
    let s = 0; for (let i = 0; i < d.length; i += 4) s += d[i] + d[i + 1] + d[i + 2];
    return s / (d.length / 4);
  }, png.toString("base64"));
}
const rscale = () => page.textContent("#rscale");

// Pinned 50 %: resize twice (normal, then tiny and odd), frame must stay lit and warning-free.
await page.goto(BASE + "/?scale=0.5", { waitUntil: "load" });
await page.waitForTimeout(4000);
const ref = await meanLum();
check((await rscale()) === "50%" && ref > 20, `pinned 50%: readout ${await rscale()}, mean ${ref.toFixed(1)}`);
for (const [w, h] of [[700, 500], [301, 157]]) {
  await page.setViewportSize({ width: w, height: h });
  await page.waitForTimeout(3000);
  const m = await meanLum();
  check((await rscale()) === "50%" && m > 20, `resize to ${w}x${h} at 50%: mean ${m.toFixed(1)}`);
}
check(diags.length === 0, `no WebGPU warnings/errors after resizes${diags.length ? ": " + diags.join(" | ") : ""}`);

// Unpinned: headless is slow enough that the controller should drop below 100 %; pausing restores it.
await page.setViewportSize({ width: 1000, height: 680 });
await page.goto(BASE + "/", { waitUntil: "load" });
let dropped = false;
for (let i = 0; i < 30 && !dropped; i++) { await page.waitForTimeout(500); dropped = (await rscale()) !== "100%"; }
if (!dropped) console.log("• SKIP pause check: this GPU never needed to scale down at 1000x680");
else {
  // DOM click: at 680 px tall the fixed, non-scrolling panel puts the button below the fold.
  await page.$eval("#playpause", (b) => b.click());
  await page.waitForTimeout(1500);
  check((await rscale()) === "100%", `pause restores full resolution (readout ${await rscale()})`);
}
await browser.close();
process.exit(failed ? 1 : 0);
