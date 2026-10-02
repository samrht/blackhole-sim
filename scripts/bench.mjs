// GPU benchmark: drives the real Renderer (loaded through the dev server) in headless Chrome and
// reports the WebGPU adapter plus the median GPU time per frame of the default animated scene at
// full internal resolution. A report, not a gate -- timings vary across machines.
//
//   npm run dev        # in another terminal (:5173)
//   npm run bench
//
// --force_high_performance_gpu makes Chrome hand WebGPU the discrete GPU on hybrid laptops; without
// it Chrome returned the Intel iGPU for every powerPreference on the dev machine (2026-09-23).
// The sky panorama is not loaded (skyStrength 0), matching the procedural fallback.
import { chromium } from "playwright-core";

const BASE = process.env.BASE || "http://localhost:5173";
const browser = await chromium.launch({
  channel: "chrome", headless: true,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--force_high_performance_gpu"],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
const diags = [];
page.on("console", (m) => { if (m.type() === "warning" || m.type() === "error") diags.push(m.text()); });
page.on("pageerror", (e) => diags.push("PAGEERROR " + e.message));
await page.goto(BASE + "/?bench", { waitUntil: "load", timeout: 20000 });

const out = await page.evaluate(async () => {
  const ad = await navigator.gpu.requestAdapter();
  const info = (ad && ad.info) || {};
  const { Renderer } = await import("/src/render/gpu.ts");
  const { buildTempLUT, buildVisibleLUT, lumNormFor } = await import("/src/physics/lookups.ts");
  const { iscoRadius } = await import("/src/physics/orbits.ts");
  async function bench(w, h, frames = 8) {
    document.body.innerHTML = "";
    const c = document.createElement("canvas");
    c.style.cssText = `width:${w}px;height:${h}px;display:block`;
    document.body.appendChild(c);
    const r = new Renderer(); await r.init(c);
    const a = 0.9, rIn = iscoRadius(a, true);
    const { jetUniforms, ETA_DEFAULT } = await import("/src/physics/synchrotron.ts");
    const { CUSTOM_DEFAULT } = await import("/src/physics/presets.ts");
    const J = jetUniforms(CUSTOM_DEFAULT.massSun, a, CUSTOM_DEFAULT.lambda, ETA_DEFAULT, 60, 2);
    r.uploadLUTs(buildTempLUT(a, true, rIn, 40, 512), buildVisibleLUT()); r.rebind();
    r.uploadHotSpots(new Float32Array([8, 0, 1.2, 1.8, 12, 2.1, 1.6, 1.2, 16, 4.3, 2.0, 0.9])); r.rebind();
    const u = (f) => ({
      resW: r.width, resH: r.height, outW: r.displayW ?? r.width, outH: r.displayH ?? r.height,
      a, incl: (72 * Math.PI) / 180, rObs: 1000, fovScale: 14, rIn, rOut: 40, Tpeak: 3e4, lumNorm: lumNormFor(3e4), lightDelay: 1, exposure: 1.6,
      time: f, frame: f, reset: f === 0 ? 1 : 0, maxSteps: 4800, blend: f === 0 ? 1 : 0.15, timeScale: 1,
      turbAmp: 0.6, breatheAmp: 0, nSpots: 3, jetStrength: 1, jetGamma: 2, jetLength: 60, jetKnots: 0.7,
      skyStrength: 0, jetB0: J.jetB0, jetQ0: J.jetQ0, rgCm: J.rgCm,
    });
    r.frame(u(0)); await r.device.queue.onSubmittedWorkDone(); // warm-up: pipeline compile
    const t = [];
    for (let f = 1; f <= frames; f++) {
      const t0 = performance.now(); r.frame(u(f)); await r.device.queue.onSubmittedWorkDone();
      t.push(performance.now() - t0);
    }
    t.sort((x, y) => x - y);
    // Geodesic cache: build every set (timed), then cached frames with the jet on and off.
    const { BuildScheduler } = await import("/src/render/cache-plan.ts");
    let build = null, cachedOn = null, cachedOff = null, bm = null;
    if (r.cacheSets) {
      r.resetCache();
      const sched = new BuildScheduler(r.cacheSets, r.displayH);
      const b0 = performance.now();
      for (let x = sched.next(); x; x = sched.next()) {
        r.frame(u(1), { build: { ...u(1), resW: r.displayW, resH: r.displayH, setIndex: x.set, rowStart: x.rowStart, rowEnd: x.rowEnd } });
        await r.device.queue.onSubmittedWorkDone();
      }
      build = (performance.now() - b0) / r.cacheSets; // per set, including the live frames it rode on
      bm = await r.readbackBookmarks();
      const cachedMedian = async (jet) => {
        const ts = [];
        for (let f = 1; f <= frames; f++) {
          const uf = { ...u(f), jetStrength: jet, setIndex: f % r.cacheSets };
          const t0 = performance.now(); r.frame(uf, { cachedSet: f % r.cacheSets }); await r.device.queue.onSubmittedWorkDone();
          ts.push(performance.now() - t0);
        }
        ts.sort((x, y) => x - y); return ts[ts.length >> 1];
      };
      cachedOn = await cachedMedian(1); cachedOff = await cachedMedian(0);
    }
    r.device.destroy();
    return { w: r.width, h: r.height, median: t[t.length >> 1], build, cachedOn, cachedOff, sets: r.cacheSets,
      bmFrac: bm ? bm.count / (r.displayW * r.displayH * r.cacheSets) : null, meanNJet: bm ? bm.meanNJet : null };
  }
  return { adapter: `${info.vendor || "?"} ${info.architecture || "?"}`, rows: [await bench(1280, 720), await bench(1920, 1080)] };
});

console.log(`adapter: ${out.adapter}`);
for (const r of out.rows) {
  console.log(`${`${r.w}x${r.h}`.padEnd(10)} live   median ${r.median.toFixed(1)} ms/frame (${(1000 / r.median).toFixed(1)} fps)`);
  if (r.sets) {
    console.log(`${"".padEnd(10)} cached median ${r.cachedOn.toFixed(1)} ms jet on (${(1000 / r.cachedOn).toFixed(1)} fps), ${r.cachedOff.toFixed(1)} ms jet off; ` +
      `build ${r.build.toFixed(0)} ms/set x ${r.sets}; bookmarks ${(100 * r.bmFrac).toFixed(1)} % mean nJet ${r.meanNJet.toFixed(1)}`);
  } else console.log(`${"".padEnd(10)} cache unavailable on this adapter`);
}
if (diags.length) console.log("console diagnostics:", diags.join(" | "));
await browser.close();
process.exit(diags.some((d) => d.startsWith("PAGEERROR")) ? 1 : 0);
