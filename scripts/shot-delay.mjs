// Writes delay-off.png / delay-on.png: the default view during an eruption flare (the first flux tube orbiting
// inside 10 r_g, a fifth of an orbit after it circularises; spec 2026-10-04), same scene time,
// light-travel delay off vs on (dev server on :5173). For visual review of the echoes.
import { chromium } from "playwright-core";
const BASE = process.env.BASE || "http://localhost:5173";
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1000, height: 700 } });
await page.goto(BASE + "/?bench", { waitUntil: "load" });
for (const ld of [0, 1]) {
  const png = await page.evaluate(async (ld) => {
    const { Renderer } = await import("/src/render/gpu.ts");
    const { buildTempLUT, buildVisibleLUT, lumNormFor } = await import("/src/physics/lookups.ts");
    const { iscoRadius } = await import("/src/physics/orbits.ts");
    document.body.innerHTML = ""; const c = document.createElement("canvas"); c.style.cssText = "width:960px;height:540px;display:block"; document.body.appendChild(c);
    const r = new Renderer(); await r.init(c);
    const a = 0.9, rIn = iscoRadius(a, true);
    r.uploadLUTs(buildTempLUT(a, true, rIn, 40, 512), buildVisibleLUT());
    r.rebind();
    const { tubeOf, tubeOrbit, flareZeta } = await import("/src/physics/eruption-spots.ts");
    const { jetUniforms, ETA_DEFAULT } = await import("/src/physics/synchrotron.ts");
    const { CUSTOM_DEFAULT } = await import("/src/physics/presets.ts");
    let k = 0; while (tubeOf(k, a).rc >= 10) k++;
    const tb = tubeOf(k, a), when = tb.t0 + tb.D + 0.2 * tubeOrbit(tb, a);
    const J = jetUniforms(CUSTOM_DEFAULT.massSun, a, CUSTOM_DEFAULT.lambda, ETA_DEFAULT, 60, 2, 1);
    for (let f = 0; f < 16; f++) r.frame({ resW: r.width, resH: r.height, outW: r.displayW, outH: r.displayH, a, incl: (60 * Math.PI) / 180, rObs: 1000,
      fovScale: 14, rIn, rOut: 40, Tpeak: 3e4, lumNorm: lumNormFor(3e4), lightDelay: ld, exposure: -1, time: when, frame: f, reset: f === 0 ? 1 : 0,
      maxSteps: 4800, blend: 1 / (f + 1), timeScale: 1, turbAmp: 0, breatheAmp: 0, flareStrength: 1, flarePhi: J.phi, flareZeta: flareZeta(), jetStrength: 0, jetGamma: 2, jetLength: 60, fluxVar: 1, skyStrength: 0,
      jetB0: 0, jetQ0: 0, rgCm: J.rgCm }); // jet off; rgCm feeds the flare's units
    await r.device.queue.onSubmittedWorkDone();
    return c.toDataURL("image/png");
  }, ld);
  (await import("node:fs")).writeFileSync(`delay-${ld ? "on" : "off"}.png`, Buffer.from(png.split(",")[1], "base64"));
  console.log(`• delay-${ld ? "on" : "off"}.png`);
}
await browser.close();
