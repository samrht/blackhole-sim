import { Renderer } from "./render/gpu";
import type { UniformValues } from "./render/uniforms";
import { buildTempLUT, buildColorLUT } from "./physics/lookups";
import { iscoRadius } from "./physics/orbits";

const canvas = document.getElementById("c") as HTMLCanvasElement;

if (location.search.includes("parity")) {
  // Validation entry: CPU<->GPU parity for the metric/orbit/g-factor math.
  const { runParity } = await import("./test/parity.browser");
  const res = await runParity();
  const ok = res.maxErr < 1e-3;
  document.body.innerHTML = `<pre style="color:${ok ? "#6f6" : "#f66"};font-size:18px;padding:20px">
PARITY ${ok ? "PASS" : "FAIL"} — maxRelErr=${res.maxErr.toExponential(3)} over ${res.rows} cases</pre>`;
  console.log("parity", res);
} else if (location.search.includes("shadow")) {
  // Validation entry: Schwarzschild shadow-radius check (a=0, pole-on).
  const { measureShadow } = await import("./test/shadow.browser");
  const res = await measureShadow(canvas);
  const ok = res.relErr < 0.1;
  document.body.innerHTML = `<pre style="color:${ok ? "#6f6" : "#f66"};font-size:18px;padding:20px">
SHADOW ${ok ? "PASS" : "FAIL"} — measured=${res.shadowPx.toFixed(1)}px expected=${res.expectedPx.toFixed(1)}px relErr=${(res.relErr * 100).toFixed(1)}%</pre>`;
  console.log("shadow", res);
} else {
  // Normal interactive render.
  const r = new Renderer();
  await r.init(canvas);

  const state = { a: 0.9, incl: 72, exposure: 0 };
  const rOut = 40;
  // Effective color-temperature scale (K) for visualization. The true Novikov–Thorne
  // peak for a stellar-mass disk (~1e7 K) radiates in X-ray, so we map the normalized
  // profile into the visible blackbody range spanned by the color LUT ([1000, 40000] K).
  const T_PEAK = 3.0e4;

  let rIn = iscoRadius(state.a, true);
  function rebuildLUTs() {
    rIn = iscoRadius(state.a, true);
    r.uploadLUTs(buildTempLUT(state.a, true, rIn, rOut, 512), buildColorLUT(1000, 40000, 256));
    r.rebind();
  }
  rebuildLUTs();

  let sample = 0; // progressive-accumulation sample index; reset to 0 re-converges the image
  function loop() {
    const u: UniformValues = {
      resW: r.width, resH: r.height, a: state.a, incl: state.incl * Math.PI / 180,
      rObs: 1000, fovScale: 14, rIn, rOut, Tpeak: T_PEAK, exposure: state.exposure,
      time: performance.now() / 1000, frame: sample, reset: sample === 0 ? 1 : 0, maxSteps: 6000,
    };
    r.frame(u);
    sample++;
    requestAnimationFrame(loop);
  }
  loop();
}
