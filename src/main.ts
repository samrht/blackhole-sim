import { Renderer } from "./render/gpu";
import type { UniformValues } from "./render/uniforms";
import { buildTempLUT, buildColorLUT } from "./physics/lookups";
import { iscoRadius } from "./physics/orbits";

const canvas = document.getElementById("c") as HTMLCanvasElement;
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
