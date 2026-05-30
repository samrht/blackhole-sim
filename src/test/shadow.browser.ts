import { Renderer } from "../render/gpu";
import { buildTempLUT, buildColorLUT } from "../physics/lookups";
import { iscoRadius } from "../physics/orbits";
import type { UniformValues } from "../render/uniforms";

/** Renders Schwarzschild (a=0) nearly pole-on, reads back the framebuffer, and measures the
 *  central dark span along the midline against the photon-capture shadow diameter 2*sqrt(27)*M.
 *  NOTE: the disk emission is truncated at the ISCO, so the dark region is bounded by the lensed
 *  ISCO inner edge, which is somewhat larger than sqrt(27)*M — expect a few-percent positive bias. */
export async function measureShadow(canvas: HTMLCanvasElement) {
  const r = new Renderer(); await r.init(canvas);
  const a = 0, rin = iscoRadius(a, true), rOut = 40, fovScale = 14;
  r.uploadLUTs(buildTempLUT(a, true, rin, rOut, 512), buildColorLUT(1000, 40000, 256)); r.rebind();
  // FACE-ON (i≈0.01) so the shadow is a centered disk; integrate one full frame.
  const u: UniformValues = { resW: r.width, resH: r.height, a, incl: 0.01, rObs: 1000,
    fovScale, rIn: rin, rOut, Tpeak: 3.0e4, exposure: 0, time: 0, frame: 0, reset: 1, maxSteps: 8000 };
  const { data, w, h } = await r.readbackPresented(u);
  // scan the central row for the dark (shadow) span
  const y = (h >> 1) * w * 4; let lo = -1, hi = -1;
  for (let x = 0; x < w; x++) {
    const lum = data[y + x * 4] + data[y + x * 4 + 1] + data[y + x * 4 + 2];
    if (lum < 20) { if (lo < 0) lo = x; hi = x; }
  }
  const shadowPx = hi - lo;
  // expected diameter in pixels: 2*sqrt(27)*M mapped through fovScale (half-width = fovScale*M across w/2 px)
  const expectedPx = (Math.sqrt(27) / fovScale) * (w / 2) * 2;
  const relErr = Math.abs(shadowPx - expectedPx) / expectedPx;
  return { shadowPx, expectedPx, relErr };
}
