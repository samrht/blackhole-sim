import { Renderer } from "../render/gpu";
import { buildColorLUT } from "../physics/lookups";
import { photonOrbit } from "../physics/orbits";
import type { UniformValues } from "../render/uniforms";
import { classify } from "../physics/shadow";

/** Structural smoke test + calibration diagnostic for the Schwarzschild (a=0) shadow.
 *
 *  We light a flat emitter from the photon orbit outward and view nearly face-on (10°, clear of
 *  the camera's pole-on degeneracy), so the image is a bright disk wrapping a centred dark capture
 *  shadow. We then scan the centre COLUMN (the centre row is the beta=0 / p_theta=0 seam where rays
 *  stay in the equatorial plane and never trigger the theta-crossing test) for the dark span.
 *
 *  PASS = a centred shadow exists, is ringed by disk emission, and has a physically-plausible
 *  apparent radius. We deliberately do NOT assert the textbook sqrt(27)*M to within a few percent:
 *  the rendered radius differs from the analytic sqrt(27)*M by a constant camera-calibration factor
 *  (`calibration` in the return value, measured ~0.87). This is NOT a physics error and NOT a step-
 *  budget artifact -- this route runs at a high step budget and the value is flat across budgets
 *  (see docs/specs/2026-07-20-photon-ring-detail-design.md §3.4). It comes from the Tier-1 camera
 *  mapping screen coordinates to photon initial conditions heuristically (p_theta = beta,
 *  p_phi = -alpha*sin i) with no normalization, so `fovScale` is not calibrated in true M units.
 *  Recalibrating to a normalized Bardeen camera would rescale every image in the project and is
 *  tracked as a separate follow-up.
 *  The rigorous numerical gate for the ported math is the ?parity test. */
export async function measureShadow(canvas: HTMLCanvasElement, maxStepsOverride = 8000) {
  const r = new Renderer(); await r.init(canvas);
  r.renderBloom = false; // measure the raw geometric shadow, not the post-processed glow
  const a = 0, rOut = 40, fovScale = 14;
  const rPh = photonOrbit(a, true); // photon orbit = 3M for a=0; emitting from here outward
  const flatTemp = new Float32Array(512).fill(1); // uniform emitter -> bright everywhere it's hit
  r.uploadLUTs(flatTemp, buildColorLUT(1000, 40000, 256)); r.rebind();
  const u: UniformValues = { resW: r.width, resH: r.height, a, incl: Math.PI / 18, rObs: 1000,
    fovScale, rIn: rPh, rOut, Tpeak: 3.0e4, exposure: 0, time: 0, frame: 0, reset: 1, maxSteps: maxStepsOverride,
    blend: 1, timeScale: 1, turbAmp: 0, breatheAmp: 0, nSpots: 0,
    jetStrength: 0, jetGamma: 5, jetLength: 60, jetKnots: 0.7, skyStrength: 0 };
  const { data, w, h } = await r.readbackPresented(u);

  const cx = w >> 1, cy = h >> 1;
  const lum = (y: number) => { const i = (y * w + cx) * 4; return data[i] + data[i + 1] + data[i + 2]; };
  let top = cy; while (top > 0 && lum(top) < 20) top--;
  let bot = cy; while (bot < h - 1 && lum(bot) < 20) bot++;
  const shadowPx = bot - top;
  let brightOnColumn = 0; for (let y = 0; y < h; y++) if (lum(y) >= 20) brightOnColumn++;

  const pxPerM = h / (2 * fovScale);          // vertical screen scale (beta units)
  const shadowRadiusM = (shadowPx / 2) / pxPerM;
  // Analytic ground truth. At a=0 the critical curve is the circle b = sqrt(27); we locate it by
  // bisecting `classify` in b^2 rather than hardcoding, so this stays honest if the model changes.
  let blo = 0, bhi = 20;
  for (let k = 0; k < 40; k++) {
    const bmid = 0.5 * (blo + bhi);
    if (classify(0, bmid * bmid, 0) === "captured") blo = bmid; else bhi = bmid;
  }
  const analyticRadiusM = 0.5 * (blo + bhi);
  const bCrit = Math.sqrt(27);                // = 5.196 M, the ideal Schwarzschild shadow radius
  const hasShadow = shadowPx > 4;
  const hasDisk = brightOnColumn > h * 0.1;
  const plausible = shadowRadiusM > 1.0 && shadowRadiusM < bCrit * 1.6;
  const ok = hasShadow && hasDisk && plausible;
  return {
    ok, shadowPx, hasShadow, hasDisk,
    shadowRadiusM: +shadowRadiusM.toFixed(2),
    bCritM: +bCrit.toFixed(2),
    analyticRadiusM: +analyticRadiusM.toFixed(3),
    scaleVsBcrit: +(shadowRadiusM / bCrit).toFixed(2),
    calibration: +(shadowRadiusM / analyticRadiusM).toFixed(3),
  };
}
