import { Renderer } from "../render/gpu";
import { buildVisibleLUT, lumNormFor } from "../physics/lookups";
import { photonOrbit } from "../physics/orbits";
import type { UniformValues } from "../render/uniforms";
import { classify } from "../physics/shadow";
import { analyticClassImage, compareShadow, type ShadowGate } from "./shadow-gate";
import { JITTER } from "../render/cache-plan";

/** Critical-curve gate (three spins/inclinations) plus the structural Schwarzschild (a=0) smoke test.
 *
 *  We light a flat emitter from the photon orbit outward and view nearly face-on (10°, clear of
 *  the camera's pole-on degeneracy), so the image is a bright disk wrapping a centred dark capture
 *  shadow. We then scan the centre COLUMN (the centre row is the beta=0 / p_theta=0 seam where rays
 *  stay in the equatorial plane and never trigger the theta-crossing test) for the dark span.
 *
 *  PASS = a centred shadow exists, is ringed by disk emission, and has a physically-plausible
 *  apparent radius. We deliberately do NOT assert the textbook sqrt(27)*M: the dark span this
 *  route measures is NOT the critical curve. With the emitter starting at rIn = 3 M (the photon
 *  orbit, inside the capture region) the centre column's captured-class rays with 4.2 <= beta < 5.196
 *  (eta < 27) cross the equatorial plane at r = 3.05-4.02 M and register as disk hits BEFORE they
 *  reach the horizon (on the beta < 0 side even |beta| = 3.9 hits at r = 3.3 M), verified on the
 *  f64 CPU twin. So the dark span is the lensed silhouette of the emitter's inner edge, and
 *  `calibration` (= shadowRadiusM / analyticRadiusM, currently 0.761) is the ratio of that
 *  silhouette to the critical curve -- a regression number, not a camera-calibration factor. The
 *  earlier reading of it as "camera calibration, not physics" (~0.87 before the polar-axis fix)
 *  was a misdiagnosis: the 0.87 also contained the old POLE_S2 floor's tunnelling artefact, which
 *  hid the far-side crossings of the xi = 0 rays on this very column; with the axis handled
 *  physically the number moved to 0.749 and is self-consistent. (0.749 -> 0.761 with exact metric
 *  derivatives: a lit axis pixel inside the shadow had stopped this column's scan ~3 px early.)
 *  The true critical-curve gate (2026-10-03) is the (xi, eta) classification image below
 *  (shadow-gate.ts); this structural check is kept alongside it. The field names are kept for stability.
 *  The rigorous numerical gate for the ported math is the ?parity test. */
export async function measureShadow(canvas: HTMLCanvasElement, maxStepsOverride = 8000) {
  const r = new Renderer(); await r.init(canvas);
  r.renderBloom = false; // measure the raw geometric shadow, not the post-processed glow
  const a = 0, rOut = 40, fovScale = 14;
  const rPh = photonOrbit(a, true); // photon orbit = 3M for a=0; emitting from here outward
  const flatTemp = new Float32Array(512).fill(1); // uniform emitter -> bright everywhere it's hit
  r.uploadLUTs(flatTemp, buildVisibleLUT()); r.rebind();
  const u: UniformValues = { resW: r.width, resH: r.height, outW: r.displayW, outH: r.displayH, a, incl: Math.PI / 18, rObs: 1000,
    fovScale, rIn: rPh, rOut, Tpeak: 3.0e4, lumNorm: lumNormFor(3.0e4), lightDelay: 0, exposure: 0, time: 0, frame: 0, reset: 1, maxSteps: maxStepsOverride,
    blend: 1, timeScale: 1, turbAmp: 0, breatheAmp: 0, flareStrength: 0, flarePhi: 0, flareZeta: 0,
    jetStrength: 0, jetGamma: 2, jetLength: 60, fluxVar: 1, skyStrength: 0,
    jetB0: 0, jetQ0: 0, rgCm: 0 }; // jet off
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
  const structural = hasShadow && hasDisk && plausible;
  // Critical-curve gate: the same renderer traces every pixel with the disk removed (rIn beyond rOut, so the
  // step controller still sees the usual rOut) and the jet off; each ray ends captured or escaped, and the
  // image must match the exact classification of the same rays (shadow-gate.ts).
  const gates: { name: string; a: number; inclDeg: number; g: ShadowGate }[] = [];
  for (const [name, ga, gi] of [["schwarzschild", 0, 10], ["default", 0.9, 72], ["edge-on", 0.998, 85]] as const) {
    const incl = (gi * Math.PI) / 180;
    const words = await r.auditKinds({ ...u, a: ga, incl, rIn: 1e9, rOut, setIndex: 0 });
    const g = compareShadow(words, analyticClassImage(r.width, r.height, fovScale, ga, incl, JITTER[0]), r.width, r.height);
    gates.push({ name, a: ga, inclDeg: gi, g });
  }
  const ok = structural && gates.every((x) => x.g.pass);
  return {
    ok, structural, gates, shadowPx, hasShadow, hasDisk,
    shadowRadiusM: +shadowRadiusM.toFixed(2),
    bCritM: +bCrit.toFixed(2),
    analyticRadiusM: +analyticRadiusM.toFixed(3),
    scaleVsBcrit: +(shadowRadiusM / bCrit).toFixed(2),
    calibration: +(shadowRadiusM / analyticRadiusM).toFixed(3),
  };
}
