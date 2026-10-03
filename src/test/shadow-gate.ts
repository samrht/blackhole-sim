// Critical-curve gate for ?shadow (roadmap: "a true critical-curve gate"). The GPU's `audit` pass traces
// every pixel with the disk and jet removed, so each ray ends captured (shadow) or escaped (sky); here we
// compare that image with the exact answer: classify(xi, eta, a) on the same rays' conserved constants.
// Rays that ran out of steps are flagged by the audit pass (the renderer would fall back on that same
// classifier, which would make the comparison circular), excluded from the comparison and counted.
import { classify } from "../physics/shadow";
import { screenToXiEta } from "../physics/camera";

/** Audit word bit set when the ray's step budget ran out (twin: the `audit` entry point in raytrace.wgsl). */
export const AUDIT_UNRESOLVED = 4;
const KIND_SHADOW = 0, KIND_SKY = 2;

/** 1 = captured, 0 = escaped, for each pixel's ray: the renderer's pixel mapping (pixelImpact) with the
 *  sub-pixel offset `jit`, then the exact (xi, eta) of that ray and the analytic capture test. */
export function analyticClassImage(w: number, h: number, fovScale: number, a: number, incl: number,
  jit: readonly [number, number]): Uint8Array {
  const out = new Uint8Array(w * h), aspect = w / h;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const alpha = (((x + 0.5 + jit[0]) / w) * 2 - 1) * fovScale * aspect;
    const beta = -(((y + 0.5 + jit[1]) / h) * 2 - 1) * fovScale;
    const [xi, eta] = screenToXiEta(alpha, beta, a, incl);
    out[y * w + x] = classify(xi, eta, a) === "captured" ? 1 : 0;
  }
  return out;
}

export interface ShadowGate {
  offBand: number;           // resolved pixels that disagree with the analytic class away from the curve (gated: 0)
  inBand: number;            // disagreements within one pixel of the curve (reported)
  unresolved: number;        // rays that ran out of steps (excluded from the comparison)
  unresolvedOffBand: number; // ... of them away from the curve (gated: 0)
  other: number;             // any kind but shadow/sky (gated: 0; the audit scene has no emitter)
  areaRatio: number;         // traced shadow area / analytic shadow area (gated: within areaTol)
  pass: boolean;
}

/** A pixel is "on the curve" when the analytic class changes anywhere in its 3x3 neighbourhood. */
export function compareShadow(words: Uint32Array, analytic: Uint8Array, w: number, h: number, areaTol = 0.005): ShadowGate {
  let offBand = 0, inBand = 0, unresolved = 0, unresolvedOffBand = 0, other = 0, gpuCap = 0, anCap = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, c = analytic[i];
    anCap += c;
    let band = false;
    for (let dy = -1; dy <= 1 && !band; dy++) for (let dx = -1; dx <= 1; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx >= 0 && yy >= 0 && xx < w && yy < h && analytic[yy * w + xx] !== c) { band = true; break; }
    }
    const kind = words[i] & 3;
    if (kind !== KIND_SHADOW && kind !== KIND_SKY) { other++; continue; }
    const cap = kind === KIND_SHADOW ? 1 : 0;
    gpuCap += cap;
    if (words[i] & AUDIT_UNRESOLVED) { unresolved++; if (!band) unresolvedOffBand++; continue; }
    if (cap !== c) { if (band) inBand++; else offBand++; }
  }
  const areaRatio = anCap > 0 ? gpuCap / anCap : NaN;
  const pass = offBand === 0 && unresolvedOffBand === 0 && other === 0 && Math.abs(areaRatio - 1) <= areaTol;
  return { offBand, inBand, unresolved, unresolvedOffBand, other, areaRatio, pass };
}
