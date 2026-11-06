// Pure parts of the ?imgerr experiment (image-error.browser.ts): the views, the pixel rays, and how two images are compared.
import { summarize, type Summary } from "./drift-analysis";

export interface ImageView { name: string; a: number; inclDeg: number; fov: number; panX: number; panY: number; time: number; note: string }
export const IMAGE_VIEWS: ImageView[] = [
  { name: "default", a: 0.9, inclDeg: 72, fov: 14, panX: 0, panY: 0, time: 37, note: "the app's default view" },
  { name: "edge", a: 0.998, inclDeg: 85, fov: 14, panX: 0, panY: 0, time: 37, note: "high spin, near edge-on" },
  { name: "schw", a: 0, inclDeg: 30, fov: 14, panX: 0, panY: 0, time: 37, note: "no spin" },
  { name: "zoom-pole", a: 0.998, inclDeg: 85, fov: 14 / 16, panX: 0, panY: 4.7, time: 37, note: "16x zoom onto the alpha = 0 column, where near-pole rays hit the disk" },
  { name: "zoom-ring", a: 0.9, inclDeg: 72, fov: 14 / 16, panX: 5.2, panY: 0, time: 37, note: "16x zoom onto the bright inner disk at the shadow's edge" },
];

/** raytrace.wgsl pixelImpact, op by op in f32: (alpha, beta) per pixel, row by row. */
export function pixelRays(w: number, h: number, fov: number, panX: number, panY: number, jit: readonly [number, number]): Float32Array {
  const F = Math.fround, out = new Float32Array(w * h * 2), aspect = F(F(w) / F(h)), ff = F(fov), px = F(panX), py = F(panY);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const nx = F(F(F(F(F(x + 0.5) + F(jit[0])) / F(w)) * 2) - 1), ny = F(F(F(F(F(y + 0.5) + F(jit[1])) / F(h)) * 2) - 1), i = (y * w + x) * 2;
    out[i] = F(F(F(nx * ff) * aspect) + px); out[i + 1] = F(F(-ny * ff) + py);
  }
  return out;
}

export interface LinearDiff { fluxRel: number; pxRel: Summary; nonFinite: number }
/** Two linear RGBA images: total luminance difference, and per-pixel |dL| / max(L_ref, 1e-3 of the mean), so dark pixels cannot blow it up. */
export function compareLinear(ref: Float32Array, got: Float32Array, mask?: (i: number) => boolean): LinearDiff {
  const n = ref.length / 4, lum = (a: Float32Array, i: number) => 0.2126 * a[i * 4] + 0.7152 * a[i * 4 + 1] + 0.0722 * a[i * 4 + 2];
  let sumR = 0, sumG = 0, count = 0, nonFinite = 0;
  for (let i = 0; i < n; i++) {
    if (mask && !mask(i)) continue;
    const lr = lum(ref, i), lg = lum(got, i);
    if (!Number.isFinite(lg) || !Number.isFinite(lr)) { nonFinite++; continue; }
    sumR += lr; sumG += lg; count++;
  }
  const floor = 1e-3 * (count ? sumR / count : 0), rel = new Float64Array(count); let k = 0;
  for (let i = 0; i < n; i++) {
    if (mask && !mask(i)) continue;
    const lr = lum(ref, i), lg = lum(got, i);
    if (!Number.isFinite(lg) || !Number.isFinite(lr)) continue;
    rel[k++] = Math.abs(lg - lr) / Math.max(lr, floor, 1e-300);
  }
  return { fluxRel: sumR > 0 ? Math.abs(sumG - sumR) / sumR : 0, pxRel: summarize(rel), nonFinite };
}

export interface DisplayDiff { pixels: number; ge1: number; ge2: number; ge5: number; maxDiff: number; psnr: number }
/** Two 8-bit RGBA images, as displayed: pixels by their largest channel difference, and PSNR over the colour channels. */
export function compareDisplay(ref: Uint8Array | Uint8ClampedArray, got: Uint8Array | Uint8ClampedArray): DisplayDiff {
  const pixels = ref.length / 4; let ge1 = 0, ge2 = 0, ge5 = 0, maxDiff = 0, se = 0;
  for (let i = 0; i < pixels; i++) {
    let m = 0;
    for (let c = 0; c < 3; c++) { const d = Math.abs(ref[i * 4 + c] - got[i * 4 + c]); se += d * d; if (d > m) m = d; }
    if (m >= 1) ge1++; if (m >= 2) ge2++; if (m >= 5) ge5++; if (m > maxDiff) maxDiff = m;
  }
  return { pixels, ge1, ge2, ge5, maxDiff, psnr: se === 0 ? Infinity : 10 * Math.log10((255 * 255) / (se / (pixels * 3))) };
}
