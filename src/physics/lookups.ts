import { temperatureShape } from "./disk";
import { blackbodyVisibleRGB, relLuminance } from "./color";

/** Normalized T(r) sampled uniformly over [rIn, rOut]; returns Float32Array(N), peak=1. */
export function buildTempLUT(a: number, prograde: boolean, rIn: number, rOut: number, N: number): Float32Array {
  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const r = rIn + (rOut - rIn) * (i / (N - 1));
    out[i] = temperatureShape(r, a, prograde);
  }
  // Renormalize to exact peak=1 over the discrete grid (temperatureShape peaks between samples)
  const peak = Math.max(...out);
  if (peak > 0) for (let i = 0; i < N; i++) out[i] /= peak;
  return out;
}
/** Visible-radiance LUT range and size (spec 2026-10-01 §2.3; 4096 entries: plan Task 2). */
export const VIS_TMIN = 100, VIS_TMAX = 1e9, VIS_LUT_N = 4096, VIS_TREF = 1e4;
const LREF = relLuminance(blackbodyVisibleRGB(VIS_TREF));

/** Visible luminance of a blackbody at T relative to one at VIS_TREF. */
export const visibleLuminance = (T: number) => relLuminance(blackbodyVisibleRGB(T)) / LREF;
/** Cap for `lumNorm`: f32-safe with headroom (LUT entries reach ~1e5, exposure 2^4). A disk whose
 *  visible luminance is below 1e-30 of a 1e4 K one (T_peak below ~365 K) renders dark, not NaN. */
export const LUM_NORM_MAX = 1e30;
/** The `lumNorm` uniform: 1 / visible luminance of the disk's rest-frame peak temperature, so the
 *  exposure slider means the same for every preset. */
export const lumNormFor = (tPeak: number) => Math.min(LUM_NORM_MAX, 1 / visibleLuminance(Math.max(tPeak, VIS_TMIN)));

/** Visible-band linear-sRGB radiance (RGBA, A = 1) at T log-spaced over [VIS_TMIN, VIS_TMAX],
 *  relative to VIS_TREF's luminance so f32 holds it (Wien-tail entries underflow towards 0). */
export function buildVisibleLUT(n = VIS_LUT_N): Float32Array {
  const out = new Float32Array(n * 4);
  const l0 = Math.log(VIS_TMIN), l1 = Math.log(VIS_TMAX);
  for (let i = 0; i < n; i++) {
    const c = blackbodyVisibleRGB(Math.exp(l0 + ((l1 - l0) * i) / (n - 1)));
    out[i * 4] = c[0] / LREF; out[i * 4 + 1] = c[1] / LREF; out[i * 4 + 2] = c[2] / LREF; out[i * 4 + 3] = 1;
  }
  return out;
}
/** CPU twin of sampleColor in raytrace.wgsl: log-T index, clamped, linear interpolation. */
export function sampleVisibleLUT(lut: Float32Array, T: number): [number, number, number] {
  const n = lut.length / 4, l0 = Math.log(VIS_TMIN), l1 = Math.log(VIS_TMAX);
  const u = Math.min(1, Math.max(0, (Math.log(T) - l0) / (l1 - l0))) * (n - 1);
  const i0 = Math.floor(u), i1 = Math.min(i0 + 1, n - 1), f = u - i0;
  const at = (i: number, k: number) => lut[i * 4 + k];
  return [0, 1, 2].map((k) => at(i0, k) + (at(i1, k) - at(i0, k)) * f) as [number, number, number];
}
