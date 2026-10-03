// Horizon magnetic-flux history Phi(t) of a magnetically arrested disk (specs 2026-10-03 jet flux knots and flux
// statistics). Phi / Phi_sat = (1 - d(t)) (1 + eps n(t)): d is the eruption sawtooth (recurrence ~1500 M,
// arXiv 2510.25842; exponential drop with the converged 500 M e-folding, Ripperda et al. 2022; linear refill),
// n is fast red-noise flicker (Gaussian lattice, 50 M cells, clipped at +-3). Per spin, (dbar, eps) make the rms
// within 1000 M windows (Narayan et al. 2022 eq. 13) equal their absolute-flux Phi trend line (Fig. 9) and the
// eruptions 2 sigma deep (2510.25842 SIV.3). WGSL twin: fluxRatioJ in emission-shared.wgsl.
import { hash4, gaussNode } from "./emission";

export const FLUX = {
  T: 1500, jitter: 1 / 3, tauD: 500, spread: 0.5, floor: 0.05, sMax: 1.4,
  flickerCell: 50, flickerClip: 3, c2: 0.995007, // E[clip(n)^2] for a standard normal clipped at 3
  saltT: 0x464c, saltD: 0x4458, saltN: 0x464e,
} as const;
/** (dbar, eps) and the eruption moments <d^k> at s = 1 per spin (scripts/calibrate-flux.ts). */
export const FLUX_SPIN = [
  { a: 0, dbar: 0.1455, eps: 0.05499, d1: 0.073273, d2: 0.007298, d3: 0.000833 },
  { a: 0.3, dbar: 0.1682, eps: 0.06295, d1: 0.084751, d2: 0.009761, d3: 0.001288 },
  { a: 0.6, dbar: 0.191, eps: 0.07072, d1: 0.096304, d2: 0.0126, d3: 0.001889 },
  { a: 0.9, dbar: 0.2138, eps: 0.0783, d1: 0.107885, d2: 0.015808, d3: 0.002653 },
] as const;
/** Narayan et al. 2022 Fig. 9 Phi trend line, 1000 t_g modulation index (prograde; a > 0.9 uses 0.9). */
export function fluxTarget(a: number): number { return 0.067 + (0.031 * Math.min(0.9, Math.max(0, a))) / 0.9; }
export function fluxParams(a: number) {
  const x = Math.min(0.9, Math.max(0, a));
  let i = 0; while (i < FLUX_SPIN.length - 2 && x > FLUX_SPIN[i + 1].a) i++;
  const lo = FLUX_SPIN[i], hi = FLUX_SPIN[i + 1], w = (x - lo.a) / (hi.a - lo.a);
  if (w === 0) return { dbar: lo.dbar, eps: lo.eps, d1: lo.d1, d2: lo.d2, d3: lo.d3 };
  if (w === 1) return { dbar: hi.dbar, eps: hi.eps, d1: hi.d1, d2: hi.d2, d3: hi.d3 };
  const L = (p: number, q: number) => p + (q - p) * w;
  return { dbar: L(lo.dbar, hi.dbar), eps: L(lo.eps, hi.eps), d1: L(lo.d1, hi.d1), d2: L(lo.d2, hi.d2), d3: L(lo.d3, hi.d3) };
}

export function fluxHash(k: number, salt: number): number { return (hash4(k, 0x7a11, 0, salt) & 0xffffff) / 16777216 - 0.5; }
export function eruptionTime(k: number): number { return FLUX.T * (k + 0.5 + FLUX.jitter * fluxHash(k, FLUX.saltT)); }
export function eruptionDepth(k: number, dbar: number): number { return dbar * (1 + FLUX.spread * fluxHash(k, FLUX.saltD)); }
/** Deficit at time k T + loc, loc in [0, T) (twin: fluxDeficitJ). */
export function fluxDeficitLocal(k: number, loc: number, dbar: number): number {
  let kk = k, dt = loc - FLUX.T * (0.5 + FLUX.jitter * fluxHash(k, FLUX.saltT));
  if (dt < 0) { kk = k - 1; dt = loc + FLUX.T - FLUX.T * (0.5 + FLUX.jitter * fluxHash(kk, FLUX.saltT)); }
  const gap = FLUX.T * (1 + FLUX.jitter * (fluxHash(kk + 1, FLUX.saltT) - fluxHash(kk, FLUX.saltT)));
  const depth = eruptionDepth(kk, dbar), D = -FLUX.tauD * Math.log(1 - depth);
  if (dt < D) return 1 - Math.exp(-dt / FLUX.tauD);
  return (depth * (gap - dt)) / (gap - D);
}
export function fluxDeficit(t: number, dbar: number): number { const k = Math.floor(t / FLUX.T); return fluxDeficitLocal(k, t - k * FLUX.T, dbar); }
const smooth = (x: number) => x * x * (3 - 2 * x);
const flickerNode = (i: number) => gaussNode(i, 0x0f1c, 0, FLUX.saltN);
/** Unit-Gaussian flicker (exact N(0,1) before the clip), lattice cells of 50 M (twin: fluxFlickerJ). */
export function fluxFlicker(t: number): number {
  const x = t / FLUX.flickerCell, i = Math.floor(x), f = smooth(x - i), w0 = 1 - f, w1 = f;
  const n = (w0 * flickerNode(i) + w1 * flickerNode(i + 1)) / Math.sqrt(w0 * w0 + w1 * w1);
  return Math.max(-FLUX.flickerClip, Math.min(FLUX.flickerClip, n));
}
/** f = Phi / <Phi> at time t, slider s, spin a (s = 0: exactly 1). */
export function fluxRatio(t: number, s: number, a: number): number {
  if (s === 0) return 1;
  const p = fluxParams(a);
  return Math.max(FLUX.floor, (1 - s * fluxDeficit(t, p.dbar)) * (1 + s * p.eps * fluxFlicker(t))) / (1 - s * p.d1);
}
/** <f^n>, n = 2, 3: flicker independent of the eruptions and symmetric (odd moments 0). */
export function fluxMoment(n: 2 | 3, s: number, a: number): number {
  const p = fluxParams(a), mu = 1 - s * p.d1, e2 = s * s * p.eps * p.eps * FLUX.c2;
  if (n === 2) return ((1 - 2 * s * p.d1 + s * s * p.d2) * (1 + e2)) / (mu * mu);
  return ((1 - 3 * s * p.d1 + 3 * s * s * p.d2 - s ** 3 * p.d3) * (1 + 3 * e2)) / mu ** 3;
}
/** Phi / Phi_sat sampled at midpoints every dt over [0, span) at s = 1 (calibration). */
export function fluxSeries(dbar: number, eps: number, span = 1e6, dt = 5): number[] {
  const out: number[] = [];
  for (let t = dt / 2; t < span; t += dt) out.push((1 - fluxDeficit(t, dbar)) * (1 + eps * fluxFlicker(t)));
  return out;
}
/** Narayan et al. 2022 eq. 13: rms / mean within consecutive windows, averaged. */
export function modulationIndex(series: number[], window: number, dt = 5): number {
  const n = Math.round(window / dt); let s = 0, c = 0;
  for (let i = 0; i + n <= series.length; i += n) {
    let m = 0, q = 0; for (let j = i; j < i + n; j++) { m += series[j]; q += series[j] * series[j]; }
    m /= n; s += Math.sqrt(Math.max(0, q / n - m * m)) / m; c++;
  }
  return s / c;
}
export function seriesSigma(series: number[]): number {
  let m = 0, q = 0; for (const v of series) { m += v; q += v * v; }
  m /= series.length; return Math.sqrt(q / series.length - m * m) / m;
}
