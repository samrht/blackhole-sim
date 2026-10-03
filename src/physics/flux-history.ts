// Horizon magnetic-flux history Phi(t) of a magnetically arrested disk (specs 2026-10-03 jet flux knots and flux
// statistics). Phi / Phi_sat = (1 - d(t)) (1 + eps n(t)): d is the eruption sawtooth (recurrence ~1500 M,
// arXiv 2510.25842; exponential drop with the converged 500 M e-folding, Ripperda et al. 2022; linear refill),
// n is fast red-noise flicker (Gaussian lattice, 50 M cells, clipped at +-3). (dbar, eps) make the rms
// within 1000 M windows (Narayan et al. 2022 eq. 13) equal their absolute-flux Phi value (Fig. 9, S3.3: flat in prograde
// spin) and the eruptions 2 sigma deep (2510.25842 SIV.3's ansatz delta-phi = 2 sigma-phi, carried over to Phi). WGSL twin: fluxRatioJ in emission-shared.wgsl.
import { hash4, gaussNode } from "./emission";

export const FLUX = {
  T: 1500, jitter: 1 / 3, tauD: 500, spread: 0.5, floor: 0.05, sMax: 1.4,
  flickerCell: 50, flickerClip: 3, c2: 0.995007, // E[clip(n)^2] for a standard normal clipped at 3
  saltT: 0x464c, saltD: 0x4458, saltN: 0x464e,
  // scripts/calibrate-flux.ts: dbar = 2 sigma/mu of the series, eps sets the 1000 M modulation index to fluxTarget()
  dbar: 0.1842, eps: 0.06844, d1: 0.092856, d2: 0.011715, d3: 0.001693,
} as const;
/** Narayan et al. 2022's absolute-flux (Phi) modulation index, 1000 t_g windows: flat in prograde spin. Their S3.3:
 *  Phi variability is "largely independent of prograde spin"; the Fig. 9 dashed lines only connect a = 0 to 0.9.
 *  The value is the mean of the five prograde points 0.067, 0.079, 0.098, 0.081, 0.098 (final review). */
export function fluxTarget(_a = 0): number { return 0.0846; }
/** Calibration (dbar, eps, <d^k>); spin-independent, so the same for every a (twin: fluxParamsJ). */
export function fluxParams(_a: number) {
  return { dbar: FLUX.dbar, eps: FLUX.eps, d1: FLUX.d1, d2: FLUX.d2, d3: FLUX.d3 };
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
