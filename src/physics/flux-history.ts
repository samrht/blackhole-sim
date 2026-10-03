// Horizon magnetic-flux history phi_BH(t) of a magnetically arrested disk (spec 2026-10-03 jet flux knots, 2.2):
// a deterministic sawtooth -- eruptions every ~1500 M (arXiv 2510.25842), each an exponential drop with the
// converged reconnection e-folding of 500 M (Ripperda et al. 2022), then a linear refill as gas re-advects flux.
// Depth calibrated so phi's rms swing is 20.9 % (Narayan et al. 2022, a = 0.9). WGSL twin: fluxRatioJ in
// emission-shared.wgsl (same hash, same split into whole periods + remainder).
import { hash4 } from "./emission";

export const FLUX = {
  T: 1500,            // mean recurrence (M)
  jitter: 1 / 3,      // eruption k at T (k + 0.5 + jitter u_k): gaps in [1000, 2000] M
  tauD: 500,          // drop e-folding (M)
  spread: 0.5,        // depth_k = dbar (1 + spread v_k)
  floor: 0.05,        // phi never below 5 % of saturation
  sMax: 1.4,          // Flux variability slider maximum (floor unreachable below it)
  target: 0.209,      // sigma/mu of phi at s = 1
  dbar: 0.5058,       // mean depth (scripts/calibrate-flux.ts)
  d1: 0.262371, d2: 0.092607, d3: 0.037282, // <d>, <d^2>, <d^3> at s = 1 over 1e6 M (same script)
  saltT: 0x464c, saltD: 0x4458,
} as const;

/** Uniform in [-0.5, 0.5) from eruption index k (twin: fluxHashJ). */
export function fluxHash(k: number, salt: number): number {
  return (hash4(k, 0x7a11, 0, salt) & 0xffffff) / 16777216 - 0.5;
}
export function eruptionTime(k: number): number { return FLUX.T * (k + 0.5 + FLUX.jitter * fluxHash(k, FLUX.saltT)); }
export function eruptionDepth(k: number, dbar: number = FLUX.dbar): number { return dbar * (1 + FLUX.spread * fluxHash(k, FLUX.saltD)); }

/** Deficit at time k T + loc, loc in [0, T): eruption k (at T(0.5 + jitter u_k) into its period) or, before it,
 *  eruption k - 1 still refilling. Eruption k + 1 is always later than (k + 1) T + T/3 > t. */
export function fluxDeficitLocal(k: number, loc: number, dbar: number = FLUX.dbar): number {
  let kk = k, dt = loc - FLUX.T * (0.5 + FLUX.jitter * fluxHash(k, FLUX.saltT));
  if (dt < 0) { kk = k - 1; dt = loc + FLUX.T - FLUX.T * (0.5 + FLUX.jitter * fluxHash(kk, FLUX.saltT)); }
  const gap = FLUX.T * (1 + FLUX.jitter * (fluxHash(kk + 1, FLUX.saltT) - fluxHash(kk, FLUX.saltT)));
  const depth = eruptionDepth(kk, dbar), D = -FLUX.tauD * Math.log(1 - depth);
  if (dt < D) return 1 - Math.exp(-dt / FLUX.tauD);
  return (depth * (gap - dt)) / (gap - D);
}
export function fluxDeficit(t: number, dbar: number = FLUX.dbar): number {
  const k = Math.floor(t / FLUX.T);
  return fluxDeficitLocal(k, t - k * FLUX.T, dbar);
}
/** f = phi / mean(phi) at time t for slider s (s = 0: exactly 1). */
export function fluxRatio(t: number, s: number): number {
  if (s === 0) return 1;
  return Math.max(FLUX.floor, 1 - s * fluxDeficit(t)) / (1 - s * FLUX.d1);
}
/** <f^n> for n = 2, 3 from the generator's deficit moments (floor unreachable for s <= sMax). */
export function fluxMoment(n: 2 | 3, s: number): number {
  const mu = 1 - s * FLUX.d1;
  if (n === 2) return (1 - 2 * s * FLUX.d1 + s * s * FLUX.d2) / (mu * mu);
  return (1 - 3 * s * FLUX.d1 + 3 * s * s * FLUX.d2 - s ** 3 * FLUX.d3) / mu ** 3;
}
/** Deficit moments and phi's sigma/mu at s = 1, sampled every dt over [0, span) (midpoints). */
export function measureFlux(dbar: number, span = 1e6, dt = 1) {
  let n = 0, a = 0, b = 0, c = 0;
  for (let t = dt / 2; t < span; t += dt) { const d = fluxDeficit(t, dbar); a += d; b += d * d; c += d * d * d; n++; }
  a /= n; b /= n; c /= n;
  return { d1: a, d2: b, d3: c, sigmaOverMu: Math.sqrt(b - a * a) / (1 - a) };
}
