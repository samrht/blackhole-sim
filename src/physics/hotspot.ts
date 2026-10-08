// Hotspot flares in the 1.3 mm view (spec 2026-10-04 mm hotspots). One Gaussian blob per horizon-flux eruption
// (flux-history.ts) on a prograde, rigid, Keplerian equatorial orbit at 8-12 r_g (GRAVITY Collaboration 2018: 6-10 r_g;
// Wielgus et al. 2022: ~11 r_g, 74 +- 6 min for Sgr A*). It boosts the hot flow's electrons by A_k L(tau) G(d) at the flow's
// own temperature and field. WGSL twin: hotspotStateJ / hotspotBoostJ / hotspotShiftJ in emission-shared.wgsl.
import { FLUX, fluxHash, eruptionTime } from "./flux-history";
import { metricLower } from "./kerr";

export const HOTSPOT = {
  rMin: 8, rSpan: 4, sigma: 2.548, cut: 4, rise: 0.1, cutFrom: 2.5, cutTo: 3, saltR: 0x4853, saltPhi: 0x4850,
  // scripts/calibrate-hotspot.ts: Sgr A*, r_c = 10, mean depth, peak of L -> +0.3 Jy at 229 GHz (Wielgus et al. 2022 S3.1);
  // recalibrated 2026-10-08 for the R-beta electrons (was 9.22: the midplane electrons are now ~9x colder, 2.8e9 vs 2.6e10 K at r 10, and emit far less)
  A0: 692,
  // M: half-width of the live-tracing window; bounds |light-travel delay| of every ray through the blob region
  // (tests/sweep-hotspot.test.ts, 48^2: Sgr A* 77.9, Gargantua 68.8, M87* 67.2; the Custom extremes a = 0.998 at i = 1 and
  // 89 deg reach 91.9 and 94.5. Longer delays are higher-order ring images, ~e^-pi fainter per half orbit, and at the
  // window's edges the light curve is ~0, so 100 stands)
  pad: 100,
} as const;
/** CPU twin of ?hotflow's hotspot measurement (scripts/calibrate-hotspot.ts, 96^2): Sgr A* at the peak of eruption 0
 *  (t = 846.25 M), flux the hotspot adds (Jy) and its centroid (alpha, beta in M), with light-travel delay on and off. */
export const HOTSPOT_TWIN = {
  delay: { jy: 0.2019, cx: -0.221, cy: 0.537 },
  instant: { jy: 0.8440, cx: 0.753, cy: -0.474 },
} as const;
/** No blob reaches beyond this radius (r_c max + the 4 sigma cut). */
export const HOTSPOT_REACH = HOTSPOT.rMin + HOTSPOT.rSpan + HOTSPOT.cut * HOTSPOT.sigma;

const sstep = (e0: number, e1: number, x: number) => { const u = Math.min(1, Math.max(0, (x - e0) / (e1 - e0))); return u * u * (3 - 2 * u); };
/** Light curve: rise over 0.1 P, e-fold in P, smooth cut to 0 between 2.5 P and 3 P. */
export function hotspotLight(tau: number, P: number): number {
  if (tau < 0 || tau >= HOTSPOT.cutTo * P) return 0;
  return sstep(0, HOTSPOT.rise * P, tau) * Math.exp(-tau / P) * (1 - sstep(HOTSPOT.cutFrom * P, HOTSPOT.cutTo * P, tau));
}
/** max_tau L and its tau / P (independent of P): the calibration's "peak of L". */
export const [HOTSPOT_LPEAK, HOTSPOT_TPEAK] = (() => {
  let m = 0, at = 0; for (let i = 0; i <= 20000; i++) { const x = i * 1e-5, v = hotspotLight(x, 1); if (v > m) { m = v; at = x; } }
  return [m, at];
})();
export function hotspotRadius(k: number): number { return HOTSPOT.rMin + HOTSPOT.rSpan * (fluxHash(k, HOTSPOT.saltR) + 0.5); }
export function hotspotPeriod(rc: number, a: number): number { return 2 * Math.PI * (rc ** 1.5 + a); }
export interface HotspotState { alive: boolean; k: number; rc: number; phiC: number; amp: number; Om: number }
/** The hotspot at absolute time t: that of the latest eruption k (t_k <= t), alive while its light curve is on.
 *  Lifetimes (3P <= 802 M) are shorter than the shortest eruption gap (1000 M), so at most one is alive. */
export function hotspotAt(t: number, s: number, a: number, A0: number = HOTSPOT.A0): HotspotState {
  let k = Math.floor(t / FLUX.T); if (eruptionTime(k) > t) k--;
  const tau = t - eruptionTime(k), rc = hotspotRadius(k), q = rc ** 1.5 + a, P = 2 * Math.PI * q;
  const phiC = 2 * Math.PI * (fluxHash(k, HOTSPOT.saltPhi) + 0.5) + tau / q;
  const alive = s > 0 && tau < HOTSPOT.cutTo * P;
  const amp = alive ? A0 * s * (1 + FLUX.spread * fluxHash(k, FLUX.saltD)) * hotspotLight(tau, P) : 0;
  return { alive, k, rc, phiC, amp, Om: 1 / q };
}
/** Gaussian profile G(d) about (r_c, pi/2, phi_c), truncated at 4 sigma (twin: hotspotBoostJ). */
export function hotspotBoost(r: number, th: number, ph: number, rc: number, phiC: number): number {
  const d2 = r * r + rc * rc - 2 * r * rc * Math.sin(th) * Math.cos(ph - phiC), sg = HOTSPOT.sigma;
  if (d2 >= (HOTSPOT.cut * sg) ** 2) return 0;
  return Math.exp(-d2 / (2 * sg * sg));
}
/** nu_plasma / nu_obs of matter at (r, th) moving at Omega (no radial motion), for the camera-normalised past-directed
 *  conserved momenta p_t, p_phi; null where that motion is not timelike (twin: hotspotShiftJ). */
export function hotspotShift(r: number, th: number, pt: number, pphi: number, a: number, Om: number): number | null {
  const g = metricLower(r, th, a), K = -(g.tt + 2 * Om * g.tphi + Om * Om * g.phph);
  if (K <= 0) return null;
  return (pt + Om * pphi) / Math.sqrt(K);
}
export function hotspotPeakTime(k: number, a: number): number { return eruptionTime(k) + HOTSPOT_TPEAK * hotspotPeriod(hotspotRadius(k), a); }
export function hotspotEndTime(k: number, a: number): number { return eruptionTime(k) + HOTSPOT.cutTo * hotspotPeriod(hotspotRadius(k), a); }
/** Is some hotspot alive at a time within [t - pad, t + pad]? main.ts traces live while it is (spec 3). */
export function hotspotAliveWindow(t: number, s: number, a: number, pad: number = HOTSPOT.pad): boolean {
  if (s <= 0) return false;
  for (let k = Math.floor((t - pad) / FLUX.T) - 1; k <= Math.floor((t + pad) / FLUX.T); k++)
    if (eruptionTime(k) < t + pad && hotspotEndTime(k, a) > t - pad) return true;
  return false;
}
