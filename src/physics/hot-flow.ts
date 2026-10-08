// Hot accretion flow for the 1.3 mm (EHT) view (spec 2026-10-04; electrons spec 2026-10-08): the semi-analytic RIAF of
// Broderick et al. 2011/2016 (n_e ~ r^-1.1 e^{-z^2/2 rho^2}) with magnetically-arrested R-beta electron heating
// (Moscibrodzka et al. 2016, as in the EHT's GRMHD libraries): virial ions, a field set by the midplane density at
// midplane beta 1, T_e = T_i / R(beta) with R_high 160, R_low 1. Sub-Keplerian dynamics of Pu, Akiyama & Asada 2016
// (Keplerian / zero-angular-momentum free fall mixed 50/50 in u^r and Omega; equatorial profiles of r, u^t from the
// local metric), thermal synchrotron (Mahadevan et al. 1996 isotropic fit) with Kirchhoff absorption. n0 calibrated
// to the measured 230 GHz flux (scripts/calibrate-hotflow.ts). WGSL twin in emission-shared.wgsl.
import { metricLower, metricUpper } from "./kerr";
import { iscoRadius } from "./orbits";
import { PRESETS } from "./presets";

export const HOTFLOW = { nu: 230e9, betaEq: 1, rHigh: 160, rLow: 1, rMax: 50, dl: 0.25, lambdaHot: 0.01, kTb: 6.1528e13 } as const;
const C = 2.99792458e10, QE = 4.80320471e-10, ME = 9.1093837e-28, MP = 1.67262192e-24, KB = 1.380649e-16;

export function mahadevanM(X: number): number {
  return 4.0505 * X ** (-1 / 6) * (1 + 0.4 * X ** -0.25 + 0.5316 * X ** -0.5) * Math.exp(-1.8899 * Math.cbrt(X));
}
/** Midplane density n0 (r / 2M)^-1.1: the Broderick profile without its Gaussian, and the MAD field's reference. */
export function flowDensityEq(r: number, n0: number): number { return n0 * (r / 2) ** -1.1; }
export function flowDensity(r: number, th: number, n0: number): number {
  const z = r * Math.cos(th), rho = r * Math.sin(th);
  if (rho < 1e-6) return 0;
  return flowDensityEq(r, n0) * Math.exp(-(z * z) / (2 * rho * rho));
}
/** Virial ion temperature: k T_i = m_p c^2 / (3 r), r in M. */
export function flowIonTemperature(r: number): number { return (MP * C * C) / (3 * KB * r); }
/** Ion plasma beta 8 pi n k T_i / B^2 with the field set by the midplane density: 2 beta_eq e^{-z^2 / 2 rho^2}. 0 on the axis. */
export function flowBeta(r: number, th: number): number {
  const z = r * Math.cos(th), rho = r * Math.sin(th);
  if (rho < 1e-6) return 0;
  return 2 * HOTFLOW.betaEq * Math.exp(-(z * z) / (2 * rho * rho));
}
/** T_i / T_e = (R_high beta^2 + R_low) / (1 + beta^2) (Moscibrodzka et al. 2016). */
export function flowRatio(beta: number): number { const b2 = beta * beta; return (HOTFLOW.rHigh * b2 + HOTFLOW.rLow) / (1 + b2); }
export function flowTemperature(r: number, th: number): number { return flowIonTemperature(r) / flowRatio(flowBeta(r, th)); }
/** Magnetically arrested field from the midplane density: B^2 / 8 pi = n_eq m_p c^2 r_S / (12 r beta_eq), r_S = 2 M. */
export function flowField(r: number, n0: number): number { return Math.sqrt((8 * Math.PI * flowDensityEq(r, n0) * MP * C * C * (2 / r)) / (12 * HOTFLOW.betaEq)); }
/** Energy and angular momentum per unit mass of the prograde ISCO orbit (M = 1). */
export function iscoEL(a: number): [number, number] {
  const r = iscoRadius(a, true), sq = Math.sqrt(r), den = r ** 0.75 * Math.sqrt(r * sq - 3 * sq + 2 * a);
  return [(r * sq - 2 * sq + a) / den, (r * r - 2 * a * sq + a * a) / den];
}
/** Pu et al. 2016 eqs. 1-3: u^r and Omega mixed between Keplerian (plunging inside the ISCO) and zero-angular-momentum
 *  free fall, as functions of r on the equatorial metric; u^t from the local metric. null where K0 <= 0. */
export function flowVelocity(r: number, th: number, a: number, kUr = 0.5, kOm = 0.5): { ut: number; ur: number; Om: number } | null {
  const gu = metricUpper(r, Math.PI / 2, a), risco = iscoRadius(a, true);
  let urK = 0, OmK = 1 / (r ** 1.5 + a);
  if (r < risco) {
    const [E, L] = iscoEL(a);
    const uT = -gu.tt * E + gu.tphi * L, uP = -gu.tphi * E + gu.phph * L;
    const rest = -1 - (gu.tt * E * E - 2 * gu.tphi * E * L + gu.phph * L * L);
    urK = -Math.sqrt(Math.max(0, rest * gu.rr)); OmK = uP / uT;
  }
  const urFF = -Math.sqrt(Math.max(0, gu.rr * (-1 - gu.tt))), OmFF = gu.tphi / gu.tt;
  // Pu eq. 2/3 written with weights on the Keplerian part: kUr = alpha, kOm = beta (1 = Keplerian, 0 = free fall)
  const ur = urK + (1 - kUr) * (urFF - urK), Om = OmK + (1 - kOm) * (OmFF - OmK);
  const g = metricLower(r, th, a), K0 = -(g.tt + 2 * Om * g.tphi + Om * Om * g.phph);
  if (K0 <= 0) return null;
  return { ut: Math.sqrt((1 + g.rr * ur * ur) / K0), ur, Om };
}
/** nu_plasma / nu_observed for the camera-normalised, past-directed covariant momentum of state s (t, r, th, phi, p...). */
export function flowShift(s: Float64Array, a: number): number | null {
  const u = flowVelocity(s[1], s[2], a); if (!u) return null;
  return u.ut * s[4] + u.ur * s[5] + u.Om * u.ut * s[7];
}
/** Plasma-frame thermal synchrotron j_nu (erg s^-1 cm^-3 Hz^-1 sr^-1) and alpha_nu (cm^-1) at nu. */
export function flowCoeffs(r: number, th: number, nu: number, n0: number): [number, number] {
  const n = flowDensity(r, th, n0); if (n <= 0) return [0, 0];
  const T = flowTemperature(r, th), the = (KB * T) / (ME * C * C), B = flowField(r, n0);
  const nuc = (QE * B) / (2 * Math.PI * ME * C), X = (2 * nu) / (3 * nuc * the * the);
  const j = ((n * QE * QE * nu) / (2 * Math.sqrt(3) * C * the * the)) * mahadevanM(X);
  return [j, j / ((2 * nu * nu * KB * T) / (C * C))];
}
export function isHotFlow(lambda: number): boolean { return lambda < HOTFLOW.lambdaHot; }
/** n0 calibrated to the measured 230 GHz flux (scripts/calibrate-hotflow.ts). Any object scales from the calibrated
 *  object nearest in log mass as density ~ Mdot / (r_g^2 c) ~ lambda / M, so a calibrated object's own mass and accretion
 *  give its calibrated n0 exactly, whatever the selector says (a spin or inclination nudge, a Custom link). */
export const HOTFLOW_N0: Record<string, number> = { sgra: 1.43e7, m87: 1.14e5 }; // from scripts/calibrate-hotflow.ts (96^2, flux-matched)
export function flowN0(mSun: number, lambda: number): number {
  let best = PRESETS[0], dBest = Infinity;
  for (const p of PRESETS) {
    if (HOTFLOW_N0[p.id] === undefined) continue;
    const d = Math.abs(Math.log(mSun / p.massSun)); if (d < dBest) { dBest = d; best = p; }
  }
  return HOTFLOW_N0[best.id] * (lambda / best.lambda) * (best.massSun / mSun);
}
