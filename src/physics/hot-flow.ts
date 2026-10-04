// Hot accretion flow for the 1.3 mm (EHT) view (spec 2026-10-04): the semi-analytic RIAF of Broderick et al. 2011/2016
// (n_e ~ r^-1.1 e^{-z^2/2 rho^2}, T_e ~ r^-0.84, toroidal field at beta = 10) with the sub-Keplerian dynamics of
// Pu, Akiyama & Asada 2016 (Keplerian / zero-angular-momentum free fall mixed 50/50 in u^r and Omega; equatorial
// profiles of r, u^t from the local metric), thermal synchrotron (Mahadevan et al. 1996 isotropic fit) with
// Kirchhoff absorption. n0 calibrated to the measured 230 GHz flux (scripts/calibrate-hotflow.ts). WGSL twin in
// emission-shared.wgsl.
import { metricLower, metricUpper } from "./kerr";
import { iscoRadius } from "./orbits";
import { PRESETS } from "./presets";

export const HOTFLOW = { nu: 230e9, T0: 1e11, beta: 10, rMax: 50, dl: 0.25, lambdaHot: 0.01, kTb: 6.1528e13 } as const;
const C = 2.99792458e10, QE = 4.80320471e-10, ME = 9.1093837e-28, MP = 1.67262192e-24, KB = 1.380649e-16;

export function mahadevanM(X: number): number {
  return 4.0505 * X ** (-1 / 6) * (1 + 0.4 * X ** -0.25 + 0.5316 * X ** -0.5) * Math.exp(-1.8899 * Math.cbrt(X));
}
export function flowDensity(r: number, th: number, n0: number): number {
  const z = r * Math.cos(th), rho = r * Math.sin(th);
  if (rho < 1e-6) return 0;
  return n0 * (r / 2) ** -1.1 * Math.exp(-(z * z) / (2 * rho * rho));
}
export function flowTemperature(r: number): number { return HOTFLOW.T0 * (r / 2) ** -0.84; }
/** B^2 / 8 pi = n m_p c^2 r_S / (12 r beta), r_S = 2 M (Broderick et al. 2011). */
export function flowField(r: number, n: number): number { return Math.sqrt((8 * Math.PI * n * MP * C * C * (2 / r)) / (12 * HOTFLOW.beta)); }
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
  const T = flowTemperature(r), the = (KB * T) / (ME * C * C), B = flowField(r, n);
  const nuc = (QE * B) / (2 * Math.PI * ME * C), X = (2 * nu) / (3 * nuc * the * the);
  const j = ((n * QE * QE * nu) / (2 * Math.sqrt(3) * C * the * the)) * mahadevanM(X);
  return [j, j / ((2 * nu * nu * KB * T) / (C * C))];
}
export function isHotFlow(lambda: number): boolean { return lambda < HOTFLOW.lambdaHot; }
/** n0 calibrated to the measured 230 GHz flux (scripts/calibrate-hotflow.ts); other objects scale from Sgr A*'s as
 *  density ~ Mdot / (r_g^2 c) ~ lambda / M. */
export const HOTFLOW_N0: Record<string, number> = { sgra: 1.5e7, m87: 5.03e5 }; // from scripts/calibrate-hotflow.ts (96^2, flux-matched)
export function flowN0(mSun: number, lambda: number, presetId?: string): number {
  if (presetId && HOTFLOW_N0[presetId] !== undefined) return HOTFLOW_N0[presetId];
  const sg = PRESETS.find((p) => p.id === "sgra")!;
  return HOTFLOW_N0.sgra * (lambda / sg.lambda) * (sg.massSun / mSun);
}
