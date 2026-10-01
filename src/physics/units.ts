import { peakFluxShape } from "./disk";
import { efficiency, iscoRadius } from "./orbits";

// SI constants (CODATA 2018; IAU 2015 nominal solar mass; Julian year).
export const G = 6.6743e-11;
export const C = 2.99792458e8;
export const SIGMA_SB = 5.670374419e-8;
export const M_SUN = 1.98847e30;
export const YEAR = 3.15576e7;
const AU = 1.495978707e11;
/** Eddington luminosity per solar mass, electron scattering in ionised hydrogen (W). */
export const L_EDD_PER_MSUN = 1.2572e31;

/** r_g = G M / c^2 (m): the renderer's unit of length, "M". */
export const gravRadius = (mSun: number) => (G * mSun * M_SUN) / (C * C);
/** t_g = G M / c^3 (s): the renderer's unit of time. */
export const gravTime = (mSun: number) => (G * mSun * M_SUN) / (C * C * C);
export const eddingtonLuminosity = (mSun: number) => L_EDD_PER_MSUN * mSun;

/** Accretion rate (kg/s) of a thin disk radiating lambda L_Edd: Mdot = lambda L_Edd / (eta(a) c^2),
 *  eta = 1 - E_ISCO (spec 2026-10-01 §2.1). */
export const mdotFromLambda = (mSun: number, a: number, lambda: number) =>
  (lambda * eddingtonLuminosity(mSun)) / (efficiency(a, true) * C * C);
/** lambda for an accretion rate given in solar masses per year. */
export const lambdaFromMdot = (mSun: number, a: number, mdotSunPerYear: number) =>
  ((mdotSunPerYear * M_SUN) / YEAR) * efficiency(a, true) * C * C / eddingtonLuminosity(mSun);

/** Peak effective temperature (K) of the Novikov-Thorne disk, one face:
 *  F = Mdot c^2 shape_peak / (4 pi r_g^2), T = (F / sigma)^(1/4). pageThorneFluxShape is the flux
 *  for Mdot = M = 1 in geometric units, so this is the same profile temperatureShape normalises. */
export function peakTemperature(mSun: number, a: number, lambda: number): number {
  const rg = gravRadius(mSun);
  const flux = (mdotFromLambda(mSun, a, lambda) * C * C * peakFluxShape(a, true)) / (4 * Math.PI * rg * rg);
  return Math.pow(flux / SIGMA_SB, 0.25);
}
/** The lambda that gives a disk peak temperature tK (T_peak scales as lambda^(1/4)). */
export const lambdaForPeakTemperature = (mSun: number, a: number, tK: number) =>
  Math.pow(tK / peakTemperature(mSun, a, 1), 4);

/** Coordinate-time period of the prograde circular orbit at the ISCO: 2 pi (r^(3/2) + a) t_g. */
export const iscoPeriod = (mSun: number, a: number) =>
  2 * Math.PI * (Math.pow(iscoRadius(a, true), 1.5) + a) * gravTime(mSun);

const sig3 = (v: number) => v.toLocaleString("en-US", { maximumSignificantDigits: 3 });
export function formatLength(m: number): string {
  if (m < 1e3) return `${sig3(m)} m`;
  if (m < 0.1 * AU) return `${sig3(m / 1e3)} km`;
  return `${sig3(m / AU)} AU`;
}
export function formatDuration(s: number): string {
  const ladder: [number, string][] = [[1e-6, "µs"], [1e-3, "ms"], [1, "s"], [60, "min"], [3600, "h"], [86400, "days"], [YEAR, "yr"]];
  let unit = ladder[0];
  for (const step of ladder) if (s >= step[0]) unit = step;
  return `${sig3(s / unit[0])} ${unit[1]}`;
}
