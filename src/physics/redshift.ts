import { metricLower } from "./kerr";
import { omegaKepler } from "./orbits";

/** General g = sqrt(-(g_tt + 2Ω g_tφ + Ω² g_φφ)) / (1 - Ω ξ), emitter at equator, observer at infinity. */
export function gFactor(r: number, a: number, xi: number, Omega: number): number {
  const g = metricLower(r, Math.PI / 2, a);
  const rad = -(g.tt + 2 * Omega * g.tphi + Omega * Omega * g.phph);
  return Math.sqrt(Math.max(0, rad)) / (1 - Omega * xi);
}
/** Convenience for a Keplerian disk element. */
export function gFactorKepler(r: number, a: number, xi: number, prograde = true): number {
  return gFactor(r, a, xi, omegaKepler(r, a, prograde));
}
