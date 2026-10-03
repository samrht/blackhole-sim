// Eruption flares (spec 2026-10-04): each horizon-flux eruption ejects a reconnection-heated flux tube that spirals
// out to r_c (5-30 r_g), orbits at the Keplerian rate for up to two orbits and shines as optically thin synchrotron
// (Porth et al. 2021; Ripperda et al. 2022; GRAVITY). Field by flux conservation; radiates zeta x its field energy,
// zeta fixed so a default Sgr A* tube injects a typical NIR flare's 1e38 erg (Yusef-Zadeh et al. 2006). WGSL twins
// in emission-shared.wgsl.
import { FLUX, fluxHash, eruptionTime, eruptionDepth } from "./flux-history";
import { GL6 } from "./jet";
import { iscoRadius } from "./orbits";
import { jetEnergetics, C_CGS } from "./synchrotron";
import { PRESETS } from "./presets";

export const FLARE = { rMin: 5, rMax: 30, sizeFrac: 0.2, lifeOrbits: 2, saltR: 0x5243, saltPhi: 0x5048, anchorErg: 1e38, cA: 0.48452046 } as const;
export interface Tube { k: number; t0: number; D: number; depth: number; rc: number; R: number; phi0: number }
const omegaK = (r: number, a: number) => 1 / (r ** 1.5 + a);
const smooth = (x: number) => { const c = Math.min(1, Math.max(0, x)); return c * c * (3 - 2 * c); };

export function tubeOf(k: number, a: number): Tube {
  const depth = eruptionDepth(k, FLUX.dbar), D = -FLUX.tauD * Math.log(1 - depth);
  const u = fluxHash(k, FLARE.saltR) + 0.5, v = fluxHash(k, FLARE.saltPhi) + 0.5;
  const rc = Math.max(FLARE.rMin + (FLARE.rMax - FLARE.rMin) * u, iscoRadius(a, true) / (1 - 2 * FLARE.sizeFrac));
  return { k, t0: eruptionTime(k), D, depth, rc, R: FLARE.sizeFrac * rc, phi0: 2 * Math.PI * v };
}
export const tubeOrbit = (t: Tube, a: number) => 2 * Math.PI / omegaK(t.rc, a);
export const tubeLife = (t: Tube, a: number) => t.D + FLARE.lifeOrbits * tubeOrbit(t, a);
/** Azimuth gained spiralling from r_in to r (Omega_K integrated along r(tau) linear in tau; GL6). */
function spiralPhase(r: number, rin: number, t: Tube, a: number): number {
  if (r <= rin) return 0;
  const h = 0.5 * (r - rin), m = 0.5 * (r + rin); let s = 0;
  for (const [x, w] of GL6) s += w * omegaK(m + h * x, a);
  return (t.D / (t.rc - rin)) * s * h;
}
export function tubeAt(t: Tube, time: number, a: number): { r: number; phi: number; A: number } {
  const tau = time - t.t0, rin = iscoRadius(a, true);
  if (tau < 0 || tau >= tubeLife(t, a)) return { r: t.rc, phi: t.phi0, A: 0 };
  if (tau < t.D) { const r = rin + (t.rc - rin) * tau / t.D; return { r, phi: t.phi0 + spiralPhase(r, rin, t, a), A: tau / t.D }; }
  const tc = tau - t.D, P = tubeOrbit(t, a);
  return { r: t.rc, phi: t.phi0 + spiralPhase(t.rc, rin, t, a) + omegaK(t.rc, a) * tc, A: Math.exp(-2 * tc / P) * (1 - smooth((tc / P - 1.5) / 0.5)) };
}
/** Top-hat field through the tube cross-section pi R^2 (flux conservation): B = delta s Phi / (pi R^2 r_g^2). */
export function tubeField(t: Tube, s: number, phi: number, rgCm: number): number {
  return (t.depth * s * phi) / (Math.PI * (t.R * rgCm) ** 2);
}
export function tubeEnergy(t: Tube, s: number, phi: number, rgCm: number): number {
  return ((t.depth * s) ** 2 * phi * phi) / (4 * Math.PI ** 2 * t.R * rgCm);
}
export function tubePower(t: Tube, time: number, a: number, s: number, f: number, phi: number, zeta: number, rgCm: number): number {
  if (f === 0 || s === 0) return 0;
  const A = tubeAt(t, time, a).A; if (A === 0) return 0;
  const norm = (t.D / 2 + tubeOrbit(t, a) * FLARE.cA) * rgCm / C_CGS; // integral of A in seconds
  return (zeta * f * tubeEnergy(t, s, phi, rgCm) * A) / norm;
}
export function meanTubeEnergy(s: number, phi: number, rgCm: number): number {
  return ((FLUX.dbar * s) ** 2 * (1 + 1 / 48) * phi * phi * (Math.log(6) / 25)) / (0.8 * Math.PI ** 2 * rgCm);
}
let zetaCache = 0;
/** zeta = anchor / <E> at the Sgr A* preset (s = 1). */
export function flareZeta(): number {
  if (!zetaCache) { const p = PRESETS.find((q) => q.id === "sgra")!, E = jetEnergetics(p.massSun, p.a, p.lambda);
    zetaCache = FLARE.anchorErg / meanTubeEnergy(1, E.phi, E.rgCm); }
  return zetaCache;
}
