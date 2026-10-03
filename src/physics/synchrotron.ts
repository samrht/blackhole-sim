// Synchrotron jet physics (specs 2026-10-02 synchrotron-jet and cooled-jet). CPU twin of the jet code in
// src/render/emission-shared.wgsl, plus what the shader cannot do (the energy budget). The coefficients come
// from the cooled population's exact cyclo-synchrotron table (cyclosynch.ts). Gaussian cgs; lengths in r_g.

import { metricUpper, metricLower, horizonOuter } from "./kerr";
import { fluxMoment } from "./flux-history";
import { I_G, lookup, type SynchTable } from "./cyclosynch";
import { mdotFromLambda } from "./units";
import { funnelEdge, wallProfile, lengthFalloff, gammaProfile, JET } from "./jet";
import { cieX, cieY, cieZ } from "./color";
import { VIS_LREF } from "./lookups";
export { gammaProfile, GAMMA_REF_Z, GAMMA_SLOPE } from "./jet";

export const Q_E = 4.80320471e-10, M_E = 9.1093837e-28, C_CGS = 2.99792458e10, G_CGS = 6.6743e-8, MSUN_G = 1.98847e33;
/** Magnetically arrested horizon flux (Gaussian units, Tchekhovskoy et al. 2011) and BZ coefficient. */
export const PHI_MAD = 50, KAPPA_BZ = 0.05;

/** |B| (G) at cylindrical (rho, z) in r_g: poloidal flux conservation in the funnel, B_p = b0 / rho_f^2, plus
 *  the force-free toroidal field B_phi = B_p rho Omega_F / c with Omega_F = Omega_H / 2 = a / (4 r_H). */
export function jetField(rho: number, z: number, a: number, b0: number): number {
  const rf = funnelEdge(z), bp = b0 / (rf * rf), w = (rho * a) / (4 * horizonOuter(a));
  return bp * Math.sqrt(1 + w * w);
}
/** Unit flow direction (n_r, n_theta) at BL (r, theta): outward along the self-similar funnel family
 *  rho = q rho_f(z) through the point. */
export function streamlineDir(r: number, th: number): [number, number] {
  const z = r * Math.cos(th), rho = r * Math.sin(th), az = Math.max(Math.abs(z), 1e-6);
  const vr = (rho / funnelEdge(z)) * (JET.slope / (2 * Math.sqrt(az))), vz = z >= 0 ? 1 : -1;
  const n = Math.hypot(vr, vz), ur = vr / n, uz = vz / n;
  return [ur * Math.sin(th) + uz * Math.cos(th), ur * Math.cos(th) - uz * Math.sin(th)];
}
/** nu_plasma / nu_observed = p.u for the camera-normalised, past-directed photon state s, with the plasma
 *  moving at Lorentz factor gamma along (nR, nTh) in the zero-angular-momentum observer's frame. */
export function plasmaShift(s: Float64Array, a: number, gamma: number, nR: number, nTh: number): number {
  const g = metricUpper(s[1], s[2], a);
  const alpha = Math.sqrt(-1 / g.tt), omega = g.tphi / g.tt, beta = Math.sqrt(Math.max(0, 1 - 1 / (gamma * gamma)));
  return gamma * ((s[4] + omega * s[7]) / alpha + beta * (nR * Math.sqrt(g.rr) * s[5] + nTh * Math.sqrt(g.thth) * s[6]));
}
export const SIGMA_T = 6.6524587e-25;
/** r_g, Mdot, horizon flux, Blandford-Znajek power with the high-spin correction f(Omega_H) (Tchekhovskoy,
 *  Narayan & McKinney 2010), b0 = Phi / (pi r_g^2). */
export function jetEnergetics(mSun: number, a: number, lambda: number) {
  const rgCm = (G_CGS * mSun * MSUN_G) / C_CGS ** 2, mdot = mdotFromLambda(mSun, a, lambda) * 1e3;
  const phi = PHI_MAD * Math.sqrt(mdot * C_CGS) * rgCm, w = a / (2 * horizonOuter(a)); // Omega_H r_g / c
  const fH = 1 + 1.38 * w * w - 9.2 * w ** 4;
  const pBZ = (KAPPA_BZ / (4 * Math.PI * C_CGS)) * phi * phi * (w * C_CGS / rgCm) ** 2 * fH;
  return { rgCm, mdot, phi, pBZ, b0: phi / (Math.PI * rgCm * rgCm), fH };
}
/** Plasma proper time (s) since leaving the base z_b: Int dz r_g / (c Gamma beta), Gamma beta = U0 (|z|/280)^0.58. */
export function flowTime(z: number, g280: number, rgCm: number): number {
  const U0 = Math.sqrt(Math.max(1e-12, g280 * g280 - 1)), az = Math.max(Math.abs(z), JET.zBase);
  return (rgCm / C_CGS) * Math.pow(280, 0.58) / U0 * (Math.pow(az, 0.42) - Math.pow(JET.zBase, 0.42)) / 0.42;
}
/** Energy at infinity per flat volume element: lapse^2 Gamma sqrt(g_rr g_thth g_phph) / (r^2 sin th) (spec 2.4). */
export function volumeWeight(r: number, th: number, a: number, g280: number): number {
  const gl = metricLower(r, th, a), lapse2 = -1 / metricUpper(r, th, a).tt;
  return lapse2 * gammaProfile(r * Math.cos(th), g280) * Math.sqrt(gl.rr * gl.thth * gl.phph) / (r * r * Math.sin(th));
}
/** Sum of f(rho, z) weighted by volumeWeight over the rendered region (both lobes), in r_g^3, out to
 *  rho = qMax rho_f(z) (1.2: the wall's cut; wider for a flux-widened jet). */
function regionSum(a: number, jetLength: number, g280: number, f: (rho: number, z: number) => number, qMax = 1.2): number {
  const NZ = 160, NR = 48, zb = JET.zBase; let L = 0;
  for (let iz = 0; iz < NZ; iz++) {
    const dz = (jetLength - zb) / NZ, z = zb + dz * (iz + 0.5), dr = (qMax * funnelEdge(z)) / NR;
    for (let ir = 0; ir < NR; ir++) {
      const rho = dr * (ir + 0.5), v = f(rho, z); if (v === 0) continue;
      L += 2 * 2 * Math.PI * rho * dr * dz * volumeWeight(Math.hypot(rho, z), Math.atan2(rho, z), a, g280) * v;
    }
  }
  return L;
}
/** Injected power at infinity for q0 = 1 with the whole jet at flux ratio f (wall sqrt(f) wider, density x f;
 *  filaments are mean-one and drop out). f = 1 is the steady jet. */
export function injUnit(a: number, b0: number, rgCm: number, jetLength: number, g280: number, f = 1): number {
  const sw = Math.sqrt(f);
  return regionSum(a, jetLength, g280, (rho, z) => { const sh = f * wallProfile(rho / sw, z) * lengthFalloff(z, jetLength);
    if (sh <= 0) return 0; const B = jetField(rho, z, a, b0); return B * B * sh; }, 1.2 * sw) * rgCm ** 3;
}
/** Time average of the injected power over the flux distribution at slider s (spec corrections 2): injUnit is
 *  ~ f^2 (A + C f) (B_phi grows with rho), so the exact cubic through f = 0.5, 1, 1.5, 2 is averaged with the
 *  generator's moments <f> = 1, <f^2>, <f^3>. s = 0: the steady jet. */
export function meanInjection(a: number, b0: number, rgCm: number, jetLength: number, g280: number, fluxVar: number): number {
  if (fluxVar === 0) return injUnit(a, b0, rgCm, jetLength, g280, 1);
  const F = [0.5, 1, 1.5, 2], P = F.map((f) => injUnit(a, b0, rgCm, jetLength, g280, f));
  // Power-basis coefficients c0..c3 of the cubic through (F, P): solve the 4x4 Vandermonde system.
  const M = F.map((f, i) => [1, f, f * f, f * f * f, P[i]]);
  for (let c = 0; c < 4; c++) {
    for (let r = c + 1; r < 4; r++) { const k = M[r][c] / M[c][c]; for (let j = c; j < 5; j++) M[r][j] -= k * M[c][j]; }
  }
  const co = [0, 0, 0, 0];
  for (let r = 3; r >= 0; r--) { let s = M[r][4]; for (let j = r + 1; j < 4; j++) s -= M[r][j] * co[j]; co[r] = s / M[r][r]; }
  return co[0] + co[1] + co[2] * fluxMoment(2, fluxVar) + co[3] * fluxMoment(3, fluxVar);
}
// j' = C_J q0 shape B J^(x, s), alpha' = C_A q0 shape J^... (spec 2.3): logs of the prefactors, nu_B / B, k / B^2,
// and of the flow-time constant 280^0.58 / (0.42 c).
export const LN_CJ = Math.log((3 * Q_E ** 3) / (2 * I_G * SIGMA_T * M_E * C_CGS ** 3));
export const LN_CA = Math.log((3 * Math.PI ** 2 * Q_E) / (C_CGS * I_G * SIGMA_T));
export const LN_NUB0 = Math.log(Q_E / (2 * Math.PI * M_E * C_CGS));
export const LN_K0 = Math.log(SIGMA_T / (6 * Math.PI * M_E * C_CGS));
export const LN_T0 = 0.58 * Math.log(280) - Math.log(C_CGS) - Math.log(0.42);
/** Plasma-frame j' (erg s^-1 cm^-3 Hz^-1 sr^-1) and alpha' (cm^-1) at nu' (twin of synchSampleJ). */
export function jetCoeffs(t: SynchTable, nuP: number, B: number, s: number, q0: number, shape: number): [number, number] {
  const lnB = Math.log(B), lnx = Math.log(nuP) - LN_NUB0 - lnB, [lJ, lA] = lookup(t, lnx, Math.max(Math.log(s), -30));
  const base = Math.log(q0) + Math.log(shape);
  return [Math.exp(LN_CJ + base + lnB + lJ), Math.exp(LN_CA + base - lnB - 2 * lnx + lA)];
}
/** Comoving, optically thin nu L_nu (energy at infinity) of the rendered jet at q0. */
export function nuLnuAt(t: SynchTable, nu: number, a: number, b0: number, rgCm: number, jetLength: number, g280: number, q0: number): number {
  return 4 * Math.PI * nu * rgCm ** 3 * regionSum(a, jetLength, g280, (rho, z) => {
    const sh = wallProfile(rho, z) * lengthFalloff(z, jetLength); if (sh <= 0) return 0;
    const B = jetField(rho, z, a, b0), s = Math.exp(LN_K0) * B * B * flowTime(z, g280, rgCm);
    return jetCoeffs(t, nu, B, s, q0, sh)[0];
  });
}
/** Default eta: M87*'s nu L_nu(550 nm) = 1e41 erg/s (spec 2.4; the anchor test re-derives it). */
export const ETA_DEFAULT = 0.05527;
/** The jet's uniforms: q0 = eta P_BZ / (time-averaged injected power at q0 = 1); 0 at spin 0 (no BZ power). */
export function jetUniforms(mSun: number, a: number, lambda: number, eta: number, jetLength: number, g280: number, fluxVar: number) {
  const E = jetEnergetics(mSun, a, lambda);
  return { jetB0: E.b0, jetQ0: E.pBZ > 0 ? (eta * E.pBZ) / meanInjection(a, E.b0, E.rgCm, jetLength, g280, fluxVar) : 0, rgCm: E.rgCm, pBZ: E.pBZ };
}
/** One band of one jet sample: exact solution across a uniform slab of path ds (cm) behind optical depth
 *  tau already accumulated from the camera. Twin of jetSlabJ in emission-shared.wgsl. */
export function slabStep(I: number, tau: number, j: number, alpha: number, ds: number): [number, number] {
  const dt = alpha * ds, fac = dt < 1e-4 ? 1 - 0.5 * dt : (1 - Math.exp(-dt)) / dt;
  return [I + j * ds * fac * Math.exp(-tau), Math.min(tau + dt, 1e30)];
}

/** Colour bands (nm): 450, 550, 650; I_nu is taken piecewise constant over 360-500 / 500-600 / 600-830 nm. */
export const JET_BANDS_NM = [450, 550, 650] as const;
/** 3x3 matrix (row R,G,B; column band) from I_nu (cgs) per band to linear sRGB in the disk's units
 *  (blackbodyVisibleRGB / VIS_LREF). blackbodyVisibleRGB sums planck_m(lambda) = lambda_m^-5 / (e^x - 1) over
 *  5 nm bins; the cgs B_lambda = 2 h c^2 1e-10 planck_m, so an I_lambda = I_nu c / lambda^2 maps the same way. */
export function jetBandMatrix(): number[][] {
  const H_CGS = 6.62607015e-27, xyz = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let nm = 360; nm <= 830; nm += 5) {
    const b = nm < 500 ? 0 : nm < 600 ? 1 : 2, lcm = nm * 1e-7, f = C_CGS / (lcm * lcm) / (2 * H_CGS * C_CGS ** 2 * 1e-10);
    xyz[0][b] += cieX(nm) * f; xyz[1][b] += cieY(nm) * f; xyz[2][b] += cieZ(nm) * f;
  }
  const T = [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]];
  return [0, 1, 2].map((r) => [0, 1, 2].map((b) => (T[r][0] * xyz[0][b] + T[r][1] * xyz[1][b] + T[r][2] * xyz[2][b]) / VIS_LREF));
}
