// Synchrotron jet physics (spec docs/specs/2026-10-02-synchrotron-jet-design.md). CPU twin of the jet
// code in src/render/emission-shared.wgsl, plus what the shader cannot do (the energy-budget density
// scale). Gaussian cgs; lengths in r_g where noted.

import { metricUpper, horizonOuter } from "./kerr";
import { mdotFromLambda } from "./units";
import { funnelEdge, wallProfile, lengthFalloff, JET } from "./jet";
export { gammaProfile, GAMMA_REF_Z, GAMMA_SLOPE } from "./jet";

export const Q_E = 4.80320471e-10, M_E = 9.1093837e-28, C_CGS = 2.99792458e10, G_CGS = 6.6743e-8, MSUN_G = 1.98847e33;
/** Electron power law N(gamma) = K gamma^-p for gamma >= GAMMA_MIN (spec 2.3). */
export const SYN_P = 2.4, GAMMA_MIN = 10;

export function lnGamma(x: number): number {
  const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lnGamma(1 - x);
  x -= 1; let s = c[0]; const t = x + g + 0.5;
  for (let i = 1; i < 9; i++) s += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(s);
}
/** <sin^x alpha> over an isotropic pitch-angle distribution. */
export const avgSinPow = (x: number) => (Math.sqrt(Math.PI) / 2) * Math.exp(lnGamma((x + 2) / 2) - lnGamma((x + 3) / 2));

/** ln of Cj, Ca in j = Cj K B^((p+1)/2) nu^(-(p-1)/2) (per sr, RL 6.36 with j = P_omega / 2) and
 *  alpha = Ca K B^((p+2)/2) nu^(-(p+4)/2) (RL 6.53, whose C is per unit ENERGY: C_E = K (m c^2)^(p-1)),
 *  both averaged over pitch angle; and ln of nu_min / B = 3 gmin^2 q / (4 pi m c). */
export function synchConsts(p = SYN_P, gmin = GAMMA_MIN) {
  const lnCj = 0.5 * Math.log(3) + 3 * Math.log(Q_E) - Math.log(4 * Math.PI * M_E * C_CGS ** 2 * (p + 1))
    + lnGamma(p / 4 + 19 / 12) + lnGamma(p / 4 - 1 / 12) + Math.log(avgSinPow((p + 1) / 2))
    - ((p - 1) / 2) * Math.log((2 * Math.PI * M_E * C_CGS) / (3 * Q_E));
  const lnCa = 0.5 * Math.log(3) + 3 * Math.log(Q_E) - Math.log(8 * Math.PI * M_E)
    + (p / 2) * Math.log((3 * Q_E) / (2 * Math.PI * M_E ** 3 * C_CGS ** 5))
    + lnGamma((3 * p + 2) / 12) + lnGamma((3 * p + 22) / 12) + Math.log(avgSinPow((p + 2) / 2))
    + (p - 1) * Math.log(M_E * C_CGS ** 2);
  const lnNuMin0 = Math.log((3 * gmin * gmin * Q_E) / (4 * Math.PI * M_E * C_CGS));
  return { lnCj, lnCa, lnNuMin0 };
}
const SC = synchConsts();
/** Plasma-frame emissivity j (erg s^-1 cm^-3 Hz^-1 sr^-1) and absorption alpha (cm^-1); below nu_min
 *  the low-frequency forms matched at nu_min (spec 2.3). Computed in logs (B up to ~1e9 G). */
export function synchCoeffs(K: number, B: number, nu: number): [number, number] {
  const lnB = Math.log(B), lnNu = Math.log(nu), lnNuMin = SC.lnNuMin0 + lnB, le = Math.max(lnNu, lnNuMin);
  let lj = SC.lnCj + Math.log(K) + ((SYN_P + 1) / 2) * lnB - ((SYN_P - 1) / 2) * le;
  let la = SC.lnCa + Math.log(K) + ((SYN_P + 2) / 2) * lnB - ((SYN_P + 4) / 2) * le;
  if (lnNu < lnNuMin) { lj += (lnNu - lnNuMin) / 3; la -= (5 / 3) * (lnNu - lnNuMin); }
  return [Math.exp(lj), Math.exp(la)];
}

/** Magnetically arrested horizon flux (Gaussian units, Tchekhovskoy et al. 2011) and BZ coefficient. */
export const PHI_MAD = 50, KAPPA_BZ = 0.05;
/** Visible band for the energy budget (400-750 nm). */
export const VIS_NU_LO = C_CGS / 750e-7, VIS_NU_HI = C_CGS / 400e-7;

/** r_g (cm), Mdot (g/s), horizon flux Phi = phi sqrt(Mdot c) r_g (G cm^2), BZ power (erg/s), b0 = Phi/(pi r_g^2) (G). */
export function jetEnergetics(mSun: number, a: number, lambda: number) {
  const rgCm = (G_CGS * mSun * MSUN_G) / C_CGS ** 2;
  const mdot = mdotFromLambda(mSun, a, lambda) * 1e3;
  const phi = PHI_MAD * Math.sqrt(mdot * C_CGS) * rgCm;
  const omegaH = (a * C_CGS) / (2 * horizonOuter(a) * rgCm);
  const pBZ = (KAPPA_BZ / (4 * Math.PI * C_CGS)) * phi * phi * omegaH * omegaH;
  return { rgCm, mdot, phi, pBZ, b0: phi / (Math.PI * rgCm * rgCm) };
}
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
/** Comoving, optically thin visible-band luminosity (erg/s) of the rendered jet (both lobes,
 *  zBase <= |z| <= jetLength, rho <= 1.2 rho_f) at k_scale = 1, i.e. K = B^2 * wall * falloff. Knots and
 *  turbulence are mean-one modulations (1 + amp (noise - 1/2) 2), so this is the time-averaged budget. */
export function visLuminanceUnit(a: number, b0: number, rgCm: number, jetLength: number): number {
  const NZ = 120, NR = 48, NNU = 16, l0 = Math.log(VIS_NU_LO), l1 = Math.log(VIS_NU_HI), zb = JET.zBase;
  let L = 0;
  for (let iz = 0; iz < NZ; iz++) {
    const dz = (jetLength - zb) / NZ, z = zb + dz * (iz + 0.5), rf = funnelEdge(z), dr = (1.2 * rf) / NR;
    for (let ir = 0; ir < NR; ir++) {
      const rho = dr * (ir + 0.5), shape = wallProfile(rho, z) * lengthFalloff(z, jetLength);
      if (shape <= 0) continue;
      const B = jetField(rho, z, a, b0), K = B * B * shape;
      let jv = 0;
      for (let k = 0; k < NNU; k++) { const nu = Math.exp(l0 + ((l1 - l0) * (k + 0.5)) / NNU); jv += synchCoeffs(K, B, nu)[0] * nu * ((l1 - l0) / NNU); }
      L += 2 * (2 * Math.PI * rho * dr * dz) * rgCm ** 3 * 4 * Math.PI * jv;
    }
  }
  return L;
}
/** The jet's three physics uniforms (spec 2.3): k_scale makes the visible luminance equal eps * P_BZ. */
export function jetUniforms(mSun: number, a: number, lambda: number, eps: number, jetLength: number) {
  const E = jetEnergetics(mSun, a, lambda);
  return { jetB0: E.b0, jetKScale: (eps * E.pBZ) / visLuminanceUnit(a, E.b0, E.rgCm, jetLength), rgCm: E.rgCm, pBZ: E.pBZ };
}
/** One band of one jet sample: exact solution across a uniform slab of path ds (cm) behind optical depth
 *  tau already accumulated from the camera. Twin of jetSlabJ in emission-shared.wgsl. */
export function slabStep(I: number, tau: number, j: number, alpha: number, ds: number): [number, number] {
  const dt = alpha * ds, fac = dt < 1e-4 ? 1 - 0.5 * dt : (1 - Math.exp(-dt)) / dt;
  return [I + j * ds * fac * Math.exp(-tau), Math.min(tau + dt, 1e30)];
}
