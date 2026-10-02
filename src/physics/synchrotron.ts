// Synchrotron jet physics (spec docs/specs/2026-10-02-synchrotron-jet-design.md). CPU twin of the jet
// code in src/render/emission-shared.wgsl, plus what the shader cannot do (the energy-budget density
// scale). Gaussian cgs; lengths in r_g where noted.

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
