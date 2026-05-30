import { M } from "./kerr";
// prograde uses upper signs; s = +1 prograde, -1 retrograde.

export function omegaKepler(r: number, a: number, prograde = true): number {
  const s = prograde ? 1 : -1;
  return s * Math.sqrt(M) / (Math.pow(r, 1.5) + s * a * Math.sqrt(M));
}
export function circE(r: number, a: number, prograde = true): number {
  const s = prograde ? 1 : -1, rt = Math.sqrt(r), sM = Math.sqrt(M);
  const num = r * rt - 2 * M * rt + s * a * sM;
  const den = Math.pow(r, 0.75) * Math.sqrt(r * rt - 3 * M * rt + 2 * s * a * sM);
  return num / den;
}
export function circL(r: number, a: number, prograde = true): number {
  const s = prograde ? 1 : -1, rt = Math.sqrt(r), sM = Math.sqrt(M);
  const den = Math.pow(r, 0.75) * Math.sqrt(r * rt - 3 * M * rt + 2 * s * a * sM);
  return s * sM * (r * r - s * 2 * a * sM * rt + a * a) / den;
}
export function iscoRadius(a: number, prograde = true): number {
  const as = a / M;
  const Z1 = 1 + Math.cbrt(1 - as * as) * (Math.cbrt(1 + as) + Math.cbrt(1 - as));
  const Z2 = Math.sqrt(3 * as * as + Z1 * Z1);
  const s = prograde ? -1 : 1;
  return M * (3 + Z2 + s * Math.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2)));
}
export function photonOrbit(a: number, prograde = true): number {
  const as = a / M, sgn = prograde ? -1 : 1;
  return 2 * M * (1 + Math.cos((2 / 3) * Math.acos(sgn * as)));
}
export function marginallyBound(a: number, prograde = true): number {
  const s = prograde ? -1 : 1;
  return 2 * M + s * a + 2 * Math.sqrt(M) * Math.sqrt(M + s * a);
}
export function efficiency(a: number, prograde = true): number {
  return 1 - circE(iscoRadius(a, prograde), a, prograde);
}
