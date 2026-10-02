// Exact cyclo-synchrotron emission and the cooled electron population (spec 2026-10-02 cooled jet).
// Self-contained (no imports): scripts/build-synch-table.mjs runs it directly in Node to build the table.
// Kernel units: e = m_e = c = omega_B = 1, x = nu / nu_B; a kernel is the power per unit angular frequency
// (units e^2 omega_B / c), integrated over emission direction and averaged over an isotropic pitch angle.

// ---- Bessel J_n(z), J_n'(z) ------------------------------------------------------------------------------
const NM = 40;
/** Exact J_n, J_n' by Miller's backward recurrence (normalised with J_0 + 2 sum J_2k = 1). */
export function besselRecurrence(n: number, z: number): [number, number] {
  if (z === 0) return [n === 0 ? 1 : 0, n === 1 ? 0.5 : 0];
  const N = Math.max(n, Math.ceil(z)) + 40 + Math.ceil(Math.sqrt(40 * Math.max(n, z)));
  let jp1 = 0, j = 1e-300, sum = 0, Jn = 0, Jn1 = 0;
  for (let k = N; k >= 1; k--) {
    const jm1 = (2 * k / z) * j - jp1; jp1 = j; j = jm1;
    if (Math.abs(j) > 1e250) { j *= 1e-250; jp1 *= 1e-250; sum *= 1e-250; Jn *= 1e-250; Jn1 *= 1e-250; }
    if (k - 1 === n) { Jn = j; Jn1 = jp1; }
    if (k - 1 > 0 && (k - 1) % 2 === 0) sum += 2 * j;
  }
  sum += j;
  const jn = Jn / sum, jn1 = Jn1 / sum;
  return [jn, (n / z) * jn - jn1];
}
function airy(t: number): [number, number] {
  if (t > 6) {
    const zeta = (2 / 3) * Math.pow(t, 1.5), e = Math.exp(-zeta) / (2 * Math.sqrt(Math.PI));
    return [e * Math.pow(t, -0.25) * (1 - 5 / (72 * zeta) + 385 / (10368 * zeta * zeta)),
      -e * Math.pow(t, 0.25) * (1 + 7 / (72 * zeta) - 455 / (10368 * zeta * zeta))];
  }
  if (t < -6) {
    const at = -t, zeta = (2 / 3) * Math.pow(at, 1.5), ph = zeta + Math.PI / 4;
    return [Math.pow(at, -0.25) / Math.sqrt(Math.PI) * (Math.sin(ph) * (1 - 385 / (10368 * zeta * zeta)) - Math.cos(ph) * 5 / (72 * zeta)),
      -Math.pow(at, 0.25) / Math.sqrt(Math.PI) * (Math.cos(ph) * (1 + 455 / (10368 * zeta * zeta)) + Math.sin(ph) * 7 / (72 * zeta))];
  }
  const c1 = 0.355028053887817, c2 = 0.258819403792807;
  let f = 1, g = t, fp = 0, gp = 1, tf = 1, tg = t; const t3 = t * t * t;
  for (let k = 1; k < 60; k++) {
    tf *= t3 / ((3 * k - 1) * (3 * k)); tg *= t3 / ((3 * k) * (3 * k + 1));
    f += tf; g += tg; if (t !== 0) { fp += tf * 3 * k / t; gp += tg * (3 * k + 1) / t; }
    if (Math.abs(tf) + Math.abs(tg) < 1e-17 * (Math.abs(f) + Math.abs(g))) break;
  }
  return [c1 * f - c2 * g, c1 * fp - c2 * gp];
}
/** J_nu(z) for real order nu > 0 from the uniform (Airy) expansion. */
function jAiry(nu: number, z: number): number {
  const w = z / nu;
  if (Math.abs(w - 1) < 1e-9) return Math.pow(2, 1 / 3) * airy(0)[0] / Math.pow(nu, 1 / 3);
  let zeta: number;
  if (w > 1) { const s = Math.sqrt(w * w - 1); zeta = -Math.pow(1.5 * (s - Math.acos(1 / w)), 2 / 3); }
  else { const s = Math.sqrt(1 - w * w); zeta = Math.pow(1.5 * (Math.log((1 + s) / w) - s), 2 / 3); }
  return Math.pow(4 * zeta / (1 - w * w), 0.25) * airy(Math.pow(nu, 2 / 3) * zeta)[0] / Math.pow(nu, 1 / 3);
}
/** [J_n(z), J_n'(z)]: exact recurrence for n <= 40; above, the uniform expansion for J (6e-4 at n = 41, falling
 *  with n) and J' = (J_{n-1} - J_{n+1}) / 2 (the expansion's own J' term is 1.8 % off at n = 41). */
export function bessel(n: number, z: number): [number, number] {
  if (n <= NM) return besselRecurrence(n, z);
  return [jAiry(n, z), 0.5 * (jAiry(n - 1, z) - jAiry(n + 1, z))];
}

// ---- quadrature ------------------------------------------------------------------------------------------
export function gaussLegendre(n: number): [number[], number[]] {
  const x: number[] = [], w: number[] = [];
  for (let i = 1; i <= n; i++) {
    let r = Math.cos(Math.PI * (i - 0.25) / (n + 0.5)), dp = 0;
    for (let it = 0; it < 100; it++) {
      let p0 = 1, p1 = r;
      for (let k = 2; k <= n; k++) { const p2 = ((2 * k - 1) * r * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
      dp = n * (r * p1 - p0) / (r * r - 1);
      const dr = p1 / dp; r -= dr; if (Math.abs(dr) < 1e-15) break;
    }
    x.push(r); w.push(2 / ((1 - r * r) * dp * dp));
  }
  return [x, w];
}
const [GX, GW] = gaussLegendre(48);
const GL8 = gaussLegendre(8);

// ---- single electron -------------------------------------------------------------------------------------
/** Harmonic n's contribution at x for an electron of Lorentz factor gamma (Bekefi 1966 emissivity; the resonance
 *  delta removes the direction integral; mu = cos pitch, mu' = cos(view angle to B), mu mu' = C). */
export function harmonicPower(x: number, gamma: number, n: number): number {
  const beta = Math.sqrt(1 - 1 / (gamma * gamma)), Om = 1 / gamma;
  const C = (1 - n * Om / x) / beta, aC = Math.abs(C); if (aC > 1) return 0;
  const oneMinus = 1 - beta * C;
  const wMax = Math.min(1, beta * (1 - aC) / oneMinus); // J_n(n w) ~ exp(-n (atanh s - s)): prune the negligible
  if (wMax < 1) { const sq = Math.sqrt(1 - wMax * wMax); if (n * (Math.atanh(sq) - sq) > 60) return 0; }
  let part = 0;
  for (const sgn of [1, -1]) for (let k = 0; k < GX.length; k++) {
    const v = 0.5 * (GX[k] + 1), dv = 0.5 * GW[k], mu = sgn * (aC + (1 - aC) * v * v), dmu = (1 - aC) * 2 * v * dv;
    const mup = C / mu; if (Math.abs(mup) > 1) continue;
    const sinxi = Math.sqrt(Math.max(0, 1 - mu * mu)), sinth = Math.sqrt(Math.max(0, 1 - mup * mup));
    const bpar = beta * mu, bperp = beta * sinxi, z = n * bperp * sinth / oneMinus, [J, Jp] = bessel(n, z);
    const ang = sinth > 1e-14 ? Math.pow((mup - bpar) / sinth, 2) * J * J : 0;
    part += (x * x / (2 * Math.PI)) * (ang + bperp * bperp * Jp * Jp) / (x * beta * Math.abs(mu)) * dmu;
  }
  return Math.PI * part;
}
/** Exact single-electron kernel: the sum over every harmonic. */
export function kernelExact(x: number, gamma: number): number {
  const beta = Math.sqrt(1 - 1 / (gamma * gamma)), Om = 1 / gamma;
  const nLo = Math.max(1, Math.ceil(x * (1 - beta) / Om - 1e-12)), nHi = Math.floor(x * (1 + beta) / Om + 1e-12);
  let tot = 0;
  for (let n = nLo; n <= nHi; n++) tot += harmonicPower(x, gamma, n);
  return tot;
}
/** Harmonic n's frequency interval [x_lo, x_hi] (pitch-averaged). */
export function harmonicRange(gamma: number, n: number): [number, number] {
  const beta = Math.sqrt(1 - 1 / (gamma * gamma));
  return [n / (gamma * (1 + beta)), n / (gamma * (1 - beta))];
}

// ---- synchrotron kernel (gamma >> 1) ---------------------------------------------------------------------
// F(y) = y Int_y^inf K_5/3 = y Int_0^inf exp(-y cosh tau) cosh(5 tau / 3) / cosh(tau) dtau, tabulated once.
function fDirect(y: number): number {
  const N = 4000, tmax = Math.acosh(1 + 60 / y), h = tmax / N; let s = 0;
  for (let i = 0; i <= N; i++) { const tau = i * h, w = i === 0 || i === N ? 1 : i % 2 ? 4 : 2, c = Math.cosh(tau);
    s += w * Math.exp(-y * c) * Math.cosh((5 * tau) / 3) / c; }
  return (y * s * h) / 3;
}
// Built on first use: the browser imports this module (table lookup) but never evaluates F.
let F_TAB: { a: number; b: number; n: number; lnF: number[] } | null = null;
function fTab() {
  if (!F_TAB) {
    const n = 1200, a = Math.log(1e-6), b = Math.log(60), lnF: number[] = [];
    for (let i = 0; i <= n; i++) lnF.push(Math.log(fDirect(Math.exp(a + (b - a) * i / n))));
    F_TAB = { a, b, n, lnF };
  }
  return F_TAB;
}
/** Synchrotron function F(y) (Rybicki & Lightman 6.31c). */
export function synchF(y: number): number {
  if (y <= 1e-6) return 2.1495282 * Math.cbrt(y);
  if (y >= 60) return Math.sqrt(Math.PI / 2) * Math.sqrt(y) * Math.exp(-y) * (1 + 55 / (72 * y));
  const T = fTab(), f = (Math.log(y) - T.a) / (T.b - T.a) * T.n, i = Math.min(T.n - 1, Math.floor(f)), t = f - i;
  return Math.exp(T.lnF[i] * (1 - t) + T.lnF[i + 1] * t);
}
/** Pitch-averaged synchrotron kernel, same units as kernelExact. */
export function kernelSync(x: number, gamma: number): number {
  let s = 0; const N = 64;
  for (let i = 0; i < N; i++) { const mu = (i + 0.5) / N, sx = Math.sqrt(1 - mu * mu); s += (Math.sqrt(3) * sx / (2 * Math.PI)) * synchF(x / (1.5 * gamma * gamma * sx)); }
  return s / N;
}
/** Exact sum up to the seam, synchrotron kernel above (spec 2.3: 1.3 % power-weighted at gamma = 10). */
export const GAMMA_SEAM = 10;
/** Kernel averaged over the bin [xa, xb]: harmonic lines (few per bin) integrated over their overlap with the bin,
 *  the bin centre where more than 50 overlap (the sum is smooth there). */
export function kernelBin(xa: number, xb: number, gamma: number): number {
  if (gamma > GAMMA_SEAM) return kernelSync(Math.sqrt(xa * xb), gamma);
  const beta = Math.sqrt(1 - 1 / (gamma * gamma));
  const nLo = Math.max(1, Math.ceil(xa * gamma * (1 - beta) - 1e-12)), nHi = Math.floor(xb * gamma * (1 + beta) + 1e-12);
  if (nHi < nLo) return 0;
  if (nHi - nLo > 50) return kernelExact(Math.sqrt(xa * xb), gamma);
  let tot = 0; const [gx, gw] = GL8;
  for (let n = nLo; n <= nHi; n++) {
    const [ra, rb] = harmonicRange(gamma, n), a = Math.max(xa, ra), b = Math.min(xb, rb);
    if (b <= a) continue;
    for (let k = 0; k < gx.length; k++) { // x = a + (b-a)(1 - cos(pi t))/2 clusters nodes at the line edges
      const t = 0.5 * (gx[k] + 1), x = a + (b - a) * (1 - Math.cos(Math.PI * t)) / 2, dx = (b - a) * Math.PI * Math.sin(Math.PI * t) / 2 * 0.5 * gw[k];
      tot += harmonicPower(x, gamma, n) * dx;
    }
  }
  return tot / (xb - xa);
}
