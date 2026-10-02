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

// ---- cooled electron population (spec 2.1-2.2) -----------------------------------------------------------
/** Injection fitted to M87's core SED: N ~ gamma^-2.2 (alpha 1.1) for 40..1200, gamma^-3.0 (alpha 1.5) above. */
export const G_MIN = 40, G_BR = 1200, P1 = 2.2, P2 = 3.0;
export const gInj = (g: number) => (g < G_MIN ? 0 : g <= G_BR ? Math.pow(g, -P1) : Math.pow(G_BR, P2 - P1) * Math.pow(g, -P2));
/** Int_gamma^inf g. */
export function gTail(g: number): number {
  if (!Number.isFinite(g)) return 0;
  const a = Math.max(g, G_MIN), t2 = Math.pow(G_BR, P2 - P1) * Math.pow(Math.max(a, G_BR), 1 - P2) / (P2 - 1);
  return a >= G_BR ? t2 : (Math.pow(a, 1 - P1) - Math.pow(G_BR, 1 - P1)) / (P1 - 1) + t2;
}
/** I_g = Int (gamma - 1) g dgamma: injected kinetic energy per unit Q0, in m_e c^2. */
export const I_G = (Math.pow(G_BR, 2 - P1) - Math.pow(G_MIN, 2 - P1)) / (2 - P1) - (Math.pow(G_MIN, 1 - P1) - Math.pow(G_BR, 1 - P1)) / (P1 - 1)
  + Math.pow(G_BR, P2 - P1) * (Math.pow(G_BR, 2 - P2) / (P2 - 2) - Math.pow(G_BR, 1 - P2) / (P2 - 1));
/** g(G) (G^2 - 1) without overflow for huge G. */
const gInjG2 = (G: number) => (G > 1e100 ? Math.pow(G_BR, P2 - P1) * Math.pow(G, 2 - P2) : gInj(G) * (G * G - 1));
/** acoth(gamma) from u = gamma beta without cancellation: ln((gamma + 1) / u). */
const acothU = (u: number) => Math.log((1 + Math.sqrt(1 + u * u)) / u);
/** Injection energy that cools to momentum u within cooling depth s: coth(acoth(gamma) - s), or Infinity. */
export function coolFromU(u: number, s: number): number { const d = acothU(u) - s; return d > 0 ? 1 / Math.tanh(d) : Infinity; }
/** N^ = N k / Q0 per unit gamma at momentum u: (1/u^2) Int_gamma^G g (spec 2.2). Written in u: gamma rounds to 1
 *  for u < 1e-8 in double precision, and the cold electrons matter for absorption at the fundamental. */
export function nHatU(u: number, s: number): number { const g = Math.sqrt(1 + u * u); return (gTail(g) - gTail(coolFromU(u, s))) / (u * u); }
/** d/dgamma [N^ / (gamma u)] (analytic; dG/dgamma = (G^2 - 1) / u^2). */
export function dNOverGuU(u: number, s: number): number {
  const g = Math.sqrt(1 + u * u), u2 = u * u, G = coolFromU(u, s), T = gTail(g) - gTail(G);
  const dT = -gInj(g) + (Number.isFinite(G) ? gInjG2(G) / u2 : 0);
  const den = g * u2 * u;
  return (dT * den - T * (u2 * u + 3 * g * g * u)) / (den * den);
}
export const nHat = (g: number, s: number) => nHatU(Math.sqrt(g * g - 1), s);
export const dNOverGu = (g: number, s: number) => dNOverGuU(Math.sqrt(g * g - 1), s);

// ---- the table (spec 2.3) --------------------------------------------------------------------------------
export interface Grid { lnx0: number; lnx1: number; nx: number; lns0: number; lns1: number; ns: number }
export const TABLE_GRID: Grid = { lnx0: Math.log(1e-4), lnx1: Math.log(1e10), nx: 1612, lns0: Math.log(1e-3), lns1: Math.log(40), ns: 160 };
/** Below this momentum the dipole limit: the fundamental line alone, carrying the Larmor power. */
export const U_COLD = 0.01;
/** Momentum grid for the gamma integral: cold (1e-18..0.01), exact (to the seam), synchrotron (to 1e7). */
export function uGrid(): { u: number[]; dlnu: number[] } {
  const u: number[] = [], d: number[] = [];
  const seg = (a: number, b: number, step: number) => { const n = Math.ceil((Math.log(b) - Math.log(a)) / step), h = (Math.log(b) - Math.log(a)) / n;
    for (let i = 0; i < n; i++) { u.push(Math.exp(Math.log(a) + h * (i + 0.5))); d.push(h); } };
  seg(1e-18, U_COLD, 0.1);
  seg(U_COLD, Math.sqrt(GAMMA_SEAM * GAMMA_SEAM - 1), 0.03);
  seg(Math.sqrt(GAMMA_SEAM * GAMMA_SEAM - 1), 1e7, 0.05);
  return { u, dlnu: d };
}
/** Bin-averaged kernel for one momentum over all x bins (independent of s). */
export function kernelRow(u: number, g: Grid, out: Float64Array) {
  const gam = Math.sqrt(1 + u * u), h = (g.lnx1 - g.lnx0) / g.nx;
  out.fill(0);
  if (u < U_COLD) { // fundamental line [1/(gamma(1+beta)), gamma(1+beta)] with power (4/9) u^2, spread over its bins
    const beta = u / gam, a = 1 / (gam * (1 + beta)), b = gam * (1 + beta), P = (4 / 9) * u * u;
    if (!(b - a > 1e-9 * a)) { // narrower than double precision resolves: all in the bin holding x = 1/gamma
      const i = Math.floor((Math.log(1 / gam) - g.lnx0) / h);
      if (i >= 0 && i < g.nx) out[i] = P / (Math.exp(g.lnx0 + h * (i + 1)) - Math.exp(g.lnx0 + h * i));
      return;
    }
    const ia = Math.floor((Math.log(a) - g.lnx0) / h), ib = Math.floor((Math.log(b) - g.lnx0) / h);
    for (let i = Math.max(0, ia); i <= Math.min(g.nx - 1, ib); i++) {
      const xa = Math.exp(g.lnx0 + h * i), xb = Math.exp(g.lnx0 + h * (i + 1)), ov = Math.max(0, Math.min(b, xb) - Math.max(a, xa));
      out[i] = P * (ov / (b - a)) / (xb - xa);
    }
    return;
  }
  const xmax = 90 * gam * gam + 10; // beyond 60 x the critical frequency the kernel is below exp(-60)
  for (let i = 0; i < g.nx; i++) {
    const xa = Math.exp(g.lnx0 + h * i), xb = Math.exp(g.lnx0 + h * (i + 1));
    if (xa > xmax) break;
    out[i] = kernelBin(xa, xb, gam);
  }
}
export interface Population { n(u: number, s: number): number; d(u: number, s: number): number }
const COOLED: Population = { n: nHatU, d: dNOverGuU };
/** J^ = Int N^ p^ dgamma, A^ = -Int p^ gamma u d/dgamma(N^/(gamma u)) dgamma on the (x, s) grid (p^ = 2 pi kernel). */
export function contract(rows: Float64Array[], uu: { u: number[]; dlnu: number[] }, g: Grid, pop: Population = COOLED) {
  const J = new Float64Array(g.nx * g.ns), A = new Float64Array(g.nx * g.ns);
  for (let js = 0; js < g.ns; js++) {
    const s = Math.exp(g.lns0 + (g.lns1 - g.lns0) * js / (g.ns - 1));
    for (let k = 0; k < uu.u.length; k++) {
      const u = uu.u[k], gam = Math.sqrt(1 + u * u), dg = (u * u / gam) * uu.dlnu[k];
      const n = pop.n(u, s), dn = pop.d(u, s); if (n === 0 && dn === 0) continue;
      const wj = n * 2 * Math.PI * dg, wa = -2 * Math.PI * gam * u * dn * dg, row = rows[k];
      for (let i = 0; i < g.nx; i++) { const r = row[i]; if (r === 0) continue; J[js * g.nx + i] += wj * r; A[js * g.nx + i] += wa * r; }
    }
  }
  return { J, A };
}
// File: Uint32 [magic 'SYNT', version 1, nx, ns], Float32 [lnx0, lnx1, lns0, lns1, G_MIN, G_BR, P1, P2, GAMMA_SEAM,
// U_COLD, 0, 0], then Float32 pairs (ln J^, ln A^), s rows of x columns. Non-positive values (frequencies no
// electron reaches; a few slow-cooling maser cells) are stored as ln = -80, i.e. zero.
const MAGIC = 0x544e5953, HEAD = 16;
export function encodeTable(J: Float64Array, A: Float64Array, g: Grid): ArrayBuffer {
  const buf = new ArrayBuffer(HEAD * 4 + g.nx * g.ns * 8), u32 = new Uint32Array(buf, 0, 4), f32 = new Float32Array(buf);
  u32.set([MAGIC, 1, g.nx, g.ns]);
  f32.set([g.lnx0, g.lnx1, g.lns0, g.lns1, G_MIN, G_BR, P1, P2, GAMMA_SEAM, U_COLD, 0, 0], 4);
  for (let i = 0; i < g.nx * g.ns; i++) { f32[HEAD + 2 * i] = J[i] > 0 ? Math.log(J[i]) : -80; f32[HEAD + 2 * i + 1] = A[i] > 0 ? Math.log(A[i]) : -80; }
  return buf;
}
export interface SynchTable { nx: number; ns: number; lnx0: number; lnx1: number; lns0: number; lns1: number; data: Float32Array<ArrayBuffer> }
export function parseTable(buf: ArrayBuffer): SynchTable {
  const u32 = new Uint32Array(buf, 0, 4), f32 = new Float32Array(buf);
  if (u32[0] !== MAGIC || u32[1] !== 1) throw new Error("synch-table.bin: bad magic or version");
  const c = [G_MIN, G_BR, P1, P2, GAMMA_SEAM, U_COLD];
  c.forEach((v, i) => { if (Math.abs(f32[8 + i] - v) > 1e-6 * Math.abs(v)) throw new Error("synch-table.bin: built for other constants; rebuild it"); });
  return { nx: u32[2], ns: u32[3], lnx0: f32[4], lnx1: f32[5], lns0: f32[6], lns1: f32[7], data: f32.subarray(HEAD) };
}
/** Bilinear (ln J^, ln A^) at (ln x, ln s) between cell centres. Above the x grid: the fast-cooled gamma^-4
 *  asymptote (J^ ~ x^-1.5, A^ ~ x^-2); below: the first column; below the s grid: slope 1 in ln s (N^ ~ s);
 *  above: the last row. Twin of synchLookupJ in emission-shared.wgsl. */
export function lookup(t: SynchTable, lnx: number, lns: number): [number, number] {
  const h = (t.lnx1 - t.lnx0) / t.nx;
  let fx = (lnx - t.lnx0) / h - 0.5, extX = 0;
  if (fx > t.nx - 1) { extX = (fx - (t.nx - 1)) * h; fx = t.nx - 1; }
  if (fx < 0) fx = 0;
  let fs = (lns - t.lns0) / (t.lns1 - t.lns0) * (t.ns - 1), extS = 0;
  if (fs < 0) { extS = lns - t.lns0; fs = 0; }
  if (fs > t.ns - 1) fs = t.ns - 1;
  const ix = Math.min(t.nx - 2, Math.floor(fx)), is = Math.min(t.ns - 2, Math.floor(fs)), ax = fx - ix, as = fs - is;
  const v = (i: number, j: number, c: number) => t.data[2 * (j * t.nx + i) + c];
  const bil = (c: number) => (1 - as) * ((1 - ax) * v(ix, is, c) + ax * v(ix + 1, is, c)) + as * ((1 - ax) * v(ix, is + 1, c) + ax * v(ix + 1, is + 1, c));
  return [bil(0) - 1.5 * extX + extS, bil(1) - 2 * extX + extS];
}
