import { describe, it, expect } from "vitest";
import { synchConsts, synchCoeffs, lnGamma, Q_E, M_E, C_CGS, SYN_P, GAMMA_MIN } from "../src/physics/synchrotron";

// Direct numerical integration of the single-electron spectrum F(x) = x Int_x^inf K_5/3 over the power
// law and an isotropic pitch-angle distribution (Rybicki & Lightman 6.18, 6.50).
function K53(t: number) { let s = 0; const N = 4000, umax = Math.log(200 / t + 2) + 2, du = umax / N;
  for (let i = 0; i <= N; i++) { const u = i * du, w = i === 0 || i === N ? 1 : i % 2 ? 4 : 2; s += w * Math.exp(-t * Math.cosh(u)) * Math.cosh((5 * u) / 3); }
  return (s * du) / 3; }
const FX: { x: number[]; F: number[] } = (() => { const n = 500, lx0 = -8, lx1 = Math.log(60), ts: number[] = [], ks: number[] = [];
  for (let i = 0; i <= n; i++) { ts.push(Math.exp(lx0 + ((lx1 - lx0) * i) / n)); ks.push(K53(ts[i])); }
  const cum = new Array(ts.length).fill(0); for (let i = ts.length - 2; i >= 0; i--) cum[i] = cum[i + 1] + 0.5 * (ks[i] + ks[i + 1]) * (ts[i + 1] - ts[i]);
  return { x: ts, F: ts.map((t, i) => t * cum[i]) }; })();
function F(x: number) { const { x: xs, F: Fs } = FX; if (x <= xs[0]) return 2.15 * Math.pow(x, 1 / 3); if (x >= xs[xs.length - 1]) return 0;
  let lo = 0, hi = xs.length - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] > x) hi = m; else lo = m; }
  const f = (Math.log(x) - Math.log(xs[lo])) / (Math.log(xs[hi]) - Math.log(xs[lo])); return Fs[lo] + (Fs[hi] - Fs[lo]) * f; }
function numeric(K: number, B: number, nu: number): [number, number] {
  const p = SYN_P; let J = 0, A = 0, W = 0; const NA = 60, NG = 600, lg0 = Math.log(GAMMA_MIN), lg1 = Math.log(1e8), wB = (Q_E * B) / (M_E * C_CGS);
  for (let ia = 0; ia < NA; ia++) { const s = Math.sin(((ia + 0.5) / NA) * (Math.PI / 2)); let Pw = 0, Ia = 0;
    for (let ig = 0; ig < NG; ig++) { const lg = lg0 + ((lg1 - lg0) * (ig + 0.5)) / NG, g = Math.exp(lg), dg = (g * (lg1 - lg0)) / NG;
      const x = (2 * Math.PI * nu) / (1.5 * g * g * wB * s), P1 = ((Math.sqrt(3) * Q_E ** 3 * B * s) / (2 * Math.PI * M_E * C_CGS ** 2)) * F(x);
      Pw += K * Math.pow(g, -p) * P1 * dg; Ia += 2 * Math.PI * P1 * K * Math.pow(g, -p - 1) / (M_E * C_CGS ** 2) * dg; }
    J += (Pw / 2) * s; A += (((p + 2) * C_CGS ** 2) / (8 * Math.PI * nu * nu)) * Ia * s; W += s; }
  return [J / W, A / W];
}

describe("synchrotron coefficients (Rybicki & Lightman 6.36 / 6.53, pitch-averaged)", () => {
  it("lnGamma matches known values", () => {
    expect(lnGamma(5)).toBeCloseTo(Math.log(24), 10);
    expect(lnGamma(0.5)).toBeCloseTo(0.5 * Math.log(Math.PI), 10);
  });
  it("closed forms agree with direct numerical integration within 0.5 %", () => {
    const K = 1e6, B = 10, nu = 5.45e14;
    const [j, a] = synchCoeffs(K, B, nu), [jn, an] = numeric(K, B, nu);
    expect(Math.abs(j / jn - 1)).toBeLessThan(5e-3);
    expect(Math.abs(a / an - 1)).toBeLessThan(5e-3);
  }, 120000);
  it("scales as K B^((p+1)/2) nu^(-(p-1)/2) and K B^((p+2)/2) nu^(-(p+4)/2)", () => {
    const [j1, a1] = synchCoeffs(1e6, 10, 5e14), [j2, a2] = synchCoeffs(2e6, 20, 1e15), p = SYN_P;
    expect(j2 / j1).toBeCloseTo(2 * 2 ** ((p + 1) / 2) * 2 ** (-(p - 1) / 2), 9);
    expect(a2 / a1).toBeCloseTo(2 * 2 ** ((p + 2) / 2) * 2 ** (-(p + 4) / 2), 9);
  });
  it("below nu_min the low-frequency tail is continuous (X-ray binaries: B ~ 1e8 G)", () => {
    const B = 1e8, numin = Math.exp(synchConsts().lnNuMin0) * B;
    const [jl, al] = synchCoeffs(1e20, B, numin * (1 - 1e-9)), [jh, ah] = synchCoeffs(1e20, B, numin * (1 + 1e-9));
    expect(jl / jh).toBeCloseTo(1, 6); expect(al / ah).toBeCloseTo(1, 6);
    const [ja] = synchCoeffs(1e20, B, numin / 8), [jb] = synchCoeffs(1e20, B, numin / 64);
    expect(ja / jb).toBeCloseTo(2, 6); // nu^(1/3)
    for (const v of synchCoeffs(1e20, B, 5e14)) expect(Number.isFinite(v)).toBe(true);
  });
});
