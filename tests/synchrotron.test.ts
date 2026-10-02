import { describe, it, expect } from "vitest";
import { synchConsts, synchCoeffs, lnGamma, Q_E, M_E, C_CGS, SYN_P, GAMMA_MIN } from "../src/physics/synchrotron";
import { jetEnergetics, gammaProfile, plasmaShift, streamlineDir, jetUniforms, visLuminanceUnit, slabStep } from "../src/physics/synchrotron";
import { screenToState } from "../src/physics/camera";
import { lambdaFromMdot } from "../src/physics/units";

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

describe("jet model and energy budget", () => {
  it("M87*: Blandford-Znajek power ~0.98 Mdot c^2 and inside 1e43-1e44 erg/s (spec 2.1)", () => {
    const E = jetEnergetics(6.5e9, 0.9, lambdaFromMdot(6.5e9, 0.9, 7.7e-4));
    expect(E.pBZ).toBeGreaterThan(1e43); expect(E.pBZ).toBeLessThan(1e44);       // measured 4.26e43
    expect(E.pBZ / (E.mdot * C_CGS ** 2)).toBeCloseTo(0.977, 2);
  });
  it("Gamma(z) = max(1, G280 (|z|/280)^0.58)", () => {
    expect(gammaProfile(280, 2)).toBe(2); expect(gammaProfile(-280, 2)).toBe(2);
    expect(gammaProfile(60, 2)).toBe(1);                                          // 0.82 -> clamped
    expect(gammaProfile(1000, 2)).toBeCloseTo(2 * Math.pow(1000 / 280, 0.58), 12);
  });
  it("plasma frequency shift: static at r = 1000 ~1 (gravitational), toward/away Doppler (Review Focus 5)", () => {
    const s = screenToState(0, 0, 0, Math.PI / 2, 1000), b = Math.sqrt(1 - 1 / 9);
    expect(plasmaShift(s, 0, 1, 1, 0)).toBeCloseTo(1 / Math.sqrt(1 - 2 / 1000), 4);
    expect(plasmaShift(s, 0, 3, 1, 0) / (3 * (1 - b))).toBeCloseTo(1, 2);       // toward camera: blueshift
    expect(plasmaShift(s, 0, 3, -1, 0) / (3 * (1 + b))).toBeCloseTo(1, 2);      // counter-jet: redshift, finite
  });
  it("streamlines are unit vectors, mirror-symmetric between lobes, more collimated than radial", () => {
    const [ur, ut] = streamlineDir(20, 0.2), [lr, lt] = streamlineDir(20, Math.PI - 0.2);
    expect(Math.hypot(ur, ut)).toBeCloseTo(1, 12);
    expect(lr).toBeCloseTo(ur, 12); expect(lt).toBeCloseTo(-ut, 12);
    expect(ut).toBeLessThan(0); // upper lobe bends toward the axis relative to radial
  });
  it("energy budget: visible luminance at the chosen k_scale equals eps * P_BZ", () => {
    for (const [m, a, l] of [[6.5e9, 0.9, lambdaFromMdot(6.5e9, 0.9, 7.7e-4)], [1e8, 0.9, 3.66e-4], [21.2, 0.998, 0.02]]) {
      const U = jetUniforms(m, a, l, 2e-3, 60);
      expect((visLuminanceUnit(a, U.jetB0, U.rgCm, 60) * U.jetKScale) / (2e-3 * U.pBZ)).toBeCloseTo(1, 9);
      expect(jetUniforms(m, a, l, 4e-3, 60).jetKScale / U.jetKScale).toBeCloseTo(2, 9);
    }
  });
  it("slab transfer: thin -> j ds, thick -> source function j/alpha, earlier light attenuated (spec 5)", () => {
    const [It] = slabStep(0, 0, 2, 1e-9, 3);
    expect(It).toBeCloseTo(6, 6);                                                 // optically thin
    const [Ik, tk] = slabStep(0, 0, 2, 1e3, 3);
    expect(Ik).toBeCloseTo(2 / 1e3, 12); expect(tk).toBe(3e3);                    // saturates at S = j/alpha
    let I = 0, tau = 0;
    for (let k = 0; k < 400; k++) [I, tau] = slabStep(I, tau, 2, 0.5, 0.05);      // 400 thin pieces = one slab
    expect(I).toBeCloseTo((2 / 0.5) * (1 - Math.exp(-10)), 6);                    // exact uniform-slab result
    const [Ib] = slabStep(5, 1, 2, 1, 1);                                         // gas behind tau = 1 dims by e^-1
    expect(Ib).toBeCloseTo(5 + 2 * (1 - Math.exp(-1)) * Math.exp(-1), 12);
  });
  it("extremes stay finite and positive (Review Focus 3)", () => {
    for (const [m, l, e] of [[1, 1e-10, 1e-5], [1e10, 1e-10, 1e-1], [1, 1, 1e-1], [1e10, 1, 1e-5]]) {
      const U = jetUniforms(m, 0.9, l, e, 60);
      for (const v of [U.jetB0, U.jetKScale, U.rgCm]) { expect(Number.isFinite(v)).toBe(true); expect(v).toBeGreaterThan(0); }
    }
  });
});
