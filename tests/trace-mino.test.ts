import { describe, it, expect } from "vitest";
import { minoRay, minoInit, minoRhs, minoStep, minoDense, minoToState, minoCrossing, minoTrace, minoRw, minoU, MINO_TOL, MINO_UFRAC } from "../src/physics/trace-mino";
import { screenToState, screenToXiEta } from "../src/physics/camera";
import { rhs } from "../src/physics/geodesic";
import { metricUpper } from "../src/physics/kerr";
import { traceRay, H_TOL } from "../src/physics/trace";
import { MINO_MAX_REJECT } from "../src/physics/trace-mino";
import { readFileSync } from "node:fs";
import { join } from "node:path";

describe("WGSL twin constants (integrator-shared.wgsl)", () => {
  const W = readFileSync(join(__dirname, "../src/render/integrator-shared.wgsl"), "utf8");
  const wconst = (n: string) => { const m = W.match(new RegExp(`const ${n}\\s*=\\s*([^;]+);`)); if (!m) throw new Error(`no ${n}`); return Number(m[1].trim().replace(/u$/, "")); };
  it("match trace-mino.ts", () => {
    expect(wconst("MINO_TOL")).toBe(MINO_TOL);
    expect(wconst("MINO_UFRAC")).toBe(MINO_UFRAC);
    expect(wconst("MINO_MAX_REJECT")).toBe(MINO_MAX_REJECT);
  });
});

// Carter's radial potential in r (the textbook form) and its w = 1/r version, cross-checked below.
const Rr = (r: number, c: ReturnType<typeof minoRay>) => { const P = r * r + c.a * c.a - c.a * c.xi, D = r * r - 2 * r + c.a * c.a; return P * P - D * c.K; };
const R = (y: Float64Array, c: ReturnType<typeof minoRay>) => minoRw(y[1], c);
const U = (y: Float64Array, c: ReturnType<typeof minoRay>) => minoU(y[2], c);
const setup = (al: number, be: number, a: number, iDeg: number) => {
  const i = (iDeg * Math.PI) / 180, [xi, eta] = screenToXiEta(al, be, a, i), c = minoRay(a, xi, eta);
  return { c, y: minoInit(1000, i, be, c), s: screenToState(al, be, a, i, 1000) };
};

describe("Mino-time integrator (spec 2026-10-07)", () => {
  it("initial state satisfies both first integrals and maps to the camera State", () => {
    for (const [al, be, a, i] of [[3, 2, 0.9, 72], [0.1, 6, 0, 8], [-5, -1, 0.998, 85]] as const) {
      const { c, y, s } = setup(al, be, a, i);
      expect(Math.abs(y[5] * y[5] / R(y, c) - 1)).toBeLessThan(1e-12); expect(y[5]).toBeGreaterThan(0); // w grows inward
      for (const r of [1000, 37, 2.5]) expect(minoRw(1 / r, c)).toBeCloseTo(Rr(r, c) / r ** 4, 12);
      expect(Math.abs(y[6] * y[6] - U(y, c))).toBeLessThan(1e-12 * Math.max(1, U(y, c)));
      const s2 = minoToState(y, c);
      for (let k = 0; k < 8; k++) expect(s2[k]).toBeCloseTo(s[k], 8);
    }
  });
  it("RHS equals Sigma x the Hamiltonian velocities (t, r, theta, phi) and R'/2, U'/2", () => {
    for (const [al, be, a, i] of [[3, 2, 0.9, 72], [-4, 5, 0.5, 30]] as const) {
      const { c } = setup(al, be, a, i);
      const r = 7.3, y = new Float64Array([-990, 1 / r, 0.31, 0.4, 0, 0, 0]);
      y[5] = Math.sqrt(R(y, c)); y[6] = Math.sqrt(Math.max(0, U(y, c)));
      const f = minoRhs(y, c), s = minoToState(y, c), h = rhs(s, a), Sig = r * r + a * a * y[2] ** 2;
      expect(s[1]).toBeCloseTo(r, 12);
      expect(f[0]).toBeCloseTo(Sig * h[0], 9); expect(-f[1] * r * r).toBeCloseTo(Sig * h[1], 9); // w' = -r'/r^2
      expect(f[3]).toBeCloseTo(Sig * h[3], 9); expect(f[4]).toBeCloseTo(Sig, 12);
      expect(f[2]).toBeCloseTo(-Math.sin(s[2]) * Sig * h[2], 9);
      const e = 1e-6, yp = y.slice(), ym = y.slice(); yp[1] += e; ym[1] -= e;
      expect(f[5]).toBeCloseTo((R(yp, c) - R(ym, c)) / (4 * e), 4);
      const up = y.slice(), um = y.slice(); up[2] += e; um[2] -= e;
      expect(f[6]).toBeCloseTo((U(up, c) - U(um, c)) / (4 * e), 6);
    }
  });
  it("a step keeps the first integrals and the null condition (f64)", () => {
    const { c, y } = setup(3, 2, 0.9, 72);
    let s = y, h = 1e-5;
    for (let k = 0; k < 400 && 1 / s[1] > 3; k++) { const o = minoStep(s, h, c, MINO_TOL, MINO_UFRAC); s = o.y; h = o.hNext; }
    expect(1 / s[1]).toBeLessThan(3.5);
    expect(Math.abs(s[5] ** 2 - R(s, c))).toBeLessThan(1e-6);
    const st = minoToState(s, c), g = metricUpper(st[1], st[2], 0.9);
    const H = g.tt + 2 * g.tphi * st[7] + g.rr * st[5] ** 2 + g.thth * st[6] ** 2 + g.phph * st[7] ** 2;
    expect(Math.abs(H) / Math.abs(g.tt)).toBeLessThan(1e-6);
  });
  it("dense output is exact at the ends and a cubic between; the u = 0 root is found", () => {
    const { c, y } = setup(0, -3, 0.5, 80); // u' < 0: heads for the equator before anything else
    let s = y, h = 1e-5, found = false;
    for (let k = 0; k < 4000 && !found && 1 / s[1] > 2; k++) {
      const o = minoStep(s, h, c, MINO_TOL, MINO_UFRAC);
      const d0 = minoDense(s, o.y, o.f0, o.f1, o.h, 0), d1 = minoDense(s, o.y, o.f0, o.f1, o.h, 1);
      for (let j = 0; j < 7; j++) { expect(d0[j]).toBe(s[j]); expect(d1[j]).toBeCloseTo(o.y[j], 12); }
      const th = minoCrossing(s, o.y, o.f0, o.f1, o.h);
      if (th >= 0) { found = true; expect(Math.abs(minoDense(s, o.y, o.f0, o.f1, o.h, th)[2])).toBeLessThan(1e-9); }
      s = o.y; h = o.hNext;
    }
    expect(found).toBe(true);
  });
  it("a xi = 0 ray through the pole gains pi of azimuth, as the Hamiltonian reflection does", () => {
    // alpha = 0 at i = 8 deg: xi = 0 exactly, beta = 6 aims across the axis; it crosses the pole once, then the disk.
    const i = 8, o = { rIn: 3, rOut: 40, rObs: 1000 };
    const old = traceRay(screenToState(0, 6, 0, (i * Math.PI) / 180, 1000), 0, { ...o, hTol: 1e-9 });
    const nu = minoTrace(0, 6, 0, i, { ...o, tol: 1e-10 });
    expect(nu.fate).toBe("disk"); expect(old.fate).toBe("disk");
    const d = Math.atan2(Math.sin(nu.phiHit! - old.phiHit!), Math.cos(nu.phiHit! - old.phiHit!));
    expect(Math.abs(d)).toBeLessThan(1e-3);
    expect(Math.abs(nu.rHit! - old.rHit!)).toBeLessThan(1e-3);
  });
  it("whole rays agree with today's integrator on fate and disk radius (default view)", () => {
    for (const [al, be] of [[3, 2], [-6, 1], [0.5, -7], [10, 10]]) {
      const i = (72 * Math.PI) / 180, o = { rIn: 2.32, rOut: 40, rObs: 1000 };
      const old = traceRay(screenToState(al, be, 0.9, i, 1000), 0.9, { ...o, hTol: H_TOL });
      const nu = minoTrace(al, be, 0.9, 72, o);
      expect(nu.fate).toBe(old.fate);
      if (old.fate === "disk") expect(Math.abs(nu.rHit! - old.rHit!)).toBeLessThan(0.02);
    }
  });
});
