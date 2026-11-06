import { describe, it, expect, afterEach } from "vitest";
import { minoHemi, minoInit, minoRay, minoRayF32, minoRhs, minoRhsF32, minoTrace, minoTraceRay, minoTry, setMinoRounding } from "../src/physics/trace-mino";
import { f32Inputs } from "../src/test/limit-score";

const a = 0.9, inc = 72, i = (inc * Math.PI) / 180, al = 6, be = 3, xi = -al * Math.sin(i);
const eta = be * be + (xi * xi * Math.cos(i) ** 2) / Math.sin(i) ** 2 - a * a * Math.cos(i) ** 2;
const c = minoRay(a, xi, eta, 1000), F = Math.fround;
const o = { rIn: 2.32, rOut: 40, rObs: 1000, tol: 1e-6 };
const isF32 = (v: number) => F(v) === v;
afterEach(() => setMinoRounding());

describe("rounding switches", () => {
  it("off, the trace is bit-identical to no switch at all", () => {
    const u = minoTrace(al, be, a, inc, o);
    setMinoRounding({ rhs: true, incr: true, store: true, pi: true }); minoTrace(al, be, a, inc, o); setMinoRounding();
    const v = minoTrace(al, be, a, inc, o);
    expect(v.rHit).toBe(u.rHit); expect(v.phiHit).toBe(u.phiHit); expect(v.tHit).toBe(u.tHit); expect(v.steps).toBe(u.steps);
  });
  it("store: a step's result is f32 in every component", () => {
    setMinoRounding({ store: true });
    const y = minoInit(1000, i, be, c).map(F) as Float64Array, r = minoTry(y, 1e-3, c, minoRhs(y, c));
    expect([...r.y1].every(isF32)).toBe(true);
  });
  it("store off: the state is not rounded even with the other switches on", () => {
    setMinoRounding({ rhs: true, incr: true });
    const y = minoInit(1000, i, be, c), r = minoTry(y, 1e-3, c, minoRhsF32(y, c));
    expect([...r.y1].slice(0, 7).some((v) => !isF32(v))).toBe(true);
  });
  it("rhs: the force is computed in f32, close to the f64 force", () => {
    const y = minoInit(1000, i, be, c); y[1] = 0.1; y[5] = 0.8; y[6] = 0.5;
    const f = minoRhsF32(y, minoRayF32(F(a), F(xi), F(eta), 1000)), g = minoRhs(y, c);
    expect([...f].every(isF32)).toBe(true);
    for (const k of [1, 2, 3, 5, 6]) expect(Math.abs(f[k] - g[k])).toBeLessThan(1e-5 * Math.max(1, Math.abs(g[k])));
    expect([...f]).not.toEqual([...g]);
  });
  it("pi: the mirror uses the f32 value of pi, 8.74e-8 off", () => {
    const y = minoInit(1000, i, be, c); y[2] = 2;
    const exact = minoHemi(y)[2]; setMinoRounding({ pi: true });
    expect(minoHemi(y)[2] - exact).toBeCloseTo(F(Math.PI) - Math.PI, 12);
  });
  it("minoRayF32 is minoRay in f32, op by op", () => {
    const u = minoRayF32(F(a), F(xi), F(eta), 1000), v = minoRay(F(a), F(xi), F(eta), 1000);
    for (const k of ["K", "A1", "c2", "c3", "c4"] as const) { expect(isF32(u[k])).toBe(true); expect(Math.abs(u[k] - v[k])).toBeLessThan(1e-5 * Math.max(1, Math.abs(v[k]))); }
    expect(u.tp.every(isF32)).toBe(true);
  });
  it("all switches on, a ray still lands where it should, to about f32 accuracy", () => {
    const exact = minoTrace(al, be, a, inc, { ...o, tol: 1e-11 });
    setMinoRounding({ rhs: true, incr: true, store: true, pi: true });
    const q = f32Inputs(al, be, a, i), r = minoTraceRay(minoRayF32(q.a, q.xi, q.eta, 1000), q.incl, q.beta, o);
    expect(r.fate).toBe(exact.fate);
    expect(Math.abs(r.rHit! - exact.rHit!)).toBeLessThan(1e-3); expect(r.rHit).not.toBe(exact.rHit);
  });
});

describe("the drift check's noise floor", () => {
  it("eps alone keeps exact arithmetic but takes fewer steps, because the check is weaker", () => {
    const u = minoTrace(al, be, a, inc, { ...o, tol: 1e-5 });
    setMinoRounding({ eps: true });
    const v = minoTrace(al, be, a, inc, { ...o, tol: 1e-5 });
    setMinoRounding();
    expect(v.attempts).toBeLessThan(u.attempts);
    expect(F(v.rHit!)).not.toBe(v.rHit); // no rounding anywhere in the arithmetic
  });
});
