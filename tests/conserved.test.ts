import { describe, it, expect } from "vitest";
import { minoHQ, driftMetric, minoQ, preciseDrift } from "../src/physics/conserved";
import { minoInit, minoRay, minoRw, minoTheta, minoTrace, minoHemi } from "../src/physics/trace-mino";

const rad = (d: number) => (d * Math.PI) / 180;
/** The camera's (xi, eta): trace-mino.ts minoTrace's expression. */
function ray(alpha: number, beta: number, a: number, inclDeg: number, r0 = 1000) {
  const i = rad(inclDeg), si = Math.sin(i), ci = Math.cos(i), xi = -alpha * si;
  const eta = beta * beta + (xi * xi * ci * ci) / (si * si) - a * a * ci * ci;
  return { c: minoRay(a, xi, eta, r0), i, xi, eta };
}
/** A state ON the null cone at (r, theta), from the exact first integrals. */
function onShell(r: number, th: number, alpha: number, beta: number, a: number, inclDeg: number) {
  const { c } = ray(alpha, beta, a, inclDeg), w = 1 / r;
  return { c, y: new Float64Array([0, w, th, 0, 0, Math.sqrt(minoRw(w, c)), Math.sqrt(Math.max(0, minoTheta(th, c))), 1]) };
}

describe("conserved quantities of a Mino state", () => {
  it("Q at the camera is the ray's eta and H is 0", () => {
    for (const [al, be, a, inc] of [[3, 4, 0.998, 85], [-7, 1, 0.9, 72], [0.5, -6, 0.5, 30], [9, 9, 0.998, 17]]) {
      const { c, i, eta } = ray(al, be, a, inc), y = minoInit(1000, i, be, c);
      expect(minoQ(y, c).q).toBeCloseTo(eta, 10);
      const m = minoHQ(y, c);
      expect(Math.abs(m.h)).toBeLessThan(1e-12);
    }
  });

  it("H vanishes on the null cone right down to the horizon", () => {
    for (const r of [500, 10, 2, 1.1, 1.07]) {
      const { y, c } = onShell(r, 1.2, 1.5, 0.8, 0.998, 85), m = minoHQ(y, c);
      expect(Math.abs(m.h)).toBeLessThan(1e-12 * Math.max(1, m.hScale));
    }
  });

  it("Q responds to theta' as 2 theta' d, and is even about the equator", () => {
    const { y, c } = onShell(5, 1.2, 1.5, 0.8, 0.998, 85), d = 1e-6;
    const q0 = minoQ(y, c).q, yp = y.slice(); yp[6] += d;
    expect(minoQ(yp, c).q - q0).toBeCloseTo(2 * y[6] * d, 11);
    const south = minoHemi(new Float64Array([0, y[1], Math.PI - 1.2, 0, 0, y[5], y[6], 1]));
    expect(minoQ(south, c).q).toBeCloseTo(q0, 12);
  });

  it("an error in the radial first integral reaches H amplified by r^4 / (2 Sigma Delta)", () => {
    const a = 0.998, th = 1.2;
    for (const r of [50, 4, 1.07]) {
      const { y, c } = onShell(r, th, 1.5, 0.8, a, 85), d = 1e-8, yp = y.slice(); yp[5] += d;
      const dW = yp[5] * yp[5] - y[5] * y[5];
      const Sig = r * r + a * a * Math.cos(th) ** 2, Del = r * r - 2 * r + a * a;
      const got = minoHQ(yp, c).h - minoHQ(y, c).h;
      expect(got / ((dW * r ** 4) / (2 * Sig * Del))).toBeCloseTo(1, 4);
    }
  });

  it("reports the radial first integral w'^2 - R~(w)", () => {
    const { y, c } = onShell(5, 1.2, 1.5, 0.8, 0.998, 85), yp = y.slice(); yp[5] += 1e-6;
    const m = minoHQ(yp, c);
    expect(m.wRes).toBeCloseTo(yp[5] * yp[5] - minoRw(yp[1], c), 15);
  });

  it("the drift metric is |dQ| / |Q0| + |H_end|", () => {
    expect(driftMetric(2, 2.002, 3e-4)).toBeCloseTo(1e-3 + 3e-4, 12);
    expect(driftMetric(-2, -2.002, -3e-4)).toBeCloseTo(1e-3 + 3e-4, 12);
    expect(driftMetric(0, 1e-9, 0)).toBe(Infinity); // Q0 = 0 has no relative drift: callers must treat it as undefined
  });

  it("minoTrace hands back the end state, and its f64 drift is small for each fate", () => {
    const a = 0.998;
    for (const [al, be, fate] of [[8, 3, "disk"], [0.5, 0.5, "captured"], [13, 12, "escaped"]] as const) {
      const { c, eta, i } = ray(al, be, a, 85);
      const rIn = 1.237, res = minoTrace(al, be, a, 85, { rIn, rOut: 40, rObs: 1000 });
      expect(res.fate).toBe(fate);
      const y0 = minoInit(1000, i, be, c), q0 = minoQ(y0, c).q, m = minoHQ(res.y, c);
      expect(q0).toBeCloseTo(eta, 10);
      expect(driftMetric(q0, minoQ(res.y, c).q, m.h)).toBeLessThan(1e-2);
    }
  });

  it("f32 evaluation alone sets an H floor that grows toward the horizon", () => {
    const fr = Math.fround;
    const far = onShell(500, 1.2, 1.5, 0.8, 0.998, 85), near = onShell(1.07, 1.2, 1.5, 0.8, 0.998, 85);
    const floorFar = minoHQ(far.y, far.c, fr), floorNear = minoHQ(near.y, near.c, fr);
    expect(floorNear.hScale).toBeGreaterThan(100 * floorFar.hScale);
    expect(Math.abs(floorFar.h)).toBeLessThan(1e-6);
  });
});

describe("precise measure of a stored f32 end state", () => {
  it("is the f64 measure of those f32 numbers, and removes the f32 evaluation error near the horizon", () => {
    const a = 0.998, fr = Math.fround;
    const { y, c } = onShell(1.068, 1.2, 1.5, 0.8, a, 85);
    const end = { w: fr(y[1]), wp: fr(y[5]), v: fr(y[2]), vp: fr(y[6]) };
    const ray32 = { a: fr(a), xi: fr(c.xi), eta: fr(c.eta), r0: 1000 };
    const p = preciseDrift(end, { v: fr(1.48), vp: fr(0.8) }, ray32);
    const c32 = minoRay(ray32.a, ray32.xi, ray32.eta, 1000), y32 = Float64Array.from([0, end.w, end.v, 0, 0, end.wp, end.vp, 1]);
    expect(p.h1).toBe(minoHQ(y32, c32).h);
    expect(p.q1).toBe(minoQ(y32, c32).q);
    expect(p.q0).toBe(minoQ(Float64Array.from([0, 0, fr(1.48), 0, 0, 0, fr(0.8), 1]), c32).q);
    expect(Math.abs(p.h1)).toBeLessThan(0.1 * Math.abs(minoHQ(y32, c32, fr).h));
    expect(p.total).toBe(driftMetric(p.q0, p.q1, p.h1));
  });
});
