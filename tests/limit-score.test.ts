import { describe, it, expect } from "vitest";
import { obsVector, jacobianOf, rayError, summarizeArm, f32Inputs, roundObs, type Obs } from "../src/test/limit-score";
import { minoTrace, minoTraceRay, minoRay, MINO_TOL } from "../src/physics/trace-mino";

const disk = (r: number, p: number, t = -1010): Obs => ({ f: "disk", r, p, t, steps: 10, attempts: 11 });
const sky = (dir: number[]): Obs => ({ f: "escaped", dir, steps: 10, attempts: 11 });

describe("observables", () => {
  it("a disk hit is its Cartesian point, so phi wrap-around is not an error", () => {
    const u = obsVector(disk(5, 0.001))!, v = obsVector(disk(5, 2 * Math.PI + 0.001))!;
    expect(Math.hypot(u[0] - v[0], u[1] - v[1])).toBeLessThan(1e-12);
  });
  it("a sky hit is its direction; captured rays have no vector", () => {
    expect(obsVector(sky([1, 0, 0]))).toEqual([1, 0, 0]);
    expect(obsVector({ f: "captured", steps: 1, attempts: 1 })).toBeNull();
  });
});

describe("image-plane error", () => {
  // Observable = (2 alpha, 3 beta) per pixel: a known linear map.
  const J = jacobianOf([2, 0], [-2, 0], [0, 3], [0, -3], 1)!;
  it("the Jacobian is the central difference per pixel", () => { expect(J).toEqual([[2, 0], [0, 3]]); });
  it("no Jacobian without all four neighbours", () => { expect(jacobianOf(null, [0, 0], [0, 0], [0, 0], 1)).toBeUndefined(); });
  it("maps an observable error back to on-screen pixels", () => {
    const e = rayError(disk(1, 0), J, disk(1 + 2 * 0.5, 0)); // dx = 1 -> 0.5 px through d(x)/d(alpha) = 2
    expect(e.flip).toBe(false); expect(e.px).toBeCloseTo(0.5, 10);
  });
  it("reports the delay error of disk rays in M", () => {
    expect(rayError(disk(5, 0, -1010), J, disk(5, 0, -1010.25)).delay).toBeCloseTo(0.25, 10);
  });
  it("a different fate is a flip with no pixel error", () => {
    const e = rayError(disk(5, 0), J, { f: "captured", steps: 1, attempts: 1 });
    expect(e.flip).toBe(true); expect(e.px).toBeNaN();
  });
  it("a captured ray matching a captured reference has no error to score", () => {
    const e = rayError({ f: "captured", steps: 1, attempts: 1 }, undefined, { f: "captured", steps: 1, attempts: 1 });
    expect(e.flip).toBe(false); expect(e.px).toBeNaN();
  });
});

describe("arm summary", () => {
  it("bins by distance from the critical curve and counts flips and unscored rays", () => {
    const s = summarizeArm([0.5, 2, 0.5, 2], [
      { flip: false, px: 1e-3, delay: NaN, steps: 10, attempts: 12 }, { flip: false, px: 3e-3, delay: 0.1, steps: 30, attempts: 30 },
      { flip: true, px: NaN, delay: NaN, steps: 20, attempts: 20 }, { flip: false, px: NaN, delay: NaN, steps: 20, attempts: 20 }]);
    expect(s.flips).toBe(1); expect(s.meanSteps).toBe(20); expect(s.meanAttempts).toBe(20.5);
    expect(s.px.n).toBe(2); expect(s.px.max).toBe(3e-3);
    expect(s.byCurve["[0.1, 1)"].px.n).toBe(1); expect(s.byCurve["[1, Infinity)"].px.max).toBe(3e-3);
    expect(s.delay.n).toBe(1);
  });
});

describe("the f32 inputs arm", () => {
  it("rounds what the shader rounds", () => {
    const f = f32Inputs(3.25, -2.5, 0.998, (85 * Math.PI) / 180);
    for (const v of [f.a, f.incl, f.xi, f.eta, f.beta]) expect(Math.fround(v)).toBe(v);
    const xi64 = -3.25 * Math.sin((85 * Math.PI) / 180);
    expect(Math.abs(f.xi - xi64)).toBeLessThan(1e-6); expect(f.xi).not.toBe(xi64);
  });
});

describe("minoTraceRay", () => {
  it("is minoTrace with the ray given directly", () => {
    const a = 0.9, inc = 72, i = (inc * Math.PI) / 180, al = 6, be = 3, xi = -al * Math.sin(i);
    const eta = be * be + (xi * xi * Math.cos(i) ** 2) / Math.sin(i) ** 2 - a * a * Math.cos(i) ** 2;
    const o = { rIn: 2.32, rOut: 40, rObs: 1000, tol: MINO_TOL };
    const u = minoTrace(al, be, a, inc, o), v = minoTraceRay(minoRay(a, xi, eta, 1000), i, be, o);
    expect(v.fate).toBe(u.fate); expect(v.rHit).toBe(u.rHit); expect(v.phiHit).toBe(u.phiHit); expect(v.steps).toBe(u.steps);
  });
});

describe("output rounding and groups", () => {
  it("roundObs stores what the renderer would store, in f32", () => {
    const o = roundObs({ f: "disk", r: 5.123456789, p: 1.23456789, t: -1012.3456789, steps: 1, attempts: 1 });
    expect(Math.fround(o.r!)).toBe(o.r); expect(Math.fround(o.p!)).toBe(o.p);
    expect(Math.fround(-o.t! - 1000)).toBeCloseTo(-o.t! - 1000, 10); // the delay is the stored f32, not t
    const s = roundObs({ f: "escaped", dir: [0.6000000001, 0.7999999999, 0], steps: 1, attempts: 1 });
    expect(s.dir!.every((v) => Math.fround(v) === v)).toBe(true);
  });
  it("summarizeArm reports named groups of rays", () => {
    const rows = [{ flip: false, px: 1, delay: NaN, steps: 1, attempts: 1 }, { flip: false, px: 3, delay: NaN, steps: 1, attempts: 1 }];
    const s = summarizeArm([1, 1], rows, { axis: [true, false], other: [false, true] });
    expect(s.groups!.axis.px.max).toBe(1); expect(s.groups!.other.px.max).toBe(3); expect(s.groups!.axis.rays).toBe(1);
  });
});
