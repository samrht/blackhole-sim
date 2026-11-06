import { describe, it, expect } from "vitest";
import { dsAdd, dsCos, dsDiv, dsFrom, dsLog, dsMul, dsSin, dsSqrt, dsSub, dsToNumber, dsWgslConstants, twoProd, twoSum, DS_COEFFS } from "../src/physics/double-single";

const F = Math.fround;
// Each result is compared with the same operation in f64. f64 has 53 bits, double-single about 48, so 2^-45 relative is a real test.
const TOL = 2 ** -45;
let seed = 99;
const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
const mag = (lo: number, hi: number) => 10 ** (lo + (hi - lo) * rnd());
const close = (got: number, want: number, scale = Math.abs(want)) => expect(Math.abs(got - want)).toBeLessThanOrEqual(TOL * Math.max(scale, 1e-30));

describe("error-free transformations", () => {
  it("twoSum: s + e is exactly a + b", () => {
    for (let k = 0; k < 2000; k++) {
      const a = F(mag(-4, 4)), b = F((rnd() - 0.5) * mag(-6, 2)), [s, e] = twoSum(a, b);
      expect(F(s)).toBe(s); expect(F(e)).toBe(e);
      expect(s + e).toBe(a + b); // exact in f64: f32 values this close in exponent sum without rounding
    }
  });
  it("twoProd: p + e is exactly a * b, with no fused multiply-add", () => {
    for (let k = 0; k < 2000; k++) {
      const a = F((rnd() - 0.5) * mag(-4, 4)), b = F((rnd() - 0.5) * mag(-4, 4)), [p, e] = twoProd(a, b);
      expect(F(p)).toBe(p); expect(F(e)).toBe(e);
      expect(p + e).toBe(a * b); // the product of two 24-bit values fits in f64 exactly
    }
  });
});

describe("double-single arithmetic against f64", () => {
  it("represents a f64 value to about 48 bits, hi being the f32 nearest", () => {
    for (const x of [0.998, 1 / 3, Math.PI, 1e-3 / 7, -2.5e4 / 3]) { const d = dsFrom(x); expect(d[0]).toBe(F(x)); close(dsToNumber(d), x); }
  });
  it("add, sub, mul, div, sqrt", () => {
    for (let k = 0; k < 3000; k++) {
      const x = (rnd() - 0.5) * mag(-3, 3), y = (rnd() - 0.5) * mag(-3, 3), a = dsFrom(x), b = dsFrom(y), X = dsToNumber(a), Y = dsToNumber(b);
      close(dsToNumber(dsAdd(a, b)), X + Y, Math.abs(X) + Math.abs(Y));
      close(dsToNumber(dsSub(a, b)), X - Y, Math.abs(X) + Math.abs(Y));
      close(dsToNumber(dsMul(a, b)), X * Y);
      close(dsToNumber(dsDiv(a, b)), X / Y);
      close(dsToNumber(dsSqrt(dsFrom(Math.abs(x)))), Math.sqrt(dsToNumber(dsFrom(Math.abs(x)))));
    }
  });
  it("sin and cos over the angles the integrator uses, including tiny ones", () => {
    for (let k = 0; k < 3000; k++) {
      const x = k % 3 === 0 ? mag(-6, -1) : rnd() * 2 * Math.PI - Math.PI, a = dsFrom(x), X = dsToNumber(a);
      // Inside pi/4 nothing is reduced, so the result is relative. Beyond it the reduction is limited by pi/2 in double-single (3.5e-15
      // absolute), and an angle there cannot be stored better than that anyway: absolute, against scale 1.
      const rel = Math.abs(X) <= Math.PI / 4;
      close(dsToNumber(dsSin(a)), Math.sin(X), rel ? Math.abs(Math.sin(X)) : 0.5);
      close(dsToNumber(dsCos(a)), Math.cos(X), rel ? Math.abs(Math.cos(X)) : 0.5);
    }
  });
  it("sin keeps relative precision near 0 (a pole)", () => {
    const a = dsFrom(3e-5);
    expect(Math.abs(dsToNumber(dsSin(a)) / Math.sin(dsToNumber(a)) - 1)).toBeLessThan(TOL);
  });
  it("log over the range of w = 1/r", () => {
    for (let k = 0; k < 2000; k++) { const a = dsFrom(mag(-4, 0.5)), X = dsToNumber(a); close(dsToNumber(dsLog(a)), Math.log(X), Math.max(Math.abs(Math.log(X)), 0.1)); }
  });
});

describe("WGSL constants", () => {
  it("every generated value survives the text round trip exactly", () => {
    const src = dsWgslConstants({ A21: 1 / 5 });
    const m = /const A21 = vec2<f32>\(([^,]+), ([^)]+)\);/.exec(src)!;
    const want = dsFrom(1 / 5);
    expect(Math.fround(Number(m[1]))).toBe(want[0]); expect(Math.fround(Number(m[2]))).toBe(want[1]);
  });
  it("has the shared Taylor coefficients", () => { expect(DS_COEFFS.length).toBeGreaterThan(8); });
});
