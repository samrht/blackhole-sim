import { describe, it, expect } from "vitest";
import { blackbodyLinearSRGB, blackbodyVisibleRGB, relLuminance } from "../src/physics/color";

describe("blackbody color", () => {
  it("returns 3 finite, non-negative channels normalized to luminance=1", () => {
    const c = blackbodyLinearSRGB(6500);
    expect(c.length).toBe(3);
    for (const x of c) { expect(Number.isFinite(x)).toBe(true); expect(x).toBeGreaterThanOrEqual(0); }
    const Y = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    expect(Y).toBeCloseTo(1, 6);
  });
  it("cool stars are red-dominant (r>b)", () => {
    const c = blackbodyLinearSRGB(3000);
    expect(c[0]).toBeGreaterThan(c[2]);
  });
  it("hot stars are blue-dominant (b>r)", () => {
    const c = blackbodyLinearSRGB(20000);
    expect(c[2]).toBeGreaterThan(c[0]);
  });
});

describe("visible-band radiance", () => {
  it("is the same colour as the normalised function, with luminance increasing in T", () => {
    for (const T of [2000, 5800, 3e4, 1e7]) {
      const v = blackbodyVisibleRGB(T), n = blackbodyLinearSRGB(T), L = relLuminance(v);
      for (let k = 0; k < 3; k++) expect(v[k] / L).toBeCloseTo(n[k], 10);
    }
    let prev = 0;
    for (let lt = 2; lt <= 9; lt += 0.05) {
      const L = relLuminance(blackbodyVisibleRGB(10 ** lt));
      expect(L).toBeGreaterThan(prev); prev = L;
    }
  });
});
