import { describe, it, expect } from "vitest";
import { blackbodyLinearSRGB } from "../src/physics/color";

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
