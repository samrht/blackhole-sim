import { describe, it, expect } from "vitest";
import { buildTempLUT, buildColorLUT } from "../src/physics/lookups";

describe("lookups", () => {
  it("temp LUT: length N, ~0 at inner edge, has an interior peak of 1", () => {
    const lut = buildTempLUT(0, true, 6, 40, 256);
    expect(lut.length).toBe(256);
    expect(lut[0]).toBeLessThan(0.1);
    expect(Math.max(...lut)).toBeCloseTo(1, 5);
    const argmax = lut.indexOf(Math.max(...lut));
    expect(argmax).toBeGreaterThan(0);          // peak is not at the inner edge
    expect(argmax).toBeLessThan(255);
  });
  it("color LUT: length 4N (RGBA), all finite in [0,1]", () => {
    const lut = buildColorLUT(1000, 40000, 64);
    expect(lut.length).toBe(64 * 4);
    for (const x of lut) { expect(x).toBeGreaterThanOrEqual(0); expect(x).toBeLessThanOrEqual(1); }
  });
});
