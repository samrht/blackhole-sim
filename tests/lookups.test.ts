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
  it("color LUT: length 4N (RGBA), all finite and non-negative", () => {
    const lut = buildColorLUT(1000, 40000, 64);
    expect(lut.length).toBe(64 * 4);
    // Entries are luminance-normalized, so an individual channel may exceed 1 (a 40000 K blue is
    // ~4.3 in blue). That is intentional and safe: the LUT is an f32 storage buffer, not unorm8.
    for (const x of lut) { expect(Number.isFinite(x)).toBe(true); expect(x).toBeGreaterThanOrEqual(0); }
  });
});
