import { describe, it, expect } from "vitest";
import { buildTempLUT, buildVisibleLUT, sampleVisibleLUT, visibleLuminance, lumNormFor, VIS_LUT_N, VIS_TMIN } from "../src/physics/lookups";
import { blackbodyVisibleRGB, relLuminance } from "../src/physics/color";

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
  it("visible LUT: RGBA, finite, non-negative; log-T lookup within 0.5 % of direct evaluation", () => {
    const lut = buildVisibleLUT();
    expect(lut.length).toBe(VIS_LUT_N * 4);
    for (const x of lut) { expect(Number.isFinite(x)).toBe(true); expect(x).toBeGreaterThanOrEqual(0); }
    const ref = relLuminance(blackbodyVisibleRGB(1e4));
    for (let lt = Math.log10(1500); lt < 9; lt += 0.0371) { // off-grid sample points
      const T = 10 ** lt, got = relLuminance(sampleVisibleLUT(lut, T)), want = relLuminance(blackbodyVisibleRGB(T)) / ref;
      expect(Math.abs(got / want - 1)).toBeLessThan(5e-3);
    }
  });
  it("lumNorm normalises the peak temperature to luminance 1; capped at 1e30 for cold disks", () => {
    for (const T of [4100, 3e4, 6e6]) expect(lumNormFor(T) * visibleLuminance(T)).toBeCloseTo(1, 10);
    // A 124 K disk (Mass 1e10, Accretion 1e-10) has visible luminance ~1e-90: uncapped, lumNorm would
    // be Infinity in f32 and Infinity * 0 = NaN on the GPU (Review Focus 2).
    expect(lumNormFor(124)).toBe(1e30);
    expect(lumNormFor(VIS_TMIN / 2)).toBe(1e30);
    expect(lumNormFor(400)).toBeLessThan(1e30);
  });
});
