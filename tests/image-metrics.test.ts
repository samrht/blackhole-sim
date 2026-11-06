import { describe, it, expect } from "vitest";
import { compareDisplay, compareLinear, pixelRays, IMAGE_VIEWS } from "../src/test/image-metrics";

const rgba = (rows: number[][]) => Float32Array.from(rows.flat());

describe("linear image comparison", () => {
  it("is zero for identical images", () => {
    const a = rgba([[1, 1, 1, 1], [2, 2, 2, 1]]), c = compareLinear(a, a.slice());
    expect(c.fluxRel).toBe(0); expect(c.pxRel.max).toBe(0); expect(c.nonFinite).toBe(0);
  });
  it("reports the total flux difference and the per-pixel relative difference in luminance", () => {
    const ref = rgba([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]), got = rgba([[1.1, 1.1, 1.1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]);
    const c = compareLinear(ref, got);
    expect(c.fluxRel).toBeCloseTo(0.025, 6);
    expect(c.pxRel.max).toBeCloseTo(0.1, 5);
  });
  it("does not divide by a dark reference pixel: dark pixels are measured against a floor of the mean", () => {
    const ref = rgba([[0, 0, 0, 1], [1000, 1000, 1000, 1]]), got = rgba([[1, 1, 1, 1], [1000, 1000, 1000, 1]]);
    const c = compareLinear(ref, got); // floor = 1e-3 of the mean luminance (500) = 0.5
    expect(c.pxRel.max).toBeCloseTo(2, 5);
  });
  it("counts non-finite pixels rather than letting them poison the statistics", () => {
    const c = compareLinear(rgba([[1, 1, 1, 1], [1, 1, 1, 1]]), rgba([[NaN, 1, 1, 1], [1, 1, 1, 1]]));
    expect(c.nonFinite).toBe(1); expect(Number.isFinite(c.pxRel.max)).toBe(true);
  });
});

describe("displayed (8-bit) comparison", () => {
  it("counts pixels by the largest channel difference, ignoring alpha", () => {
    const ref = Uint8Array.from([10, 10, 10, 255, 10, 10, 10, 255, 10, 10, 10, 255, 10, 10, 10, 255]);
    const got = Uint8Array.from([10, 10, 10, 0, 11, 10, 10, 255, 10, 13, 10, 255, 10, 10, 20, 255]);
    const c = compareDisplay(ref, got);
    expect(c.pixels).toBe(4); expect(c.ge1).toBe(3); expect(c.ge2).toBe(2); expect(c.ge5).toBe(1); expect(c.maxDiff).toBe(10);
  });
  it("PSNR is infinite for identical images and falls as the error grows", () => {
    const ref = Uint8Array.from([10, 10, 10, 255, 200, 200, 200, 255]);
    expect(compareDisplay(ref, ref.slice()).psnr).toBe(Infinity);
    const a = compareDisplay(ref, Uint8Array.from([11, 10, 10, 255, 200, 200, 200, 255])).psnr, b = compareDisplay(ref, Uint8Array.from([30, 10, 10, 255, 200, 200, 200, 255])).psnr;
    expect(a).toBeGreaterThan(b); expect(b).toBeGreaterThan(0);
  });
});

describe("pixel rays", () => {
  it("are raytrace.wgsl's pixelImpact in f32, with the jitter and pan", () => {
    const r = pixelRays(4, 2, 14, 1, 0.5, [0.25, -0.25]);
    expect(r.length).toBe(16);
    // pixel (1, 0): ndc = (1 + 0.5 + 0.25) / 4 * 2 - 1 = -0.125, alpha = -0.125 * 14 * 2 + 1
    expect(r[2]).toBeCloseTo(-2.5, 5);
    // ndc y = (0 + 0.5 - 0.25) / 2 * 2 - 1 = -0.75, beta = 0.75 * 14 + 0.5
    expect(r[3]).toBeCloseTo(11, 5);
    expect(r.every((v) => Math.fround(v) === v)).toBe(true);
  });
});

describe("views", () => {
  it("cover the ordinary views and the two 16x zooms", () => {
    expect(IMAGE_VIEWS.map((v) => v.name)).toEqual(["default", "edge", "schw", "zoom-pole", "zoom-pole-far", "zoom-pole-hit", "zoom-ring"]);
    expect(IMAGE_VIEWS.filter((v) => v.fov < 14).every((v) => v.fov === 14 / 16)).toBe(true);
  });
});
