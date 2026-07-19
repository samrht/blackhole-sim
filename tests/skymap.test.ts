import { describe, it, expect } from "vitest";
import { dirToEquirectUV, tiltDir, SKY_TEXW } from "../src/render/skymap";

describe("equirectangular sky map", () => {
  it("maps cardinal directions to the expected UV", () => {
    expect(SKY_TEXW).toBe(4096);
    const [u0, v0] = dirToEquirectUV([1, 0, 0]);
    expect(u0).toBeCloseTo(0.5); expect(v0).toBeCloseTo(0.5);      // +X → centre
    expect(dirToEquirectUV([0, 1, 0])[1]).toBeCloseTo(0.0);        // +Y → north pole (v=0)
    expect(dirToEquirectUV([0, -1, 0])[1]).toBeCloseTo(1.0);       // -Y → south pole (v=1)
    expect(dirToEquirectUV([0, 0, 1])[0]).toBeCloseTo(0.75);       // +Z
    expect(dirToEquirectUV([0, 0, -1])[0]).toBeCloseTo(0.25);      // -Z
  });

  it("wraps continuously across the ±Z seam behind -X", () => {
    const uHi = dirToEquirectUV([-1, 0, 1e-4])[0];   // just above the seam → u ≈ 1
    const uLo = dirToEquirectUV([-1, 0, -1e-4])[0];  // just below the seam → u ≈ 0
    expect(uHi).toBeGreaterThan(0.99);
    expect(uLo).toBeLessThan(0.01);
  });

  it("tiltDir is an orthonormal rotation (preserves length)", () => {
    const len = (d: number[]) => Math.hypot(d[0], d[1], d[2]);
    expect(len(tiltDir([1, 0, 0]))).toBeCloseTo(1);
    expect(len(tiltDir([0.3, -0.5, 0.81]))).toBeCloseTo(len([0.3, -0.5, 0.81]));
  });
});
