import { describe, it, expect } from "vitest";
import { criticalXiEta, photonShellRange, classify, shadowBoundary, A_EPS } from "../src/physics/shadow";

describe("Kerr critical curve", () => {
  it("photon shell endpoints match the equatorial photon orbits at a=0.9", () => {
    const [lo, hi] = photonShellRange(0.9);
    expect(lo).toBeCloseTo(1.5579, 3);
    expect(hi).toBeCloseTo(3.9103, 3);
  });

  it("eta vanishes at both shell endpoints", () => {
    const [lo, hi] = photonShellRange(0.9);
    expect(criticalXiEta(lo, 0.9)[1]).toBeCloseTo(0, 6);
    expect(criticalXiEta(hi, 0.9)[1]).toBeCloseTo(0, 6);
  });

  it("xi is strictly decreasing across the shell (precondition for bisection)", () => {
    const [lo, hi] = photonShellRange(0.9);
    let prev = Infinity;
    for (let i = 0; i <= 200; i++) {
      const r = lo + ((hi - lo) * i) / 200;
      const xi = criticalXiEta(r, 0.9)[0];
      expect(xi).toBeLessThan(prev);
      prev = xi;
    }
  });

  it("reduces to the Schwarzschild circle of radius sqrt(27) as a -> 0", () => {
    const [lo, hi] = photonShellRange(0);
    expect(lo).toBeCloseTo(3, 6);
    expect(hi).toBeCloseTo(3, 6);
    // at a=0 the shadow is a circle: eta + xi^2 = 27 on the critical curve
    const [xi, eta] = criticalXiEta(3, 0);
    expect(eta + xi * xi).toBeCloseTo(27, 4);
  });

  it("is continuous across the A_EPS branch cutover", () => {
    const below = criticalXiEta(3, A_EPS * 0.5);
    const above = criticalXiEta(3, A_EPS * 1.5);
    expect(below[1] + below[0] * below[0]).toBeCloseTo(above[1] + above[0] * above[0], 3);
  });

  it("classifies deep-interior rays as captured and distant rays as escaped", () => {
    // xi=0, eta=0 is a radial ray straight into the hole
    expect(classify(0, 0, 0.9)).toBe("captured");
    // a huge Carter constant means a large impact parameter -> escapes
    expect(classify(0, 400, 0.9)).toBe("escaped");
    // xi far outside the shell's xi range escapes
    expect(classify(50, 0, 0.9)).toBe("escaped");
  });

  it("brackets the Schwarzschild capture threshold at b = sqrt(27)", () => {
    // at a->0, pole-on, eta = b^2; capture iff b < sqrt(27)
    expect(classify(0, 27 * 0.98, 1e-6)).toBe("captured");
    expect(classify(0, 27 * 1.02, 1e-6)).toBe("escaped");
  });

  it("the a=0 boundary is a symmetric circle of radius sqrt(27)", () => {
    const pts = shadowBoundary(0, Math.PI / 2, 400);
    const al = pts.map((p) => p[0]);
    expect(Math.min(...al)).toBeCloseTo(-Math.sqrt(27), 6);
    expect(Math.max(...al)).toBeCloseTo(Math.sqrt(27), 6);
    for (const [x, y] of pts) expect(Math.hypot(x, y)).toBeCloseTo(Math.sqrt(27), 6);
  });

  it("the a=0.9 boundary is flattened on the prograde side (the Kerr signature)", () => {
    const al = shadowBoundary(0.9, Math.PI / 2, 400).map((p) => p[0]);
    const mn = Math.min(...al), mx = Math.max(...al);
    expect(mn).toBeCloseTo(-2.829, 2);
    expect(mx).toBeCloseTo(6.794, 2);
    // strongly asymmetric about alpha = 0, unlike Schwarzschild where this is exactly 0
    expect(Math.abs(mn + mx)).toBeGreaterThan(3.5);
  });
});
