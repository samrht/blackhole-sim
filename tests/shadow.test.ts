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

  it("the GENERAL formula (a above A_EPS, not the hardcoded short-circuit) converges to eta + xi^2 = 27", () => {
    // a=1e-3 is above A_EPS=1e-4, so this exercises the actual Bardeen formula, not the
    // Schwarzschild short-circuit. Derived via scratch computation: at a=1e-3,
    // eta + xi^2 = 27.000004003774002, i.e. |diff from 27| ~= 4.0e-6, scaling ~O(a^2)
    // (diff was ~4.0e-4 at a=1e-2 and ~3.3e-7 at a=3e-4). toBeCloseTo(27, 5) requires
    // |diff| < 5e-6, which the measured 4.0e-6 satisfies with ~20% margin.
    const [xi, eta] = criticalXiEta(3, 1e-3);
    expect(eta + xi * xi).toBeCloseTo(27, 5);
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

  it("the GENERAL formula (a=1e-3, above A_EPS) still brackets the sqrt(27) capture threshold", () => {
    // a=1e-3 is above A_EPS=1e-4, so classify() runs the bisection over the actual photon
    // shell instead of the hardcoded eta + xi^2 < 27 shortcut. Verified by scratch computation
    // that the bracket direction is unchanged at a=1e-2, 3e-3, 1e-3, 3e-4 -- no sign flip near
    // the threshold, so no special tolerance is needed here (this is a discrete classification).
    expect(classify(0, 27 * 0.98, 1e-3)).toBe("captured");
    expect(classify(0, 27 * 1.02, 1e-3)).toBe("escaped");
  });

  it("the a=0 boundary is a symmetric circle of radius sqrt(27)", () => {
    const pts = shadowBoundary(0, Math.PI / 2, 400);
    const al = pts.map((p) => p[0]);
    expect(Math.min(...al)).toBeCloseTo(-Math.sqrt(27), 6);
    expect(Math.max(...al)).toBeCloseTo(Math.sqrt(27), 6);
    for (const [x, y] of pts) expect(Math.hypot(x, y)).toBeCloseTo(Math.sqrt(27), 6);
  });

  it("the GENERAL formula (a=1e-3, above A_EPS) still traces a near-circular boundary near radius sqrt(27)", () => {
    // a=1e-3 is above A_EPS=1e-4, so shadowBoundary() walks the actual photon shell instead of
    // emitting the hardcoded circle. Derived via scratch computation: at a=1e-3 the sampled
    // radii range over [5.194152135035858, 5.198142138672847] against sqrt(27) = 5.196152422706632,
    // i.e. max |radius - sqrt(27)| ~= 2.0e-3, scaling ~O(a) (it was ~2.0e-2 at a=1e-2 and
    // ~6.0e-4 at a=3e-4). toBeCloseTo(sqrt(27), 2) requires |diff| < 5e-3, which the measured
    // 2.0e-3 satisfies with better than 2x margin.
    const pts = shadowBoundary(1e-3, Math.PI / 2, 400);
    const target = Math.sqrt(27);
    for (const [x, y] of pts) expect(Math.hypot(x, y)).toBeCloseTo(target, 2);
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
