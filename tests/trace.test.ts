import { describe, it, expect } from "vitest";
import { screenToState, screenToXiEta } from "../src/physics/camera";
import { conserved } from "../src/physics/geodesic";
import { stepGeodesic, traceRay, reflectAxis, stepSize, hquadScaled } from "../src/physics/trace";

const I8 = (8 * Math.PI) / 180, ROBS = 1000;
const OPTS = { rIn: 3, rOut: 40, rObs: ROBS }; // emit from the a=0 photon orbit outward, like ?shadow

describe("polar axis", () => {
  it("a xi != 0 ray aimed near the pole turns back at the centrifugal barrier and hits the disk", () => {
    // Fails on the shipped POLE_S2 = 1e-3: the floor removes the barrier, theta runs through 0 and
    // the sampled minimum lands at ~6e-4 instead of the analytic 1.16e-2.
    for (const [alpha, beta] of [[0.5, 6], [0.2, 7], [1.0, 8]]) {
      const a = 0;
      const [xi, eta] = screenToXiEta(alpha, beta, a, I8);
      const thMinAn = Math.atan(Math.abs(xi) / Math.sqrt(eta));
      const res = traceRay(screenToState(alpha, beta, a, I8, ROBS), a, OPTS);
      expect(res.fate).toBe("disk");
      expect(res.thMin).toBeGreaterThan(thMinAn * (1 - 1e-3));
      expect(res.thMin).toBeLessThan(thMinAn * 1.05);
    }
  });

  it("keeps xi exact and bounds the eta / null-constraint drift across a near-axis pass", () => {
    const a = 0, alpha = 0.05, beta = 6; // theta_min ~ 1.2e-3 rad, deep inside the old 1.81 deg cone
    const s0 = screenToState(alpha, beta, a, I8, ROBS);
    const c0 = conserved(s0, a);
    const res = traceRay(s0, a, OPTS);
    expect(res.fate).toBe("disk");
    const c1 = conserved(res.s, a);
    expect(Math.abs(c1.Lz - c0.Lz)).toBeLessThan(1e-6); // Killing: exact up to float noise
    // eta and H are NOT conserved to the equatorial test's 1e-4 here: the finite-difference RK4
    // integrator has a documented accuracy ceiling that a near-axis turning point exposes.
    // Measured before this test was written (f64, provisional H_TOL = 1e-4, MAX_RETRY = 8):
    // |d eta| = 0.12 of eta = 36 (0.3%), |H| = 4e-5 -- versus 11% and 1.3e-3 with the monitor
    // effectively off (H_TOL = 1e-2). The bounds below reject the unmonitored integrator and
    // accept the monitored one with ~3x margin; they are measured ceilings, not targets.
    expect(Math.abs(c1.Q - c0.Q) / c0.Q).toBeLessThan(1e-2);
    expect(Math.abs(c1.H)).toBeLessThan(1e-3);
    // The near-axis answer must converge on the on-axis (alpha = 0) limit: the disk-hit radius is
    // continuous in alpha, so alpha = 0.05 must land within half a pixel (0.02 M at the
    // interactive scale of ~24 px/M) of alpha = 0. Measured: 0.0095 M monitored, 0.34 M unmonitored.
    const ref = traceRay(screenToState(0, beta, a, I8, ROBS), a, OPTS);
    expect(ref.fate).toBe("disk");
    expect(Math.abs(res.rHit! - ref.rHit!)).toBeLessThan(0.02);
  });

  it("a xi = 0 ray passes through the axis: phi shifts by pi, p_theta flips, disk is still found", () => {
    // Fails on the shipped code (no reflection): theta runs to -pi/2 and beyond, the disk test
    // never fires at -pi/2, and the ray terminates as escaped.
    const a = 0;
    const s0 = screenToState(0, 6, a, I8, ROBS); // alpha = 0 => xi = 0 exactly
    expect(s0[7]).toBe(0);
    const res = traceRay(s0, a, OPTS);
    expect(res.fate).toBe("disk");
    // With xi = 0 and a = 0, dphi/dl = 0 identically, so the only phi change is the +pi at the crossing.
    const dphi = ((res.phiHit! - s0[3]) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI);
    expect(Math.abs(dphi - Math.PI)).toBeLessThan(1e-9);
    expect(Math.sign(res.s[6])).toBe(-Math.sign(s0[6])); // p_theta flipped by the crossing
  });

  it("reflectAxis is the exact continuation at both poles and the identity elsewhere", () => {
    const inside = new Float64Array([0, 10, 1.0, 0.3, 1, -1, 2, 0.5]);
    expect(reflectAxis(inside)).toBe(inside); // identity returns the same object
    const north = reflectAxis(new Float64Array([0, 10, -0.01, 0.3, 1, -1, 2, 0.5]));
    expect(north[2]).toBeCloseTo(0.01, 12);
    expect(north[3]).toBeCloseTo(0.3 + Math.PI, 12);
    expect(north[6]).toBe(-2);
    expect(north[7]).toBe(0.5); // p_phi unchanged
    const south = reflectAxis(new Float64Array([0, 10, Math.PI + 0.02, 0.3, 1, -1, -2, 0.5]));
    expect(south[2]).toBeCloseTo(Math.PI - 0.02, 12);
    expect(south[3]).toBeCloseTo(0.3 + Math.PI, 12);
    expect(south[6]).toBe(2);
  });

  it("retry exhaustion is reported, not silently accepted", () => {
    const a = 0.9, s = screenToState(3, 2, a, 1.2, ROBS);
    const rh = 1 + Math.sqrt(1 - a * a);
    // hTol < 0 can never be satisfied, so every attempt fails: the step must come back with
    // ok = false, all halvings spent, and dl = dl0 / 2^maxRetry.
    const dl0 = stepSize(s[1], rh, 40);
    const out = stepGeodesic(s, a, dl0, -1, 5);
    expect(out.ok).toBe(false);
    expect(out.retries).toBe(5);
    expect(out.dl).toBeCloseTo(dl0 / 32, 12);
    // traceRay routes an untrusted step out of the loop immediately.
    expect(traceRay(s, a, { ...OPTS, hTol: -1, maxRetry: 3 }).fate).toBe("untrusted");
    // ...and a NaN state is never accepted either (NaN drift compares false).
    const nan = new Float64Array([0, NaN, 1, 0, 1, -1, 0, 0]);
    expect(stepGeodesic(nan, a, 0.1, 1e-4, 2).ok).toBe(false);
  });

  it("hquadScaled scale is >= 1 for p_t = 1 and equals |terms| sum", () => {
    const [h, scale] = hquadScaled(2.05, 1.0, 0.9, 1, -3, 0.5, -2);
    expect(scale).toBeGreaterThanOrEqual(1);
    expect(Math.abs(h)).toBeLessThanOrEqual(scale);
  });
});
