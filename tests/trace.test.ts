import { describe, it, expect } from "vitest";
import { screenToState, screenToXiEta } from "../src/physics/camera";
import { conserved } from "../src/physics/geodesic";
import { stepGeodesic, traceRay, reflectAxis, stepSize, hquadScaled } from "../src/physics/trace";
import { iscoRadius } from "../src/physics/orbits";

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
    // eta and H across a near-axis turning point. History: under the H_TOL monitor alone the
    // measured drift was |d eta|/eta 0.3 % and |H| 4e-5 (bounds 1e-2 / 1e-3), against 11 % and 1.3e-3
    // with the monitor effectively off. Since the angular caps also bound the near field
    // (2026-10-01) the passage is resolved: measured |dQ|/Q 2.8e-6, |H| 6.0e-8, max |H| over the
    // trajectory (spec 4.1) 1.4e-7, |rHit - rHit(alpha = 0)| 2.1e-4 M (was 0.0095 M; 0.34 M
    // unmonitored). The bounds below lock that in with ~10x margin; they are measured ceilings.
    expect(Math.abs(c1.Q - c0.Q) / c0.Q).toBeLessThan(3e-5);
    expect(Math.abs(c1.H)).toBeLessThan(1e-6);
    expect(res.maxAbsH).toBeLessThan(2e-6);
    // The near-axis answer must converge on the on-axis (alpha = 0) limit: the disk-hit radius is
    // continuous in alpha, so alpha = 0.05 must land well within half a pixel (0.02 M at the
    // interactive scale of ~24 px/M) of alpha = 0.
    const ref = traceRay(screenToState(0, beta, a, I8, ROBS), a, OPTS);
    expect(ref.fate).toBe("disk");
    expect(Math.abs(res.rHit! - ref.rHit!)).toBeLessThan(2e-3);
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

  it("rays within 1e-3 of alpha = 0 hit the disk at the on-axis radius (no seam at the axis column)", () => {
    // The interactive view (a = 0.9, i = 72 deg), beta = 8: the alpha = 0 ray crosses the axis
    // exactly (reflectAxis) and hits the disk; rays a hair off-axis pass within ~1e-4 rad of the
    // pole where the 1/sin^3 barrier exhausts the retry budget. Routing those to the (xi, eta)
    // classifier painted starfield where the disk belongs -- a 1-px dark seam on the alpha = 0
    // pixel column. By continuity they must land where the on-axis ray does.
    const a = 0.9, incl = (72 * Math.PI) / 180;
    const opts = { rIn: iscoRadius(a, true), rOut: 40, rObs: ROBS };
    const ref = traceRay(screenToState(0, 8, a, incl, ROBS), a, opts);
    expect(ref.fate).toBe("disk");
    for (const alpha of [0, 1e-4, 1e-3, 1e-2]) {
      const res = traceRay(screenToState(alpha, 8, a, incl, ROBS), a, opts);
      expect(res.fate, `alpha = ${alpha}`).toBe("disk");
      expect(Math.abs(res.rHit! - ref.rHit!), `alpha = ${alpha}`).toBeLessThan(0.02); // half a pixel at ~24 px/M
    }
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
    const dl0 = stepSize(s, rh, 40);
    const out = stepGeodesic(s, a, dl0, -1, 5);
    expect(out.ok).toBe(false);
    expect(out.retries).toBe(5);
    expect(out.dl).toBeCloseTo(dl0 / 32, 12);
    // traceRay counts the exhaustion but proceeds with the smallest-step attempt: the ray still
    // terminates normally.
    const t = traceRay(s, a, { ...OPTS, hTol: -1, maxRetry: 3 });
    expect(t.exhausted).toBeGreaterThan(0);
    // alpha = 3, beta = 2 at a = 0.9 is inside the critical curve: the ray is captured (observed
    // with 2552 exhausted steps of 2717), i.e. exhaustion neither strands it in the budget nor
    // ends it early.
    expect(t.fate).toBe("captured");
    // ...and a NaN state is never accepted either (NaN drift compares false).
    const nan = new Float64Array([0, NaN, 1, 0, 1, -1, 0, 0]);
    expect(stepGeodesic(nan, a, 0.1, 1e-4, 2).ok).toBe(false);
  });

  it("hquadScaled scale is >= 1 for p_t = 1 and equals |terms| sum", () => {
    const [h, scale] = hquadScaled(2.05, 1.0, 0.9, 1, -3, 0.5, -2);
    expect(scale).toBeGreaterThanOrEqual(1);
    expect(Math.abs(h)).toBeLessThanOrEqual(scale);
    // Cross-check against the independent conserved() Hamiltonian at a non-equatorial a = 0.9
    // state from the camera, where the 2 g^{t phi} p_t p_phi cross term is non-zero: a wrong
    // factor on it in both twins would pass every parity gate, so pin it here.
    const s = screenToState(4, 3, 0.9, 1.2, ROBS);
    const [h2, scale2] = hquadScaled(s[1], s[2], 0.9, s[4], s[5], s[6], s[7]);
    expect(h2).toBeCloseTo(2 * conserved(s, 0.9).H, 12);
    expect(scale2).toBeGreaterThanOrEqual(1);
  });

  it("a step that moves theta by more than 0.5 rad is not accepted as a disk crossing", () => {
    // reflectAxis assumes a single crossing per step (theta -> -theta); a step that carried theta
    // through several radians is a diverged state, and interpolating a disk hit from it would be
    // garbage. The render loop and traceRay guard the disk test with |delta theta| < 0.5. Since the
    // angular caps also bound the near field (2026-10-01) a legitimate stride moves theta by at most
    // ~F_AXIS of its distance to the axis, so the guard is a backstop: it is reachable only where
    // the cap is clamped at its floor (0.002) and garbage momenta still move theta a lot. Exercise
    // exactly that: r = 20, theta = pi/2 - 0.3, and a p_theta so large that the capped stride is the
    // floor and one floor-sized step moves theta by ~1 rad across the plane (dtheta/dl = p_theta /
    // Sigma = p_theta / 400). The monitor is switched off (hTol = 1e30) so the step is taken as is,
    // and maxSteps = 1 isolates that single step. The same state with p_theta scaled to a ~0.4 rad
    // move crosses the plane inside the guard and IS a disk hit near r = 20.
    const a = 0, rh = 2, r = 20, th = Math.PI / 2 - 0.3, FLOOR = 0.002;
    const mk = (dth: number) => new Float64Array([0, r, th, 0, 1, -1, dth * r * r / FLOOR, 0]);
    expect(stepSize(mk(1.0), rh, 40)).toBe(FLOOR);
    expect(stepSize(mk(0.4), rh, 40)).toBe(FLOOR);
    // ...whereas a moderate p_theta is capped well below a plane-jumping move: ~0.1 * theta_d.
    const mod = new Float64Array([0, r, th, 0, 1, -1, 5, 0]);
    expect(stepSize(mod, rh, 40) * 5 / (r * r)).toBeLessThan(0.11 * th);
    const big = traceRay(mk(1.0), a, { ...OPTS, hTol: 1e30, maxSteps: 1 });
    expect(big.fate).toBe("budget"); // crossed the plane, |delta theta| ~ 1 > 0.5: not a hit
    expect(big.s[2]).toBeGreaterThan(Math.PI / 2);
    const small = traceRay(mk(0.4), a, { ...OPTS, hTol: 1e30, maxSteps: 1 });
    expect(small.fate).toBe("disk");
    expect(Math.abs(small.rHit! - r)).toBeLessThan(2); // the (unphysical) p_theta also kicks p_r
  });
});
