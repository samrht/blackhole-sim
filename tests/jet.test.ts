import { describe, it, expect } from "vitest";
import {
  JET, funnelEdge, wallProfile, lengthFalloff, knots, dopplerBoost, jetEmission,
} from "../src/physics/jet";

describe("jet geometry", () => {
  it("funnel widens with height (parabolic)", () => {
    expect(funnelEdge(0)).toBeCloseTo(JET.rho0, 12);
    expect(funnelEdge(4)).toBeCloseTo(JET.rho0 + JET.slope * 2, 12); // sqrt(4)=2
    expect(funnelEdge(16)).toBeGreaterThan(funnelEdge(4));
  });

  it("wall profile peaks at q = qPeak (limb-brightened, hollow)", () => {
    const z = 9; const edge = funnelEdge(z);
    const atPeak = wallProfile(JET.qPeak * edge, z);
    const atAxis = wallProfile(0.0, z);
    const outside = wallProfile(1.3 * edge, z);
    expect(atPeak).toBeCloseTo(1.0, 6);   // gaussian peak == 1
    expect(atAxis).toBeLessThan(atPeak);  // dimmer on the axis (hollow tube)
    expect(outside).toBe(0);              // nothing beyond the wall
  });
});

describe("jet beaming", () => {
  it("Doppler boost is >1 approaching, collapses receding, monotonic in mu", () => {
    const g = 5;
    expect(dopplerBoost(0.9, g)).toBeGreaterThan(1);       // toward observer -> boosted
    expect(dopplerBoost(-0.9, g)).toBeLessThan(0.05);      // away -> counter-jet vanishes
    expect(dopplerBoost(0.9, g)).toBeGreaterThan(dopplerBoost(0.3, g));
    expect(dopplerBoost(0.3, g)).toBeGreaterThan(dopplerBoost(-0.3, g));
  });
});

describe("jet living emission field", () => {
  it("knots form a traveling wave (advancing t shifts the pattern)", () => {
    const a = knots(10, 0.0, 1, 0.7);
    const b = knots(10, 0.5, 1, 0.7);
    expect(a).not.toBeCloseTo(b, 6); // time changes the local knot brightness
  });

  it("emission is exactly 0 when jetStrength = 0 (features-off invariant)", () => {
    expect(jetEmission(6, 0.15, 3.2, 1, 0, 60, 0.7)).toBe(0);
  });

  it("emission is 0 outside the axial band and inside the funnel band it is positive", () => {
    const thAxis = 0.12;                 // near the pole -> inside a funnel
    const rIn = 8;
    expect(jetEmission(rIn, thAxis, 0, 1, 1, 60, 0.7)).toBeGreaterThan(0);
    expect(jetEmission(1.5, thAxis, 0, 1, 1, 60, 0.7)).toBe(0); // below zBase launch
    expect(jetEmission(400, thAxis, 0, 1, 1, 60, 0.7)).toBe(0); // beyond jetLength
    expect(jetEmission(8, Math.PI / 2, 0, 1, 1, 60, 0.7)).toBe(0); // equatorial: outside funnel
  });
});
