import { describe, it, expect } from "vitest";
import {
  sigma, delta, bigA, horizonOuter, ergosphere, omegaHorizon,
  metricLower, metricUpper, omegaZAMO,
} from "../src/physics/kerr";

const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * (1 + Math.abs(b));

describe("kerr metric", () => {
  it("Schwarzschild horizon is 2M", () => { expect(close(horizonOuter(0), 2)).toBe(true); });
  it("extremal horizon is M", () => { expect(close(horizonOuter(1), 1)).toBe(true); });
  it("ergosphere at equator is 2M for any spin", () => {
    expect(close(ergosphere(Math.PI / 2, 0.9), 2)).toBe(true);
  });
  it("lower and upper metric are inverse in the (t,phi) block and diagonal", () => {
    const r = 8, th = 1.1, a = 0.7;
    const g = metricLower(r, th, a), gi = metricUpper(r, th, a);
    // (M·M^-1)_tt = g_tt g^tt + g_tφ g^φt = 1
    expect(close(g.tt * gi.tt + g.tphi * gi.tphi, 1)).toBe(true);
    // (M·M^-1)_φφ = g_φt g^tφ + g_φφ g^φφ = 1
    expect(close(g.tphi * gi.tphi + g.phph * gi.phph, 1)).toBe(true);
    // off-diagonal (M·M^-1)_tφ = g_tt g^tφ + g_tφ g^φφ = 0
    expect(close(g.tt * gi.tphi + g.tphi * gi.phph, 0, 1e-9)).toBe(true);
    expect(close(g.rr * gi.rr, 1)).toBe(true);
    expect(close(g.thth * gi.thth, 1)).toBe(true);
  });
  it("ZAMO omega -> Omega_H at the horizon", () => {
    const a = 0.6, rp = horizonOuter(a);
    expect(close(omegaZAMO(rp, Math.PI / 2, a), omegaHorizon(a), 1e-6)).toBe(true);
  });
  it("Sigma, Delta, A basic identities", () => {
    expect(close(sigma(5, Math.PI / 2, 0.5), 25)).toBe(true);       // r^2 at equator
    expect(close(delta(2, 0), 0)).toBe(true);                        // r_+=2M for a=0
    expect(bigA(6, 1.0, 0.5)).toBeGreaterThan(0);
  });
});
