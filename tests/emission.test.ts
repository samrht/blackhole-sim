import { describe, it, expect } from "vitest";
import { patternPhase, hotspotField, emissionField, lognormalFactor, T_BREATHE, type HotSpot } from "../src/physics/emission";
import { omegaKepler } from "../src/physics/orbits";

const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * (1 + Math.abs(b));

describe("emission", () => {
  it("patternPhase = phi_hit at t=0, and advances by -Omega*t*timeScale", () => {
    expect(close(patternPhase(8, 1.3, 0, 1, 0.9), 1.3)).toBe(true);
    const om = omegaKepler(8, 0.9, true);
    expect(close(patternPhase(8, 1.3, 5, 2, 0.9), 1.3 - om * 5 * 2)).toBe(true);
  });

  it("inner annuli sweep faster than outer (differential rotation)", () => {
    const dInner = 0 - patternPhase(6, 0, 1, 1, 0.9);   // phase swept in unit time at r=6
    const dOuter = 0 - patternPhase(20, 0, 1, 1, 0.9);  // at r=20
    expect(dInner).toBeGreaterThan(dOuter);             // |Omega(6)| > |Omega(20)|
  });

  it("hotspotField peaks at the spot center and decays far away, and is periodic in psi", () => {
    const spots: HotSpot[] = [{ r: 10, psi: 1.0, sigma: 1.0, amp: 2.0 }];
    const peak = hotspotField(10, 1.0, spots);
    expect(close(peak, 2.0, 1e-6)).toBe(true);
    expect(hotspotField(10, 1.0 + 3, spots)).toBeLessThan(0.05);            // ~3 sigma away in arc
    expect(close(hotspotField(10, 1.0, spots), hotspotField(10, 1.0 + 2 * Math.PI, spots), 1e-6)).toBe(true);
  });

  it("emissionField reduces to exactly 1 when all features are off (Tier-1 regression gate)", () => {
    expect(emissionField(9, 2.0, 123.4, 0.9, 0, 0, [])).toBe(1);
    expect(emissionField(15, -1.0, -5.0, 0.0, 0, 0, [])).toBe(1);
  });
  it("lognormal factor has mean exactly 1 over a unit Gaussian (quadrature), so the disk's light is conserved", () => {
    for (const s of [0.1, 0.55, 1.7]) {
      let I = 0; const h = 1e-3;
      for (let x = -14; x <= 14 + 1e-12; x += h) I += Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI) * lognormalFactor(x, s) * h;
      expect(Math.abs(I - 1)).toBeLessThan(1e-3);
    }
  });
  it("emissionField = lognormal turbulence x breathing + hot spots at the co-rotating phase", () => {
    const spots: HotSpot[] = [{ r: 10, psi: 1.0, sigma: 1.0, amp: 2.0 }];
    const t = 37, phi = 1.0 + omegaKepler(10, 0.9, true) * t; // the spot's centre at time t
    expect(emissionField(10, phi, t, 0.9, 0, 0, spots)).toBeCloseTo(3.0, 9);
  });
  it("emissionField is non-negative and finite at the slider's top sigma with strong features", () => {
    const spots: HotSpot[] = [{ r: 8, psi: 0, sigma: 1.5, amp: 3 }];
    for (let p = 0; p < 6.28; p += 0.5) {
      const e = emissionField(8, p, 10, 0.9, 1.8, 0.9, spots);
      expect(Number.isFinite(e)).toBe(true); expect(e).toBeGreaterThanOrEqual(0);
    }
  });

  it("T_BREATHE is the documented period constant", () => { expect(T_BREATHE).toBe(2000); });
});
