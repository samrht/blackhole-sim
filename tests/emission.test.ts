import { describe, it, expect } from "vitest";
import { patternPhase, hotspotField, lognormalFactor, diskShadeFactors, T_BREATHE, type HotSpot } from "../src/physics/emission";
import { omegaKepler } from "../src/physics/orbits";
import { turbulenceAt } from "../src/physics/emission";

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

  it("lognormal factor has mean exactly 1 over a unit Gaussian (quadrature), so the disk's light is conserved", () => {
    for (const s of [0.1, 0.55, 1.7]) {
      let I = 0; const h = 1e-3;
      for (let x = -14; x <= 14 + 1e-12; x += h) I += Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI) * lognormalFactor(x, s) * h;
      expect(Math.abs(I - 1)).toBeLessThan(1e-3);
    }
  });
  it("turbulence modulates the local flux, so it shows as temperature T x F^(1/4) (final review, Important 2)", () => {
    const f = diskShadeFactors(9, 1.1, 50, 0.9, 0.553, 0, []);
    const F = lognormalFactor(turbulenceAt(9, 1.1, 50, 0.9), 0.553);
    expect(f.tempScale).toBeCloseTo(Math.pow(F, 0.25), 12);
    expect(f.grey).toBe(1);                                      // no illustrative features
    expect(diskShadeFactors(9, 1.1, 50, 0.9, 0, 0, [])).toEqual({ tempScale: 1, grey: 1 });
  });
  it("bolometric light is conserved: the mean of tempScale^4 over a unit Gaussian is exactly 1", () => {
    for (const s of [0.553, 1.76]) {
      let I = 0; const h = 1e-3;
      for (let x = -14; x <= 14 + 1e-12; x += h) I += Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI) * Math.pow(Math.pow(lognormalFactor(x, s), 0.25), 4) * h;
      expect(Math.abs(I - 1)).toBeLessThan(1e-3);
    }
  });
  it("all features off shades exactly as Novikov-Thorne (Tier-1 regression gate)", () => {
    expect(diskShadeFactors(15, -1.0, -5.0, 0.0, 0, 0, [])).toEqual({ tempScale: 1, grey: 1 });
  });
  it("finite at the slider's top sigma with strong illustrative features", () => {
    const spots: HotSpot[] = [{ r: 8, psi: 0, sigma: 1.5, amp: 3 }];
    for (let p = 0; p < 6.28; p += 0.5) {
      const f = diskShadeFactors(8, p, 10, 0.9, 1.8, 0.9, spots);
      expect(Number.isFinite(f.tempScale) && Number.isFinite(f.grey)).toBe(true); expect(f.grey).toBeGreaterThanOrEqual(0);
    }
  });
  it("illustrative breathing and hot spots stay a grey brightness factor, never negative", () => {
    const spots: HotSpot[] = [{ r: 10, psi: 1.0, sigma: 1.0, amp: 2.0 }];
    const t = 37, phi = 1.0 + omegaKepler(10, 0.9, true) * t;
    expect(diskShadeFactors(10, phi, t, 0.9, 0, 0, spots).grey).toBeCloseTo(3.0, 9);
    expect(diskShadeFactors(10, phi + 3, 500, 0.9, 0, 1.5, []).grey).toBeGreaterThanOrEqual(0);
  });
  it("T_BREATHE is the documented period constant", () => { expect(T_BREATHE).toBe(2000); });
});
