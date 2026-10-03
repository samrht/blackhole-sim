import { describe, it, expect } from "vitest";
import { patternPhase, lognormalFactor, diskShadeFactors, T_BREATHE } from "../src/physics/emission";
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

  it("lognormal factor has mean exactly 1 over a unit Gaussian (quadrature), so the disk's light is conserved", () => {
    for (const s of [0.1, 0.55, 1.7]) {
      let I = 0; const h = 1e-3;
      for (let x = -14; x <= 14 + 1e-12; x += h) I += Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI) * lognormalFactor(x, s) * h;
      expect(Math.abs(I - 1)).toBeLessThan(1e-3);
    }
  });
  it("turbulence modulates the local flux, so it shows as temperature T x F^(1/4) (final review, Important 2)", () => {
    const f = diskShadeFactors(9, 1.1, 50, 0.9, 0.553, 0);
    const F = lognormalFactor(turbulenceAt(9, 1.1, 50, 0.9), 0.553);
    expect(f.tempScale).toBeCloseTo(Math.pow(F, 0.25), 12);
    expect(f.grey).toBe(1);                                      // no illustrative features
    expect(diskShadeFactors(9, 1.1, 50, 0.9, 0, 0)).toEqual({ tempScale: 1, grey: 1 });
  });
  it("bolometric light is conserved: the mean of tempScale^4 over a unit Gaussian is exactly 1", () => {
    for (const s of [0.553, 1.76]) {
      let I = 0; const h = 1e-3;
      for (let x = -14; x <= 14 + 1e-12; x += h) I += Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI) * Math.pow(Math.pow(lognormalFactor(x, s), 0.25), 4) * h;
      expect(Math.abs(I - 1)).toBeLessThan(1e-3);
    }
  });
  it("all features off shades exactly as Novikov-Thorne (Tier-1 regression gate)", () => {
    expect(diskShadeFactors(15, -1.0, -5.0, 0.0, 0, 0)).toEqual({ tempScale: 1, grey: 1 });
  });
  it("finite at the slider's top sigma with strong breathing", () => {
    for (let p = 0; p < 6.28; p += 0.5) {
      const f = diskShadeFactors(8, p, 10, 0.9, 1.8, 0.9);
      expect(Number.isFinite(f.tempScale) && Number.isFinite(f.grey)).toBe(true); expect(f.grey).toBeGreaterThanOrEqual(0);
    }
  });
  it("the illustrative hot spots are gone (eruption flares replace them, spec 2026-10-04): grey is the breathing only", () => {
    expect(diskShadeFactors.length).toBe(6);
    expect(diskShadeFactors(10, 1.0, 500, 0.9, 0, 0.5).grey).toBeCloseTo(1 + 0.5 * Math.sin(2 * Math.PI * 500 / T_BREATHE), 12);
  });
  it("T_BREATHE is the documented period constant", () => { expect(T_BREATHE).toBe(2000); });
});
