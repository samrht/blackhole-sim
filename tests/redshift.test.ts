import { describe, it, expect } from "vitest";
import { gFactor, gFactorKepler } from "../src/physics/redshift";
const close = (a: number, b: number, tol = 1e-6) => Math.abs(a - b) <= tol * (1 + Math.abs(b));

describe("redshift g-factor", () => {
  it("static emitter in Schwarzschild gives g = sqrt(1-2M/r)", () => {
    // Ω=0, ξ=0 -> g = sqrt(-g_tt) = sqrt(1-2/r)
    expect(close(gFactor(10, 0, 0, 0), Math.sqrt(1 - 2 / 10))).toBe(true);
    expect(close(gFactor(4, 0, 0, 0), Math.sqrt(1 - 2 / 4))).toBe(true);
  });
  it("orbiting disk: approaching side (ξ>0) brighter than receding (ξ<0)", () => {
    const r = 10, a = 0.5, b = 3;
    expect(gFactorKepler(r, a, +b)).toBeGreaterThan(gFactorKepler(r, a, -b));
  });
  it("g-factor is positive and finite in the emitting region", () => {
    expect(gFactorKepler(8, 0.9, 2)).toBeGreaterThan(0);
    expect(Number.isFinite(gFactorKepler(8, 0.9, 2))).toBe(true);
  });
});
