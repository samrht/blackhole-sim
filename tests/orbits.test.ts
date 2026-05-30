import { describe, it, expect } from "vitest";
import { omegaKepler, circE, circL, iscoRadius, photonOrbit, marginallyBound, efficiency } from "../src/physics/orbits";
const close = (a: number, b: number, tol = 1e-4) => Math.abs(a - b) <= tol * (1 + Math.abs(b));

describe("orbits", () => {
  it("ISCO: 6M (a=0), ->M (a=1 pro), 9M (a=1 retro)", () => {
    expect(close(iscoRadius(0), 6)).toBe(true);
    expect(close(iscoRadius(1, true), 1, 2e-3)).toBe(true);
    expect(close(iscoRadius(1, false), 9)).toBe(true);
  });
  it("photon orbit: 3M (a=0), M (a=1 pro), 4M (a=1 retro)", () => {
    expect(close(photonOrbit(0), 3)).toBe(true);
    expect(close(photonOrbit(1, true), 1, 2e-3)).toBe(true);
    expect(close(photonOrbit(1, false), 4)).toBe(true);
  });
  it("marginally bound: 4M (a=0)", () => { expect(close(marginallyBound(0), 4)).toBe(true); });
  it("efficiency: ~0.057 (a=0), ~0.42 (a=1 pro)", () => {
    expect(close(efficiency(0), 0.0572, 2e-3)).toBe(true);
    expect(efficiency(0.998, true)).toBeGreaterThan(0.3);
  });
  it("Keplerian Omega -> sqrt(1/r^3) far out", () => {
    expect(close(omegaKepler(1000, 0), Math.sqrt(1 / 1e9), 1e-2)).toBe(true);
  });
  it("circular E,L finite at r=6M (a=0): E=sqrt(8/9), L=2sqrt(3)", () => {
    expect(close(circE(6, 0), Math.sqrt(8 / 9))).toBe(true);
    expect(close(circL(6, 0), 2 * Math.sqrt(3))).toBe(true);
  });
});
