import { describe, it, expect } from "vitest";
import { diskFlickerRms, FULL_GRID } from "../src/physics/turbulence-calibration";
import { FLICKER_TABLE, FLICKER_DEFAULT, sigmaForFlicker } from "../src/physics/emission";

describe("flicker calibration (spec 2026-10-03 §2.2)", () => {
  it("the table starts at (0, 0), is strictly increasing and reaches the slider's 10 %", () => {
    expect(FLICKER_TABLE[0]).toEqual([0, 0]);
    for (let i = 1; i < FLICKER_TABLE.length; i++) {
      expect(FLICKER_TABLE[i][0]).toBeGreaterThan(FLICKER_TABLE[i - 1][0]);
      expect(FLICKER_TABLE[i][1]).toBeGreaterThan(FLICKER_TABLE[i - 1][1]);
    }
    expect(FLICKER_TABLE[FLICKER_TABLE.length - 1][1]).toBeGreaterThan(0.1);
  });
  it("sigmaForFlicker inverts the table: 0 -> 0, linear in between, finite at 10 %", () => {
    expect(sigmaForFlicker(0)).toBe(0);
    const [s1, r1] = FLICKER_TABLE[1], [s2, r2] = FLICKER_TABLE[2];
    expect(sigmaForFlicker((r1 + r2) / 2)).toBeCloseTo((s1 + s2) / 2, 12);
    const top = sigmaForFlicker(0.1);
    expect(Number.isFinite(top)).toBe(true);
    expect(top).toBeLessThan(3);
    expect(sigmaForFlicker(1)).toBe(FLICKER_TABLE[FLICKER_TABLE.length - 1][0]); // clamps
  });
  it("small-sigma slope kappa matches the prototype (0.0347 +- 10 %)", () => {
    const [s, r] = FLICKER_TABLE[1];
    expect(r / s).toBeGreaterThan(0.031); expect(r / s).toBeLessThan(0.038);
  });
  it("snapshots start at the grid's t0, so a re-measurement can use its own time window", () => {
    const g = { ...FULL_GRID, ne: 4, np: 8, ns: 3 };
    const [a] = diskFlickerRms([0.5], { ...g, t0: 1000 }), [b] = diskFlickerRms([0.5], { ...g, t0: 7e5 });
    expect(a).not.toBe(b);
  });
  it("the default slider gives 2.0 % +- 0.2 % intrinsic rms, re-measured on an independent coarse grid", () => {
    // Own time window: FULL_GRID's 1500 snapshots end at t = 1000 + 1500 x 397 = 596,500 M; these start at 7e5.
    const coarse = { ...FULL_GRID, ne: 80, np: 128, ns: 300, t0: 7e5 };
    expect(coarse.t0).toBeGreaterThan(FULL_GRID.t0 + FULL_GRID.ns * FULL_GRID.dt);
    const [rms] = diskFlickerRms([sigmaForFlicker(FLICKER_DEFAULT)], coarse);
    expect(rms).toBeGreaterThan(0.018); expect(rms).toBeLessThan(0.022);
  }, 120000);
});
