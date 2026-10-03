import { describe, it, expect } from "vitest";
import { advanceSimTime, splitTime, MAX_FRAME_DT_MS, TIME_EPOCH } from "../src/render/sim-clock";

describe("simulation clock (final review, Important 1)", () => {
  it("advances by dt x SPEED x Motion for ordinary frames", () => {
    expect(advanceSimTime(100, 16, 20, 1)).toBeCloseTo(100.32, 12);
    expect(advanceSimTime(100, 16, 20, 0)).toBe(100);
  });
  it("caps one frame's step, so a tab left in the background does not jump hours ahead on return", () => {
    expect(MAX_FRAME_DT_MS).toBe(100);
    expect(advanceSimTime(0, 8 * 3600 * 1000, 20, 1)).toBeCloseTo(2, 12); // 100 ms at 20 M/s
    expect(advanceSimTime(5, -40, 20, 1)).toBe(5);                           // clock went backwards: no step
  });
  it("splits time into an epoch (a multiple of TIME_EPOCH) plus a small remainder that sums back exactly", () => {
    expect(TIME_EPOCH).toBe(2048);
    for (const t of [0, 37, 2047.9, 2048, 3.6e5 + 0.123, 1.8e6 + 17.25, -50.5]) {
      const { epoch, rel } = splitTime(t);
      expect(Math.abs(epoch % TIME_EPOCH)).toBe(0);
      expect(rel).toBeGreaterThanOrEqual(0); expect(rel).toBeLessThan(TIME_EPOCH);
      expect(epoch + rel).toBe(t);
      expect(Math.fround(epoch)).toBe(epoch);          // the epoch survives f32 exactly
    }
    expect(splitTime(37)).toEqual({ epoch: 0, rel: 37 }); // small times: unchanged (golden scenes)
  });
});
