import { describe, it, expect } from "vitest";
import { DARK_I, intensityScored, frozenFailures } from "../src/test/accuracy-judge";

describe("?accuracy intensity judging (spec 2026-10-08 MAD R-beta; final review)", () => {
  it("a pixel is dark (unscored) only when BOTH the reference and the renderer are below DARK_I of the peak", () => {
    expect(intensityScored(1e-23, 0, 1)).toBe(false);          // Gargantua's corners: invisible in both
    expect(intensityScored(1e-23, 0.3, 1)).toBe(true);         // the renderer puts visible light where the reference has none
    expect(intensityScored(0.3, 0, 1)).toBe(true);             // the renderer loses visible light
    expect(intensityScored(DARK_I * 2, DARK_I / 2, 1)).toBe(true);
    expect(intensityScored(5, 5, 1e9)).toBe(true);             // relative to the scene's peak
  });
  it("frozen bounds: a scene fails when more pixels exceed their floor, or the worst exceeds its bound", () => {
    const b = { over: 15, worst: 3.8 };
    expect(frozenFailures(5, 1.28, b)).toEqual([]);
    expect(frozenFailures(16, 1.28, b)).toEqual(["16 px above floor > bound 15"]);
    expect(frozenFailures(5, 4.1, b)).toEqual(["worst 4.10x floor > bound 3.8x"]);
  });
});
