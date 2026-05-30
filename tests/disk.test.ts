import { describe, it, expect } from "vitest";
import { pageThorneFluxShape, temperatureShape } from "../src/physics/disk";
import { iscoRadius } from "../src/physics/orbits";

describe("Novikov-Thorne / Page-Thorne disk", () => {
  it("flux vanishes at the ISCO (zero-torque inner boundary)", () => {
    const a = 0, ri = iscoRadius(a);
    expect(temperatureShape(ri * 1.0001, a)).toBeLessThan(0.05);
  });
  it("temperature peaks just OUTSIDE the ISCO, not at it", () => {
    const a = 0, ri = iscoRadius(a);
    const tAtEdge = temperatureShape(ri * 1.001, a);
    const tPeakish = temperatureShape(ri * 1.5, a);
    expect(tPeakish).toBeGreaterThan(tAtEdge);
  });
  it("far-field flux ~ r^-3 => T ~ r^-3/4", () => {
    const a = 0;
    const ratio = temperatureShape(400, a) / temperatureShape(100, a);
    expect(ratio).toBeCloseTo(Math.pow(4, -0.75), 1); // ≈0.354
  });
  it("flux is positive in the emitting region", () => {
    expect(pageThorneFluxShape(20, 0.9)).toBeGreaterThan(0);
  });
});
