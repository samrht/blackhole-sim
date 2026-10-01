import { describe, it, expect } from "vitest";
import { computeReadouts } from "../src/physics/readouts";
import { gravRadius, gravTime, peakTemperature } from "../src/physics/units";
import { lumNormFor } from "../src/physics/lookups";

describe("physical readouts", () => {
  it("radii in metres, ISCO period, playback scale", () => {
    const r = computeReadouts({ massSun: 10, a: 0, lambda: 0.1, timeScale: 2 }, 20);
    expect(r.horizonM).toBeCloseTo(2 * gravRadius(10), 6);
    expect(r.iscoM).toBeCloseTo(6 * gravRadius(10), 4);
    expect(r.photonM).toBeCloseTo(3 * gravRadius(10), 4);
    expect(r.iscoPeriodS / gravTime(10)).toBeCloseTo(2 * Math.PI * Math.pow(6, 1.5), 4);
    expect(r.realSecondsPerScreenSecond).toBeCloseTo(40 * gravTime(10), 12);
    expect(r.lumNorm).toBeCloseTo(lumNormFor(r.tPeakK), 12);
  });
  it("changing spin at fixed mass and accretion changes T_peak (Review Focus 3)", () => {
    const lo = computeReadouts({ massSun: 1e8, a: 0, lambda: 1e-3, timeScale: 1 }, 20);
    const hi = computeReadouts({ massSun: 1e8, a: 0.998, lambda: 1e-3, timeScale: 1 }, 20);
    expect(lo.tPeakK).toBeCloseTo(peakTemperature(1e8, 0, 1e-3), 6);
    expect(hi.tPeakK).toBeGreaterThan(lo.tPeakK);
    expect(hi.iscoM).toBeLessThan(lo.iscoM);
  });
});
