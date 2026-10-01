import { describe, it, expect } from "vitest";
import {
  gravRadius, gravTime, eddingtonLuminosity, mdotFromLambda, lambdaFromMdot, peakTemperature,
  lambdaForPeakTemperature, iscoPeriod, formatLength, formatDuration, M_SUN, YEAR,
} from "../src/physics/units";
import { efficiency } from "../src/physics/orbits";

describe("physical units", () => {
  it("gravitational radius and time of one solar mass", () => {
    expect(gravRadius(1)).toBeCloseTo(1476.6, 0);       // m
    expect(gravTime(1) * 1e6).toBeCloseTo(4.9255, 3);   // µs
  });
  it("Eddington luminosity and thin-disk efficiency", () => {
    expect(eddingtonLuminosity(1)).toBeCloseTo(1.2572e31, -27);
    expect(efficiency(0, true)).toBeCloseTo(0.0572, 4);
    expect(efficiency(0.998, true)).toBeGreaterThan(0.31);
    expect(efficiency(0.998, true)).toBeLessThan(0.33);
  });
  it("lambda <-> Mdot round trip", () => {
    const l = lambdaFromMdot(6.5e9, 0.9, 7.7e-4);
    expect(mdotFromLambda(6.5e9, 0.9, l) * YEAR / M_SUN).toBeCloseTo(7.7e-4, 10);
  });
  it("peak temperature: 10 M_sun, a = 0, Eddington -> ~7e6 K; scales as lambda^1/4 and M^-1/4", () => {
    const t = peakTemperature(10, 0, 1);
    expect(t).toBeGreaterThan(6.3e6); expect(t).toBeLessThan(7.7e6);
    expect(peakTemperature(10, 0, 1e-4) / t).toBeCloseTo(0.1, 6);
    expect(peakTemperature(1e5, 0, 1) / t).toBeCloseTo(0.1, 6);
  });
  it("lambdaForPeakTemperature inverts peakTemperature", () => {
    const l = lambdaForPeakTemperature(1e8, 0.6, 5800);
    expect(peakTemperature(1e8, 0.6, l)).toBeCloseTo(5800, 6);
  });
  it("extremes stay finite and positive (Review Focus 2)", () => {
    for (const [m, l] of [[1, 1e-10], [1e10, 1e-10], [1, 1], [1e10, 1]]) {
      const t = peakTemperature(m, 0.998, l);
      expect(Number.isFinite(t)).toBe(true); expect(t).toBeGreaterThan(0);
    }
    expect(peakTemperature(1e10, 0, 1e-10)).toBeLessThan(200); // ~124 K: visibly dark (Task 2 caps lumNorm)
  });
  it("ISCO orbital period: a = 0 is 2 pi 6^1.5 t_g", () => {
    expect(iscoPeriod(1, 0) / gravTime(1)).toBeCloseTo(2 * Math.PI * Math.pow(6, 1.5), 6);
  });
  it("formats lengths and durations across 20 orders of magnitude (Review Focus 5)", () => {
    expect(formatLength(512)).toBe("512 m");
    expect(formatLength(31_300)).toBe("31.3 km");
    expect(formatLength(9.6e12)).toBe("64.2 AU");
    expect(formatDuration(1.56e-3)).toBe("1.56 ms");
    expect(formatDuration(42)).toBe("42 s");
    expect(formatDuration(3 * 3600)).toBe("3 h");
    expect(formatDuration(8.9e5)).toBe("10.3 days");
    expect(formatDuration(3.2e9)).toBe("101 yr");
    expect(formatDuration(3e-7)).toBe("0.3 µs");
  });
});
