import { describe, it, expect } from "vitest";
import { heatColor, logUnit, summarize, curveDistance, parseDriftParams, HEAT_RAMP } from "../src/test/drift-analysis";

describe("drift heat-map scale", () => {
  it("logUnit maps [lo, hi] to [0, 1] in log and clamps; non-finite is NaN", () => {
    expect(logUnit(1e-9, 1e-9, 1e-1)).toBe(0);
    expect(logUnit(1e-5, 1e-9, 1e-1)).toBeCloseTo(0.5, 12);
    expect(logUnit(1, 1e-9, 1e-1)).toBe(1);
    expect(logUnit(0, 1e-9, 1e-1)).toBe(0);
    expect(logUnit(Infinity, 1e-9, 1e-1)).toBeNaN();
    expect(logUnit(NaN, 1e-9, 1e-1)).toBeNaN();
  });
  it("the ramp darkens monotonically and undefined values leave it", () => {
    const lum = ([r, g, b]: number[]) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let prev = Infinity;
    for (let k = 0; k <= 32; k++) { const l = lum(heatColor(k / 32)); expect(l).toBeLessThan(prev); prev = l; }
    expect(HEAT_RAMP.length).toBeGreaterThan(4);
    const nan = heatColor(NaN);
    expect(nan[0]).toBe(nan[1]); expect(nan[1]).toBe(nan[2]); // neutral grey, not a value on the blue ramp
  });
});

describe("drift statistics", () => {
  it("summarize reports quantiles over finite values and counts the rest", () => {
    const s = summarize(Float64Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, Infinity, NaN]));
    expect(s.n).toBe(10); expect(s.undefined).toBe(2);
    expect(s.p50).toBe(6); expect(s.max).toBe(10);
  });
  it("summarize survives an empty set", () => { expect(summarize(new Float64Array(0)).n).toBe(0); });
  it("curveDistance is the distance to the nearest curve point", () => {
    const pts: [number, number][] = [[0, 0], [3, 0], [0, 4]];
    expect(curveDistance(3, 4, pts)).toBe(3);
    expect(curveDistance(1, 0, pts)).toBe(1);
  });
});

describe("?drift parameters", () => {
  it("defaults to a high-spin edge-on frame with the disk", () => {
    expect(parseDriftParams("?drift")).toEqual({ a: 0.998, inclDeg: 85, disk: true, fov: 14, panX: 0, panY: 0, maxSteps: 4800, cpuStride: 8 });
  });
  it("reads overrides", () => {
    const p = parseDriftParams("?drift&a=0.9&i=17&disk=0&fov=1.5&px=2&py=-3&steps=9000&cpu=16");
    expect(p).toEqual({ a: 0.9, inclDeg: 17, disk: false, fov: 1.5, panX: 2, panY: -3, maxSteps: 9000, cpuStride: 16 });
  });
});
