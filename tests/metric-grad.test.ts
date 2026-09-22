import { describe, it, expect } from "vitest";
import { metricUpper, metricUpperGrad, horizonOuter, type Metric } from "../src/physics/kerr";
import { rhs } from "../src/physics/geodesic";

const KEYS: (keyof Metric)[] = ["tt", "tphi", "rr", "thth", "phph"];

/** Richardson-extrapolated central difference: O(h^4) truncation. */
function rich(f: (x: number) => number, x: number, h: number): number {
  const d = (hh: number) => (f(x + hh) - f(x - hh)) / (2 * hh);
  return (4 * d(h / 2) - d(h)) / 3;
}
const close = (got: number, want: number) => Math.abs(got - want) <= 1e-8 * (1 + Math.abs(want));

function states() {
  const out: { r: number; th: number; a: number }[] = [];
  for (const a of [0, 0.5, 0.998]) {
    const rh = horizonOuter(a);
    for (const r of [rh * 1.05, 2.5, 6, 40, 900]) {
      if (r <= rh * 1.01) continue;
      for (const th of [1e-3, 0.3, Math.PI / 2, 2.8, Math.PI - 2e-3]) out.push({ r, th, a });
    }
  }
  return out;
}
const hr = (r: number, a: number) => 1e-3 * (r - horizonOuter(a));
// 1e-2, not 1e-3: at r = 1.05 r_h, a = 0.998, theta = 1e-3 the f64 roundoff of a 1e-6 step on
// g^tt ~ -235 (eps * 235 / 1e-6 ~ 5e-8) exceeds the 1e-8 tolerance; the analytic value agrees with
// a 60-digit reference there to 1e-15. Truncation at 1e-2 is O(1e-8) relative, well inside.
const ht = (th: number) => 1e-2 * Math.min(th, Math.PI - th);

describe("metricUpperGrad", () => {
  it("returns the same metric as metricUpper", () => {
    for (const { r, th, a } of states()) {
      const m = metricUpperGrad(r, th, a).g, ref = metricUpper(r, th, a);
      for (const k of KEYS) expect(close(m[k], ref[k])).toBe(true);
    }
  });
  it("matches a Richardson reference for d/dr and d/dtheta to 1e-8 relative", () => {
    let n = 0;
    for (const { r, th, a } of states()) {
      const d = metricUpperGrad(r, th, a);
      for (const k of KEYS) {
        const wantR = rich((x) => metricUpper(x, th, a)[k], r, hr(r, a));
        const wantT = rich((x) => metricUpper(r, x, a)[k], th, ht(th));
        if (!close(d.dr[k], wantR)) throw new Error(`d${k}/dr at r=${r} th=${th} a=${a}: ${d.dr[k]} vs ${wantR}`);
        if (!close(d.dth[k], wantT)) throw new Error(`d${k}/dth at r=${r} th=${th} a=${a}: ${d.dth[k]} vs ${wantT}`);
        n++;
      }
    }
    expect(n).toBeGreaterThan(300);
  });
});

describe("analytic rhs", () => {
  it("momentum forces equal -1/2 dH/dx from a Richardson reference", () => {
    const hq = (r: number, th: number, a: number, p: number[]) => {
      const g = metricUpper(r, th, a);
      return g.tt * p[0] ** 2 + 2 * g.tphi * p[0] * p[3] + g.rr * p[1] ** 2 + g.thth * p[2] ** 2 + g.phph * p[3] ** 2;
    };
    const p = [1, -0.7, 2.3, -3.1];
    for (const { r, th, a } of states()) {
      const d = rhs(new Float64Array([0, r, th, 0, ...p]), a);
      expect(close(d[5], -0.5 * rich((x) => hq(x, th, a, p), r, hr(r, a)))).toBe(true);
      expect(close(d[6], -0.5 * rich((x) => hq(r, x, a, p), th, ht(th)))).toBe(true);
      expect(d[4]).toBe(0);
      expect(d[7]).toBe(0);
    }
  });
});
