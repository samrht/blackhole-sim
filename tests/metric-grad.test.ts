import { describe, it, expect } from "vitest";
import { metricUpper, metricUpperGrad, horizonOuter, sigma, delta, POLE_S2, type Metric } from "../src/physics/kerr";
import { rhs, rhsFD } from "../src/physics/geodesic";

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
// 3e-3, not 1e-3: at r = 1.05 r_h, a = 0.998, theta = 1e-3 the f64 roundoff of a 1e-6 step on
// g^tt ~ -235 (eps * 235 / 1e-6 ~ 5e-8) exceeds the 1e-8 tolerance; the analytic value agrees with
// a 60-digit reference there to 1e-15. Measured worst reference error over these states: 3.0e-9
// at 3e-3 (tt, r = 1.05 r_h, theta = pi - 2e-3, a = 0.998) vs 7.5e-9 at 1e-2 (phph, theta = 1e-3),
// i.e. about 3x inside the 1e-8 tolerance, not "well inside".
const ht = (th: number) => 3e-3 * Math.min(th, Math.PI - th);

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
  it("the POLE_S2 floor: below it, the floored denominator's theta-derivative is zero", () => {
    const pole = [3e-7, Math.PI - 4e-7];
    for (const th of pole) expect(Math.sin(th) ** 2).toBeLessThan(POLE_S2); // the floored branch runs
    for (const r of [2.5, 6, 40]) {
      for (const th of pole) {
        // a = 0: every component is theta-independent below the floor (tphi = 0; tt, rr, thth depend
        // on theta only through a^2; phph = Delta / (Sigma Delta POLE_S2) is constant), so exactly 0.
        const d0 = metricUpperGrad(r, th, 0).dth;
        for (const k of KEYS) expect(Math.abs(d0[k]), `a=0 d${k}/dth at r=${r} th=${th}`).toBe(0);
        // a = 0.9: phph = N / P with N = Delta - a^2 s^2, P = Sigma Delta POLE_S2 (floor constant),
        // so d(phph)/dtheta = (dN - phph dP) / P with dN = -2 a^2 s c and dP = dSigma/dtheta Delta POLE_S2.
        const a = 0.9, s = Math.sin(th), c = Math.cos(th);
        const D = delta(r, a), Sig = sigma(r, th, a), N = D - a * a * s * s, P = Sig * D * POLE_S2;
        const want = (-2 * a * a * s * c - (N / P) * (-2 * a * a * s * c * D * POLE_S2)) / P;
        const got = metricUpperGrad(r, th, a).dth;
        expect(Math.abs(got.phph - want) / Math.abs(want), `phph at r=${r} th=${th}`).toBeLessThanOrEqual(1e-12);
        // The other components are smooth (even) through the axis, so a centred difference may cross
        // it; an absolute 1e-3 step keeps f64 roundoff (~eps |g| / h) far below the tolerance.
        for (const k of ["tt", "tphi", "rr", "thth"] as const) {
          const ref = rich((x) => metricUpper(r, x, a)[k], th, 1e-3);
          if (!close(got[k], ref)) throw new Error(`d${k}/dth at r=${r} th=${th} a=${a}: ${got[k]} vs ${ref}`);
        }
      }
    }
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

describe("rhsFD (finite-difference reference)", () => {
  it("agrees with the analytic rhs to 1e-5 relative on moderate states", () => {
    const cases: [number, number, number][] = [[3, 0.3, 0], [8, 1.2, 0], [40, 2.8, 0], [3, 2.8, 0.9], [8, 2.0, 0.9], [40, 0.3, 0.9]];
    const p = [-1, 0.4, 1.7, -2.2];
    for (const [r, th, a] of cases) {
      const s = new Float64Array([0, r, th, 0, ...p]);
      const fd = rhsFD(s, a), an = rhs(s, a);
      for (let i = 0; i < 8; i++) {
        if (!(Math.abs(fd[i] - an[i]) <= 1e-5 * (1 + Math.abs(an[i]))))
          throw new Error(`component ${i} at r=${r} th=${th} a=${a}: FD ${fd[i]} vs analytic ${an[i]}`);
      }
      expect(Math.abs(an[6])).toBeGreaterThan(1e-3); // the theta force is exercised, not trivially zero
    }
  });
});
