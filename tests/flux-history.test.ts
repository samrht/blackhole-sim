import { describe, it, expect } from "vitest";
import { FLUX, fluxHash, eruptionTime, eruptionDepth, fluxDeficit, fluxRatio, fluxMoment, measureFlux } from "../src/physics/flux-history";

describe("flux history (spec 2026-10-03 jet flux knots 2.2)", () => {
  it("eruptions recur every 1500 M on average, every gap inside [1000, 2000] M", () => {
    const gaps: number[] = [];
    for (let k = 0; k < 700; k++) gaps.push(eruptionTime(k + 1) - eruptionTime(k));
    const mean = gaps.reduce((x, y) => x + y) / gaps.length;
    expect(Math.abs(mean - 1500)).toBeLessThan(50);
    for (const g of gaps) { expect(g).toBeGreaterThanOrEqual(1000); expect(g).toBeLessThanOrEqual(2000); }
  });
  it("hash is uniform in [-0.5, 0.5) and deterministic", () => {
    for (let k = -50; k < 50; k++) {
      const u = fluxHash(k, FLUX.saltT);
      expect(u).toBeGreaterThanOrEqual(-0.5); expect(u).toBeLessThan(0.5); expect(fluxHash(k, FLUX.saltT)).toBe(u);
    }
  });
  it("drops are exponential with e-folding 500 M (Ripperda+2022), then refill linearly to zero", () => {
    for (const k of [3, 17, 401]) {
      const tk = eruptionTime(k), dk = eruptionDepth(k), D = -FLUX.tauD * Math.log(1 - dk);
      expect(fluxDeficit(tk + 1e-6)).toBeLessThan(1e-6);                       // starts at 0
      expect(fluxDeficit(tk + 100)).toBeCloseTo(1 - Math.exp(-100 / 500), 12);  // exponential
      expect(fluxDeficit(tk + D - 1e-6)).toBeCloseTo(dk, 6);                     // reaches the depth
      const tn = eruptionTime(k + 1), mid = 0.5 * (tk + D + tn);
      expect(fluxDeficit(mid)).toBeCloseTo(0.5 * dk, 6);                         // linear refill
      expect(fluxDeficit(tn - 1e-6)).toBeLessThan(1e-6);                         // continuous at the next eruption
    }
  });
  it("swing sigma/mu = 0.209 at s = 1 (Narayan+2022 a = 0.9) and the hardcoded moments match the generator", () => {
    const m = measureFlux(FLUX.dbar, 1e6, 1);
    expect(m.sigmaOverMu).toBeCloseTo(FLUX.target, 3);
    expect(Math.abs(m.d1 / FLUX.d1 - 1)).toBeLessThan(1e-4);
    expect(Math.abs(m.d2 / FLUX.d2 - 1)).toBeLessThan(1e-4);
    expect(Math.abs(m.d3 / FLUX.d3 - 1)).toBeLessThan(1e-4);
  });
  it("the floor is never reached for s <= sMax (Review Focus 3)", () => {
    let maxDepth = 0;
    for (let k = -10; k < 5000; k++) maxDepth = Math.max(maxDepth, eruptionDepth(k));
    expect(maxDepth * FLUX.sMax).toBeLessThan(1 - FLUX.floor);
  });
  it("s = 0 is a steady jet: f = 1 at all times (Review Focus 3)", () => {
    for (const t of [0, 123.4, 9e5, -50]) expect(fluxRatio(t, 0)).toBe(1);
  });
  it("f has mean one and the closed-form moments match a brute-force time average", () => {
    for (const s of [0.5, 1, 1.4]) {
      let m1 = 0, m2 = 0, m3 = 0, n = 0;
      // the constants' own 1e6 M window (a shorter one samples different eruptions: ~0.5 % off)
      for (let t = 0.5; t < 1e6; t += 1) { const f = fluxRatio(t, s); m1 += f; m2 += f * f; m3 += f * f * f; n++; }
      expect(m1 / n).toBeCloseTo(1, 2);
      expect(Math.abs(m2 / n / fluxMoment(2, s) - 1)).toBeLessThan(3e-3);
      expect(Math.abs(m3 / n / fluxMoment(3, s) - 1)).toBeLessThan(5e-3);
    }
  });
  it("negative and huge times work (the clock never runs backwards, but launch times can precede 0)", () => {
    for (const t of [-1400, -1, 0, 4.2e7]) { const d = fluxDeficit(t); expect(d).toBeGreaterThanOrEqual(0); expect(d).toBeLessThan(1); }
  });
});
