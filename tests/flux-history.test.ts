import { describe, it, expect } from "vitest";
import { FLUX, fluxHash, eruptionTime, eruptionDepth, fluxDeficit, fluxFlicker, fluxParams, fluxTarget,
  fluxRatio, fluxMoment, fluxSeries, modulationIndex, seriesSigma } from "../src/physics/flux-history";

describe("flux history v2: eruptions x flicker, per spin (spec 2026-10-03 flux statistics)", () => {
  it("eruptions recur every 1500 M on average, every gap in [1000, 2000] M", () => {
    const gaps: number[] = [];
    for (let k = 0; k < 700; k++) gaps.push(eruptionTime(k + 1) - eruptionTime(k));
    expect(Math.abs(gaps.reduce((x, y) => x + y) / gaps.length - 1500)).toBeLessThan(50);
    for (const g of gaps) { expect(g).toBeGreaterThanOrEqual(1000); expect(g).toBeLessThanOrEqual(2000); }
    for (let k = -5; k < 5; k++) { const u = fluxHash(k, FLUX.saltT); expect(u).toBeGreaterThanOrEqual(-0.5); expect(u).toBeLessThan(0.5); }
  });
  it("drops are exponential with e-folding 500 M, then refill linearly (unchanged)", () => {
    const db = 0.2138;
    for (const k of [3, 17, 401]) {
      const tk = eruptionTime(k), dk = eruptionDepth(k, db), D = -FLUX.tauD * Math.log(1 - dk);
      expect(fluxDeficit(tk + 100, db)).toBeCloseTo(1 - Math.exp(-0.2), 12);
      expect(fluxDeficit(tk + D - 1e-6, db)).toBeCloseTo(dk, 6);
      expect(fluxDeficit(0.5 * (tk + D + eruptionTime(k + 1)), db)).toBeCloseTo(0.5 * dk, 6);
    }
  });
  it("flicker: deterministic, clipped at 3, clipped variance c2, decorrelated after 100 M", () => {
    let m = 0, v = 0, mx = 0, n = 0; const xs: number[] = [];
    for (let t = 2.5; t < 1e6; t += 5) { const x = fluxFlicker(t); xs.push(x); m += x; v += x * x; mx = Math.max(mx, Math.abs(x)); n++; }
    expect(fluxFlicker(1234.5)).toBe(fluxFlicker(1234.5));
    expect(mx).toBeLessThanOrEqual(FLUX.flickerClip);
    expect(Math.abs(v / n - FLUX.c2)).toBeLessThan(0.01);
    expect(Math.abs(m / n)).toBeLessThan(0.02);
    let num = 0; for (let i = 0; i + 20 < xs.length; i++) num += xs[i] * xs[i + 20];
    expect(Math.abs(num / v)).toBeLessThan(0.05); // lag 100 M
  });
  it("Phi's 1000 M modulation index is flat in prograde spin at 0.0846 (Narayan+2022 S3.3; final review)", () => {
    // Their Fig. 9 dashed lines only connect a = 0 to 0.9; the text: Phi variability "largely independent of
    // prograde spin". Target = mean of the five prograde points 0.067, 0.079, 0.098, 0.081, 0.098.
    for (const a of [0, 0.3, 0.45, 0.9, 0.998]) { expect(fluxTarget(a)).toBe(0.0846); expect(fluxParams(a)).toEqual(fluxParams(0)); }
  });
  it("the calibration reproduces the windowed index, window insensitivity and the 2-sigma depth", () => {
    const s = fluxSeries(FLUX.dbar, FLUX.eps);
    expect(Math.abs(modulationIndex(s, 1000) - 0.0846)).toBeLessThan(0.002);
    expect(modulationIndex(s, 500) / modulationIndex(s, 2000)).toBeGreaterThanOrEqual(0.8);
    expect(Math.abs(2 * seriesSigma(s) / FLUX.dbar - 1)).toBeLessThan(0.01);
    let d1 = 0, d2 = 0, d3 = 0, n = 0;
    for (let t = 0.5; t < 1e6; t += 1) { const d = fluxDeficit(t, FLUX.dbar); d1 += d; d2 += d * d; d3 += d * d * d; n++; }
    // the constants print 6 decimals: compare at that precision
    expect(Math.abs(d1 / n - FLUX.d1)).toBeLessThan(1e-6);
    expect(Math.abs(d2 / n - FLUX.d2)).toBeLessThan(1e-6);
    expect(Math.abs(d3 / n - FLUX.d3)).toBeLessThan(1e-6);
  });
  it("s = 0 is a steady jet; s <= sMax never reaches the floor at any spin (Review Focus 3)", () => {
    for (const t of [0, 123.4, 9e5]) expect(fluxRatio(t, 0, 0.9)).toBe(1);
    for (const a of [0, 0.45, 0.9]) {
      const p = fluxParams(a);
      let maxDepth = 0; for (let k = -10; k < 5000; k++) maxDepth = Math.max(maxDepth, eruptionDepth(k, p.dbar));
      expect((1 - FLUX.sMax * maxDepth) * (1 - FLUX.sMax * p.eps * FLUX.flickerClip)).toBeGreaterThan(FLUX.floor);
    }
  });
  it("closed-form <f> = 1, <f^2>, <f^3> match brute force (Review Focus 4)", () => {
    for (const a of [0, 0.45, 0.9, 0.998]) for (const s of [0.5, 1, 1.4]) {
      let m1 = 0, m2 = 0, m3 = 0, n = 0;
      for (let t = 0.5; t < 1e6; t += 1) { const f = fluxRatio(t, s, a); m1 += f; m2 += f * f; m3 += f * f * f; n++; }
      expect(m1 / n).toBeCloseTo(1, 2);
      expect(Math.abs(m2 / n / fluxMoment(2, s, a) - 1)).toBeLessThan(3e-3);
      expect(Math.abs(m3 / n / fluxMoment(3, s, a) - 1)).toBeLessThan(5e-3);
    }
  });
});

describe("Flux variability tooltip (final review: it still advertised the old phi_BH 21 %)", () => {
  it("describes the absolute-flux calibration, not 21 %", async () => {
    const { readFileSync } = await import("node:fs");
    const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
    const tip = html.match(/<div class="ctrl" title="([^"]*)">\s*<div class="row"><label>Flux variability/)![1];
    expect(tip).not.toMatch(/21 %/);
    expect(tip).toMatch(/8\.5 %/);
    expect(tip).toMatch(/Φ/);
  });
});
