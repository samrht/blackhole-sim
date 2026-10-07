import { describe, it, expect } from "vitest";
import { HOTSPOT, HOTSPOT_REACH, HOTSPOT_LPEAK, HOTSPOT_TPEAK, hotspotLight, hotspotRadius, hotspotPeriod, hotspotAt,
  hotspotBoost, hotspotShift, hotspotPeakTime, hotspotEndTime, hotspotAliveWindow } from "../src/physics/hotspot";
import { FLUX, eruptionTime, eruptionDepth, fluxHash } from "../src/physics/flux-history";
import { omegaKepler } from "../src/physics/orbits";
import { gFactorKepler } from "../src/physics/redshift";

describe("hotspot schedule (spec 2026-10-04 mm hotspots 2.2)", () => {
  it("one hotspot per eruption: radius in [8, 12), every lifetime 3P shorter than the shortest eruption gap", () => {
    let rmin = Infinity, rmax = 0;
    for (let k = -5; k < 2000; k++) {
      const rc = hotspotRadius(k); rmin = Math.min(rmin, rc); rmax = Math.max(rmax, rc);
      expect(rc).toBeGreaterThanOrEqual(8); expect(rc).toBeLessThan(12);
      for (const a of [0, 0.998]) expect(hotspotEndTime(k, a)).toBeLessThan(eruptionTime(k + 1));
    }
    expect(rmin).toBeLessThan(8.05); expect(rmax).toBeGreaterThan(11.95); // the range is used, not a corner of it
    expect(HOTSPOT_REACH).toBeCloseTo(8 + 4 + 4 * 2.548, 10);
  });
  it("hotspotAt: the latest eruption, alive while 0 <= tau < 3P; amplitude A0 s (delta_k / dbar) L; none at s = 0", () => {
    const a = 0.94;
    for (let t = -3000; t < 3e5; t += 7.3) {
      const h = hotspotAt(t, 1, a), k = h.k;
      expect(eruptionTime(k)).toBeLessThanOrEqual(t); expect(eruptionTime(k + 1)).toBeGreaterThan(t);
      const tau = t - eruptionTime(k), P = hotspotPeriod(hotspotRadius(k), a);
      expect(h.alive).toBe(tau < 3 * P);
      if (!h.alive) continue;
      expect(h.rc).toBe(hotspotRadius(k));
      expect(h.amp).toBeCloseTo(HOTSPOT.A0 * (eruptionDepth(k, FLUX.dbar) / FLUX.dbar) * hotspotLight(tau, P), 9);
      expect(hotspotAt(t, 1.4, a).amp).toBeCloseTo(1.4 * h.amp, 9);
      expect(hotspotAt(t, 0, a).alive).toBe(false);
    }
  });
  it("Keplerian, prograde, rigid: Omega = omegaKepler(r_c), phi_c advances 2 pi per period from 2 pi (u_phi + 0.5)", () => {
    for (const a of [0, 0.5, 0.94]) for (const k of [0, 1, 7]) {
      const t0 = eruptionTime(k), h0 = hotspotAt(t0, 1, a), P = hotspotPeriod(h0.rc, a);
      expect(h0.Om).toBeCloseTo(omegaKepler(h0.rc, a, true), 12);
      expect(h0.phiC).toBeCloseTo(2 * Math.PI * (fluxHash(k, HOTSPOT.saltPhi) + 0.5), 9);
      expect(hotspotAt(t0 + P, 1, a).phiC - h0.phiC).toBeCloseTo(2 * Math.PI, 9);
    }
  });
});

describe("hotspot light curve, boost and Doppler factor", () => {
  it("L: 0 before birth, rises over 0.1 P, e-folds in P, smooth cut to 0 by 3 P", () => {
    const P = 200;
    expect(hotspotLight(-1, P)).toBe(0); expect(hotspotLight(0, P)).toBe(0);
    expect(hotspotLight(0.05 * P, P)).toBeCloseTo(0.5 * Math.exp(-0.05), 12);
    expect(hotspotLight(0.1 * P, P)).toBeCloseTo(Math.exp(-0.1), 12);
    expect(hotspotLight(P, P)).toBeCloseTo(Math.exp(-1), 12);
    expect(hotspotLight(2.5 * P, P)).toBeCloseTo(Math.exp(-2.5), 12);
    expect(hotspotLight(2.75 * P, P)).toBeCloseTo(0.5 * Math.exp(-2.75), 12);
    expect(hotspotLight(3 * P, P)).toBe(0);
    expect(HOTSPOT_LPEAK).toBeGreaterThan(0.904); expect(HOTSPOT_LPEAK).toBeLessThan(0.907);
    expect(HOTSPOT_TPEAK).toBeGreaterThan(0.09); expect(HOTSPOT_TPEAK).toBeLessThan(0.1);
    const a = 0.94, tp = hotspotPeakTime(3, a), h = hotspotAt(tp, 1, a);
    expect(h.amp / (HOTSPOT.A0 * (eruptionDepth(3, FLUX.dbar) / FLUX.dbar))).toBeCloseTo(HOTSPOT_LPEAK, 9);
  });
  it("G: 1 at the centre, 1/2 at 3 M (FWHM 6 M), 0 beyond 4 sigma", () => {
    const rc = 10, ph = 1.3;
    expect(hotspotBoost(rc, Math.PI / 2, ph, rc, ph)).toBeCloseTo(1, 12);
    expect(hotspotBoost(rc + 3, Math.PI / 2, ph, rc, ph)).toBeCloseTo(0.5, 3);
    const z = (d: number) => Math.atan2(rc, d); // a point d above the centre: r = hypot(rc, d), th = atan2(rc, d)
    expect(hotspotBoost(Math.hypot(rc, 3), z(3), ph, rc, ph)).toBeCloseTo(0.5, 3);
    expect(hotspotBoost(rc + 4 * 2.548 + 0.01, Math.PI / 2, ph, rc, ph)).toBe(0);
    expect(hotspotBoost(rc + 4 * 2.548 - 0.01, Math.PI / 2, ph, rc, ph)).toBeGreaterThan(0);
  });
  it("D_h: static far-observer limit 1, and 1 / g of a Keplerian equatorial emitter (redshift.ts)", () => {
    expect(hotspotShift(1e6, Math.PI / 2, 1, 0, 0.5, 1e-9)!).toBeCloseTo(1, 5);
    for (const [r, a, xi] of [[10, 0.94, 3], [8, 0, -4], [12, 0.5, 0]] as const) {
      const D = hotspotShift(r, Math.PI / 2, 1, -xi, a, omegaKepler(r, a, true))!;
      expect(D * gFactorKepler(r, a, xi, true)).toBeCloseTo(1, 9);
    }
    expect(hotspotShift(1.5, Math.PI / 2, 1, 2, 0.94, 1 / (10 ** 1.5 + 0.94))).toBeNull(); // inside the ergosphere: not timelike
  });
});

describe("live-tracing window", () => {
  it("true iff some hotspot is alive within +-pad of t (brute force); never at s = 0", () => {
    const a = 0.94, pad = 300;
    for (let i = 0; i < 300; i++) {
      const t = 137.77 * i - 2000;
      let brute = false; for (let x = t - pad; x <= t + pad && !brute; x += 0.25) brute = hotspotAt(x, 1, a).alive;
      expect(hotspotAliveWindow(t, 1, a, pad)).toBe(brute);
      expect(hotspotAliveWindow(t, 0, a, pad)).toBe(false);
    }
  });
});
