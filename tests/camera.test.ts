import { describe, it, expect } from "vitest";
import { screenToState } from "../src/physics/camera";
import { rk4, nullRadialMomentum, conserved } from "../src/physics/geodesic";
import { metricUpper } from "../src/physics/kerr";

const A = 0.9, INCL = (72 * Math.PI) / 180, ROBS = 1000;

/** integrate a backward ray inward; return the first equatorial crossing */
function diskHit(s0: Float64Array, a = A) {
  let s: Float64Array = s0.slice();
  const rh = 1 + Math.sqrt(1 - a * a);
  for (let i = 0; i < 300000; i++) {
    const dl = Math.min(0.5, Math.max(0.002, 0.02 * (s[1] - rh)));
    const nx = rk4(s, a, dl);
    if (!isFinite(nx[1]) || nx[1] <= rh * 1.005) return { fate: "hole" as const };
    if (nx[1] > ROBS * 1.2) return { fate: "sky" as const };
    const f0 = s[2] - Math.PI / 2, f1 = nx[2] - Math.PI / 2;
    if (f0 * f1 < 0) {
      const fr = f0 / (f0 - f1);
      return { fate: "disk" as const, r: s[1] + fr * (nx[1] - s[1]) };
    }
    s = nx;
  }
  return { fate: "budget" as const };
}

/** a PHYSICAL photon leaving the equator at rE with conserved (xi, eta), future-directed, outgoing */
function shootOut(rE: number, xi: number, eta: number) {
  const pth0 = -Math.sqrt(eta); // toward the northern hemisphere
  const pr = nullRadialMomentum(rE, Math.PI / 2, A, -1, xi, pth0);
  let s: Float64Array = new Float64Array([0, rE, Math.PI / 2, 0, -1, pr, pth0, xi]);
  for (let i = 0; i < 500000 && s[1] < ROBS; i++) {
    const nx = rk4(s, A, Math.min(0.5, Math.max(0.002, 0.02 * s[1])));
    if (!isFinite(nx[1]) || nx[1] < 2.4) return null;
    s = nx;
  }
  return s[1] >= ROBS ? s : null;
}

describe("camera initial conditions", () => {
  it("produces a null initial 4-momentum", () => {
    for (const [al, be] of [[-6, 1], [8, 4], [0, 0], [12, -7]]) {
      const H = conserved(screenToState(al, be, A, INCL, ROBS), A).H;
      expect(Math.abs(H)).toBeLessThan(1e-12);
    }
  });

  it("traces PAST-directed, retracing the photon that arrives (not one leaving)", () => {
    // dt/dl < 0: going back in coordinate time. dr/dl < 0: inward.
    // The buggy convention gave dt/dl = +1.002 here.
    for (const [al, be] of [[-6, 1], [8, 4], [12, -7]]) {
      const s = screenToState(al, be, A, INCL, ROBS);
      const g = metricUpper(s[1], s[2], A);
      expect(g.tt * s[4] + g.tphi * s[7]).toBeLessThan(0);
      expect(g.rr * s[5]).toBeLessThan(0);
    }
  });

  it("reciprocity: a photon shot from the disk traces back to its emitter", () => {
    // Ground truth independent of the camera code: shoot a real photon out from r=8, find the
    // one that arrives at the camera's latitude, convert to a pixel, then trace that pixel back.
    const rE = 8;
    let checked = 0;
    for (const xi of [2.8, -3.5, 1.5]) {
      let lo = 0.02, hi = 60;
      const arrive = (e: number) => { const s = shootOut(rE, xi, e); return s ? s[2] : NaN; };
      let alo = arrive(lo), ahi = arrive(hi);
      if (!isFinite(alo) || !isFinite(ahi) || (alo - INCL) * (ahi - INCL) > 0) continue;
      let eta = 0;
      for (let k = 0; k < 60; k++) {
        eta = 0.5 * (lo + hi);
        const am = arrive(eta);
        if (!isFinite(am)) break;
        if ((alo - INCL) * (am - INCL) <= 0) { hi = eta; ahi = am; } else { lo = eta; alo = am; }
      }
      const out = shootOut(rE, xi, eta);
      if (!out) continue;
      const hit = diskHit(screenToState(-xi / Math.sin(INCL), out[6], A, INCL, ROBS));
      expect(hit.fate).toBe("disk");
      expect((hit as { r: number }).r).toBeCloseTo(rE, 2);
      checked++;
    }
    expect(checked).toBeGreaterThanOrEqual(3); // guard: the loop must not silently skip everything
  }, 120_000);
});
