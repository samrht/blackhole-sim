import { describe, it, expect } from "vitest";
import {
  JET, funnelEdge, wallProfile, lengthFalloff, knots, jetShape, inJetEnvelope,
  launchDelay, fieldLineOmega, comovingAzimuth, gammaProfile,
} from "../src/physics/jet";
import { vnoise } from "../src/physics/emission";

describe("jet geometry", () => {
  it("funnel widens with height (parabolic)", () => {
    expect(funnelEdge(0)).toBeCloseTo(JET.rho0, 12);
    expect(funnelEdge(4)).toBeCloseTo(JET.rho0 + JET.slope * 2, 12); // sqrt(4)=2
    expect(funnelEdge(16)).toBeGreaterThan(funnelEdge(4));
  });

  it("wall profile peaks at q = qPeak (limb-brightened, hollow)", () => {
    const z = 9; const edge = funnelEdge(z);
    const atPeak = wallProfile(JET.qPeak * edge, z);
    const atAxis = wallProfile(0.0, z);
    const outside = wallProfile(1.3 * edge, z);
    expect(atPeak).toBeCloseTo(1.0, 6);   // gaussian peak == 1
    expect(atAxis).toBeLessThan(atPeak);  // dimmer on the axis (hollow tube)
    expect(outside).toBe(0);              // nothing beyond the wall
  });
});

describe("jet living emission field", () => {
  it("knots ride the jet flow at beta(Gamma) < c (a pattern faster than light washed out under light-travel delay)", () => {
    const gamma = 5, beta = Math.sqrt(1 - 1 / (gamma * gamma)); // 0.9798
    for (const [z, t, dt] of [[10, 0, 3], [25, 12, 7.5], [-14, 4, 2]]) {
      const zs = Math.sign(z) * (Math.abs(z) + beta * dt); // outward along its own lobe
      expect(knots(zs, t + dt, gamma, 0.7)).toBeCloseTo(knots(z, t, gamma, 0.7), 5);
    }
  });
  it("knots form a traveling wave (advancing t shifts the pattern)", () => {
    const a = knots(10, 0.0, 5, 0.7);
    const b = knots(10, 0.5, 5, 0.7);
    expect(a).not.toBeCloseTo(b, 6); // time changes the local knot brightness
  });

  it("emission is 0 outside the axial band and inside the funnel band it is positive", () => {
    const thAxis = 0.12;                 // near the pole -> inside a funnel
    const rIn = 8;
    expect(jetShape(rIn, thAxis, 0, 60, 0.7, 2)).toBeGreaterThan(0);
    expect(jetShape(1.5, thAxis, 0, 60, 0.7, 2)).toBe(0); // below zBase launch
    expect(jetShape(400, thAxis, 0, 60, 0.7, 2)).toBe(0); // beyond jetLength
    expect(jetShape(8, Math.PI / 2, 0, 60, 0.7, 2)).toBe(0); // equatorial: outside funnel
  });
  it("jetShape is a mean-one modulation of the wall profile (energy budget assumes it, synchrotron.ts)", () => {
    // average over many times at a fixed point in the wall: the knots stream past and average out.
    // g280 = 8 so the flow moves here (Gamma(20) = 1.73; at g280 = 2 the plasma is still at Gamma = 1
    // below z ~ 84 and the knot pattern is static there - a consequence of M87's measured profile).
    const r = 20, th = Math.atan2(funnelEdge(20) * JET.qPeak, 20), z = r * Math.cos(th), rho = r * Math.sin(th);
    let m = 0; const N = 4000;
    for (let k = 0; k < N; k++) m += jetShape(r, th, k * 0.37, 60, 0.7, 8);
    const turb = 1 + JET.turbAmpJet * (vnoise(Math.log(1 + rho), JET.kz * z) - 0.5) * 2;
    expect(m / N / (wallProfile(rho, z) * lengthFalloff(z, 60) * turb)).toBeCloseTo(1, 1);
  });
});

describe("jet envelope (geodesic-cache bookmark region)", () => {
  it("contains every point where the jet can emit", () => {
    // jetShape > 0 anywhere => inJetEnvelope true, over a grid and several times
    for (let r = 1.2; r < 80; r *= 1.07) {
      for (let th = 0.001; th < Math.PI; th += 0.013) {
        for (const t of [0, 3.3, 77]) {
          if (jetShape(r, th, t, 60, 0.7, 2) > 0) expect(inJetEnvelope(r, th, 60)).toBe(true);
        }
      }
    }
  });
  it("excludes below the launch height, beyond the length, and outside the wall", () => {
    expect(inJetEnvelope(1.5, 0.01, 60)).toBe(false);             // |z| < zBase
    expect(inJetEnvelope(70, 0.01, 60)).toBe(false);              // |z| > jetLength
    expect(inJetEnvelope(20, Math.PI / 2 - 0.2, 60)).toBe(false); // far outside the funnel
    expect(inJetEnvelope(20, 0.03, 60)).toBe(true);               // on the axis, inside
    expect(inJetEnvelope(20, Math.PI - 0.03, 60)).toBe(true);     // counter-jet
  });
  it("does not depend on jet strength (it takes none)", () => {
    expect(inJetEnvelope.length).toBe(3);
  });
});

describe("plasma travel time and co-moving azimuth (spec 2.3, 2.5 + corrections)", () => {
  // 20 000-point midpoint reference in v = z^(1-p): tau = Int sqrt(1 + A^2 v^(2p/(1-p))) dv / (A (1-p)).
  const tauRef = (z: number, g: number) => {
    const p = 0.58, A = Math.sqrt(g * g - 1) / Math.pow(280, p), e = (2 * p) / (1 - p);
    const v0 = Math.pow(2, 1 - p), v1 = Math.pow(Math.abs(z), 1 - p), N = 20000; let s = 0;
    for (let i = 0; i < N; i++) { const v = v0 + ((i + 0.5) / N) * (v1 - v0); s += Math.sqrt(1 + A * A * Math.pow(v, e)); }
    return (s * (v1 - v0)) / N / (A * (1 - p));
  };
  it("6-point Gauss quadrature matches the reference to 1e-4 (G280 1.5-8, z to 1000 M)", () => {
    for (const g of [1.5, 2, 3, 5, 8]) for (const z of [2.5, 5, 10, 30, 60, 200, 1000]) {
      expect(Math.abs(launchDelay(z, g) / tauRef(z, g) - 1)).toBeLessThan(1e-4);
      expect(launchDelay(-z, g)).toBe(launchDelay(z, g)); // counter-jet uses |z| (Review Focus 4)
    }
    expect(launchDelay(1.5, 2)).toBe(0);                  // below the base
    expect(launchDelay(60, 2)).toBeCloseTo(166, 0);       // spec 2.3
  });
  it("a parcel integrated along the flow keeps its launch time and co-moving azimuth", () => {
    const g = 2, a = 0.9, OmF = fieldLineOmega(a);
    let z = 2, ph = 0.3, t = 1000; const dt = 0.01;
    const beta = (zz: number) => { const G = gammaProfile(zz, g); return Math.sqrt(1 - 1 / (G * G)); };
    for (let k = 0; k < 20000; k++) {
      const b1 = beta(z), b2 = beta(z + 0.5 * dt * b1), b3 = beta(z + 0.5 * dt * b2), b4 = beta(z + dt * b3);
      const dz = (dt / 6) * (b1 + 2 * b2 + 2 * b3 + b4);
      ph += OmF * (dt - dz); z += dz; t += dt;   // dphi/dt = Omega_F (1 - beta)
    }
    expect(z).toBeGreaterThan(30);
    expect(t - launchDelay(z, g)).toBeCloseTo(1000, 2);
    expect(comovingAzimuth(ph, z, a, g)).toBeCloseTo(0.3, 3);
  });
  it("field-line rotation is half the horizon's: a / (4 r+), zero at spin 0 (Review Focus 5)", () => {
    expect(fieldLineOmega(0)).toBe(0);
    expect(fieldLineOmega(0.9)).toBeCloseTo(0.9 / (4 * (1 + Math.sqrt(1 - 0.81))), 12);
    expect(comovingAzimuth(1.0, 30, 0, 2)).toBe(1.0);
  });
});
