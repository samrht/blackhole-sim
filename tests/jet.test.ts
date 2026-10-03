import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  JET, funnelEdge, wallProfile, lengthFalloff, jetShape, inJetEnvelope, FILAMENT, JET_ENV_Q, vnoise3, filaments,
  launchDelay, fieldLineOmega, comovingAzimuth, gammaProfile, GL6,
} from "../src/physics/jet";
import { FLUX, fluxRatio } from "../src/physics/flux-history";
import { FLARE } from "../src/physics/eruption-spots";

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

describe("flux-driven jet shape (spec 2.4, 2.5)", () => {
  const thWall = (r: number) => Math.atan2(funnelEdge(r) * JET.qPeak, r); // approx. on the wall peak
  it("zero outside the jet band, positive inside (both lobes)", () => {
    expect(jetShape(8, 0.12, 0, 0, 60, 1, 2, 0.9)).toBeGreaterThan(0);
    expect(jetShape(8, Math.PI - 0.12, 0, 0, 60, 1, 2, 0.9)).toBeGreaterThan(0);
    expect(jetShape(1.5, 0.12, 0, 0, 60, 1, 2, 0.9)).toBe(0);        // below zBase
    expect(jetShape(400, 0.12, 0, 0, 60, 1, 2, 0.9)).toBe(0);        // beyond jetLength
    expect(jetShape(8, Math.PI / 2, 0, 0, 60, 1, 2, 0.9)).toBe(0);   // equatorial
  });
  it("density x f and width x sqrt(f) at the plasma's launch time", () => {
    const r = 30, th = thWall(30), z = r * Math.cos(th), rho = r * Math.sin(th);
    for (const t of [500, 2100, 7777]) {
      const tl = t - launchDelay(z, 2), f = fluxRatio(tl, 1, 0.9), sw = Math.sqrt(f);
      const q = rho / (sw * funnelEdge(z));
      const want = f * wallProfile(rho / sw, z) * lengthFalloff(z, 60) * filaments(q, comovingAzimuth(0.4, z, 0.9, 2), tl);
      expect(jetShape(r, th, 0.4, t, 60, 1, 2, 0.9)).toBeCloseTo(want, 12);
    }
  });
  it("the pattern rides the flow: what the base launched appears at height z after tau(z)", () => {
    // Same launch time and co-moving azimuth => same flux ratio and filament value at both heights.
    const a = 0.9, g = 2, t0 = 3000, ph0 = 1.1;
    for (const z of [10, 40]) {
      const tl = t0, t = tl + launchDelay(z, g), ph = ph0 + fieldLineOmega(a) * (launchDelay(z, g) - (z - JET.zBase));
      expect(comovingAzimuth(ph, z, a, g)).toBeCloseTo(ph0, 10);
      expect(fluxRatio(t - launchDelay(z, g), 1, 0.9)).toBeCloseTo(fluxRatio(tl, 1, 0.9), 12);
    }
  });
  it("s = 0 gives the steady jet: wall x falloff x filaments, no width change (Review Focus 3)", () => {
    const r = 20, th = thWall(20), z = r * Math.cos(th), rho = r * Math.sin(th), tl = 900 - launchDelay(z, 2);
    const q = rho / funnelEdge(z);
    expect(jetShape(r, th, 0.2, 900, 60, 0, 2, 0.9))
      .toBeCloseTo(wallProfile(rho, z) * lengthFalloff(z, 60) * filaments(q, comovingAzimuth(0.2, z, 0.9, 2), tl), 12);
  });
  it("filaments: mean one, amplitude bound 0.35, periodic and continuous across the azimuth wrap (Review Focus 4)", () => {
    let m = 0, n = 0;
    for (let i = 0; i < 20000; i++) { const v = filaments(0.3 + (i % 7) * 0.13, i * 0.731, i * 3.17); m += v; n++;
      expect(v).toBeGreaterThanOrEqual(1 - FILAMENT.amp - 1e-12); expect(v).toBeLessThanOrEqual(1 + FILAMENT.amp + 1e-12); }
    expect(m / n).toBeCloseTo(1, 1);
    for (const ph of [0, 1, -2.5]) expect(filaments(0.8, ph + 2 * Math.PI, 444)).toBeCloseTo(filaments(0.8, ph, 444), 10);
    expect(Math.abs(filaments(0.8, 2 * Math.PI - 1e-9, 444) - filaments(0.8, 1e-9, 444))).toBeLessThan(1e-6);
  });
  it("vnoise3 interpolates node values and wraps y with period n", () => {
    expect(vnoise3(3, 2, 8, 5, 7)).toBeCloseTo(vnoise3(3, 10, 8, 5, 7), 12);
    expect(vnoise3(3.5, 2.25, 8, 5.75, 7)).toBeGreaterThanOrEqual(0);
    expect(vnoise3(3.5, 2.25, 8, 5.75, 7)).toBeLessThanOrEqual(1);
  });
});

describe("jet envelope (geodesic-cache bookmark region), widened for the flux-driven width", () => {
  it("JET_ENV_Q covers the widest jet the slider allows", () => {
    const fMax = (1 + FLUX.sMax * FLUX.eps * FLUX.flickerClip) / (1 - FLUX.sMax * FLUX.d1);
    expect(JET_ENV_Q).toBeGreaterThanOrEqual(1.2 * Math.sqrt(fMax));
  });
  it("contains every point where the jet can emit, at any flux and slider value (Review Focus 2)", () => {
    for (let r = 1.2; r < 80; r *= 1.07)
      for (let th = 0.001; th < Math.PI; th += 0.013)
        for (const t of [0, 333, 1777, 2950]) for (const s of [0, 1, FLUX.sMax]) for (const a of [0.3, 0.9, 0.998])
          if (jetShape(r, th, 0.5, t, 60, s, 2, a) > 0) expect(inJetEnvelope(r, th, 60)).toBe(true);
  });
  it("excludes below the launch height, beyond the length, and outside the wall", () => {
    expect(inJetEnvelope(1.5, 0.01, 60)).toBe(false);
    expect(inJetEnvelope(70, 0.01, 60)).toBe(false);
    expect(inJetEnvelope(20, Math.PI / 2 - 0.2, 60)).toBe(false);
    expect(inJetEnvelope(20, 0.03, 60)).toBe(true);
    expect(inJetEnvelope(20, Math.PI - 0.03, 60)).toBe(true);
  });
  it("does not depend on jet strength or the flux slider (it takes neither)", () => {
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

const WGSL_E = readFileSync(join(__dirname, "../src/render/emission-shared.wgsl"), "utf8");
const wconst = (name: string) => { const m = WGSL_E.match(new RegExp(`const ${name}\\s*=\\s*([^;]+);`)); if (!m) throw new Error(`no ${name}`); return m[1]; };
describe("emission-shared.wgsl flux/filament constants match the CPU twins", () => {
  it("flux generator, filaments, envelope, Gauss nodes", () => {
    const pairs: [string, number][] = [["FLUX_T", FLUX.T], ["FLUX_JIT", FLUX.jitter], ["FLUX_TAUD", FLUX.tauD],
      ["FLUX_SPREAD", FLUX.spread], ["FLUX_FLOOR", FLUX.floor], ["FLUX_CELL_N", FLUX.flickerCell], ["FLUX_CLIP", FLUX.flickerClip], ["FLUX_SALT_N", FLUX.saltN], ["FLUX_DBAR", FLUX.dbar], ["FLUX_EPS", FLUX.eps], ["FLUX_D1", FLUX.d1], ["FLARE_RMIN", 5], ["FLARE_RMAX", 30], ["FLARE_SIZE", 0.2], ["FLARE_CA", FLARE.cA], ["FLARE_SALT_R", FLARE.saltR], ["FLARE_SALT_PHI", FLARE.saltPhi],
      ["FLUX_SALT_T", FLUX.saltT], ["FLUX_SALT_D", FLUX.saltD], ["FIL_AMP", FILAMENT.amp], ["FIL_CELL_T", FILAMENT.cellT],
      ["FIL_CELLS_PHI", FILAMENT.cellsPhi], ["FIL_CELLS_Q", FILAMENT.cellsQ], ["FIL_SALT", FILAMENT.salt], ["JET_ENV_Q", JET_ENV_Q]];
    // Number() parses decimals and the hex salts (0x464c); WGSL's u suffix is stripped first.
    for (const [n, v] of pairs) expect(Math.abs(Number(wconst(n).trim().replace(/u$/, "")) - v)).toBeLessThan(1e-7 * Math.max(1, Math.abs(v)));
    const nodes = wconst("GL6_X").match(/-?\d+\.\d+/g)!.map(Number), wts = wconst("GL6_W").match(/-?\d+\.\d+/g)!.map(Number);
    GL6.forEach(([x, w], i) => { expect(nodes[i]).toBeCloseTo(x, 7); expect(wts[i]).toBeCloseTo(w, 7); });
  });
  it("the old knot and churn noise is gone from the shared jet code", () => {
    expect(WGSL_E).not.toMatch(/knotsJ|JET_KZ|JET_TURB|JET_SEED/);
  });
});

describe("jet shape cost (final review: the live trace slowed ~30 % when launch time ran before the wall test)", () => {
  it("jetShapeJ rejects samples outside the envelope before any launch-time / flux work", () => {
    const body = WGSL_E.slice(WGSL_E.indexOf("fn jetShapeJ("), WGSL_E.indexOf("struct SynchOut"));
    const reject = body.search(/r \* sin\(th\) > JET_ENV_Q \* funnelEdgeJ\(z\)\) \{ return 0\.0; \}/);
    expect(reject).toBeGreaterThan(0);
    expect(reject).toBeLessThan(body.indexOf("launchDelayJ("));
    expect(reject).toBeLessThan(body.indexOf("fluxRatioJ("));
  });
  it("the CPU twin rejects the same region with the same result (zero there for any flux)", () => {
    for (const s of [0, 1, FLUX.sMax]) for (const t of [0, 1777]) {
      const z = 20, rho = JET_ENV_Q * funnelEdge(z) * 1.001, r = Math.hypot(rho, z), th = Math.atan2(rho, z);
      expect(jetShape(r, th, 0.3, t, 60, s, 2, 0.9)).toBe(0);
    }
  });
});
