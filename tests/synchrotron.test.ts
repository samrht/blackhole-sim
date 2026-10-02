import { describe, it, expect } from "vitest";
import { gammaProfile, plasmaShift, streamlineDir, slabStep } from "../src/physics/synchrotron";
import { screenToState } from "../src/physics/camera";
import { jetBandMatrix, JET_BANDS_NM } from "../src/physics/synchrotron";
import { blackbodyVisibleRGB, relLuminance } from "../src/physics/color";
import { VIS_LREF } from "../src/physics/lookups";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseTable } from "../src/physics/cyclosynch";
import { jetEnergetics, jetUniforms, injUnit, nuLnuAt, flowTime, jetField, ETA_DEFAULT, C_CGS, LN_NUB0 } from "../src/physics/synchrotron";
import { TABLE_GRID } from "../src/physics/cyclosynch";
import { PRESETS } from "../src/physics/presets";
import { JET } from "../src/physics/jet";
const T = parseTable(readFileSync(join(__dirname, "../public/synch-table.bin")).buffer.slice(0) as ArrayBuffer);

describe("jet model and energy budget", () => {
  it("proper speed Gamma*beta = sqrt(G280^2 - 1) (|z|/280)^0.58: anchored at 280 r_g, moving everywhere in view", () => {
    expect(gammaProfile(280, 2)).toBeCloseTo(2, 12); expect(gammaProfile(-280, 2)).toBeCloseTo(2, 12);
    const ub = (z: number) => Math.sqrt(3) * Math.pow(z / 280, 0.58);
    expect(gammaProfile(60, 2)).toBeCloseTo(Math.sqrt(1 + ub(60) ** 2), 12);      // beta = 0.58: no static plasma
    expect(Math.sqrt(1 - 1 / gammaProfile(60, 2) ** 2)).toBeGreaterThan(0.5);
    expect(Math.sqrt(1 - 1 / gammaProfile(10, 2) ** 2)).toBeGreaterThan(0.2);     // mildly relativistic near the base
    expect(gammaProfile(1e5, 2) / (Math.sqrt(3) * Math.pow(1e5 / 280, 0.58))).toBeCloseTo(1, 3); // Gamma ~ z^0.58 when Gamma >> 1
    expect(gammaProfile(0, 2)).toBe(1);
  });
  it("plasma frequency shift: static at r = 1000 ~1 (gravitational), toward/away Doppler (Review Focus 5)", () => {
    const s = screenToState(0, 0, 0, Math.PI / 2, 1000), b = Math.sqrt(1 - 1 / 9);
    expect(plasmaShift(s, 0, 1, 1, 0)).toBeCloseTo(1 / Math.sqrt(1 - 2 / 1000), 4);
    expect(plasmaShift(s, 0, 3, 1, 0) / (3 * (1 - b))).toBeCloseTo(1, 2);       // toward camera: blueshift
    expect(plasmaShift(s, 0, 3, -1, 0) / (3 * (1 + b))).toBeCloseTo(1, 2);      // counter-jet: redshift, finite
  });
  it("streamlines are unit vectors, mirror-symmetric between lobes, more collimated than radial", () => {
    const [ur, ut] = streamlineDir(20, 0.2), [lr, lt] = streamlineDir(20, Math.PI - 0.2);
    expect(Math.hypot(ur, ut)).toBeCloseTo(1, 12);
    expect(lr).toBeCloseTo(ur, 12); expect(lt).toBeCloseTo(-ut, 12);
    expect(ut).toBeLessThan(0); // upper lobe bends toward the axis relative to radial
  });
  it("slab transfer: thin -> j ds, thick -> source function j/alpha, earlier light attenuated (spec 5)", () => {
    const [It] = slabStep(0, 0, 2, 1e-9, 3);
    expect(It).toBeCloseTo(6, 6);                                                 // optically thin
    const [Ik, tk] = slabStep(0, 0, 2, 1e3, 3);
    expect(Ik).toBeCloseTo(2 / 1e3, 12); expect(tk).toBe(3e3);                    // saturates at S = j/alpha
    let I = 0, tau = 0;
    for (let k = 0; k < 400; k++) [I, tau] = slabStep(I, tau, 2, 0.5, 0.05);      // 400 thin pieces = one slab
    expect(I).toBeCloseTo((2 / 0.5) * (1 - Math.exp(-10)), 6);                    // exact uniform-slab result
    const [Ib] = slabStep(5, 1, 2, 1, 1);                                         // gas behind tau = 1 dims by e^-1
    expect(Ib).toBeCloseTo(5 + 2 * (1 - Math.exp(-1)) * Math.exp(-1), 12);
  });
});

describe("jet colour in the disk's units (spec 2.4/2.5)", () => {
  it("a blackbody fed through the three bands has the disk path's luminance within 1 %", () => {
    const M = jetBandMatrix(), h = 6.62607015e-27, kB = 1.380649e-16;
    const Bnu = (T: number, nu: number) => (2 * h * nu ** 3) / C_CGS ** 2 / (Math.exp((h * nu) / (kB * T)) - 1);
    for (const T of [6000, 3e4, 1e7]) {
      const I = JET_BANDS_NM.map((nm) => Bnu(T, C_CGS / (nm * 1e-7)));
      const rgb = [0, 1, 2].map((r) => M[r][0] * I[0] + M[r][1] * I[1] + M[r][2] * I[2]) as [number, number, number];
      const ref = blackbodyVisibleRGB(T).map((v) => v / VIS_LREF) as [number, number, number];
      expect(Math.abs(relLuminance(rgb) / relLuminance(ref) - 1)).toBeLessThan(0.01); // measured 0.5-0.6 %
    }
  });
});

describe("full-spectrum energy budget (spec 2.4)", () => {
  it("P_BZ carries the high-spin correction f = 1 + 1.38 w^2 - 9.2 w^4; M87* stays in 1e43-1e44", () => {
    expect(jetEnergetics(6.5e9, 0.9, 1e-5).fH).toBeCloseTo(1.0468, 3);
    expect(jetEnergetics(21.2, 0.998, 0.02).fH).toBeCloseTo(0.8576, 3);
    const m = PRESETS.find((p) => p.id === "m87")!, E = jetEnergetics(m.massSun, m.a, m.lambda);
    expect(E.pBZ).toBeGreaterThan(1e43); expect(E.pBZ).toBeLessThan(1e44); // measured 4.46e43
  });
  it("injected power at infinity equals eta P_BZ", () => {
    for (const [m, a, l, eta] of [[6.5e9, 0.9, 8e-6, 0.05], [21.2, 0.998, 0.02, 1e-3], [1e8, 0.5, 3e-4, 1]]) {
      const U = jetUniforms(m, a, l, eta, 60, 2);
      expect((U.jetQ0 * injUnit(a, U.jetB0, U.rgCm, 60, 2)) / (eta * U.pBZ)).toBeCloseTo(1, 9);
    }
  });
  it("the default eta reproduces M87's optical nucleus: nu L_nu(550 nm) = 1e41 erg/s within 1 %", () => {
    const m = PRESETS.find((p) => p.id === "m87")!, U = jetUniforms(m.massSun, m.a, m.lambda, ETA_DEFAULT, 60, 2);
    const L = nuLnuAt(T, C_CGS / 550e-7, m.a, U.jetB0, U.rgCm, 60, 2, U.jetQ0);
    expect(Math.abs(L / 1e41 - 1)).toBeLessThan(0.01);
  });
  it("flow time is the integral of dz / (Gamma beta c)", () => {
    const rg = 1e15, z = 37; let s = 0; const N = 20000;
    for (let i = 0; i < N; i++) { const zz = 2 + (z - 2) * (i + 0.5) / N, G = gammaProfile(zz, 2.5); s += (z - 2) / N / Math.sqrt(G * G - 1); }
    expect(flowTime(z, 2.5, rg) / (s * rg / C_CGS)).toBeCloseTo(1, 5);
  });
  it("slider corners: q0 finite and >= 0 (0 at spin 0); the strongest field keeps x inside the table (Review Focus 1, 2)", () => {
    for (const [m, a, l, eta, g] of [[1, 0, 1e-10, 1e-4, 1.5], [1e10, 0.998, 1, 1, 8], [1, 0.998, 1, 1, 1.5], [1e10, 0.001, 1e-10, 1e-4, 8]]) {
      const U = jetUniforms(m, a, l, eta, 60, g);
      expect(Number.isFinite(U.jetQ0)).toBe(true); expect(U.jetQ0).toBeGreaterThanOrEqual(0);
      if (a === 0) expect(U.jetQ0).toBe(0);
      // the reddest plasma-frame frequency (650 nm, D = 0.1) at the strongest field (the funnel base)
      const B = jetField(0.5, JET.zBase + 0.05, a, U.jetB0), lnx = Math.log(0.1 * C_CGS / 650e-7) - LN_NUB0 - Math.log(B);
      expect(lnx).toBeGreaterThan(TABLE_GRID.lnx0);
      expect(Number.isFinite(Math.log(flowTime(30, g, U.rgCm)))).toBe(true);
    }
  });
});
