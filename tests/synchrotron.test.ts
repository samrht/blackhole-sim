import { describe, it, expect } from "vitest";
import { gammaProfile, plasmaShift, streamlineDir, slabStep } from "../src/physics/synchrotron";
import { screenToState } from "../src/physics/camera";
import { jetBandMatrix, JET_BANDS_NM } from "../src/physics/synchrotron";
import { blackbodyVisibleRGB, relLuminance } from "../src/physics/color";
import { VIS_LREF } from "../src/physics/lookups";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseTable } from "../src/physics/cyclosynch";
import { jetEnergetics, jetUniforms, injUnit, meanInjection, nuLnuAt, flowTime, jetField, ETA_DEFAULT, C_CGS, LN_NUB0 } from "../src/physics/synchrotron";
import { fluxRatio } from "../src/physics/flux-history";
import { TABLE_GRID } from "../src/physics/cyclosynch";
import { PRESETS } from "../src/physics/presets";
import { JET } from "../src/physics/jet";
import { LN_CJ, LN_CA, LN_K0, LN_T0 } from "../src/physics/synchrotron";
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
      const U = jetUniforms(m, a, l, eta, 60, 2, 0);
      expect((U.jetQ0 * injUnit(a, U.jetB0, U.rgCm, 60, 2)) / (eta * U.pBZ)).toBeCloseTo(1, 9);
    }
  });
  it("the default eta reproduces M87's optical nucleus: nu L_nu(550 nm) = 1e41 erg/s within 1 %", () => {
    const m = PRESETS.find((p) => p.id === "m87")!, U = jetUniforms(m.massSun, m.a, m.lambda, ETA_DEFAULT, 60, 2, 0); // the anchor is the steady jet
    const L = nuLnuAt(T, C_CGS / 550e-7, m.a, U.jetB0, U.rgCm, 60, 2, U.jetQ0);
    expect(Math.abs(L / 1e41 - 1)).toBeLessThan(0.01);
  });
  it("M87's anchor also holds at the default Flux variability (1x), averaged over the flux distribution (review minor)", () => {
    const m = PRESETS.find((p) => p.id === "m87")!, U = jetUniforms(m.massSun, m.a, m.lambda, ETA_DEFAULT, 60, 2, 1);
    const fq = fluxQuantiles(1, m.a);
    const L = fq.reduce((acc, f) => acc + nuLnuAt(T, C_CGS / 550e-7, m.a, U.jetB0, U.rgCm, 60, 2, U.jetQ0, f), 0) / fq.length;
    expect(Math.abs(L / 1e41 - 1)).toBeLessThan(0.01);
  });
  it("flow time is the integral of dz / (Gamma beta c)", () => {
    const rg = 1e15, z = 37; let s = 0; const N = 20000;
    for (let i = 0; i < N; i++) { const zz = 2 + (z - 2) * (i + 0.5) / N, G = gammaProfile(zz, 2.5); s += (z - 2) / N / Math.sqrt(G * G - 1); }
    expect(flowTime(z, 2.5, rg) / (s * rg / C_CGS)).toBeCloseTo(1, 5);
  });
  it("slider corners: q0 finite and >= 0 (0 at spin 0); the strongest field keeps x inside the table (Review Focus 1, 2)", () => {
    for (const [m, a, l, eta, g] of [[1, 0, 1e-10, 1e-4, 1.5], [1e10, 0.998, 1, 1, 8], [1, 0.998, 1, 1, 1.5], [1e10, 0.001, 1e-10, 1e-4, 8]]) {
      const U = jetUniforms(m, a, l, eta, 60, g, 0);
      expect(Number.isFinite(U.jetQ0)).toBe(true); expect(U.jetQ0).toBeGreaterThanOrEqual(0);
      if (a === 0) expect(U.jetQ0).toBe(0);
      // the reddest plasma-frame frequency (650 nm, D = 0.1) at the strongest field (the funnel base)
      const B = jetField(0.5, JET.zBase + 0.05, a, U.jetB0), lnx = Math.log(0.1 * C_CGS / 650e-7) - LN_NUB0 - Math.log(B);
      expect(lnx).toBeGreaterThan(TABLE_GRID.lnx0);
      expect(Number.isFinite(Math.log(flowTime(30, g, U.rgCm)))).toBe(true);
    }
  });
});

describe("mean injected power with the flux-driven jet (spec 2.4 + corrections 2)", () => {
  const E = jetEnergetics(6.5e9, 0.9, 1e-5);
  const P = (f: number) => injUnit(0.9, E.b0, E.rgCm, 60, 2, f);
  const F = [0.5, 1, 1.5, 2];
  const lagrange = (Pn: number[]) => (f: number) =>
    F.reduce((s, fi, i) => s + Pn[i] * F.reduce((p, fj, j) => (j === i ? p : (p * (f - fj)) / (fi - fj)), 1), 0);
  it("injected power is ~f^2 (A + C f): the cubic through f = 0.5, 1, 1.5, 2 holds at other f within 0.5 %", () => {
    const lag = lagrange(F.map(P));
    for (const f of [0.17, 0.3, 0.75, 1.25, 1.58]) expect(Math.abs(lag(f) / P(f) - 1)).toBeLessThan(5e-3);
    expect(P(1)).toBeCloseTo(injUnit(0.9, E.b0, E.rgCm, 60, 2), 12); // default argument f = 1
  });
  it("s = 0 reproduces the steady jet; s > 0 averages the cubic with <f^2>, <f^3>", () => {
    expect(meanInjection(0.9, E.b0, E.rgCm, 60, 2, 0)).toBe(P(1));
    const lag = lagrange(F.map(P));
    for (const s of [0.5, 1, 1.4]) {
      // over the moments' own 1e6 M window (2e5 M samples different eruptions: <f> = 0.986)
      let m = 0, n = 0; for (let t = 0.5; t < 1e6; t += 3) { m += lag(fluxRatio(t, s, 0.9)); n++; }
      expect(Math.abs(meanInjection(0.9, E.b0, E.rgCm, 60, 2, s) / (m / n) - 1)).toBeLessThan(1e-2);
    }
  });
  it("time-averaged injected power: a direct average of injUnit over the flux distribution equals meanInjection within 1 %", () => {
    // Non-circular replacement of the q0 x meanInjection = eta P_BZ checks (true by construction; review minors).
    for (const a of [0.3, 0.9]) {
      const Ea = jetEnergetics(6.5e9, a, 1e-5);
      for (const s of [1, 1.4]) {
        const fq = fluxQuantiles(s, a);
        const direct = fq.reduce((acc, f) => acc + injUnit(a, Ea.b0, Ea.rgCm, 60, 2, f), 0) / fq.length;
        expect(Math.abs(meanInjection(a, Ea.b0, Ea.rgCm, 60, 2, s) / direct - 1)).toBeLessThan(0.01);
      }
    }
  });
  it("q0 stays finite at every slider value", () => {
    for (const s of [0, 0.5, 1, 1.4]) expect(Number.isFinite(jetUniforms(6.5e9, 0.9, 1e-5, 2e-3, 60, 2, s).jetQ0)).toBe(true);
  });
  it("spin 0: no BZ power, q0 = 0, no NaN at any slider value (Review Focus 5)", () => {
    for (const s of [0, 1, 1.4]) expect(jetUniforms(10, 0, 0.1, 2e-3, 60, 2, s).jetQ0).toBe(0);
  });
});

/** 64 equal-count bin means of f = Phi/<Phi> over the generator's 1e6 M window (quadrature of the f distribution). */
function fluxQuantiles(s: number, a: number): number[] {
  const v: number[] = []; for (let t = 2.5; t < 1e6; t += 5) v.push(fluxRatio(t, s, a));
  v.sort((x, y) => x - y); const n = 64, out: number[] = [];
  for (let i = 0; i < n; i++) { const lo = Math.floor((i * v.length) / n), hi = Math.floor(((i + 1) * v.length) / n); let m = 0; for (let j = lo; j < hi; j++) m += v[j]; out.push(m / (hi - lo)); }
  return out;
}
const WGSL = readFileSync(join(__dirname, "../src/render/emission-shared.wgsl"), "utf8");
const rawConst = (name: string) => { const m = WGSL.match(new RegExp(`const ${name}\\s*=\\s*([^;]+);`)); if (!m) throw new Error(`no ${name}`); return m[1]; };
const constOf = (name: string) => +rawConst(name);
describe("emission-shared.wgsl constants match the CPU twin", () => {
  it("coefficient prefactors, cyclotron and cooling scales, table grid", () => {
    for (const [n, v] of [["SYN_LNCJ", LN_CJ], ["SYN_LNCA", LN_CA], ["SYN_LNNUB0", LN_NUB0], ["SYN_LNK0", LN_K0], ["SYN_LNT0", LN_T0],
      ["SYN_LNX0", TABLE_GRID.lnx0], ["SYN_LNX1", TABLE_GRID.lnx1], ["SYN_LNS0", TABLE_GRID.lns0], ["SYN_LNS1", TABLE_GRID.lns1]] as [string, number][])
      expect(constOf(n)).toBeCloseTo(v, 6);
  });
  it("band frequencies and the band matrix (column b = band b)", () => {
    const raw = rawConst;
    const args = (s: string) => s.slice(s.indexOf("(") + 1).split(",").map((v) => +v.replace(")", ""));
    const nu = args(raw("JET_LNNU")); expect(nu.length).toBe(3);
    nu.forEach((v, b) => expect(v).toBeCloseTo(Math.log(C_CGS / (JET_BANDS_NM[b] * 1e-7)), 6));
    const M = jetBandMatrix(), m = args(raw("JET_BAND_M")); expect(m.length).toBe(9);
    for (let b = 0; b < 3; b++) for (let r = 0; r < 3; r++) expect(Math.abs(m[b * 3 + r] / M[r][b] - 1)).toBeLessThan(1e-7);
  });
});
