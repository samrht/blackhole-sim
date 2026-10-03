import { describe, it, expect } from "vitest";
import { FLARE, tubeOf, tubeAt, tubeLife, tubeField, tubeEnergy, tubePower, flareZeta, meanTubeEnergy } from "../src/physics/eruption-spots";
import { eruptionTime, eruptionDepth, FLUX } from "../src/physics/flux-history";
import { iscoRadius } from "../src/physics/orbits";
import { jetEnergetics } from "../src/physics/synchrotron";
import { PRESETS } from "../src/physics/presets";
import { readFileSync } from "node:fs"; import { join } from "node:path";
import { parseTable } from "../src/physics/cyclosynch";
import { tubeLight } from "../src/physics/eruption-spots";
import { jetCoeffs, C_CGS as C } from "../src/physics/synchrotron";

const sgra = PRESETS.find((p) => p.id === "sgra")!, E = jetEnergetics(sgra.massSun, sgra.a, sgra.lambda);
describe("eruption flux tubes (spec 2026-10-04 eruption flares)", () => {
  it("one tube per eruption, born at the eruption time with its depth and drop duration", () => {
    for (const k of [0, 7, 123]) {
      const t = tubeOf(k, 0.9);
      expect(t.t0).toBe(eruptionTime(k)); expect(t.depth).toBe(eruptionDepth(k, FLUX.dbar));
      expect(t.D).toBeCloseTo(-FLUX.tauD * Math.log(1 - t.depth), 12);
    }
  });
  it("orbit radius in [5, 30] r_g, size 0.2 r_c, outside ISCO + 2R at every spin (Review Focus 2)", () => {
    for (const a of [0, 0.5, 0.94]) for (let k = 0; k < 500; k++) {
      const t = tubeOf(k, a);
      expect(t.rc).toBeLessThanOrEqual(30); expect(t.R).toBeCloseTo(0.2 * t.rc, 12);
      expect(t.rc).toBeGreaterThanOrEqual(Math.max(5, iscoRadius(a, true) + 2 * t.R) - 1e-9);
      expect(t.phi0).toBeGreaterThanOrEqual(0); expect(t.phi0).toBeLessThan(2 * Math.PI);
    }
  });
  it("spirals from the ISCO to r_c over D with continuous azimuth, then orbits at the Keplerian rate", () => {
    const a = 0.9, t = tubeOf(5, a), rin = iscoRadius(a, true);
    expect(tubeAt(t, t.t0 + 1e-6, a).r).toBeCloseTo(rin, 4);
    const end = tubeAt(t, t.t0 + t.D - 1e-6, a), after = tubeAt(t, t.t0 + t.D + 1e-6, a);
    expect(end.r).toBeCloseTo(t.rc, 4); expect(after.phi).toBeCloseTo(end.phi, 4);
    const Om = 1 / (t.rc ** 1.5 + a), p1 = tubeAt(t, t.t0 + t.D + 50, a).phi, p2 = tubeAt(t, t.t0 + t.D + 150, a).phi;
    expect((p2 - p1) / 100).toBeCloseTo(Om, 10);
    // spiral azimuth = brute-force integral of Omega_K along r(tau)
    const tau = 0.6 * t.D; let ph = t.phi0; const n = 20000;
    for (let i = 0; i < n; i++) { const s = (i + 0.5) / n * tau, r = rin + (t.rc - rin) * s / t.D; ph += tau / n / (r ** 1.5 + a); }
    expect(tubeAt(t, t.t0 + tau, a).phi).toBeCloseTo(ph, 6);
  });
  it("light curve: linear rise over D, continuous, half-orbit e-fold, zero after 2 orbits and before birth", () => {
    const a = 0.9, t = tubeOf(9, a), P = 2 * Math.PI * (t.rc ** 1.5 + a);
    expect(tubeAt(t, t.t0 - 1, a).A).toBe(0);
    expect(tubeAt(t, t.t0 + 0.5 * t.D, a).A).toBeCloseTo(0.5, 12);
    expect(tubeAt(t, t.t0 + t.D + 1e-9, a).A).toBeCloseTo(1, 6);
    expect(tubeAt(t, t.t0 + t.D + 0.5 * P, a).A).toBeCloseTo(Math.exp(-1), 10);
    expect(tubeAt(t, t.t0 + t.D + 2 * P + 1e-6, a).A).toBe(0);
    expect(tubeLife(t, a)).toBeCloseTo(t.D + 2 * P, 9);
    let I = 0; const n = 200000, L = tubeLife(t, a);
    for (let i = 0; i < n; i++) I += tubeAt(t, t.t0 + (i + 0.5) / n * L, a).A * L / n;
    expect(I / (t.D / 2 + P * FLARE.cA)).toBeCloseTo(1, 4);
  });
  it("at most three tubes alive at once", () => {
    for (let tt = 0; tt < 3e5; tt += 37) {
      const k0 = Math.floor(tt / FLUX.T); let alive = 0;
      for (let k = k0 - 4; k <= k0 + 1; k++) if (tubeAt(tubeOf(k, 0.9), tt, 0.9).A > 0) alive++;
      expect(alive).toBeLessThanOrEqual(3);
      for (let k = k0 - 6; k < k0 - 2; k++) expect(tubeAt(tubeOf(k, 0.9), tt, 0.9).A).toBe(0); // k-2..k suffice
    }
  });
  it("field by flux conservation and energy (delta s)^2 Phi^2 / (4 pi^2 R r_g)", () => {
    const t = tubeOf(3, sgra.a);
    expect(tubeField(t, 1, E.phi, E.rgCm)).toBeCloseTo(t.depth * E.phi / (Math.PI * (t.R * E.rgCm) ** 2), 6);
    const B = tubeField(t, 1, E.phi, E.rgCm), V = Math.PI * (t.R * E.rgCm) ** 2 * 2 * t.R * E.rgCm;
    expect(tubeEnergy(t, 1, E.phi, E.rgCm) / (B * B * V / (8 * Math.PI))).toBeCloseTo(1, 10);
  });
  it("anchor: zeta = 1e38 / <E> at the Sgr A* preset (~1 % of the tube field energy)", () => {
    expect(flareZeta() * meanTubeEnergy(1, E.phi, E.rgCm)).toBeCloseTo(1e38, -33);
    expect(flareZeta()).toBeGreaterThan(1.0e-2); expect(flareZeta()).toBeLessThan(1.12e-2);
    let m = 0; const n = 20000; for (let k = 0; k < n; k++) m += tubeEnergy(tubeOf(k, sgra.a), 1, E.phi, E.rgCm);
    expect(m / n / meanTubeEnergy(1, E.phi, E.rgCm)).toBeCloseTo(1, 1); // closed form vs the hashed draws (sampling)
  });
  it("injected energy over a tube's life = zeta f E; f = 0 or s = 0 gives nothing (Review Focus 3)", () => {
    const a = sgra.a, t = tubeOf(11, a), L = tubeLife(t, a), z = flareZeta();
    let W = 0; const n = 100000; for (let i = 0; i < n; i++) W += tubePower(t, t.t0 + (i + 0.5) / n * L, a, 1, 1, E.phi, z, E.rgCm) * L / n * E.rgCm / 2.99792458e10;
    expect(W / (z * tubeEnergy(t, 1, E.phi, E.rgCm))).toBeCloseTo(1, 3);
    expect(tubePower(t, t.t0 + 0.3 * L, a, 1, 0, E.phi, z, E.rgCm)).toBe(0);
    expect(tubePower(t, t.t0 + 0.3 * L, a, 0, 1, E.phi, z, E.rgCm)).toBe(0);
  });
});

const TAB = parseTable(readFileSync(join(__dirname, "../public/synch-table.bin")).buffer.slice(0) as ArrayBuffer);
describe("tube light", () => {
  const a = sgra.a, z = flareZeta(), t = tubeOf(4, a), when = t.t0 + t.D + 0.2 * 2 * Math.PI * (t.rc ** 1.5 + a);
  const at = tubeAt(t, when, a);
  it("bright at the tube centre, Gaussian fall-off, zero far away and with flares off", () => {
    const c = tubeLight(TAB, at.r, at.phi, 1, when, a, 1, 1, E.phi, z, E.rgCm);
    expect(c[1]).toBeGreaterThan(0);
    const off = tubeLight(TAB, at.r + t.R, at.phi, 1, when, a, 1, 1, E.phi, z, E.rgCm);
    expect(off[1] / c[1]).toBeCloseTo(Math.exp(-0.5), 2);
    expect(tubeLight(TAB, at.r + 10 * t.R, at.phi, 1, when, a, 1, 1, E.phi, z, E.rgCm)[1]).toBe(0);
    expect(tubeLight(TAB, at.r, at.phi, 1, when, a, 1, 0, E.phi, z, E.rgCm)).toEqual([0, 0, 0]);
  });
  it("radiated / injected >= 0.4 over the r_c range (outer tubes are not fully fast-cooling; spec corrections 3)", () => {
    for (const k of [1, 2, 3, 4, 5, 6]) {
      const tk = tubeOf(k, a), B = tubeField(tk, 1, E.phi, E.rgCm), ageS = 600 * E.rgCm / C;
      const sc = Math.exp(-20.46682379) * B * B * ageS; let rad = 0; const n = 3000, l0 = Math.log(1e3), l1 = Math.log(1e24);
      for (let i = 0; i < n; i++) { const nu = Math.exp(l0 + (l1 - l0) * (i + 0.5) / n); rad += jetCoeffs(TAB, nu, B, sc, 1, 1)[0] * nu * (l1 - l0) / n; }
      expect(4 * Math.PI * rad / (B * B)).toBeGreaterThan(0.4);
    }
  });
});
