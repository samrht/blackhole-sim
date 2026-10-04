import { describe, it, expect } from "vitest";
import { HOTFLOW, mahadevanM, flowDensity, flowTemperature, flowField, iscoEL, flowVelocity, flowCoeffs, isHotFlow, flowN0, HOTFLOW_N0 } from "../src/physics/hot-flow";
import { ringDiameterUas } from "../src/physics/hot-flow-image";
import { metricLower } from "../src/physics/kerr";
import { iscoRadius } from "../src/physics/orbits";
import { PRESETS } from "../src/physics/presets";

describe("hot flow (spec 2026-10-04)", () => {
  it("Mahadevan's isotropic fit M(X) at reference points (formula of Mahadevan et al. 1996 / Leung et al. 2011)", () => {
    const M = (X: number) => 4.0505 * X ** (-1 / 6) * (1 + 0.4 * X ** -0.25 + 0.5316 * X ** -0.5) * Math.exp(-1.8899 * X ** (1 / 3));
    for (const X of [0.1, 1, 10, 100, 1e4]) expect(mahadevanM(X)).toBeCloseTo(M(X), 12);
    expect(mahadevanM(1e9)).toBeLessThan(1e-200 + 1e-300);
  });
  it("Broderick profiles: density r^-1.1 with a Gaussian in z/rho, temperature r^-0.84, beta = 10 toroidal field", () => {
    expect(flowDensity(4, Math.PI / 2, 1e7)).toBeCloseTo(1e7 * 2 ** -1.1, 3);
    const th = Math.atan2(4, 3), r = 5; // rho = 4, z = 3
    expect(flowDensity(r, th, 1e7)).toBeCloseTo(1e7 * (r / 2) ** -1.1 * Math.exp(-9 / 32), 3);
    expect(flowDensity(10, 1e-9, 1e7)).toBe(0); // on the axis
    expect(flowTemperature(2)).toBeCloseTo(1e11, 0); expect(flowTemperature(8)).toBeCloseTo(1e11 * 4 ** -0.84, 0);
    const n = 3e6, r2 = 6, B = flowField(r2, n);
    expect((B * B) / (8 * Math.PI)).toBeCloseTo((n * 1.67262192e-24 * 2.99792458e10 ** 2 * (2 / r2)) / (12 * 10), 6);
  });
  it("Pu et al. velocity: normalised, Keplerian / free-fall limits, plunge inside the ISCO, K0 <= 0 -> null", () => {
    const a = 0.94, risco = iscoRadius(a, true);
    for (const [r, th] of [[10, 1.4], [risco * 0.7, 1.5], [3, 0.8]] as [number, number][]) {
      const u = flowVelocity(r, th, a)!; const g = metricLower(r, th, a);
      const uphi = u.Om * u.ut;
      const norm = g.tt * u.ut ** 2 + 2 * g.tphi * u.ut * uphi + g.phph * uphi ** 2 + g.rr * u.ur ** 2;
      expect(norm).toBeCloseTo(-1, 9);
    }
    const kep = flowVelocity(10, Math.PI / 2, a, 1, 1)!; expect(kep.ur).toBe(0); expect(kep.Om).toBeCloseTo(1 / (10 ** 1.5 + a), 12);
    const ff = flowVelocity(10, Math.PI / 2, a, 0, 0)!; expect(ff.ur).toBeLessThan(0);
    const ge = metricLower(10, Math.PI / 2, a); expect(ff.Om).toBeCloseTo(-ge.tphi / ge.phph, 9); // zero angular momentum = ZAMO rate
    expect(flowVelocity(risco * 0.8, Math.PI / 2, a, 1, 1)!.ur).toBeLessThan(0);  // Keplerian plunge inside the ISCO
    const [E, L] = iscoEL(0); expect(E).toBeCloseTo(Math.sqrt(8 / 9), 9); expect(L).toBeCloseTo(2 * Math.sqrt(3), 9);
  });
  it("coefficients: Kirchhoff with the Rayleigh-Jeans source function; zero where there is no gas", () => {
    const [j, al] = flowCoeffs(6, 1.3, 230e9, 1e7);
    expect(j).toBeGreaterThan(0);
    expect((j / al) / ((2 * 230e9 ** 2 * 1.380649e-16 * flowTemperature(6)) / 2.99792458e10 ** 2)).toBeCloseTo(1, 9);
    expect(flowCoeffs(6, 1e-9, 230e9, 1e7)).toEqual([0, 0]);
  });
  it("regime: hot below 1 % Eddington; n0 scaled lambda / M from the calibrated object nearest in mass", () => {
    expect(isHotFlow(0.009)).toBe(true); expect(isHotFlow(0.011)).toBe(false);
    const sg = PRESETS.find((p) => p.id === "sgra")!, m87 = PRESETS.find((p) => p.id === "m87")!;
    expect(isHotFlow(sg.lambda)).toBe(true); expect(isHotFlow(m87.lambda)).toBe(true);
    expect(isHotFlow(PRESETS.find((p) => p.id === "cygx1")!.lambda)).toBe(false);
    // Each calibrated object gets its own n0 from its mass and accretion alone (no preset id: spin, inclination or a
    // Custom link with the same mass and accretion cannot change it).
    expect(flowN0(sg.massSun, sg.lambda)).toBeCloseTo(HOTFLOW_N0.sgra, 6);
    expect(flowN0(m87.massSun, m87.lambda) / HOTFLOW_N0.m87).toBeCloseTo(1, 12);
    expect(flowN0(2 * sg.massSun, 3 * sg.lambda) / HOTFLOW_N0.sgra).toBeCloseTo(1.5, 9);   // n0 ~ lambda / M near Sgr A*
    expect(flowN0(m87.massSun / 2, m87.lambda) / HOTFLOW_N0.m87).toBeCloseTo(2, 9);        // ... and near M87*
    expect(HOTFLOW.nu).toBe(230e9);
  });
});

describe("ring diameter measure (hot-flow-image.ts)", () => {
  // A thin Gaussian ring of radius R M; the measure must not depend on the frame or the pixel grid (the 60-bin profile
  // alone quantised it to ~2.3 uas: 46.0 vs 48.2 at 48^2 vs 96^2, 47.1 on the GPU's 28 M frame vs 48.2 on the CPU's 26 M).
  // Grids as fine as the gates use (CPU 96^2, GPU 512^2); at 48^2 the 0.54 M pixels are coarser than the bins.
  it("recovers a ring's diameter to 0.05 M (0.26 uas for Sgr A*) for any frame and resolution", () => {
    for (const R of [4.35, 4.6, 4.83, 5.1]) for (const [N, half] of [[96, 13], [128, 14], [512, 14]]) {
      const I = new Float64Array(N * N), px = (2 * half) / N;
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const b = Math.hypot(-half + px * (i + 0.5), -half + px * (j + 0.5)); I[j * N + i] = Math.exp(-((b - R) ** 2) / (2 * 0.6 ** 2));
      }
      expect(Math.abs(ringDiameterUas(I, N, half, 1) - 2 * R)).toBeLessThan(0.05);
    }
  });
});
