import { describe, it, expect } from "vitest";
import { bessel, besselRecurrence, kernelExact, kernelSync, synchF, nHatU, nHat, dNOverGu, gInj, I_G } from "../src/physics/cyclosynch";

describe("Bessel functions", () => {
  it("recurrence matches reference values", () => {
    expect(bessel(5, 3)[0]).toBeCloseTo(0.04302843487704758, 12);
    expect(bessel(1, 1)[0]).toBeCloseTo(0.44005058574493355, 12);
    expect(bessel(1, 1)[1]).toBeCloseTo(0.7651976865579666 - 0.44005058574493355, 12); // J1' = J0 - J1/z
  });
  it("large orders (uniform expansion, J' by recurrence) agree with the exact recurrence within 1e-3", () => {
    for (const n of [41, 100, 400]) for (const w of [0.5, 0.9, 0.99]) {
      const a = besselRecurrence(n, n * w), b = bessel(n, n * w);
      expect(Math.abs(b[0] / a[0] - 1)).toBeLessThan(1e-3);
      expect(Math.abs(b[1] / a[1] - 1)).toBeLessThan(1e-3);
    }
  });
});
describe("single-electron cyclo-synchrotron spectrum", () => {
  it("integrated over frequency it equals the Larmor power (4/9) u^2 within 0.5 %", () => {
    for (const u of [0.3, 1]) {
      const g = Math.sqrt(1 + u * u), beta = u / g, N = 1500;
      const l0 = Math.log(0.5 / g / (1 + beta)), l1 = Math.log(Math.min(4e4, 40 * (1 + 1.5 * g ** 3) / (1 - beta) / g)); let s = 0;
      for (let i = 0; i < N; i++) { const x = Math.exp(l0 + (l1 - l0) * (i + 0.5) / N); s += kernelExact(x, g) * x * (l1 - l0) / N; }
      expect(Math.abs(s / ((4 / 9) * u * u) - 1)).toBeLessThan(5e-3); // measured 0.07 % / 0.13 %
    }
  }, 120000);
  it("synchrotron function F matches reference values", () => {
    expect(Math.abs(synchF(1) / 0.651423 - 1)).toBeLessThan(1e-4);
    expect(Math.abs(synchF(0.29) / 0.918 - 1)).toBeLessThan(5e-4);
    expect(synchF(1e-6 * 1.001) / synchF(1e-6 * 0.999)).toBeCloseTo(Math.cbrt(1.001 / 0.999), 3);
  });
  it("at the seam (gamma = 10) the synchrotron kernel is within 1.5 % of the exact sum, power-weighted", () => {
    const g = 10; let num = 0, den = 0;
    for (let i = 0; i < 40; i++) { const x = Math.exp(Math.log(0.5) + (Math.log(40 * 1.5 * g * g) - Math.log(0.5)) * i / 39);
      const e = kernelExact(x, g), s = kernelSync(x, g); num += Math.abs(s - e) * x; den += e * x; }
    expect(num / den).toBeLessThan(0.015); // measured 1.3e-2 (14 % at gamma 3, 5.1 % at 5)
  }, 120000);
});
describe("cooled population (exact, constant injection for a cooling depth s)", () => {
  it("solves the continuity equation d/dgamma [u^2 N^] = -g in the fully cooled range", () => {
    for (const g of [50, 500, 5000]) { const h = 1e-4 * g;
      const d = ((g + h) ** 2 - 1) * nHat(g + h, 10) - ((g - h) ** 2 - 1) * nHat(g - h, 10);
      expect(d / (2 * h) / -gInj(g)).toBeCloseTo(1, 6); }
  });
  it("radiates what it is given when fast cooling, less when slow (calorimetry)", () => {
    const frac = (s: number) => { let r = 0; const N = 200000, l0 = Math.log(1e-12), l1 = Math.log(1e7);
      for (let i = 0; i < N; i++) { const u = Math.exp(l0 + (l1 - l0) * (i + 0.5) / N); r += nHatU(u, s) * u * u * (u * u / Math.sqrt(1 + u * u)) * (l1 - l0) / N; } return r / I_G; };
    expect(frac(10)).toBeCloseTo(1, 4);
    expect(frac(1e-3)).toBeLessThan(0.3); // measured 0.285: most injected energy not yet radiated
  });
  it("I_G matches a direct integral", () => {
    let s = 0; const N = 400000, l0 = Math.log(40), l1 = Math.log(1e9);
    for (let i = 0; i < N; i++) { const g = Math.exp(l0 + (l1 - l0) * (i + 0.5) / N); s += (g - 1) * gInj(g) * g * (l1 - l0) / N; }
    expect(s / I_G).toBeCloseTo(1, 5);
  });
  it("the absorption derivative is the analytic derivative of N^ / (gamma u)", () => {
    for (const [g, s] of [[3, 10], [50, 10], [2000, 10], [60, 0.01]]) { const h = 1e-6 * g, f = (x: number) => nHat(x, s) / (x * Math.sqrt(x * x - 1));
      expect(dNOverGu(g, s) / ((f(g + h) - f(g - h)) / (2 * h))).toBeCloseTo(1, 5); }
  });
  it("slow cooling (s -> 0) gives N^ = s g above gamma_min", () => {
    expect(nHat(100, 1e-6) / (1e-6 * gInj(100))).toBeCloseTo(1, 3);
  });
});
