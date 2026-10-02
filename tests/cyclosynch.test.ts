import { describe, it, expect } from "vitest";
import { bessel, besselRecurrence, kernelExact, kernelSync, synchF, nHatU, nHat, dNOverGu, gInj, I_G } from "../src/physics/cyclosynch";
import { parseTable, lookup, TABLE_GRID, uGrid, U_COLD, kernelBin, dNOverGuU, kernelRow, contract, encodeTable } from "../src/physics/cyclosynch";
import { readFileSync } from "node:fs";
import { join } from "node:path";

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
const TAB = () => parseTable(readFileSync(join(__dirname, "../public/synch-table.bin")).buffer.slice(0) as ArrayBuffer);
describe("coefficient table (public/synch-table.bin)", () => {
  it("header matches the code's grid and injection constants (a stale table fails here)", () => {
    const t = TAB();
    expect([t.nx, t.ns]).toEqual([TABLE_GRID.nx, TABLE_GRID.ns]);
    expect(t.lnx0).toBeCloseTo(TABLE_GRID.lnx0, 5); expect(t.lns1).toBeCloseTo(TABLE_GRID.lns1, 5);
    const bad = new Uint8Array(readFileSync(join(__dirname, "../public/synch-table.bin"))); bad[0] ^= 1;
    expect(() => parseTable(bad.buffer)).toThrow();
  });
  it("calorimetry: Int J^ d omega over the table equals the population's radiated power (within 0.5 %)", () => {
    const t = TAB(), h = (t.lnx1 - t.lnx0) / t.nx;
    // s >= 1e-3: below it uncooled electrons up to gamma ~ 1/s emit beyond the table's top (x = 1e10), so the sum over
    // the table is incomplete by construction (57 % at s = 1e-7); the top-edge extension covers that range instead.
    for (const sWant of [1e-3, 0.58, 40]) {
      const js = Math.round((Math.log(sWant) - t.lns0) / (t.lns1 - t.lns0) * (t.ns - 1));
      const s = Math.exp(t.lns0 + (t.lns1 - t.lns0) * js / (t.ns - 1)); let tot = 0, pop = 0;
      for (let i = 0; i < t.nx; i++) { const x = Math.exp(t.lnx0 + h * (i + 0.5)); tot += Math.exp(t.data[2 * (js * t.nx + i)]) / (2 * Math.PI) * x * h; }
      const uu = uGrid(); uu.u.forEach((u, k) => { pop += nHatU(u, s) * u * u * (u * u / Math.sqrt(1 + u * u)) * uu.dlnu[k]; });
      expect(Math.abs(tot / ((4 / 9) * pop) - 1)).toBeLessThan(5e-3); // measured 4e-3 (s = 1e-3), 1e-4 elsewhere
    }
  });
  it("lookup agrees with a direct computation at off-grid points (1 % emission, 2 % absorption)", () => {
    const t = TAB(), uu = uGrid(), h = (t.lnx1 - t.lnx0) / t.nx; void U_COLD; void kernelBin; void dNOverGuU;
    for (const [x, s] of [[3.7e2, 5], [2.1e4, 0.3], [2.5e6, 12], [1.6, 30], [55, 0.01], [1e8, 1e-6]] as [number, number][]) {
      const g = { lnx0: Math.log(x) - h / 2, lnx1: Math.log(x) + h / 2, nx: 1, lns0: Math.log(s), lns1: Math.log(s) + 1, ns: 2 };
      const rows = uu.u.map((u) => { const r = new Float64Array(1); kernelRow(u, g, r); return r; }), d = contract(rows, uu, g);
      const [lJ, lA] = lookup(t, Math.log(x), Math.log(s));
      expect(Math.abs(Math.exp(lJ) / d.J[0] - 1)).toBeLessThan(1e-2);
      expect(Math.abs(Math.exp(lA) / d.A[0] - 1)).toBeLessThan(2e-2);
    }
  }, 300000);
  it("Kirchhoff: for a thermal population the contraction gives A^ = J^ / Theta", () => {
    const g = { ...TABLE_GRID, nx: 40, lnx0: Math.log(0.5), lnx1: Math.log(400), ns: 2 }, uu = uGrid();
    const keep = uu.u.map((u, k) => k).filter((k) => uu.u[k] > 0.05 && uu.u[k] < 30);
    const sub = { u: keep.map((k) => uu.u[k]), dlnu: keep.map((k) => uu.dlnu[k]) };
    const rows = sub.u.map((u) => { const r = new Float64Array(g.nx); kernelRow(u, g, r); return r; });
    for (const Th of [0.3, 1, 3]) {
      // thermal (Maxwell-Juttner) N = gamma u exp(-gamma/Theta) per unit gamma: N/(gamma u) = exp(-gamma/Theta)
      const th = { n: (u: number) => { const gm = Math.sqrt(1 + u * u); return gm * u * Math.exp(-gm / Th); } };
      const { J, A } = contract(rows, sub, g, th);
      // finite-volume absorption: exact up to the cell discretisation (measured 1.5e-4)
      for (let i = 5; i < g.nx; i += 7) if (J[i] > 0) expect(Math.abs(J[i] / (A[i] * Th) - 1)).toBeLessThan(2e-3);
    }
  }, 120000);
  it("edge extensions are continuous (top of x, bottom of s) and the slider range stays inside x >= 1e-4", () => {
    const t = TAB(), e = 1e-6;
    for (const lns of [Math.log(0.1), Math.log(10)]) {
      const a = lookup(t, t.lnx1 - (t.lnx1 - t.lnx0) / t.nx * 0.5 - e, lns), b = lookup(t, t.lnx1 - (t.lnx1 - t.lnx0) / t.nx * 0.5 + e, lns);
      expect(Math.abs(a[0] - b[0])).toBeLessThan(1e-4); expect(Math.abs(a[1] - b[1])).toBeLessThan(1e-4);
    }
    const c = lookup(t, Math.log(1e5), t.lns0 + e), d = lookup(t, Math.log(1e5), t.lns0 - e);
    expect(Math.abs(c[0] - d[0])).toBeLessThan(1e-4);
  });
});

describe("final-review fixes", () => {
  it("large-order Bessel functions stay finite and exact right at the turning point (z within 1e-8 of n)", () => {
    for (const [n, z] of [[47, 46.0000002], [46, 46.0000002], [100, 100.00000003], [60, 59.99999998]] as [number, number][]) {
      const [J, Jp] = bessel(n, z), [Jr, Jpr] = besselRecurrence(n, z);
      expect(Number.isFinite(J) && Number.isFinite(Jp)).toBe(true);
      expect(Math.abs(J / Jr - 1)).toBeLessThan(1e-3); expect(Math.abs(Jp / Jpr - 1)).toBeLessThan(1e-3);
    }
  });
  it("absorption is converged on the momentum grid where the cooled population ramps (vs a 10x finer grid)", () => {
    for (const [x, s] of [[0.6, 0.31], [0.35, 0.29], [1e-2, 1e-3], [1e-2, 1.82e-3], [1e-2, 3.1e-3], [50, 0.02]] as [number, number][]) {
      const g = { lnx0: Math.log(x) - 0.01, lnx1: Math.log(x) + 0.01, nx: 1, lns0: Math.log(s), lns1: Math.log(s) + 1, ns: 2 };
      const run = (refine: number) => { const uu = uGrid(refine), rows = uu.u.map((u) => { const r = new Float64Array(1); kernelRow(u, g, r); return r; }); return contract(rows, uu, g); };
      const a = run(1), b = run(10);
      expect(Math.abs(a.J[0] / b.J[0] - 1)).toBeLessThan(1e-2);
      expect(a.A[0]).toBeGreaterThan(0);
      expect(Math.abs(a.A[0] / b.A[0] - 1)).toBeLessThan(3e-2);
    }
    }, 300000);
  it("the shipped table has no holes: no emission cell is zero between non-zero neighbours, no absorption is negative", () => {
    const t = TAB(); let holes = 0, negA = 0;
    for (let j = 0; j < t.ns; j++) for (let i = 1; i < t.nx - 1; i++) {
      const lj = (k: number) => t.data[2 * (j * t.nx + k)];
      if (lj(i) <= -79 && lj(i - 1) > -60 && lj(i + 1) > -60) holes++;
      // zero absorption where electrons emit is allowed only in the cyclotron-maser band: the low edge of the
      // fundamental of electrons at the cooled population's switch-on edge, where the converged absorption is
      // genuinely negative (measured: (x 0.16, s 0.143) -5.4e-6 at 1x and 10x grids) and is stored as zero
      if (t.data[2 * (j * t.nx + i) + 1] <= -79 && lj(i) > -60) {
        const x = Math.exp(t.lnx0 + ((t.lnx1 - t.lnx0) / t.nx) * (i + 0.5)), s = Math.exp(t.lns0 + (t.lns1 - t.lns0) * j / (t.ns - 1));
        if (!(x > 0.1 && x < 2 && s > 0.05 && s < 2)) negA++;
      }
    }
    expect(holes).toBe(0); expect(negA).toBe(0);
  });
  it("above the x grid the extension follows the spectrum's own slope (uncooled at tiny s, cooled otherwise)", () => {
    const t = TAB(), uu = uGrid();
    for (const s of [1e-6, 10]) {
      const x = 1e11, g = { lnx0: Math.log(x) - 0.01, lnx1: Math.log(x) + 0.01, nx: 1, lns0: Math.log(s), lns1: Math.log(s) + 1, ns: 2 };
      const rows = uu.u.map((u) => { const r = new Float64Array(1); kernelRow(u, g, r); return r; }), d = contract(rows, uu, g);
      const [lJ, lA] = lookup(t, Math.log(x), Math.log(s));
      // measured 6.8 % at s = 1e-6, x = 1e11: the cooling break sits just past the grid (was 64 % with a fixed slope)
      expect(Math.abs(Math.exp(lJ) / d.J[0] - 1)).toBeLessThan(0.1);
      expect(Math.abs(Math.exp(lA) / d.A[0] - 1)).toBeLessThan(0.1);
    }
  }, 300000);
  it("the table build refuses non-finite values", () => {
    const g = { ...TABLE_GRID, nx: 2, ns: 2 };
    expect(() => encodeTable(new Float64Array([1, NaN, 1, 1]), new Float64Array(4).fill(1), g)).toThrow();
  });
  it("cold-electron absorption at the fundamental grows linearly with s past the table (so the extension is linear)", () => {
    const g = { lnx0: Math.log(0.995), lnx1: Math.log(1.005), nx: 1, lns0: Math.log(20), lns1: Math.log(40), ns: 3 };
    const uu = uGrid(), rows = uu.u.map((u) => { const r = new Float64Array(1); kernelRow(u, g, r); return r; });
    const { A } = contract(rows, uu, g), s = [20, Math.sqrt(800), 40];
    const k1 = (A[1] - A[0]) / (s[1] - s[0]), k2 = (A[2] - A[1]) / (s[2] - s[1]);
    expect(k1).toBeGreaterThan(0); expect(Math.abs(k2 / k1 - 1)).toBeLessThan(0.05); // measured ~4.2 per unit s
    const t = TAB(), at = (ss: number) => Math.exp(lookup(t, 0, Math.log(ss))[1]);
    expect(at(40 * (1 + 1e-9)) / at(40)).toBeCloseTo(1, 6);                        // continuous at the edge
    const kT = (at(80) - at(40)) / 40; expect(kT).toBeGreaterThan(0);
    expect(Math.abs((at(400) - at(40)) / 360 / kT - 1)).toBeLessThan(1e-6);      // linear beyond
    const far = (ss: number) => Math.exp(lookup(t, Math.log(1e4), Math.log(ss))[1]);
    expect(far(400) / far(40)).toBeLessThan(1.05);                              // away from the line: ~s-independent
  }, 300000);
});
