import { describe, it, expect } from "vitest";
import { turbulenceAt, boxMuller, boxMullerAngle, TURB } from "../src/physics/emission";
import { omegaKepler, iscoRadius } from "../src/physics/orbits";

const TWO_PI = 2 * Math.PI, A = 0.9;
/** Seeded PRNG (mulberry32): every statistic below is deterministic. */
function rng(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) >>> 0; let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const torb = (r: number, a = A) => TWO_PI / omegaKepler(r, a, true);

/** Real DFT power |X_m|^2, m = 0..N/2. */
function power(x: number[]): number[] {
  const N = x.length, P: number[] = [];
  for (let m = 0; m <= N / 2; m++) {
    let re = 0, im = 0;
    for (let n = 0; n < N; n++) { const th = (-TWO_PI * m * n) / N; re += x[n] * Math.cos(th); im += x[n] * Math.sin(th); }
    P.push(re * re + im * im);
  }
  return P;
}
function solve3(A3: number[][], b: number[]): number[] {
  const M = A3.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < 3; c++) {
    let p = c; for (let r = c + 1; r < 3; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < 3; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k < 4; k++) M[r][k] -= f * M[c][k]; }
  }
  return [M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2]];
}
/** Harmonic where the variance per ln k (m P(m)) peaks: log-bin, then a least-squares parabola in ln m
 *  over +-0.6 of the highest bin (SKH06: "the variance peaks at this break wavenumber"). */
function peakHarmonic(P: number[]): number {
  const xs: number[] = [], ys: number[] = [], mMax = P.length / 2;
  for (let lo = 1; lo < mMax; ) {
    const hi = Math.max(lo + 1, Math.round(lo * Math.exp(0.1)));
    let sx = 0, sy = 0, n = 0;
    for (let m = lo; m < hi && m < mMax; m++) { sx += Math.log(m); sy += m * P[m]; n++; }
    xs.push(sx / n); ys.push(Math.log(sy / n)); lo = hi;
  }
  let im = 0; for (let i = 1; i < ys.length; i++) if (ys[i] > ys[im]) im = i;
  const M3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], b = [0, 0, 0];
  for (let i = 0; i < xs.length; i++) {
    const d = xs[i] - xs[im]; if (Math.abs(d) > 0.6) continue;
    const f = [1, d, d * d];
    for (let r = 0; r < 3; r++) { b[r] += f[r] * ys[i]; for (let c = 0; c < 3; c++) M3[r][c] += f[r] * f[c]; }
  }
  const c = solve3(M3, b);
  return Math.exp(xs[im] - c[1] / (2 * c[2]));
}
/** Contour slope dphi/deta = -<g_eta g_phi>/<g_phi^2> of snapshots near time T. */
function tilt(T: number, rand: () => number): number {
  let num = 0, den = 0; const h = 1e-3;
  for (let q = 0; q < 40000; q++) {
    const r = 6 * Math.exp(rand() * 2), eta = Math.log(r), phi = rand() * TWO_PI, t = T + rand() * 500;
    const ge = (turbulenceAt(Math.exp(eta + h), phi, t, A) - turbulenceAt(Math.exp(eta - h), phi, t, A)) / (2 * h);
    const gp = (turbulenceAt(r, phi + h, t, A) - turbulenceAt(r, phi - h, t, A)) / (2 * h);
    num += ge * gp; den += gp * gp;
  }
  return -num / den;
}
function moments(T0: number, T1: number, rand: () => number, a = A, rMin = 3) {
  let s1 = 0, s2 = 0, s4 = 0; const n = 100000;
  for (let q = 0; q < n; q++) {
    const g = turbulenceAt(rMin * Math.exp(rand() * 3), rand() * 20 - 10, T0 + rand() * (T1 - T0), a);
    s1 += g; s2 += g * g; s4 += g ** 4;
  }
  return { mean: s1 / n, v: s2 / n, kurt: s4 / n / (s2 / n) ** 2 };
}

describe("MRI turbulence field (spec 2026-10-03)", () => {
  it("is a unit Gaussian at every point: mean 0, variance 1, kurtosis 3", () => {
    const m = moments(-100, 1e4, rng(1));
    expect(Math.abs(m.mean)).toBeLessThan(0.05);
    expect(Math.abs(m.v - 1)).toBeLessThan(0.05);
    expect(Math.abs(m.kurt - 3)).toBeLessThan(0.15);
  });

  it("is 2 pi-periodic in phi at any time, including late and negative times (the seam above the shadow)", () => {
    for (const [r, p, t] of [[10.46, -3.2085, 0], [6.45, 0.3, 1234.5], [25, 2.9, 3e6], [4, -1.2, -50], [2.4, 5.0, 2e5]]) {
      const v = turbulenceAt(r, p, t, A);
      for (const k of [-2, -1, 1, 3]) expect(turbulenceAt(r, p + TWO_PI * k, t, A)).toBeCloseTo(v, 9);
    }
  });

  it("is continuous in r, phi and t: no jumps at lattice rows, generation ticks or t = 0", () => {
    const rand = rng(2); let worst = 0;
    for (let q = 0; q < 20000; q++) {
      const r = 2.5 * Math.exp(rand() * 3), phi = rand() * TWO_PI;
      const t = q < 2000 ? -200 + rand() * 200 : rand() * 1e4;
      const g = turbulenceAt(r, phi, t, A);
      worst = Math.max(worst,
        Math.abs(turbulenceAt(r * (1 + 1e-6), phi, t, A) - g),
        Math.abs(turbulenceAt(r, phi + 1e-6, t, A) - g),
        Math.abs(turbulenceAt(r, phi, t + 1e-6 * torb(r), A) - g));
    }
    expect(worst).toBeLessThan(1e-3); // a jump would be O(1)
  });

  it("Box-Muller never takes log(0): the extreme hashes give finite values within 6 sigma", () => {
    for (const h1 of [0, 1, 0xffffff, 0xffffffff]) for (const h2 of [0, 0x800000, 0xffffff]) {
      const z = boxMuller(h1, h2);
      expect(Number.isFinite(z)).toBe(true);
      expect(Math.abs(z)).toBeLessThan(6);
    }
  });

  it("Box-Muller's cosine argument stays in [-pi, pi], where WGSL guarantees cos to 2^-11 absolute", () => {
    for (const u2 of [0, 0.25, 0.5, 0.75, 1 - 2 ** -24]) {
      const t = boxMullerAngle(u2);
      expect(t).toBeGreaterThanOrEqual(-Math.PI); expect(t).toBeLessThanOrEqual(Math.PI);
      expect(-Math.cos(t)).toBeCloseTo(Math.cos(2 * Math.PI * u2), 12); // the same angle, shifted by pi
    }
  });
  it("structure sizes: variance spectra peak at lambda_phi = 25 deg and lambda_eta = 0.26 (de-sheared, SKH06)", () => {
    const rand = rng(3), N = 512, span = 6, eta0 = Math.log(6), slope = -0.9 * Math.PI;
    const Pphi = new Array(N / 2 + 1).fill(0), Peta = new Array(N / 2 + 1).fill(0);
    for (let s = 0; s < 300; s++) {
      const t = 1000 + s * 1000, eta = eta0 + rand() * 3;
      power(Array.from({ length: N }, (_, n) => turbulenceAt(Math.exp(eta), (TWO_PI * n) / N, t, A))).forEach((p, m) => (Pphi[m] += p));
      const phi0 = rand() * TWO_PI;
      power(Array.from({ length: N }, (_, n) => {
        const e = eta0 + (span * n) / N, w = 0.5 - 0.5 * Math.cos((TWO_PI * n) / N); // Hann window
        return w * turbulenceAt(Math.exp(e), phi0 + slope * (e - eta0), t, A);
      })).forEach((p, m) => (Peta[m] += p));
    }
    const lphi = 360 / peakHarmonic(Pphi), leta = span / peakHarmonic(Peta);
    expect(lphi).toBeGreaterThan(22.5); expect(lphi).toBeLessThan(27.5);
    expect(leta).toBeGreaterThan(0.234); expect(leta).toBeLessThan(0.286);
  }, 60000);

  it("lifetime: the correlation following the flow falls to 1/e at 0.3 T_orb (SKH06 eq. 36)", () => {
    const rand = rng(4), lags = [0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4];
    const rho = lags.map((L) => {
      let s = 0; const n = 20000;
      for (let q = 0; q < n; q++) {
        const r = 6 + rand() * 20, phi = rand() * TWO_PI, t = rand() * 1e4, dt = L * torb(r);
        s += turbulenceAt(r, phi, t, A) * turbulenceAt(r, phi + omegaKepler(r, A, true) * dt, t + dt, A);
      }
      return s / n;
    });
    const k = rho.findIndex((v) => v < 1 / Math.E);
    expect(k).toBeGreaterThan(0);
    const life = lags[k - 1] + ((rho[k - 1] - 1 / Math.E) / (rho[k - 1] - rho[k])) * (lags[k] - lags[k - 1]);
    expect(life).toBeGreaterThan(0.27); expect(life).toBeLessThan(0.33);
  }, 60000);

  it("trailing spirals emerge from shear over one lifetime: dphi/deta = -0.9 pi within 20 %", () => {
    const s = tilt(1000, rng(5)) / Math.PI;
    expect(s).toBeLessThan(-0.72); expect(s).toBeGreaterThan(-1.08);
  }, 60000);

  it("no runaway winding: after ~1000 orbits the variance and the tilt equal those after one", () => {
    const early = tilt(1000, rng(6)), late = tilt(3e6, rng(6));
    expect(Math.abs(late / early - 1)).toBeLessThan(0.05);
    const m = moments(3e6, 3.01e6, rng(7));
    expect(Math.abs(m.v - 1)).toBeLessThan(0.05);
  }, 60000);

  it("spin extremes: finite and unit variance down to the a = 0.998 ISCO", () => {
    for (const a of [0, 0.998]) {
      const m = moments(-100, 1e4, rng(8), a, iscoRadius(a));
      expect(Number.isFinite(m.v)).toBe(true);
      expect(Math.abs(m.v - 1)).toBeLessThan(0.05);
    }
  });

  it("TURB holds the planned constants", () => {
    expect(TURB).toEqual({ cellEta: 0.086, cellsPhi: 49, clock: 0.32, octave2: 0.5 });
  });
});
