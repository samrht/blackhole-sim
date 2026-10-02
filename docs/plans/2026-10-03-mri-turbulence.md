# MRI Disk Turbulence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the decorative disk value-noise with a turbulence field whose structure size, lifetime,
spiral shape and lognormal statistics are those measured for MRI turbulence (SKH06), with amplitude set
so the disk's intrinsic integrated light flickers at the observed 2 % rms.

**Architecture:** `src/physics/emission.ts` gets a unit-Gaussian lattice field on (ln r, φ) made of
overlapping *generations* (each born at a radius-dependent clock tick, frozen into the Keplerian flow,
cross-faded out over two clock cells), and a lognormal emission factor exp(σg − σ²/2). A CPU calibration
module measures the integrated flicker rms against σ; the table goes into `emission.ts` and the panel slider
(flicker %) inverts it to σ. `emission-shared.wgsl` holds the sole WGSL twin; `raytrace.wgsl`'s disk shading
calls it with (r, φ_hit, t_emit, a). The uniform `turbAmp` carries σ (no layout change).

**Tech Stack:** TypeScript, WebGPU/WGSL, Vite, Vitest, vite-node (calibration script), Playwright (GPU gates).

**Spec:** `docs/specs/2026-10-03-mri-turbulence-design.md` (read its "Corrections from prototyping").

## Global Constraints

- Targets (SKH06): variance-spectrum peak wavelengths λ_η = 0.26 and λ_φ = 25° in coordinates de-sheared by
  φ′ = φ + 0.9π η; flow-following correlation 1/e time 0.3 T_orb(r); emergent spiral tilt dφ/dη ≈ −0.9π
  (trailing); lognormal emission with mean exactly 1.
- Constants (measured while planning, `tests/_turbproto.ts`, deleted): `TURB = { cellEta: 0.086, cellsPhi: 49,
  clock: 0.32, octave2: 0.5 }`. Prototype results with these: λ_φ 25.0–25.9°, λ_η 0.246–0.261 (at 0.084;
  0.086 centres it), ρ(0.30 T_orb) = 0.369, tilt −0.948π (t ≈ 1e3) / −0.934π (t ≈ 1e6), mean 0.015,
  variance 1.004, kurtosis 3.017, κ = 0.0346 (linear flicker per unit σ, a = 0.9, r_in = ISCO to 40); σ ≈ 0.553
  gives the 2 % default (exp(σg) is visibly nonlinear there, hence a table, not σ = flicker/κ).
- T_orb = 2π/Ω_K(r, a) in coordinate time (Ω_K = 1/(r^1.5 + a), M = 1, prograde).
- Hash: u32 arithmetic only (`Math.imul`/`>>> 0` on the CPU, wrapping u32 in WGSL) so CPU and GPU pick the same
  lattice values; Box–Muller u1 = ((h & 0xffffff) + 0.5)/2²⁴ (never 0).
- Panel: "Disk flicker (rms)" 0–10 % step 0.5 %, default 2 %; "Flares (illustrative)" default 0; breathing has
  no control and stays 0 in the app.
- The jet's `vnoise`/`vnoiseE`/`ihash`/`ihashE` are unchanged.
- Gates green after every task: `npm test`, `npm run build`. GPU gates from Task 3 on: dev server on :5173
  (`npx vite --port 5173 --strictPort`), `npm run verify:gpu` (any console warning fails),
  `node scripts/probe-axis.mjs`, `node scripts/probe-scale.mjs`.
- `?golden` re-recorded once, in Task 4, after the σ = 0 A/B check against `main`.
- No Co-Authored-By trailer on commits in this repo.

## Review Focus

1. **Long sessions** (t_emit ≫ 1e5 M after an hour at Motion 5; f32 τ = t/T_c and generation index): the field
   must stay smooth, unit-variance and equally wound. Pinned by Task 1's late-time test (t ≈ 3e6) and Task 3's
   parity case at t = 2e5.
2. **Negative emission times** (light delay near t = 0 makes t_emit < 0; negative generation indices go through
   the u32 bit pattern): finite, continuous across t = 0. Pinned by Task 1's continuity test (t ∈ [−200, 0])
   and Task 3's parity case t = −50.
3. **Spin extremes** (a = 0, a = 0.998 with ISCO 1.24 M): finite, unit variance near the ISCO. Pinned by
   Task 1's spin-extremes test and Task 3's parity cases at a = 0 and 0.998.
4. **Flicker slider at its maximum** (10 % → σ ≈ 1.7, exp(σg) up to ~e¹⁰): no overflow, no NaN, image not
   blown out to white. Pinned by Task 2's table-range test and a Task 5 screenshot at 10 %.
5. **The 2π seam above the shadow** (rays reaching one disk point with φ_hit 2π apart, now with advection
   inside the field): must agree. Pinned by Task 1's periodicity test (including large t and negative t).

---

## File Structure

- Modify `src/physics/emission.ts`: add the Gaussian lattice field (`mixHash`, `hash4`, `boxMuller`,
  `turbulenceAt`, `TURB`), `lognormalFactor`, the flicker table + `sigmaForFlicker` + `FLICKER_DEFAULT`; change
  `emissionField` to `(r, phi, t, a, sigma, breatheAmp, spots)`; remove `turbulence`/`vnoiseRing` (Task 3).
- Create `tests/turbulence.test.ts`: the field's measured properties.
- Modify `tests/emission.test.ts`: emission-factor tests on the new signature.
- Create `src/physics/turbulence-calibration.ts` (`diskFlickerRms`) and `scripts/calibrate-turbulence.ts`
  (prints the table). Not imported by the app.
- Create `tests/turbulence-calibration.test.ts`.
- Modify `src/render/emission-shared.wgsl` (WGSL twin; remove `turbulenceE`/`vnoiseRingE`),
  `src/render/turb-parity.wgsl`, `src/test/parity.browser.ts`.
- Modify `src/render/raytrace.wgsl` (`emissionFieldE`, `shadeDisk`), `src/render/uniforms.ts` (comment),
  `src/test/scenes.ts` (σ default), `src/test/golden.json` (re-record).
- Modify `index.html`, `src/main.ts`, `scripts/verify-gpu.mjs` (panel and app check).
- Modify `README.md`, `docs/ROADMAP.md`.

---

### Task 1: CPU turbulence field

**Files:**
- Modify: `src/physics/emission.ts`
- Create: `tests/turbulence.test.ts`

**Interfaces:**
- Consumes: `omegaKepler(r, a, prograde)` from `src/physics/orbits.ts`, `iscoRadius(a)`.
- Produces: `TURB`, `mixHash(n: number): number`, `hash4(ix, iy, gen, salt): number` (u32),
  `boxMuller(h1: number, h2: number): number`, `turbulenceAt(r: number, phi: number, t: number, a: number): number`
  (unit Gaussian g), `lognormalFactor(g: number, sigma: number): number`.

- [ ] **Step 1: Write the failing tests** — `tests/turbulence.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { turbulenceAt, boxMuller, TURB } from "../src/physics/emission";
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run tests/turbulence.test.ts`
Expected: FAIL — `turbulenceAt` / `boxMuller` / `TURB` are not exported.

- [ ] **Step 3: Implement** — in `src/physics/emission.ts`, after `vnoise` (keep `ihash`, `smooth`, `vnoise`,
`vnoiseRing`, `turbulence` for now; Task 3 removes the last two), add:

```ts
// --- MRI turbulence (spec 2026-10-03) -------------------------------------------------------------
// A unit-Gaussian field on (eta = ln r, phi) whose structures have the size, lifetime, shape and statistics
// measured for magnetorotational turbulence in thin disks (Schnittman, Krolik & Hawley 2006, "SKH06").
// Lattice: hashed standard normals, smoothstep-interpolated and renormalised by sqrt(sum w^2), so every
// point is exactly N(0, 1). Generations: each lattice row i runs a clock tau = t / T_c(r_i) with
// T_c = clock * T_orb; generation k is born at tick k - 1, frozen into the Keplerian flow from birth
// (sampled at phi - Omega(r) * age), and cross-faded out over the next tick (cos/sin weights, unit
// variance). Shear acting on each generation for its age makes trailing spirals; no generation lives more
// than 2 T_c, so the winding never runs away. Twin: turbulenceFieldE in emission-shared.wgsl.
export const TURB = {
  cellEta: 0.086, // ln r cell (octave 1): variance spectrum peaks at lambda_eta = 0.26 (SKH06 dr/r = 0.3)
  cellsPhi: 49,   // cells around the ring (octave 1): peaks at lambda_phi = 25 deg (SKH06)
  clock: 0.32,    // T_c / T_orb: the flow-following correlation falls to 1/e at 0.30 T_orb (SKH06 eq. 36)
  octave2: 0.5,   // weight of the half-size octave (small-scale tail of the spectrum)
} as const;

/** u32 avalanche (twin: mixT). */
export function mixHash(n: number): number {
  n = Math.imul(n ^ (n >>> 15), 2246822519) >>> 0;
  n = Math.imul(n ^ (n >>> 13), 3266489917) >>> 0;
  return (n ^ (n >>> 16)) >>> 0;
}
/** u32 hash of a lattice node (row ix, column iy, generation, octave salt). Negative ints hash through
 *  their u32 bit pattern, as WGSL's u32(i32) does. */
export function hash4(ix: number, iy: number, gen: number, salt: number): number {
  return mixHash((Math.imul(ix >>> 0, 1973) + Math.imul(iy >>> 0, 9277) + Math.imul(gen >>> 0, 26699) + Math.imul(salt >>> 0, 59359)) >>> 0);
}
/** Standard normal from two u32 hashes (Box-Muller); u1 is offset by half a step so it is never 0. */
export function boxMuller(h1: number, h2: number): number {
  const u1 = ((h1 & 0xffffff) + 0.5) / 16777216, u2 = (h2 & 0xffffff) / 16777216;
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(TWO_PI * u2);
}
function gaussNode(ix: number, iy: number, gen: number, salt: number): number {
  const h1 = hash4(ix, iy, gen, salt);
  return boxMuller(h1, mixHash((h1 ^ 0x9e3779b9) >>> 0));
}
/** One octave: rows ix of height cellEta in ln r, cellsPhi columns around the ring. */
function turbOctave(r: number, phi: number, t: number, a: number, cellEta: number, cellsPhi: number, salt: number): number {
  const x = Math.log(r) / cellEta, i0 = Math.floor(x), fx = smooth(x - i0);
  const Om = omegaKepler(r, a, true);
  let num = 0, v = 0;
  for (let d = 0; d < 2; d++) {
    const i = i0 + d, wr = d ? fx : 1 - fx;
    const Tc = (TURB.clock * TWO_PI) / omegaKepler(Math.exp(i * cellEta), a, true);
    const tau = t / Tc + (hash4(i, 0x51ed, 0, salt) & 0xffffff) / 16777216; // per-row clock phase
    const k = Math.floor(tau), f = tau - k;
    for (let e = 0; e < 2; e++) {
      const gen = k + e, wt = e ? Math.sin((Math.PI / 2) * f) : Math.cos((Math.PI / 2) * f);
      const age = (e ? f : 1 + f) * Tc;               // generation k + e was born at tick k + e - 1
      const ph = phi - Om * age, y = (ph / TWO_PI - Math.floor(ph / TWO_PI)) * cellsPhi;
      const j0 = Math.floor(y), fy = smooth(y - j0), ja = j0 % cellsPhi, jb = (j0 + 1) % cellsPhi;
      const w0 = wr * wt * (1 - fy), w1 = wr * wt * fy;
      num += w0 * gaussNode(i, ja, gen, salt) + w1 * gaussNode(i, jb, gen, salt);
      v += w0 * w0 + w1 * w1;
    }
  }
  return num / Math.sqrt(v);
}
/** Unit-Gaussian MRI turbulence g at disk radius r, azimuth phi (any branch: 2 pi-periodic), coordinate
 *  time t (the emission time; negative is fine), spin a. */
export function turbulenceAt(r: number, phi: number, t: number, a: number): number {
  const w = TURB.octave2;
  return (turbOctave(r, phi, t, a, TURB.cellEta, TURB.cellsPhi, 1) + w * turbOctave(r, phi, t, a, TURB.cellEta / 2, 2 * TURB.cellsPhi, 2)) / Math.sqrt(1 + w * w);
}
/** Lognormal brightness factor exp(sigma g - sigma^2 / 2): mean exactly 1 over a unit Gaussian g, so the
 *  turbulence redistributes the Novikov-Thorne light without changing its average (Hogg & Reynolds 2016). */
export function lognormalFactor(g: number, sigma: number): number {
  return Math.exp(sigma * g - 0.5 * sigma * sigma);
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run tests/turbulence.test.ts`
Expected: PASS, 10 tests (sizes/lifetime/tilt tests take a few seconds each). If a measured property misses its
band, the constant is wrong, not the band: re-measure with the test's own helper and ledger a Ruling with the
new value (the bands are the spec's).

- [ ] **Step 5: Commit**

```bash
git add src/physics/emission.ts tests/turbulence.test.ts
git commit -m "Turbulence: unit-Gaussian MRI field with SKH06 sizes, lifetime and emergent trailing spirals (CPU)"
```

---

### Task 2: Flicker calibration and the lognormal emission field

**Files:**
- Create: `src/physics/turbulence-calibration.ts`, `scripts/calibrate-turbulence.ts`,
  `tests/turbulence-calibration.test.ts`
- Modify: `src/physics/emission.ts`, `tests/emission.test.ts`

**Interfaces:**
- Consumes: `turbulenceAt`, `lognormalFactor` (Task 1); `pageThorneFluxShape(r, a)` (`disk.ts`); `iscoRadius`.
- Produces: `diskFlickerRms(sigmas: number[], grid: FlickerGrid): number[]`, `FULL_GRID`, `FLICKER_TABLE:
  readonly (readonly [number, number])[]` ((σ, rms) pairs, ascending), `FLICKER_DEFAULT = 0.02`,
  `sigmaForFlicker(rms: number): number`, and the new
  `emissionField(r, phi, t, a, sigma, breatheAmp, spots): number`.

- [ ] **Step 1: Write the failing tests** — `tests/turbulence-calibration.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { diskFlickerRms, FULL_GRID } from "../src/physics/turbulence-calibration";
import { FLICKER_TABLE, FLICKER_DEFAULT, sigmaForFlicker } from "../src/physics/emission";

describe("flicker calibration (spec 2026-10-03 §2.2)", () => {
  it("the table starts at (0, 0), is strictly increasing and reaches the slider's 10 %", () => {
    expect(FLICKER_TABLE[0]).toEqual([0, 0]);
    for (let i = 1; i < FLICKER_TABLE.length; i++) {
      expect(FLICKER_TABLE[i][0]).toBeGreaterThan(FLICKER_TABLE[i - 1][0]);
      expect(FLICKER_TABLE[i][1]).toBeGreaterThan(FLICKER_TABLE[i - 1][1]);
    }
    expect(FLICKER_TABLE[FLICKER_TABLE.length - 1][1]).toBeGreaterThan(0.1);
  });
  it("sigmaForFlicker inverts the table: 0 -> 0, linear in between, finite at 10 %", () => {
    expect(sigmaForFlicker(0)).toBe(0);
    const [s1, r1] = FLICKER_TABLE[1], [s2, r2] = FLICKER_TABLE[2];
    expect(sigmaForFlicker((r1 + r2) / 2)).toBeCloseTo((s1 + s2) / 2, 12);
    const top = sigmaForFlicker(0.1);
    expect(Number.isFinite(top)).toBe(true);
    expect(top).toBeLessThan(3);
    expect(sigmaForFlicker(1)).toBe(FLICKER_TABLE[FLICKER_TABLE.length - 1][0]); // clamps
  });
  it("small-sigma slope kappa matches the prototype (0.0347 +- 10 %)", () => {
    const [s, r] = FLICKER_TABLE[1];
    expect(r / s).toBeGreaterThan(0.031); expect(r / s).toBeLessThan(0.038);
  });
  it("the default slider gives 2.0 % +- 0.2 % intrinsic rms, re-measured on an independent coarse grid", () => {
    const [rms] = diskFlickerRms([sigmaForFlicker(FLICKER_DEFAULT)], { ...FULL_GRID, ne: 80, np: 128, ns: 300 });
    expect(rms).toBeGreaterThan(0.018); expect(rms).toBeLessThan(0.022);
  }, 120000);
});
```

Add to `tests/emission.test.ts` (and change the import line to
`import { patternPhase, turbulence, hotspotField, emissionField, lognormalFactor, T_BREATHE, type HotSpot } from "../src/physics/emission";`),
replacing the three `emissionField` tests:

```ts
  it("emissionField reduces to exactly 1 when all features are off (Tier-1 regression gate)", () => {
    expect(emissionField(9, 2.0, 123.4, 0.9, 0, 0, [])).toBe(1);
    expect(emissionField(15, -1.0, -5.0, 0.0, 0, 0, [])).toBe(1);
  });
  it("lognormal factor has mean exactly 1 over a unit Gaussian (quadrature), so the disk's light is conserved", () => {
    for (const s of [0.1, 0.55, 1.7]) {
      let I = 0; const h = 1e-3;
      for (let x = -14; x <= 14 + 1e-12; x += h) I += Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI) * lognormalFactor(x, s) * h;
      expect(Math.abs(I - 1)).toBeLessThan(1e-3);
    }
  });
  it("emissionField = lognormal turbulence x breathing + hot spots at the co-rotating phase", () => {
    const spots: HotSpot[] = [{ r: 10, psi: 1.0, sigma: 1.0, amp: 2.0 }];
    const t = 37, phi = 1.0 + omegaKepler(10, 0.9, true) * t; // the spot's centre at time t
    expect(emissionField(10, phi, t, 0.9, 0, 0, spots)).toBeCloseTo(3.0, 9);
  });
  it("emissionField is non-negative and finite at the slider's top sigma with strong features", () => {
    const spots: HotSpot[] = [{ r: 8, psi: 0, sigma: 1.5, amp: 3 }];
    for (let p = 0; p < 6.28; p += 0.5) {
      const e = emissionField(8, p, 10, 0.9, 1.8, 0.9, spots);
      expect(Number.isFinite(e)).toBe(true); expect(e).toBeGreaterThanOrEqual(0);
    }
  });
```

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run tests/turbulence-calibration.test.ts tests/emission.test.ts`
Expected: FAIL — module `turbulence-calibration` missing; `lognormalFactor`/new `emissionField` signature.

- [ ] **Step 3: Implement the calibration module** — `src/physics/turbulence-calibration.ts`:

```ts
// Calibration of the turbulence amplitude (spec 2026-10-03 §2.2): the intrinsic (face-on, before
// relativistic beaming) fractional rms of the disk's integrated light, for each lognormal sigma.
// Not imported by the app; scripts/calibrate-turbulence.ts prints FLICKER_TABLE from it.
import { turbulenceAt, lognormalFactor } from "./emission";
import { iscoRadius } from "./orbits";
import { pageThorneFluxShape } from "./disk";

export interface FlickerGrid { ne: number; np: number; ns: number; dt: number; a: number; rOut: number; }
/** ne log-spaced radii from the ISCO to rOut, np azimuths, ns snapshots dt apart (M). */
export const FULL_GRID: FlickerGrid = { ne: 160, np: 256, ns: 1500, dt: 397, a: 0.9, rOut: 40 };

export function diskFlickerRms(sigmas: number[], g: FlickerGrid): number[] {
  const ri = iscoRadius(g.a), l0 = Math.log(ri), l1 = Math.log(g.rOut);
  const rs: number[] = [], w: number[] = [];
  let tot = 0;
  for (let i = 0; i < g.ne; i++) {
    const r = Math.exp(l0 + ((l1 - l0) * (i + 0.5)) / g.ne);
    const wi = pageThorneFluxShape(r, g.a) * r * r; // F dA = F r dr dphi = F r^2 d(ln r) dphi
    rs.push(r); w.push(wi); tot += wi * g.np;
  }
  const L = sigmas.map(() => [] as number[]);
  for (let s = 0; s < g.ns; s++) {
    const t = 1000 + s * g.dt, sums = sigmas.map(() => 0);
    for (let i = 0; i < g.ne; i++) for (let j = 0; j < g.np; j++) {
      const gv = turbulenceAt(rs[i], (2 * Math.PI * j) / g.np, t, g.a);
      for (let q = 0; q < sigmas.length; q++) sums[q] += w[i] * lognormalFactor(gv, sigmas[q]);
    }
    sums.forEach((v, q) => L[q].push(v / tot));
  }
  return L.map((xs) => {
    const m = xs.reduce((p, c) => p + c, 0) / xs.length;
    return Math.sqrt(xs.reduce((p, c) => p + (c - m) ** 2, 0) / xs.length) / m;
  });
}
```

`scripts/calibrate-turbulence.ts`:

```ts
// Prints FLICKER_TABLE for src/physics/emission.ts. Run: npx vite-node scripts/calibrate-turbulence.ts
// (~10 min, single thread). Re-run whenever TURB changes.
import { diskFlickerRms, FULL_GRID } from "../src/physics/turbulence-calibration";
const S = [0.05, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];
const rms = diskFlickerRms(S, FULL_GRID);
console.log(`export const FLICKER_TABLE: readonly (readonly [number, number])[] = [[0, 0], ${S.map((s, i) => `[${s}, ${rms[i].toFixed(5)}]`).join(", ")}];`);
```

- [ ] **Step 4: Run the calibration** (background, 2 h limit; it prints one line)

Run: `npx vite-node scripts/calibrate-turbulence.ts > $CLAUDE_JOB_DIR/tmp/flicker-table.txt`
Expected (the planning prototype on this same grid, 1500 snapshots): 0.0017 at σ 0.05 (κ = 0.0346), 0.0087 at
0.25, 0.0179 at 0.5, 0.0279 at 0.75, 0.0394 at 1.0, 0.0535 at 1.25, 0.0720 at 1.5, 0.1389 at 2.0 — so σ ≈ 0.553
at the 2 % default and σ ≈ 1.7 at 10 %. Agreement to ~2 % is expected (same field, same grid); a larger
difference means the field differs from the prototype: stop and find out why.

- [ ] **Step 5: Implement the table, inversion and new emissionField** — in `emission.ts`, paste the printed
line (with this doc comment) and add below `lognormalFactor`:

```ts
/** Intrinsic flicker of the disk's integrated light (fractional rms, face-on, before beaming) against the
 *  lognormal sigma, measured by scripts/calibrate-turbulence.ts at a = 0.9, r_in = ISCO .. 40 M, 1500
 *  snapshots. Observed thermal-state disks flicker at ~2 % (SKH06 <~ 2 % above 10 Hz; soft states a few %). */
export const FLICKER_TABLE: readonly (readonly [number, number])[] = /* pasted from the script */;
export const FLICKER_DEFAULT = 0.02;
/** sigma giving intrinsic flicker `rms` (linear between table points; clamps to the table's top). */
export function sigmaForFlicker(rms: number): number {
  if (rms <= 0) return 0;
  for (let i = 1; i < FLICKER_TABLE.length; i++) {
    const [s0, r0] = FLICKER_TABLE[i - 1], [s1, r1] = FLICKER_TABLE[i];
    if (rms <= r1) return s0 + ((rms - r0) / (r1 - r0)) * (s1 - s0);
  }
  return FLICKER_TABLE[FLICKER_TABLE.length - 1][0];
}
```

Replace `emissionField`:

```ts
/** Dimensionless emission multiplier at disk point (r, phi) and emission time t: lognormal MRI turbulence
 *  (sigma from sigmaForFlicker) x the optional breathing + the optional hot spots, which orbit at the
 *  co-rotating phase psi = phi - Omega t. Exactly 1 when sigma = breatheAmp = 0 and there are no spots. */
export function emissionField(
  r: number, phi: number, t: number, a: number,
  sigma: number, breatheAmp: number, spots: HotSpot[],
): number {
  const turb = sigma > 0 ? lognormalFactor(turbulenceAt(r, phi, t, a), sigma) : 1;
  const breathe = 1 + breatheAmp * Math.sin(TWO_PI * t / T_BREATHE);
  return Math.max(0, turb * breathe + hotspotField(r, patternPhase(r, phi, t, 1, a), spots));
}
```

- [ ] **Step 6: Run to verify they pass**

Run: `npx vitest run tests/turbulence-calibration.test.ts tests/emission.test.ts tests/turbulence.test.ts`
Expected: PASS (the coarse-grid check takes ~20 s; it lands near 0.0205 because the coarse grid reads ~4 %
high, inside the ±10 % band).

- [ ] **Step 7: Commit**

```bash
git add src/physics/turbulence-calibration.ts scripts/calibrate-turbulence.ts tests/turbulence-calibration.test.ts src/physics/emission.ts tests/emission.test.ts
git commit -m "Turbulence: lognormal emission and the 2 % flicker calibration table"
```

---

### Task 3: WGSL twin and parity

**Files:**
- Modify: `src/render/emission-shared.wgsl`, `src/render/turb-parity.wgsl`, `src/test/parity.browser.ts`,
  `src/physics/emission.ts`, `tests/emission.test.ts`

**Interfaces:**
- Consumes: `turbulenceAt` (Task 1); `omegaKep(r, a)` from `integrator-shared.wgsl` (prepended before
  `emission-shared.wgsl` by `gpu.ts` and the parity route).
- Produces: WGSL `turbulenceFieldE(r: f32, phi: f32, t: f32, a: f32) -> f32`; removes `turbulenceE`,
  `vnoiseRingE` (WGSL) and `turbulence`, `vnoiseRing` (CPU).

- [ ] **Step 1: Write the failing parity cases** — in `src/test/parity.browser.ts`, change the import to
`import { turbulenceAt } from "../physics/emission";` and replace the turbulence-parity block's cases, buffer
sizes and comparison:

```ts
  // --- turbulence parity (CPU emission.ts turbulenceAt vs the shipped turbulenceFieldE) ---
  // Late (2e5), negative (-50) and both spin extremes are Review Focus 1-3.
  const tcases = [
    { r: 6, phi: 0.4, t: 1.7, a: 0.9 }, { r: 9, phi: 1.7, t: 500, a: 0.9 },
    { r: 14, phi: -3.9, t: 5000, a: 0.9 }, { r: 22, phi: 5.2, t: 2e5, a: 0.9 },
    { r: 4, phi: 2.2, t: -50, a: 0.9 }, { r: 7, phi: 0.9, t: 333, a: 0 },
    { r: 1.3, phi: 4.4, t: 81, a: 0.998 }, { r: 35, phi: -0.6, t: 12345, a: 0.5 },
  ];
  const tin = device.createBuffer({ size: tcases.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const tarr = new Float32Array(tcases.length * 4);
  tcases.forEach((c, i) => { tarr.set([c.r, c.phi, c.t, c.a], i * 4); });
```

(keep the `tout`/`tread`/pipeline lines as they are), and the comparison:

```ts
  let turbErr = 0;
  tcases.forEach((c, i) => {
    const cpu = turbulenceAt(Math.fround(c.r), Math.fround(c.phi), Math.fround(c.t), Math.fround(c.a));
    turbErr = Math.max(turbErr, Math.abs(tgpu[i] - cpu));
  });
  maxErr = Math.max(maxErr, turbErr / 1e3); // reported separately below; gated at 2e-3 absolute
```

and next to where the route reports its verdict add `turbErr` to the printed line and fail the route when
`turbErr > 2e-3` (follow the existing `jetLogErr` pattern in the same file: it is reported as
`turb |d g| <value>` and ANDed into PASS).

`src/render/turb-parity.wgsl`:

```wgsl
// Parity harness for the disk turbulence: turbulenceFieldE comes from emission-shared.wgsl (the sole copy,
// also prepended to raytrace.wgsl), so ?parity checks the renderer's own bytes against turbulenceAt() in
// src/physics/emission.ts. Inputs are (r, phi, t_emit, a).
@group(0) @binding(0) var<storage, read> inp: array<vec4<f32>>;   // (r, phi, t, a)
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = arrayLength(&inp);
  if (gid.x >= n) { return; }
  let c = inp[gid.x];
  outp[gid.x] = turbulenceFieldE(c.x, c.y, c.z, c.w);
}
```

- [ ] **Step 2: Run to verify it fails**

Start the dev server (`npx vite --port 5173 --strictPort`, background), then run `npm run verify:gpu`.
Expected: FAIL — the WGSL module does not compile (`turbulenceFieldE` undefined) and the route reports it.

- [ ] **Step 3: Implement the WGSL twin** — in `emission-shared.wgsl`, replace everything from
`// Value noise periodic in y` through the end of `turbulenceE` with (keep `ihashE`, `smoothE`, `vnoiseE`,
`TWO_PI_E`; update the header comment's "disk turbulence" twin reference to `turbulenceAt`):

```wgsl
// --- MRI turbulence (spec 2026-10-03; twin: turbulenceAt in src/physics/emission.ts) ------------------
// Unit-Gaussian lattice field on (ln r, phi) in overlapping generations frozen into the Keplerian flow;
// see the CPU twin for the construction. u32 arithmetic wraps exactly as Math.imul/>>>0 on the CPU.
const TURB_CELL_ETA = 0.086; const TURB_CELLS_PHI = 49u; const TURB_CLOCK = 0.32; const TURB_OCT2 = 0.5;
fn mixT(n0: u32) -> u32 {
  var n = (n0 ^ (n0 >> 15u)) * 2246822519u;
  n = (n ^ (n >> 13u)) * 3266489917u;
  return n ^ (n >> 16u);
}
fn hash4T(ix: i32, iy: u32, gen: i32, salt: u32) -> u32 {
  return mixT(u32(ix) * 1973u + iy * 9277u + u32(gen) * 26699u + salt * 59359u);
}
fn gaussT(ix: i32, iy: u32, gen: i32, salt: u32) -> f32 {
  let h1 = hash4T(ix, iy, gen, salt); let h2 = mixT(h1 ^ 0x9e3779b9u);
  let u1 = (f32(h1 & 0xffffffu) + 0.5) / 16777216.0;  // f32 may round the top value to 1: log -> 0, finite
  let u2 = f32(h2 & 0xffffffu) / 16777216.0;
  return sqrt(max(0.0, -2.0 * log(u1))) * cos(TWO_PI_E * u2);
}
fn turbOctaveT(r: f32, phi: f32, t: f32, a: f32, cellEta: f32, cellsPhi: u32, salt: u32) -> f32 {
  let x = log(r) / cellEta; let i0 = i32(floor(x)); let fx = smoothE(x - floor(x));
  let Om = omegaKep(r, a);
  var num = 0.0; var v = 0.0;
  for (var d = 0; d < 2; d++) {
    let i = i0 + d; let wr = select(1.0 - fx, fx, d == 1);
    let Tc = TURB_CLOCK * TWO_PI_E / omegaKep(exp(f32(i) * cellEta), a);
    let tau = t / Tc + f32(hash4T(i, 0x51edu, 0, salt) & 0xffffffu) / 16777216.0;
    let k = i32(floor(tau)); let f = tau - floor(tau);
    for (var e = 0; e < 2; e++) {
      let wt = select(cos(0.25 * TWO_PI_E * f), sin(0.25 * TWO_PI_E * f), e == 1);
      let age = select(1.0 + f, f, e == 1) * Tc;
      let y = fract((phi - Om * age) / TWO_PI_E) * f32(cellsPhi);
      let jf = floor(y); let fy = smoothE(y - jf);
      let ja = u32(jf) % cellsPhi; let jb = (ja + 1u) % cellsPhi;
      let w0 = wr * wt * (1.0 - fy); let w1 = wr * wt * fy;
      num += w0 * gaussT(i, ja, k + e, salt) + w1 * gaussT(i, jb, k + e, salt);
      v += w0 * w0 + w1 * w1;
    }
  }
  return num / sqrt(v);
}
// Unit-Gaussian turbulence g at (r, phi, t_emit, a). 2 pi-periodic in phi.
fn turbulenceFieldE(r: f32, phi: f32, t: f32, a: f32) -> f32 {
  return (turbOctaveT(r, phi, t, a, TURB_CELL_ETA, TURB_CELLS_PHI, 1u)
        + TURB_OCT2 * turbOctaveT(r, phi, t, a, 0.5 * TURB_CELL_ETA, 2u * TURB_CELLS_PHI, 2u)) / sqrt(1.0 + TURB_OCT2 * TURB_OCT2);
}
```

The CPU's `hash4(i, 0x51ed, 0, salt)` hashes column 0x51ed as `iy >>> 0`; the WGSL passes `0x51edu` — the
same u32. The CPU's `ja = j0 % cellsPhi` with `j0 ≥ 0` equals WGSL `u32(jf) % cellsPhi`.

Then remove the now-dead CPU `vnoiseRing` and `turbulence` from `emission.ts`, and in `tests/emission.test.ts`
delete the three `turbulence …` tests and `turbulence` from the import (Task 1's periodicity/moments tests
replace them). In `raytrace.wgsl`, `emissionFieldE` still calls `turbulenceE`: change its first line to
`let turb = 1.0;` for this task only (Task 4 wires the new field) so the renderer compiles.

- [ ] **Step 4: Run to verify it passes**

Run: `npm test` and `npm run build`; then `npm run verify:gpu` with the dev server up.
Expected: unit suite and build green; `?parity` PASS with `turb |d g|` ≤ 2e-3 (f32 vs f64: expect ~1e-5–1e-4;
the t = 2e5 case is the largest, from f32 τ); `?golden` FAILs (expected: turbulence is temporarily off — do not
re-record here); `tests/shader-twins.test.ts` green (no WGSL function defined twice).

- [ ] **Step 5: Commit**

```bash
git add src/render/emission-shared.wgsl src/render/turb-parity.wgsl src/test/parity.browser.ts src/physics/emission.ts tests/emission.test.ts src/render/raytrace.wgsl
git commit -m "Turbulence: WGSL twin in emission-shared, ?parity on (r, phi, t, a) incl. late, negative-time and spin-extreme cases"
```

---

### Task 4: Renderer, scenes and golden

**Files:**
- Modify: `src/render/raytrace.wgsl`, `src/render/uniforms.ts` (comment), `src/test/scenes.ts`, `src/main.ts`
  (uniform only), `src/test/golden.json`

**Interfaces:**
- Consumes: `turbulenceFieldE` (Task 3), `sigmaForFlicker`, `FLICKER_DEFAULT` (Task 2).
- Produces: uniform `turbAmp` = σ everywhere; `state.flicker` (fraction) in `main.ts`.

- [ ] **Step 1: A/B baseline at σ = 0 (the A/B rule).** Create a `main` worktree with its own install and serve
it on :5174:

```bash
git worktree add ../bh-main main
cd ../bh-main && npm ci && (npx vite --port 5174 --strictPort > $CLAUDE_JOB_DIR/tmp/vite5174.log 2>&1 &)
```

Write `$CLAUDE_JOB_DIR/tmp/ab-turb.mjs` (copy of the cooled-jet A/B tool with a σ = 0 override):

```js
// A/B: LIVE accum of every golden scene with turbAmp 0 from the server at argv[3] into argv[2].
import { createRequire } from "node:module";
import { writeFileSync } from "node:fs";
const require = createRequire("C:/Users/shoke/Documents/Claude/blackhole-sim/package.json");
const { chromium } = require("playwright-core");
const out = process.argv[2], base = process.argv[3];
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1000, height: 700 } });
await page.goto(base + "/?bench", { waitUntil: "load" });
const res = await page.evaluate(async () => {
  const { Renderer } = await import("/src/render/gpu.ts");
  const { SCENES, prepareScene, sceneUniforms } = await import("/src/test/scenes.ts");
  document.body.innerHTML = ""; const c = document.createElement("canvas"); c.style.cssText = "width:320px;height:180px;display:block"; document.body.appendChild(c);
  const r = new Renderer(); await r.init(c); const o = {};
  for (const s of SCENES) { const rIn = prepareScene(r, s); r.frame(sceneUniforms(r, s, rIn, { turbAmp: 0 })); await r.device.queue.onSubmittedWorkDone(); o[s.name] = Array.from(await r.readbackAccum()); }
  return o;
});
writeFileSync(out, JSON.stringify(res)); console.log("wrote", out); await browser.close();
```

Run: `node $CLAUDE_JOB_DIR/tmp/ab-turb.mjs $CLAUDE_JOB_DIR/tmp/ab-turb-main.json http://localhost:5174`
Expected: `wrote …`.

- [ ] **Step 2: Wire the renderer** — in `raytrace.wgsl` replace `emissionFieldE` and the two lines of
`shadeDisk` that use it:

```wgsl
// Emission multiplier (twin: emissionField in src/physics/emission.ts): lognormal MRI turbulence with
// sigma = U.turbAmp (mean exactly 1, so the Novikov-Thorne light is redistributed, not changed), times the
// optional breathing, plus the optional hot spots at the co-rotating phase psi. Exactly 1 with all off.
fn emissionFieldE(rHit: f32, phiHit: f32, psi: f32, tEmit: f32, a: f32) -> f32 {
  let s = U.turbAmp;
  var turb = 1.0;
  if (s > 0.0) { turb = exp(s * turbulenceFieldE(rHit, phiHit, tEmit, a) - 0.5 * s * s); }
  let breathe = 1.0 + U.breatheAmp * sin(2.0 * PI * tEmit / 2000.0);
  return max(0.0, turb * breathe + hotspotFieldE(rHit, psi));
}
```

and in `shadeDisk`: `let E = emissionFieldE(rHit, phiHit, psi, tEmit, a);`. In `uniforms.ts` change the header
comment `turbAmp` → `turbAmp (lognormal sigma of the MRI turbulence)`. In `src/test/scenes.ts` import
`{ sigmaForFlicker, FLICKER_DEFAULT }` from `../physics/emission` and set
`turbAmp: sigmaForFlicker(FLICKER_DEFAULT)` (scenes keep `breatheAmp: 0.2, nSpots: 3`: the golden scenes go on
exercising both illustrative features). In `src/main.ts` replace `turbAmp: 0.6` in `state` with
`flicker: FLICKER_DEFAULT` and the uniform's `turbAmp: state.turbAmp` with
`turbAmp: sigmaForFlicker(state.flicker)`; the existing `turb` slider handler becomes
`state.flicker = +turb.value / 100; turbv.textContent = (+turb.value).toFixed(1); reset();` (Task 5 changes
the slider's range to percent; until then the old 0–1.5 range reads as 0–1.5 %).

- [ ] **Step 3: A/B compare at σ = 0**

Run: `node $CLAUDE_JOB_DIR/tmp/ab-turb.mjs $CLAUDE_JOB_DIR/tmp/ab-turb-new.json http://localhost:5173`, then a
compare (max |a − b| / (1 + |a|) over all accum floats per scene) with a one-off node snippet.
Expected: 0 in every scene (σ = 0 skips the field; breathing and spots are unchanged). If not 0: the
difference must be compiler rescheduling — prove it as in 2026-10-01 (make the turbulence branch unreachable
and show bit-identity), ledger a Ruling, and do not proceed on an unexplained difference.
Then stop the :5174 server and remove the worktree (`git worktree remove ../bh-main`; never `--force` with a
running server).

- [ ] **Step 4: Gates and golden**

Run: `npm test`, `npm run build`, then `RECORD_GOLDEN=1 npm run verify:gpu` (re-records all five scenes on
purpose: turbulence is on in each), then `npm run verify:gpu`, `node scripts/probe-axis.mjs`,
`node scripts/probe-scale.mjs`.
Expected: all PASS; `?cachecheck` max 0.00e+0 (turbulence is shading-only: r, φ and the delay are already in the
cache record); app FPS within ~10 % of the pre-change readout (ledger both numbers). Record the five new hashes.

- [ ] **Step 5: Commit**

```bash
git add src/render/raytrace.wgsl src/render/uniforms.ts src/test/scenes.ts src/main.ts src/test/golden.json
git commit -m "Turbulence: renderer shades lognormal MRI turbulence at (r, phi, t_emit); golden re-recorded"
```

---

### Task 5: Panel

**Files:**
- Modify: `index.html`, `src/main.ts`, `scripts/verify-gpu.mjs`

**Interfaces:**
- Consumes: `state.flicker`, `FLICKER_DEFAULT` (Task 4).
- Produces: slider `#turb` in percent (0–10, step 0.5, default 2); `#flare` default 0.

- [ ] **Step 1: Write the failing app check** — in `scripts/verify-gpu.mjs`, after the light-delay toggle check
(the block ending `if (delayLeft) failed = true;`), add:

```js
// Turbulence (spec 2026-10-03): the flicker slider reads in percent with the observed 2 % default, flares
// start off, and nudging the flicker is shading-only (stays cached).
const panel = await page.evaluate(() => ({
  turb: document.getElementById("turb").value, turbMax: document.getElementById("turb").max,
  turbv: document.getElementById("turbv").textContent, flare: document.getElementById("flare").value,
  label: document.getElementById("turb").closest(".ctrl").textContent,
}));
const panelOk = panel.turb === "2" && panel.turbMax === "10" && panel.turbv === "2.0" && panel.flare === "0"
  && /flicker/i.test(panel.label);
preSteps.push(`flicker panel ${panelOk ? "ok" : "FAILED"} (${JSON.stringify(panel)})`);
if (!panelOk) failed = true;
let turbLeft = false;
await page.evaluate(() => { const t = document.getElementById("turb"); t.value = "5"; t.dispatchEvent(new Event("input", { bubbles: true })); });
for (let k = 0; k < 20; k++) { await page.waitForTimeout(75); if (!(await page.evaluate(() => document.getElementById("cmode").textContent)).startsWith("cached")) turbLeft = true; }
await page.evaluate(() => { const t = document.getElementById("turb"); t.value = "2"; t.dispatchEvent(new Event("input", { bubbles: true })); });
preSteps.push(`flicker nudge ${!turbLeft ? "ok" : "FAILED"} (stayed cached ${!turbLeft})`);
if (turbLeft) failed = true;
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm run verify:gpu` (dev server up). Expected: FAIL — `flicker panel FAILED` (`turb` is "0.6", max "1.5",
flare "1"). (If `reset()` in the turb handler makes the mode leave `cached`, the nudge check fails too: that is
the bug it exists to catch; the handler must only re-shade, like the mass slider.)

- [ ] **Step 3: Implement** — `index.html`, replace the Turbulence and Flares controls:

```html
    <div class="ctrl" title="Intrinsic rms flicker of the disk's integrated light from MRI turbulence (observed: ~2 %)">
      <div class="row"><label>Disk flicker (rms)</label><span class="val"><b id="turbv">2.0</b> %</span></div>
      <input id="turb" type="range" min="0" max="10" step="0.5" value="2">
    </div>
    <div class="ctrl" title="Illustrative orbiting hot spots (not MRI turbulence)">
      <div class="row"><label>Flares (illustrative)</label><span class="val"><b id="flarev">0.0</b>×</span></div>
      <input id="flare" type="range" min="0" max="3" step="0.1" value="0">
    </div>
```

`src/main.ts`: `flareScale: 0.0` in `state`; the turb handler from Task 4 stays (it now reads percent). Check the
handler calls the same shading-only path as the mass slider (`reset()` restarts accumulation but must not change
the geometry key; mirror the mass handler if it differs).

- [ ] **Step 4: Run to verify it passes**

Run: `npm test`, `npm run build`, `npm run verify:gpu`, `node scripts/probe-axis.mjs`, `node scripts/probe-scale.mjs`.
Expected: all PASS, `flicker panel ok`, `flicker nudge ok`, panel-reachable check still ok.
Then screenshots (Playwright, 1280×720, default view, settled 3 s): flicker 0 %, 2 %, 10 %; and the M87 preset
at 2 %. Look at them: 2 % shows trailing spiral filaments a few cells long across the whole disk; 10 % is
strongly patchy but not blown out (Review Focus 4). Save under `$CLAUDE_JOB_DIR/tmp/turb-*.png`.

- [ ] **Step 5: Commit**

```bash
git add index.html src/main.ts scripts/verify-gpu.mjs
git commit -m "Panel: Disk flicker (rms) 0-10 % at the observed 2 %; flares illustrative and off by default"
```

---

### Task 6: Docs

**Files:**
- Modify: `README.md`, `docs/ROADMAP.md`

- [ ] **Step 1: README** — add a section after the cooled-jet section, "**MRI disk turbulence
(feat/mri-turbulence, 2026-10-03)**", stating: what replaced the value noise and why (the old features never
died, so shear wound them into ever-tighter stripes); the construction (Gaussian lattice, generations frozen into
the flow, lognormal emission, mean-1); the targets with sources and the measured values from Task 1's tests
(λ_φ, λ_η, 1/e lifetime, emergent tilt early/late, moments); the calibration (κ, σ at 2 %, the slider range and
σ at 10 %); the spec corrections (spectral peak not e-fold; the sqrt reading of SKH06 eq. 33; correlation shape is
the crossfade, not exponential); that flares and breathing remain, illustrative and off; the parity value
(`turb |d g|`); the new golden hashes. Update the Tier 2A paragraph's "Motion / Turbulence / Flares sliders" to
"Motion / Disk flicker / Flares" and the `?parity` bullet's case count.

- [ ] **Step 2: ROADMAP** — replace the "Evolving alpha-disk" item with a checked entry
"**MRI disk turbulence** (2026-10-03): measured sizes, lifetime, spirals and lognormal statistics; amplitude
from the observed 2 % flicker. The viscous evolution it replaced is static on screen (t_visc ≈ 3e6 M at 10 M)."

- [ ] **Step 3: Verify and commit**

Run: `npm test`, `npm run build`.
Expected: PASS.

```bash
git add README.md docs/ROADMAP.md
git commit -m "README and roadmap: MRI disk turbulence"
```
