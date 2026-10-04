# Hot Flow and 1.3 mm View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A "1.3 mm (EHT)" band in which hot-flow objects (λ < 0.01) are rendered as a semi-analytic RIAF
(Broderick et al. 2011; sub-Keplerian dynamics of Pu et al. 2016) in thermal synchrotron at 230 GHz, density
calibrated to the measured flux, shown as brightness temperature in false colour; the visible view is unchanged.

**Architecture:** CPU twin `src/physics/hot-flow.ts` (profiles, velocity, Doppler factor, emissivity/absorption,
n₀ table, regime test) and `src/physics/hot-flow-image.ts` (CPU image tracer, flux, ring measure) used by
`scripts/calibrate-hotflow.ts` and gated tests. WGSL twin in `emission-shared.wgsl`. `raytrace.wgsl`: band switch;
in mm hot-flow mode the ray integrates the flow every 0.25 M inside r < 50 M and does not stop at the disk plane;
mm thin-disk objects shade the disk as T_b = g T; the jet runs at 230 GHz; `storeComposite` writes T_b. `present.wgsl`
maps T_b through afmhot. Cache: band and regime join the geometry key; in mm hot-flow mode entries store the pixel's
steady flow intensity.

**Tech Stack:** TypeScript, WebGPU/WGSL, Vitest, vite-node, Playwright.

**Spec:** `docs/specs/2026-10-04-hot-flow-mm-design.md` (incl. the M87* limitation in §2.4).

## Global Constraints

- Constants: ν = 230 GHz; T₀ = 1e11 K at r = r_S = 2 M (T_e = T₀ (r/2)^−0.84); n_e = n₀ (r/2)^−1.1 exp(−z²/2ρ²);
  B² = 8π n_e m_p c² (2/r)/(12 β), β = 10; Pu mixing 0.5 / 0.5; r < 50 M; sample step 0.25 M; λ_hot = 0.01.
- cgs: c 2.99792458e10, e 4.80320471e-10, m_e 9.1093837e-28, m_p 1.67262192e-24, k 1.380649e-16.
  T_b = I_ν c²/(2ν²k); K_TB = c²/(2ν²k) = 6.15280e13 (cgs).
- Calibration targets: Sgr A* 2.4 Jy at 8.2 kpc, ring 51.8 ± 2.3 µas (gate, unblurred, within 2σ);
  M87* 0.5 Jy at 16.8 Mpc, ring 42 ± 3 µas (reported, not gated). Prototype n₀: Sgr A* ≈ 1.5e7, M87* ≈ 5.0e5 cm⁻³.
- Display: v = T_b / 1e10 K · 2^exposure; afmhot r = clamp(2v), g = clamp(2v − 0.5), b = clamp(2v − 1); no bloom,
  vignette or ACES in mm.
- No Co-Authored-By trailer. Merge, push and deploy when done (standing rule + user request).

## Review Focus

1. **Visible view untouched:** every visible golden scene bit-identical (or proven compiler rescheduling) — Task 4.
2. **f32 range:** j_ν, α_ν and n₀ (5e5–1e8) in f32 on the GPU; no underflow to 0 in the ring, no inf/NaN at the
   horizon (D → large), at ρ → 0 (axis: density → 0), or where K₀ ≤ 0 (no emission) — Task 3 parity cases.
3. **Cache in mm:** cached = live exactly for the mm scenes (steady flow stored, jet replayed) — Task 5.
4. **Regime switch:** a custom object crossing λ = 0.01 while in mm rebuilds the cache and swaps flow ↔ disk — Task 4.
5. **Jet at 230 GHz:** the synch table covers x = ν'/ν_B at mm for every preset field (warn if clamped) — Task 3.

---

### Task 1: Hot-flow physics on the CPU

**Files:** Create `src/physics/hot-flow.ts`; Test `tests/hot-flow.test.ts`.

**Produces:** `HOTFLOW` (constants above); `mahadevanM(X)`; `flowDensity(r, th, n0)`; `flowTemperature(r)`;
`flowField(r, n)`; `iscoEL(a): [E, L]`; `flowVelocity(r, th, a, kUr = 0.5, kOm = 0.5): { ut; ur; Om } | null`;
`flowShift(s: Float64Array, a): number | null` (D at the state's position); `flowCoeffs(r, th, nuPlasma, n0): [j, alpha]`;
`isHotFlow(lambda): boolean`; `flowN0(mSun, lambda, presetId?: string): number`.

- [ ] **Step 1: failing tests** (`tests/hot-flow.test.ts`):

```ts
import { describe, it, expect } from "vitest";
import { HOTFLOW, mahadevanM, flowDensity, flowTemperature, flowField, iscoEL, flowVelocity, flowCoeffs, isHotFlow, flowN0 } from "../src/physics/hot-flow";
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
  it("regime: hot below 1 % Eddington; n0 from the calibration for presets, scaled lambda / M otherwise", () => {
    expect(isHotFlow(0.009)).toBe(true); expect(isHotFlow(0.011)).toBe(false);
    const sg = PRESETS.find((p) => p.id === "sgra")!;
    expect(isHotFlow(sg.lambda)).toBe(true); expect(isHotFlow(PRESETS.find((p) => p.id === "m87")!.lambda)).toBe(true);
    expect(isHotFlow(PRESETS.find((p) => p.id === "cygx1")!.lambda)).toBe(false);
    const n = flowN0(sg.massSun, sg.lambda, "sgra");
    expect(flowN0(sg.massSun, sg.lambda)).toBeCloseTo(n, 6);                      // the scaling reproduces Sgr A*
    expect(flowN0(2 * sg.massSun, 3 * sg.lambda) / n).toBeCloseTo(1.5, 9);         // n0 ~ lambda / M
    expect(HOTFLOW.nu).toBe(230e9);
  });
});
```

- [ ] **Step 2:** `npx vitest run tests/hot-flow.test.ts` — FAIL (module missing).
- [ ] **Step 3: implement** `src/physics/hot-flow.ts`:

```ts
// Hot accretion flow for the 1.3 mm (EHT) view (spec 2026-10-04): the semi-analytic RIAF of Broderick et al. 2011/2016
// (n_e ~ r^-1.1 e^{-z^2/2 rho^2}, T_e ~ r^-0.84, toroidal field at beta = 10) with the sub-Keplerian dynamics of
// Pu, Akiyama & Asada 2016 (Keplerian / zero-angular-momentum free fall mixed 50/50 in u^r and Omega; equatorial
// profiles of r, u^t from the local metric), thermal synchrotron (Mahadevan et al. 1996 isotropic fit) with
// Kirchhoff absorption. n0 calibrated to the measured 230 GHz flux (scripts/calibrate-hotflow.ts). WGSL twin in
// emission-shared.wgsl.
import { metricLower, metricUpper } from "./kerr";
import { iscoRadius } from "./orbits";
import { PRESETS } from "./presets";

export const HOTFLOW = { nu: 230e9, T0: 1e11, beta: 10, rMax: 50, dl: 0.25, lambdaHot: 0.01, kTb: 6.1528e13 } as const;
const C = 2.99792458e10, QE = 4.80320471e-10, ME = 9.1093837e-28, MP = 1.67262192e-24, KB = 1.380649e-16;

export function mahadevanM(X: number): number {
  return 4.0505 * X ** (-1 / 6) * (1 + 0.4 * X ** -0.25 + 0.5316 * X ** -0.5) * Math.exp(-1.8899 * Math.cbrt(X));
}
export function flowDensity(r: number, th: number, n0: number): number {
  const z = r * Math.cos(th), rho = r * Math.sin(th);
  if (rho < 1e-6) return 0;
  return n0 * (r / 2) ** -1.1 * Math.exp(-(z * z) / (2 * rho * rho));
}
export function flowTemperature(r: number): number { return HOTFLOW.T0 * (r / 2) ** -0.84; }
/** B^2 / 8 pi = n m_p c^2 r_S / (12 r beta), r_S = 2 M (Broderick et al. 2011). */
export function flowField(r: number, n: number): number { return Math.sqrt((8 * Math.PI * n * MP * C * C * (2 / r)) / (12 * HOTFLOW.beta)); }
/** Energy and angular momentum per unit mass of the prograde ISCO orbit (M = 1). */
export function iscoEL(a: number): [number, number] {
  const r = iscoRadius(a, true), sq = Math.sqrt(r), den = r ** 0.75 * Math.sqrt(r * sq - 3 * sq + 2 * a);
  return [(r * sq - 2 * sq + a) / den, (r * r - 2 * a * sq + a * a) / den];
}
/** Pu et al. 2016 eqs. 1-3: u^r and Omega mixed between Keplerian (plunging inside the ISCO) and zero-angular-momentum
 *  free fall, as functions of r on the equatorial metric; u^t from the local metric. null where K0 <= 0. */
export function flowVelocity(r: number, th: number, a: number, kUr = 0.5, kOm = 0.5): { ut: number; ur: number; Om: number } | null {
  const gu = metricUpper(r, Math.PI / 2, a), risco = iscoRadius(a, true);
  let urK = 0, OmK = 1 / (r ** 1.5 + a);
  if (r < risco) {
    const [E, L] = iscoEL(a);
    const uT = -gu.tt * E + gu.tphi * L, uP = -gu.tphi * E + gu.phph * L;
    const rest = -1 - (gu.tt * E * E - 2 * gu.tphi * E * L + gu.phph * L * L);
    urK = -Math.sqrt(Math.max(0, rest * gu.rr)); OmK = uP / uT;
  }
  const urFF = -Math.sqrt(Math.max(0, gu.rr * (-1 - gu.tt))), OmFF = gu.tphi / gu.tt;
  // Pu eq. 2/3 written with weights on the Keplerian part: kUr = alpha, kOm = beta (1 = Keplerian, 0 = free fall)
  const ur = urK + (1 - kUr) * (urFF - urK), Om = OmK + (1 - kOm) * (OmFF - OmK);
  const g = metricLower(r, th, a), K0 = -(g.tt + 2 * Om * g.tphi + Om * Om * g.phph);
  if (K0 <= 0) return null;
  return { ut: Math.sqrt((1 + g.rr * ur * ur) / K0), ur, Om };
}
/** nu_plasma / nu_observed for the camera-normalised, past-directed covariant momentum of state s (t, r, th, phi, p...). */
export function flowShift(s: Float64Array, a: number): number | null {
  const u = flowVelocity(s[1], s[2], a); if (!u) return null;
  return u.ut * s[4] + u.ur * s[5] + u.Om * u.ut * s[7];
}
/** Plasma-frame thermal synchrotron j_nu (erg s^-1 cm^-3 Hz^-1 sr^-1) and alpha_nu (cm^-1) at nu. */
export function flowCoeffs(r: number, th: number, nu: number, n0: number): [number, number] {
  const n = flowDensity(r, th, n0); if (n <= 0) return [0, 0];
  const T = flowTemperature(r), the = (KB * T) / (ME * C * C), B = flowField(r, n);
  const nuc = (QE * B) / (2 * Math.PI * ME * C), X = (2 * nu) / (3 * nuc * the * the);
  const j = ((n * QE * QE * nu) / (2 * Math.sqrt(3) * C * the * the)) * mahadevanM(X);
  return [j, j / ((2 * nu * nu * KB * T) / (C * C))];
}
export function isHotFlow(lambda: number): boolean { return lambda < HOTFLOW.lambdaHot; }
/** n0 calibrated to the measured 230 GHz flux (scripts/calibrate-hotflow.ts); other objects scale from Sgr A*'s as
 *  density ~ Mdot / (r_g^2 c) ~ lambda / M. */
export const HOTFLOW_N0: Record<string, number> = { sgra: 1.5e7, m87: 5.0e5 }; // replaced by the script's values in Step 4 of Task 2
export function flowN0(mSun: number, lambda: number, presetId?: string): number {
  if (presetId && HOTFLOW_N0[presetId] !== undefined) return HOTFLOW_N0[presetId];
  const sg = PRESETS.find((p) => p.id === "sgra")!;
  return HOTFLOW_N0.sgra * (lambda / sg.lambda) * (sg.massSun / mSun);
}
```

- [ ] **Step 4:** run — PASS. **Step 5:** `git commit -m "Hot flow: CPU physics (Broderick RIAF profiles, Pu velocity, thermal synchrotron, regime, n0 scaling)"`.

---

### Task 2: CPU image, calibration and the gated validation

**Files:** Create `src/physics/hot-flow-image.ts`, `scripts/calibrate-hotflow.ts`, `tests/sweep-hotflow.test.ts`;
Modify `src/physics/hot-flow.ts` (the `HOTFLOW_N0` values).

**Produces:** `HOTFLOW_TARGETS` (Sgr A* / M87*: distance kpc, Jy, ring µas ± err, a, incl, mass from presets);
`traceFlowSamples(a, inclDeg, N, half): FlowSamples` (per-ray (r, th, D, dl) for D > 0 inside r < 50, from the CPU
integrator with steps ≤ 0.5 M); `flowImage(samples, n0, rgCm): Float64Array` (I_ν cgs per pixel, slab transfer);
`imageFluxJy(I, N, half, rgCm, distCm)`; `ringDiameterUas(I, N, half, uasPerM, blurUas = 0)` (azimuthal-mean peak).

- [ ] **Step 1: failing gated test** `tests/sweep-hotflow.test.ts` (skipped unless `SWEEP=1`, like the other sweeps):

```ts
import { describe, it, expect } from "vitest";
import { HOTFLOW_TARGETS, traceFlowSamples, flowImage, imageFluxJy, ringDiameterUas } from "../src/physics/hot-flow-image";
import { flowN0 } from "../src/physics/hot-flow";
import { PRESETS } from "../src/physics/presets";
const RUN = process.env.SWEEP === "1";
describe.skipIf(!RUN)("hot flow calibration (48^2 CPU trace, ~1 min per object)", () => {
  for (const id of ["sgra", "m87"] as const) {
    it(`${id}: tabulated n0 reproduces the measured 230 GHz flux within 5 %; ring reported (Sgr A* gated within 2 sigma)`, () => {
      const p = PRESETS.find((q) => q.id === id)!, T = HOTFLOW_TARGETS[id];
      const rg = 1.476625e5 * p.massSun, dist = T.distKpc * 3.0857e21, uasPerM = (rg / dist) * 206264.806e6;
      const smp = traceFlowSamples(p.a, p.inclDeg, 48, 13);
      const I = flowImage(smp, flowN0(p.massSun, p.lambda, id), rg);
      expect(Math.abs(imageFluxJy(I, 48, 13, rg, dist) / T.jy - 1)).toBeLessThan(0.05);
      const d = ringDiameterUas(I, 48, 13, uasPerM);
      console.log(`${id}: ring ${d.toFixed(1)} uas vs EHT ${T.ringUas} +- ${T.ringErr}`);
      if (id === "sgra") expect(Math.abs(d - T.ringUas)).toBeLessThan(2 * T.ringErr);
    }, 600000);
  }
});
```

- [ ] **Step 2:** `SWEEP=1 npx vitest run tests/sweep-hotflow.test.ts` — FAIL (module missing).
- [ ] **Step 3: implement** `src/physics/hot-flow-image.ts`:

```ts
// CPU image of the hot flow at 230 GHz (calibration and validation; the renderer's twin is the GPU path).
import { screenToState } from "./camera";
import { stepGeodesic, stepSize, H_TOL, H_TOL_FAR } from "./trace";
import { horizonOuter } from "./kerr";
import { HOTFLOW, flowShift, flowCoeffs } from "./hot-flow";

export const HOTFLOW_TARGETS = {
  sgra: { distKpc: 8.2, jy: 2.4, ringUas: 51.8, ringErr: 2.3 },   // EHT 2022 (Sgr A* Papers I, IV)
  m87: { distKpc: 16800, jy: 0.5, ringUas: 42, ringErr: 3 },      // EHT 2019 (Papers I, IV, VI); not gated (spec 2.4)
} as const;
export interface FlowSamples { N: number; half: number; rays: Float64Array[] } // per ray: (r, th, D, dl) quads
export function traceFlowSamples(a: number, inclDeg: number, N: number, half: number): FlowSamples {
  const incl = (inclDeg * Math.PI) / 180, rh = horizonOuter(a), rays: Float64Array[] = [];
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const al = -half + (2 * half * (i + 0.5)) / N, be = half - (2 * half * (j + 0.5)) / N;
    let s = screenToState(al, be, a, incl, 1000); const q: number[] = [];
    for (let k = 0; k < 30000; k++) {
      const out = stepGeodesic(s, a, Math.min(stepSize(s, rh, 40), 0.5), s[1] > 60 ? H_TOL_FAR : H_TOL);
      if (!out.ok) break;
      const r = out.s[1];
      if (r < HOTFLOW.rMax && r > rh * 1.01) { const D = flowShift(out.s, a); if (D !== null && D > 0) q.push(r, out.s[2], D, out.dl); }
      s = out.s;
      if (r <= rh * 1.01 || r > 1100) break;
    }
    rays.push(Float64Array.from(q));
  }
  return { N, half, rays };
}
export function flowImage(smp: FlowSamples, n0: number, rgCm: number): Float64Array {
  const I = new Float64Array(smp.N * smp.N);
  smp.rays.forEach((q, idx) => {
    let Iv = 0, tau = 0;
    for (let k = 0; k < q.length; k += 4) {
      const [r, th, D, dl] = [q[k], q[k + 1], q[k + 2], q[k + 3]];
      const [j, al] = flowCoeffs(r, th, D * HOTFLOW.nu, n0); if (j === 0) continue;
      const ds = rgCm * D * dl, dt = al * ds, fac = dt < 1e-4 ? 1 - 0.5 * dt : (1 - Math.exp(-dt)) / dt;
      Iv += (j / D ** 3) * ds * fac * Math.exp(-tau); tau += dt;
    }
    I[idx] = Iv;
  });
  return I;
}
export function imageFluxJy(I: Float64Array, N: number, half: number, rgCm: number, distCm: number): number {
  const pix = ((2 * half) / N) * (rgCm / distCm); let F = 0; for (const v of I) F += v * pix * pix; return F / 1e-23;
}
export function ringDiameterUas(I: Float64Array, N: number, half: number, uasPerM: number, blurUas = 0): number {
  let img = I; const px = (2 * half) / N;
  if (blurUas > 0) {
    const sig = blurUas / 2.3548 / uasPerM, R = Math.ceil((3 * sig) / px); img = new Float64Array(N * N);
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { let v = 0, w = 0;
      for (let dj = -R; dj <= R; dj++) for (let di = -R; di <= R; di++) { const ii = i + di, jj = j + dj; if (ii < 0 || jj < 0 || ii >= N || jj >= N) continue;
        const ww = Math.exp(-((di * px) ** 2 + (dj * px) ** 2) / (2 * sig * sig)); v += ww * I[jj * N + ii]; w += ww; } img[j * N + i] = v / w; }
  }
  const nb = 60, prof = new Float64Array(nb), cnt = new Float64Array(nb);
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const b = Math.hypot(-half + px * (i + 0.5), -half + px * (j + 0.5)), k = Math.floor((b / half) * nb);
    if (k < nb) { prof[k] += img[j * N + i]; cnt[k]++; } }
  let kmax = 0; for (let k = 1; k < nb; k++) if (cnt[k] && prof[k] / cnt[k] > prof[kmax] / Math.max(cnt[kmax], 1)) kmax = k;
  return 2 * ((kmax + 0.5) / nb) * half * uasPerM;
}
```

  `scripts/calibrate-hotflow.ts` (96², bisection on n0 reusing the samples, prints both rings):

```ts
// Reproduces HOTFLOW_N0 in src/physics/hot-flow.ts (run: npx vite-node scripts/calibrate-hotflow.ts, ~6 min).
import { HOTFLOW_TARGETS, traceFlowSamples, flowImage, imageFluxJy, ringDiameterUas } from "../src/physics/hot-flow-image";
import { PRESETS } from "../src/physics/presets";
for (const id of ["sgra", "m87"] as const) {
  const p = PRESETS.find((q) => q.id === id)!, T = HOTFLOW_TARGETS[id], rg = 1.476625e5 * p.massSun, dist = T.distKpc * 3.0857e21;
  const smp = traceFlowSamples(p.a, p.inclDeg, 96, 13); let lo = 1e3, hi = 1e10;
  for (let k = 0; k < 40; k++) { const m = Math.sqrt(lo * hi); if (imageFluxJy(flowImage(smp, m, rg), 96, 13, rg, dist) < T.jy) lo = m; else hi = m; }
  const I = flowImage(smp, lo, rg), u = (rg / dist) * 206264.806e6;
  console.log(`${id}: n0 ${lo.toPrecision(3)} cm^-3 -> ${imageFluxJy(I, 96, 13, rg, dist).toFixed(3)} Jy; ring ${ringDiameterUas(I, 96, 13, u).toFixed(1)} uas (20 uas blur: ${ringDiameterUas(I, 96, 13, u, 20).toFixed(1)}) vs EHT ${T.ringUas} +- ${T.ringErr}; ${u.toFixed(2)} uas/M`);
}
```

- [ ] **Step 4:** `npx vite-node scripts/calibrate-hotflow.ts` (background, ≤ 15 min) — write its two n0 values (3 significant
  figures) into `HOTFLOW_N0`; record the printed rings in the ledger. Expected near the prototype: Sgr A* n0 ≈ 1.5e7,
  ring ≈ 48 µas; M87* n0 ≈ 5e5, ring ≈ 36 µas.
- [ ] **Step 5:** `SWEEP=1 npx vitest run tests/sweep-hotflow.test.ts` — PASS (record the printed rings);
  `npx vitest run tests/hot-flow.test.ts` — PASS (the scaling test uses the new values).
- [ ] **Step 6:** `git commit -m "Hot flow: CPU image, calibration script (n0 from the measured flux), gated EHT ring check"`.

---

### Task 3: WGSL twin and `?parity`

**Files:** `src/render/emission-shared.wgsl`, `src/render/jet-parity.wgsl`, `src/render/raytrace.wgsl` (only the
`synchSampleJ` call), Create `src/render/flow-parity.wgsl`, Modify `src/test/parity.browser.ts`, `src/main.ts`
(parity line), `tests/jet.test.ts` (WGSL constants).

**Produces (WGSL):** `const HF_LNNU` (ln 230 GHz), `HF_T0`, `HF_BETA`, `HF_RMAX`, `HF_KTB`; `fn iscoJ(a)` (prograde ISCO,
Bardeen–Press–Teukolsky; add only if absent — `grep -n "fn iscoJ" src/render/*.wgsl`); `fn flowVelocityJ(r, th, a) -> vec4<f32>`
(ut, ur, Om, ok ∈ {0,1}); `fn flowShiftJ(r, th, p: vec4<f32>, a) -> f32` (≤ 0 when no valid velocity);
`fn flowCoeffsJ(r, th, nu: f32, n0: f32) -> vec2<f32>` (j, α in plasma frame, computed in logs);
`synchSampleJ` gains `lnNu: vec3<f32>` as its first argument (the jet's three band frequencies, or ln 230 GHz × 3).

- [ ] **Step 1: failing constants test** — in `tests/jet.test.ts` add `["HF_T0", 1e11], ["HF_BETA", 10], ["HF_RMAX", 50], ["HF_KTB", HOTFLOW.kTb]`
  and `expect(Number(wconst("HF_LNNU"))).toBeCloseTo(Math.log(230e9), 7)` (import `HOTFLOW`). Run — FAIL.
- [ ] **Step 2: implement** in `emission-shared.wgsl` (after the jet block):

```wgsl
// --- Hot flow at 230 GHz (spec 2026-10-04; twin: src/physics/hot-flow.ts) --------------------------------------
const HF_LNNU = 26.16134515;        // ln(230e9)
const HF_T0 = 1e11; const HF_BETA = 10.0; const HF_RMAX = 50.0; const HF_KTB = 6.1528e13;
// cgs logs (Global Constraints values): ln e^2, ln(k / m_e c^2), ln(8 pi m_p c^2 * 2 / (12 beta)) [B^2 = e^(that + ln n - ln r)],
// ln(e / 2 pi m_e c), ln(2 k / c^2), ln(2 sqrt(3) c)
const HF_LN_QE2 = -42.91313517; const HF_LN_THE = -22.50327261; const HF_LN_B2 = -7.37028061;
const HF_LN_NUC = 14.84486172; const HF_LN_RJ = -84.07320297; const HF_LN_2S3C = 25.3662245;
```

  (HF_LN_NUC equals the jet's SYN_LNNUB0 — reuse `SYN_LNNUB0` instead of a second constant if the shader-twins test
  objects to duplicate values; it does not check values, only function names.) Then:

```wgsl
fn mahadevanMJ(X: f32) -> f32 { return 4.0505 * pow(X, -1.0 / 6.0) * (1.0 + 0.4 * pow(X, -0.25) + 0.5316 * pow(X, -0.5)) * exp(-1.8899 * pow(X, 1.0 / 3.0)); }
fn flowVelocityJ(r: f32, th: f32, a: f32) -> vec4<f32> {
  let gu = gUp(r, 0.5 * PI, a);                       // 0 tt, 1 tphi, 2 rr, 3 thth, 4 phph
  let risco = iscoJ(a);
  var urK = 0.0; var OmK = omegaKep(r, a);
  if (r < risco) {
    let sq = sqrt(risco); let den = pow(risco, 0.75) * sqrt(risco * sq - 3.0 * sq + 2.0 * a);
    let E = (risco * sq - 2.0 * sq + a) / den; let L = (risco * risco - 2.0 * a * sq + a * a) / den;
    let uT = -gu[0] * E + gu[1] * L; let uP = -gu[1] * E + gu[4] * L;
    let rest = -1.0 - (gu[0] * E * E - 2.0 * gu[1] * E * L + gu[4] * L * L);
    urK = -sqrt(max(0.0, rest * gu[2])); OmK = uP / uT;
  }
  let urFF = -sqrt(max(0.0, gu[2] * (-1.0 - gu[0]))); let OmFF = gu[1] / gu[0];
  let ur = urK + 0.5 * (urFF - urK); let Om = OmK + 0.5 * (OmFF - OmK);
  let g = gLow(r, th, a);                              // 0 tt, 1 tphi, 2 rr, 3 thth, 4 phph
  let K0 = -(g[0] + 2.0 * Om * g[1] + Om * Om * g[4]);
  if (K0 <= 0.0) { return vec4<f32>(0.0); }
  return vec4<f32>(sqrt((1.0 + g[2] * ur * ur) / K0), ur, Om, 1.0);
}
fn flowShiftJ(r: f32, th: f32, p: vec4<f32>, a: f32) -> f32 {
  let u = flowVelocityJ(r, th, a);
  if (u.w == 0.0) { return -1.0; }
  return u.x * p.x + u.y * p.y + u.z * u.x * p.w;        // p = (p_t, p_r, p_th, p_phi)
}
// Plasma-frame (j, alpha) at frequency e^lnNu; computed in logs (n0 ~ 1e5-1e8 and j ~ 1e-20 stay in f32 range).
fn flowCoeffsJ(r: f32, th: f32, lnNu: f32, n0: f32) -> vec2<f32> {
  let z = r * cos(th); let rho = r * sin(th);
  if (rho < 1e-6) { return vec2<f32>(0.0); }
  let lnN = log(n0) - 1.1 * log(0.5 * r) - z * z / (2.0 * rho * rho);
  if (lnN < -40.0) { return vec2<f32>(0.0); }
  let lnT = log(HF_T0) - 0.84 * log(0.5 * r);
  let lnThe = HF_LN_THE + lnT;
  let lnB = 0.5 * (HF_LN_B2 + lnN - log(r));
  let lnX = log(2.0 / 3.0) + lnNu - (HF_LN_NUC + lnB) - 2.0 * lnThe;
  let M = mahadevanMJ(exp(lnX));
  if (M <= 0.0) { return vec2<f32>(0.0); }
  let lnJ = lnN + HF_LN_QE2 + lnNu - HF_LN_2S3C - 2.0 * lnThe + log(M);
  let lnA = lnJ - (HF_LN_RJ + 2.0 * lnNu + lnT);
  return vec2<f32>(exp(lnJ), exp(lnA));
}
```

  Change `fn synchSampleJ(r: f32, …)` to `fn synchSampleJ(lnNu: vec3<f32>, r: f32, …)` and use `lnNu[b]` in place of
  `JET_LNNU[b]` inside; update its two callers: `jet-parity.wgsl` passes `JET_LNNU`, `raytrace.wgsl` passes
  `select(JET_LNNU, vec3<f32>(HF_LNNU), U.band > 0.5)` (the `band` uniform arrives in Task 4 — until then pass `JET_LNNU`).
  Add `iscoJ` if absent:

```wgsl
fn iscoJ(a: f32) -> f32 { // prograde ISCO (Bardeen, Press & Teukolsky 1972; twin: iscoRadius in orbits.ts)
  let z1 = 1.0 + pow(1.0 - a * a, 1.0 / 3.0) * (pow(1.0 + a, 1.0 / 3.0) + pow(max(1.0 - a, 0.0), 1.0 / 3.0));
  let z2 = sqrt(3.0 * a * a + z1 * z1);
  return 3.0 + z2 - sqrt(max((3.0 - z1) * (3.0 + z1 + 2.0 * z2), 0.0));
}
```

- [ ] **Step 3: parity.** `src/render/flow-parity.wgsl`: input `a = (r, th, spin, n0)`, `b = p (4 floats)`; output
  `(flowShiftJ, ln j, ln alpha, ut)` with `flowCoeffsJ(r, th, HF_LNNU + log(max(D, 1e-6)), n0)`. Cases (in
  `parity.browser.ts`): states from the CPU tracer at spins 0.1 / 0.94, r ∈ {1.5, 2.5, 5, 12, 30, 49}, latitudes near
  the plane, mid and near the axis (th 1.55, 1.0, 0.15), n0 ∈ {5e5, 1.5e7}; CPU expectation from `flowShift` and
  `flowCoeffs` (cases with j = 0 compare only D). Metric: relative on D and u^t (1e-4), absolute on ln j, ln α (2e-3).
  Report `flowErr` in the PARITY line; fail above tolerance. Jet parity numbers must be unchanged.
- [ ] **Step 4: run** — CPU tests PASS; `?parity` PASS (flow worst ≤ 1); mutation `HF_BETA = 11.0` → FAIL; revert.
- [ ] **Step 5:** `git commit -m "Hot flow: WGSL twin (velocity, shift, coefficients in logs), synchSampleJ takes band frequencies, ?parity flow cases"`.

---

### Task 4: Renderer — band switch, flow integration, mm display, cache

**Files:** `src/render/uniforms.ts`, `raytrace.wgsl`, `present.wgsl`, `bloom.wgsl` (struct only), `src/render/cache-plan.ts`
(geometry key), `src/main.ts`, `tests/uniforms.test.ts`, `tests/cache-plan.test.ts`.

**Produces:** uniforms `band` (0 visible / 1 mm), `hotFlow` (0/1), `flowN0` (cm⁻³) at floats 36–38 (block 160 B);
`GeometryInputs.band`, `.hotFlow` in the key.

- [ ] **Step 1: failing tests:** `uniforms.test.ts` — size 160, band/hotFlow/flowN0 at byte offsets 144/148/152;
  `cache-plan.test.ts` — `geometryKey` differs when band or hotFlow changes and not when shading-only values change.
- [ ] **Step 2: uniforms** — `UniformValues` gains `band?`, `hotFlow?`, `flowN0?` (default 0); `UNIFORM_FLOATS = 31`;
  `f[36] = u.band ?? 0; f[37] = u.hotFlow ?? 0; f[38] = u.flowN0 ?? 0;`. Every WGSL `Uniforms` struct that declares
  `timeEpoch` gains `band: f32, hotFlow: f32, flowN0: f32,` after it; `present.wgsl`'s shorter struct is extended with
  all fields through `flowN0` (copy the order from `raytrace.wgsl`) so it can read `U.band`.
- [ ] **Step 2b: jet at 230 GHz (spec §2.5, added in Task 3)** — `hot-flow.ts`: `jetAtMm(a, b0): boolean` = 230e9 >
  ν_B · max B over the jet base wall (`jetField(funnelEdge(JET.zBase), JET.zBase, a, b0)` with `LN_NUB0`); test: M87*
  true, Cyg X-1 and GRS 1915+105 false (b0 from `jetEnergetics`). `main.ts`: in mm with `!jetAtMm`, the jet strength
  uniform is 0 (the jet is off, and the cache rebuilds as for any jet toggle).
- [ ] **Step 3: traceRay** (`raytrace.wgsl`):
  - `let mmFlow = U.band > 0.5 && U.hotFlow > 0.5;` and a second accumulator `var flow: JetOut` (use `.x` channels).
  - Each step, when `mmFlow` and the step's chord comes within `HF_RMAX` (reuse `jetSubCount`/`jetSample` sub-sampling;
    skip when both endpoints have r > HF_RMAX): for each sample, `D = flowShiftJ(q.x, q.y, mix(s.p, sNew.p, f), a)`;
    if D > 1e-6, `c = flowCoeffsJ(q.x, q.y, HF_LNNU + log(D), U.flowN0)`; slab `jetSlabJ(flow, vec3(c.x / (D*D*D)), vec3(c.y), U.rgCm * D * dl / n)`.
  - Disk-crossing test only when `!mmFlow` (rays cross the plane in a hot flow).
  - mm thin-disk shading: when `U.band > 0.5 && !mmFlow`, a disk hit's colour is `vec3(g * U.Tpeak * sampleTemp(rHit))`
    (T_b of a Rayleigh–Jeans blackbody, I_ν/ν³ invariant ⇒ T_b,obs = g T); sky colour in mm is 0.
  - `TraceOut` gains `flowI: f32`.
- [ ] **Step 4: composite** — `storeComposite(idx, color, jet, flowI)`; in mm:
  `raw = vec3(color.x * exp(-jet.tau.x) + HF_KTB * (max(jet.I.x, 0.0) + flowI))` (T_b, K); visible unchanged.
  `main`/`shade`/`build` pass `t.flowI` (0 in visible).
- [ ] **Step 5: cache** — `build`: in mm hot-flow mode store `Entry(word, t.flowI, 0, 0)` (no disk kind can occur; sky
  needs no direction in mm). `shade`: in mm hot-flow mode `flowI = e.p0`, `color = 0`; in mm thin-disk mode DISK
  entries use the mm disk shading; jet replay unchanged. `cache-plan.ts`: `GeometryInputs` gains `band`, `hotFlow`;
  `geometryKey` appends them. `main.ts` passes them.
- [ ] **Step 6: present** (`present.wgsl`): at the top of `fs`, after `hdr` is read, `if (U.band > 0.5) { let v = hdr.x / 1e10 * exp2(U.exposure);
  return vec4<f32>(clamp(2.0 * v, 0.0, 1.0), clamp(2.0 * v - 0.5, 0.0, 1.0), clamp(2.0 * v - 1.0, 0.0, 1.0), 1.0); }`
  (no bloom, vignette or tone map in mm). Bloom still runs (harmless; its output is unused in mm).
- [ ] **Step 7: run** `npx vitest run tests/uniforms.test.ts tests/cache-plan.test.ts` — PASS; `npm run build` clean.
- [ ] **Step 8:** `git commit -m "Renderer: 1.3 mm band (hot-flow volume integration, mm disk and jet, T_b composite, afmhot display, cache)"`.

---

### Task 5: UI, scenes, validation routes and app checks

**Files:** `index.html`, `src/main.ts`, `src/share.ts` (+ `tests/share.test.ts`), `src/physics/presets.ts` (captions),
`src/test/scenes.ts`, Create `src/test/hotflow.browser.ts`, `scripts/verify-gpu.mjs`, `scripts/bench.mjs`.

- [ ] **Step 1: failing tests:** `tests/share.test.ts` — `SHARE_FIELDS` contains `{ key: "b", id: "band", kind: "select" }`
  directly after `p`; a link `#p=sgra&b=mm` decodes to `{ p: "sgra", b: "mm" }` with options `["vis", "mm"]`.
- [ ] **Step 2: UI** — `index.html`: next to Exposure, `<select id="band"><option value="vis">Visible</option><option value="mm">1.3 mm (EHT)</option></select>`
  with a title explaining the band; `main.ts`: `state.band`, change handler → `reset()` (geometry key changes);
  uniforms `band: state.band === "mm" ? 1 : 0`, `hotFlow: isHotFlow(state.lambda) ? 1 : 0`,
  `flowN0: flowN0(state.massSun, state.lambda, presetSel.value)`; share field `b` (after `p`, kind select).
  Captions (presets.ts) for Sgr A* and M87*: replace "a hot flow this renderer does not model" with a sentence on the
  1.3 mm view and, for M87*, the ring limitation (spec §2.4).
- [ ] **Step 3: scenes** — `Scene` gains `band?: "vis" | "mm"`; add `{ name: "sgra-mm", … }` (Sgr A* preset a, incl,
  obj, band mm, jet off) and `{ name: "m87-mm", … }` (jet on); `sceneUniforms` sets band/hotFlow/flowN0 from the scene.
- [ ] **Step 4: `?hotflow` route** (`src/test/hotflow.browser.ts` + a `location.search.includes("hotflow")` branch in
  `main.ts`): render `sgra-mm` at 512×512 internal, 16 converged frames, `readbackAccum()`, convert T_b → I_ν
  (÷ HF_KTB), flux = Σ I · (2·fovScale/512 · r_g/D)² → Jy, ring diameter via `ringDiameterUas` (import from
  hot-flow-image.ts, applied to the GPU image). PASS iff flux within 5 % of 2.4 Jy and the ring within 2σ of 51.8 µas;
  print both, and M87*'s flux and ring (reported). Add `await check("/?hotflow", "HOTFLOW PASS")` to `verify-gpu.mjs`.
- [ ] **Step 5: app check** (verify-gpu, after the share check): open `#p=sgra&b=mm`, wait for `cached`, the canvas
  right of the panel has a bright ring (max brightness > 120 and the centre darker than the ring) and no warnings.
- [ ] **Step 6: bench** — `bench.mjs` gains an mm run (Sgr A* uniforms, band 1): report live and cached ms at 720p next
  to the visible numbers.
- [ ] **Step 7: gates** — `npm test`, `npm run build`, `RECORD_GOLDEN=1 npm run verify:gpu`: all PASS; visible golden
  hashes unchanged (else the stubbed-kernel proof: stub the flow block and compare with main); `?cachecheck` 0.00e+0
  including the two mm scenes; `?hotflow` PASS.
- [ ] **Step 8:** `git commit -m "1.3 mm view: band control and share key, mm golden/cachecheck scenes, ?hotflow GPU flux and ring gate, app check, bench"`.

---

### Task 6: Visual check, docs, review, ship

- [ ] **Step 1:** Screenshots: Sgr A* and M87* in both bands (`#p=sgra&b=mm`, `#p=m87&b=mm`), a thin-disk object in mm;
  look at them (ring, asymmetry, shadow, jet base for M87*).
- [ ] **Step 2:** README "Hot flow and the 1.3 mm view" (model, sources, calibration table with the script's n0 and the
  measured rings, the M87* limitation, performance numbers); ROADMAP entry; spec status "implemented".
- [ ] **Step 3:** Final whole-branch review (fresh reviewer, most capable model); one fix pass; fast-forward main, push,
  `vercel deploy --prod` (PowerShell); verify the live bundle contains `flowCoeffsJ`.
