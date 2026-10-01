# Real-Object Presets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One-click presets for M87\*, Sgr A\*, Cygnus X-1, GRS 1915+105 and Gargantua with physically
derived disk temperature, Mass and Accretion sliders, physical-unit readouts, and visible-band
brightness in every view.

**Architecture:** Pure TypeScript physics modules (`units.ts`, `readouts.ts`, `presets.ts`) compute
T_peak and readouts on the CPU; a log-spaced visible-radiance LUT (`lookups.ts`) replaces the
luminance-normalised colour LUT, and the shader's disk shading becomes `LUT(T_obs) * lumNorm * E`.
The panel gains an Object selector with captions, two log sliders and new readouts. Mass and
accretion are shading-only: they never touch the geodesic cache's geometry key.

**Tech Stack:** TypeScript, WebGPU/WGSL, Vite, Vitest, Playwright (headless Chrome gates).

**Spec:** `docs/specs/2026-10-01-real-object-presets-design.md`

## Global Constraints

- Every change keeps green: `npm test`, `npm run build`, `npm run verify:gpu` (dev server on :5173;
  any console warning fails a route), `node scripts/probe-axis.mjs`, `node scripts/probe-scale.mjs`.
- `?golden` is re-recorded ONCE, deliberately, in Task 3 (`RECORD_GOLDEN=1 npm run verify:gpu`),
  because the brightness law changes; every later task must reproduce those hashes.
- `?cachecheck` must PASS unchanged in meaning (cached frame = live trace, ≤ ~4e-7).
- Slider ranges: spin 0–0.998, inclination 1–89°, Mass log10 0–10 (M_sun), Accretion log10 −10–0
  (fraction of Eddington). Every preset value lies inside them.
- Preset values and captions exactly as in the spec §3 (λ for M87\* and Sgr A\* derived from the
  cited Ṁ in code; Gargantua's λ solved for T_peak = 5,800 K; Custom default solved for 30,000 K).
- No Co-Authored-By trailer on commits in this repo.
- Commit messages: imperative, describe the change (house style, see `git log`).

## Review Focus

1. **Preset switch while the geodesic cache is built:** a preset that changes spin/inclination must
   drop to live and rebuild; one that only changes mass/accretion must re-shade at once with no
   rebuild. Pinned by Task 5's `verify-gpu` preset loop (mode returns to `cached`, image lit, no
   console warnings).
2. **Slider extremes:** Mass 10¹⁰ with Accretion 10⁻¹⁰ gives T_peak ≈ 124 K. Its visible luminance
   is ~10⁻⁹⁰ of a 10⁴ K disk, so an unclamped `lumNorm` (~10⁹⁰) would be Infinity in f32 and
   Infinity × 0 = NaN on the GPU. `lumNormFor` is capped at 10³⁰: such a disk renders dark (it does
   not glow visibly), never NaN. Pinned by Task 1's extremes test and Task 2's `lumNormFor` cap test.
3. **Changing spin in Custom keeps M and λ:** T_peak and the km readouts must update with spin
   (η(a) and the flux profile change). Pinned by Task 4's `computeReadouts` test.
4. **Dragging the canvas to tilt is an inclination change:** it must switch the selector to Custom
   just like the slider. Pinned by Task 5's verify step (drag, then read `#preset`).
5. **Unit formatting across 20 orders of magnitude:** ms for Cygnus X-1's ISCO orbit, days for
   M87\*'s, AU for M87\*'s horizon, km for stellar ones. Pinned by Task 1's formatter tests.

---

## File Structure

- Create `src/physics/units.ts` — SI constants, r_g, t_g, L_Edd, λ ↔ Ṁ, `peakTemperature`,
  `lambdaForPeakTemperature`, `iscoPeriod`, `formatLength`, `formatDuration`.
- Modify `src/physics/disk.ts` — export the existing peak-flux finder as `peakFluxShape`.
- Modify `src/physics/color.ts` — add `blackbodyVisibleRGB` (un-normalised) and `relLuminance`;
  `blackbodyLinearSRGB` becomes a normalising wrapper (identical output).
- Modify `src/physics/lookups.ts` — replace `buildColorLUT` with `buildVisibleLUT`,
  `sampleVisibleLUT`, `visibleLuminance`, `lumNormFor` and the `VIS_*` constants.
- Create `src/physics/readouts.ts` — `computeReadouts` (T_peak, lumNorm, radii in metres, ISCO
  period, playback scale).
- Create `src/physics/presets.ts` — `PRESETS`, `CUSTOM_DEFAULT`, `Preset` type.
- Modify `src/render/uniforms.ts`, `src/render/raytrace.wgsl` — `lumNorm` uniform, log LUT
  indexing, visible-band `shadeDisk`.
- Modify `src/main.ts`, `index.html` — selector, caption, sliders, readouts, Custom switching.
- Modify `src/test/scenes.ts`, `src/test/shadow.browser.ts`, `scripts/bench.mjs` — new LUT and
  `lumNorm`.
- Modify `scripts/verify-gpu.mjs` — preset loop check. Create `scripts/shot-presets.mjs`.
- Tests: create `tests/units.test.ts`, `tests/readouts.test.ts`, `tests/presets.test.ts`; modify
  `tests/color.test.ts`, `tests/lookups.test.ts`, `tests/uniforms.test.ts`.
- Docs: `README.md`, `docs/ROADMAP.md`, spec status line.

---

### Task 1: Physical units and peak temperature

**Files:**
- Create: `src/physics/units.ts`
- Modify: `src/physics/disk.ts` (rename `peakFlux` → exported `peakFluxShape`)
- Test: `tests/units.test.ts`

**Interfaces:**
- Consumes: `pageThorneFluxShape`, `temperatureShape` (disk.ts); `efficiency`, `iscoRadius` (orbits.ts).
- Produces:
  - `peakFluxShape(a: number, prograde?: boolean): number` (disk.ts)
  - `G, C, SIGMA_SB, M_SUN, YEAR, L_EDD_PER_MSUN: number`
  - `gravRadius(mSun: number): number` (m), `gravTime(mSun: number): number` (s)
  - `eddingtonLuminosity(mSun: number): number` (W)
  - `mdotFromLambda(mSun: number, a: number, lambda: number): number` (kg/s)
  - `lambdaFromMdot(mSun: number, a: number, mdotSunPerYear: number): number`
  - `peakTemperature(mSun: number, a: number, lambda: number): number` (K)
  - `lambdaForPeakTemperature(mSun: number, a: number, tK: number): number`
  - `iscoPeriod(mSun: number, a: number): number` (s)
  - `formatLength(m: number): string`, `formatDuration(s: number): string`

- [ ] **Step 1: Write the failing test** — `tests/units.test.ts`

```ts
import { describe, it, expect } from "vitest";
import {
  gravRadius, gravTime, eddingtonLuminosity, mdotFromLambda, lambdaFromMdot, peakTemperature,
  lambdaForPeakTemperature, iscoPeriod, formatLength, formatDuration, M_SUN, YEAR,
} from "../src/physics/units";
import { efficiency } from "../src/physics/orbits";

describe("physical units", () => {
  it("gravitational radius and time of one solar mass", () => {
    expect(gravRadius(1)).toBeCloseTo(1476.6, 0);       // m
    expect(gravTime(1) * 1e6).toBeCloseTo(4.9255, 3);   // µs
  });
  it("Eddington luminosity and thin-disk efficiency", () => {
    expect(eddingtonLuminosity(1)).toBeCloseTo(1.2572e31, -27);
    expect(efficiency(0, true)).toBeCloseTo(0.0572, 4);
    expect(efficiency(0.998, true)).toBeGreaterThan(0.31);
    expect(efficiency(0.998, true)).toBeLessThan(0.33);
  });
  it("lambda <-> Mdot round trip", () => {
    const l = lambdaFromMdot(6.5e9, 0.9, 7.7e-4);
    expect(mdotFromLambda(6.5e9, 0.9, l) * YEAR / M_SUN).toBeCloseTo(7.7e-4, 10);
  });
  it("peak temperature: 10 M_sun, a = 0, Eddington -> ~7e6 K; scales as lambda^1/4 and M^-1/4", () => {
    const t = peakTemperature(10, 0, 1);
    expect(t).toBeGreaterThan(6.3e6); expect(t).toBeLessThan(7.7e6);
    expect(peakTemperature(10, 0, 1e-4) / t).toBeCloseTo(0.1, 6);
    expect(peakTemperature(1e5, 0, 1) / t).toBeCloseTo(0.1, 6);
  });
  it("lambdaForPeakTemperature inverts peakTemperature", () => {
    const l = lambdaForPeakTemperature(1e8, 0.6, 5800);
    expect(peakTemperature(1e8, 0.6, l)).toBeCloseTo(5800, 6);
  });
  it("extremes stay finite and positive (Review Focus 2)", () => {
    for (const [m, l] of [[1, 1e-10], [1e10, 1e-10], [1, 1], [1e10, 1]]) {
      const t = peakTemperature(m, 0.998, l);
      expect(Number.isFinite(t)).toBe(true); expect(t).toBeGreaterThan(0);
    }
    expect(peakTemperature(1e10, 0, 1e-10)).toBeLessThan(200); // ~124 K: visibly dark (Task 2 caps lumNorm)
  });
  it("ISCO orbital period: a = 0 is 2 pi 6^1.5 t_g", () => {
    expect(iscoPeriod(1, 0) / gravTime(1)).toBeCloseTo(2 * Math.PI * Math.pow(6, 1.5), 6);
  });
  it("formats lengths and durations across 20 orders of magnitude (Review Focus 5)", () => {
    expect(formatLength(512)).toBe("512 m");
    expect(formatLength(31_300)).toBe("31.3 km");
    expect(formatLength(9.6e12)).toBe("64.2 AU");
    expect(formatDuration(1.56e-3)).toBe("1.56 ms");
    expect(formatDuration(42)).toBe("42 s");
    expect(formatDuration(3 * 3600)).toBe("3 h");
    expect(formatDuration(8.9e5)).toBe("10.3 days");
    expect(formatDuration(3.2e9)).toBe("101 yr");
    expect(formatDuration(3e-7)).toBe("0.3 µs");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run tests/units.test.ts`
Expected: FAIL — `Failed to load url ../src/physics/units`.

- [ ] **Step 3: Export the peak-flux finder** — in `src/physics/disk.ts` replace

```ts
let _cache: { a: number; prograde: boolean; peak: number } | null = null;
function peakFlux(a: number, prograde: boolean): number {
```
with
```ts
let _cache: { a: number; prograde: boolean; peak: number } | null = null;
/** Maximum of pageThorneFluxShape over r (the disk's peak flux for Mdot = M = 1). Cached per (a, prograde). */
export function peakFluxShape(a: number, prograde = true): number {
```
and in `temperatureShape` replace `f / peakFlux(a, prograde)` with `f / peakFluxShape(a, prograde)`.

- [ ] **Step 4: Write `src/physics/units.ts`**

```ts
import { peakFluxShape } from "./disk";
import { efficiency, iscoRadius } from "./orbits";

// SI constants (CODATA 2018; IAU 2015 nominal solar mass; Julian year).
export const G = 6.6743e-11;
export const C = 2.99792458e8;
export const SIGMA_SB = 5.670374419e-8;
export const M_SUN = 1.98847e30;
export const YEAR = 3.15576e7;
const AU = 1.495978707e11;
/** Eddington luminosity per solar mass, electron scattering in ionised hydrogen (W). */
export const L_EDD_PER_MSUN = 1.2572e31;

/** r_g = G M / c^2 (m): the renderer's unit of length, "M". */
export const gravRadius = (mSun: number) => (G * mSun * M_SUN) / (C * C);
/** t_g = G M / c^3 (s): the renderer's unit of time. */
export const gravTime = (mSun: number) => (G * mSun * M_SUN) / (C * C * C);
export const eddingtonLuminosity = (mSun: number) => L_EDD_PER_MSUN * mSun;

/** Accretion rate (kg/s) of a thin disk radiating lambda L_Edd: Mdot = lambda L_Edd / (eta(a) c^2),
 *  eta = 1 - E_ISCO (spec 2026-10-01 §2.1). */
export const mdotFromLambda = (mSun: number, a: number, lambda: number) =>
  (lambda * eddingtonLuminosity(mSun)) / (efficiency(a, true) * C * C);
/** lambda for an accretion rate given in solar masses per year. */
export const lambdaFromMdot = (mSun: number, a: number, mdotSunPerYear: number) =>
  ((mdotSunPerYear * M_SUN) / YEAR) * efficiency(a, true) * C * C / eddingtonLuminosity(mSun);

/** Peak effective temperature (K) of the Novikov-Thorne disk, one face:
 *  F = Mdot c^2 shape_peak / (4 pi r_g^2), T = (F / sigma)^(1/4). pageThorneFluxShape is the flux
 *  for Mdot = M = 1 in geometric units, so this is the same profile temperatureShape normalises. */
export function peakTemperature(mSun: number, a: number, lambda: number): number {
  const rg = gravRadius(mSun);
  const flux = (mdotFromLambda(mSun, a, lambda) * C * C * peakFluxShape(a, true)) / (4 * Math.PI * rg * rg);
  return Math.pow(flux / SIGMA_SB, 0.25);
}
/** The lambda that gives a disk peak temperature tK (T_peak scales as lambda^(1/4)). */
export const lambdaForPeakTemperature = (mSun: number, a: number, tK: number) =>
  Math.pow(tK / peakTemperature(mSun, a, 1), 4);

/** Coordinate-time period of the prograde circular orbit at the ISCO: 2 pi (r^(3/2) + a) t_g. */
export const iscoPeriod = (mSun: number, a: number) =>
  2 * Math.PI * (Math.pow(iscoRadius(a, true), 1.5) + a) * gravTime(mSun);

const sig3 = (v: number) => v.toLocaleString("en-US", { maximumSignificantDigits: 3 });
export function formatLength(m: number): string {
  if (m < 1e3) return `${sig3(m)} m`;
  if (m < 0.1 * AU) return `${sig3(m / 1e3)} km`;
  return `${sig3(m / AU)} AU`;
}
export function formatDuration(s: number): string {
  const ladder: [number, string][] = [[1e-6, "µs"], [1e-3, "ms"], [1, "s"], [60, "min"], [3600, "h"], [86400, "days"], [YEAR, "yr"]];
  let unit = ladder[0];
  for (const step of ladder) if (s >= step[0]) unit = step;
  return `${sig3(s / unit[0])} ${unit[1]}`;
}
```

- [ ] **Step 5: Run the tests**

Run: `npx vitest run tests/units.test.ts`
Expected: PASS (8 tests). If a formatter case is off by rounding, fix the implementation, not the
expectation: the expectations are the agreed display.

- [ ] **Step 6: Run the whole suite** — `npm test` → all pass (disk.ts rename is internal).

- [ ] **Step 7: Commit**

```bash
git add src/physics/units.ts src/physics/disk.ts tests/units.test.ts
git commit -m "Add physical units: Eddington accretion, Novikov-Thorne peak temperature, formatters"
```

---

### Task 2: Visible-band radiance LUT

**Files:**
- Modify: `src/physics/color.ts`, `src/physics/lookups.ts`
- Test: `tests/color.test.ts`, `tests/lookups.test.ts`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `blackbodyVisibleRGB(T: number): [number, number, number]` (un-normalised, fixed arbitrary units)
  - `relLuminance(c: readonly [number, number, number]): number`
  - `VIS_TMIN = 100`, `VIS_TMAX = 1e9`, `VIS_LUT_N = 4096`, `VIS_TREF = 1e4`
  - `buildVisibleLUT(n?: number): Float32Array` (RGBA, log-spaced T, relative to VIS_TREF's luminance)
  - `sampleVisibleLUT(lut: Float32Array, T: number): [number, number, number]` (CPU twin of the shader)
  - `visibleLuminance(T: number): number` (relative to VIS_TREF), `lumNormFor(tPeak: number): number`
  - `buildColorLUT` stays for now (Task 3 moves its callers to `buildVisibleLUT` and deletes it),
    so the build is green after this task.

Why 4096 entries (spec said 1024): with linear interpolation in log T, the Wien side changes by
e^(hc/λkT · Δln T) per entry; at 1,500 K and 550 nm that is 7 % per entry at 4096 (error ≈ 0.06 %)
but 31 % at 1024 (≈ 1 %). 4096 × 16 B = 64 KB.

- [ ] **Step 1: Write the failing tests**

Append to `tests/color.test.ts`:
```ts
import { blackbodyVisibleRGB, relLuminance } from "../src/physics/color";

describe("visible-band radiance", () => {
  it("is the same colour as the normalised function, with luminance increasing in T", () => {
    for (const T of [2000, 5800, 3e4, 1e7]) {
      const v = blackbodyVisibleRGB(T), n = blackbodyLinearSRGB(T), L = relLuminance(v);
      for (let k = 0; k < 3; k++) expect(v[k] / L).toBeCloseTo(n[k], 10);
    }
    let prev = 0;
    for (let lt = 2; lt <= 9; lt += 0.05) {
      const L = relLuminance(blackbodyVisibleRGB(10 ** lt));
      expect(L).toBeGreaterThan(prev); prev = L;
    }
  });
});
```
(The existing `import { blackbodyLinearSRGB }` at the top stays; merge the two imports into one line.)

Replace the "color LUT" case in `tests/lookups.test.ts` (and its import) with:
```ts
import { buildTempLUT, buildVisibleLUT, sampleVisibleLUT, visibleLuminance, lumNormFor, VIS_LUT_N, VIS_TMIN } from "../src/physics/lookups";
import { blackbodyVisibleRGB, relLuminance } from "../src/physics/color";
```
```ts
  it("visible LUT: RGBA, finite, non-negative; log-T lookup within 0.5 % of direct evaluation", () => {
    const lut = buildVisibleLUT();
    expect(lut.length).toBe(VIS_LUT_N * 4);
    for (const x of lut) { expect(Number.isFinite(x)).toBe(true); expect(x).toBeGreaterThanOrEqual(0); }
    const ref = relLuminance(blackbodyVisibleRGB(1e4));
    for (let lt = Math.log10(1500); lt < 9; lt += 0.0371) { // off-grid sample points
      const T = 10 ** lt, got = relLuminance(sampleVisibleLUT(lut, T)), want = relLuminance(blackbodyVisibleRGB(T)) / ref;
      expect(Math.abs(got / want - 1)).toBeLessThan(5e-3);
    }
  });
  it("lumNorm normalises the peak temperature to luminance 1; capped at 1e30 for cold disks", () => {
    for (const T of [4100, 3e4, 6e6]) expect(lumNormFor(T) * visibleLuminance(T)).toBeCloseTo(1, 10);
    // A 124 K disk (Mass 1e10, Accretion 1e-10) has visible luminance ~1e-90: uncapped, lumNorm would
    // be Infinity in f32 and Infinity * 0 = NaN on the GPU (Review Focus 2).
    expect(lumNormFor(124)).toBe(1e30);
    expect(lumNormFor(VIS_TMIN / 2)).toBe(1e30);
    expect(lumNormFor(400)).toBeLessThan(1e30);
  });
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/color.test.ts tests/lookups.test.ts`
Expected: FAIL — `blackbodyVisibleRGB` / `buildVisibleLUT` not exported.

- [ ] **Step 3: Implement in `src/physics/color.ts`** — replace the body of
`blackbodyLinearSRGB` by splitting it:

```ts
/** Linear-sRGB radiance of a blackbody at T (K): the CIE colour-matching integral of the Planck
 *  spectrum over 360-830 nm, NOT normalised (fixed arbitrary units). This is the colour AND the
 *  brightness a camera records; a blackbody seen with frequency shift g is a blackbody at g T. */
export function blackbodyVisibleRGB(T: number): [number, number, number] {
  let X = 0, Y = 0, Z = 0;
  for (let nm = 360; nm <= 830; nm += 5) {
    const p = planck(nm * 1e-9, T);
    X += p * cieX(nm); Y += p * cieY(nm); Z += p * cieZ(nm);
  }
  // XYZ -> linear sRGB (IEC 61966-2-1, D65)
  const r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z;
  const g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z;
  const b = 0.0557 * X - 0.2040 * Y + 1.0570 * Z;
  return [Math.max(0, r), Math.max(0, g), Math.max(0, b)];
}
export const relLuminance = (c: readonly [number, number, number]) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];

/** Linear sRGB color of a blackbody at temperature T (K), chromaticity-preserving, luminance = 1. */
export function blackbodyLinearSRGB(T: number): [number, number, number] {
  const c = blackbodyVisibleRGB(T);
  // Normalize by relative luminance, NOT by max channel (see git history for why). Channels may exceed 1.
  const lum = relLuminance(c) || 1;
  return [c[0] / lum, c[1] / lum, c[2] / lum];
}
```
Keep the existing comment block about max-channel normalisation above `const lum` (shortened as
shown is acceptable). `planck` returns 0 when `exp` overflows (T below ~40 K), which is harmless.

- [ ] **Step 4: Implement in `src/physics/lookups.ts`** — keep `buildColorLUT` for now, extend the
colour import to `import { blackbodyLinearSRGB, blackbodyVisibleRGB, relLuminance } from "./color";`
and add:

```ts
/** Visible-radiance LUT range and size (spec 2026-10-01 §2.3; 4096 entries: see plan Task 2). */
export const VIS_TMIN = 100, VIS_TMAX = 1e9, VIS_LUT_N = 4096, VIS_TREF = 1e4;
const LREF = relLuminance(blackbodyVisibleRGB(VIS_TREF));

/** Visible luminance of a blackbody at T relative to one at VIS_TREF. */
export const visibleLuminance = (T: number) => relLuminance(blackbodyVisibleRGB(T)) / LREF;
/** Cap for `lumNorm`: f32-safe with headroom (LUT entries reach ~1e5, exposure 2^4). A disk whose
 *  visible luminance is below 1e-30 of a 1e4 K one (T_peak below ~365 K) renders dark, not NaN. */
export const LUM_NORM_MAX = 1e30;
/** The `lumNorm` uniform: 1 / visible luminance of the disk's rest-frame peak temperature, so the
 *  exposure slider means the same for every preset. */
export const lumNormFor = (tPeak: number) => Math.min(LUM_NORM_MAX, 1 / visibleLuminance(Math.max(tPeak, VIS_TMIN)));

/** Visible-band linear-sRGB radiance (RGBA, A = 1) at T log-spaced over [VIS_TMIN, VIS_TMAX],
 *  relative to VIS_TREF's luminance so f32 holds it (Wien-tail entries underflow towards 0). */
export function buildVisibleLUT(n = VIS_LUT_N): Float32Array {
  const out = new Float32Array(n * 4);
  const l0 = Math.log(VIS_TMIN), l1 = Math.log(VIS_TMAX);
  for (let i = 0; i < n; i++) {
    const c = blackbodyVisibleRGB(Math.exp(l0 + ((l1 - l0) * i) / (n - 1)));
    out[i * 4] = c[0] / LREF; out[i * 4 + 1] = c[1] / LREF; out[i * 4 + 2] = c[2] / LREF; out[i * 4 + 3] = 1;
  }
  return out;
}
/** CPU twin of sampleColor in raytrace.wgsl: log-T index, clamped, linear interpolation. */
export function sampleVisibleLUT(lut: Float32Array, T: number): [number, number, number] {
  const n = lut.length / 4, l0 = Math.log(VIS_TMIN), l1 = Math.log(VIS_TMAX);
  const u = Math.min(1, Math.max(0, (Math.log(T) - l0) / (l1 - l0))) * (n - 1);
  const i0 = Math.floor(u), i1 = Math.min(i0 + 1, n - 1), f = u - i0;
  const at = (i: number, k: number) => lut[i * 4 + k];
  return [0, 1, 2].map((k) => at(i0, k) + (at(i1, k) - at(i0, k)) * f) as [number, number, number];
}
```

- [ ] **Step 5: Run the tests** — `npx vitest run tests/color.test.ts tests/lookups.test.ts` → PASS;
`npm test` → all pass; `npm run build` → clean.

- [ ] **Step 6: Commit**

```bash
git add src/physics/color.ts src/physics/lookups.ts tests/color.test.ts tests/lookups.test.ts
git commit -m "Add a log-spaced visible-band radiance LUT and its luminance normaliser"
```

---

### Task 3: Visible-band brightness in the shader

**Files:**
- Modify: `src/render/uniforms.ts`, `src/render/raytrace.wgsl`, `src/main.ts`, `src/test/scenes.ts`,
  `src/test/shadow.browser.ts`, `scripts/bench.mjs`, `src/test/golden.json` (re-recorded)
- Test: `tests/uniforms.test.ts`; GPU: `npm run verify:gpu`

**Interfaces:**
- Consumes: `buildVisibleLUT`, `lumNormFor`, `VIS_TMIN`, `VIS_TMAX` (Task 2).
- Produces: `UniformValues.lumNorm: number` (required), packed at float index 30 (byte 120);
  WGSL `U.lumNorm`; `shadeDisk` = `sampleColor(T_obs) * U.lumNorm * E`.

- [ ] **Step 1: Failing uniforms test** — in `tests/uniforms.test.ts` add `lumNorm: 2.5` to both
`UniformValues` literals and to the first test:

```ts
    expect(dv.getFloat32(120, true)).toBeCloseTo(2.5);  // lumNorm (index 30)
```
Run `npx vitest run tests/uniforms.test.ts` → FAIL (reads 0).

- [ ] **Step 2: Pack it** — `src/render/uniforms.ts`: add to the header comment
`//         + lumNorm (1, visible-band brightness)                 -> 23 floats`, add the field
```ts
  /** 1 / visible luminance of the disk's rest-frame peak temperature (lumNormFor). */
  lumNorm: number;
```
after `outW: number; outH: number;`, set `export const UNIFORM_FLOATS = 23, UNIFORM_UINTS = 8;`
(size stays 128: ceil(31 / 4) × 16), and in `packUniforms` add `f[30] = u.lumNorm;`.
Run the test → PASS.

- [ ] **Step 3: Shader** — `src/render/raytrace.wgsl`:
  - struct: `jitterMode: u32, setIndex: u32, rowStart: u32, rowEnd: u32,` → append `lumNorm: f32,`
    on a new line before `};`. (bloom.wgsl / present.wgsl keep their shorter structs: a uniform
    struct may be smaller than its buffer.)
  - replace `sampleColor`:
```wgsl
// Visible-band radiance LUT, log-spaced in T over [VIS_TMIN, VIS_TMAX] (lookups.ts), relative to a
// 1e4 K blackbody's luminance. Twin: sampleVisibleLUT in lookups.ts.
const VIS_LN_TMIN = 4.605170186;   // ln(100)
const VIS_LN_TMAX = 20.723265837;  // ln(1e9)
fn sampleColor(T_kelvin: f32) -> vec3<f32> {
  let n = arrayLength(&colorLUT);
  let u = clamp((log(max(T_kelvin, 1.0)) - VIS_LN_TMIN) / (VIS_LN_TMAX - VIS_LN_TMIN), 0.0, 1.0) * f32(n - 1u);
  let i0 = u32(floor(u)); let i1 = min(i0 + 1u, n - 1u);
  return mix(colorLUT[i0].rgb, colorLUT[i1].rgb, fract(u));
}
```
  - in `shadeDisk` replace `return sampleColor(Tobs) * pow(g * Tn, 4.0) * E;` with
```wgsl
  // Visible-band radiance of a blackbody at T_obs (I_nu / nu^3 is invariant, so a shifted blackbody
  // is a blackbody at g T): colour AND brightness a camera records, normalised so the disk's
  // rest-frame peak has luminance 1 (spec 2026-10-01 §2.3). Was the bolometric (g Tn)^4 law.
  return sampleColor(Tobs) * U.lumNorm * E;
```
  and update the `@binding(3)` comment to `// visible-band blackbody radiance (log T)`.

- [ ] **Step 4: Callers**
  - `src/physics/lookups.ts`: delete `buildColorLUT` and drop `blackbodyLinearSRGB` from its color
    import (no callers remain after the bullets below); `tests/lookups.test.ts` already does not use it.
  - `src/main.ts`: import `buildVisibleLUT, lumNormFor` instead of `buildColorLUT`; add after
    `const T_PEAK = 3.0e4;` → `const COLOR_LUT = buildVisibleLUT();` and in `rebuildLUTs` use
    `r.uploadLUTs(buildTempLUT(state.a, true, rIn, rOut, 512), COLOR_LUT);`; in the uniforms
    literal add `lumNorm: lumNormFor(T_PEAK),`. Replace the T_PEAK comment with: `// Peak disk
    temperature (K); replaced by the physical value from mass and accretion in Task 5.`
  - `src/test/scenes.ts`: import `buildVisibleLUT, lumNormFor`; upload `buildVisibleLUT()`; add
    `lumNorm: lumNormFor(3e4),` next to `Tpeak: 3e4`.
  - `src/test/shadow.browser.ts`: same LUT swap; add `lumNorm: lumNormFor(3.0e4),` beside `Tpeak`.
  - `scripts/bench.mjs`: it imports modules in-page; change its `buildColorLUT(1000, 40000, 256)`
    to `buildVisibleLUT()` (import from `/src/physics/lookups.ts`) and add `lumNorm: lumNormFor(3e4)`
    to its uniforms object.

- [ ] **Step 5: Local gates** — `npm test` → PASS; `npm run build` → clean.

- [ ] **Step 6: GPU gates and golden re-record** (dev server on :5173)

Run: `RECORD_GOLDEN=1 npm run verify:gpu` (PowerShell: `$env:RECORD_GOLDEN="1"; npm run verify:gpu`)
Expected: `• recorded src/test/golden.json`, then every line `✓ PASS`, including `?parity` (unchanged:
1.654e-4 over 55), `?shadow` (a lit-threshold radius; record the value — it may move slightly with
the brightness law; FAIL only if not structural), `?golden` (new hashes), `?cachecheck`, app check.
Then: `node scripts/probe-axis.mjs` and `node scripts/probe-scale.mjs` → all PASS. If probe-axis's
"dark" class (RGB sum < 80) now catches dimmer outer disk, raise nothing: report the numbers and
stop for a decision (the probe's thresholds were set between measured states).

- [ ] **Step 7: Visual check of the default view** — screenshot `/` after 5 s (verify:gpu writes
`render.png`); compare with the previous `render.png` from `main`. If the disk is clearly too dim
or blown out at exposure +1.6 EV, choose a new default exposure, set it in `index.html` (`#exp`
value and `#expv` text) and `main.ts` (`state.exposure`), and record old/new in the commit message.

- [ ] **Step 8: Commit**

```bash
git add src/physics/lookups.ts src/render/uniforms.ts src/render/raytrace.wgsl src/main.ts src/test/scenes.ts src/test/shadow.browser.ts scripts/bench.mjs src/test/golden.json tests/uniforms.test.ts index.html
git commit -m "Shade the disk by visible-band radiance at the observed temperature"
```

---

### Task 4: Presets and physical readouts (pure)

**Files:**
- Create: `src/physics/presets.ts`, `src/physics/readouts.ts`
- Test: `tests/presets.test.ts`, `tests/readouts.test.ts`

**Interfaces:**
- Consumes: Task 1 (`lambdaFromMdot`, `lambdaForPeakTemperature`, `peakTemperature`, `gravRadius`,
  `gravTime`, `iscoPeriod`), Task 2 (`lumNormFor`), orbits (`iscoRadius`, `photonOrbit`).
- Produces:
```ts
export interface Preset { id: string; name: string; massSun: number; a: number; inclDeg: number; lambda: number; jet: boolean; caption: string; }
export const PRESETS: readonly Preset[];
export const CUSTOM_DEFAULT: { massSun: number; lambda: number };
export interface Readouts { tPeakK: number; lumNorm: number; horizonM: number; iscoM: number; photonM: number; iscoPeriodS: number; realSecondsPerScreenSecond: number; }
export function computeReadouts(p: { massSun: number; a: number; lambda: number; timeScale: number }, speedMPerSecond: number): Readouts;
```

- [ ] **Step 1: Failing tests** — `tests/presets.test.ts`

```ts
import { describe, it, expect } from "vitest";
import { PRESETS, CUSTOM_DEFAULT } from "../src/physics/presets";
import { peakTemperature, lambdaFromMdot } from "../src/physics/units";

const T_TABLE: Record<string, number> = { m87: 4.1e3, sgra: 1.1e4, cygx1: 6.1e6, grs1915: 1.1e7, gargantua: 5.8e3 };

describe("presets", () => {
  it("covers the five agreed objects with captions", () => {
    expect(PRESETS.map((p) => p.id)).toEqual(["m87", "sgra", "cygx1", "grs1915", "gargantua"]);
    for (const p of PRESETS) expect(p.caption.length).toBeGreaterThan(80);
  });
  it("every value lies inside the slider ranges", () => {
    for (const p of PRESETS) {
      expect(p.a).toBeGreaterThanOrEqual(0); expect(p.a).toBeLessThanOrEqual(0.998);
      expect(p.inclDeg).toBeGreaterThanOrEqual(1); expect(p.inclDeg).toBeLessThanOrEqual(89);
      expect(Math.log10(p.massSun)).toBeGreaterThanOrEqual(0); expect(Math.log10(p.massSun)).toBeLessThanOrEqual(10);
      expect(Math.log10(p.lambda)).toBeGreaterThanOrEqual(-10); expect(Math.log10(p.lambda)).toBeLessThanOrEqual(0);
    }
  });
  it("peak temperatures match the spec table within 10 % (Gargantua within 5 % of 5,800 K)", () => {
    for (const p of PRESETS) {
      const t = peakTemperature(p.massSun, p.a, p.lambda), want = T_TABLE[p.id];
      expect(Math.abs(t / want - 1)).toBeLessThan(p.id === "gargantua" ? 0.05 : 0.1);
    }
  });
  it("M87* and Sgr A* lambda are derived from the cited accretion rates", () => {
    const m87 = PRESETS.find((p) => p.id === "m87")!, sgra = PRESETS.find((p) => p.id === "sgra")!;
    expect(m87.lambda).toBeCloseTo(lambdaFromMdot(6.5e9, 0.9, 7.7e-4), 12);
    expect(sgra.lambda).toBeCloseTo(lambdaFromMdot(4.3e6, 0.94, 1e-8), 15);
  });
  it("the Custom default reproduces today's 30,000 K at a = 0.9", () => {
    expect(peakTemperature(CUSTOM_DEFAULT.massSun, 0.9, CUSTOM_DEFAULT.lambda)).toBeCloseTo(3e4, 3);
  });
});
```

`tests/readouts.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { computeReadouts } from "../src/physics/readouts";
import { gravRadius, gravTime, peakTemperature } from "../src/physics/units";
import { lumNormFor } from "../src/physics/lookups";

describe("physical readouts", () => {
  it("radii in metres, ISCO period, playback scale", () => {
    const r = computeReadouts({ massSun: 10, a: 0, lambda: 0.1, timeScale: 2 }, 20);
    expect(r.horizonM).toBeCloseTo(2 * gravRadius(10), 6);
    expect(r.iscoM).toBeCloseTo(6 * gravRadius(10), 4);
    expect(r.photonM).toBeCloseTo(3 * gravRadius(10), 4);
    expect(r.iscoPeriodS / gravTime(10)).toBeCloseTo(2 * Math.PI * Math.pow(6, 1.5), 4);
    expect(r.realSecondsPerScreenSecond).toBeCloseTo(40 * gravTime(10), 12);
    expect(r.lumNorm).toBeCloseTo(lumNormFor(r.tPeakK), 12);
  });
  it("changing spin at fixed mass and accretion changes T_peak (Review Focus 3)", () => {
    const lo = computeReadouts({ massSun: 1e8, a: 0, lambda: 1e-3, timeScale: 1 }, 20);
    const hi = computeReadouts({ massSun: 1e8, a: 0.998, lambda: 1e-3, timeScale: 1 }, 20);
    expect(lo.tPeakK).toBeCloseTo(peakTemperature(1e8, 0, 1e-3), 6);
    expect(hi.tPeakK).toBeGreaterThan(lo.tPeakK);
    expect(hi.iscoM).toBeLessThan(lo.iscoM);
  });
});
```

- [ ] **Step 2: Run to verify failure** — `npx vitest run tests/presets.test.ts tests/readouts.test.ts` → FAIL (modules missing).

- [ ] **Step 3: Write `src/physics/readouts.ts`**

```ts
import { iscoRadius, photonOrbit } from "./orbits";
import { gravRadius, gravTime, iscoPeriod, peakTemperature } from "./units";
import { lumNormFor } from "./lookups";

export interface Readouts {
  tPeakK: number; lumNorm: number;
  horizonM: number; iscoM: number; photonM: number;   // metres
  iscoPeriodS: number;                                // seconds
  realSecondsPerScreenSecond: number;                 // playback scale
}
/** Everything the panel shows in physical units, from the physical state. speedMPerSecond is the
 *  animation's coordinate time (in M) per real second at timeScale 1 (main.ts SPEED). */
export function computeReadouts(p: { massSun: number; a: number; lambda: number; timeScale: number }, speedMPerSecond: number): Readouts {
  const rg = gravRadius(p.massSun), tPeakK = peakTemperature(p.massSun, p.a, p.lambda);
  return {
    tPeakK, lumNorm: lumNormFor(tPeakK),
    horizonM: (1 + Math.sqrt(Math.max(0, 1 - p.a * p.a))) * rg,
    iscoM: iscoRadius(p.a, true) * rg,
    photonM: photonOrbit(p.a, true) * rg,
    iscoPeriodS: iscoPeriod(p.massSun, p.a),
    realSecondsPerScreenSecond: speedMPerSecond * p.timeScale * gravTime(p.massSun),
  };
}
```

- [ ] **Step 4: Write `src/physics/presets.ts`** (captions verbatim from the spec §3)

```ts
import { lambdaFromMdot, lambdaForPeakTemperature } from "./units";

export interface Preset {
  id: string; name: string;
  massSun: number; a: number; inclDeg: number;
  /** Accretion: fraction of the Eddington luminosity (units.ts mdotFromLambda). */
  lambda: number;
  jet: boolean;
  caption: string;
}

/** Real-object presets (spec 2026-10-01 §3). Sources in each caption. */
export const PRESETS: readonly Preset[] = [
  {
    id: "m87", name: "M87*", massSun: 6.5e9, a: 0.9, inclDeg: 17,
    // Mdot (3-20)e-4 M_sun/yr (EHT M87* Paper VIII); log-midpoint 7.7e-4.
    lambda: lambdaFromMdot(6.5e9, 0.9, 7.7e-4), jet: true,
    caption: "EHT 2019: 6.5 billion solar masses (Paper VI); seen 17° from its jet (Mertens et al. 2016, Walker et al. 2018); accretion (3–20)×10⁻⁴ M☉/yr (Paper VIII). Spin is not measured: 0.9 is a common model value. Caveat: M87*'s real flow is hot, thick and radio-bright, not the thin disk drawn here; the colour is what a thin disk at this accretion rate would emit.",
  },
  {
    id: "sgra", name: "Sagittarius A*", massSun: 4.3e6, a: 0.94, inclDeg: 30,
    // Mdot ~1e-8 M_sun/yr (radiatively inefficient flow; L ~ 2e-9 L_Edd).
    lambda: lambdaFromMdot(4.3e6, 0.94, 1e-8), jet: false,
    caption: "Our galaxy's centre: 4.3 million solar masses (GRAVITY). Spin 0.94 and a 30° view are EHT 2022 Paper V's best-bet model values, preferences rather than measurements; accretion ~10⁻⁸ M☉/yr. Caveat: like M87*, the real flow is not a thin disk.",
  },
  {
    id: "cygx1", name: "Cygnus X-1", massSun: 21.2, a: 0.998, inclDeg: 27, lambda: 0.02, jet: true,
    caption: "The first known stellar black hole: 21.2 ± 2.2 solar masses and spin above 0.9985 (Miller-Jones et al. 2021), seen at ~27° (Orosz et al. 2011), accreting at ~2 % of Eddington. Spin sits at the slider's 0.998 cap (the Thorne limit); the measured value is higher. The real disk peaks in X-rays; you see its blue-white visible tail.",
  },
  {
    id: "grs1915", name: "GRS 1915+105", massSun: 12.4, a: 0.98, inclDeg: 60, lambda: 0.3, jet: true,
    caption: "A microquasar with a powerful jet: 12.4 solar masses, jet seen at 60° ± 5°, spin ~0.98 (Reid et al. 2014), accreting at ~30 % of Eddington. The real disk peaks in X-rays; you see its blue-white visible tail.",
  },
  {
    id: "gargantua", name: "Gargantua (Interstellar)", massSun: 1e8, a: 0.6, inclDeg: 85,
    // "Anemic" disk about as hot as the Sun's surface (Thorne, The Science of Interstellar).
    lambda: lambdaForPeakTemperature(1e8, 0.6, 5800), jet: false,
    caption: "Fictional. 100 million solar masses (Thorne, The Science of Interstellar); the film's disk was rendered at spin 0.6 for the visuals (James, von Tunzelmann, Franklin & Thorne 2015), an 'anemic' disk about as hot as the Sun's surface. The 85° view is our choice to resemble the film. The film removed Doppler colour and brightness shifts; this render keeps them, which is why one side is brighter here.",
  },
];

/** Custom mode's starting mass and accretion: today's 30,000 K disk at a = 0.9. */
export const CUSTOM_DEFAULT = { massSun: 1e8, lambda: lambdaForPeakTemperature(1e8, 0.9, 3e4) };
```

- [ ] **Step 5: Run the tests** — `npx vitest run tests/presets.test.ts tests/readouts.test.ts` → PASS;
then `npm test` → all pass.

- [ ] **Step 6: Commit**

```bash
git add src/physics/presets.ts src/physics/readouts.ts tests/presets.test.ts tests/readouts.test.ts
git commit -m "Add real-object presets and physical readouts"
```

---

### Task 5: Panel — selector, sliders, readouts, Custom switching

**Files:**
- Modify: `index.html`, `src/main.ts`, `scripts/verify-gpu.mjs`
- Create: `scripts/shot-presets.mjs`

**Interfaces:**
- Consumes: `PRESETS`, `CUSTOM_DEFAULT`, `Preset` (Task 4); `computeReadouts`, `Readouts` (Task 4);
  `formatLength`, `formatDuration` (Task 1).
- Produces: DOM ids `preset` (select; option values `custom` + preset ids), `pcap` (caption),
  `mass`/`massv`, `acc`/`accv`, `tpk`, `pisco`, `tscale`; `state.massSun`, `state.lambda`.

- [ ] **Step 1: HTML** — in `index.html`, CSS block: add
```css
  #preset{width:100%;margin-top:6px;padding:7px 8px;background:transparent;color:var(--ink);
    border:1px solid var(--hair);border-radius:8px;font:inherit;font-size:12px}
  #preset option{background:#0b0d12;color:var(--ink)}
  #pcap{margin:9px 0 0;font-size:10.5px;line-height:1.55;color:var(--muted)}
```
Directly after the first `<div class="rule"></div>` (before the Spin control) add:
```html
    <div class="ctrl">
      <div class="row"><label>Object</label></div>
      <select id="preset"><option value="custom">Custom</option></select>
      <p id="pcap" hidden></p>
    </div>
```
After the Inclination control add:
```html
    <div class="ctrl">
      <div class="row"><label>Mass</label><span class="val"><b id="massv">1.00e8</b> M☉</span></div>
      <input id="mass" type="range" min="0" max="10" step="0.01" value="8">
    </div>
    <div class="ctrl">
      <div class="row"><label>Accretion</label><span class="val"><b id="accv">3.7e-4</b> L<sub>Edd</sub></span></div>
      <input id="acc" type="range" min="-10" max="0" step="0.01" value="-3.44">
    </div>
```
In `#readout`, after the Photon orbit row add:
```html
      <div class="ro"><span>Disk peak</span><b id="tpk">—</b></div>
      <div class="ro"><span>ISCO orbit</span><b id="pisco">—</b></div>
      <div class="ro wide"><span>1 s on screen</span><b id="tscale">—</b></div>
```

- [ ] **Step 2: State and helpers in `src/main.ts`**
  - imports: `import { PRESETS, CUSTOM_DEFAULT, type Preset } from "./physics/presets";`,
    `import { computeReadouts, type Readouts } from "./physics/readouts";`,
    `import { formatLength, formatDuration } from "./physics/units";`
  - `state` gains `massSun: CUSTOM_DEFAULT.massSun, lambda: CUSTOM_DEFAULT.lambda`.
  - delete `const T_PEAK = 3.0e4;` and its comment; add after `const COLOR_LUT = ...`:
```ts
  let phys: Readouts = computeReadouts(state, SPEED);
  const refreshPhysics = () => { phys = computeReadouts(state, SPEED); };
```
    (`state` has `massSun`, `a`, `lambda`, `timeScale`, so it satisfies the parameter type.)
  - uniforms literal: `Tpeak: phys.tPeakK, ... lumNorm: phys.lumNorm,`.
  - `refreshReadouts()` becomes:
```ts
  function refreshReadouts() {
    const rh = 1 + Math.sqrt(Math.max(0, 1 - state.a * state.a));
    rhEl.innerHTML = `${inM(rh)} <i>${formatLength(phys.horizonM)}</i>`;
    riscoEl.innerHTML = `${inM(iscoRadius(state.a, true))} <i>${formatLength(phys.iscoM)}</i>`;
    rphEl.innerHTML = `${inM(photonOrbit(state.a, true))} <i>${formatLength(phys.photonM)}</i>`;
    tpkEl.textContent = `${Math.round(phys.tPeakK).toLocaleString("en-US", { maximumSignificantDigits: 3 })} K`;
    piscoEl.textContent = formatDuration(phys.iscoPeriodS);
    tscaleEl.textContent = state.timeScale > 0 ? `≈ ${formatDuration(phys.realSecondsPerScreenSecond)}` : "—";
  }
```
    with `const tpkEl = $("tpk"), piscoEl = $("pisco"), tscaleEl = $("tscale");` declared next to `rhEl`.

- [ ] **Step 3: Controls in `src/main.ts`**

```ts
  const presetSel = $("preset") as HTMLSelectElement, pcap = $("pcap");
  for (const p of PRESETS) presetSel.add(new Option(p.name, p.id));
  const mass = $("mass") as HTMLInputElement, acc = $("acc") as HTMLInputElement, massv = $("massv"), accv = $("accv");
  const showMass = () => { massv.textContent = state.massSun.toExponential(2); };
  const showAcc = () => { accv.textContent = state.lambda.toExponential(1); };
  /** Any change to spin, inclination, mass or accretion leaves the preset (Review Focus 4: drag too). */
  const markCustom = () => { if (presetSel.value !== "custom") { presetSel.value = "custom"; pcap.hidden = true; } };
  const physicsChanged = () => { refreshPhysics(); refreshReadouts(); reset(); };

  mass.addEventListener("input", () => { state.massSun = 10 ** +mass.value; showMass(); markCustom(); physicsChanged(); });
  acc.addEventListener("input", () => { state.lambda = 10 ** +acc.value; showAcc(); markCustom(); physicsChanged(); });

  function applyPreset(p: Preset) {
    state.a = p.a; state.incl = p.inclDeg; state.massSun = p.massSun; state.lambda = p.lambda;
    state.jetStrength = p.jet ? 1 : 0;
    spin.value = String(p.a); spinv.textContent = p.a.toFixed(3);
    incl.value = String(p.inclDeg); inclv.textContent = String(p.inclDeg);
    mass.value = String(Math.log10(p.massSun)); showMass();
    acc.value = String(Math.log10(p.lambda)); showAcc();
    jet.value = String(state.jetStrength); jetv.textContent = state.jetStrength.toFixed(1);
    pcap.textContent = p.caption; pcap.hidden = false;
    rebuildLUTs(); physicsChanged();
  }
  presetSel.addEventListener("change", () => {
    const p = PRESETS.find((q) => q.id === presetSel.value);
    if (p) applyPreset(p); else pcap.hidden = true;
  });
```
  Place this block AFTER the `jet`/`jetv` declarations (applyPreset uses them). Then:
  - spin handler: add `markCustom();` and replace `rebuildLUTs(); refreshReadouts(); reset();` with
    `rebuildLUTs(); markCustom(); physicsChanged();` (spin changes T_peak, Review Focus 3).
  - incl handler: add `markCustom();`.
  - drag handler: inside `if (Math.round(next) !== state.incl) { ... }` add `markCustom();`.
  - ts (Motion) handler: add `refreshPhysics(); refreshReadouts();` (playback-scale readout).
  - initial: `showMass(); showAcc();` before the first `refreshReadouts();`.

- [ ] **Step 4: Build and unit tests** — `npm test` and `npm run build` → PASS / clean.

- [ ] **Step 5: verify-gpu preset loop** — in `scripts/verify-gpu.mjs`, before the final
`process.exit`, add (after the cache-in-the-app check, reusing `waitMode`, `meanBrightness`,
`diagSince`):

```js
// Presets (spec 2026-10-01): each one must leave the app lit, back in `cached` mode, warning-free.
// Spin/inclination changes rebuild the cache; mass/accretion are shading-only (Review Focus 1).
const dPre = diags.length;
const presetIds = await page.evaluate(() => [...document.getElementById("preset").options].map((o) => o.value).filter((v) => v !== "custom"));
const preSteps = [];
for (const id of presetIds) {
  await page.selectOption("#preset", id);
  await page.waitForTimeout(300); // let the new geometry key reach the scheduler (else "cached" is the old preset's)
  const cached = await waitMode("cached");
  await page.waitForTimeout(1500);
  const lit = await meanBrightness();
  preSteps.push(`${id} ${cached && lit > 2 ? "ok" : "FAILED"} (lit ${lit.toFixed(1)})`);
  if (!cached || !(lit > 2)) failed = true;
}
// Dragging the canvas tilts the camera: that must leave the preset for Custom (Review Focus 4).
await page.mouse.move(700, 300); await page.mouse.down(); await page.mouse.move(700, 360, { steps: 6 }); await page.mouse.up();
const afterDrag = await page.evaluate(() => document.getElementById("preset").value);
preSteps.push(`drag -> ${afterDrag}`);
if (afterDrag !== "custom") failed = true;
const preDiag = diagSince(dPre);
if (preDiag) failed = true;
console.log(`${preSteps.every((s) => !s.includes("FAILED")) && afterDrag === "custom" && !preDiag ? "✓ PASS" : "✗ FAIL"}  presets: ${preSteps.join(", ")}${preDiag}`);
```

- [ ] **Step 6: Preset screenshots** — create `scripts/shot-presets.mjs`:

```js
// Writes preset-<id>.png for every Object preset (dev server on :5173), for visual review.
import { chromium } from "playwright-core";
const BASE = process.env.BASE || "http://localhost:5173";
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 760 } });
await page.goto(BASE + "/", { waitUntil: "load" });
await page.waitForFunction(() => /^\d+$/.test(document.getElementById("spp").textContent), null, { timeout: 30000 });
const ids = await page.evaluate(() => [...document.getElementById("preset").options].map((o) => o.value));
for (const id of ids) {
  await page.selectOption("#preset", id);
  await page.waitForTimeout(7000);
  await page.screenshot({ path: `preset-${id}.png` });
  console.log(`• preset-${id}.png`);
}
await browser.close();
```
Run `node scripts/shot-presets.mjs`; look at every image. Expected by eye: M87\* orange, Sgr A\*
white, Cygnus X-1 and GRS 1915+105 blue-white, Gargantua yellow-white and near edge-on, Custom like
the default view; captions visible in the panel. If a preset is unreadably dark or blown out at
the default exposure, report it with the image before changing anything (exposure is the user's
control; presets do not set it).

- [ ] **Step 7: Full gates** — `npm run verify:gpu` (all PASS incl. the new `presets` line),
`node scripts/probe-axis.mjs`, `node scripts/probe-scale.mjs` → PASS.

- [ ] **Step 8: Commit**

```bash
git add index.html src/main.ts scripts/verify-gpu.mjs scripts/shot-presets.mjs
git commit -m "Add the Object selector, Mass and Accretion sliders and physical readouts"
```

---

### Task 6: Documentation

**Files:**
- Modify: `README.md`, `docs/ROADMAP.md`, `docs/specs/2026-10-01-real-object-presets-design.md`

- [ ] **Step 1: README** — in `## Features` add two bullets:
  - "Real-object presets — M87\*, Sagittarius A\*, Cygnus X-1, GRS 1915+105 and Interstellar's
    Gargantua with published mass, spin and viewing angle, physically derived disk temperature, and
    a caption with sources and caveats"
  - "Physical units — Mass and Accretion (fraction of Eddington) sliders; horizon, ISCO and photon
    orbit in km/AU, ISCO orbital period, playback time scale"
  and replace the "Blackbody emission → CIE XYZ → linear sRGB" bullet's tail with "…— colour and
  brightness are the blackbody's visible-band radiance at the observed temperature".
  In `## Status` add a paragraph **Real-object presets (2026-10-01)** stating: the T_peak formula
  and its 10 M_sun check (7.0e6 K); the visible-band brightness law replacing (gTn)⁴ and why (a
  shifted blackbody is a blackbody at gT); the LUT (100 K–1e9 K, 4096 log entries, measured
  interpolation error); the preset table with computed T_peak; the golden re-record and new hashes;
  the `?shadow` value after the change; any default-exposure retune; the gate results.
- [ ] **Step 2: ROADMAP** — tick the presets item (`- [x] ... (2026-10-01)`).
- [ ] **Step 3: Spec status** — `**Status:** implemented on feat/real-object-presets`.
- [ ] **Step 4: Commit**

```bash
git add README.md docs/ROADMAP.md docs/specs/2026-10-01-real-object-presets-design.md
git commit -m "Document real-object presets and the visible-band brightness law"
```
