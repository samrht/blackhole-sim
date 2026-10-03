# Horizon-Flux Statistics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive the jet with the absolute horizon flux Φ, calibrated per spin to Narayan et al. 2022's windowed
modulation index, as eruptions (2σ deep) times fast clipped flicker.

**Architecture:** `src/physics/flux-history.ts` keeps the eruption sawtooth and adds a 1-D Gaussian-lattice flicker
(50 M cells, clipped at ±3) and a per-spin table (a = 0, 0.3, 0.6, 0.9) of (δ̄, ε, ⟨d⟩, ⟨d²⟩, ⟨d³⟩) interpolated in a.
`fluxRatio`/`fluxMoment` gain the spin; `jetShape` and `meanInjection` pass it. `emission-shared.wgsl`'s
`fluxRatioJ` mirrors it (spin from `jetShapeJ`, which already receives `a`).

**Tech Stack:** TypeScript, WebGPU/WGSL, Vitest, vite-node, Playwright (GPU gates).

**Spec:** `docs/specs/2026-10-03-flux-statistics-design.md`

## Global Constraints

- Flicker: cell 50 M, clip 3, salt `0x464e`, node hash row `0x0f1c`, node = `gaussNode(i, 0x0f1c, 0, 0x464e)`
  (emission.ts; WGSL `gaussT(i, 0x0f1cu, 0, 0x464eu)`), smoothstep weights renormalised by √(w₀² + w₁²).
- c₂ = E[clip(n)²] = 0.995007 (exact for a clipped standard normal).
- Calibration table (computed while planning with the project hash; Task 1's script must reproduce it):

  | a | dbar | eps | d1 | d2 | d3 |
  |---|---|---|---|---|---|
  | 0 | 0.1455 | 0.05499 | 0.073273 | 0.007298 | 0.000833 |
  | 0.3 | 0.1682 | 0.06295 | 0.084751 | 0.009761 | 0.001288 |
  | 0.6 | 0.1910 | 0.07072 | 0.096304 | 0.012600 | 0.001889 |
  | 0.9 | 0.2138 | 0.07830 | 0.107885 | 0.015808 | 0.002653 |

  Targets: 0.067 + 0.031 a / 0.9 (a clamped to [0, 0.9]). Measured M₁₀₀₀ equals the target at each spin;
  M₅₀₀/M₂₀₀₀ = 0.828 / 0.823 / 0.818 / 0.813; δ̄ = 2σ/μ exactly.
- Envelope: f_max(a = 0.9, s = 1.4) = (1 + 1.4·0.0783·3)/(1 − 1.4·0.107885) = 1.5653 → 1.2√f = 1.501 ≤ 1.51
  (`JET_ENV_Q` unchanged).
- Sampling for the calibration statistics: midpoints every 5 M over [0, 10⁶) M; moments of d every 1 M.
- No Co-Authored-By trailer in commits (user rule). Merge, push and deploy when done (user request).

## Review Focus

1. **Interpolation at the table edges.** a = 0.998 (Gargantua, M87 presets at 0.9) must use the a = 0.9 row
   exactly; a between rows must interpolate every column linearly — pinned in Task 1.
2. **Long sessions.** The flicker's 50 M lattice index at epochs up to 2048 × 8000 must match the CPU — Task 2
   parity cases.
3. **Slider at 0 and 1.4.** s = 0 gives f ≡ 1 (both twins); s = 1.4 stays inside the envelope at every spin — Task 1.
4. **Energy budget per spin.** q0 × ⟨P⟩ = η P_BZ at spins 0.3 and 0.9 and the moments match brute force — Task 1.
5. **Spin 0 jet.** q0 = 0 at a = 0, so the jet is off there; the table row still must be finite — Task 1.

---

### Task 1: Flux generator v2 and all CPU consumers

**Files:**
- Modify: `src/physics/emission.ts` (export `gaussNode`)
- Modify: `src/physics/flux-history.ts`, `scripts/calibrate-flux.ts`
- Modify: `src/physics/jet.ts` (`jetShape` passes `a` to `fluxRatio`)
- Modify: `src/physics/synchrotron.ts` (`meanInjection` gains `a`; `jetUniforms` passes it)
- Test: `tests/flux-history.test.ts` (rewrite), `tests/jet.test.ts`, `tests/synchrotron.test.ts`

**Interfaces (produces):**
- `FLUX` keeps `T, jitter, tauD, spread, floor, sMax, saltT, saltD` and adds `flickerCell: 50, flickerClip: 3,
  saltN: 0x464e, c2: 0.995007`; drops `target, dbar, d1, d2, d3`.
- `FLUX_SPIN: readonly { a: number; dbar: number; eps: number; d1: number; d2: number; d3: number }[]` (table above).
- `fluxTarget(a: number): number` = 0.067 + 0.031·clamp(a, 0, 0.9)/0.9.
- `fluxParams(a: number): { dbar; eps; d1; d2; d3 }` (linear in clamp(a, 0, 0.9)).
- `fluxFlicker(t: number): number` (clipped, unit-variance before clipping).
- `fluxDeficit(t, dbar)` / `fluxDeficitLocal(k, loc, dbar)` — `dbar` now REQUIRED.
- `fluxRatio(t: number, s: number, a: number): number`; `fluxMoment(n: 2 | 3, s: number, a: number): number`.
- `fluxSeries(dbar, eps, span = 1e6, dt = 5): number[]`; `modulationIndex(series, window, dt = 5): number`;
  `seriesSigma(series): number` (σ/μ of the whole series).
- `meanInjection(a, b0, rgCm, jetLength, g280, fluxVar)` unchanged signature (it already has `a`), now uses
  `fluxMoment(n, fluxVar, a)`.

- [ ] **Step 1: Write the failing tests.** Replace `tests/flux-history.test.ts` with:

```ts
import { describe, it, expect } from "vitest";
import { FLUX, FLUX_SPIN, fluxHash, eruptionTime, eruptionDepth, fluxDeficit, fluxFlicker, fluxParams, fluxTarget,
  fluxRatio, fluxMoment, fluxSeries, modulationIndex, seriesSigma } from "../src/physics/flux-history";

describe("flux history v2: eruptions x flicker, per spin (spec 2026-10-03 flux statistics)", () => {
  it("eruptions recur every 1500 M on average, every gap in [1000, 2000] M", () => {
    const gaps: number[] = [];
    for (let k = 0; k < 700; k++) gaps.push(eruptionTime(k + 1) - eruptionTime(k));
    expect(Math.abs(gaps.reduce((x, y) => x + y) / gaps.length - 1500)).toBeLessThan(50);
    for (const g of gaps) { expect(g).toBeGreaterThanOrEqual(1000); expect(g).toBeLessThanOrEqual(2000); }
    for (let k = -5; k < 5; k++) { const u = fluxHash(k, FLUX.saltT); expect(u).toBeGreaterThanOrEqual(-0.5); expect(u).toBeLessThan(0.5); }
  });
  it("drops are exponential with e-folding 500 M, then refill linearly (unchanged)", () => {
    const db = 0.2138;
    for (const k of [3, 17, 401]) {
      const tk = eruptionTime(k), dk = eruptionDepth(k, db), D = -FLUX.tauD * Math.log(1 - dk);
      expect(fluxDeficit(tk + 100, db)).toBeCloseTo(1 - Math.exp(-0.2), 12);
      expect(fluxDeficit(tk + D - 1e-6, db)).toBeCloseTo(dk, 6);
      expect(fluxDeficit(0.5 * (tk + D + eruptionTime(k + 1)), db)).toBeCloseTo(0.5 * dk, 6);
    }
  });
  it("flicker: deterministic, clipped at 3, clipped variance c2, decorrelated after 100 M", () => {
    let m = 0, v = 0, mx = 0, n = 0; const xs: number[] = [];
    for (let t = 2.5; t < 1e6; t += 5) { const x = fluxFlicker(t); xs.push(x); m += x; v += x * x; mx = Math.max(mx, Math.abs(x)); n++; }
    expect(fluxFlicker(1234.5)).toBe(fluxFlicker(1234.5));
    expect(mx).toBeLessThanOrEqual(FLUX.flickerClip);
    expect(Math.abs(v / n - FLUX.c2)).toBeLessThan(0.01);
    expect(Math.abs(m / n)).toBeLessThan(0.02);
    let num = 0; for (let i = 0; i + 20 < xs.length; i++) num += xs[i] * xs[i + 20];
    expect(Math.abs(num / v)).toBeLessThan(0.05); // lag 100 M
  });
  it("each table row reproduces Narayan+2022's windowed Phi index, window insensitivity and the 2-sigma depth", () => {
    for (const row of FLUX_SPIN) {
      const s = fluxSeries(row.dbar, row.eps);
      expect(Math.abs(modulationIndex(s, 1000) - fluxTarget(row.a))).toBeLessThan(0.002);
      expect(modulationIndex(s, 500) / modulationIndex(s, 2000)).toBeGreaterThanOrEqual(0.8);
      expect(Math.abs(2 * seriesSigma(s) / row.dbar - 1)).toBeLessThan(0.01);
      let d1 = 0, d2 = 0, d3 = 0, n = 0;
      for (let t = 0.5; t < 1e6; t += 1) { const d = fluxDeficit(t, row.dbar); d1 += d; d2 += d * d; d3 += d * d * d; n++; }
      expect(Math.abs(d1 / n / row.d1 - 1)).toBeLessThan(1e-4);
      expect(Math.abs(d2 / n / row.d2 - 1)).toBeLessThan(1e-4);
      expect(Math.abs(d3 / n / row.d3 - 1)).toBeLessThan(1e-4);
    }
  });
  it("spin interpolation: table rows exact, linear between, clamped outside [0, 0.9] (Review Focus 1, 5)", () => {
    for (const row of FLUX_SPIN) expect(fluxParams(row.a).eps).toBe(row.eps);
    const mid = fluxParams(0.45), lo = FLUX_SPIN[1], hi = FLUX_SPIN[2];
    expect(mid.dbar).toBeCloseTo(0.5 * (lo.dbar + hi.dbar), 12);
    expect(mid.d2).toBeCloseTo(0.5 * (lo.d2 + hi.d2), 12);
    expect(fluxParams(0.998)).toEqual(fluxParams(0.9));
    expect(fluxParams(-0.2)).toEqual(fluxParams(0));
    expect(fluxTarget(0)).toBeCloseTo(0.067, 12); expect(fluxTarget(0.998)).toBeCloseTo(0.098, 12);
  });
  it("s = 0 is a steady jet; s <= sMax never reaches the floor at any spin (Review Focus 3)", () => {
    for (const t of [0, 123.4, 9e5]) expect(fluxRatio(t, 0, 0.9)).toBe(1);
    for (const a of [0, 0.45, 0.9]) {
      const p = fluxParams(a);
      let maxDepth = 0; for (let k = -10; k < 5000; k++) maxDepth = Math.max(maxDepth, eruptionDepth(k, p.dbar));
      expect((1 - FLUX.sMax * maxDepth) * (1 - FLUX.sMax * p.eps * FLUX.flickerClip)).toBeGreaterThan(FLUX.floor);
    }
  });
  it("closed-form <f> = 1, <f^2>, <f^3> match brute force (Review Focus 4)", () => {
    for (const a of [0, 0.45, 0.9, 0.998]) for (const s of [0.5, 1, 1.4]) {
      let m1 = 0, m2 = 0, m3 = 0, n = 0;
      for (let t = 0.5; t < 1e6; t += 1) { const f = fluxRatio(t, s, a); m1 += f; m2 += f * f; m3 += f * f * f; n++; }
      expect(m1 / n).toBeCloseTo(1, 2);
      expect(Math.abs(m2 / n / fluxMoment(2, s, a) - 1)).toBeLessThan(3e-3);
      expect(Math.abs(m3 / n / fluxMoment(3, s, a) - 1)).toBeLessThan(5e-3);
    }
  });
});
```

  In `tests/jet.test.ts`: every `fluxRatio(x, s)` call becomes `fluxRatio(x, s, 0.9)` (the shape tests use a = 0.9);
  the envelope bound test becomes

```ts
  it("JET_ENV_Q covers the widest jet the slider allows at every spin", () => {
    for (const row of FLUX_SPIN) {
      const fMax = (1 + FLUX.sMax * row.eps * FLUX.flickerClip) / (1 - FLUX.sMax * row.d1);
      expect(JET_ENV_Q).toBeGreaterThanOrEqual(1.2 * Math.sqrt(fMax));
    }
  });
```

  (import `FLUX_SPIN`); the envelope sweep adds an outer loop `for (const a of [0.3, 0.9, 0.998])` and passes `a`
  as `jetShape`'s last argument. In the WGSL-constants test replace `["FLUX_DBAR", FLUX.dbar], ["FLUX_D1", FLUX.d1]`
  with `["FLUX_CELL_N", FLUX.flickerCell], ["FLUX_CLIP", FLUX.flickerClip], ["FLUX_SALT_N", FLUX.saltN]` and add:

```ts
    const tab = wconst("FLUX_TAB").match(/-?\d+\.\d+(e-?\d+)?/g)!.map(Number); // 4 rows x (dbar, eps, d1)
    FLUX_SPIN.forEach((r, i) => { expect(tab[3 * i]).toBeCloseTo(r.dbar, 7); expect(tab[3 * i + 1]).toBeCloseTo(r.eps, 7); expect(tab[3 * i + 2]).toBeCloseTo(r.d1, 7); });
```

  In `tests/synchrotron.test.ts`: the distribution-average test uses `fluxRatio(t, s, 0.9)`, and add

```ts
  it("q0 x <P> = eta P_BZ at spins 0.3 and 0.9 (Review Focus 4)", () => {
    for (const a of [0.3, 0.9]) for (const s of [1, 1.4]) {
      const U = jetUniforms(6.5e9, a, 1e-5, 2e-3, 60, 2, s);
      expect((U.jetQ0 * meanInjection(a, U.jetB0, U.rgCm, 60, 2, s)) / (2e-3 * U.pBZ)).toBeCloseTo(1, 9);
    }
  });
```

- [ ] **Step 2: Run to verify failure.** `npx vitest run tests/flux-history.test.ts tests/jet.test.ts tests/synchrotron.test.ts`
  — Expected: FAIL (`FLUX_SPIN`/`fluxFlicker` undefined; the WGSL test fails on `FLUX_CELL_N` until Task 2 — that
  one stays red until Task 2 and is excluded from this task's completion run with `-t "^(?!.*emission-shared)"`
  or by running it in Task 2).

- [ ] **Step 3: Implement.** In `emission.ts` change `function gaussNode(` to `export function gaussNode(`.
  Rewrite `src/physics/flux-history.ts`:

```ts
// Horizon magnetic-flux history Phi(t) of a magnetically arrested disk (specs 2026-10-03 jet flux knots and flux
// statistics). Phi / Phi_sat = (1 - d(t)) (1 + eps n(t)): d is the eruption sawtooth (recurrence ~1500 M,
// arXiv 2510.25842; exponential drop with the converged 500 M e-folding, Ripperda et al. 2022; linear refill),
// n is fast red-noise flicker (Gaussian lattice, 50 M cells, clipped at +-3). Per spin, (dbar, eps) make the rms
// within 1000 M windows (Narayan et al. 2022 eq. 13) equal their absolute-flux Phi trend line (Fig. 9) and the
// eruptions 2 sigma deep (2510.25842 SIV.3). WGSL twin: fluxRatioJ in emission-shared.wgsl.
import { hash4, gaussNode } from "./emission";

export const FLUX = {
  T: 1500, jitter: 1 / 3, tauD: 500, spread: 0.5, floor: 0.05, sMax: 1.4,
  flickerCell: 50, flickerClip: 3, c2: 0.995007, // E[clip(n)^2] for a standard normal clipped at 3
  saltT: 0x464c, saltD: 0x4458, saltN: 0x464e,
} as const;
/** (dbar, eps) and the eruption moments <d^k> at s = 1 per spin (scripts/calibrate-flux.ts). */
export const FLUX_SPIN = [
  { a: 0, dbar: 0.1455, eps: 0.05499, d1: 0.073273, d2: 0.007298, d3: 0.000833 },
  { a: 0.3, dbar: 0.1682, eps: 0.06295, d1: 0.084751, d2: 0.009761, d3: 0.001288 },
  { a: 0.6, dbar: 0.191, eps: 0.07072, d1: 0.096304, d2: 0.0126, d3: 0.001889 },
  { a: 0.9, dbar: 0.2138, eps: 0.0783, d1: 0.107885, d2: 0.015808, d3: 0.002653 },
] as const;
/** Narayan et al. 2022 Fig. 9 Phi trend line, 1000 t_g modulation index (prograde; a > 0.9 uses 0.9). */
export function fluxTarget(a: number): number { return 0.067 + (0.031 * Math.min(0.9, Math.max(0, a))) / 0.9; }
export function fluxParams(a: number) {
  const x = Math.min(0.9, Math.max(0, a));
  let i = 0; while (i < FLUX_SPIN.length - 2 && x > FLUX_SPIN[i + 1].a) i++;
  const lo = FLUX_SPIN[i], hi = FLUX_SPIN[i + 1], w = (x - lo.a) / (hi.a - lo.a);
  if (w === 0) return { dbar: lo.dbar, eps: lo.eps, d1: lo.d1, d2: lo.d2, d3: lo.d3 };
  if (w === 1) return { dbar: hi.dbar, eps: hi.eps, d1: hi.d1, d2: hi.d2, d3: hi.d3 };
  const L = (p: number, q: number) => p + (q - p) * w;
  return { dbar: L(lo.dbar, hi.dbar), eps: L(lo.eps, hi.eps), d1: L(lo.d1, hi.d1), d2: L(lo.d2, hi.d2), d3: L(lo.d3, hi.d3) };
}

export function fluxHash(k: number, salt: number): number { return (hash4(k, 0x7a11, 0, salt) & 0xffffff) / 16777216 - 0.5; }
export function eruptionTime(k: number): number { return FLUX.T * (k + 0.5 + FLUX.jitter * fluxHash(k, FLUX.saltT)); }
export function eruptionDepth(k: number, dbar: number): number { return dbar * (1 + FLUX.spread * fluxHash(k, FLUX.saltD)); }
/** Deficit at time k T + loc, loc in [0, T) (twin: fluxDeficitJ). */
export function fluxDeficitLocal(k: number, loc: number, dbar: number): number {
  let kk = k, dt = loc - FLUX.T * (0.5 + FLUX.jitter * fluxHash(k, FLUX.saltT));
  if (dt < 0) { kk = k - 1; dt = loc + FLUX.T - FLUX.T * (0.5 + FLUX.jitter * fluxHash(kk, FLUX.saltT)); }
  const gap = FLUX.T * (1 + FLUX.jitter * (fluxHash(kk + 1, FLUX.saltT) - fluxHash(kk, FLUX.saltT)));
  const depth = eruptionDepth(kk, dbar), D = -FLUX.tauD * Math.log(1 - depth);
  if (dt < D) return 1 - Math.exp(-dt / FLUX.tauD);
  return (depth * (gap - dt)) / (gap - D);
}
export function fluxDeficit(t: number, dbar: number): number { const k = Math.floor(t / FLUX.T); return fluxDeficitLocal(k, t - k * FLUX.T, dbar); }
const smooth = (x: number) => x * x * (3 - 2 * x);
const flickerNode = (i: number) => gaussNode(i, 0x0f1c, 0, FLUX.saltN);
/** Unit-Gaussian flicker (exact N(0,1) before the clip), lattice cells of 50 M (twin: fluxFlickerJ). */
export function fluxFlicker(t: number): number {
  const x = t / FLUX.flickerCell, i = Math.floor(x), f = smooth(x - i), w0 = 1 - f, w1 = f;
  const n = (w0 * flickerNode(i) + w1 * flickerNode(i + 1)) / Math.sqrt(w0 * w0 + w1 * w1);
  return Math.max(-FLUX.flickerClip, Math.min(FLUX.flickerClip, n));
}
/** f = Phi / <Phi> at time t, slider s, spin a (s = 0: exactly 1). */
export function fluxRatio(t: number, s: number, a: number): number {
  if (s === 0) return 1;
  const p = fluxParams(a);
  return Math.max(FLUX.floor, (1 - s * fluxDeficit(t, p.dbar)) * (1 + s * p.eps * fluxFlicker(t))) / (1 - s * p.d1);
}
/** <f^n>, n = 2, 3: flicker independent of the eruptions and symmetric (odd moments 0). */
export function fluxMoment(n: 2 | 3, s: number, a: number): number {
  const p = fluxParams(a), mu = 1 - s * p.d1, e2 = s * s * p.eps * p.eps * FLUX.c2;
  if (n === 2) return ((1 - 2 * s * p.d1 + s * s * p.d2) * (1 + e2)) / (mu * mu);
  return ((1 - 3 * s * p.d1 + 3 * s * s * p.d2 - s ** 3 * p.d3) * (1 + 3 * e2)) / mu ** 3;
}
/** Phi / Phi_sat sampled at midpoints every dt over [0, span) at s = 1 (calibration). */
export function fluxSeries(dbar: number, eps: number, span = 1e6, dt = 5): number[] {
  const out: number[] = [];
  for (let t = dt / 2; t < span; t += dt) out.push((1 - fluxDeficit(t, dbar)) * (1 + eps * fluxFlicker(t)));
  return out;
}
/** Narayan et al. 2022 eq. 13: rms / mean within consecutive windows, averaged. */
export function modulationIndex(series: number[], window: number, dt = 5): number {
  const n = Math.round(window / dt); let s = 0, c = 0;
  for (let i = 0; i + n <= series.length; i += n) {
    let m = 0, q = 0; for (let j = i; j < i + n; j++) { m += series[j]; q += series[j] * series[j]; }
    m /= n; s += Math.sqrt(Math.max(0, q / n - m * m)) / m; c++;
  }
  return s / c;
}
export function seriesSigma(series: number[]): number {
  let m = 0, q = 0; for (const v of series) { m += v; q += v * v; }
  m /= series.length; return Math.sqrt(q / series.length - m * m) / m;
}
```

  `scripts/calibrate-flux.ts` becomes:

```ts
// Reproduces FLUX_SPIN in src/physics/flux-history.ts (run: npx vite-node scripts/calibrate-flux.ts, ~2 min).
// Per spin: dbar = 2 sigma/mu of the series (self-consistent) and eps so the 1000 M modulation index hits fluxTarget(a).
import { fluxSeries, modulationIndex, seriesSigma, fluxTarget, fluxDeficit } from "../src/physics/flux-history";
for (const a of [0, 0.3, 0.6, 0.9]) {
  const tg = fluxTarget(a); let db = 0.2, eps = 0.06;
  const solveEps = (d: number, it: number) => { let lo = 0, hi = 0.3; for (let k = 0; k < it; k++) { const e = (lo + hi) / 2;
    if (modulationIndex(fluxSeries(d, e), 1000) < tg) lo = e; else hi = e; } return (lo + hi) / 2; };
  for (let it = 0; it < 12; it++) { db = Math.round(db * 1e4) / 1e4; eps = solveEps(db, 22);
    const nd = Math.round(2 * seriesSigma(fluxSeries(db, eps)) * 1e4) / 1e4; if (Math.abs(nd - db) < 2e-4) { db = nd; break; } db = nd; }
  eps = Math.round(solveEps(db, 30) * 1e5) / 1e5;
  let d1 = 0, d2 = 0, d3 = 0, n = 0;
  for (let t = 0.5; t < 1e6; t += 1) { const d = fluxDeficit(t, db); d1 += d; d2 += d * d; d3 += d * d * d; n++; }
  console.log(`{ a: ${a}, dbar: ${db}, eps: ${eps}, d1: ${(d1 / n).toFixed(6)}, d2: ${(d2 / n).toFixed(6)}, d3: ${(d3 / n).toFixed(6)} },`);
}
```

  `src/physics/jet.ts` line `f = fluxRatio(tl, fluxVar)` → `f = fluxRatio(tl, fluxVar, a)`.
  `src/physics/synchrotron.ts`: `fluxMoment(2, fluxVar)` / `fluxMoment(3, fluxVar)` → `fluxMoment(2, fluxVar, a)` /
  `fluxMoment(3, fluxVar, a)`.

- [ ] **Step 4: Reproduce the table and run the tests.** `npx vite-node scripts/calibrate-flux.ts` — Expected: the
  four rows of the Global Constraints table, digit for digit (if not: the hash wiring differs — fix it, never edit
  the constants to match). Then `npx vitest run tests/flux-history.test.ts tests/synchrotron.test.ts` and
  `npx vitest run tests/jet.test.ts -t "^(?!emission-shared)"` — Expected: PASS.

- [ ] **Step 5: Commit.** `git add src/physics/emission.ts src/physics/flux-history.ts scripts/calibrate-flux.ts src/physics/jet.ts src/physics/synchrotron.ts tests/flux-history.test.ts tests/jet.test.ts tests/synchrotron.test.ts`
  `git commit -m "Flux history v2: absolute flux Phi, per-spin windowed calibration (Narayan+2022 Fig. 9), eruptions x clipped 50 M flicker"`

---

### Task 2: WGSL twin and `?parity`

**Files:**
- Modify: `src/render/emission-shared.wgsl`, `src/render/flux-parity.wgsl`, `src/test/parity.browser.ts`

**Interfaces:** `fn fluxRatioJ(epoch: f32, rel: f32, s: f32, a: f32) -> f32`; `fn fluxFlickerJ(epoch: f32, rel: f32) -> f32`;
WGSL constants `FLUX_CELL_N = 50.0`, `FLUX_CLIP = 3.0`, `FLUX_SALT_N = 0x464eu`,
`FLUX_TAB = array<f32, 12>(dbar0, eps0, d1_0, …, dbar3, eps3, d1_3)` (rows a = 0, 0.3, 0.6, 0.9).

- [ ] **Step 1: Failing check.** The Task 1 WGSL-constants test (`npx vitest run tests/jet.test.ts`) — Expected:
  FAIL `no FLUX_CELL_N`.
- [ ] **Step 2: Implement.** In `emission-shared.wgsl` delete `FLUX_DBAR` and `FLUX_D1`; add the constants above and:

```wgsl
// Flicker: unit-Gaussian lattice in time, 50 M cells (gaussT nodes), smoothstep weights renormalised, clipped
// at +-3 (twin: fluxFlicker). The time is split into whole cells + remainder (exact at any epoch).
fn fluxFlickerJ(epoch: f32, rel: f32) -> f32 {
  let kc = splitPeriodJ(epoch, rel, FLUX_CELL_N);
  let i = i32(kc.x); let f = smoothE(kc.y / FLUX_CELL_N); let w0 = 1.0 - f; let w1 = f;
  let n = (w0 * gaussT(i, 0x0f1cu, 0, FLUX_SALT_N) + w1 * gaussT(i + 1, 0x0f1cu, 0, FLUX_SALT_N)) / sqrt(w0 * w0 + w1 * w1);
  return clamp(n, -FLUX_CLIP, FLUX_CLIP);
}
// (dbar, eps, d1) at spin a: linear in clamp(a, 0, 0.9) between the rows a = 0, 0.3, 0.6, 0.9 (twin: fluxParams).
fn fluxParamsJ(a: f32) -> vec3<f32> {
  var tab = FLUX_TAB;
  let x = clamp(a, 0.0, 0.9) / 0.3; let i = min(u32(floor(x)), 2u); let w = x - f32(i);
  let lo = vec3<f32>(tab[3u * i], tab[3u * i + 1u], tab[3u * i + 2u]);
  let hi = vec3<f32>(tab[3u * i + 3u], tab[3u * i + 4u], tab[3u * i + 5u]);
  return mix(lo, hi, w);
}
```

  and change `fluxDeficitJ(kf, loc)` to take `dbar` as a third argument (use it in place of `FLUX_DBAR`), and
  `fluxRatioJ` to:

```wgsl
fn fluxRatioJ(epoch: f32, rel: f32, s: f32, a: f32) -> f32 {
  if (s == 0.0) { return 1.0; }
  let p = fluxParamsJ(a);
  let kl = splitPeriodJ(epoch, rel, FLUX_T);
  return max(FLUX_FLOOR, (1.0 - s * fluxDeficitJ(kl.x, kl.y, p.x)) * (1.0 + s * p.y * fluxFlickerJ(epoch, rel))) / (1.0 - s * p.z);
}
```

  In `jetShapeJ`: `fluxRatioJ(epoch, relL, fluxVar)` → `fluxRatioJ(epoch, relL, fluxVar, a)`. In
  `flux-parity.wgsl`: input `b = (q, ph, spin, s)` already carries the spin — call `fluxRatioJ(c.a.x, c.a.y, c.b.w, c.b.z)`
  and update the header comment. In `parity.browser.ts`: the flux case list becomes
  `[[5, 2, 0.8, 0.3, 0.9, 1], [-40, 5, 1.1, -7.2, 0.998, 1.4], [59, 1.5, 0.2, 12.9, 0, 0.5], [20, 3, 0.6, 2.2, 0.6, 1], [33, 2, 0.9, 4.4, 0.45, 1.2]]`
  and the CPU expectation `fluxRatio(t, s)` → `fluxRatio(t, s, a)`.
- [ ] **Step 3: Run.** `npx vitest run tests/jet.test.ts tests/shader-twins.test.ts` — PASS. Dev server on :5173,
  `?parity` route — Expected PARITY PASS with the flux line ≤ 1 (100 flux cases now).
- [ ] **Step 4: Mutations.** `FLUX_CELL_N = 51.0` → `?parity` FAIL; revert. Change the a = 0.6 row's eps in
  `FLUX_TAB` by +0.01 → FAIL; revert (revert by exact text replacement, never `git checkout` of a file with other
  uncommitted work).
- [ ] **Step 5: Commit.** `git commit -m "WGSL twin: fluxRatioJ with per-spin table and 50 M flicker; ?parity flux cases across spins"`

---

### Task 3: Gates, visual check, docs, ship

- [ ] **Step 1:** `npm test` (all pass), `npm run build` (clean).
- [ ] **Step 2:** `RECORD_GOLDEN=1 npm run verify:gpu` — every route PASS; jet-off hash: if changed, the jet-free
  kernel experiment (stub `jetStep` to `return accIn;` on main and branch, record both, hashes equal; control:
  main unstubbed reproduces its committed hashes).
- [ ] **Step 3:** Visual: `node scripts/shot-flux.mjs` from a scratch directory; contact sheet; measure the jet-box
  brightness series (expect small, fast fluctuations plus a ~20 % eruption dip, not 51 %); estimate the knot
  spacing from one frame's brightness profile along the jet.
- [ ] **Step 4:** Docs: README jet-knots section gets a "Correction (flux statistics, 2026-10-03)" paragraph with the
  two errors, the table, the windowed check numbers, golden hashes and test counts; ROADMAP follow-up 1 marked done;
  spec status "implemented".
- [ ] **Step 5:** Commit, fast-forward main, push, `vercel deploy --prod` (PowerShell), verify the live bundle
  contains `fluxFlickerJ`.
