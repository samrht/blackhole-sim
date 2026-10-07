# Hotspot Flares in the 1.3 mm View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In the 1.3 mm view, a Gaussian hotspot is born at each horizon-flux eruption, orbits Keplerian at 8–12 r_g for up
to three orbits and fades. It is integrated with the hot flow along each ray (Doppler, lensing, light-travel delay,
absorption), and its brightness is calibrated so Sgr A*'s hotspot adds 0.3 Jy (ALMA, Wielgus et al. 2022).

**Architecture:** CPU twin `src/physics/hotspot.ts` (schedule from `flux-history.ts`, light curve, Gaussian boost,
Doppler factor, alive window). `hot-flow-image.ts` gains an optional hotspot so the calibration script and the gated sweep
can image it. WGSL twin in `emission-shared.wgsl` (sole copy, gated by `?parity`). `raytrace.wgsl`'s `flowStep` adds the
hotspot's emission and absorption to each flow sample, on live frames only. `main.ts` forces live tracing while a hotspot
can be in view, and sends `fluxVar = 0` in mm otherwise, so hotspot-free frames skip the hotspot code exactly. The cache
stays steady and is never rebuilt for a hotspot.

**Tech Stack:** TypeScript, WebGPU/WGSL, Vitest, vite-node, Playwright (`verify:gpu`, `bench`).

**Spec:** `docs/specs/2026-10-04-mm-hotspots-design.md`. This plan departs from it in three places, recorded in Task 5
Step 3: (1) the mode-switch window half-width is measured, `HOTSPOT.pad` (sweep, Task 2), not a fixed 100 M; (2) in mm,
`main.ts` sends `fluxVar = 0` while no hotspot can be in view. In mm, `fluxVar` drives only the hotspots (no jet at
1.3 mm), so this is exact and keeps hotspot-free live frames at their old cost; (3) a dev-only `window.__bhSetTime` hook
lets the app check jump to a hotspot.

## Global Constraints

- Orbit radius r_c,k = 8 + 4 u_k r_g, with u_k = `fluxHash(k, 0x4853) + 0.5`; starting azimuth φ₀,k = 2π (`fluxHash(k, 0x4850)` + 0.5).
- Orbit: prograde, equatorial, rigid: φ_c = φ₀,k + τ / q, with q = r_c^{3/2} + a, Ω_c = 1/q, P = 2π q, τ = t − t_k, and t_k = `eruptionTime(k)`.
- Gaussian σ = 2.548 M (6 r_g FWHM), truncated at d = 4σ. d² = r² + r_c² − 2 r r_c sinθ cos(φ − φ_c) (BL pseudo-Cartesian).
- Light curve: L(τ) = smoothstep(0, 0.1 P, τ) · exp(−τ/P) · (1 − smoothstep(2.5 P, 3 P, τ)); 0 for τ < 0 and τ ≥ 3 P.
- Amplitude: A_k L = A₀ · s · (1 + 0.5 `fluxHash(k, FLUX.saltD)`) · L, where (1 + 0.5 hash) = δ_k / δ̄. s = Flux variability; s = 0 means no hotspots.
- Emission: j′_h = A_k L G · j′_flow(r, θ; ν D_h) and α′_h = A_k L G · α′_flow(r, θ; ν D_h), with D_h = u^t (p_t + Ω_c p_φ) and
  u^t = (−(g_tt + 2Ω_c g_tφ + Ω_c² g_φφ))^{−1/2}. There is no contribution where that is not timelike. Both emitters share
  one slab per sample.
- Calibration: Sgr A* preset, r_c = 10, δ_k = δ̄, s = 1, at the peak of L (amp = A₀ · L_peak), emission time frozen,
  added flux averaged over φ_c ∈ {0, π/2, π, 3π/2} = 0.3 Jy. Prototype: amp ≈ 8.5, so A₀ ≈ 9.4.
- Visible golden hashes and the existing mm hashes (`sgra-mm` f298812a, `m87-mm` 3ea2d532) stay bit-identical. No hotspot
  is alive at their times 0 and 37 M (the first eruption is at t₀ ≈ 831 M).
- No Co-Authored-By trailer. Standing rule: when the gates pass, fast-forward into main and push without asking. Deploy only on request.

## Review Focus

1. **Cached frame shown while a hotspot is visible** (expected: never). Rays seen through the blob region have
   light-travel delays bounded by `HOTSPOT.pad` (Task 2 sweep). The mode is live whenever a hotspot is alive within ±pad
   of the clock (Task 1 window test), and the app goes live and comes back to cached without a rebuild (Task 4 app check).
2. **Long sessions** (clock ~1e6–8e6 M): the hotspot keeps its exact position and brightness in f32 (Task 3 parity cases at late epochs).
3. **Slider extremes**: s = 0 gives no hotspot anywhere and the mode is never forced live; s = 1.4 scales the amplitude by
   1.4 with no NaN (Task 1 tests, Task 3 parity at s ∈ {0, 0.6, 1.4}).
4. **Blob region near the horizon** (d < 4σ reaches r < 2): where the hotspot's circular orbit is not timelike, D_h = −1
   and there is no contribution, with no NaN in the image (Task 3 parity at r = 1.6; Task 4 golden scene).
5. **Light-delay toggle off**: every sample is seen at one instant, and the GPU still matches the CPU twin (Task 4 `?hotflow`, both settings).

---

### Task 1: Hotspot model on the CPU

**Files:** Create `src/physics/hotspot.ts`, `tests/hotspot.test.ts`; Modify `src/render/cache-plan.ts` (`chooseMode`), `tests/cache-plan.test.ts`.

**Interfaces:**
- Consumes: `FLUX`, `fluxHash(k, salt)`, `eruptionTime(k)` (flux-history.ts); `metricLower(r, th, a)` (kerr.ts).
- Produces: `HOTSPOT` constants `{ rMin: 8, rSpan: 4, sigma: 2.548, cut: 4, rise: 0.1, cutFrom: 2.5, cutTo: 3, saltR: 0x4853, saltPhi: 0x4850, A0, pad }`;
  `HOTSPOT_REACH` (= rMin + rSpan + cut·sigma = 22.192); `HOTSPOT_LPEAK`, `HOTSPOT_TPEAK` (max of L and its τ/P);
  `hotspotLight(tau, P): number`; `hotspotRadius(k): number`; `hotspotPeriod(rc, a): number`;
  `interface HotspotState { alive: boolean; k: number; rc: number; phiC: number; amp: number; Om: number }`;
  `hotspotAt(t, s, a, A0 = HOTSPOT.A0): HotspotState`; `hotspotBoost(r, th, ph, rc, phiC): number` (G, 0 beyond 4σ);
  `hotspotShift(r, th, pt, pphi, a, Om): number | null`; `hotspotPeakTime(k, a): number`; `hotspotEndTime(k, a): number`;
  `hotspotAliveWindow(t, s, a, pad = HOTSPOT.pad): boolean`;
  `chooseMode(playing, completedSets, enabled, forceLive = false)`.

- [ ] **Step 1: failing tests** `tests/hotspot.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { HOTSPOT, HOTSPOT_REACH, HOTSPOT_LPEAK, HOTSPOT_TPEAK, hotspotLight, hotspotRadius, hotspotPeriod, hotspotAt,
  hotspotBoost, hotspotShift, hotspotPeakTime, hotspotEndTime, hotspotAliveWindow } from "../src/physics/hotspot";
import { FLUX, eruptionTime, eruptionDepth, fluxHash } from "../src/physics/flux-history";
import { omegaKepler } from "../src/physics/orbits";
import { gFactorKepler } from "../src/physics/redshift";

describe("hotspot schedule (spec 2026-10-04 mm hotspots 2.2)", () => {
  it("one hotspot per eruption: radius in [8, 12), every lifetime 3P shorter than the shortest eruption gap", () => {
    let rmin = Infinity, rmax = 0;
    for (let k = -5; k < 2000; k++) {
      const rc = hotspotRadius(k); rmin = Math.min(rmin, rc); rmax = Math.max(rmax, rc);
      expect(rc).toBeGreaterThanOrEqual(8); expect(rc).toBeLessThan(12);
      for (const a of [0, 0.998]) expect(hotspotEndTime(k, a)).toBeLessThan(eruptionTime(k + 1));
    }
    expect(rmin).toBeLessThan(8.05); expect(rmax).toBeGreaterThan(11.95); // the range is used, not a corner of it
    expect(HOTSPOT_REACH).toBeCloseTo(8 + 4 + 4 * 2.548, 10);
  });
  it("hotspotAt: the latest eruption, alive while 0 <= tau < 3P; amplitude A0 s (delta_k / dbar) L; none at s = 0", () => {
    const a = 0.94;
    for (let t = -3000; t < 3e5; t += 7.3) {
      const h = hotspotAt(t, 1, a), k = h.k;
      expect(eruptionTime(k)).toBeLessThanOrEqual(t); expect(eruptionTime(k + 1)).toBeGreaterThan(t);
      const tau = t - eruptionTime(k), P = hotspotPeriod(hotspotRadius(k), a);
      expect(h.alive).toBe(tau < 3 * P);
      if (!h.alive) continue;
      expect(h.rc).toBe(hotspotRadius(k));
      expect(h.amp).toBeCloseTo(HOTSPOT.A0 * (eruptionDepth(k, FLUX.dbar) / FLUX.dbar) * hotspotLight(tau, P), 9);
      expect(hotspotAt(t, 1.4, a).amp).toBeCloseTo(1.4 * h.amp, 9);
      expect(hotspotAt(t, 0, a).alive).toBe(false);
    }
  });
  it("Keplerian, prograde, rigid: Omega = omegaKepler(r_c), phi_c advances 2 pi per period from 2 pi (u_phi + 0.5)", () => {
    for (const a of [0, 0.5, 0.94]) for (const k of [0, 1, 7]) {
      const t0 = eruptionTime(k), h0 = hotspotAt(t0, 1, a), P = hotspotPeriod(h0.rc, a);
      expect(h0.Om).toBeCloseTo(omegaKepler(h0.rc, a, true), 12);
      expect(h0.phiC).toBeCloseTo(2 * Math.PI * (fluxHash(k, HOTSPOT.saltPhi) + 0.5), 9);
      expect(hotspotAt(t0 + P, 1, a).phiC - h0.phiC).toBeCloseTo(2 * Math.PI, 9);
    }
  });
});

describe("hotspot light curve, boost and Doppler factor", () => {
  it("L: 0 before birth, rises over 0.1 P, e-folds in P, smooth cut to 0 by 3 P", () => {
    const P = 200;
    expect(hotspotLight(-1, P)).toBe(0); expect(hotspotLight(0, P)).toBe(0);
    expect(hotspotLight(0.05 * P, P)).toBeCloseTo(0.5 * Math.exp(-0.05), 12);
    expect(hotspotLight(0.1 * P, P)).toBeCloseTo(Math.exp(-0.1), 12);
    expect(hotspotLight(P, P)).toBeCloseTo(Math.exp(-1), 12);
    expect(hotspotLight(2.5 * P, P)).toBeCloseTo(Math.exp(-2.5), 12);
    expect(hotspotLight(2.75 * P, P)).toBeCloseTo(0.5 * Math.exp(-2.75), 12);
    expect(hotspotLight(3 * P, P)).toBe(0);
    expect(HOTSPOT_LPEAK).toBeGreaterThan(0.904); expect(HOTSPOT_LPEAK).toBeLessThan(0.907);
    expect(HOTSPOT_TPEAK).toBeGreaterThan(0.09); expect(HOTSPOT_TPEAK).toBeLessThan(0.1);
    const a = 0.94, tp = hotspotPeakTime(3, a), h = hotspotAt(tp, 1, a);
    expect(h.amp / (HOTSPOT.A0 * (eruptionDepth(3, FLUX.dbar) / FLUX.dbar))).toBeCloseTo(HOTSPOT_LPEAK, 9);
  });
  it("G: 1 at the centre, 1/2 at 3 M (FWHM 6 M), 0 beyond 4 sigma", () => {
    const rc = 10, ph = 1.3;
    expect(hotspotBoost(rc, Math.PI / 2, ph, rc, ph)).toBeCloseTo(1, 12);
    expect(hotspotBoost(rc + 3, Math.PI / 2, ph, rc, ph)).toBeCloseTo(0.5, 3);
    const z = (d: number) => Math.atan2(rc, d); // a point d above the centre: r = hypot(rc, d), th = atan2(rc, d)
    expect(hotspotBoost(Math.hypot(rc, 3), z(3), ph, rc, ph)).toBeCloseTo(0.5, 3);
    expect(hotspotBoost(rc + 4 * 2.548 + 0.01, Math.PI / 2, ph, rc, ph)).toBe(0);
    expect(hotspotBoost(rc + 4 * 2.548 - 0.01, Math.PI / 2, ph, rc, ph)).toBeGreaterThan(0);
  });
  it("D_h: static far-observer limit 1, and 1 / g of a Keplerian equatorial emitter (redshift.ts)", () => {
    expect(hotspotShift(1e6, Math.PI / 2, 1, 0, 0.5, 1e-9)!).toBeCloseTo(1, 5);
    for (const [r, a, xi] of [[10, 0.94, 3], [8, 0, -4], [12, 0.5, 0]] as const) {
      const D = hotspotShift(r, Math.PI / 2, 1, -xi, a, omegaKepler(r, a, true))!;
      expect(D * gFactorKepler(r, a, xi, true)).toBeCloseTo(1, 9);
    }
    expect(hotspotShift(1.5, Math.PI / 2, 1, 2, 0.94, 1 / (10 ** 1.5 + 0.94))).toBeNull(); // inside the ergosphere: not timelike
  });
});

describe("live-tracing window", () => {
  it("true iff some hotspot is alive within +-pad of t (brute force); never at s = 0", () => {
    const a = 0.94, pad = 300;
    for (let i = 0; i < 300; i++) {
      const t = 137.77 * i - 2000;
      let brute = false; for (let x = t - pad; x <= t + pad && !brute; x += 0.25) brute = hotspotAt(x, 1, a).alive;
      expect(hotspotAliveWindow(t, 1, a, pad)).toBe(brute);
      expect(hotspotAliveWindow(t, 0, a, pad)).toBe(false);
    }
  });
});
```

  In `tests/cache-plan.test.ts`, inside `describe("chooseMode")`, add:

```ts
  it("forceLive (a 1.3 mm hotspot can be in view) traces live even with a complete cache", () => {
    expect(chooseMode(true, 4, true, true)).toBe("live");
    expect(chooseMode(true, 4, true, false)).toBe("cached");
  });
```

- [ ] **Step 2:** `npx vitest run tests/hotspot.test.ts tests/cache-plan.test.ts`. Expected: FAIL (module missing; `chooseMode` ignores the 4th argument).
- [ ] **Step 3: implement** `src/physics/hotspot.ts`:

```ts
// Hotspot flares in the 1.3 mm view (spec 2026-10-04 mm hotspots). One Gaussian blob per horizon-flux eruption
// (flux-history.ts) on a prograde, rigid, Keplerian equatorial orbit at 8-12 r_g (GRAVITY Collaboration 2018: 6-10 r_g;
// Wielgus et al. 2022: ~11 r_g, 74 +- 6 min for Sgr A*). It boosts the hot flow's electrons by A_k L(tau) G(d) at the flow's
// own temperature and field. WGSL twin: hotspotStateJ / hotspotBoostJ / hotspotShiftJ in emission-shared.wgsl.
import { FLUX, fluxHash, eruptionTime } from "./flux-history";
import { metricLower } from "./kerr";

export const HOTSPOT = {
  rMin: 8, rSpan: 4, sigma: 2.548, cut: 4, rise: 0.1, cutFrom: 2.5, cutTo: 3, saltR: 0x4853, saltPhi: 0x4850,
  // scripts/calibrate-hotspot.ts: Sgr A*, r_c = 10, mean depth, peak of L -> +0.3 Jy at 229 GHz (Wielgus et al. 2022 S3.1)
  A0: 9.4,
  // M: half-width of the live-tracing window; bounds |light-travel delay| of every ray through the blob region (sweep-hotspot)
  pad: 300,
} as const;
/** No blob reaches beyond this radius (r_c max + the 4 sigma cut). */
export const HOTSPOT_REACH = HOTSPOT.rMin + HOTSPOT.rSpan + HOTSPOT.cut * HOTSPOT.sigma;

const sstep = (e0: number, e1: number, x: number) => { const u = Math.min(1, Math.max(0, (x - e0) / (e1 - e0))); return u * u * (3 - 2 * u); };
/** Light curve: rise over 0.1 P, e-fold in P, smooth cut to 0 between 2.5 P and 3 P. */
export function hotspotLight(tau: number, P: number): number {
  if (tau < 0 || tau >= HOTSPOT.cutTo * P) return 0;
  return sstep(0, HOTSPOT.rise * P, tau) * Math.exp(-tau / P) * (1 - sstep(HOTSPOT.cutFrom * P, HOTSPOT.cutTo * P, tau));
}
/** max_tau L and its tau / P (independent of P): the calibration's "peak of L". */
export const [HOTSPOT_LPEAK, HOTSPOT_TPEAK] = (() => {
  let m = 0, at = 0; for (let i = 0; i <= 20000; i++) { const x = i * 1e-5, v = hotspotLight(x, 1); if (v > m) { m = v; at = x; } }
  return [m, at];
})();
export function hotspotRadius(k: number): number { return HOTSPOT.rMin + HOTSPOT.rSpan * (fluxHash(k, HOTSPOT.saltR) + 0.5); }
export function hotspotPeriod(rc: number, a: number): number { return 2 * Math.PI * (rc ** 1.5 + a); }
export interface HotspotState { alive: boolean; k: number; rc: number; phiC: number; amp: number; Om: number }
/** The hotspot at absolute time t: that of the latest eruption k (t_k <= t), alive while its light curve is on.
 *  Lifetimes (3P <= 802 M) are shorter than the shortest eruption gap (1000 M), so at most one is alive. */
export function hotspotAt(t: number, s: number, a: number, A0: number = HOTSPOT.A0): HotspotState {
  let k = Math.floor(t / FLUX.T); if (eruptionTime(k) > t) k--;
  const tau = t - eruptionTime(k), rc = hotspotRadius(k), q = rc ** 1.5 + a, P = 2 * Math.PI * q;
  const phiC = 2 * Math.PI * (fluxHash(k, HOTSPOT.saltPhi) + 0.5) + tau / q;
  const alive = s > 0 && tau < HOTSPOT.cutTo * P;
  const amp = alive ? A0 * s * (1 + FLUX.spread * fluxHash(k, FLUX.saltD)) * hotspotLight(tau, P) : 0;
  return { alive, k, rc, phiC, amp, Om: 1 / q };
}
/** Gaussian profile G(d) about (r_c, pi/2, phi_c), truncated at 4 sigma (twin: hotspotBoostJ). */
export function hotspotBoost(r: number, th: number, ph: number, rc: number, phiC: number): number {
  const d2 = r * r + rc * rc - 2 * r * rc * Math.sin(th) * Math.cos(ph - phiC), sg = HOTSPOT.sigma;
  if (d2 >= (HOTSPOT.cut * sg) ** 2) return 0;
  return Math.exp(-d2 / (2 * sg * sg));
}
/** nu_plasma / nu_obs of matter at (r, th) moving at Omega (no radial motion), for the camera-normalised past-directed
 *  conserved momenta p_t, p_phi; null where that motion is not timelike (twin: hotspotShiftJ). */
export function hotspotShift(r: number, th: number, pt: number, pphi: number, a: number, Om: number): number | null {
  const g = metricLower(r, th, a), K = -(g.tt + 2 * Om * g.tphi + Om * Om * g.phph);
  if (K <= 0) return null;
  return (pt + Om * pphi) / Math.sqrt(K);
}
export function hotspotPeakTime(k: number, a: number): number { return eruptionTime(k) + HOTSPOT_TPEAK * hotspotPeriod(hotspotRadius(k), a); }
export function hotspotEndTime(k: number, a: number): number { return eruptionTime(k) + HOTSPOT.cutTo * hotspotPeriod(hotspotRadius(k), a); }
/** Is some hotspot alive at a time within [t - pad, t + pad]? main.ts traces live while it is (spec 3). */
export function hotspotAliveWindow(t: number, s: number, a: number, pad: number = HOTSPOT.pad): boolean {
  if (s <= 0) return false;
  for (let k = Math.floor((t - pad) / FLUX.T) - 1; k <= Math.floor((t + pad) / FLUX.T); k++)
    if (eruptionTime(k) < t + pad && hotspotEndTime(k, a) > t - pad) return true;
  return false;
}
```

  The `hotspotAt` test expects `hotspotAt(t, 0, a).alive === false` and an amplitude of 0 when dead. In
  `src/render/cache-plan.ts`:

```ts
/** forceLive: a 1.3 mm hotspot can be in view (main.ts, hotspotAliveWindow); the steady cache cannot show it. */
export function chooseMode(playing: boolean, completedSets: number, enabled: boolean, forceLive = false): Mode {
  if (forceLive) return "live";
  // ...existing body unchanged
```

- [ ] **Step 4:** `npx vitest run tests/hotspot.test.ts tests/cache-plan.test.ts`. Expected: PASS. (`HOTSPOT.A0` 9.4 and `pad` 300 are placeholders until Task 2 measures them; no test depends on their values.)
- [ ] **Step 5:** `git commit -m "Hotspots: CPU model (eruption schedule, Keplerian orbit, light curve, Gaussian boost, Doppler factor, live window); chooseMode forceLive"`

---

### Task 2: CPU image with a hotspot, calibration and gated sweep

**Files:** Modify `src/physics/hot-flow-image.ts`, `src/physics/hotspot.ts` (A0, pad, `HOTSPOT_TWIN`); Create
`scripts/calibrate-hotspot.ts`, `tests/sweep-hotspot.test.ts`.

**Interfaces:**
- Consumes: Task 1's `HOTSPOT`, `HOTSPOT_REACH`, `HOTSPOT_LPEAK`, `hotspotAt`, `hotspotBoost`, `hotspotShift`, `hotspotPeakTime`.
- Produces: `FlowSamples` = `{ N, half, a, rays: Float64Array[] /* per sample (r, th, D, dl, phi, t); D = -1 where the flow
  has no velocity */, mom: Float64Array /* (p_t, p_phi) per ray */ }`;
  `type HotspotFn = (tEmit: number) => { rc: number; phiC: number; amp: number; Om: number } | null`;
  `flowImage(smp, n0, rgCm, hs?: HotspotFn, tObs = 0, lightDelay = 1): Float64Array`;
  `imageCentroid(I, N, half): [number, number]` (α, β in M); `HOTSPOT_TWIN = { delay: { jy, cx, cy }, instant: { jy, cx, cy } }`
  (CPU twin of Sgr A* at the peak of eruption 0, read by Task 4's `?hotflow`).

- [ ] **Step 1: failing gated test** `tests/sweep-hotspot.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { traceFlowSamples, flowImage, imageFluxJy, HOTFLOW_TARGETS } from "../src/physics/hot-flow-image";
import { flowN0 } from "../src/physics/hot-flow";
import { HOTSPOT, HOTSPOT_REACH, HOTSPOT_LPEAK } from "../src/physics/hotspot";
import { PRESETS } from "../src/physics/presets";
const RUN = process.env.SWEEP === "1";
describe.skipIf(!RUN)("hotspot calibration and the live-window bound (48^2 CPU traces)", () => {
  it("tabulated A0 adds 0.3 Jy to Sgr A* within 5 % (r_c 10, peak of L, frozen, 4 phases)", () => {
    const p = PRESETS.find((q) => q.id === "sgra")!, T = HOTFLOW_TARGETS.sgra;
    const rg = 1.476625e5 * p.massSun, dist = T.distKpc * 3.0857e21, n0 = flowN0(p.massSun, p.lambda);
    const smp = traceFlowSamples(p.a, p.inclDeg, 48, 14), F0 = imageFluxJy(flowImage(smp, n0, rg), 48, 14, rg, dist);
    let add = 0;
    for (const phiC of [0, Math.PI / 2, Math.PI, 1.5 * Math.PI]) {
      const hs = () => ({ rc: 10, phiC, amp: HOTSPOT.A0 * HOTSPOT_LPEAK, Om: 1 / (10 ** 1.5 + p.a) });
      add += (imageFluxJy(flowImage(smp, n0, rg, hs), 48, 14, rg, dist) - F0) / 4;
    }
    console.log(`hotspot adds ${add.toFixed(3)} Jy (target 0.3)`);
    expect(Math.abs(add / 0.3 - 1)).toBeLessThan(0.05);
  }, 600000);
  for (const id of ["sgra", "gargantua"]) {
    it(`${id}: every ray through the blob region has |light-travel delay| < HOTSPOT.pad`, () => {
      const p = PRESETS.find((q) => q.id === id)!, smp = traceFlowSamples(p.a, p.inclDeg, 48, 14);
      let worst = 0;
      for (const q of smp.rays) for (let k = 0; k < q.length; k += 6) {
        const r = q[k], th = q[k + 1], t = q[k + 5];
        if (r < HOTSPOT_REACH && Math.abs(r * Math.cos(th)) < HOTSPOT.cut * HOTSPOT.sigma) worst = Math.max(worst, Math.abs(-t - 1000));
      }
      console.log(`${id}: max |delay| through the blob region ${worst.toFixed(1)} M (pad ${HOTSPOT.pad})`);
      expect(worst).toBeLessThan(HOTSPOT.pad);
    }, 600000);
  }
});
```

- [ ] **Step 2:** `SWEEP=1 npx vitest run tests/sweep-hotspot.test.ts`. Expected: FAIL (`flowImage` takes no hotspot; the samples have stride 4).
- [ ] **Step 3: implement** in `src/physics/hot-flow-image.ts`. Before changing the sample layout, run
  `grep -rn "rays\[\|\.rays\|k += 4" src tests scripts`; every hit must move to stride 6. Replace `FlowSamples`,
  `traceFlowSamples` and `flowImage` with:

```ts
import { HOTSPOT, HOTSPOT_REACH, hotspotBoost, hotspotShift } from "./hotspot";
// per ray: (r, th, D, dl, phi, t) per sample, D = -1 where the flow has no velocity (the hotspot can still emit there);
// mom: the ray's conserved (p_t, p_phi). t is the coordinate time (0 at the camera, decreasing): delay = -t - 1000.
export interface FlowSamples { N: number; half: number; a: number; rays: Float64Array[]; mom: Float64Array }
export function traceFlowSamples(a: number, inclDeg: number, N: number, half: number): FlowSamples {
  const incl = (inclDeg * Math.PI) / 180, rh = horizonOuter(a), rays: Float64Array[] = [], mom = new Float64Array(2 * N * N);
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const al = -half + (2 * half * (i + 0.5)) / N, be = half - (2 * half * (j + 0.5)) / N;
    let s = screenToState(al, be, a, incl, 1000); const q: number[] = [];
    mom[2 * (j * N + i)] = s[4]; mom[2 * (j * N + i) + 1] = s[7];
    for (let k = 0; k < 30000; k++) {
      const out = stepGeodesic(s, a, Math.min(stepSize(s, rh, 40), 0.5), s[1] > 60 ? H_TOL_FAR : H_TOL);
      if (!out.ok) break;
      const r = out.s[1];
      if (r < HOTFLOW.rMax && r > rh * 1.01) { const D = flowShift(out.s, a); q.push(r, out.s[2], D !== null && D > 0 ? D : -1, out.dl, out.s[3], out.s[0]); }
      s = out.s;
      if (r <= rh * 1.01 || r > 1100) break;
    }
    rays.push(Float64Array.from(q));
  }
  return { N, half, a, rays, mom };
}
/** The hotspot seen at emission time tEmit, or null (frozen calibrations return a constant). */
export type HotspotFn = (tEmit: number) => { rc: number; phiC: number; amp: number; Om: number } | null;
/** I_nu per pixel. With hs, each sample also carries the hotspot's boosted coefficients at its own Doppler factor, both
 *  emitters in one slab (spec 2.3); the sample's emission time is tObs - lightDelay * delay. */
export function flowImage(smp: FlowSamples, n0: number, rgCm: number, hs?: HotspotFn, tObs = 0, lightDelay = 1): Float64Array {
  const I = new Float64Array(smp.N * smp.N), zCut = HOTSPOT.cut * HOTSPOT.sigma;
  smp.rays.forEach((q, idx) => {
    const pt = smp.mom[2 * idx], pphi = smp.mom[2 * idx + 1];
    let Iv = 0, tau = 0;
    for (let k = 0; k < q.length; k += 6) {
      const r = q[k], th = q[k + 1], D = q[k + 2], dl = q[k + 3];
      let jE = 0, dTau = 0;
      if (D > 0) {
        const [j, al] = flowCoeffs(r, th, D * HOTFLOW.nu, n0);
        if (j > 0) { const ds = rgCm * D * dl; jE = (j / D ** 3) * ds; dTau = al * ds; }
      }
      if (hs && r < HOTSPOT_REACH && Math.abs(r * Math.cos(th)) < zCut) {
        const h = hs(tObs - lightDelay * (-q[k + 5] - 1000));
        const b = h ? h.amp * hotspotBoost(r, th, q[k + 4], h.rc, h.phiC) : 0;
        const Dh = b > 0 ? hotspotShift(r, th, pt, pphi, smp.a, h!.Om) : null;
        if (Dh !== null && Dh > 1e-6) {
          const [jh, ah] = flowCoeffs(r, th, Dh * HOTFLOW.nu, n0), dsh = rgCm * Dh * dl;
          jE += ((b * jh) / Dh ** 3) * dsh; dTau += b * ah * dsh;
        }
      }
      if (jE === 0 && dTau === 0) continue;
      const fac = dTau < 1e-4 ? 1 - 0.5 * dTau : (1 - Math.exp(-dTau)) / dTau;
      Iv += jE * fac * Math.exp(-tau); tau += dTau;
    }
    I[idx] = Iv;
  });
  return I;
}
/** Intensity-weighted centre (alpha, beta) in M (pixel convention of traceFlowSamples). */
export function imageCentroid(I: Float64Array, N: number, half: number): [number, number] {
  let w = 0, x = 0, y = 0;
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const v = I[j * N + i]; w += v; x += v * (-half + (2 * half * (i + 0.5)) / N); y += v * (half - (2 * half * (j + 0.5)) / N);
  }
  return [x / w, y / w];
}
```

  Without `hs`, the image is unchanged: the same samples, and the same operations in the same order. Check this with
  `SWEEP=1 npx vitest run tests/sweep-hotflow.test.ts`, which must print the same rings as before (48.2 / 35.6 µas).

  `scripts/calibrate-hotspot.ts`:

```ts
// Reproduces HOTSPOT.A0 and HOTSPOT_TWIN in src/physics/hotspot.ts (run: npx vite-node scripts/calibrate-hotspot.ts, ~10 min).
// A0: Sgr A* (96^2, half 14 = ?hotflow's view), hotspot at r_c = 10 with mean depth at the peak of L (amp = A0 L_peak),
// emission time frozen; the flux it adds, averaged over four orbital phases, is ALMA's 0.3 Jy (Wielgus et al. 2022 S3.1).
// HOTSPOT_TWIN: the CPU twin of ?hotflow's GPU measurement (eruption 0 at its peak, the real schedule, light delay 1 and 0).
import { HOTFLOW_TARGETS, traceFlowSamples, flowImage, imageFluxJy, imageCentroid } from "../src/physics/hot-flow-image";
import { flowN0 } from "../src/physics/hot-flow";
import { HOTSPOT_LPEAK, hotspotAt, hotspotPeakTime } from "../src/physics/hotspot";
import { PRESETS } from "../src/physics/presets";
const p = PRESETS.find((q) => q.id === "sgra")!, T = HOTFLOW_TARGETS.sgra, N = 96, H = 14;
const rg = 1.476625e5 * p.massSun, dist = T.distKpc * 3.0857e21, n0 = flowN0(p.massSun, p.lambda);
const smp = traceFlowSamples(p.a, p.inclDeg, N, H), base = flowImage(smp, n0, rg), F0 = imageFluxJy(base, N, H, rg, dist);
const added = (amp: number) => {
  let s = 0;
  for (const phiC of [0, Math.PI / 2, Math.PI, 1.5 * Math.PI])
    s += imageFluxJy(flowImage(smp, n0, rg, () => ({ rc: 10, phiC, amp, Om: 1 / (10 ** 1.5 + p.a) })), N, H, rg, dist) - F0;
  return s / 4;
};
let lo = 0.1, hi = 200;
for (let k = 0; k < 40; k++) { const m = Math.sqrt(lo * hi); if (added(m) < 0.3) lo = m; else hi = m; }
const A0 = Number((lo / HOTSPOT_LPEAK).toPrecision(3));
console.log(`A0 ${A0} (peak amp ${lo.toFixed(3)}, adds ${added(A0 * HOTSPOT_LPEAK).toFixed(4)} Jy over ${F0.toFixed(3)} Jy)`);
const t = hotspotPeakTime(0, p.a), hs = (te: number) => { const h = hotspotAt(te, 1, p.a, A0); return h.alive ? h : null; };
for (const [name, ld] of [["delay", 1], ["instant", 0]] as const) {
  const I = flowImage(smp, n0, rg, hs, t, ld), d = I.map((v, i) => v - base[i]);
  const [cx, cy] = imageCentroid(d, N, H);
  console.log(`HOTSPOT_TWIN.${name} = { jy: ${(imageFluxJy(d, N, H, rg, dist)).toFixed(4)}, cx: ${cx.toFixed(3)}, cy: ${cy.toFixed(3)} }  (t = ${t.toFixed(2)} M)`);
}
```

- [ ] **Step 4:** Run `npx vite-node scripts/calibrate-hotspot.ts` in the background, with a 30 min timeout. Expected:
  A0 ≈ 9.4 (prototype amp ≈ 8.5). Write the printed A0 into `HOTSPOT.A0`, and add to `hotspot.ts`:

```ts
/** CPU twin of ?hotflow's hotspot measurement (scripts/calibrate-hotspot.ts, 96^2): Sgr A* at the peak of eruption 0,
 *  flux the hotspot adds (Jy) and its centroid (alpha, beta in M), with light-travel delay on and off. */
export const HOTSPOT_TWIN = { delay: { jy: /* printed */, cx: /* printed */, cy: /* printed */ }, instant: { jy: /* printed */, cx: /* printed */, cy: /* printed */ } } as const;
```

  Fill in all six printed numbers as written. Record the full output in the ledger.
- [ ] **Step 5:** `SWEEP=1 npx vitest run tests/sweep-hotspot.test.ts`. Expected: PASS. It prints the added flux and both
  max delays. If a delay reaches the 300 M pad, set `HOTSPOT.pad` to the next multiple of 50 above 1.2 × the larger
  printed value, and rerun. Then run `npx vitest run`. Expected: the whole suite PASSES.
- [ ] **Step 6:** `git commit -m "Hotspots: CPU image (shared slab with the flow), calibration A0 = <value> (Sgr A* +0.3 Jy), measured live-window pad, ?hotflow twin targets"`

---

### Task 3: WGSL twin and `?parity`

**Files:** `src/render/emission-shared.wgsl`; Create `src/render/hotspot-parity.wgsl`; Modify `src/test/parity.browser.ts`,
`src/main.ts` (parity line), `tests/hotspot.test.ts` (WGSL constants).

**Interfaces:**
- Consumes: `splitPeriodJ`, `fluxHashJ`, `FLUX_T`, `FLUX_JIT`, `FLUX_SPREAD`, `FLUX_SALT_T`, `FLUX_SALT_D`, `TWO_PI_E`, `smoothstepJ`
  (emission-shared.wgsl); `gLow` (integrator-shared.wgsl); Task 1/2 CPU functions as the expected values.
- Produces (WGSL): constants `HS_RMIN, HS_RSPAN, HS_SIGMA, HS_CUT, HS_RISE, HS_CUT_FROM, HS_CUT_TO, HS_SALT_R, HS_SALT_PHI, HS_A0, HS_REACH`;
  `fn hotspotLightJ(tau: f32, P: f32) -> f32`; `fn hotspotStateJ(epoch: f32, rel: f32, s: f32, a: f32) -> vec4<f32>`
  (r_c, φ_c, amp, alive 0/1; all 0 when dead); `fn hotspotBoostJ(r: f32, th: f32, ph: f32, rc: f32, phiC: f32) -> f32`;
  `fn hotspotShiftJ(r: f32, th: f32, p: vec4<f32>, a: f32, Om: f32) -> f32` (−1 if not timelike). `runParity()` result gains `hsErr`, `hsWorst`.

- [ ] **Step 1: failing constants test**. Append to `tests/hotspot.test.ts`:

```ts
import { readFileSync } from "node:fs";
import { join } from "node:path";
describe("WGSL twin constants (emission-shared.wgsl)", () => {
  const W = readFileSync(join(__dirname, "../src/render/emission-shared.wgsl"), "utf8");
  const wconst = (n: string) => { const m = W.match(new RegExp(`const ${n}\\s*=\\s*([^;]+);`)); if (!m) throw new Error(`no ${n}`); return Number(m[1].trim().replace(/u$/, "")); };
  it("match HOTSPOT", () => {
    for (const [n, v] of [["HS_RMIN", HOTSPOT.rMin], ["HS_RSPAN", HOTSPOT.rSpan], ["HS_SIGMA", HOTSPOT.sigma], ["HS_CUT", HOTSPOT.cut],
      ["HS_RISE", HOTSPOT.rise], ["HS_CUT_FROM", HOTSPOT.cutFrom], ["HS_CUT_TO", HOTSPOT.cutTo], ["HS_SALT_R", HOTSPOT.saltR],
      ["HS_SALT_PHI", HOTSPOT.saltPhi], ["HS_A0", HOTSPOT.A0]] as const) expect(wconst(n)).toBe(v);
    expect(wconst("HS_REACH")).toBeCloseTo(HOTSPOT_REACH, 6);
  });
});
```

  Run `npx vitest run tests/hotspot.test.ts`. Expected: FAIL (`no HS_RMIN`).
- [ ] **Step 2: implement**. Append to `emission-shared.wgsl`, after the hot-flow block (it needs `gLow`, which is
  prepended there like `flowVelocityOrbJ`'s):

```wgsl
// --- Hotspot flares at 1.3 mm (spec 2026-10-04 mm hotspots; twin: src/physics/hotspot.ts) -----------------------------
const HS_RMIN = 8.0; const HS_RSPAN = 4.0; const HS_SIGMA = 2.548; const HS_CUT = 4.0;
const HS_RISE = 0.1; const HS_CUT_FROM = 2.5; const HS_CUT_TO = 3.0;
const HS_SALT_R = 0x4853u; const HS_SALT_PHI = 0x4850u;
const HS_A0 = 9.4;        // scripts/calibrate-hotspot.ts (twin: HOTSPOT.A0) -- write Task 2's value
const HS_REACH = 22.192;  // HS_RMIN + HS_RSPAN + HS_CUT * HS_SIGMA: no blob reaches beyond this radius
fn hotspotLightJ(tau: f32, P: f32) -> f32 {
  if (tau < 0.0 || tau >= HS_CUT_TO * P) { return 0.0; }
  return smoothstepJ(0.0, HS_RISE * P, tau) * exp(-tau / P) * (1.0 - smoothstepJ(HS_CUT_FROM * P, HS_CUT_TO * P, tau));
}
// (r_c, phi_c, A_k L, 1) of the hotspot at absolute time epoch + rel, or 0 when none is alive (twin: hotspotAt). The
// latest eruption k comes from fluxDeficitJ's split (exact at any epoch); at most one hotspot is alive at a time.
fn hotspotStateJ(epoch: f32, rel: f32, s: f32, a: f32) -> vec4<f32> {
  if (s <= 0.0) { return vec4<f32>(0.0); }
  let kl = splitPeriodJ(epoch, rel, FLUX_T);
  var k = i32(kl.x);
  var tau = kl.y - FLUX_T * (0.5 + FLUX_JIT * fluxHashJ(k, FLUX_SALT_T));
  if (tau < 0.0) { k = k - 1; tau = kl.y + FLUX_T - FLUX_T * (0.5 + FLUX_JIT * fluxHashJ(k, FLUX_SALT_T)); }
  let rc = HS_RMIN + HS_RSPAN * (fluxHashJ(k, HS_SALT_R) + 0.5);
  let q = pow(rc, 1.5) + a; let P = TWO_PI_E * q;
  if (tau >= HS_CUT_TO * P) { return vec4<f32>(0.0); }
  let phiC = TWO_PI_E * (fluxHashJ(k, HS_SALT_PHI) + 0.5) + tau / q;
  let amp = HS_A0 * s * (1.0 + FLUX_SPREAD * fluxHashJ(k, FLUX_SALT_D)) * hotspotLightJ(tau, P);
  return vec4<f32>(rc, phiC, amp, 1.0);
}
// Gaussian G(d) about (r_c, pi/2, phi_c), 0 beyond 4 sigma (twin: hotspotBoost).
fn hotspotBoostJ(r: f32, th: f32, ph: f32, rc: f32, phiC: f32) -> f32 {
  let d2 = r * r + rc * rc - 2.0 * r * rc * sin(th) * cos(ph - phiC);
  if (d2 >= HS_CUT * HS_CUT * HS_SIGMA * HS_SIGMA) { return 0.0; }
  return exp(-d2 / (2.0 * HS_SIGMA * HS_SIGMA));
}
// nu_plasma / nu_obs of matter at (r, th) moving at Om, p = (p_t, p_r, p_th, p_phi); -1 where not timelike (twin: hotspotShift).
fn hotspotShiftJ(r: f32, th: f32, p: vec4<f32>, a: f32, Om: f32) -> f32 {
  let g = gLow(r, th, a);                                // 0 tt, 1 tphi, 2 rr, 3 thth, 4 phph
  let K = -(g[0] + 2.0 * Om * g[1] + Om * Om * g[4]);
  if (K <= 0.0) { return -1.0; }
  return (p.x + Om * p.w) / sqrt(K);
}
```

  Set `HS_A0` to Task 2's `HOTSPOT.A0`. Check `smoothstepJ`'s argument order (`grep -n "fn smoothstepJ" -A3`);
  it must be (edge0, edge1, x).
- [ ] **Step 3: parity**. Create `src/render/hotspot-parity.wgsl`:

```wgsl
// Parity harness for the 1.3 mm hotspots: hotspotStateJ / hotspotBoostJ / hotspotShiftJ come from emission-shared.wgsl
// (sole copies, also prepended to raytrace.wgsl), so ?parity checks the renderer's own bytes against src/physics/hotspot.ts.
// Per case: a = (epoch, rel, s, spin), b = (r, th, ph, 0), c = (p_t, p_phi, 0, 0). Output per case, two vec4: the state
// (r_c, phi_c, amp, alive), then (G at b about the GPU's own state, D_h at b, Omega_c, 0); a dead state uses r_c = HS_RMIN.
struct HIn { a: vec4<f32>, b: vec4<f32>, c: vec4<f32> };
@group(0) @binding(0) var<storage, read> inp: array<HIn>;
@group(0) @binding(1) var<storage, read_write> outp: array<vec4<f32>>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = inp[gid.x];
  let st = hotspotStateJ(c.a.x, c.a.y, c.a.z, c.a.w);
  let rc = max(st.x, HS_RMIN); let Om = 1.0 / (pow(rc, 1.5) + c.a.w);
  outp[2u * gid.x] = st;
  outp[2u * gid.x + 1u] = vec4<f32>(hotspotBoostJ(c.b.x, c.b.y, c.b.z, rc, st.y),
    hotspotShiftJ(c.b.x, c.b.y, vec4<f32>(c.c.x, 0.0, 0.0, c.c.y), c.a.w, Om), Om, 0.0);
}
```

  In `src/test/parity.browser.ts`, after the hot-flow block and before the `return`, add the hotspot block. Import
  `hotspotParityWGSL from "../render/hotspot-parity.wgsl?raw"` and `{ HOTSPOT, hotspotAt, hotspotBoost, hotspotShift,
  hotspotPeriod, hotspotRadius }` from `../physics/hotspot`, plus `eruptionTime` from `../physics/flux-history`:

```ts
  // --- hotspots (CPU hotspot.ts vs the SHIPPED hotspotStateJ / hotspotBoostJ / hotspotShiftJ in emission-shared.wgsl) ---
  // Times across eruption k's life (before birth, the rise, the peak, mid-life, the cut, after) for k from 0 to 5000
  // (clock up to 7.5e6 M: the f32 epoch split must stay exact), spins 0 / 0.94 / 0.998, slider 0 / 0.6 / 1.4; points at the
  // centre, 2-3 M off it in r, above the plane, ahead in phi, and near the horizon (r 1.6: not timelike, D_h = -1).
  const scs: { a: number[]; b: number[]; c: number[] }[] = [];
  for (const k of [0, 1, 3, 700, 5000]) for (const a of [0, 0.94, 0.998]) for (const fr of [-0.02, 0.003, 0.05, 0.098, 0.5, 1.7, 2.6, 2.95, 3.05]) {
    const t = eruptionTime(k) + fr * hotspotPeriod(hotspotRadius(k), a), epoch = 2048 * Math.floor(t / 2048);
    for (const s of fr === 0.5 ? [0, 0.6, 1.4] : [1]) {
      const h = hotspotAt(t, 1, a), rc = h.rc;
      for (const [r, th, dph] of [[rc, Math.PI / 2, 0], [rc + 2, Math.PI / 2, 0.25], [rc - 3, 1.3, -0.2], [1.6, Math.PI / 2, 0], [3, 1.4, 0.1]])
        scs.push({ a: [epoch, t - epoch, s, a], b: [r, th, h.phiC + dph, 0], c: [1, (k & 1) ? 3 : -2.5, 0, 0] });
    }
  }
  const sarr = new Float32Array(scs.length * 12);
  scs.forEach((c, i) => sarr.set([...c.a, ...c.b, ...c.c], i * 12));
  const sin_ = device.createBuffer({ size: sarr.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(sin_, 0, sarr);
  const sout = device.createBuffer({ size: scs.length * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const sread = device.createBuffer({ size: scs.length * 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const smod = device.createShaderModule({ code: integratorSharedWGSL + emissionSharedWGSL + hotspotParityWGSL });
  const spipe = device.createComputePipeline({ layout: "auto", compute: { module: smod, entryPoint: "main" } });
  const sbind = device.createBindGroup({ layout: spipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: sin_ } }, { binding: 1, resource: { buffer: sout } }] });
  const senc = device.createCommandEncoder();
  const scp = senc.beginComputePass(); scp.setPipeline(spipe); scp.setBindGroup(0, sbind); scp.dispatchWorkgroups(scs.length); scp.end();
  senc.copyBufferToBuffer(sout, 0, sread, 0, scs.length * 32);
  device.queue.submit([senc.finish()]);
  await sread.mapAsync(GPUMapMode.READ);
  const sgpu = new Float32Array(sread.getMappedRange().slice(0));
  // Errors / tolerance: alive must agree exactly (no case sits within 0.02 P of a boundary); absolute 1e-4 on r_c, 2e-3 on
  // phi_c (f32 tau ~ 1e3 M carries ~1e-4 M), 1e-4 on G; relative 1e-3 on the amplitude, Omega_c and D_h (D_h's against the
  // size of its terms, as for the flow); D_h = -1 exactly where the CPU has no timelike motion.
  let hsErr = 0, hsWorst = "";
  const hset = (i: number, k: number, g: number, w: number, tol: number) => { const e = Math.abs(g - w) / tol;
    if (!(e <= hsErr)) { hsErr = Number.isFinite(e) ? e : Infinity; hsWorst = `case ${i} out ${k}: gpu ${g} cpu ${w}`; } };
  scs.forEach((_, i) => {
    const v = Array.from(sarr.subarray(i * 12, i * 12 + 12)), [epoch, rel, s, a] = v, [r, th, ph] = v.slice(4, 7), [pt, pphi] = v.slice(8, 10);
    const g = Array.from(sgpu.subarray(i * 8, i * 8 + 8)), h = hotspotAt(epoch + rel, s, a);
    hset(i, 3, g[3], h.alive ? 1 : 0, 1e-6);
    if (h.alive) { hset(i, 0, g[0], h.rc, 1e-4); hset(i, 1, g[1], h.phiC, 2e-3); hset(i, 2, g[2], h.amp, 1e-3 * Math.max(1, h.amp)); }
    const rc = h.alive ? g[0] : HOTSPOT.rMin, Om = 1 / (rc ** 1.5 + a);
    hset(i, 4, g[4], hotspotBoost(r, th, ph, rc, g[1]), 1e-4);
    hset(i, 6, g[6], Om, 1e-3 * Om);
    const Dh = hotspotShift(r, th, pt, pphi, a, g[6]);
    if (Dh === null) hset(i, 5, g[5], -1, 1e-6);
    else hset(i, 5, g[5], Dh, 1e-3 * (Math.abs(pt) + Math.abs(Om * pphi)) * Math.max(1, Math.abs(Dh)));
  });
  console.log("hotspot parity worst (|err| / tol)", hsErr.toExponential(2), hsWorst, "cases", scs.length);
```

  Extend the return type and value with `hsErr, hsWorst`, and add `scs.length` to `rows`. In `src/main.ts`'s parity
  branch, `ok` gains `&& res.hsErr <= 1`, and the PARITY line gains
  `; hotspots worst ${res.hsErr.toFixed(3)} of tolerance (${res.hsWorst})`.
- [ ] **Step 4: run**. `npx vitest run` must PASS (constants, shader-twins). Start the dev server on :5173, then run
  `npm run verify:gpu`. Expected: `?parity` PASS, with the hotspot worst ≤ 1 and every other parity number unchanged.
  Also expected: all golden hashes unchanged, since the new functions have no caller in the renderer yet. Mutation
  check: set `HS_SIGMA = 2.6` and `?parity` must FAIL (G off by ~2 % at 2–3 M); then revert.
- [ ] **Step 5:** `git commit -m "Hotspots: WGSL twin (state on the f32 epoch split, Gaussian boost, Doppler factor) and ?parity cases"`

---

### Task 4: Renderer, mode switch, golden scene, `?hotflow` gate, app check, bench

**Files:** `src/render/raytrace.wgsl`, `src/main.ts`, `src/test/scenes.ts`, `src/test/hotflow.browser.ts`,
`scripts/verify-gpu.mjs`, `scripts/bench.mjs`, `src/test/golden.json` (recorded).

**Interfaces:**
- Consumes: Task 3's WGSL functions; Task 1's `hotspotAliveWindow`, `hotspotPeakTime`, `hotspotEndTime`, `HOTSPOT`;
  Task 2's `HOTSPOT_TWIN`, `imageCentroid`; `chooseMode(..., forceLive)`.
- Produces: `traceRay(pix, jit, record, hs: bool)`; `flowStep(..., accIn, hs: bool)`; scene `sgra-mm-hotspot`; dev hook
  `window.__bhSetTime(t: number)`.

- [ ] **Step 1: shader**. In `raytrace.wgsl`, replace `flowStep` with the version below. Update its header comment's
  last sentence to: "Hotspots (spec 2026-10-04 mm hotspots) add their boosted coefficients to the same slab on live frames (hs)."

```wgsl
fn flowStep(s: State, sNew: State, dl: f32, rh: f32, orb: vec3<f32>, accIn: vec2<f32>, hs: bool) -> vec2<f32> {
  let p0 = cartOf(s.x);
  let dvec = cartOf(sNew.x) - p0;
  if (dot(dvec, dvec) <= 1e-12 || chordMisses(p0, dvec, HF_RMAX)) { return accIn; }
  let n = clamp(u32(ceil(dl / (JET_DL * max(1.0, min(s.x.y, sNew.x.y) / 8.0)))), 1u, JET_NSUB_MAX);
  // Hotspot only on live frames (hs; the cache stores the steady flow), with the slider on (main.ts sends fluxVar 0 in mm
  // while no hotspot can be in view), and only on steps whose chord passes within HS_REACH of the hole.
  let hsOn = hs && U.fluxVar > 0.0 && !chordMisses(p0, dvec, HS_REACH);
  var acc = accIn;
  for (var k = 0u; k < n; k++) {
    let q = jetSample(s, p0, dvec, k, n);
    if (q.x >= HF_RMAX || q.x <= rh * 1.01) { continue; }
    let f = f32(k) / f32(n);
    let pk = mix(s.p, sNew.p, f);
    var jE = 0.0; var dTau = 0.0; // this sample's observed emission and optical depth (flow + hotspot, one slab)
    let D = flowShiftOrbJ(q.x, q.y, pk, U.a, orb);
    if (D > 1e-6) {
      let c = flowCoeffsJ(q.x, q.y, HF_LNNU + log(D), U.flowN0);
      if (c.x > 0.0) { let ds = U.rgCm * D * dl / f32(n); jE = c.x / (D * D * D) * ds; dTau = c.y * ds; }
    }
    if (hsOn && q.x < HS_REACH && abs(q.x * cos(q.y)) < HS_CUT * HS_SIGMA) {
      let tS = select(s.x.x + (sNew.x.x - s.x.x) * f, s.x.x, k == 0u);
      let st = hotspotStateJ(U.timeEpoch, emitRel(-tS - U.rObs), U.fluxVar, U.a);
      if (st.w > 0.0) {
        let b = st.z * hotspotBoostJ(q.x, q.y, q.z, st.x, st.y);
        if (b > 0.0) {
          let Dh = hotspotShiftJ(q.x, q.y, pk, U.a, 1.0 / (pow(st.x, 1.5) + U.a));
          if (Dh > 1e-6) {
            let ch = flowCoeffsJ(q.x, q.y, HF_LNNU + log(Dh), U.flowN0);
            let dsh = U.rgCm * Dh * dl / f32(n);
            jE += b * ch.x / (Dh * Dh * Dh) * dsh; dTau += b * ch.y * dsh;
          }
        }
      }
    }
    if (jE > 0.0 || dTau > 0.0) {
      let fac = select((1.0 - exp(-dTau)) / max(dTau, 1e-30), 1.0 - 0.5 * dTau, dTau < 1e-4);
      acc = vec2<f32>(acc.x + jE * fac * exp(-acc.y), min(acc.y + dTau, 1e30));
    }
  }
  return acc;
}
```

  `traceRay` gains a 4th parameter `hs: bool`, passed through: `flow = flowStep(s, sNew, dl, rh, orb, flow, hs);`.
  Callers: `main` → `traceRay(gid.xy, pixelJitter(gid.xy), false, true)`; `build` → `(…, !bandMm(), false)`;
  `audit` → `(…, false, false)`; `shade`'s KIND_LIVE fallback → `(…, false, false)` (a cached frame must equal the
  steady image).
- [ ] **Step 2: main loop** (`src/main.ts`). Import `{ HOTSPOT, hotspotAliveWindow }` from `./physics/hotspot`. In
  `loop`, after `n0` is computed:

```ts
    // 1.3 mm hotspots (spec 2026-10-04 mm hotspots): the cache holds the steady flow, so frames trace live while a hotspot
    // can be in view (alive within +-HOTSPOT.pad, the measured bound on light-travel delay). Otherwise mm frames get
    // fluxVar 0, which in mm drives only the hotspots (no jet at 1.3 mm), so live frames skip the hotspot code exactly.
    const hsLive = hot === 1 && hotspotAliveWindow(simTime, state.fluxVar, state.a, HOTSPOT.pad);
```

  Then: `chooseMode(state.playing, sched.completedSets, cacheOn, hsLive)`, and in `u`
  `fluxVar: mm ? (hsLive ? state.fluxVar : 0) : state.fluxVar`. Dev hook, next to `let simTime`:

```ts
  // verify:gpu's hotspot check jumps the clock (dev server only; absent from production builds).
  if (import.meta.env.DEV) (window as unknown as { __bhSetTime?: (t: number) => void }).__bhSetTime = (t) => { simTime = t; };
```

- [ ] **Step 3: scene**. In `src/test/scenes.ts`, import `hotspotPeakTime` from `../physics/hotspot` and append to
  `SCENES`, after `m87-mm`:

```ts
  // Sgr A* at the peak of eruption 0's hotspot, light delay on (spec 2026-10-04 mm hotspots; ?hotflow's hotspot gate).
  { ...mmScene("sgra-mm-hotspot", "sgra", 0, hotspotPeakTime(0, preset("sgra").a)), lightDelay: 1 },
```

  (`?golden` hashes it, and judges only recorded scenes, so this scene does not fail before it is recorded. Do not add it
  to `?cachecheck`: cached frames never show a hotspot.)
- [ ] **Step 4: `?hotflow` gate** (`src/test/hotflow.browser.ts`). Split `measure` into an
  `image(r, s, extra = {}): Promise<Float64Array>` that returns I_ν, and a flux helper. Keep `measure`'s rows and
  verdicts as they are. Then add:

```ts
// Hotspot (spec 2026-10-04 mm hotspots 5): Sgr A* at the peak of eruption 0's hotspot minus the same frame with the slider
// at 0 (in mm, fluxVar drives only the hotspots): the flux it adds and its centroid must match the CPU twin
// (HOTSPOT_TWIN, scripts/calibrate-hotspot.ts) within 5 % and 0.5 M, with light-travel delay on and off.
async function hotspotRow(r: Renderer, ld: 0 | 1) {
  const s = { ...SCENES.find((x) => x.name === "sgra-mm-hotspot")!, lightDelay: ld };
  const on = await image(r, s), off = await image(r, s, { fluxVar: 0 }), d = on.map((v, i) => v - off[i]);
  const want = ld ? HOTSPOT_TWIN.delay : HOTSPOT_TWIN.instant, jy = fluxJy(d, "sgra"), [cx, cy] = imageCentroid(d, N, 14);
  const ok = Math.abs(jy / want.jy - 1) < 0.05 && Math.hypot(cx - want.cx, cy - want.cy) < 0.5;
  return { ok, line: `hotspot (light delay ${ld}): adds ${jy.toFixed(3)} Jy at (${cx.toFixed(2)}, ${cy.toFixed(2)}) M; CPU twin ${want.jy} Jy at (${want.cx}, ${want.cy}) ${ok ? "ok" : "FAILED"}` };
}
```

  `runHotFlow` calls `hotspotRow(r, 1)` and `hotspotRow(r, 0)`, ANDs both `ok` into the verdict, and appends both lines.
  `fluxJy(I, id)` is the factored-out sum from `measure`: rg, dist and `half = 14`, as there.
- [ ] **Step 5: app check** (`scripts/verify-gpu.mjs`, after the 1.3 mm check, on the same `#p=sgra&b=mm` page):

```js
// 1.3 mm hotspots (spec 2026-10-04 mm hotspots): with the cache complete, jumping the clock to just before a hotspot's
// peak must switch the mode to live and change the image; jumping past its window must return to cached without a rebuild.
{
  const dHs = diags.length, steps = [];
  const clip = async () => { const vp = page.viewportSize(), cx = Math.round(vp.width / 2), cy = Math.round(vp.height / 2), R = 180;
    return (await page.screenshot({ clip: { x: cx - R, y: cy - R, width: 2 * R, height: 2 * R } })).toString("base64"); };
  const times = await page.evaluate(async () => {
    const hs = await import("/src/physics/hotspot.ts"), { PRESETS } = await import("/src/physics/presets.ts");
    const a = PRESETS.find((p) => p.id === "sgra").a;
    return { peak: hs.hotspotPeakTime(0, a), after: hs.hotspotEndTime(0, a) + hs.HOTSPOT.pad + 50 };
  });
  await page.evaluate(() => window.__bhSetTime(0)); await waitMode("cached"); await page.waitForTimeout(1000);
  const steady = await clip();
  await page.evaluate((t) => window.__bhSetTime(t), times.peak - 30);
  const live = step("hotspot->live", await waitMode("live", 5000)); await page.waitForTimeout(1500);
  const flare = await clip();
  const diff = await page.evaluate(async ([a, b]) => {
    const px = async (d) => { const img = new Image(); img.src = "data:image/png;base64," + d; await img.decode();
      const c = document.createElement("canvas"); c.width = img.width; c.height = img.height; const g = c.getContext("2d");
      g.drawImage(img, 0, 0); return g.getImageData(0, 0, c.width, c.height).data; };
    // Pixels that brightened by > 90 (sum over channels): the blob covers thousands; the live frame's lower internal
    // resolution only moves thin ring edges, both ways.
    const x = await px(a), y = await px(b); let n = 0;
    for (let i = 0; i < x.length; i += 4) if (y[i] - x[i] + y[i + 1] - x[i + 1] + y[i + 2] - x[i + 2] > 90) n++;
    return n;
  }, [steady, flare]);
  steps.push(`${diff} pixels brightened`);
  await page.evaluate((t) => window.__bhSetTime(t), times.after);
  const back = step("->cached (no rebuild)", await waitMode("cached", 3000));
  const ok = live && back && diff > 2000, hsDiag = diagSince(dHs);
  console.log(`${ok && !hsDiag ? "✓ PASS" : "✗ FAIL"}  1.3 mm hotspot: ${steps.join(", ")}${hsDiag}`);
  if (!ok || hsDiag) failed = true;
}
```

  Use the file's existing `step(name, ok)` helper if it has one with that shape (`grep -n "const step" scripts/verify-gpu.mjs`).
  Otherwise define it locally as `const step = (n, ok) => { steps.push(`${n} ${ok ? "ok" : "FAILED"}`); return ok; };`
  before its first use. The 3 s limit on `->cached` is what proves there was no rebuild, because a rebuild takes 16
  slices × 4 sets.
- [ ] **Step 6: bench** (`scripts/bench.mjs`). Change `bench(w, h, frames = 8, mm = false, t0 = 0)`, with `time: t0 + f`
  in `u` and, for mm, `fluxVar: t0 ? 1 : 0` (what the app sends: 0 with no hotspot in view). The `ONLY_MM` rows become
  `[await bench(1280, 720, 16, true), await bench(1280, 720, 16, true, hsPeak)]`. In the full run, add the hotspot row
  after the mm row. Compute `hsPeak` in the page with `(await import("/src/physics/hotspot.ts")).hotspotPeakTime(0, sg.a)`.
  Label the hotspot row `mm+hs` in the printout. To pass `t0` into the label, return it in the row object.
- [ ] **Step 7: gates**. With the dev server on :5173:
  - `npx vitest run` and `npm run build`: PASS.
  - `npm run verify:gpu` before recording. `?golden` must PASS, which means every recorded hash is unchanged (visible and
    `sgra-mm` / `m87-mm`); the new scene is not recorded yet. If a recorded hash moved, apply the stubbed-kernel proof:
    with `hsOn` forced to `false` the kernel must reproduce main's hashes. If it does, the move is compiler rescheduling
    and you re-record; if it does not, it is a bug.
  - `RECORD_GOLDEN=1 npm run verify:gpu`: every route PASSES, including `?parity`, `?shadow`, `?cachecheck` (0 incl.
    sgra-mm / m87-mm), `?hotflow` (both hotspot rows ok) and the hotspot app check. Commit the new `golden.json`.
  - `ONLY_MM=1 npm run bench` three times, interleaved with the same on `main` (git stash or a second checkout). Record
    the bench minima. Accept only if the `mm` row (no hotspot, fluxVar 0) is within 3 % of main. Report the `mm+hs` row.
- [ ] **Step 8:** `git commit -m "Renderer: 1.3 mm hotspots on live frames (one slab with the flow), live while a hotspot can be in view, sgra-mm-hotspot golden, ?hotflow twin gate, app check, bench"`

---

### Task 5: Captions, visual check, docs, review, ship

**Files:** `src/physics/presets.ts`, `index.html`, `README.md`, `docs/ROADMAP.md`, `docs/specs/2026-10-04-mm-hotspots-design.md`.

- [ ] **Step 1: copy**. Append these captions (`presets.ts`). Sgr A*: " At 1.3 mm a hotspot flares at each horizon-flux
  eruption (about every 1500 M, ~8.5 h here): a blob on a Keplerian orbit at 8–12 r_g that fades within three orbits, as
  ALMA saw after the X-ray flare of 2017 April 11 (Wielgus et al. 2022: ~11 r_g, 74 ± 6 min, up to ~0.3 Jy). Its
  brightness is set so it adds 0.3 Jy here." M87* and Gargantua: " At 1.3 mm hotspots flare at each flux eruption,
  scaled from Sgr A*'s (a model prediction)." Flux-variability tooltip (`index.html`), appended: " At 1.3 mm it also sets
  the hotspot flares (one per eruption; 0 turns them off)." Run `npx vitest run tests/presets.test.ts`. Expected: PASS.
- [ ] **Step 2: look at it**. Take screenshots with the clock at `hotspotPeakTime(0, a)` and at peak + P/2 (dev hook),
  for `#p=sgra&b=mm`, `#p=m87&b=mm` and `#p=gargantua&b=mm`. Check that the blob sits on its orbit, is brighter on the
  approaching side, has a lensed secondary image near the ring, and is clearly visible but fainter than the ring peak.
  Save them under `.superpowers/sdd/2026-10-07-mm-hotspots/` and describe them in the ledger.
- [ ] **Step 3: docs**. Add a README section "Hotspot flares at 1.3 mm" covering: the model and sources (Wielgus et al.
  2022; GRAVITY 2018), A₀ and the twin numbers from Task 2, the measured pad, live-frame cost from the bench, and the
  limitations (spec §6). Add the new gate numbers to the gates paragraph: npm test count, parity rows, the
  `sgra-mm-hotspot` hash. Add a ROADMAP entry. Spec status: "implemented (plan 2026-10-07)". Add a short "Departures"
  note listing the three items from this plan's header.
- [ ] **Step 4: final review**. Fresh reviewer on the most capable model, over the whole branch against the spec and
  this plan; then one fix pass, re-running the gates it touches.
- [ ] **Step 5: ship**. `git commit` the docs. Fast-forward `main` to the branch and `git push origin main` (standing
  rule). Do not deploy; report and offer `vercel deploy --prod` (PowerShell).
