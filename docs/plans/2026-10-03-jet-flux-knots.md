# Jet Knots from Horizon-Flux Variability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the jet's decorative knot noise and static churn with (1) the imprint of a MAD's horizon
magnetic-flux history, launched at the jet base and carried outward by the plasma, and (2) filaments frozen into
the moving, rotating plasma.

**Architecture:** A deterministic flux-history generator φ(t) (new `src/physics/flux-history.ts`) gives the flux
ratio f = φ/φ̄ at any time. Each jet sample looks up f at its plasma's *launch time* t − τ(z), with τ(z) the flow's
travel time from the base (6-point Gauss quadrature). The jet's width scales by √f and its density by f; the
mean-power normalisation `jetQ0` averages the real injected power over the flux distribution. Filaments are a 3-D
value noise on (launch time, co-moving azimuth, field-line label). `emission-shared.wgsl` stays the sole GPU copy,
prepended to the renderer and to `?parity`; the uniform slot `jetKnots` becomes `fluxVar`.

**Tech Stack:** TypeScript, WebGPU/WGSL, Vite, Vitest, vite-node (calibration script), Playwright (GPU gates).

**Spec:** `docs/specs/2026-10-03-jet-flux-knots-design.md` — read its "Corrections from planning" section; it
overrides §2.3 (no table: Gauss quadrature from z_base), §2.4 (mean power is a cubic average) and §2.5 (twist uses
τ − (|z| − z_base)).

## Global Constraints

- Flux generator: T = 1500 M; jitter j = 1/3 (gaps T(1 + j(u_{k+1} − u_k)) ∈ [1000, 2000] M); drop e-folding
  τ_d = 500 M; depth δ_k = δ̄ (1 + 0.5 v_k); linear refill; floor φ ≥ 0.05; slider s ∈ [0, 1.4], default 1.
- Calibrated constants (computed while planning with the project's `hash4`, salts below; Task 1's script
  reproduces them): `dbar: 0.5058`, `d1: 0.262371`, `d2: 0.092607`, `d3: 0.037282`; σ/μ = 0.2090; gaps
  1027.6–1984.5 M, mean 1499.97 M; max δ_k = 0.6322; max δ_k · 1.4 = 0.885 < 0.95 (floor never reached).
- Hash: u = (hash4(k, 0x7a11, 0, salt) & 0xffffff) / 16777216 − 0.5 with salt 0x464c (times) and 0x4458 (depths);
  filament salt 0x46494c. `hash4` / `hash4T` already exist (emission.ts / emission-shared.wgsl).
- Travel time from z_base = 2 M; τ(60 M) = 166 M at G₂₈₀ = 2. Field-line rotation Ω_F = a / (4 r₊),
  r₊ = 1 + √(1 − a²).
- Jet envelope widened: largest flux ratio f_max = 1/(1 − s_max d1) = 1.5806, width factor √f_max = 1.2572, so the
  envelope bound is q ≤ `JET_ENV_Q = 1.51` (was 1.2).
- Filaments: amplitude 0.35, cells 25 M in launch time, 8 around (periodic), 2.5 per unit q (3 across q ≤ 1.2).
- Gates before merge: `npm test`, `npm run build`, `npm run verify:gpu` (dev server on :5173 first; Prisma/Neon
  rule does not apply here, but run `vercel` only in PowerShell if ever asked). `?golden` is re-recorded by design
  (`RECORD_GOLDEN=1 npm run verify:gpu`); jet-off scenes must stay unchanged or differ only by the known
  compiler-rescheduling effect, proven with a jet-free kernel.
- Never junction `node_modules` into a worktree. No Co-Authored-By trailer in commits (user rule).

## Review Focus

1. **A long session (large clock epoch).** After hours of play the epoch is ~10⁶ M; the GPU must give the same flux
   and filaments as the CPU's f64 evaluation, without per-pixel grain. Pinned by Task 5's flux parity cases at
   epochs up to 2048 × 8000.
2. **The still-camera cache while the jet widens.** A ray cached as "misses the jet" must never be crossed by the
   widened jet. Pinned by Task 3's envelope sweep over f up to f_max and Task 6's `?cachecheck`.
3. **Slider at the extremes.** Flux variability 0 must reproduce a steady jet exactly (f ≡ 1, no width change);
   1.4 must never hit the 0.05 floor and must keep `jetQ0` finite. Pinned in Tasks 1, 3 and 4.
4. **Counter-jet and azimuth wrap.** The lower lobe (z < 0) must use |z| for travel time and twist; azimuths that
   wrap past 2π or are negative (unwrapped ray φ) must give continuous filaments. Pinned in Tasks 2 and 3.
5. **Spin 0 / low spin.** Ω_F = 0 at a = 0 (filaments stop twisting) and q0 = 0 (no jet) must not produce NaN.
   Pinned in Task 4 (q0 at a = 0) and Task 2 (Ω_F(0) = 0).

---

### Task 1: Flux-history generator (CPU)

**Files:**
- Create: `src/physics/flux-history.ts`
- Create: `scripts/calibrate-flux.ts`
- Test: `tests/flux-history.test.ts`

**Interfaces:**
- Consumes: `hash4(ix, iy, gen, salt): number` from `src/physics/emission.ts`.
- Produces:
  - `FLUX` constants object (fields `T, jitter, tauD, spread, floor, sMax, target, dbar, d1, d2, d3, saltT, saltD`).
  - `fluxHash(k: number, salt: number): number` in [−0.5, 0.5).
  - `eruptionTime(k: number): number`, `eruptionDepth(k: number, dbar?: number): number`.
  - `fluxDeficitLocal(k: number, loc: number, dbar?: number): number` — deficit given whole periods k and the
    remainder loc ∈ [0, T) (mirrors the WGSL split).
  - `fluxDeficit(t: number, dbar?: number): number` — deficit d(t) at s = 1.
  - `fluxRatio(t: number, s: number): number` — f = max(floor, 1 − s d)/(1 − s d1).
  - `fluxMoment(n: 2 | 3, s: number): number` — ⟨f^n⟩.
  - `measureFlux(dbar: number, span?: number, dt?: number): { d1: number; d2: number; d3: number; sigmaOverMu: number }`.

- [ ] **Step 1: Write the failing tests**

```ts
// tests/flux-history.test.ts
import { describe, it, expect } from "vitest";
import { FLUX, fluxHash, eruptionTime, eruptionDepth, fluxDeficit, fluxRatio, fluxMoment, measureFlux } from "../src/physics/flux-history";

describe("flux history (spec 2026-10-03 jet flux knots 2.2)", () => {
  it("eruptions recur every 1500 M on average, every gap inside [1000, 2000] M", () => {
    const gaps: number[] = [];
    for (let k = 0; k < 700; k++) gaps.push(eruptionTime(k + 1) - eruptionTime(k));
    const mean = gaps.reduce((x, y) => x + y) / gaps.length;
    expect(Math.abs(mean - 1500)).toBeLessThan(50);
    for (const g of gaps) { expect(g).toBeGreaterThanOrEqual(1000); expect(g).toBeLessThanOrEqual(2000); }
  });
  it("hash is uniform in [-0.5, 0.5) and deterministic", () => {
    for (let k = -50; k < 50; k++) { const u = fluxHash(k, FLUX.saltT); expect(u).toBeGreaterThanOrEqual(-0.5); expect(u).toBeLessThan(0.5); expect(fluxHash(k, FLUX.saltT)).toBe(u); }
  });
  it("drops are exponential with e-folding 500 M (Ripperda+2022), then refill linearly to zero", () => {
    for (const k of [3, 17, 401]) {
      const tk = eruptionTime(k), dk = eruptionDepth(k), D = -FLUX.tauD * Math.log(1 - dk);
      expect(fluxDeficit(tk + 1e-6)).toBeLessThan(1e-6);                       // starts at 0
      expect(fluxDeficit(tk + 100)).toBeCloseTo(1 - Math.exp(-100 / 500), 12);  // exponential
      expect(fluxDeficit(tk + D - 1e-6)).toBeCloseTo(dk, 6);                     // reaches the depth
      const tn = eruptionTime(k + 1), mid = 0.5 * (tk + D + tn);
      expect(fluxDeficit(mid)).toBeCloseTo(0.5 * dk, 6);                         // linear refill
      expect(fluxDeficit(tn - 1e-6)).toBeLessThan(1e-6);                         // continuous at the next eruption
    }
  });
  it("swing sigma/mu = 0.209 at s = 1 (Narayan+2022 a = 0.9) and the hardcoded moments match the generator", () => {
    const m = measureFlux(FLUX.dbar, 1e6, 1);
    expect(m.sigmaOverMu).toBeCloseTo(FLUX.target, 3);
    expect(Math.abs(m.d1 / FLUX.d1 - 1)).toBeLessThan(1e-4);
    expect(Math.abs(m.d2 / FLUX.d2 - 1)).toBeLessThan(1e-4);
    expect(Math.abs(m.d3 / FLUX.d3 - 1)).toBeLessThan(1e-4);
  });
  it("the floor is never reached for s <= sMax (Review Focus 3)", () => {
    let maxDepth = 0;
    for (let k = -10; k < 5000; k++) maxDepth = Math.max(maxDepth, eruptionDepth(k));
    expect(maxDepth * FLUX.sMax).toBeLessThan(1 - FLUX.floor);
  });
  it("s = 0 is a steady jet: f = 1 at all times (Review Focus 3)", () => {
    for (const t of [0, 123.4, 9e5, -50]) expect(fluxRatio(t, 0)).toBe(1);
  });
  it("f has mean one and the closed-form moments match a brute-force time average", () => {
    for (const s of [0.5, 1, 1.4]) {
      let m1 = 0, m2 = 0, m3 = 0, n = 0;
      for (let t = 0.5; t < 3e5; t += 1) { const f = fluxRatio(t, s); m1 += f; m2 += f * f; m3 += f * f * f; n++; }
      expect(m1 / n).toBeCloseTo(1, 2);
      expect(Math.abs(m2 / n / fluxMoment(2, s) - 1)).toBeLessThan(3e-3);
      expect(Math.abs(m3 / n / fluxMoment(3, s) - 1)).toBeLessThan(5e-3);
    }
  });
  it("negative and huge times work (the clock never runs backwards, but launch times can precede 0)", () => {
    for (const t of [-1400, -1, 0, 4.2e7]) { const d = fluxDeficit(t); expect(d).toBeGreaterThanOrEqual(0); expect(d).toBeLessThan(1); }
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run tests/flux-history.test.ts`
Expected: FAIL — `Cannot find module '../src/physics/flux-history'`.

- [ ] **Step 3: Write the implementation**

```ts
// src/physics/flux-history.ts
// Horizon magnetic-flux history phi_BH(t) of a magnetically arrested disk (spec 2026-10-03 jet flux knots, 2.2):
// a deterministic sawtooth -- eruptions every ~1500 M (arXiv 2510.25842), each an exponential drop with the
// converged reconnection e-folding of 500 M (Ripperda et al. 2022), then a linear refill as gas re-advects flux.
// Depth calibrated so phi's rms swing is 20.9 % (Narayan et al. 2022, a = 0.9). WGSL twin: fluxRatioJ in
// emission-shared.wgsl (same hash, same split into whole periods + remainder).
import { hash4 } from "./emission";

export const FLUX = {
  T: 1500,            // mean recurrence (M)
  jitter: 1 / 3,      // eruption k at T (k + 0.5 + jitter u_k): gaps in [1000, 2000] M
  tauD: 500,          // drop e-folding (M)
  spread: 0.5,        // depth_k = dbar (1 + spread v_k)
  floor: 0.05,        // phi never below 5 % of saturation
  sMax: 1.4,          // Flux variability slider maximum (floor unreachable below it)
  target: 0.209,      // sigma/mu of phi at s = 1
  dbar: 0.5058,       // mean depth (scripts/calibrate-flux.ts)
  d1: 0.262371, d2: 0.092607, d3: 0.037282, // <d>, <d^2>, <d^3> at s = 1 over 1e6 M (same script)
  saltT: 0x464c, saltD: 0x4458,
} as const;

/** Uniform in [-0.5, 0.5) from eruption index k (twin: fluxHashJ). */
export function fluxHash(k: number, salt: number): number {
  return (hash4(k, 0x7a11, 0, salt) & 0xffffff) / 16777216 - 0.5;
}
export function eruptionTime(k: number): number { return FLUX.T * (k + 0.5 + FLUX.jitter * fluxHash(k, FLUX.saltT)); }
export function eruptionDepth(k: number, dbar: number = FLUX.dbar): number { return dbar * (1 + FLUX.spread * fluxHash(k, FLUX.saltD)); }

/** Deficit at time k T + loc, loc in [0, T): eruption k (at T(0.5 + jitter u_k) into its period) or, before it,
 *  eruption k - 1 still refilling. Eruption k + 1 is always later than (k + 1) T + T/3 > t. */
export function fluxDeficitLocal(k: number, loc: number, dbar: number = FLUX.dbar): number {
  let kk = k, dt = loc - FLUX.T * (0.5 + FLUX.jitter * fluxHash(k, FLUX.saltT));
  if (dt < 0) { kk = k - 1; dt = loc + FLUX.T - FLUX.T * (0.5 + FLUX.jitter * fluxHash(kk, FLUX.saltT)); }
  const gap = FLUX.T * (1 + FLUX.jitter * (fluxHash(kk + 1, FLUX.saltT) - fluxHash(kk, FLUX.saltT)));
  const depth = eruptionDepth(kk, dbar), D = -FLUX.tauD * Math.log(1 - depth);
  if (dt < D) return 1 - Math.exp(-dt / FLUX.tauD);
  return (depth * (gap - dt)) / (gap - D);
}
export function fluxDeficit(t: number, dbar: number = FLUX.dbar): number {
  const k = Math.floor(t / FLUX.T);
  return fluxDeficitLocal(k, t - k * FLUX.T, dbar);
}
/** f = phi / mean(phi) at time t for slider s (s = 0: exactly 1). */
export function fluxRatio(t: number, s: number): number {
  if (s === 0) return 1;
  return Math.max(FLUX.floor, 1 - s * fluxDeficit(t)) / (1 - s * FLUX.d1);
}
/** <f^n> for n = 2, 3 from the generator's deficit moments (floor unreachable for s <= sMax). */
export function fluxMoment(n: 2 | 3, s: number): number {
  const mu = 1 - s * FLUX.d1;
  if (n === 2) return (1 - 2 * s * FLUX.d1 + s * s * FLUX.d2) / (mu * mu);
  return (1 - 3 * s * FLUX.d1 + 3 * s * s * FLUX.d2 - s ** 3 * FLUX.d3) / mu ** 3;
}
/** Deficit moments and phi's sigma/mu at s = 1, sampled every dt over [0, span) (midpoints). */
export function measureFlux(dbar: number, span = 1e6, dt = 1) {
  let n = 0, a = 0, b = 0, c = 0;
  for (let t = dt / 2; t < span; t += dt) { const d = fluxDeficit(t, dbar); a += d; b += d * d; c += d * d * d; n++; }
  a /= n; b /= n; c /= n;
  return { d1: a, d2: b, d3: c, sigmaOverMu: Math.sqrt(b - a * a) / (1 - a) };
}
```

```ts
// scripts/calibrate-flux.ts -- reproduces FLUX.dbar, d1, d2, d3 (run: npx vite-node scripts/calibrate-flux.ts).
import { FLUX, measureFlux } from "../src/physics/flux-history";
let lo = 0.3, hi = 0.7;
for (let i = 0; i < 40; i++) { const m = (lo + hi) / 2; if (measureFlux(m, 1e6, 2).sigmaOverMu < FLUX.target) lo = m; else hi = m; }
const dbar = Math.round(lo * 1e4) / 1e4, M = measureFlux(dbar, 1e6, 1);
console.log(`dbar: ${dbar}, d1: ${M.d1.toFixed(6)}, d2: ${M.d2.toFixed(6)}, d3: ${M.d3.toFixed(6)}, sigma/mu ${M.sigmaOverMu.toFixed(4)}`);
```

- [ ] **Step 4: Reproduce the constants and run the tests**

Run: `npx vite-node scripts/calibrate-flux.ts`
Expected: `dbar: 0.5058, d1: 0.262371, d2: 0.092607, d3: 0.037282, sigma/mu 0.2090`. If any digit differs, the
hash wiring differs from the planning prototype: fix the wiring, do not edit the constants to match.

Run: `npx vitest run tests/flux-history.test.ts`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/physics/flux-history.ts scripts/calibrate-flux.ts tests/flux-history.test.ts
git commit -m "Flux history: MAD horizon-flux sawtooth (1500 M recurrence, 500 M drop e-folding, 20.9 % swing), calibration script"
```

---

### Task 2: Travel time and co-moving azimuth (CPU)

**Files:**
- Modify: `src/physics/jet.ts` (add after `gammaProfile`)
- Test: `tests/jet.test.ts` (new `describe`)

**Interfaces:**
- Consumes: `JET.zBase`, `GAMMA_REF_Z`, `GAMMA_SLOPE`, `gammaProfile` (jet.ts); `horizonOuter(a)` (kerr.ts).
- Produces:
  - `launchDelay(z: number, g280: number): number` — τ(|z|) from z_base (0 below it).
  - `fieldLineOmega(a: number): number` — Ω_F = a / (4 r₊).
  - `comovingAzimuth(ph: number, z: number, a: number, g280: number): number` — φ − Ω_F (τ − (|z| − z_base)).

- [ ] **Step 1: Write the failing tests** (append to `tests/jet.test.ts`; extend its import from
  `../src/physics/jet` with `launchDelay, fieldLineOmega, comovingAzimuth, gammaProfile`)

```ts
describe("plasma travel time and co-moving azimuth (spec 2.3, 2.5 + corrections)", () => {
  // 20 000-point midpoint reference in v = z^(1-p): tau = Int sqrt(1 + A^2 v^(2p/(1-p))) dv / (A (1-p)).
  const tauRef = (z: number, g: number) => {
    const p = 0.58, A = Math.sqrt(g * g - 1) / Math.pow(280, p), e = (2 * p) / (1 - p);
    const v0 = Math.pow(2, 1 - p), v1 = Math.pow(Math.abs(z), 1 - p), N = 20000; let s = 0;
    for (let i = 0; i < N; i++) { const v = v0 + ((i + 0.5) / N) * (v1 - v0); s += Math.sqrt(1 + A * A * Math.pow(v, e)); }
    return (s * (v1 - v0)) / N / (A * (1 - p));
  };
  it("6-point Gauss quadrature matches the reference to 1e-4 (G280 1.5-8, z to 1000 M)", () => {
    for (const g of [1.5, 2, 3, 5, 8]) for (const z of [2.5, 5, 10, 30, 60, 200, 1000]) {
      expect(Math.abs(launchDelay(z, g) / tauRef(z, g) - 1)).toBeLessThan(1e-4);
      expect(launchDelay(-z, g)).toBe(launchDelay(z, g)); // counter-jet uses |z| (Review Focus 4)
    }
    expect(launchDelay(1.5, 2)).toBe(0);                  // below the base
    expect(launchDelay(60, 2)).toBeCloseTo(166, 0);       // spec 2.3
  });
  it("a parcel integrated along the flow keeps its launch time and co-moving azimuth", () => {
    const g = 2, a = 0.9, OmF = fieldLineOmega(a);
    let z = 2, ph = 0.3, t = 1000; const dt = 0.01;
    const beta = (zz: number) => { const G = gammaProfile(zz, g); return Math.sqrt(1 - 1 / (G * G)); };
    // start just above the base with an exact first step (beta(2) > 0)
    for (let k = 0; k < 20000; k++) {
      const b1 = beta(z), b2 = beta(z + 0.5 * dt * b1), b3 = beta(z + 0.5 * dt * b2), b4 = beta(z + dt * b3);
      const dz = (dt / 6) * (b1 + 2 * b2 + 2 * b3 + b4);
      ph += OmF * (dt - dz); z += dz; t += dt;   // dphi/dt = Omega_F (1 - beta)
    }
    expect(z).toBeGreaterThan(30);
    expect(t - launchDelay(z, g)).toBeCloseTo(1000, 2);
    expect(comovingAzimuth(ph, z, a, g)).toBeCloseTo(0.3, 3);
  });
  it("field-line rotation is half the horizon's: a / (4 r+), zero at spin 0 (Review Focus 5)", () => {
    expect(fieldLineOmega(0)).toBe(0);
    expect(fieldLineOmega(0.9)).toBeCloseTo(0.9 / (4 * (1 + Math.sqrt(1 - 0.81))), 12);
    expect(comovingAzimuth(1.0, 30, 0, 2)).toBe(1.0);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/jet.test.ts`
Expected: FAIL — `launchDelay is not a function` (import undefined).

- [ ] **Step 3: Implement** (in `src/physics/jet.ts`; add `import { horizonOuter } from "./kerr";` at the top)

```ts
/** Gauss-Legendre 6-point nodes and weights on [-1, 1] (twin: GL6 constants in emission-shared.wgsl). */
export const GL6: readonly [number, number][] = [
  [-0.9324695142031521, 0.1713244923791704], [-0.6612093864662645, 0.3607615730481386],
  [-0.2386191860831969, 0.4679139345726910], [0.2386191860831969, 0.4679139345726910],
  [0.6612093864662645, 0.3607615730481386], [0.9324695142031521, 0.1713244923791704],
];
/** Coordinate time (M) for plasma to climb from the jet base z_base to |z| at the flow law Gamma beta = A z^p
 *  (spec 2026-10-03 jet flux knots, corrections 1): with v = z^(1-p), tau = Int sqrt(1 + A^2 v^(2p/(1-p))) dv
 *  / (A (1-p)) -- smooth in v, so 6-point Gauss is accurate to 3e-5. Twin: launchDelayJ. */
export function launchDelay(z: number, g280: number): number {
  const az = Math.abs(z);
  if (az <= JET.zBase) return 0;
  const p = GAMMA_SLOPE, A = Math.sqrt(Math.max(1e-12, g280 * g280 - 1)) / Math.pow(GAMMA_REF_Z, p), e = (2 * p) / (1 - p);
  const v0 = Math.pow(JET.zBase, 1 - p), v1 = Math.pow(az, 1 - p), h = 0.5 * (v1 - v0), m = 0.5 * (v1 + v0);
  let s = 0;
  for (const [x, w] of GL6) s += w * Math.sqrt(1 + A * A * Math.pow(m + h * x, e));
  return (s * h) / (A * (1 - p));
}
/** Blandford-Znajek field-line angular velocity Omega_F = Omega_H / 2 = a / (4 r+) (twin: fieldLineOmegaJ). */
export function fieldLineOmega(a: number): number { return a / (4 * horizonOuter(a)); }
/** Azimuth a plasma parcel had at launch: the parcel turns at Omega = Omega_F (1 - beta) (v_phi = Omega_F rho +
 *  v_p B_phi / B_p, B_phi / B_p = -Omega_F rho / c), so Int Omega dt = Omega_F (tau - (|z| - z_base)). */
export function comovingAzimuth(ph: number, z: number, a: number, g280: number): number {
  const az = Math.abs(z);
  if (az <= JET.zBase) return ph;
  return ph - fieldLineOmega(a) * (launchDelay(z, g280) - (az - JET.zBase));
}
```

- [ ] **Step 4: Run tests**

Run: `npx vitest run tests/jet.test.ts`
Expected: PASS (new 3 tests plus the existing ones).

- [ ] **Step 5: Commit**

```bash
git add src/physics/jet.ts tests/jet.test.ts
git commit -m "Jet: plasma travel time from the base (6-point Gauss) and co-moving azimuth (Omega_F (1 - beta))"
```

---

### Task 3: Jet shape driven by the flux history, filaments, widened envelope (CPU)

**Files:**
- Modify: `src/physics/jet.ts` (replace `knots`, `jetShape`, `inJetEnvelope`; drop `JET.kz`, `JET.knotSeed`,
  `JET.turbAmpJet`; add `FILAMENT`, `JET_ENV_Q`, `vnoise3`, `filaments`)
- Modify: `tests/jet.test.ts` (replace the "jet living emission field" and envelope describes)

**Interfaces:**
- Consumes: `fluxRatio(t, s)`, `FLUX` (Task 1); `launchDelay`, `comovingAzimuth` (Task 2); `hash4` (emission.ts).
- Produces:
  - `FILAMENT = { amp: 0.35, cellT: 25, cellsPhi: 8, cellsQ: 2.5, salt: 0x46494c }`.
  - `JET_ENV_Q = 1.51` (exported const).
  - `vnoise3(x: number, y: number, n: number, w: number, salt: number): number` in [0, 1]; periodic in y with period n.
  - `filaments(q: number, phiC: number, tLaunch: number): number` (mean one).
  - `jetShape(r: number, th: number, ph: number, t: number, jetLength: number, fluxVar: number, g280: number, a: number): number`.
  - `inJetEnvelope(r: number, th: number, jetLength: number): boolean` (same signature, wider bound).

- [ ] **Step 1: Write the failing tests** — in `tests/jet.test.ts` replace the import of `knots` with
  `FILAMENT, JET_ENV_Q, vnoise3, filaments`, delete `import { vnoise } from "../src/physics/emission";`, add
  `import { FLUX, fluxRatio } from "../src/physics/flux-history";`,
  delete the describe "jet living emission field" and the envelope describe, and add:

```ts
describe("flux-driven jet shape (spec 2.4, 2.5)", () => {
  const thWall = (r: number) => Math.atan2(funnelEdge(r) * JET.qPeak, r); // approx. on the wall peak
  it("zero outside the jet band, positive inside (both lobes)", () => {
    expect(jetShape(8, 0.12, 0, 0, 60, 1, 2, 0.9)).toBeGreaterThan(0);
    expect(jetShape(8, Math.PI - 0.12, 0, 0, 60, 1, 2, 0.9)).toBeGreaterThan(0);
    expect(jetShape(1.5, 0.12, 0, 0, 60, 1, 2, 0.9)).toBe(0);        // below zBase
    expect(jetShape(400, 0.12, 0, 0, 60, 1, 2, 0.9)).toBe(0);        // beyond jetLength
    expect(jetShape(8, Math.PI / 2, 0, 0, 60, 1, 2, 0.9)).toBe(0);   // equatorial
  });
  it("density x f and width x sqrt(f) at the plasma's launch time", () => {
    const r = 30, th = thWall(30), z = r * Math.cos(th), rho = r * Math.sin(th);
    for (const t of [500, 2100, 7777]) {
      const tl = t - launchDelay(z, 2), f = fluxRatio(tl, 1), sw = Math.sqrt(f);
      const q = rho / (sw * funnelEdge(z));
      const want = f * wallProfile(rho / sw, z) * lengthFalloff(z, 60) * filaments(q, comovingAzimuth(0.4, z, 0.9, 2), tl);
      expect(jetShape(r, th, 0.4, t, 60, 1, 2, 0.9)).toBeCloseTo(want, 12);
    }
  });
  it("the pattern rides the flow: what the base launched appears at height z after tau(z)", () => {
    // Same launch time and co-moving azimuth => same flux ratio and filament value at both heights.
    const a = 0.9, g = 2, t0 = 3000, ph0 = 1.1;
    for (const z of [10, 40]) {
      const tl = t0, t = tl + launchDelay(z, g), ph = ph0 + fieldLineOmega(a) * (launchDelay(z, g) - (z - JET.zBase));
      expect(comovingAzimuth(ph, z, a, g)).toBeCloseTo(ph0, 10);
      expect(fluxRatio(t - launchDelay(z, g), 1)).toBeCloseTo(fluxRatio(tl, 1), 12);
    }
  });
  it("s = 0 gives the steady jet: wall x falloff x filaments, no width change (Review Focus 3)", () => {
    const r = 20, th = thWall(20), z = r * Math.cos(th), rho = r * Math.sin(th), tl = 900 - launchDelay(z, 2);
    const q = rho / funnelEdge(z);
    expect(jetShape(r, th, 0.2, 900, 60, 0, 2, 0.9))
      .toBeCloseTo(wallProfile(rho, z) * lengthFalloff(z, 60) * filaments(q, comovingAzimuth(0.2, z, 0.9, 2), tl), 12);
  });
  it("filaments: mean one, amplitude bound 0.35, periodic and continuous across the azimuth wrap (Review Focus 4)", () => {
    let m = 0, n = 0;
    for (let i = 0; i < 20000; i++) { const v = filaments(0.3 + (i % 7) * 0.13, i * 0.731, i * 3.17); m += v; n++;
      expect(v).toBeGreaterThanOrEqual(1 - FILAMENT.amp - 1e-12); expect(v).toBeLessThanOrEqual(1 + FILAMENT.amp + 1e-12); }
    expect(m / n).toBeCloseTo(1, 1);
    for (const ph of [0, 1, -2.5]) expect(filaments(0.8, ph + 2 * Math.PI, 444)).toBeCloseTo(filaments(0.8, ph, 444), 10);
    expect(Math.abs(filaments(0.8, 2 * Math.PI - 1e-9, 444) - filaments(0.8, 1e-9, 444))).toBeLessThan(1e-6);
  });
  it("vnoise3 interpolates node values and wraps y with period n", () => {
    expect(vnoise3(3, 2, 8, 5, 7)).toBeCloseTo(vnoise3(3, 10, 8, 5, 7), 12);
    expect(vnoise3(3.5, 2.25, 8, 5.75, 7)).toBeGreaterThanOrEqual(0);
    expect(vnoise3(3.5, 2.25, 8, 5.75, 7)).toBeLessThanOrEqual(1);
  });
});

describe("jet envelope (geodesic-cache bookmark region), widened for the flux-driven width", () => {
  it("JET_ENV_Q covers the widest jet the slider allows", () => {
    expect(JET_ENV_Q).toBeGreaterThanOrEqual(1.2 * Math.sqrt(1 / (1 - FLUX.sMax * FLUX.d1)));
  });
  it("contains every point where the jet can emit, at any flux and slider value (Review Focus 2)", () => {
    for (let r = 1.2; r < 80; r *= 1.07)
      for (let th = 0.001; th < Math.PI; th += 0.013)
        for (const t of [0, 333, 1777, 2950]) for (const s of [0, 1, FLUX.sMax])
          if (jetShape(r, th, 0.5, t, 60, s, 2, 0.9) > 0) expect(inJetEnvelope(r, th, 60)).toBe(true);
  });
  it("excludes below the launch height, beyond the length, and outside the wall", () => {
    expect(inJetEnvelope(1.5, 0.01, 60)).toBe(false);
    expect(inJetEnvelope(70, 0.01, 60)).toBe(false);
    expect(inJetEnvelope(20, Math.PI / 2 - 0.2, 60)).toBe(false);
    expect(inJetEnvelope(20, 0.03, 60)).toBe(true);
    expect(inJetEnvelope(20, Math.PI - 0.03, 60)).toBe(true);
  });
  it("does not depend on jet strength or the flux slider (it takes neither)", () => {
    expect(inJetEnvelope.length).toBe(3);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/jet.test.ts`
Expected: FAIL — `filaments`/`vnoise3`/`JET_ENV_Q` undefined, `jetShape` arity mismatch.

- [ ] **Step 3: Implement** in `src/physics/jet.ts`: delete `knots` (and its two doc comments), delete `kz`,
  `turbAmpJet`, `knotSeed` from `JET`, change the import to `import { hash4 } from "./emission";` (drop `vnoise`),
  add `import { fluxRatio } from "./flux-history";`, update the file header comment (no longer "reuses the Tier 2A
value-noise basis"; filaments use the turbulence hash), and replace `jetShape` / `inJetEnvelope` with:

```ts
/** Filaments frozen into the moving plasma (spec 2.5): value noise on (launch time, co-moving azimuth, field-line
 *  label). Strength is illustrative (no measurement fixes it); the motion is physical. Twin: filamentsJ. */
export const FILAMENT = { amp: 0.35, cellT: 25, cellsPhi: 8, cellsQ: 2.5, salt: 0x46494c } as const;
/** Envelope bound on q = rho / rho_f: 1.2 (the wall's cut) times the widest width factor the slider allows,
 *  sqrt(1 / (1 - sMax d1)) = 1.2572, rounded up. Twin: JET_ENV_Q in emission-shared.wgsl. */
export const JET_ENV_Q = 1.51;

const smooth = (t: number) => t * t * (3 - 2 * t);
const node3 = (ix: number, iy: number, iw: number, salt: number) => (hash4(ix, iy, iw, salt) & 0xffffff) / 16777216;
/** Smooth value noise in [0, 1] on (x, y, w), periodic in y with period n (twin: vnoise3J). */
export function vnoise3(x: number, y: number, n: number, w: number, salt: number): number {
  const ix = Math.floor(x), iy = Math.floor(y), iw = Math.floor(w);
  const fx = smooth(x - ix), fy = smooth(y - iy), fw = smooth(w - iw);
  const y0 = ((iy % n) + n) % n, y1 = (y0 + 1) % n;
  let v = 0;
  for (let c = 0; c < 8; c++) {
    const dx = c & 1, dy = (c >> 1) & 1, dw = (c >> 2) & 1;
    const wt = (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy) * (dw ? fw : 1 - fw);
    v += wt * node3(ix + dx, dy ? y1 : y0, iw + dw, salt);
  }
  return v;
}
export function filaments(q: number, phiC: number, tLaunch: number): number {
  const turns = phiC / (2 * Math.PI), y = (turns - Math.floor(turns)) * FILAMENT.cellsPhi;
  return 1 + FILAMENT.amp * (vnoise3(tLaunch / FILAMENT.cellT, y, FILAMENT.cellsPhi, q * FILAMENT.cellsQ, FILAMENT.salt) - 0.5) * 2;
}

/** Density modulation of the synchrotron jet (spec 2026-10-03 jet flux knots 2.4): the plasma at height z left
 *  the base at t - tau(z), when the horizon flux ratio was f; the funnel there is sqrt(f) wider and f denser
 *  (power per length ~ f^2 ~ phi^2), times the co-moving filaments. 0 outside the emitting region.
 *  t is absolute coordinate time (M); a only sets the filaments' twist. Twin: jetShapeJ. */
export function jetShape(r: number, th: number, ph: number, t: number, jetLength: number, fluxVar: number, g280: number, a: number): number {
  const z = r * Math.cos(th), az = Math.abs(z);
  if (az < JET.zBase || az > jetLength) return 0;
  const tl = t - launchDelay(z, g280), f = fluxRatio(tl, fluxVar), sw = Math.sqrt(f);
  const rho = r * Math.sin(th), w = wallProfile(rho / sw, z);
  if (w <= 0) return 0;
  const q = rho / (sw * funnelEdge(z));
  return Math.max(0, f * w * lengthFalloff(z, jetLength) * filaments(q, comovingAzimuth(ph, z, a, g280), tl));
}

/** True where jetShape can be non-zero for SOME time and slider value: zBase <= |z| <= jetLength and
 *  q <= JET_ENV_Q. Purely geometric, so the geodesic cache's jet bookmark never depends on the jet switch,
 *  brightness or flux (spec 2026-10-01 3.3). WGSL twin: inJetEnvelope in raytrace.wgsl. */
export function inJetEnvelope(r: number, th: number, jetLength: number): boolean {
  const z = r * Math.cos(th), az = Math.abs(z);
  if (az < JET.zBase || az > jetLength) return false;
  return (r * Math.sin(th)) / funnelEdge(z) <= JET_ENV_Q;
}
```

Then fix the other CPU callers of `jetShape` so the type-check passes (they get the full update in Tasks 4–5;
here only the arity): `src/physics/synchrotron.ts` does not call `jetShape` (verify with
`grep -n "jetShape" src/physics/*.ts`); `src/test/parity.browser.ts` is updated in Task 5 — leave it, `npm test`
does not compile it.

- [ ] **Step 4: Run tests**

Run: `npx vitest run tests/jet.test.ts tests/flux-history.test.ts`
Expected: PASS. The envelope sweep is ~180 k jetShape calls (62 radii x 242 angles x 4 times x 3 slider values),
a few seconds.

- [ ] **Step 5: Commit**

```bash
git add src/physics/jet.ts tests/jet.test.ts
git commit -m "Jet shape from the flux history at launch time (width x sqrt f, density x f), co-moving filaments, envelope q <= 1.51"
```

---

### Task 4: Mean-power normalisation (CPU energy budget)

**Files:**
- Modify: `src/physics/synchrotron.ts` (`regionSum`, `injUnit`, new `meanInjection`, `jetUniforms`)
- Modify: `tests/synchrotron.test.ts` (calls of `jetUniforms` gain the slider argument; new tests)

**Interfaces:**
- Consumes: `fluxMoment(n, s)` (Task 1); `wallProfile`, `lengthFalloff`, `funnelEdge` (jet.ts).
- Produces:
  - `injUnit(a, b0, rgCm, jetLength, g280, f = 1): number` — injected power at q0 = 1 with the whole jet at flux ratio f.
  - `meanInjection(a, b0, rgCm, jetLength, g280, fluxVar): number` — ⟨injUnit(f)⟩ over the flux distribution.
  - `jetUniforms(mSun, a, lambda, eta, jetLength, g280, fluxVar)` — `fluxVar` is now REQUIRED (7th argument).

- [ ] **Step 1: Write the failing tests** (in `tests/synchrotron.test.ts`; add `meanInjection` to the synchrotron
  import and `import { fluxMoment, fluxRatio } from "../src/physics/flux-history";`)

```ts
describe("mean injected power with the flux-driven jet (spec 2.4 + corrections 2)", () => {
  const E = jetEnergetics(6.5e9, 0.9, 1e-5);
  const P = (f: number) => injUnit(0.9, E.b0, E.rgCm, 60, 2, f);
  it("injected power is ~f^2 (A + C f): the cubic through f = 0.5, 1, 1.5, 2 holds at other f within 0.5 %", () => {
    // Recover the cubic from meanInjection's own construction indirectly: evaluate P at extra points and compare
    // with the Lagrange cubic through the four nodes.
    const F = [0.5, 1, 1.5, 2], Pn = F.map(P);
    const lag = (f: number) => F.reduce((s, fi, i) => s + Pn[i] * F.reduce((p, fj, j) => (j === i ? p : (p * (f - fj)) / (fi - fj)), 1), 0);
    for (const f of [0.17, 0.3, 0.75, 1.25, 1.58]) expect(Math.abs(lag(f) / P(f) - 1)).toBeLessThan(5e-3);
    expect(P(1)).toBeCloseTo(injUnit(0.9, E.b0, E.rgCm, 60, 2), 12); // default argument f = 1
  });
  it("s = 0 reproduces the steady jet; s > 0 averages the cubic with <f^2>, <f^3>", () => {
    expect(meanInjection(0.9, E.b0, E.rgCm, 60, 2, 0)).toBe(P(1));
    // brute-force distribution average of the cubic over the generator (exact P only at the nodes; cubic checked above)
    const F = [0.5, 1, 1.5, 2], Pn = F.map(P);
    const lag = (f: number) => F.reduce((s, fi, i) => s + Pn[i] * F.reduce((p, fj, j) => (j === i ? p : (p * (f - fj)) / (fi - fj)), 1), 0);
    for (const s of [0.5, 1, 1.4]) {
      let m = 0, n = 0; for (let t = 0.5; t < 2e5; t += 3) { m += lag(fluxRatio(t, s)); n++; }
      expect(Math.abs(meanInjection(0.9, E.b0, E.rgCm, 60, 2, s) / (m / n) - 1)).toBeLessThan(1e-2);
    }
  });
  it("time-averaged injected power equals eta P_BZ at every slider value (spec 5 energy test)", () => {
    for (const s of [0, 0.5, 1, 1.4]) {
      const U = jetUniforms(6.5e9, 0.9, 1e-5, 2e-3, 60, 2, s);
      expect((U.jetQ0 * meanInjection(0.9, U.jetB0, U.rgCm, 60, 2, s)) / (2e-3 * U.pBZ)).toBeCloseTo(1, 9);
      expect(Number.isFinite(U.jetQ0)).toBe(true);
    }
  });
  it("spin 0: no BZ power, q0 = 0, no NaN at any slider value (Review Focus 5)", () => {
    for (const s of [0, 1, 1.4]) expect(jetUniforms(10, 0, 0.1, 2e-3, 60, 2, s).jetQ0).toBe(0);
  });
});
```

Update the existing calls: `jetUniforms(m, a, l, eta, 60, 2)` → `jetUniforms(m, a, l, eta, 60, 2, 0)` (the "equals
eta P_BZ" and "default eta reproduces M87's optical nucleus" tests: the anchor is the steady jet, s = 0) and
`jetUniforms(m, a, l, eta, 60, g)` → `jetUniforms(m, a, l, eta, 60, g, 0)`.

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/synchrotron.test.ts`
Expected: FAIL — `meanInjection is not a function`.

- [ ] **Step 3: Implement** in `src/physics/synchrotron.ts` (add `import { fluxMoment } from "./flux-history";`):

```ts
/** Sum of f(rho, z) weighted by volumeWeight over the rendered region (both lobes), in r_g^3, out to
 *  rho = qMax rho_f(z) (1.2: the wall's cut; wider for a flux-widened jet). */
function regionSum(a: number, jetLength: number, g280: number, f: (rho: number, z: number) => number, qMax = 1.2): number {
  const NZ = 160, NR = 48, zb = JET.zBase; let L = 0;
  for (let iz = 0; iz < NZ; iz++) {
    const dz = (jetLength - zb) / NZ, z = zb + dz * (iz + 0.5), dr = (qMax * funnelEdge(z)) / NR;
    for (let ir = 0; ir < NR; ir++) {
      const rho = dr * (ir + 0.5), v = f(rho, z); if (v === 0) continue;
      L += 2 * 2 * Math.PI * rho * dr * dz * volumeWeight(Math.hypot(rho, z), Math.atan2(rho, z), a, g280) * v;
    }
  }
  return L;
}
/** Injected power at infinity for q0 = 1 with the whole jet at flux ratio f (wall sqrt(f) wider, density x f;
 *  filaments are mean-one and drop out). f = 1 is the steady jet. */
export function injUnit(a: number, b0: number, rgCm: number, jetLength: number, g280: number, f = 1): number {
  const sw = Math.sqrt(f);
  return regionSum(a, jetLength, g280, (rho, z) => { const sh = f * wallProfile(rho / sw, z) * lengthFalloff(z, jetLength);
    if (sh <= 0) return 0; const B = jetField(rho, z, a, b0); return B * B * sh; }, 1.2 * sw) * rgCm ** 3;
}
/** Time average of the injected power over the flux distribution at slider s (spec corrections 2): injUnit is
 *  ~ f^2 (A + C f) (B_phi grows with rho), so the exact cubic through f = 0.5, 1, 1.5, 2 is averaged with the
 *  generator's moments <f> = 1, <f^2>, <f^3>. s = 0: the steady jet. */
export function meanInjection(a: number, b0: number, rgCm: number, jetLength: number, g280: number, fluxVar: number): number {
  if (fluxVar === 0) return injUnit(a, b0, rgCm, jetLength, g280, 1);
  const F = [0.5, 1, 1.5, 2], P = F.map((f) => injUnit(a, b0, rgCm, jetLength, g280, f));
  // Power-basis coefficients c0..c3 of the cubic through (F, P): solve the 4x4 Vandermonde system.
  const M = F.map((f, i) => [1, f, f * f, f * f * f, P[i]]);
  for (let c = 0; c < 4; c++) {
    for (let r = c + 1; r < 4; r++) { const k = M[r][c] / M[c][c]; for (let j = c; j < 5; j++) M[r][j] -= k * M[c][j]; }
  }
  const co = [0, 0, 0, 0];
  for (let r = 3; r >= 0; r--) { let s = M[r][4]; for (let j = r + 1; j < 4; j++) s -= M[r][j] * co[j]; co[r] = s / M[r][r]; }
  return co[0] + co[1] + co[2] * fluxMoment(2, fluxVar) + co[3] * fluxMoment(3, fluxVar);
}
```

and change `jetUniforms`:

```ts
/** The jet's uniforms: q0 = eta P_BZ / (time-averaged injected power at q0 = 1); 0 at spin 0 (no BZ power). */
export function jetUniforms(mSun: number, a: number, lambda: number, eta: number, jetLength: number, g280: number, fluxVar: number) {
  const E = jetEnergetics(mSun, a, lambda);
  return { jetB0: E.b0, jetQ0: E.pBZ > 0 ? (eta * E.pBZ) / meanInjection(a, E.b0, E.rgCm, jetLength, g280, fluxVar) : 0, rgCm: E.rgCm, pBZ: E.pBZ };
}
```

Fix every other `jetUniforms(` caller found by `grep -rn "jetUniforms(" src scripts tests`: in `src/main.ts`
pass `state.jetKnots` for now (renamed in Task 6); in any test/script pass `1` unless the test is about the
steady anchor (then `0`).

- [ ] **Step 4: Run tests**

Run: `npx vitest run tests/synchrotron.test.ts tests/presets.test.ts`
Expected: PASS. Note the time: the four-node `meanInjection` is ~4× `injUnit`; if the suite slows by more than
10 s, reduce nothing — report the number.

- [ ] **Step 5: Commit**

```bash
git add src/physics/synchrotron.ts src/main.ts tests/synchrotron.test.ts
git commit -m "Energy budget: q0 normalises the time-averaged injected power over the flux distribution (cubic in f)"
```

---

### Task 5: Shared WGSL twins and `?parity` coverage

**Files:**
- Modify: `src/render/emission-shared.wgsl` (remove knots/churn noise use; add flux, delay, filaments, new `jetShapeJ`)
- Create: `src/render/flux-parity.wgsl`
- Modify: `src/render/jet-parity.wgsl` (new `jetShapeJ` arguments)
- Modify: `src/test/parity.browser.ts` (flux cases; jet cases call the new CPU `jetShape`)
- Modify: `tests/jet.test.ts` (WGSL constants match the CPU twins)

**Interfaces:**
- Consumes: `hash4T`, `smoothE`, `funnelEdgeJ`, `wallJ`, `lengthFalloffJ`, `jetGammaAt` (emission-shared.wgsl);
  CPU twins from Tasks 1–3.
- Produces (WGSL, all in `emission-shared.wgsl`):
  - `const JET_ENV_Q = 1.51;` and flux/filament constants named `FLUX_T, FLUX_JIT, FLUX_TAUD, FLUX_SPREAD,
    FLUX_FLOOR, FLUX_DBAR, FLUX_D1, FLUX_SALT_T, FLUX_SALT_D, FIL_AMP, FIL_CELL_T, FIL_CELLS_PHI, FIL_CELLS_Q, FIL_SALT`.
  - `fn splitPeriodJ(epoch: f32, rel: f32, P: f32) -> vec2<f32>` — (whole periods k, remainder in [0, P)).
  - `fn fluxRatioJ(epoch: f32, rel: f32, s: f32) -> f32`.
  - `fn launchDelayJ(z: f32, g280: f32) -> f32`.
  - `fn fieldLineOmegaJ(a: f32) -> f32`, `fn comovingAzimuthJ(ph: f32, z: f32, a: f32, g280: f32) -> f32`.
  - `fn filamentsJ(q: f32, phiC: f32, epoch: f32, relL: f32) -> f32`.
  - `fn jetShapeJ(r: f32, th: f32, ph: f32, epoch: f32, rel: f32, jetLength: f32, fluxVar: f32, g280: f32, a: f32) -> f32`.

- [ ] **Step 1: Write the failing CPU-side constant test** (append to `tests/jet.test.ts`; add imports
  `readFileSync` from `node:fs`, `join` from `node:path`, and `GL6` from jet.ts)

```ts
const WGSL_E = readFileSync(join(__dirname, "../src/render/emission-shared.wgsl"), "utf8");
const wconst = (name: string) => { const m = WGSL_E.match(new RegExp(`const ${name}\\s*=\\s*([^;]+);`)); if (!m) throw new Error(`no ${name}`); return m[1]; };
describe("emission-shared.wgsl flux/filament constants match the CPU twins", () => {
  it("flux generator, filaments, envelope, Gauss nodes", () => {
    const pairs: [string, number][] = [["FLUX_T", FLUX.T], ["FLUX_JIT", FLUX.jitter], ["FLUX_TAUD", FLUX.tauD],
      ["FLUX_SPREAD", FLUX.spread], ["FLUX_FLOOR", FLUX.floor], ["FLUX_DBAR", FLUX.dbar], ["FLUX_D1", FLUX.d1],
      ["FLUX_SALT_T", FLUX.saltT], ["FLUX_SALT_D", FLUX.saltD], ["FIL_AMP", FILAMENT.amp], ["FIL_CELL_T", FILAMENT.cellT],
      ["FIL_CELLS_PHI", FILAMENT.cellsPhi], ["FIL_CELLS_Q", FILAMENT.cellsQ], ["FIL_SALT", FILAMENT.salt], ["JET_ENV_Q", JET_ENV_Q]];
    // Number() parses decimals and the hex salts (0x464c); WGSL's u suffix is stripped first.
    for (const [n, v] of pairs) expect(Math.abs(Number(wconst(n).trim().replace(/u$/, "")) - v)).toBeLessThan(1e-7 * Math.max(1, Math.abs(v)));
    const nodes = wconst("GL6_X").match(/-?\d+\.\d+/g)!.map(Number), wts = wconst("GL6_W").match(/-?\d+\.\d+/g)!.map(Number);
    GL6.forEach(([x, w], i) => { expect(nodes[i]).toBeCloseTo(x, 7); expect(wts[i]).toBeCloseTo(w, 7); });
  });
  it("the old knot and churn noise is gone from the shared jet code", () => {
    expect(WGSL_E).not.toMatch(/knotsJ|JET_KZ|JET_TURB|JET_SEED/);
  });
});
```

(Salts are written as hex in WGSL, e.g. `const FLUX_SALT_T = 0x464cu;`.)

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/jet.test.ts`
Expected: FAIL — `no FLUX_T`.

- [ ] **Step 3: Implement the WGSL** in `src/render/emission-shared.wgsl`. Delete `knotsJ` and the constants
  `JET_KZ`, `JET_TURB`, `JET_SEED` (keep `vnoiseE`/`ihashE` only if still used elsewhere: `grep -n vnoiseE src/render/*.wgsl`;
  if unused, delete them and update the section comment). Then add, after `jetGammaAt`:

```wgsl
// --- Horizon-flux history and launch time (spec 2026-10-03 jet flux knots; twins: flux-history.ts, jet.ts) ----
const FLUX_T = 1500.0; const FLUX_JIT = 0.33333334; const FLUX_TAUD = 500.0; const FLUX_SPREAD = 0.5;
const FLUX_FLOOR = 0.05; const FLUX_DBAR = 0.5058; const FLUX_D1 = 0.262371;
const FLUX_SALT_T = 0x464cu; const FLUX_SALT_D = 0x4458u;
const JET_ENV_Q = 1.51;   // envelope bound on rho / rho_f: 1.2 x the widest flux-driven width (jet.ts JET_ENV_Q)
// Absolute time epoch + rel as (whole periods k, remainder in [0, P)): the epoch is a multiple of 2048, so for
// an integer P both epoch and kE P are exact f32 integers below 2^24 and the remainder keeps rel's precision.
fn splitPeriodJ(epoch: f32, rel: f32, P: f32) -> vec2<f32> {
  let kE = floor(epoch / P);
  let x = (epoch - kE * P) + rel;
  let kl = floor(x / P);
  return vec2<f32>(kE + kl, x - kl * P);
}
fn fluxHashJ(k: i32, salt: u32) -> f32 { return f32(hash4T(k, 0x7a11u, 0, salt) & 0xffffffu) / 16777216.0 - 0.5; }
fn fluxDeficitJ(kf: f32, loc: f32) -> f32 {
  var k = i32(kf);
  var dt = loc - FLUX_T * (0.5 + FLUX_JIT * fluxHashJ(k, FLUX_SALT_T));
  if (dt < 0.0) { k = k - 1; dt = loc + FLUX_T - FLUX_T * (0.5 + FLUX_JIT * fluxHashJ(k, FLUX_SALT_T)); }
  let gap = FLUX_T * (1.0 + FLUX_JIT * (fluxHashJ(k + 1, FLUX_SALT_T) - fluxHashJ(k, FLUX_SALT_T)));
  let depth = FLUX_DBAR * (1.0 + FLUX_SPREAD * fluxHashJ(k, FLUX_SALT_D));
  let D = -FLUX_TAUD * log(1.0 - depth);
  if (dt < D) { return 1.0 - exp(-dt / FLUX_TAUD); }
  return depth * (gap - dt) / (gap - D);
}
// f = phi / mean(phi) at absolute time epoch + rel for slider s (s = 0: exactly 1).
fn fluxRatioJ(epoch: f32, rel: f32, s: f32) -> f32 {
  if (s == 0.0) { return 1.0; }
  let kl = splitPeriodJ(epoch, rel, FLUX_T);
  return max(FLUX_FLOOR, 1.0 - s * fluxDeficitJ(kl.x, kl.y)) / (1.0 - s * FLUX_D1);
}
const GL6_X = array<f32, 6>(-0.9324695142031521, -0.6612093864662645, -0.2386191860831969, 0.2386191860831969, 0.6612093864662645, 0.9324695142031521);
const GL6_W = array<f32, 6>(0.1713244923791704, 0.3607615730481386, 0.4679139345726910, 0.4679139345726910, 0.3607615730481386, 0.1713244923791704);
// Coordinate time for plasma to climb from z_base to |z| (6-point Gauss in v = z^(1-p); twin: launchDelay).
fn launchDelayJ(z: f32, g280: f32) -> f32 {
  let az = abs(z);
  if (az <= JET_ZBASE) { return 0.0; }
  let A = sqrt(max(1e-12, g280 * g280 - 1.0)) / pow(GAMMA_REF_Z, GAMMA_SLOPE);
  let e = 2.0 * GAMMA_SLOPE / (1.0 - GAMMA_SLOPE);
  let v0 = pow(JET_ZBASE, 1.0 - GAMMA_SLOPE); let v1 = pow(az, 1.0 - GAMMA_SLOPE);
  let h = 0.5 * (v1 - v0); let m = 0.5 * (v1 + v0);
  // Function-scope copies: WGSL guarantees dynamic indexing for var arrays.
  var xs = GL6_X; var ws = GL6_W;
  var s = 0.0;
  for (var i = 0; i < 6; i++) { s += ws[i] * sqrt(1.0 + A * A * pow(m + h * xs[i], e)); }
  return s * h / (A * (1.0 - GAMMA_SLOPE));
}
fn fieldLineOmegaJ(a: f32) -> f32 { return a / (4.0 * (1.0 + sqrt(max(0.0, 1.0 - a * a)))); }
fn comovingAzimuthJ(ph: f32, z: f32, a: f32, g280: f32) -> f32 {
  let az = abs(z);
  if (az <= JET_ZBASE) { return ph; }
  return ph - fieldLineOmegaJ(a) * (launchDelayJ(z, g280) - (az - JET_ZBASE));
}
// --- Filaments frozen into the moving plasma (spec 2.5; twin: filaments in jet.ts) ---------------------------
const FIL_AMP = 0.35; const FIL_CELL_T = 25.0; const FIL_CELLS_PHI = 8u; const FIL_CELLS_Q = 2.5; const FIL_SALT = 0x46494cu;
fn node3J(ix: i32, iy: u32, iw: i32) -> f32 { return f32(hash4T(ix, iy, iw, FIL_SALT) & 0xffffffu) / 16777216.0; }
// Value noise on (ix + fx, y periodic in FIL_CELLS_PHI, w); x is passed split (time cell index + fraction).
fn vnoise3J(ix: i32, fx0: f32, y: f32, w: f32) -> f32 {
  let iyf = floor(y); let iw = i32(floor(w));
  let fx = smoothE(fx0); let fy = smoothE(y - iyf); let fw = smoothE(w - floor(w));
  let y0 = u32(iyf) % FIL_CELLS_PHI; let y1 = (y0 + 1u) % FIL_CELLS_PHI;
  var v = 0.0;
  for (var c = 0u; c < 8u; c++) {
    let dx = c & 1u; let dy = (c >> 1u) & 1u; let dw = (c >> 2u) & 1u;
    let wt = select(1.0 - fx, fx, dx == 1u) * select(1.0 - fy, fy, dy == 1u) * select(1.0 - fw, fw, dw == 1u);
    v += wt * node3J(ix + i32(dx), select(y0, y1, dy == 1u), iw + i32(dw));
  }
  return v;
}
fn filamentsJ(q: f32, phiC: f32, epoch: f32, relL: f32) -> f32 {
  let kt = splitPeriodJ(epoch, relL, FIL_CELL_T);
  let turns = phiC / TWO_PI_E;
  let y = (turns - floor(turns)) * f32(FIL_CELLS_PHI);
  return 1.0 + FIL_AMP * (vnoise3J(i32(kt.x), kt.y / FIL_CELL_T, y, q * FIL_CELLS_Q) - 0.5) * 2.0;
}
```

(`TWO_PI_E` is already defined at the top of `emission-shared.wgsl`. `y` is in [0, 8) so `u32(iyf)` is safe.)

Replace `jetShapeJ`:

```wgsl
// Density modulation (spec 2026-10-03 jet flux knots 2.4; twin: jetShape in jet.ts): the plasma at height z left
// the base at (epoch + rel) - tau(z) with flux ratio f; the funnel is sqrt(f) wider and f denser, times the
// co-moving filaments. rel carries the per-pixel emission time (light delay); 0 outside the emitting jet.
fn jetShapeJ(r: f32, th: f32, ph: f32, epoch: f32, rel: f32, jetLength: f32, fluxVar: f32, g280: f32, a: f32) -> f32 {
  let z = r * cos(th); let az = abs(z);
  if (az < JET_ZBASE || az > jetLength) { return 0.0; }
  let relL = rel - launchDelayJ(z, g280);
  let f = fluxRatioJ(epoch, relL, fluxVar); let sw = sqrt(f);
  let rho = r * sin(th);
  let w = wallJ(rho / sw, z);
  if (w <= 0.0) { return 0.0; }
  let q = rho / (sw * funnelEdgeJ(z));
  return max(0.0, f * w * lengthFalloffJ(z, jetLength) * filamentsJ(q, comovingAzimuthJ(ph, z, a, g280), epoch, relL));
}
```

`launchDelayJ` uses `GAMMA_REF_Z` / `GAMMA_SLOPE`, which are declared just above `jetGammaAt` — place the new block
after `jetGammaAt` so the reading order matches (WGSL itself does not require declaration order).

- [ ] **Step 4: Write the GPU parity shader** `src/render/flux-parity.wgsl`:

```wgsl
// Parity harness for the flux history, launch delay, co-moving azimuth and filaments: the functions come from
// emission-shared.wgsl (sole copies, also prepended to raytrace.wgsl), so ?parity checks the renderer's own bytes
// against flux-history.ts and jet.ts. Per case: a = (epoch, rel, z, g280), b = (q, ph, spin, s).
// Output: (fluxRatioJ(epoch, rel, s), launchDelayJ(z, g280), comovingAzimuthJ(ph, z, spin, g280), filamentsJ(q, ph, epoch, rel)).
struct FIn { a: vec4<f32>, b: vec4<f32> };
@group(0) @binding(0) var<storage, read> finp: array<FIn>;
@group(0) @binding(1) var<storage, read_write> foutp: array<vec4<f32>>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = finp[gid.x];
  foutp[gid.x] = vec4<f32>(fluxRatioJ(c.a.x, c.a.y, c.b.w), launchDelayJ(c.a.z, c.a.w),
                           comovingAzimuthJ(c.b.y, c.a.z, c.b.z, c.a.w), filamentsJ(c.b.x, c.b.y, c.a.x, c.a.y));
}
```

In `src/render/jet-parity.wgsl` replace `const P_JETLEN = 60.0; const P_KNOTS = 0.7;` with
`const P_JETLEN = 60.0; const P_FLUX = 1.0;`, update the header comment (`c = (a, g280, emission time REMAINDER,
slab path ds in cm), k = (b0, q0, rgCm, clock epoch)`), and the shape line to:

```wgsl
  let shape = jetShapeJ(r, th, c.x.w, c.k.w, c.c.z, P_JETLEN, P_FLUX, c.c.y, a);
```

- [ ] **Step 5: Wire the parity harness** in `src/test/parity.browser.ts`:
  1. imports: `import fluxParityWGSL from "../render/flux-parity.wgsl?raw";`,
     `import { fluxRatio } from "../physics/flux-history";`,
     `import { jetShape, launchDelay, comovingAzimuth, filaments } from "../physics/jet";` (replace the old jetShape import).
  2. Jet cases: `const J_LEN = 60, J_FLUX = 1, T_EM = 1.7, J_EPOCH = 2048 * 3;` (replace `J_KNOTS`), and every
     CPU `jetShape(r, th, T_EM, J_LEN, J_KNOTS, g280)` becomes
     `jetShape(r, th, s[3], J_EPOCH + T_EM, J_LEN, J_FLUX, g280, a)` — in `planeCoeffs` (has `s`, `a`), in the
     ray walk loop (`s[1], s[2], s[3]`, spin from the ray tuple) and in the comparison loop (`c.a`). In the input
     packing, the last slot `0` becomes `J_EPOCH`:
     `jarr.set([...c.s.slice(0, 8), c.a, c.g280, T_EM, c.ds, J.b0, J.q0, J.rg, J_EPOCH], i * J_IN);`
  3. Flux cases, after the jet block (before the final `return`), modelled on the turbulence block:

```ts
  // --- flux history / launch delay / co-moving azimuth / filaments (CPU flux-history.ts + jet.ts vs the SHIPPED
  // emission-shared.wgsl). Epochs up to 2048 x 8000 (~16 M M: hours of play, Review Focus 1); rel spans eruption
  // drops and refills and negative launch remainders; both lobes; spins 0, 0.9, 0.998.
  const fcases: number[][] = [];
  for (const epoch of [0, 2048 * 7, 2048 * 5000, 2048 * 8000])
    for (const rel of [-900.5, 13.25, 377.75, 1024.5, 1999.875])
      for (const [z, g, q, ph, a, s] of [[5, 2, 0.8, 0.3, 0.9, 1], [-40, 5, 1.1, -7.2, 0.998, 1.4], [59, 1.5, 0.2, 12.9, 0, 0.5]])
        fcases.push([epoch, rel, z, g, q, ph, a, s]);
  const farr = new Float32Array(fcases.length * 8);
  fcases.forEach((c, i) => farr.set([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]], i * 8));
  const fin = device.createBuffer({ size: farr.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(fin, 0, farr);
  const fout = device.createBuffer({ size: fcases.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const fread = device.createBuffer({ size: fcases.length * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const fmod = device.createShaderModule({ code: integratorSharedWGSL + emissionSharedWGSL + fluxParityWGSL });
  const fpipe = device.createComputePipeline({ layout: "auto", compute: { module: fmod, entryPoint: "main" } });
  const fbind = device.createBindGroup({ layout: fpipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: fin } }, { binding: 1, resource: { buffer: fout } }] });
  const fenc = device.createCommandEncoder();
  const fcp = fenc.beginComputePass(); fcp.setPipeline(fpipe); fcp.setBindGroup(0, fbind); fcp.dispatchWorkgroups(fcases.length); fcp.end();
  fenc.copyBufferToBuffer(fout, 0, fread, 0, fcases.length * 16);
  device.queue.submit([fenc.finish()]);
  await fread.mapAsync(GPUMapMode.READ);
  const fgpu = new Float32Array(fread.getMappedRange().slice(0));
  let fluxErr = 0, fluxWorst = "";
  fcases.forEach((_, i) => {
    const [epoch, rel, z, g, q, ph, a, s] = Array.from(farr.subarray(i * 8, i * 8 + 8)); // f32-rounded, as the GPU got
    const t = epoch + rel, tau = launchDelay(z, g);
    // Errors: absolute on f and filaments (bounded ~0.2-1.6 and 0.65-1.35), relative on tau, absolute on the azimuth.
    const want = [fluxRatio(t, s), tau, comovingAzimuth(ph, z, a, g), filaments(q, ph, t)];
    const tol = [2e-3, 1e-4 * Math.max(1, tau), 2e-3, 2e-3];
    for (let k = 0; k < 4; k++) { const e = Math.abs(fgpu[i * 4 + k] - want[k]) / tol[k];
      if (e > fluxErr) { fluxErr = e; fluxWorst = `case ${i} out ${k}: gpu ${fgpu[i * 4 + k]} cpu ${want[k]}`; } }
  });
  console.log("flux parity worst (|err| / tol)", fluxErr.toExponential(2), fluxWorst);
```

  Note the filaments case passes `ph` (not a co-moving azimuth) and `rel` directly: it tests `filamentsJ` itself.
  4. Report it: add `fluxErr: number; fluxWorst: string` to the return type, include
     `fcases.length` in `rows`, and return `fluxErr, fluxWorst`. Then make the route's PASS condition include
     `fluxErr <= 1`: find where `runParity()` is judged (`grep -n "PARITY PASS" src -r`) and add
     `&& res.fluxErr <= 1` plus `flux ${res.fluxErr.toFixed(3)}` to the printed line, mirroring `turbErr`.

- [ ] **Step 6: Run the CPU tests and the GPU parity gate**

Run: `npx vitest run tests/jet.test.ts tests/shader-twins.test.ts`
Expected: PASS (constants match; the shader-twins guard sees no duplicated WGSL functions).

Run (dev server on :5173 in another terminal: `npm run dev`): `npm run verify:gpu`
Expected for this task: `?parity` PASS with the flux line ≤ 1 and the jet cases passing; the renderer routes
(`?golden`, `?cachecheck`, app) FAIL to compile because `raytrace.wgsl` still calls the old `jetShapeJ` — that is
Task 6. To check parity alone, run with `ONLY_APP` unset and stop after the parity line, or temporarily read the
`?parity` output in the browser at `http://localhost:5173/?parity`.

- [ ] **Step 7: Mutation check** — temporarily change `FLUX_TAUD = 500.0` to `400.0` in `emission-shared.wgsl`,
  reload `?parity`: the flux line must exceed 1 (FAIL). Revert. Then temporarily change `FIL_CELL_T = 25.0` to
  `26.0`: FAIL again. Revert. Record both numbers for the report.

- [ ] **Step 8: Commit**

```bash
git add src/render/emission-shared.wgsl src/render/flux-parity.wgsl src/render/jet-parity.wgsl src/test/parity.browser.ts tests/jet.test.ts
git commit -m "Shared WGSL: flux history, launch delay, co-moving filaments, flux-driven jetShapeJ; ?parity flux cases (epochs to 2048 x 8000)"
```

---

### Task 6: Renderer wiring, uniform rename, panel slider, golden re-record

**Files:**
- Modify: `src/render/raytrace.wgsl` (jet sample φ, epoch-split time, envelope), `src/render/present.wgsl` (struct field name)
- Modify: `src/render/uniforms.ts` (`jetKnots` → `fluxVar`)
- Modify: `src/main.ts`, `index.html` (slider), `src/test/scenes.ts`, `src/test/shadow.browser.ts`,
  `scripts/bench.mjs`, `scripts/shot-delay.mjs`, `tests/uniforms.test.ts`, `tests/cache-plan.test.ts`
- Modify: `src/test/golden.json` (re-recorded)

**Interfaces:**
- Consumes: `jetShapeJ(r, th, ph, epoch, rel, jetLength, fluxVar, g280, a)`, `JET_ENV_Q` (Task 5);
  `jetUniforms(..., fluxVar)` (Task 4).
- Produces: uniform field `fluxVar` (float index 22, byte offset 88) in `UniformValues` and both WGSL structs.

- [ ] **Step 1: Update the uniform test first** — in `tests/uniforms.test.ts` rename `jetKnots` to `fluxVar` in
  both objects and the offset assertion:

```ts
    expect(dv.getFloat32(88, true)).toBeCloseTo(0.7);   // fluxVar (index 22)
```

and in `tests/cache-plan.test.ts` replace `jetKnots: 0.1` with `fluxVar: 0.1`.

Run: `npx vitest run tests/uniforms.test.ts tests/cache-plan.test.ts`
Expected: FAIL (TypeScript property `fluxVar` does not exist on `UniformValues`).

- [ ] **Step 2: Rename the uniform** — `src/render/uniforms.ts`: in the layout comment, the interface
  (`jetStrength: number; jetGamma: number; jetLength: number; fluxVar: number;`) and the writer
  (`f[22] = u.fluxVar;`). In `src/render/raytrace.wgsl` and `src/render/present.wgsl` the struct line becomes
  `jetStrength: f32, jetGamma: f32, jetLength: f32, fluxVar: f32,`. Update every TS caller found by
  `grep -rn "jetKnots" src scripts tests index.html`: `src/test/scenes.ts` (`fluxVar: 1`),
  `src/test/shadow.browser.ts` (`fluxVar: 1`), `scripts/bench.mjs` (`fluxVar: 1`), `scripts/shot-delay.mjs`
  (`fluxVar: 0`).

Run: `npx vitest run tests/uniforms.test.ts tests/cache-plan.test.ts` — Expected: PASS.

- [ ] **Step 3: Renderer jet sampling** in `src/render/raytrace.wgsl`:

```wgsl
// Sample k of n: (r, theta, phi). k = 0 is the step's start state itself (its phi may be unwrapped; the
// filaments use it modulo 2 pi); interior samples take phi from the chord's Cartesian point.
fn jetSample(s: State, p0: vec3<f32>, dvec: vec3<f32>, k: u32, n: u32) -> vec3<f32> {
  if (k == 0u) { return vec3<f32>(s.x.y, s.x.z, s.x.w); }
  let p = p0 + dvec * (f32(k) / f32(n));
  let r = length(p);
  return vec3<f32>(r, acos(clamp(p.z / r, -1.0, 1.0)), atan2(p.y, p.x));
}
```

In `jetChordMisses`: `let fe = JET_ENV_Q * funnelEdgeJ(U.jetLength);` and update its comment
(`rho <= JET_ENV_Q funnelEdge(jetLength)`). In `inJetEnvelope`: `return r * sin(th) / funnelEdgeJ(z) <= JET_ENV_Q;`.
In `jetStep` the shape line becomes:

```wgsl
    let shape = jetShapeJ(q.x, q.y, q.z, U.timeEpoch, emitRel(-tS - U.rObs), U.jetLength, U.fluxVar, U.jetGamma, U.a);
```

`jetTouches` keeps `inJetEnvelope(q.x, q.y)` (q is now vec3; `.x .y` unchanged). If `emitTime` has no other
caller (`grep -n "emitTime(" src/render/raytrace.wgsl`), delete it and keep its comment's content on `emitRel`.

- [ ] **Step 4: Panel slider and state** — `index.html`, replace the "Jet knots" control with:

```html
    <div class="ctrl" title="Swing of the horizon magnetic flux that drives the jet: 1× is the GRMHD value at spin 0.9 (σ/μ = 21 %, Narayan et al. 2022). Each flux eruption sends a dimming, narrowing front up the jet. The twisting filaments are illustrative.">
      <div class="row"><label>Flux variability</label><span class="val"><b id="fvv">1.00</b>×</span></div>
      <input id="fv" type="range" min="0" max="1.4" step="0.05" value="1">
    </div>
```

`src/main.ts`: in `state` replace `jetKnots: 0.7` with `fluxVar: 1.0`; replace the `jk` element lookups and
listener with `fv` / `fvv` and

```ts
  fv.addEventListener("input", () => { state.fluxVar = +fv.value; fvv.textContent = state.fluxVar.toFixed(2); physicsChanged(); });
```

(`physicsChanged`, not `reset`: q0 depends on the slider.) In `refreshJet` add `|${state.fluxVar}` to the key
and pass `state.fluxVar` as the 7th argument of both `jetUniforms` calls; the frame uniforms pass
`fluxVar: state.fluxVar`. If presets apply jet fields (`grep -n "jet" src/main.ts | grep -i preset`), set
`state.fluxVar = 1` and the slider/label there too.

- [ ] **Step 5: Build and full CPU suite**

Run: `npm run build` — Expected: no type errors.
Run: `npm test` — Expected: all pass (previous 189 + new; SWEEP-gated files skipped).

- [ ] **Step 6: GPU gates and golden re-record** (dev server running)

Run: `RECORD_GOLDEN=1 npm run verify:gpu` (PowerShell: `$env:RECORD_GOLDEN="1"; npm run verify:gpu; Remove-Item Env:RECORD_GOLDEN`)
Expected: `?parity` PASS (flux line ≤ 1), `?shadow` PASS (unchanged 4.01-class numbers from README), golden
recorded then PASS, `?cachecheck` PASS (cached = live, jet scenes included), app checks warning-free.

Compare the new `src/test/golden.json` with the previous one (`git diff src/test/golden.json`). The jet-on scenes
change by design. If the **jet-off** hash changed: build a jet-free kernel on both `main` and this branch (make
`jetStep` return `accIn` on its first line in each), record golden on both, and confirm the jet-off hashes are
equal between them — the README's documented compiler-rescheduling ruling; revert the experiment. Report the
outcome either way.

- [ ] **Step 7: Commit**

```bash
git add src/render/raytrace.wgsl src/render/present.wgsl src/render/uniforms.ts src/main.ts index.html src/test/scenes.ts src/test/shadow.browser.ts scripts/bench.mjs scripts/shot-delay.mjs tests/uniforms.test.ts tests/cache-plan.test.ts src/test/golden.json
git commit -m "Renderer: jet samples carry phi and epoch-split time, envelope q <= 1.51, Flux variability slider (uniform fluxVar), golden re-recorded"
```

---

### Task 7: Visual check, docs, final gates

**Files:**
- Create: `scripts/shot-flux.mjs`
- Modify: `README.md`, `docs/ROADMAP.md`, `docs/specs/2026-10-03-jet-flux-knots-design.md` (status line)

- [ ] **Step 1: Frame-capture script**

```js
// scripts/shot-flux.mjs -- frames across one flux eruption (dev server on :5173): the default view and M87*,
// Motion at its maximum (100 M/s, one eruption cycle ~15 s), one frame per second for 20 s.
import { chromium } from "playwright-core";
const BASE = process.env.BASE || "http://localhost:5173";
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 760 } });
await page.goto(BASE + "/", { waitUntil: "load" });
await page.waitForFunction(() => /^\d+$/.test(document.getElementById("spp").textContent), null, { timeout: 30000 });
for (const id of ["default", "m87"]) {
  await page.selectOption("#preset", id);
  await page.evaluate(() => { const ts = document.getElementById("ts"); ts.value = "5"; ts.dispatchEvent(new Event("input")); });
  await page.waitForTimeout(3000);
  for (let k = 0; k < 20; k++) { await page.screenshot({ path: `flux-${id}-${String(k).padStart(2, "0")}.png` }); await page.waitForTimeout(1000); }
  console.log(`• flux-${id}-00..19.png`);
}
await browser.close();
```

Run: `node scripts/shot-flux.mjs` (dev server running). Look at the frames: in the default view a dimmer,
narrower stretch should climb the jet over ~2 s (166 M at 100 M/s) after an eruption and the jet should
re-brighten and re-widen over the following ~10 s; filaments should stream outward and twist. If no eruption
falls inside the 20 s window, rerun once. Keep 3–4 representative frames for the report (do not commit PNGs).

- [ ] **Step 2: Docs** — `README.md`: add a "Jet knots from horizon-flux variability (feat/jet-flux-knots,
  2026-10-03)" paragraph after the MRI-turbulence paragraph covering: the sources table numbers (recurrence
  1000–2500 M, e-fold 500 M, σ/μ 0.209, P ∝ φ², thin–thick pattern), the generator and its constants, launch
  time and τ(60 M) = 166 M, width √f / density f, the mean-power cubic, the filaments (illustrative strength,
  physical motion, Ω = Ω_F(1 − β)), the slider, the widened envelope (1.51), and the new golden hashes plus
  test counts. In the old Tier 2B paragraph's mention of "Jet knots" slider, add "(replaced 2026-10-03 by Flux
  variability)". `docs/ROADMAP.md`: mark queue item #3 done with one line and list follow-ups: spin-dependent
  flux statistics; shell collisions beyond the view; coupling eruptions to the disk (Ṁ drops, hot spots).
  Spec status line: "implemented on feat/jet-flux-knots".

- [ ] **Step 3: Final gates**

Run: `npm test` — Expected: all pass. Run: `npm run build` — Expected: clean.
Run: `npm run verify:gpu` — Expected: every route PASS without re-record. Run `node scripts/probe-axis.mjs` and
`node scripts/probe-scale.mjs` — Expected: PASS as before.

- [ ] **Step 4: Commit**

```bash
git add scripts/shot-flux.mjs README.md docs/ROADMAP.md docs/specs/2026-10-03-jet-flux-knots-design.md
git commit -m "Docs: jet knots from horizon-flux variability; frame-capture script for the eruption front"
```

- [ ] **Step 5: Integrate** (standing user rule for blackhole-sim): after the final review passes, fast-forward
  `main` to the branch, push `main` to origin, delete the branch. Do NOT deploy unless asked.
