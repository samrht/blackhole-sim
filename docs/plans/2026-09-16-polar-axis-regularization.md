# Polar-Axis Regularization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `sin²θ` physics cap (`POLE_S2 = 1e-3`, a 1.81° black cone at each pole) with a pure NaN guard (`1e-12`), and keep the integrator stable on the restored near-axis forces with constraint-monitored adaptive stepping plus exact analytic axis crossing — with every new piece of integrator logic gated CPU↔GPU on the shipped shader bytes.

**Architecture:** A new CPU module `src/physics/trace.ts` becomes the f64 twin of the render loop's integration logic (step controller, constraint-monitored step, axis reflection, ray tracing loop). The matching WGSL moves out of `raytrace.wgsl` into `src/render/integrator-shared.wgsl`, the **sole** WGSL copy, prepended as a string to `raytrace.wgsl` by `gpu.ts` and to the parity entry points by `parity.browser.ts` (the same pattern as `shadow-shared.wgsl` / `camera-shared.wgsl`). A new `integrator-parity.wgsl` entry point runs one `stepGeodesic` on the GPU and compares state, retry count and the tolerance constants against the CPU. The `H_TOL` / `MAX_RETRY` constants are chosen from a measured sweep, not guessed.

**Tech Stack:** TypeScript, WGSL, WebGPU, Vite (`?raw` imports), Vitest, playwright-core (headless Chrome for GPU routes).

**Spec:** `docs/specs/2026-07-21-polar-axis-regularization-design.md`

## Deviations from the spec (read before starting — each is deliberate)

These were established by a scratch f64 experiment against the CPU math before this plan was written. Keep them; do not "fix" the code back toward the spec's literal wording.

1. **Corrected failure mechanism (spec §1 says rays are *captured*).** With the barrier removed inside the cone, θ runs straight *through* 0 to large negative values (θ_min ≈ −4.7 rad measured) and nothing reflects it. The disk test `f = θ − π/2` never fires at θ = −π/2, so the ray tunnels through the disk plane undetected and terminates as **escaped**, which with the sky off renders black. Same black wedge, different route. Consequence: the axis reflection (§3.3) is load-bearing for **every** ray that crosses, not just the measure-zero ξ = 0 set, and the ξ = 0 ray still escapes under the new floor unless reflection is present.
2. **Monitor the per-step change ΔH, not |H|.** H drifts secularly (RK4 is non-symplectic, forces are finite differences). Once |H| has drifted past a tolerance, halving the *current* step can never bring it back, so an `|H| > tol` test would retry to the cap on every subsequent step. The test is `|H(s_new) − H(s_old)| ≤ tol`.
3. **The tolerance is relative, not absolute.** Near the capture margin `g^tt = −A/(ΣΔ)` reaches ~2·10², so H is a cancellation of O(100) terms and f32 noise on it is ~1e-5 absolute. An absolute tolerance small enough to matter near the axis would make the GPU retry spuriously near the horizon. The monitor compares `|ΔH|` against `H_TOL · Σ|terms|` where the terms are the five products in `g^{μν} p_μ p_ν`. Because the camera fixes `p_t = 1` and `|g^tt| ≥ 1` outside the horizon (A − ΣΔ = 2Mr(r² + a²) ≥ 0), the scale is always ≥ 1, so this is never *looser* than an absolute tolerance of `H_TOL`.
4. **On retry exhaustion the ray `break`s out of the loop immediately** with `resolved = false`, so the existing conserved-quantity classifier decides captured/escaped. The spec says "proceeds with the smallest permitted step and is marked resolved = false", but in the current loop any later disk/horizon/escape termination overwrites `resolved = true`, so "proceeding" would let an untrusted trajectory be accepted — the opposite of the spec's intent. Verified against `raytrace.wgsl:270-354` (the `resolved` flag is only consulted after the loop).
5. **`?parity` will NOT move from the `POLE_S2` change** — the parity metric cases (`parity.browser.ts:19-22`) are all at θ ≥ 0.9 where `sin²θ ≫ 1e-3`. It **will** move when the new integrator parity block lands (Task 4), because a full RK4 step in f32 has larger relative error than a single metric component. That is the legitimate re-baseline the spec anticipates; record the new number, do not tune anything to keep the old one.

## Global Constraints

- **No `Co-Authored-By` trailer in any commit.** Verify after each commit with `git log -1 --format=%B | grep -ci "co-authored"` — must print `0`. (This is a project rule that overrides the session's default attribution guidance.)
- **Do NOT push to origin. Do NOT merge. Do NOT switch branches.** Work stays on `fix/polar-axis`, created from `main` at `0bc295a`.
- Hard gate values **before** this branch (baseline, recorded 2026-07-21): `?parity` maxRelErr **exactly 9.690e-7 over 45 cases**; `?shadow` PASS at **4.49 M**, calibration **0.864**; `npm test` **62 passing**; `npm run build` clean. Task 3 must reproduce these exactly. Tasks 4–5 re-baseline them explicitly (see each task) and the new values become the gate for the rest of the branch.
- **`H_TOL` and `MAX_RETRY` are chosen by the sweep in Task 2, never by hand.** Task 1 ships provisional values clearly marked as such. A value changed to make a test pass is a plan failure.
- Never loosen `?parity`'s `1e-3` pass threshold (`src/main.ts:13`). If a parity block disagrees, understand why.
- GPU routes need the dev server running on **:5173** first (`npm run dev` in a separate terminal; on Windows use a second PowerShell window). `npm run verify:gpu` works without a `BASE=` override.
- **`verify-gpu.mjs` is blind to WGSL compile errors.** Chrome reports them as console *warnings* and the output buffer stays zeroed, so a broken shader shows up as a strange downstream number (e.g. parity maxRelErr = 1.0, shadow radius 0), not as an error. If a gate prints a nonsense number after a shader edit, attach `page.on("console", m => console.log(m.type(), m.text()))` in a scratch `.mjs` under `scripts/` (playwright-core resolves only from inside the repo; `package.json` has `"type": "module"` so use `import`, not `require`) and read the WGSL diagnostics. Delete the scratch file afterwards.
- Prepend (never append) the shared fragments so the concatenation order stays the one `gpu.ts` documents.
- Existing gates: `npm test`, `npm run build`, `npm run verify:gpu`. Run all three at the end of every task.
- Momentum sign conventions come **only** from `src/physics/camera.ts` / `src/render/camera-shared.wgsl` (`p_t = 1`, `p_φ = −ξ`, `p_θ = −β`, `p_r < 0`, integrate with `dl > 0`). Do not re-derive them.

## File Structure

| File | Responsibility |
|---|---|
| `src/physics/kerr.ts` | **Modify.** `POLE_S2` 1e-3 → 1e-12, comment rewritten to say it is a NaN guard. |
| `src/physics/trace.ts` | **Create.** CPU twin of the integration loop: `stepSize`, `hquadScaled`, `reflectAxis`, `stepGeodesic`, `traceRay`, constants `H_TOL`, `MAX_RETRY`. No emission, no jet, no colour — geometry only. |
| `tests/trace.test.ts` | **Create.** The spec's three tests (§4.1) plus exhaustion routing and the south-pole reflection. |
| `tests/sweep-htol.test.ts` | **Create.** Env-gated (`SWEEP=1`) measurement that prints the tolerance/retry table. Skipped in normal `npm test`. |
| `src/render/integrator-shared.wgsl` | **Create.** Sole WGSL copy of: `PI`, `delta_`, `sigma_`, `bigA_`, `POLE_S2`, `gUp`, `gLow`, `omegaKep`, `hquad`, `State`, `rhs`, `addS`, `rk4` (moved verbatim from `raytrace.wgsl`), plus new `stepSize`, `H_TOL`, `MAX_RETRY`, `hquadScaled`, `reflectAxis`, `StepOut`, `stepGeodesic`. |
| `src/render/raytrace.wgsl` | **Modify.** Delete lines 16–59 (moved to the fragment); main loop calls `stepGeodesic` / `stepSize`. |
| `src/render/parity.wgsl` | **Modify.** Delete its private copies of `delta_`…`omegaKep` (lines 3–14); receives the fragment by prepending. Closes the metric-twin gap as a side effect. |
| `src/render/integrator-parity.wgsl` | **Create.** Entry point only: one `stepGeodesic` per case, writes state + retries + ok + constants. No math. |
| `src/render/gpu.ts` | **Modify.** Import and prepend `integrator-shared.wgsl`. |
| `src/test/parity.browser.ts` | **Modify.** Prepend the fragment to `parity.wgsl`; add the integrator parity block. |
| `scripts/probe-axis.mjs` | **Create.** Drives the DOM sliders, screenshots a = 0 / sky 0 / jet 0 at i = 72° and 8°, measures the axis column, asserts the wedge is gone. |
| `README.md` | **Modify.** Replace the false "regularized" paragraph; record the new gate numbers and the chosen constants. |
| `docs/specs/2026-07-21-polar-axis-regularization-design.md` | **Modify.** Status line → implemented; add a short "as built" note pointing at the deviations above. |

---

### Task 0: Branch and baseline

**Files:** none modified.

- [ ] **Step 1: Create the branch**

```bash
git checkout -b fix/polar-axis
git log -1 --oneline   # expect 0bc295a Spec: polar-axis regularization
```

- [ ] **Step 2: Record the baseline gates**

Terminal 1: `npm run dev` (leave running for the whole branch). Terminal 2:

```bash
npm test 2>&1 | tail -5          # expect "Tests  62 passed (62)"
npm run build                     # expect clean
npm run verify:gpu                # expect PARITY PASS maxRelErr=9.690e-7 over 45 cases; SHADOW PASS apparent radius ≈ 4.49 M ... calibration = 0.864
```

If any number differs from the Global Constraints baseline, STOP and report — the branch base is not what this plan assumes.

---

### Task 1: CPU twin of the integration loop, and the spec's regression tests

The deliverable is a CPU module whose behaviour the later WGSL will mirror byte-for-byte in structure, and tests that **fail on the shipped physics** (old floor) and pass on the new. Order matters: the tests are written first, run against the old `POLE_S2` to prove they discriminate, then `POLE_S2` is changed.

**Files:**
- Create: `src/physics/trace.ts`
- Create: `tests/trace.test.ts`
- Modify: `src/physics/kerr.ts:3-9`

**Interfaces:**
- Consumes: `rk4(s, a, dl)` from `src/physics/geodesic.ts` (state layout `[t, r, θ, φ, p_t, p_r, p_θ, p_φ]` as `Float64Array`), `metricUpper(r, θ, a)` from `kerr.ts`, `screenToState(alpha, beta, a, incl, rObs)` / `screenToXiEta` from `camera.ts`.
- Produces (Tasks 2, 4 and the parity harness rely on these exact names/signatures):
  - `export const H_TOL: number`, `export const MAX_RETRY: number`
  - `export function stepSize(r: number, rh: number, rOut: number): number`
  - `export function hquadScaled(r, th, a, pt, pr, pth, pphi): [number, number]` → `[H_quad, Σ|terms|]`
  - `export function reflectAxis(s: Float64Array): Float64Array`
  - `export interface StepOut { s: Float64Array; ok: boolean; retries: number; dl: number }`
  - `export function stepGeodesic(s, a, dl0, hTol = H_TOL, maxRetry = MAX_RETRY): StepOut`
  - `export type Fate = "disk" | "captured" | "escaped" | "budget" | "untrusted"`
  - `export interface TraceOpts { rIn: number; rOut: number; rObs: number; maxSteps?: number; hTol?: number; maxRetry?: number }`
  - `export interface TraceResult { fate: Fate; s: Float64Array; steps: number; retries: number; thMin: number; rHit?: number; phiHit?: number }`
  - `export function traceRay(s0: Float64Array, a: number, o: TraceOpts): TraceResult`

- [ ] **Step 1: Write `src/physics/trace.ts`**

```ts
import { metricUpper } from "./kerr";
import { rk4 } from "./geodesic";

/**
 * CPU twin of the render loop's INTEGRATION logic (src/render/integrator-shared.wgsl is the sole
 * WGSL copy; ?parity compares stepGeodesic between the two). Geometry only: no emission, jet or
 * colour. Keep the control flow of stepGeodesic/traceRay structurally identical to the shader.
 */

/** Null-constraint tolerance, RELATIVE to the magnitude of the Hamiltonian's terms (see
 *  hquadScaled). PROVISIONAL until tests/sweep-htol.test.ts picks the final value. */
export const H_TOL = 1e-4;
/** Maximum number of step halvings before a step is accepted as untrusted. PROVISIONAL (see above). */
export const MAX_RETRY = 8;

/** Baseline step length: fine in the strong-field/disk region, long strides through the near-flat
 *  far field. Twin of stepSize() in integrator-shared.wgsl. */
export function stepSize(r: number, rh: number, rOut: number): number {
  if (r > rOut * 1.5) return Math.min(6, Math.max(0.6, 0.04 * r));
  return Math.min(0.5, Math.max(0.002, 0.02 * (r - rh)));
}

/** g^{mu nu} p_mu p_nu (= 2H, exactly 0 for a null geodesic) together with the sum of |terms|.
 *  The constraint drift is judged relative to that sum because near the capture margin
 *  g^tt ~ -A/(Sigma Delta) reaches ~1e2 and H becomes a cancellation of large terms; an absolute
 *  tolerance would sit at the f32 noise floor there. With p_t = 1 and |g^tt| >= 1 outside the
 *  horizon the scale is always >= 1, so this is never looser than an absolute tolerance. */
export function hquadScaled(r: number, th: number, a: number, pt: number, pr: number, pth: number, pphi: number): [number, number] {
  const g = metricUpper(r, th, a);
  const t0 = g.tt * pt * pt, t1 = 2 * g.tphi * pt * pphi, t2 = g.rr * pr * pr, t3 = g.thth * pth * pth, t4 = g.phph * pphi * pphi;
  return [t0 + t1 + t2 + t3 + t4, Math.abs(t0) + Math.abs(t1) + Math.abs(t2) + Math.abs(t3) + Math.abs(t4)];
}

/** Exact analytic continuation through the Boyer-Lindquist polar axis: theta -> -theta (or
 *  2pi - theta at the south pole), phi -> phi + pi, p_theta -> -p_theta. p_phi is unchanged.
 *  Reachable only when a step carries theta past 0 or pi; identity otherwise. */
export function reflectAxis(s: Float64Array): Float64Array {
  const th = s[2];
  if (th >= 0 && th <= Math.PI) return s;
  const o = s.slice();
  o[2] = th < 0 ? -th : 2 * Math.PI - th;
  o[3] = s[3] + Math.PI;
  o[6] = -s[6];
  return o;
}

export interface StepOut { s: Float64Array; ok: boolean; retries: number; dl: number; }

/** One constraint-monitored RK4 step. The step is accepted when the per-step change in the null
 *  constraint is within tolerance; otherwise dl is halved and the step redone, up to maxRetry
 *  halvings. The final attempt is returned either way, with ok = false if it still failed, so the
 *  caller can route the ray to the conserved-quantity classifier instead of trusting it.
 *  NaN drift compares false against the tolerance, so a diverged step is never accepted. */
export function stepGeodesic(s: Float64Array, a: number, dl0: number, hTol = H_TOL, maxRetry = MAX_RETRY): StepOut {
  const [h0] = hquadScaled(s[1], s[2], a, s[4], s[5], s[6], s[7]);
  let dl = dl0;
  for (let k = 0; k < maxRetry; k++) {
    const sN = rk4(s, a, dl);
    const [h1, scale] = hquadScaled(sN[1], sN[2], a, sN[4], sN[5], sN[6], sN[7]);
    if (Math.abs(h1 - h0) <= hTol * scale) return { s: reflectAxis(sN), ok: true, retries: k, dl };
    dl *= 0.5;
  }
  const sN = rk4(s, a, dl);
  const [h1, scale] = hquadScaled(sN[1], sN[2], a, sN[4], sN[5], sN[6], sN[7]);
  const ok = Math.abs(h1 - h0) <= hTol * scale;
  return { s: reflectAxis(sN), ok, retries: maxRetry, dl };
}

export type Fate = "disk" | "captured" | "escaped" | "budget" | "untrusted";
export interface TraceOpts { rIn: number; rOut: number; rObs: number; maxSteps?: number; hTol?: number; maxRetry?: number; }
export interface TraceResult { fate: Fate; s: Float64Array; steps: number; retries: number; thMin: number; rHit?: number; phiHit?: number; }

/** Backward ray trace with the render loop's termination order: disk crossing, then capture, then
 *  escape. thMin is the smallest sampled angular distance from either pole along the trajectory. */
export function traceRay(s0: Float64Array, a: number, o: TraceOpts): TraceResult {
  const rh = 1 + Math.sqrt(Math.max(0, 1 - a * a));
  const maxSteps = o.maxSteps ?? 20000;
  let s = s0, retries = 0, thMin = Math.min(s0[2], Math.PI - s0[2]);
  for (let step = 1; step <= maxSteps; step++) {
    const out = stepGeodesic(s, a, stepSize(s[1], rh, o.rOut), o.hTol, o.maxRetry);
    retries += out.retries;
    if (!out.ok) return { fate: "untrusted", s: out.s, steps: step, retries, thMin };
    const sN = out.s;
    const f0 = s[2] - Math.PI / 2, f1 = sN[2] - Math.PI / 2;
    if (f0 * f1 < 0) {
      const frac = f0 / (f0 - f1);
      const rHit = s[1] + frac * (sN[1] - s[1]);
      if (rHit >= o.rIn && rHit <= o.rOut) {
        return { fate: "disk", s: sN, steps: step, retries, thMin, rHit, phiHit: s[3] + frac * (sN[3] - s[3]) };
      }
    }
    s = sN;
    thMin = Math.min(thMin, s[2], Math.PI - s[2]);
    if (s[1] <= rh * 1.005) return { fate: "captured", s, steps: step, retries, thMin };
    if (s[1] > o.rObs * 1.2) return { fate: "escaped", s, steps: step, retries, thMin };
  }
  return { fate: "budget", s, steps: maxSteps, retries, thMin };
}
```

- [ ] **Step 2: Write `tests/trace.test.ts`**

The near-axis rays are all at a = 0, i = 8° (the spec's worst case). Analytic turning point for ξ ≠ 0, a = 0: Θ(θ) = η − ξ² cot²θ = 0 ⇒ θ_min = atan(|ξ| / √η). `thMin` is sampled at discrete steps, so it can only *overestimate* the true minimum, by at most one step's Δθ; the upper bound of 5 % is that sampling slack, the lower bound of 0.1 % is float slack.

```ts
import { describe, it, expect } from "vitest";
import { screenToState, screenToXiEta } from "../src/physics/camera";
import { conserved } from "../src/physics/geodesic";
import { stepGeodesic, traceRay, reflectAxis, stepSize, hquadScaled } from "../src/physics/trace";

const I8 = (8 * Math.PI) / 180, ROBS = 1000;
const OPTS = { rIn: 3, rOut: 40, rObs: ROBS }; // emit from the a=0 photon orbit outward, like ?shadow

describe("polar axis", () => {
  it("a xi != 0 ray aimed near the pole turns back at the centrifugal barrier and hits the disk", () => {
    // Fails on the shipped POLE_S2 = 1e-3: the floor removes the barrier, theta runs through 0 and
    // the sampled minimum lands at ~6e-4 instead of the analytic 1.16e-2.
    for (const [alpha, beta] of [[0.5, 6], [0.2, 7], [1.0, 8]]) {
      const a = 0;
      const [xi, eta] = screenToXiEta(alpha, beta, a, I8);
      const thMinAn = Math.atan(Math.abs(xi) / Math.sqrt(eta));
      const res = traceRay(screenToState(alpha, beta, a, I8, ROBS), a, OPTS);
      expect(res.fate).toBe("disk");
      expect(res.thMin).toBeGreaterThan(thMinAn * (1 - 1e-3));
      expect(res.thMin).toBeLessThan(thMinAn * 1.05);
    }
  });

  it("keeps xi exact and bounds the eta / null-constraint drift across a near-axis pass", () => {
    const a = 0, alpha = 0.05, beta = 6; // theta_min ~ 1.2e-3 rad, deep inside the old 1.81 deg cone
    const s0 = screenToState(alpha, beta, a, I8, ROBS);
    const c0 = conserved(s0, a);
    const res = traceRay(s0, a, OPTS);
    expect(res.fate).toBe("disk");
    const c1 = conserved(res.s, a);
    expect(Math.abs(c1.Lz - c0.Lz)).toBeLessThan(1e-6); // Killing: exact up to float noise
    // eta and H are NOT conserved to the equatorial test's 1e-4 here: the finite-difference RK4
    // integrator has a documented accuracy ceiling that a near-axis turning point exposes.
    // Measured before this test was written (f64, provisional H_TOL = 1e-4, MAX_RETRY = 8):
    // |d eta| = 0.12 of eta = 36 (0.3%), |H| = 4e-5 -- versus 11% and 1.3e-3 with the monitor
    // effectively off (H_TOL = 1e-2). The bounds below reject the unmonitored integrator and
    // accept the monitored one with ~3x margin; they are measured ceilings, not targets.
    expect(Math.abs(c1.Q - c0.Q) / c0.Q).toBeLessThan(1e-2);
    expect(Math.abs(c1.H)).toBeLessThan(1e-3);
    // The near-axis answer must converge on the on-axis (alpha = 0) limit: the disk-hit radius is
    // continuous in alpha, so alpha = 0.05 must land within half a pixel (0.02 M at the
    // interactive scale of ~24 px/M) of alpha = 0. Measured: 0.0095 M monitored, 0.34 M unmonitored.
    const ref = traceRay(screenToState(0, beta, a, I8, ROBS), a, OPTS);
    expect(ref.fate).toBe("disk");
    expect(Math.abs(res.rHit! - ref.rHit!)).toBeLessThan(0.02);
  });

  it("a xi = 0 ray passes through the axis: phi shifts by pi, p_theta flips, disk is still found", () => {
    // Fails on the shipped code (no reflection): theta runs to -pi/2 and beyond, the disk test
    // never fires at -pi/2, and the ray terminates as escaped.
    const a = 0;
    const s0 = screenToState(0, 6, a, I8, ROBS); // alpha = 0 => xi = 0 exactly
    expect(s0[7]).toBe(0);
    const res = traceRay(s0, a, OPTS);
    expect(res.fate).toBe("disk");
    // With xi = 0 and a = 0, dphi/dl = 0 identically, so the only phi change is the +pi at the crossing.
    const dphi = ((res.phiHit! - s0[3]) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI);
    expect(Math.abs(dphi - Math.PI)).toBeLessThan(1e-9);
    expect(Math.sign(res.s[6])).toBe(-Math.sign(s0[6])); // p_theta flipped by the crossing
  });

  it("reflectAxis is the exact continuation at both poles and the identity elsewhere", () => {
    const inside = new Float64Array([0, 10, 1.0, 0.3, 1, -1, 2, 0.5]);
    expect(reflectAxis(inside)).toBe(inside); // identity returns the same object
    const north = reflectAxis(new Float64Array([0, 10, -0.01, 0.3, 1, -1, 2, 0.5]));
    expect(north[2]).toBeCloseTo(0.01, 12);
    expect(north[3]).toBeCloseTo(0.3 + Math.PI, 12);
    expect(north[6]).toBe(-2);
    expect(north[7]).toBe(0.5); // p_phi unchanged
    const south = reflectAxis(new Float64Array([0, 10, Math.PI + 0.02, 0.3, 1, -1, -2, 0.5]));
    expect(south[2]).toBeCloseTo(Math.PI - 0.02, 12);
    expect(south[3]).toBeCloseTo(0.3 + Math.PI, 12);
    expect(south[6]).toBe(2);
  });

  it("retry exhaustion is reported, not silently accepted", () => {
    const a = 0.9, s = screenToState(3, 2, a, 1.2, ROBS);
    const rh = 1 + Math.sqrt(1 - a * a);
    // hTol < 0 can never be satisfied, so every attempt fails: the step must come back with
    // ok = false, all halvings spent, and dl = dl0 / 2^maxRetry.
    const dl0 = stepSize(s[1], rh, 40);
    const out = stepGeodesic(s, a, dl0, -1, 5);
    expect(out.ok).toBe(false);
    expect(out.retries).toBe(5);
    expect(out.dl).toBeCloseTo(dl0 / 32, 12);
    // traceRay routes an untrusted step out of the loop immediately.
    expect(traceRay(s, a, { ...OPTS, hTol: -1, maxRetry: 3 }).fate).toBe("untrusted");
    // ...and a NaN state is never accepted either (NaN drift compares false).
    const nan = new Float64Array([0, NaN, 1, 0, 1, -1, 0, 0]);
    expect(stepGeodesic(nan, a, 0.1, 1e-4, 2).ok).toBe(false);
  });

  it("hquadScaled scale is >= 1 for p_t = 1 and equals |terms| sum", () => {
    const [h, scale] = hquadScaled(2.05, 1.0, 0.9, 1, -3, 0.5, -2);
    expect(scale).toBeGreaterThanOrEqual(1);
    expect(Math.abs(h)).toBeLessThanOrEqual(scale);
  });
});
```

- [ ] **Step 3: Run the new tests against the SHIPPED floor — they must discriminate**

`POLE_S2` is still `1e-3` at this point.

```bash
npx vitest run tests/trace.test.ts
```

Expected: the first test ("turns back at the centrifugal barrier") FAILS on the `thMin` lower bound (sampled minimum ~6e-4 vs analytic 1.16e-2). The ξ = 0 test **passes** even now, because `traceRay` already includes `reflectAxis` — the reflection is new code this task introduces; the `POLE_S2` half of the fix is what the first test discriminates. The drift-bound test may pass or fail on the old floor (the floored metric does not conserve η inside the cone); either is fine. The reflectAxis, exhaustion and hquadScaled tests pass. **If the first test passes here, something is wrong — STOP and investigate before touching `kerr.ts`.**

- [ ] **Step 4: Change `POLE_S2` to a NaN guard**

Replace `src/physics/kerr.ts:3-9` with:

```ts
// Boyer–Lindquist coordinates are singular on the polar axis: g^{φφ} carries a 1/sin²θ that
// diverges as θ→0,π. POLE_S2 floors sin²θ in that denominator ONLY to stop an exactly-on-axis
// ray with p_φ = 0 from evaluating inf·0 = NaN. It is a NaN guard, not a physics cap: at 1e-12
// the affected cone has half-angle 1e-6 rad, thousands of times below a pixel, so the centrifugal
// barrier is exact everywhere a ray can feel it. (It used to be 1e-3 — a 1.8° cone inside which
// the barrier vanished and rays tunnelled through the axis; see trace.ts for the integrator side.)
export const POLE_S2 = 1e-12;
```

- [ ] **Step 5: Run the tests — all pass**

```bash
npx vitest run tests/trace.test.ts   # expect 6 passed
npm test 2>&1 | tail -5              # expect 68 passed (62 + 6)
npm run build                        # clean
```

If the drift bounds fail, do NOT loosen them — report the measured values (expected at the provisional constants: |Δη|/η ≈ 3e-3, |H| ≈ 4e-5, |ΔrHit| ≈ 0.0095 M). Task 2 settles the constants, but the plan wants to know if these differ materially from the scratch measurement.

- [ ] **Step 6: Commit**

```bash
git add src/physics/trace.ts tests/trace.test.ts src/physics/kerr.ts
git commit -m "Add the CPU integrator twin and make POLE_S2 a NaN guard

The sin^2(th) floor at 1e-3 did not regularize the axis, it removed the
centrifugal barrier inside a 1.81 deg cone: theta ran straight through 0
to large negative values, the disk test (theta - pi/2) never fired at
-pi/2, and the ray escaped, rendering black. Dropping the floor to 1e-12
restores the barrier; stepGeodesic() keeps RK4 honest on the restored
forces by halving the step when the per-step drift of the null constraint
exceeds a RELATIVE tolerance (H is a cancellation of ~1e2 terms near the
capture margin, so an absolute tolerance would sit at f32 noise there);
reflectAxis() continues the xi = 0 rays that genuinely reach the axis.

H_TOL / MAX_RETRY are provisional until the sweep."
git log -1 --format=%B | grep -ci "co-authored"   # must print 0
```

---

### Task 2: Choose `H_TOL` and `MAX_RETRY` from measurement

Spec §3.2: "sweep candidate values, report the resulting step-count and gate impact, and pick from measured data." The sweep is a permanently checked-in, env-gated test so the measurement is reproducible.

**Files:**
- Create: `tests/sweep-htol.test.ts`
- Modify: `src/physics/trace.ts:11-15` (the two constants and their comments)

**Interfaces:**
- Consumes: `traceRay`, `screenToState`, `photonOrbit(a, prograde)` from `src/physics/orbits.ts`.
- Produces: final numeric values of `H_TOL` and `MAX_RETRY`, recorded in the commit message and in `trace.ts`. Task 4 copies them into `integrator-shared.wgsl`.

- [ ] **Step 1: Write `tests/sweep-htol.test.ts`**

```ts
import { describe, it, expect } from "vitest";
import { screenToState } from "../src/physics/camera";
import { traceRay, type Fate } from "../src/physics/trace";
import { photonOrbit } from "../src/physics/orbits";

/**
 * Measurement, not a test: prints the H_TOL x MAX_RETRY table the constants in trace.ts were
 * chosen from. Skipped unless SWEEP=1 (PowerShell: $env:SWEEP=1; npx vitest run tests/sweep-htol.test.ts).
 * Two views: the spec's worst case (a = 0, i = 8 deg, broad axis wedge) and the default UI view
 * (a = 0.9, i = 72 deg). 32x32 rays per view at fovScale 14, the interactive default.
 */
const SWEEP = !!process.env.SWEEP;
const ROBS = 1000, FOV = 14, N = 32, STEPS = 4800;
const VIEWS = [{ a: 0, incl: (8 * Math.PI) / 180 }, { a: 0.9, incl: (72 * Math.PI) / 180 }];

function sweepOne(hTol: number, maxRetry: number) {
  let steps = 0, retries = 0, rays = 0;
  const fates: Record<Fate, number> = { disk: 0, captured: 0, escaped: 0, budget: 0, untrusted: 0 };
  for (const v of VIEWS) {
    const rIn = photonOrbit(v.a, true);
    for (let iy = 0; iy < N; iy++) for (let ix = 0; ix < N; ix++) {
      const alpha = (((ix + 0.5) / N) * 2 - 1) * FOV, beta = -(((iy + 0.5) / N) * 2 - 1) * FOV;
      const r = traceRay(screenToState(alpha, beta, v.a, v.incl, ROBS), v.a,
        { rIn, rOut: 40, rObs: ROBS, maxSteps: STEPS, hTol, maxRetry });
      steps += r.steps; retries += r.retries; rays++; fates[r.fate]++;
    }
  }
  return { meanSteps: steps / rays, retries, fates };
}

/** Disk-hit radius of the spec's reference near-axis ray (theta_min ~ 1.2e-3 rad). */
function nearAxisHit(hTol: number, maxRetry: number) {
  const r = traceRay(screenToState(0.05, 6, 0, (8 * Math.PI) / 180, ROBS), 0,
    { rIn: 3, rOut: 40, rObs: ROBS, hTol, maxRetry });
  return r.fate === "disk" ? r.rHit! : NaN;
}

describe.skipIf(!SWEEP)("H_TOL / MAX_RETRY sweep (SWEEP=1)", () => {
  it("prints the table", () => {
    // Converged reference: two successively tighter settings must agree to 1e-4 M or the
    // "truth" is not converged and the table is meaningless.
    const truthA = nearAxisHit(1e-8, 24), truthB = nearAxisHit(1e-9, 28);
    expect(Math.abs(truthA - truthB)).toBeLessThan(1e-4);
    console.log(`near-axis reference rHit (converged) = ${truthA.toFixed(5)} M`);
    console.log("hTol      retry  meanSteps  retries  untrusted  budget  disk  captured  escaped  |rHit-truth| M");
    for (const hTol of [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]) {
      for (const maxRetry of [4, 8, 12]) {
        const s = sweepOne(hTol, maxRetry);
        const err = Math.abs(nearAxisHit(hTol, maxRetry) - truthA);
        console.log(`${hTol.toExponential(0).padEnd(9)} ${String(maxRetry).padEnd(6)} ${s.meanSteps.toFixed(1).padEnd(10)} ${String(s.retries).padEnd(8)} ${String(s.fates.untrusted).padEnd(10)} ${String(s.fates.budget).padEnd(7)} ${String(s.fates.disk).padEnd(5)} ${String(s.fates.captured).padEnd(9)} ${String(s.fates.escaped).padEnd(8)} ${err.toFixed(4)}`);
      }
    }
  }, 1_800_000);
});
```

- [ ] **Step 2: Run it (takes several minutes)**

```powershell
$env:SWEEP = "1"; npx vitest run tests/sweep-htol.test.ts; Remove-Item Env:SWEEP
```

Paste the full table into the task report. Confirm `npm test` still reports the file as **skipped** (1 skipped, 68 passed) without `SWEEP`.

- [ ] **Step 3: Choose the constants by these rules, in this order**

1. **f32 floor:** discard any `hTol < 1e-5`. Reason: `hquadScaled` is a sum of five products each carrying f32 rounding ~6e-8 relative, and the RK4 state feeding it carries ~1e-6 relative error after one f32 step; a relative tolerance below ~1e-5 would make the GPU (but not the f64 CPU) retry on noise. Task 4's GPU parity block checks this assumption directly.
2. Among the remaining rows, keep those with **`untrusted = 0` on both views** and **`|rHit − truth| < 0.02 M`** (half a pixel at the interactive scale of ~24 px/M). For orientation, the scratch measurement gave 0.34 M at 1e-2 (fails), 0.0095 M at 1e-3 and 1e-4, 0.0000 M at 1e-5; and on the default a = 0.9 / 72° view the monitor at 1e-4 added ~4 retries per ~500-step ray, i.e. under 1 % cost.
3. Take the **largest `hTol`** that survives (cheapest), then the **smallest `maxRetry`** that survives at that `hTol`, and use **the next size up** from the table as the margin (e.g. survives at 4 → ship 8).
4. If no row survives rules 1–2, STOP and report the table; do not ship a value that fails them.

Record the chosen pair, the surviving rows, and the `meanSteps` delta versus the `1e-2 / 4` row (the cheapest configuration, effectively "no monitoring") in the commit message.

- [ ] **Step 4: Write the chosen values into `trace.ts`**

Replace the two constant declarations and comments:

```ts
/** Null-constraint tolerance, RELATIVE to the magnitude of the Hamiltonian's terms (see
 *  hquadScaled). Chosen from tests/sweep-htol.test.ts (SWEEP=1): <one line: which rows survived
 *  and why this one>. Twin constant in integrator-shared.wgsl. */
export const H_TOL = <chosen>;
/** Maximum step halvings before a step is returned as untrusted. Chosen from the same sweep:
 *  <one line>. Twin constant in integrator-shared.wgsl. */
export const MAX_RETRY = <chosen>;
```

- [ ] **Step 5: Re-run the unit tests with the final constants**

```bash
npm test 2>&1 | tail -5     # 68 passed, 1 skipped
```

If `tests/trace.test.ts` now fails on a tolerance, that is a finding about the chosen constants, not about the test. Report it; do not adjust the test.

- [ ] **Step 6: Commit**

```bash
git add tests/sweep-htol.test.ts src/physics/trace.ts
git commit -m "Pick H_TOL and MAX_RETRY from a measured sweep

<paste the table>

Rules: hTol >= 1e-5 (f32 floor), untrusted = 0 on both views, near-axis
rHit within 0.02 M of the converged reference; largest hTol, then
smallest retry cap with one size of margin. Chosen: H_TOL = <x>,
MAX_RETRY = <y>. Mean steps/ray vs unmonitored: <delta>."
git log -1 --format=%B | grep -ci "co-authored"   # 0
```

---

### Task 3: Move the integrator WGSL into a shared fragment (pure refactor, bit-identical)

No physics changes in this task. The deliverable is "the renderer behaves identically while the metric + integrator WGSL live in one file that the parity route also compiles." The gates ARE the test: they must reproduce the baseline **exactly**.

**Files:**
- Create: `src/render/integrator-shared.wgsl`
- Modify: `src/render/raytrace.wgsl:16-59` (delete), comments at `:22-26` go with the block
- Modify: `src/render/parity.wgsl:3-14` (delete the duplicate helpers)
- Modify: `src/render/gpu.ts:2-6` (import) and `:93-98` (prepend)
- Modify: `src/test/parity.browser.ts:4` (import) and `:31` (prepend)

**Interfaces:**
- Produces: WGSL symbols `PI`, `delta_`, `sigma_`, `bigA_`, `POLE_S2`, `gUp`, `gLow`, `omegaKep`, `hquad`, `State`, `rhs`, `addS`, `rk4`, `stepSize(r: f32, rh: f32, rOut: f32) -> f32` available to any module that prepends the fragment. Task 4 adds `stepGeodesic` here.

- [ ] **Step 1: Create `src/render/integrator-shared.wgsl`**

Cut `raytrace.wgsl` lines 16–59 (from `const PI` through the closing brace of `rk4`) **verbatim** into the new file, under this header, and append `stepSize`:

```wgsl
// ---- Kerr metric + geodesic integrator. Twin of src/physics/kerr.ts, geodesic.ts, trace.ts. ----
//
// This is the SOLE WGSL copy of the metric helpers and the integrator. It is prepended as a plain
// string to raytrace.wgsl (by gpu.ts) and to parity.wgsl / integrator-parity.wgsl (by
// parity.browser.ts), so the ?parity route compiles the same bytes the renderer runs. Do not
// inline a second copy anywhere: a duplicate would let the renderer and the gate disagree.
//
// Self-contained: no uniforms, no bindings. Everything the render loop needs from U is passed in.

<lines 16-59 of raytrace.wgsl, verbatim, including the POLE_S2 comment and value 1e-3>

// Baseline step length: fine in the strong-field/disk region, long strides through the near-flat
// far field (curvature ~M/r^3 is negligible there) so we don't burn thousands of steps just
// travelling in from the distant observer. Twin of stepSize() in trace.ts.
fn stepSize(r: f32, rh: f32, rOut: f32) -> f32 {
  if (r > rOut * 1.5) { return clamp(0.04 * r, 0.6, 6.0); }
  return clamp(0.02 * (r - rh), 0.002, 0.5);
}
```

`POLE_S2` stays at `1e-3` in this task (bit-identical refactor). Task 4 changes it.

- [ ] **Step 2: Use `stepSize` in the render loop**

In `raytrace.wgsl`, replace (post-deletion line numbers will have shifted; match on text):

```wgsl
    var dl = clamp(0.02 * (r - rh), 0.002, 0.5);
    if (r > U.rOut * 1.5) { dl = clamp(0.04 * r, 0.6, 6.0); }
    let sNew = rk4(s, a, dl);
```

with

```wgsl
    let dl = stepSize(r, rh, U.rOut);
    let sNew = rk4(s, a, dl);
```

and delete the three comment lines above it that describe the distance-adaptive step (they moved with `stepSize`). Keep the `// dl > 0 with p_r < 0 integrates INWARD along the reversed worldline.` line.

- [ ] **Step 3: Prepend in `gpu.ts`**

Add `import integratorSharedWGSL from "./integrator-shared.wgsl?raw";` after line 5, and change line 98 to:

```ts
    const cMod = this.device.createShaderModule({ code: shadowSharedWGSL + cameraSharedWGSL + integratorSharedWGSL + raytraceWGSL });
```

Extend the comment block above it with one sentence: `integrator-shared.wgsl (metric + RK4 + step controller) is likewise the sole copy; it defines PI, so raytrace.wgsl no longer does.`

- [ ] **Step 4: De-duplicate `parity.wgsl` and prepend there too**

Delete `parity.wgsl` lines 3–14 (`delta_` through `omegaKep`) and change its header comment to:

```wgsl
// Parity test shader for the metric/orbit/g-factor helpers. The helpers themselves are NOT
// copied here: integrator-shared.wgsl is prepended by parity.browser.ts, exactly as gpu.ts
// prepends it to raytrace.wgsl, so this compares the TypeScript core against the shipped bytes.
```

In `parity.browser.ts` add `import integratorSharedWGSL from "../render/integrator-shared.wgsl?raw";` next to the other `?raw` imports and change line 31 to:

```ts
  const mod = device.createShaderModule({ code: integratorSharedWGSL + parityWGSL });
```

- [ ] **Step 5: Gates — must be bit-identical to the baseline**

```bash
npm run build                 # clean (tsc + vite)
npm test 2>&1 | tail -5       # 68 passed, 1 skipped
npm run verify:gpu            # PARITY PASS maxRelErr=9.690e-7 over 45 cases; SHADOW PASS 4.49 M / 0.864
```

`?parity` must print **exactly 9.690e-7**. A different number means a shader compile problem (see Global Constraints) or an accidental edit inside the moved block — `git diff` the fragment against `git show main:src/render/raytrace.wgsl` lines 16–59 to prove it is verbatim.

- [ ] **Step 6: Commit**

```bash
git add src/render/integrator-shared.wgsl src/render/raytrace.wgsl src/render/parity.wgsl src/render/gpu.ts src/test/parity.browser.ts
git commit -m "Move the metric and RK4 integrator into a shared WGSL fragment

Pure refactor: raytrace.wgsl and parity.wgsl now both compile
integrator-shared.wgsl, so the metric parity case checks shipped bytes
instead of a hand-synced copy. Gates unchanged: parity 9.690e-7 over 45,
shadow 4.49 M / 0.864."
git log -1 --format=%B | grep -ci "co-authored"   # 0
```

---

### Task 4: GPU side of the fix, gated by an integrator parity case

Now the physics lands on the GPU: `POLE_S2 → 1e-12`, `stepGeodesic`, `reflectAxis`, the loop rewire — and a parity block that runs the shipped `stepGeodesic` bytes on cases that exercise the retry path and the reflection, comparing against `trace.ts`.

**Files:**
- Modify: `src/render/integrator-shared.wgsl` (POLE_S2 value + comment; append the new functions)
- Modify: `src/render/raytrace.wgsl` (main loop)
- Create: `src/render/integrator-parity.wgsl`
- Modify: `src/test/parity.browser.ts` (new block before the `return`)

**Interfaces:**
- Consumes: `H_TOL`, `MAX_RETRY`, `stepSize`, `stepGeodesic`, `traceRay` from `trace.ts`; `screenToState` from `camera.ts`.
- Produces: WGSL `struct StepOut { s: State, dl: f32, retries: u32, ok: bool }`, `fn stepGeodesic(s: State, a: f32, dl0: f32) -> StepOut`, `fn reflectAxis(s: State) -> State`, `fn hquadScaled(r, th, a, p: vec4<f32>) -> vec2<f32>`, consts `H_TOL`, `MAX_RETRY`.

- [ ] **Step 1: Update `POLE_S2` in the fragment**

Replace the `POLE_S2` comment and constant in `integrator-shared.wgsl` with:

```wgsl
// POLE_S2 floors sin^2(th) in the divergent 1/sin^2 denominator of g^{phi phi}. It is a NaN guard
// (inf * 0 for an exactly-on-axis ray with p_phi = 0), NOT a physics cap: at 1e-12 the affected
// cone is 1e-6 rad, far below a pixel. Matches src/physics/kerr.ts. The old 1e-3 removed the
// centrifugal barrier inside a 1.8 deg cone and let rays tunnel through the axis.
const POLE_S2 = 1e-12;
```

- [ ] **Step 2: Append the monitored step to the fragment**

Append after `stepSize`. `H_TOL` / `MAX_RETRY` are the values Task 2 chose — copy them from `trace.ts`, do not retype from memory.

```wgsl
// ---- Constraint-monitored stepping. Twin of stepGeodesic() in trace.ts. --------------------------
// For a null geodesic H = 1/2 g^{mu nu} p_mu p_nu = 0 exactly. RK4 with finite-difference forces
// drifts off that, worst where the theta force is steep (the restored 1/sin^2 barrier). After each
// step the CHANGE in H is compared against a tolerance RELATIVE to the size of the terms being
// cancelled (near the capture margin g^tt ~ 1e2, so an absolute tolerance would sit at f32 noise);
// on failure dl is halved and the step redone, up to MAX_RETRY halvings.
const H_TOL = <value from trace.ts>;
const MAX_RETRY = <value from trace.ts>u;

// vec2(g^{mu nu} p_mu p_nu, sum of |terms|). gUp indices: 0=tt, 1=tphi, 2=rr, 3=thth, 4=phph.
fn hquadScaled(r: f32, th: f32, a: f32, p: vec4<f32>) -> vec2<f32> {
  let g = gUp(r, th, a);
  let t0 = g[0]*p.x*p.x; let t1 = 2.0*g[1]*p.x*p.w; let t2 = g[2]*p.y*p.y;
  let t3 = g[3]*p.z*p.z; let t4 = g[4]*p.w*p.w;
  return vec2<f32>(t0 + t1 + t2 + t3 + t4, abs(t0) + abs(t1) + abs(t2) + abs(t3) + abs(t4));
}

// Exact analytic continuation through the polar axis: th -> -th (2pi - th at the south pole),
// phi -> phi + pi, p_th -> -p_th; p_phi unchanged. Identity unless a step carried th past 0 or pi.
fn reflectAxis(s: State) -> State {
  let th = s.x.z;
  if (th >= 0.0 && th <= PI) { return s; }
  let thR = select(2.0*PI - th, -th, th < 0.0);
  return State(vec4<f32>(s.x.x, s.x.y, thR, s.x.w + PI), vec4<f32>(s.p.x, s.p.y, -s.p.z, s.p.w));
}

struct StepOut { s: State, dl: f32, retries: u32, ok: bool };
// The final attempt is returned either way; ok = false tells the caller the trajectory can no
// longer be trusted. A NaN drift compares false against the tolerance, so it is never accepted.
fn stepGeodesic(s: State, a: f32, dl0: f32) -> StepOut {
  let h0 = hquadScaled(s.x.y, s.x.z, a, s.p).x;
  var dl = dl0;
  for (var k = 0u; k < MAX_RETRY; k++) {
    let sN = rk4(s, a, dl);
    let hs = hquadScaled(sN.x.y, sN.x.z, a, sN.p);
    if (abs(hs.x - h0) <= H_TOL * hs.y) { return StepOut(reflectAxis(sN), dl, k, true); }
    dl = dl * 0.5;
  }
  let sN = rk4(s, a, dl);
  let hs = hquadScaled(sN.x.y, sN.x.z, a, sN.p);
  let ok = abs(hs.x - h0) <= H_TOL * hs.y;
  return StepOut(reflectAxis(sN), dl, MAX_RETRY, ok);
}
```

- [ ] **Step 3: Rewire the render loop in `raytrace.wgsl`**

Replace

```wgsl
    let dl = stepSize(r, rh, U.rOut);
    let sNew = rk4(s, a, dl);
```

with

```wgsl
    let st = stepGeodesic(s, a, stepSize(r, rh, U.rOut));
    // The constraint monitor spent all its retries: this trajectory can no longer be trusted.
    // Leave the loop with resolved == false so the conserved-quantity classifier below decides
    // captured/escaped from (xi, eta) instead of a garbage state being accepted as a real hit.
    if (!st.ok) { break; }
    let dl = st.dl; let sNew = st.s;
```

Everything after (`jetAccum` uses `dl`; disk crossing uses `s` / `sNew`; `s = sNew`) is unchanged. Confirm by reading the loop that `resolved` is still only set in the three real terminations and consulted only after the loop.

- [ ] **Step 4: Create `src/render/integrator-parity.wgsl`**

```wgsl
// Entry point for CPU<->GPU parity of the constraint-monitored integrator step.
// Input per case: state (x, p) + a + dl0. Output per case: new state (x, p) + meta
// (retries, ok ? 1 : 0, H_TOL, MAX_RETRY) so the constants are checked too, not just the step.
//
// This file deliberately contains NO copy of the integrator. integrator-shared.wgsl is prepended
// by parity.browser.ts, exactly as gpu.ts prepends it to raytrace.wgsl.
struct StepIn { x: vec4<f32>, p: vec4<f32>, a: f32, dl0: f32, pad0: f32, pad1: f32 };
struct StepRes { x: vec4<f32>, p: vec4<f32>, meta: vec4<f32> };
@group(0) @binding(0) var<storage, read> inp: array<StepIn>;
@group(0) @binding(1) var<storage, read_write> outp: array<StepRes>;

@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= arrayLength(&inp)) { return; }
  let v = inp[gid.x];
  let o = stepGeodesic(State(v.x, v.p), v.a, v.dl0);
  outp[gid.x] = StepRes(o.s.x, o.s.p, vec4<f32>(f32(o.retries), select(0.0, 1.0, o.ok), H_TOL, f32(MAX_RETRY)));
}
```

- [ ] **Step 5: Add the integrator parity block to `parity.browser.ts`**

Add the imports:

```ts
import { stepGeodesic, stepSize, H_TOL, MAX_RETRY } from "../physics/trace";
import integratorParityWGSL from "../render/integrator-parity.wgsl?raw";
```

Insert before the final `return { maxErr, rows: ... }`:

```ts
  // --- integrator parity (CPU trace.ts vs the SHIPPED stepGeodesic in integrator-shared.wgsl) ---
  // Same shared-fragment discipline as the shadow and camera blocks. The cases are found by
  // walking real trajectories on the CPU until a state with the wanted property appears, so each
  // one is guaranteed to exercise the branch it is named for; a case that cannot be found throws,
  // which fails the route rather than silently testing nothing.
  const I8 = (8 * Math.PI) / 180;
  type ICase = { label: string; s: Float64Array; a: number; dl0: number };
  const icases: ICase[] = [];
  const rhOf = (a: number) => 1 + Math.sqrt(Math.max(0, 1 - a * a));
  /** Walk from s0 with the shipped step controller until pred(s, out) holds; return that pre-step state. */
  function findState(label: string, s0: Float64Array, a: number, pred: (s: Float64Array, out: ReturnType<typeof stepGeodesic>) => boolean): ICase {
    let s = s0;
    for (let k = 0; k < 20000; k++) {
      const dl0 = stepSize(s[1], rhOf(a), 40);
      const out = stepGeodesic(s, a, dl0);
      if (pred(s, out)) return { label, s, a, dl0 };
      if (!out.ok || out.s[1] <= rhOf(a) * 1.005 || out.s[1] > 1200) break;
      s = out.s;
    }
    throw new Error(`integrator parity: no state found for case "${label}"`);
  }
  // far field, long stride, a != 0
  icases.push({ label: "far", s: screenToState(4, 3, 0.9, 1.2, 1000), a: 0.9, dl0: stepSize(1000, rhOf(0.9), 40) });
  // strong field, equatorial (beta = 0 at i = pi/2 stays in the plane); alpha = 2 => L_z = 2, well
  // inside the prograde critical curve at a = 0.9, so the ray reaches r < 6 before capture
  icases.push(findState("strong-eq", screenToState(2, 0, 0.9, Math.PI / 2, 1000), 0.9, (s) => s[1] < 6));
  // near the capture margin (a = 0, b = 4 is captured; rh = 2)
  icases.push(findState("near-horizon", screenToState(4, 0, 0, Math.PI / 2, 1000), 0, (s) => s[1] < 2.3));
  // approaching the axis, before the turning point, no retry expected
  icases.push(findState("near-axis", screenToState(0.05, 6, 0, I8, 1000), 0, (s) => s[2] < 0.01));
  // a step that the monitor actually halves (retries >= 1) -- the whole point of the feature
  icases.push(findState("retry", screenToState(0.05, 6, 0, I8, 1000), 0, (_, out) => out.retries >= 1));
  // a step that crosses the axis (xi = 0): only reflectAxis can flip the sign of p_theta here
  icases.push(findState("reflect", screenToState(0, 6, 0, I8, 1000), 0, (s, out) => out.s[6] * s[6] < 0));
  const iin = device.createBuffer({ size: icases.length * 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const iarr = new Float32Array(icases.length * 12);
  icases.forEach((c, i) => { iarr.set([c.s[0], c.s[1], c.s[2], c.s[3], c.s[4], c.s[5], c.s[6], c.s[7], c.a, c.dl0, 0, 0], i * 12); });
  device.queue.writeBuffer(iin, 0, iarr);
  const iout = device.createBuffer({ size: icases.length * 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const iread = device.createBuffer({ size: icases.length * 48, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const imod = device.createShaderModule({ code: integratorSharedWGSL + integratorParityWGSL });
  const ipipe = device.createComputePipeline({ layout: "auto", compute: { module: imod, entryPoint: "main" } });
  const ibind = device.createBindGroup({ layout: ipipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: iin } }, { binding: 1, resource: { buffer: iout } }] });
  const ienc = device.createCommandEncoder();
  const icp = ienc.beginComputePass(); icp.setPipeline(ipipe); icp.setBindGroup(0, ibind); icp.dispatchWorkgroups(icases.length); icp.end();
  ienc.copyBufferToBuffer(iout, 0, iread, 0, icases.length * 48);
  device.queue.submit([ienc.finish()]);
  await iread.mapAsync(GPUMapMode.READ);
  const igpu = new Float32Array(iread.getMappedRange().slice(0));
  icases.forEach((c, i) => {
    // The GPU starts from the f32-rounded state, so compare against the CPU stepping that same
    // rounded state; otherwise the input rounding (not the shader) would dominate the error.
    const s32 = Float64Array.from(Array.from(c.s, Math.fround));
    const cpu = stepGeodesic(s32, c.a, Math.fround(c.dl0));
    for (let k = 0; k < 8; k++) {
      const got = igpu[i * 12 + k], want = cpu.s[k];
      maxErr = Math.max(maxErr, Math.abs(got - want) / (1 + Math.abs(want)));
    }
    // Retry count and ok flag: any mismatch scores >= 1.0 and fails the route outright. A retry
    // mismatch on the "retry" case means H_TOL is at the f32 noise floor -- see the plan, Task 4.
    maxErr = Math.max(maxErr, Math.abs(igpu[i * 12 + 8] - cpu.retries));
    maxErr = Math.max(maxErr, Math.abs(igpu[i * 12 + 9] - (cpu.ok ? 1 : 0)));
    // The constants themselves, so a desync between trace.ts and the fragment cannot hide.
    maxErr = Math.max(maxErr, Math.abs(igpu[i * 12 + 10] - H_TOL) / H_TOL);
    maxErr = Math.max(maxErr, Math.abs(igpu[i * 12 + 11] - MAX_RETRY));
  });
  console.log("integrator parity cases", icases.map((c) => c.label), "cpu retries", icases.map((c) => stepGeodesic(Float64Array.from(Array.from(c.s, Math.fround)), c.a, Math.fround(c.dl0)).retries));
  return { maxErr, rows: cases.length + tcases.length + jcases.length + scases.length + ccases.length + icases.length };
```

If `findState("retry", …)` throws because the chosen `H_TOL` never halves a step on the α = 0.05 ray, move the ray closer to the axis (α = 0.02, then 0.01, then 0.005 — each has a smaller θ_min and a steeper barrier) until a retry appears, and note the α used. If none of those triggers a retry, the monitor is inert at the chosen constants on the worst ray this project renders — report that as a finding rather than shipping a gate that tests nothing.

- [ ] **Step 6: Run the gates and re-baseline**

```bash
npm run build
npm test 2>&1 | tail -5     # 68 passed, 1 skipped
npm run verify:gpu
```

Expected: `PARITY PASS` with **51 cases** (45 + 6) and a **new** maxRelErr — record it to four significant digits. It will be larger than 9.690e-7 because a full f32 RK4 step near the axis carries more error than a single metric component; anything up to ~1e-4 is plausible. `SHADOW PASS` with a possibly slightly different radius/calibration — record both.

Then read the browser console for the `integrator parity cases … cpu retries` line (open `http://localhost:5173/?parity` in Chrome DevTools, or a scratch playwright script with `page.on("console")`). Confirm the "retry" case shows `retries >= 1` on the CPU. If it shows 0 the case is vacuous — fix `findState`, do not proceed.

**If parity FAILS on the retry-count comparison** (maxRelErr ≥ 1): that is the f32 floor finding. The GPU halves where the CPU does not (or vice versa). Go back to Task 2's table, choose the next-larger `H_TOL` that survived rules 2–3, update **both** `trace.ts` and the fragment, re-run `npm test` and `verify:gpu`, and record the bounce in the commit message. Do not touch the parity threshold.

- [ ] **Step 7: Commit**

```bash
git add src/render/integrator-shared.wgsl src/render/raytrace.wgsl src/render/integrator-parity.wgsl src/test/parity.browser.ts
git commit -m "Restore the true near-axis forces on the GPU with a monitored step

POLE_S2 -> 1e-12 (NaN guard only), stepGeodesic halves the step when the
per-step drift of the null constraint exceeds H_TOL relative to the size
of the cancelled terms, reflectAxis continues xi = 0 rays exactly, and an
untrusted step breaks to the (xi, eta) classifier. A new ?parity block
runs the shipped stepGeodesic bytes on far/strong/near-horizon/near-axis
/retry/reflect states against trace.ts, including the retry count and the
constants.

Re-baselined gates: parity maxRelErr <new> over 51 cases (was 9.690e-7
over 45; the new integrator cases dominate), shadow <new> M /
calibration <new> (was 4.49 / 0.864)."
git log -1 --format=%B | grep -ci "co-authored"   # 0
```

---

### Task 5: Visual confirmation probe and documentation

**Files:**
- Create: `scripts/probe-axis.mjs`
- Modify: `README.md:33` (routes list) and `:58` (the false "regularized" paragraph), plus the Status section's gate numbers
- Modify: `docs/specs/2026-07-21-polar-axis-regularization-design.md:4` (status)

**Interfaces:** consumes the DOM slider ids `#spin`, `#incl`, `#sky`, `#jet` and the canvas `#c` (`index.html:66-116`). The panel is fixed at the left (`left:26px; width:298px`), so the viewport's centre column is clear of it.

- [ ] **Step 1: Write `scripts/probe-axis.mjs`**

The measurement is self-normalising: the axis column is compared against a column 60 px to its right (outside any wedge: the wedge half-width is ~29 px at i = 8° and ~4 px at i = 72°), over the rows above the shadow. Legitimately dark rows (empty sky at the top of the 72° view) are dark in both columns and cancel.

```js
// Visual confirmation for the polar-axis fix: renders a = 0 with the sky and jet off at i = 72
// and i = 8 degrees, and checks the column through the pole is no darker than a column beside it.
// main.ts reads only the `steps` query parameter, so the sliders are driven through the DOM and
// each value is read back to prove it was applied.
//
//   npm run dev                    # in another terminal
//   node scripts/probe-axis.mjs    # writes axis-i72.png / axis-i8.png, exits non-zero on failure
import { chromium } from "playwright-core";

const BASE = process.env.BASE || "http://localhost:5173";
const browser = await chromium.launch({
  channel: "chrome", headless: true,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist"],
});
const page = await browser.newPage({ viewport: { width: 1000, height: 680 } });
const diags = [];
page.on("console", (m) => { if (m.type() === "warning" || m.type() === "error") diags.push(m.text()); });
page.on("pageerror", (e) => diags.push("PAGEERROR " + e.message));
await page.goto(BASE + "/", { waitUntil: "load", timeout: 20000 });

async function setSlider(id, value) {
  const applied = await page.evaluate(([id, v]) => {
    const el = document.getElementById(id);
    el.value = v; el.dispatchEvent(new Event("input", { bubbles: true }));
    return el.value;
  }, [id, String(value)]);
  if (Number(applied) !== Number(value)) throw new Error(`slider #${id} did not take ${value} (got ${applied})`);
}

/** Dark-pixel fraction of column x over rows [y0, y1). Decoded in-page from the PNG so no PNG lib is needed. */
async function darkFraction(png, x, y0, y1) {
  return page.evaluate(async ([b64, x, y0, y1]) => {
    const img = new Image(); img.src = "data:image/png;base64," + b64; await img.decode();
    const cv = document.createElement("canvas"); cv.width = img.width; cv.height = img.height;
    const g = cv.getContext("2d"); g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, cv.width, cv.height).data;
    let dark = 0;
    for (let y = y0; y < y1; y++) { const i = (y * cv.width + x) * 4; if (d[i] + d[i + 1] + d[i + 2] < 20) dark++; }
    return dark / Math.max(1, y1 - y0);
  }, [png.toString("base64"), x, y0, y1]);
}
/** Top edge of the dark run containing the frame centre on column x (the shadow's upper edge). */
async function shadowTop(png, x) {
  return page.evaluate(async ([b64, x]) => {
    const img = new Image(); img.src = "data:image/png;base64," + b64; await img.decode();
    const cv = document.createElement("canvas"); cv.width = img.width; cv.height = img.height;
    const g = cv.getContext("2d"); g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, cv.width, cv.height).data;
    const lum = (y) => { const i = (y * cv.width + x) * 4; return d[i] + d[i + 1] + d[i + 2]; };
    let top = cv.height >> 1; while (top > 0 && lum(top) < 20) top--;
    return top;
  }, [png.toString("base64"), x]);
}

let failed = false;
for (const incl of [72, 8]) {
  await setSlider("spin", 0); await setSlider("sky", 0); await setSlider("jet", 0); await setSlider("incl", incl);
  await page.waitForTimeout(6000); // EMA at ~15 fps converges well inside this
  const png = await page.screenshot({ type: "png" });
  const file = `axis-i${incl}.png`;
  (await import("node:fs")).writeFileSync(file, png);
  const cx = 500, top = await shadowTop(png, cx);
  const y0 = 0, y1 = Math.max(1, top - 4);
  const axis = await darkFraction(png, cx, y0, y1), side = await darkFraction(png, cx + 60, y0, y1);
  // Before the fix the axis column is ~100% dark above the shadow at i = 8 and carries a solid
  // band at i = 72; the side column is lit disk. Allow 10% for the ISCO edge and AA jitter.
  const ok = top > 10 && axis - side < 0.10;
  console.log(`${ok ? "✓ PASS" : "✗ FAIL"}  i=${incl}°  shadowTop=${top}px  darkFrac axis=${axis.toFixed(3)} side=${side.toFixed(3)}  -> ${file}`);
  if (!ok) failed = true;
}
if (diags.length) console.log("console diagnostics:", diags.join(" | "));
await browser.close();
process.exit(failed ? 1 : 0);
```

- [ ] **Step 2: Prove the probe discriminates**

First against the **old** shader, so the probe is known to catch the defect:

```bash
git status --short   # must be clean before this step
git checkout main -- src/render/raytrace.wgsl src/render/gpu.ts src/render/parity.wgsl   # old GPU path, no fragment
node scripts/probe-axis.mjs   # expect ✗ FAIL at i=8 (axis darkFrac ~1.0 vs side ~0), probably FAIL at i=72
git checkout fix/polar-axis -- src/render/raytrace.wgsl src/render/gpu.ts src/render/parity.wgsl
git status --short            # must show only scripts/probe-axis.mjs (untracked) — nothing else modified
```

(`git checkout main -- <files>` only touches those three files; it does not switch branches. If `git status` shows anything else modified, restore it before continuing.) Then on the branch:

```bash
node scripts/probe-axis.mjs   # expect ✓ PASS at both inclinations
```

Look at `axis-i8.png` and `axis-i72.png` yourself: the black wedge from the shadow to the frame edge must be gone, and the image otherwise unchanged. Both PNGs are build artefacts — do not commit them; add `axis-i*.png` to `.gitignore` if it is not already covered (check `git status`).

- [ ] **Step 3: README**

Replace the paragraph at `README.md:58` (starts `**Visual polish:**`) so the axis sentence reads:

> The Boyer–Lindquist polar axis (where g^φφ ∝ 1/sin²θ diverges) is handled physically rather than capped: `POLE_S2` is now a 1e-12 NaN guard (it was a 1e-3 floor that removed the centrifugal barrier inside a 1.8° cone, letting rays tunnel through the axis undetected and render a black wedge from each pole). The integrator halves its step whenever the per-step drift of the null constraint H = ½ g^μν p_μ p_ν exceeds `H_TOL` = <value> relative to the magnitude of the cancelled terms (up to `MAX_RETRY` = <value> halvings, chosen from a measured sweep in `tests/sweep-htol.test.ts`), rays that exhaust the retries are handed to the (ξ, η) classifier, and the rare ξ = 0 rays that genuinely reach the axis are continued analytically (θ → −θ, φ → φ + π, p_θ → −p_θ). The step controller, monitored step and axis crossing are gated CPU↔GPU by `?parity` on the shipped shader bytes. This changes pixels near the axis and, because the monitor is global, marginally elsewhere.

Update the `?parity` line at `README.md:33` to mention the integrator: `… including the shadow-edge classifier, the camera mapping and the constraint-monitored integrator step, all compiled from the same shared WGSL fragments the renderer uses`.

In the Status section, replace the recorded gate numbers with the Task 4 values (parity maxRelErr over 51 cases; shadow radius and calibration) and add one line: `Mean steps per ray rose by <Task 2 delta> with monitoring on.`

- [ ] **Step 4: Spec status**

Change `docs/specs/2026-07-21-polar-axis-regularization-design.md:4` to `**Status:** implemented on fix/polar-axis (plan: docs/plans/2026-09-16-polar-axis-regularization.md, which records five deviations from this text — read those before trusting §1's "captured" mechanism or §3.2's literal |H| test).`

- [ ] **Step 5: Final gates**

```bash
npm run build
npm test 2>&1 | tail -5       # 68 passed, 1 skipped
npm run verify:gpu            # PARITY PASS at the Task 4 number over 51 cases; SHADOW PASS at the Task 4 numbers
node scripts/probe-axis.mjs   # ✓ PASS x2
```

- [ ] **Step 6: Commit**

```bash
git add scripts/probe-axis.mjs README.md docs/specs/2026-07-21-polar-axis-regularization-design.md .gitignore
git commit -m "Add the axis probe and document the polar-axis fix

scripts/probe-axis.mjs drives the sliders (verifying each took), renders
a = 0 with sky and jet off at i = 72 and 8 deg, and asserts the column
through the pole is no darker than a column beside it. Fails on main,
passes here. README no longer claims the axis was regularized."
git log -1 --format=%B | grep -ci "co-authored"   # 0
```

Do not merge. Report the final gate numbers, the chosen constants, the sweep table, and the two PNGs' locations. Merging is the user's call.

---

## Self-review against the spec

- §3.1 floor → 1e-12 in both twins: Task 1 (CPU), Task 4 (GPU). ✓
- §3.2 constraint-monitored stepping with bounded retries, empirically chosen tolerance: Task 1 (mechanism), Task 2 (measurement), Task 4 (GPU). Deviations 2–4 documented. ✓
- §3.2 "confirm the routing against the current loop": Deviation 4 + Task 4 Step 3 instruction. ✓
- §3.3 analytic axis crossing: `reflectAxis` in Task 1 and Task 4. ✓
- §4.1 tests 1–3: Task 1 tests 2, 1, 3 respectively; test 1 fails on the shipped floor (Step 3 proves it). ✓
- §4.2 re-baseline recorded explicitly, threshold never loosened: Task 4 Step 6, Global Constraints. ✓
- §4.3 visual confirmation via DOM sliders with applied-value assertion: Task 5. ✓
- §6 out of scope respected: no Carter equations, no Kerr–Schild, no camera recalibration. ✓
- Type consistency: `StepOut` fields `{ s, ok, retries, dl }` in TS and `{ s, dl, retries, ok }` in WGSL — different field order is fine (constructed positionally in WGSL, by name in TS); `stepGeodesic` signature `(s, a, dl0)` on both sides with TS carrying the two optional overrides; `hquadScaled` returns `[H, scale]` / `vec2(H, scale)`. ✓
- Small side scope, declared: Task 3 also makes the pre-existing metric parity case compile shipped bytes (by deleting `parity.wgsl`'s private copies). Zero pixel or gate impact; closes a twin gap the memory notes list as a follow-up.
