# Faster Integrator (Carter Equations in Mino Time) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the renderer's Hamiltonian RK4 integrator with Carter's separated null-geodesic equations in Mino
time. Use an adaptive Dormand–Prince 5(4) step and dense output for emission sampling and the disk crossing. Live frames
must get faster with nothing scientific changed and no ray less accurate.

**Architecture:**
- **CPU twin** `src/physics/trace-mino.ts`: constants, RHS, DP5(4) step with error norm and guard, Hermite dense output,
  conversion to the existing `State`, and a ray loop.
- **Accuracy sweep** `tests/sweep-mino.test.ts`: sets the tolerance and decides go/no-go against a converged reference.
- **WGSL twin** in `integrator-shared.wgsl`, gated by `?parity`.
- **`raytrace.wgsl`:** the ray loop steps in Mino time. Jet, flow and hotspot sample points and the disk crossing come
  from the dense output. Every emitter keeps reading (x, p, dl) as today.
- **Untouched:** the CPU references (`trace.ts`, `hot-flow-image.ts`) and every calibration.

**Tech Stack:** TypeScript, WebGPU/WGSL, Vitest, vite-node, Playwright (`verify:gpu`, `bench`).

**Spec:** `docs/specs/2026-10-07-mino-integrator-design.md`

## Global Constraints

- **Science unchanged:**
  - no edit to emission code's physics (disk, jet, flow, hotspot, sky);
  - no edit to any physics constant;
  - no edit to `trace.ts`, `kerr.ts`, `hot-flow*.ts`, `hotspot.ts`, `flux-history.ts`, `synchrotron.ts`, `cyclosynch.ts`,
    `presets.ts`, or to the calibration scripts or their outputs.
- **Accuracy:**
  - no fate flips;
  - per ray and quantity, error_new ≤ max(error_old, floor), with floors 1e-5 relative (r, t, intensity), 1e-5 rad (φ),
    0.01 px (sky);
  - first-integral drift < 1e-4.
- **Go/no-go:** after Tasks 2–3, (steps per ray × cost per step) must be ≤ ½ of today's on the visible rays and on the
  1.3 mm flow rays. Otherwise stop, write the numbers to the ledger and the README, and do not swap the renderer.
- **Termination unchanged:** captured at r ≤ 1.005 r₊, escaped at r > 1.2 r_obs, the 1.3 mm outgoing exit at r > 50 M,
  `maxSteps`, and the classifier on budget exhaustion. The emission sample spacing is unchanged (`JET_DL` 0.25 M; the
  flow's r/8 rule beyond 8 M).
- No Co-Authored-By trailer. When finished: merge to main and push (standing rule); deploy only on request.

## Review Focus

1. **Single precision far from the hole** (r_obs = 1000, R ~ 1e12). Expected: the GPU step matches the CPU twin there
   (Task 3 parity case at r = 1000 and 60).
2. **Near-axis rays** (ξ ≈ 0, sin θ → 0). Expected: p_θ is finite with the right sign, no NaN, and the disk hits of the
   polar-axis set are no worse (Task 2 sweep set N/A/B; Task 3 parity near-axis case).
3. **Near-critical rays** (photon ring). Expected: no fate flips and no ray worse than today (Task 2 sweep critical band).
4. **Cache replay.** Expected: bookmark replay reproduces the live jet sum bit-for-bit (Task 4 `?cachecheck` = 0).
5. **1.3 mm flow intensity and hotspot** on curved samples. Expected: `?hotflow` flux, ring and hotspot twins stay within
   tolerance (Task 4).

---

### Task 1: GPU cost split (measurement only)

**Files:** scratch only (`scripts/scratch-split.mjs`, excluded from git by `.git/info/exclude` `scripts/scratch-*`).
`src/render/raytrace.wgsl` is edited temporarily and restored with `git checkout`.

- [ ] **Step 1:** With the dev server on :5173, run `ONLY_MM=1 npm run bench` and `npm run bench` (full rows). Record the
  live minima for `1280x720` visible and `1280x720 mm` (`fluxVar` 0 row).
- [ ] **Step 2: geometry-only variant.** In `traceRay`, comment out `jet = jetStep(...)` and `flow = flowStep(...)`, and
  make `shadeDisk` / `shadeDiskMm` return `vec3(1.0)` at their first line. Rerun both benches, then
  `git checkout src/render/raytrace.wgsl`. Repeat steps 1 and 2 twice more, interleaved, and record the minima.
- [ ] **Step 3:** Ledger: per view, integrator share = geometry-only / full, and emission share = 1 − that. Include the
  ceiling for an integrator k× cheaper: full / (geometry/k + emission). No commit (nothing kept).

---

### Task 2: CPU twin, unit tests, accuracy sweep and CPU go/no-go

**Files:** Create `src/physics/trace-mino.ts`, `tests/trace-mino.test.ts`, `tests/sweep-mino.test.ts`.

**Interfaces (Produces):**
- `interface MinoRay { a: number; xi: number; eta: number; K: number; omMax: number }`
- `minoRay(a: number, xi: number, eta: number): MinoRay`
- `minoInit(r0: number, th0: number, beta: number, ray: MinoRay): Float64Array`: y = [t, r, u, φ, l, r′, u′]
- `minoRhs(y: Float64Array, ray: MinoRay): Float64Array`
- `minoStep(y: Float64Array, h: number, ray: MinoRay, tol: number, uFrac: number): MinoStepOut`, where
  `MinoStepOut = { y: Float64Array; f0: Float64Array; f1: Float64Array; h: number; hNext: number; attempts: number }`
- `minoDense(y0, y1, f0, f1, h, th: number): Float64Array`
- `minoToState(y: Float64Array, ray: MinoRay): Float64Array`: today's 8-float State
- `minoCrossing(y0, y1, f0, f1, h): number`: θ* in [0, 1] of the u = 0 root, or −1
- `minoTrace(al, be, a, inclDeg, o: { rIn; rOut; rObs; maxSteps?; tol?; uFrac? }): MinoTraceResult`, where
  `MinoTraceResult = { fate: Fate; steps: number; attempts: number; rHit?; phiHit?; tHit?; s: Float64Array; drift: number }`
- constants `MINO_TOL`, `MINO_UFRAC`, `MINO_MAX_REJECT = 10`

- [ ] **Step 1: failing unit tests** `tests/trace-mino.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { minoRay, minoInit, minoRhs, minoStep, minoDense, minoToState, minoCrossing, minoTrace, MINO_TOL, MINO_UFRAC } from "../src/physics/trace-mino";
import { screenToState, screenToXiEta } from "../src/physics/camera";
import { rhs } from "../src/physics/geodesic";
import { metricUpper } from "../src/physics/kerr";
import { traceRay, H_TOL } from "../src/physics/trace";

const R = (y: Float64Array, c: ReturnType<typeof minoRay>) => { const r = y[1], P = r * r + c.a * c.a - c.a * c.xi, D = r * r - 2 * r + c.a * c.a; return P * P - D * c.K; };
const U = (y: Float64Array, c: ReturnType<typeof minoRay>) => { const u = y[2]; return c.eta * (1 - u * u) + c.a * c.a * u * u * (1 - u * u) - c.xi * c.xi * u * u; };
const setup = (al: number, be: number, a: number, iDeg: number) => {
  const i = (iDeg * Math.PI) / 180, [xi, eta] = screenToXiEta(al, be, a, i), c = minoRay(a, xi, eta);
  return { c, y: minoInit(1000, i, be, c), s: screenToState(al, be, a, i, 1000) };
};

describe("Mino-time integrator (spec 2026-10-07)", () => {
  it("initial state satisfies both first integrals and maps to the camera State", () => {
    for (const [al, be, a, i] of [[3, 2, 0.9, 72], [0.1, 6, 0, 8], [-5, -1, 0.998, 85]] as const) {
      const { c, y, s } = setup(al, be, a, i);
      expect(y[5] * y[5] / R(y, c) - 1).toBeLessThan(1e-12); expect(y[5]).toBeLessThan(0);
      expect(Math.abs(y[6] * y[6] - U(y, c))).toBeLessThan(1e-12 * Math.max(1, U(y, c)));
      const s2 = minoToState(y, c);
      for (let k = 0; k < 8; k++) expect(s2[k]).toBeCloseTo(s[k], 8);
    }
  });
  it("RHS equals Sigma x the Hamiltonian velocities (t, r, theta, phi) and R'/2, U'/2", () => {
    for (const [al, be, a, i] of [[3, 2, 0.9, 72], [-4, 5, 0.5, 30]] as const) {
      const { c } = setup(al, be, a, i);
      const y = new Float64Array([-990, 7.3, 0.31, 0.4, 0, 0, 0]);
      y[5] = -Math.sqrt(R(y, c)); y[6] = Math.sqrt(Math.max(0, U(y, c)));
      const f = minoRhs(y, c), s = minoToState(y, c), h = rhs(s, a), Sig = y[1] ** 2 + a * a * y[2] ** 2;
      expect(f[0]).toBeCloseTo(Sig * h[0], 9); expect(f[1]).toBeCloseTo(Sig * h[1], 9);
      expect(f[3]).toBeCloseTo(Sig * h[3], 9); expect(f[4]).toBeCloseTo(Sig, 12);
      expect(f[2]).toBeCloseTo(-Math.sin(s[2]) * Sig * h[2], 9);
      const e = 1e-6, yp = y.slice(), ym = y.slice(); yp[1] += e; ym[1] -= e;
      expect(f[5]).toBeCloseTo((R(yp, c) - R(ym, c)) / (4 * e), 4);
      const up = y.slice(), um = y.slice(); up[2] += e; um[2] -= e;
      expect(f[6]).toBeCloseTo((U(up, c) - U(um, c)) / (4 * e), 6);
    }
  });
  it("a step keeps the first integrals and the null condition (f64)", () => {
    const { c, y } = setup(3, 2, 0.9, 72);
    let s = y, h = 1e-5;
    for (let k = 0; k < 400 && s[1] > 3; k++) { const o = minoStep(s, h, c, MINO_TOL, MINO_UFRAC); s = o.y; h = o.hNext; }
    const P2 = (s[1] ** 2 + 0.81) ** 2;
    expect(Math.abs(s[5] ** 2 - R(s, c)) / P2).toBeLessThan(1e-6);
    const st = minoToState(s, c), g = metricUpper(st[1], st[2], 0.9);
    const H = g.tt + 2 * g.tphi * st[7] + g.rr * st[5] ** 2 + g.thth * st[6] ** 2 + g.phph * st[7] ** 2;
    expect(Math.abs(H) / Math.abs(g.tt)).toBeLessThan(1e-6);
  });
  it("dense output is exact at the ends and a cubic between; the u = 0 root is found", () => {
    const { c, y } = setup(0, -3, 0.5, 80); // u' < 0: heads for the equator before anything else
    let s = y, h = 1e-5, found = false;
    for (let k = 0; k < 4000 && !found && s[1] > 2; k++) {
      const o = minoStep(s, h, c, MINO_TOL, MINO_UFRAC);
      const d0 = minoDense(s, o.y, o.f0, o.f1, o.h, 0), d1 = minoDense(s, o.y, o.f0, o.f1, o.h, 1);
      for (let j = 0; j < 7; j++) { expect(d0[j]).toBe(s[j]); expect(d1[j]).toBeCloseTo(o.y[j], 12); }
      const th = minoCrossing(s, o.y, o.f0, o.f1, o.h);
      if (th >= 0) { found = true; expect(Math.abs(minoDense(s, o.y, o.f0, o.f1, o.h, th)[2])).toBeLessThan(1e-9); }
      s = o.y; h = o.hNext;
    }
    expect(found).toBe(true);
  });
  it("whole rays agree with today's integrator on fate and disk radius (default view)", () => {
    for (const [al, be] of [[3, 2], [-6, 1], [0.5, -7], [10, 10]]) {
      const i = (72 * Math.PI) / 180, o = { rIn: 2.32, rOut: 40, rObs: 1000 };
      const old = traceRay(screenToState(al, be, 0.9, i, 1000), 0.9, { ...o, hTol: H_TOL });
      const nu = minoTrace(al, be, 0.9, 72, o);
      expect(nu.fate).toBe(old.fate);
      if (old.fate === "disk") expect(Math.abs(nu.rHit! - old.rHit!)).toBeLessThan(0.02);
    }
  });
});
```

- [ ] **Step 2:** Run `npx vitest run tests/trace-mino.test.ts`. Expected: FAIL (module missing).
- [ ] **Step 3: implement** `src/physics/trace-mino.ts`:

```ts
// Carter's separated null-geodesic equations in Mino time (spec 2026-10-07 mino integrator). Same Kerr geodesics as
// trace.ts (which stays the reference); M = 1, camera-normalised past-directed p_t = 1, xi = -p_phi, eta = Carter's
// constant. y = [t, r, u = cos(theta), phi, l (affine), r', u'] with ' = d/d(lambda), d(lambda) = dl / Sigma.
// WGSL twin: the Mino block of integrator-shared.wgsl.
import type { Fate } from "./trace";

export const MINO_TOL = 1e-5;     // DP5(4) error-norm tolerance, set by tests/sweep-mino.test.ts (SWEEP=1)
export const MINO_UFRAC = 0.25;   // step <= MINO_UFRAC x the u-oscillation half-period bound (no hidden double crossing)
export const MINO_MAX_REJECT = 10;

export interface MinoRay { a: number; xi: number; eta: number; K: number; omMax: number }
export function minoRay(a: number, xi: number, eta: number): MinoRay {
  // |d(u'')/du| <= (eta + xi^2 - a^2) + 6 a^2: a bound on the u-oscillation's angular frequency
  const om2 = Math.max(0, eta + xi * xi - a * a) + 6 * a * a;
  return { a, xi, eta, K: eta + (xi - a) * (xi - a), omMax: Math.sqrt(om2) };
}
const Rpoly = (r: number, c: MinoRay) => { const P = r * r + c.a * c.a - c.a * c.xi, D = r * r - 2 * r + c.a * c.a; return P * P - D * c.K; };
/** Camera at (r0, th0) with screen beta: r' = -sqrt(R) (inward), u' = sin(th0) beta (= -sin th p_theta, p_theta = -beta). */
export function minoInit(r0: number, th0: number, beta: number, c: MinoRay): Float64Array {
  return new Float64Array([0, r0, Math.cos(th0), 0, 0, -Math.sqrt(Math.max(0, Rpoly(r0, c))), Math.sin(th0) * beta]);
}
export function minoRhs(y: Float64Array, c: MinoRay): Float64Array {
  const r = y[1], u = y[2], a = c.a, xi = c.xi;
  const P = r * r + a * a - a * xi, D = r * r - 2 * r + a * a, s2 = Math.max(1 - u * u, 1e-300);
  return new Float64Array([
    -((r * r + a * a) * P / D + a * (xi - a * s2)),     // t'
    y[5],                                              // r'
    y[6],                                              // u'
    -(a * P / D - a + xi / s2),                        // phi'
    r * r + a * a * u * u,                             // l' = Sigma
    2 * r * P - (r - 1) * c.K,                         // r'' = R'/2
    u * (a * a - c.eta - xi * xi) - 2 * a * a * u * u * u, // u'' = U'/2
  ]);
}
// Dormand-Prince 5(4) (FSAL). A: stage matrix rows 2..7; B: 5th-order weights (= row 7); E = B - B*.
const A = [[], [1 / 5], [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9], [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
  [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656], [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]];
const E = [71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40];
export interface MinoStepOut { y: Float64Array; f0: Float64Array; f1: Float64Array; h: number; hNext: number; attempts: number }
/** Error scale per component: t, phi, l relative to the step's change (floor 1), r relative, r' against max(r^2, |r'|),
 *  u absolute, u' against max(1, |u'|). */
function errNorm(y0: Float64Array, y1: Float64Array, err: Float64Array, tol: number): number {
  const sc = [Math.max(1, Math.abs(y1[0] - y0[0])), Math.max(y0[1], y1[1]), 1, Math.max(1, Math.abs(y1[3] - y0[3])),
    Math.max(1, Math.abs(y1[4] - y0[4])), Math.max(y1[1] * y1[1], Math.abs(y1[5])), Math.max(1, Math.abs(y1[6]))];
  let m = 0; for (let j = 0; j < 7; j++) m = Math.max(m, Math.abs(err[j]) / (tol * sc[j]));
  return m;
}
export function minoStep(y: Float64Array, h0: number, c: MinoRay, tol = MINO_TOL, uFrac = MINO_UFRAC, f0In?: Float64Array): MinoStepOut {
  const hMax = c.omMax > 0 ? (uFrac * Math.PI) / c.omMax : Infinity;
  const f0 = f0In ?? minoRhs(y, c);
  let h = Math.min(h0, hMax);
  for (let att = 1; ; att++) {
    const k: Float64Array[] = [f0];
    for (let s = 1; s < 7; s++) {
      const ys = y.slice();
      for (let j = 0; j < 7; j++) { let acc = 0; for (let q = 0; q < s; q++) acc += A[s][q] * k[q][j]; ys[j] = y[j] + h * acc; }
      k.push(minoRhs(ys, c));
    }
    // row 7 of A is the 5th-order solution, so stage 7's argument IS y1 and k[6] = f(y1) (FSAL)
    const y1 = y.slice(); for (let j = 0; j < 7; j++) { let acc = 0; for (let q = 0; q < 6; q++) acc += A[6][q] * k[q][j]; y1[j] = y[j] + h * acc; }
    const err = new Float64Array(7); for (let j = 0; j < 7; j++) { let acc = 0; for (let q = 0; q < 7; q++) acc += E[q] * k[q][j]; err[j] = h * acc; }
    const en = errNorm(y, y1, err, tol);
    const fac = en > 0 ? 0.9 * Math.pow(en, -0.2) : 5;
    if ((en <= 1 && Number.isFinite(en)) || att > MINO_MAX_REJECT) {
      return { y: y1, f0, f1: k[6], h, hNext: Math.min(hMax, h * Math.min(5, Math.max(0.2, fac))), attempts: att };
    }
    h *= Math.max(0.2, Math.min(0.9, Number.isFinite(fac) ? fac : 0.2));
  }
}
const herm = (th: number) => { const t2 = th * th, t3 = t2 * th; return [2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + th, -2 * t3 + 3 * t2, t3 - t2]; };
/** Cubic Hermite in lambda for every component (values y0, y1, derivatives f0, f1) at fraction th of the step. */
export function minoDense(y0: Float64Array, y1: Float64Array, f0: Float64Array, f1: Float64Array, h: number, th: number): Float64Array {
  if (th === 0) return y0.slice();
  const [h00, h10, h01, h11] = herm(th), o = new Float64Array(7);
  for (let j = 0; j < 7; j++) o[j] = h00 * y0[j] + h10 * h * f0[j] + h01 * y1[j] + h11 * h * f1[j];
  return o;
}
/** Fraction of the step where u = 0 (the equatorial plane), or -1 if u does not change sign across the step. */
export function minoCrossing(y0: Float64Array, y1: Float64Array, f0: Float64Array, f1: Float64Array, h: number): number {
  if (!(y0[2] * y1[2] < 0 || (y1[2] === 0 && y0[2] !== 0))) return -1;
  let lo = 0, hi = 1; const ulo = y0[2];
  for (let k = 0; k < 40; k++) { const m = 0.5 * (lo + hi), um = minoDense(y0, y1, f0, f1, h, m)[2]; if (um * ulo > 0) lo = m; else hi = m; }
  return 0.5 * (lo + hi);
}
/** Today's State (t, r, theta, phi, p_t, p_r, p_theta, p_phi). Near the axis p_theta comes from Theta, never 0/0. */
export function minoToState(y: Float64Array, c: MinoRay): Float64Array {
  const r = y[1], u = Math.max(-1, Math.min(1, y[2])), s2 = 1 - u * u, D = r * r - 2 * r + c.a * c.a;
  let pth: number;
  if (s2 > 1e-8) pth = -y[6] / Math.sqrt(s2);
  else { const Th = c.eta + c.a * c.a * u * u - (c.xi * c.xi * u * u) / Math.max(s2, 1e-12); pth = -Math.sign(y[6]) * Math.sqrt(Math.max(Th, 0)); }
  return new Float64Array([y[0], r, Math.acos(u), y[3], 1, y[5] / D, pth, -c.xi]);
}
export interface MinoTraceResult { fate: Fate; steps: number; attempts: number; rHit?: number; phiHit?: number; tHit?: number; s: Float64Array; drift: number }
/** The render loop's termination order: disk crossing (u = 0 root of the dense output), capture, escape, budget. */
export function minoTrace(al: number, be: number, a: number, inclDeg: number, o: { rIn: number; rOut: number; rObs: number; maxSteps?: number; tol?: number; uFrac?: number }): MinoTraceResult {
  const incl = (inclDeg * Math.PI) / 180, xi = -al * Math.sin(incl), ci = Math.cos(incl), si = Math.sin(incl);
  const c = minoRay(a, xi, be * be + (xi * xi * ci * ci) / Math.max(si * si, 1e-8) - a * a * ci * ci);
  const rh = 1 + Math.sqrt(Math.max(0, 1 - a * a)), maxSteps = o.maxSteps ?? 4800;
  let y = minoInit(o.rObs, incl, be, c), h = 50 / (o.rObs * o.rObs), f: Float64Array | undefined, attempts = 0, drift = 0;
  for (let step = 1; step <= maxSteps; step++) {
    const st = minoStep(y, h, c, o.tol ?? MINO_TOL, o.uFrac ?? MINO_UFRAC, f);
    attempts += st.attempts;
    const P = st.y[1] ** 2 + a * a - a * xi; drift = Math.max(drift, Math.abs(st.y[5] ** 2 - Rpoly(st.y[1], c)) / (P * P));
    const th = minoCrossing(y, st.y, st.f0, st.f1, st.h);
    if (th >= 0) {
      const d = minoDense(y, st.y, st.f0, st.f1, st.h, th);
      if (d[1] >= o.rIn && d[1] <= o.rOut) return { fate: "disk", steps: step, attempts, rHit: d[1], phiHit: d[3], tHit: d[0], s: minoToState(d, c), drift };
    }
    y = st.y; h = st.hNext; f = st.f1;
    if (y[1] <= rh * 1.005) return { fate: "captured", steps: step, attempts, s: minoToState(y, c), drift };
    if (y[1] > o.rObs * 1.2) return { fate: "escaped", steps: step, attempts, s: minoToState(y, c), drift };
  }
  return { fate: "budget", steps: maxSteps, attempts, s: minoToState(y, c), drift };
}
```

  (`MINO_TOL` and `MINO_UFRAC` are provisional; Step 6 sets them.)
- [ ] **Step 4:** Run `npx vitest run tests/trace-mino.test.ts`. Expected: PASS. If a check fails, debug the formula
  (systematic-debugging); never loosen a test tolerance to pass.
- [ ] **Step 5: accuracy sweep** `tests/sweep-mino.test.ts` (`SWEEP=1`; `HOLDOUT=1` for the holdout views). Build it from
  `tests/sweep-farmonitor.test.ts`:
  - **Copied as they are:** `skyDirCPU`, `capLocal` / `stepSizeLocal` / `localTrace` with `REF1` / `REF2`, the convergence
    rule, and the lensing-Jacobian pixel metric (its `scoreSky` logic).
  - **Ray sets:** V1 a=0 i=8; V2 a=0.9 i=72; E a=0.99 i=85 (32²); A, B, N (the axis sets); C, the critical band (for
    a = 0, 0.9 and 0.998 at i = 72: 64 points on the critical curve from `criticalXiEta` / `photonShellRange` in
    `shadow.ts`, each offset by ±0.005, ±0.01, ±0.02 M along the screen radius); mm, Sgr A* (0.94, 30) and Gargantua
    (0.6, 85) at 24² over ±14 M.
  - **OLD** = `localTrace(ray, SHIP)`, today's shipped controller; it must equal `traceRay` (assert as the farmonitor
    sweep does).
  - **NEW** = `minoTrace` with the candidate (tol, uFrac).
  - **Quantities:**
    - fate;
    - disk rHit, φHit (wrapped to (−π, π]) and tHit; extend the local trace and REF to return φHit and tHit by the same
      linear interpolation they use for rHit;
    - sky px;
    - for the mm sets, the flow intensity I: integrate `flowCoeffs` along each method's path the way the renderer does,
      with samples ≤ 0.25 M apart (r/32 beyond 8 M).
      - OLD: on the chord between steps with linearly interpolated momentum (the GPU's `jetSample` / `mix`).
      - NEW: from `minoDense` + `minoToState`, with sub-sample weights Δl from the dense l.
      - REF: per REF1 step with REF's own fine steps;
      
      with the hot-flow `n0`/`rgCm` of Sgr A* and Gargantua from `flowN0` / `jetUniforms`, as `hot-flow-image.ts` does.
  - **Pass** (Global Constraints): count newFlips, worse rays, and maxDrift. Cost = mean `attempts` (NEW) against mean
    steps + retries (OLD), per set.
  - **Print one table row per candidate:** tol ∈ {1e-4, 3e-5, 1e-5, 3e-6, 1e-6} × uFrac ∈ {0.5, 0.25}.
  - **Assert** that the shipped (MINO_TOL, MINO_UFRAC) row passes.
- [ ] **Step 6:** Run `SWEEP=1 npx vitest run tests/sweep-mino.test.ts` (background, 1 h timeout). Pick the loosest
  (tol, uFrac) that passes, and ship tol / 2 (the 2× margin). Write both into `trace-mino.ts` and the table into the
  sweep file's header comment. Then run `SWEEP=1 HOLDOUT=1`: it must pass at the shipped values. If NO candidate passes,
  that is the CPU no-go: ledger it and stop the plan with a report to the user.
- [ ] **Step 7:** Ledger the CPU go/no-go numbers: the cost ratio per set (NEW attempts / OLD steps+retries).
- [ ] **Step 8:** Commit: `git commit -m "Mino integrator: CPU twin (Carter equations in Mino time, DP5(4), Hermite dense output), unit tests, accuracy sweep (tol <value>, uFrac <value>)"`

---

### Task 3: WGSL twin, parity and per-step cost (GPU go/no-go)

**Files:** `src/render/integrator-shared.wgsl` (add the Mino block; the old integrator stays until Task 4); Create
`src/render/mino-parity.wgsl`; Modify `src/test/parity.browser.ts`, `src/main.ts` (parity line), `tests/trace-mino.test.ts`
(constants). Scratch `scripts/scratch-stepcost.mjs`.

**Produces (WGSL):**
- `struct MinoRay { a: f32, xi: f32, eta: f32, K: f32, omMax: f32 }`; `fn minoRay(a, xi, eta) -> MinoRay`
- `struct Mino { q: vec4<f32>, v: vec4<f32> }`: q = (t, r, u, φ), v = (l, r′, u′, 0)
- `fn minoInit(r0, th0, beta, c) -> Mino`; `fn minoRhs(y: Mino, c) -> Mino`
- `struct MinoStepOut { y: Mino, f0: Mino, f1: Mino, h: f32, hNext: f32, attempts: u32 }`; `fn minoStep(y, f0, h0, c) -> MinoStepOut`
- `fn minoDense(y0, y1, f0, f1, h, th) -> Mino`; `fn minoCrossing(y0, y1, f0, f1, h) -> f32`
- `fn minoToState(y: Mino, c) -> State`
- constants `MINO_TOL`, `MINO_UFRAC`, `MINO_MAX_REJECT`

- [ ] **Step 1: failing constants test** (append to `tests/trace-mino.test.ts`): read `integrator-shared.wgsl` and
  assert `MINO_TOL`, `MINO_UFRAC` and `MINO_MAX_REJECT` equal the TS constants. Use the regex helper from
  `tests/hotspot.test.ts`, written with the Edit tool so the template literal's `\\s` survives. Run it. Expected: FAIL.
- [ ] **Step 2: implement** the Mino block in `integrator-shared.wgsl`. It is a line-for-line port of `trace-mino.ts`:
  - same component order, DP coefficients as f32 literals, same `errNorm` scales, same reject loop bounded by
    `MINO_MAX_REJECT`;
  - `minoStep` takes the FSAL `f0` from the caller;
  - `minoCrossing` uses 24 bisection iterations (f32);
  - `minoToState` has the same s² > 1e-8 branch and returns `State(vec4(t, r, acos(u), φ), vec4(1, r′/Δ, pθ, −ξ))`.
- [ ] **Step 3: parity.** `mino-parity.wgsl`: input per case (r0, th0, beta, a, xi, eta, h0, nSteps). The shader inits
  and runs nSteps `minoStep`s (FSAL threaded) and outputs y, hNext, attempts, the converted State and
  `minoDense(…, 0.37)` of the last step. CPU expectation: the same from `trace-mino.ts`. Cases:
  - far field: r0 = 1000, 1 step and 20 steps;
  - a ray stepped to r ≈ 60;
  - near the horizon, a = 0.9 and 0.998;
  - near the axis, ξ = 0.001;
  - at an r turning point (photon-ring ray);
  - a forced reject (h0 = 10× the accepted step).
  
  Tolerances: relative 1e-4 on r and l, absolute 1e-4 on u, φ relative 1e-4 to the step's change, attempts exact.
  Report `minoErr`; `?parity` fails above 1. Mutation `MINO_TOL = 1e-3` must fail (attempts or state differ).
- [ ] **Step 4: per-step cost.** `scripts/scratch-stepcost.mjs` (dev server) builds two compute pipelines from
  `integrator-shared.wgsl`:
  - (i) 400 iterations of today's `stepGeodesic(s, a, stepSize(...), H_TOL)` from 1280×720 camera states (a 0.9, 72°);
  - (ii) 400 iterations of `minoStep`.
  
  Time each over 5 runs (minimum) and output ms per 1e6 steps. Ledger the ratio.
- [ ] **Step 5: go/no-go.** Combine the per-step cost ratio with Task 2's step ratio, per set, and ledger it. If it
  exceeds ½ for both the visible and the mm sets, stop: no renderer swap; README "Investigated" section; report. If only
  one passes, continue: the swap applies to both, since there is one integrator, but report the per-view gains honestly.
- [ ] **Step 6:** `npx vitest run`; dev server; `npm run verify:gpu`. Expected: `?parity` PASS and golden unchanged (the
  renderer does not call the new code yet).
- [ ] **Step 7:** Commit: `git commit -m "Mino integrator: WGSL twin and ?parity cases; per-step cost <ratio>"`

---

### Task 4: Renderer swap

**Files:** `src/render/raytrace.wgsl`, `src/render/integrator-shared.wgsl` (remove the unused old integrator),
`src/render/integrator-parity.wgsl` + its block in `parity.browser.ts` (remove), `src/test/golden.json` (re-record),
`README.md` (`?parity` description).

- [ ] **Step 1: segment type and sampling** (`raytrace.wgsl`):
  - `struct Seg { y0: Mino, y1: Mino, f0: Mino, f1: Mino, h: f32 }`.
  - `fn segAt(g: Seg, th: f32) -> Mino { return minoDense(g.y0, g.y1, g.f0, g.f1, g.h, th); }`
  - `fn segRMin(g: Seg) -> f32`: the minimum of the r-Hermite cubic on [0, 1] (endpoints, plus the real roots of its
    derivative quadratic inside [0, 1]). This exactly replaces `chordMisses(p0, dvec, R)` as `segRMin(g) > R` for the
    jet's bounding sphere, `HF_RMAX` and `HS_REACH`.
  - **Sample k of n** sits at th = k/n: position `minoToState(segAt(g, k/n), ray)`. Its weight is
    `segAt(g, (k+1)/n).v.x − segAt(g, k/n).v.x` (affine Δl; k = 0 uses `g.y0` exactly).
  - The sample's momentum is that same `State`'s p; its time is that `State`'s t.
  - The count stays `n = jetSubCount(dl)`, with dl = y1.l − y0.l. The flow uses its r/8 rule with min r = `segRMin`.
- [ ] **Step 2: emitters.** `jetStep(g: Seg, accIn)`, `flowStep(g: Seg, orb, accIn, hs)` and `jetTouches(g: Seg)` take
  their sample points, weights and momenta from Step 1. Every coefficient line stays the same. `tS` (the emission time)
  is the sample's t.
- [ ] **Step 3: ray loop.**
  - `traceRay` inits `c = minoRay(a, xi, eta)` and `y = minoInit(r0, th0, beta, c)`, with h = 50 / r0² and
    f = `minoRhs(y, c)`.
  - Each iteration: `st = minoStep(y, f, h, c)`; `g = Seg(y, st.y, st.f0, st.f1, st.h)`; then jet, flow and bookmarks
    exactly as now with `g`.
  - **Disk:** `th = minoCrossing(…)`; if th ≥ 0, `d = segAt(g, th)`. rHit = d.q.y and φHit = d.q.w. The delay comes from
    d.q.x, the same expression as now. Then the same rIn/rOut test and shading.
  - **Then:** y = st.y, f = st.f1, h = st.hNext.
  - **Capture / escape / mm exit** as now on y.q.y. The escape direction is `skyDir(minoToState(y, c), a)`.
  - **Budget exhaustion:** the classifier on (ξ, η) as now, with the `usable` guard on `minoToState(y, c)`.
- [ ] **Step 4: cache.** `Bookmark { x: vec4, p: vec4, nJet }` now stores (y.q, y.v with v.w = h at the bookmark, and
  f is recomputed). `replayJet` rebuilds `c` from the pixel's (ξ, η) via `pixelImpact`/`cameraXiEta`, which is why
  `shade` passes the pixel. It then runs nJet `minoStep`s from it with f = `minoRhs(y, c)`, summing `jetStep`. The build
  pass records the bookmark state with its h.
- [ ] **Step 5: remove** `stepGeodesic`, `stepSize`, `angularCap`, `reflectAxis`, `hquadScaled`, `rk4`, `rhs`, `gUpGrad`
  and their constants from `integrator-shared.wgsl`, but only those that `grep -rn` shows unused by `raytrace.wgsl`,
  `emission-shared.wgsl`, the parity shaders and `shadow-shared.wgsl`. Remove `integrator-parity.wgsl` and its block
  (the Mino cases replace it), and update `tests/shader-twins.test.ts` if it lists them. `trace.ts` stays (reference).
- [ ] **Step 6: gates.** `npx vitest run`; `npm run build`; dev server; `npm run verify:gpu`. Expected:
  - `?parity` PASS;
  - `?shadow` 0 mismatches;
  - `?hotflow` PASS (flux ±5 %, ring ±0.5 µas, hotspot twins);
  - `?cachecheck` exactly 0 on every scene;
  - every app check PASS;
  - `?golden` FAILS (expected: every hash moves).
  
  Investigate any other failure with systematic-debugging. Then `RECORD_GOLDEN=1 npm run verify:gpu`: all PASS. Look at
  screenshots of the default view, face-on, edge-on, Sgr A* mm and M87* mm against `main`'s: no visible change beyond
  noise.
- [ ] **Step 7:** Commit: `git commit -m "Renderer: Mino-time integrator (Carter equations, DP5(4), dense-output emission sampling and disk crossing); old Hamiltonian step removed from the shader; golden re-recorded"`

---

### Task 5: Bench, docs, review, ship

- [ ] **Step 1: bench.** Run `npm run bench` and `ONLY_MM=1 npm run bench`, 3× each, interleaved with `main`'s
  `raytrace.wgsl` + `integrator-shared.wgsl` (swap both files in from `git show main:…`, restore with `git checkout`).
  Ledger the minima: visible 720p and 1080p, live and cached, mm, mm+hs.
- [ ] **Step 2: docs.**
  - README: a section "Mino-time integrator" covering the equations and sources (Carter 1968; Mino 2003), the sweep
    table, the cost split, the bench, the removed machinery, and the go/no-go numbers. Also the gates paragraph.
  - ROADMAP entry.
  - Spec status "implemented".
- [ ] **Step 3: final review** by a fresh reviewer on the most capable model, then one fix pass.
- [ ] **Step 4: ship.** Fast-forward main and push. Do not deploy; report.
