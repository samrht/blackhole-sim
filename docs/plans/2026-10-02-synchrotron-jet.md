# Synchrotron Jet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-made jet with synchrotron emission (magnetically-arrested field, M87's
measured acceleration, p = 2.4 electrons), transferred in general relativity along the existing
geodesics, with absolute brightness fixed by energy conservation (ε × Blandford–Znajek power) and
shared physical units with the disk.

**Architecture:** A CPU module `src/physics/synchrotron.ts` holds the physics (coefficients, field,
flow, plasma frequency shift, BZ power, the energy-budget density scale) and computes three new
uniforms. The sole WGSL copy of the per-sample physics lives in `emission-shared.wgsl` (parity-tested
against the CPU twin); `raytrace.wgsl` integrates emission and absorption per jet quadrature sample in
three visible bands, attenuates what lies behind, and converts to the disk's colour units with a
constant band matrix. The panel trades the "Jet" strength slider for a Jet checkbox and a
radiative-efficiency slider.

**Tech Stack:** TypeScript, WebGPU/WGSL, Vite, Vitest, Playwright (headless Chrome gates).

**Spec:** `docs/specs/2026-10-02-synchrotron-jet-design.md`

## Global Constraints

- Gaussian cgs in all jet physics; shader lengths in r_g, converted with `U.rgCm`.
- p = 2.4, γ_min = 10, φ = 50 (Gaussian), κ = 0.05, Γ(z) = max(1, Γ₂₈₀ (|z|/280)^0.58), Γ₂₈₀ default 2,
  ε default 2 × 10⁻³ (slider log 10⁻⁵–10⁻¹), bands 450 / 550 / 650 nm, visible band 400–750 nm.
- Rybicki & Lightman closed forms: emission per unit γ (6.36), absorption per unit energy (6.53,
  C_E = K (m_e c²)^(p−1)); both pitch-angle averaged; below ν_min = 3γ_min²qB/(4πm_ec): j ∝ (ν/ν_min)^(1/3),
  α ∝ (ν/ν_min)^(−5/3).
- Gates green after every task: `npm test`, `npm run build`, `npm run verify:gpu` (dev server :5173;
  any console warning fails), `node scripts/probe-axis.mjs`, `node scripts/probe-scale.mjs`.
- `?golden` is re-recorded once, in Task 5, on purpose (jet scenes change). The `jet-off` scene must stay
  bit-identical; if it does not, apply the 2026-10-01 A/B rule (capture, compare, explain with an
  experiment that restores the old data flow, ledger the ruling).
- `?cachecheck` stays exact (cached = live, ~0).
- No Co-Authored-By trailer on commits in this repo.

## Review Focus

1. **Jet over the disk with absorption in cached mode:** a cached frame must attenuate the cached disk
   colour by the replayed τ exactly as the live trace does. Pinned by `?cachecheck` (jet-on scenes) in Task 5.
2. **X-ray binaries (B ~ 10⁸ G, optical below ν_min, τ ≫ 1):** no NaN/Inf, sensible (dark or source-
   function-limited) jet. Pinned by Task 1's low-frequency-tail test and Task 6's verify-gpu preset loop
   (Cygnus X-1, GRS 1915+105 lit, warning-free).
3. **Slider extremes:** ε 10⁻⁵ and 10⁻¹, mass 1 and 10¹⁰, accretion 10⁻¹⁰: k_scale finite and positive.
   Pinned by Task 2's extremes test.
4. **Jet off is exactly the old jet-off path** (`color·e⁰ + 0`). Pinned by `?golden` jet-off in Task 5.
5. **Plasma moving away from the camera (counter-jet) or D → 0:** no division blow-up. Pinned by Task 2's
   `plasmaShift` test (receding flow) and the shader's `D > 0` guard in Task 4/5.

---

## File Structure

- Create `src/physics/synchrotron.ts` (+ `tests/synchrotron.test.ts`): constants, coefficients,
  field/flow/shift, BZ power, energy budget, `jetUniforms`, `jetBandMatrix`.
- Modify `src/physics/color.ts` (export the CIE functions), `src/physics/lookups.ts` (export `VIS_LREF`).
- Modify `src/physics/jet.ts` (+ test): add `jetShape`; remove `jetEmission`, `dopplerBoost`.
- Modify `src/render/emission-shared.wgsl`: synchrotron constants and functions (sole copy).
- Modify `src/render/jet-parity.wgsl`, `src/test/parity.browser.ts`: new jet parity cases.
- Modify `src/render/uniforms.ts` (+ test), `src/render/raytrace.wgsl`: 144-byte uniforms, transfer.
- Modify `src/test/scenes.ts`, `src/test/shadow.browser.ts`, `scripts/bench.mjs`, `src/main.ts`,
  `index.html`, `src/physics/presets.ts` (+ test), `scripts/probe-axis.mjs`, `scripts/verify-gpu.mjs`.
- Docs: `README.md`, `docs/ROADMAP.md`, spec status.

---

### Task 1: Synchrotron coefficients (CPU)

**Files:**
- Create: `src/physics/synchrotron.ts` (coefficient part), `tests/synchrotron.test.ts`

**Interfaces:**
- Produces: `Q_E, M_E, C_CGS, G_CGS, MSUN_G, SYN_P, GAMMA_MIN: number`; `lnGamma(x): number`;
  `avgSinPow(x): number`; `synchConsts(p?, gmin?): { lnCj: number; lnCa: number; lnNuMin0: number }`;
  `synchCoeffs(K: number, B: number, nu: number): [j: number, alpha: number]` (plasma frame, cgs, j per sr).

- [ ] **Step 1: Failing tests** — `tests/synchrotron.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { synchConsts, synchCoeffs, lnGamma, Q_E, M_E, C_CGS, SYN_P, GAMMA_MIN } from "../src/physics/synchrotron";

// Direct numerical integration of the single-electron spectrum F(x) = x Int_x^inf K_5/3 over the power
// law and an isotropic pitch-angle distribution (Rybicki & Lightman 6.18, 6.50).
function K53(t: number) { let s = 0; const N = 4000, umax = Math.log(200 / t + 2) + 2, du = umax / N;
  for (let i = 0; i <= N; i++) { const u = i * du, w = i === 0 || i === N ? 1 : i % 2 ? 4 : 2; s += w * Math.exp(-t * Math.cosh(u)) * Math.cosh((5 * u) / 3); }
  return (s * du) / 3; }
const FX: { x: number[]; F: number[] } = (() => { const n = 500, lx0 = -8, lx1 = Math.log(60), ts: number[] = [], ks: number[] = [];
  for (let i = 0; i <= n; i++) { ts.push(Math.exp(lx0 + ((lx1 - lx0) * i) / n)); ks.push(K53(ts[i])); }
  const cum = new Array(ts.length).fill(0); for (let i = ts.length - 2; i >= 0; i--) cum[i] = cum[i + 1] + 0.5 * (ks[i] + ks[i + 1]) * (ts[i + 1] - ts[i]);
  return { x: ts, F: ts.map((t, i) => t * cum[i]) }; })();
function F(x: number) { const { x: xs, F: Fs } = FX; if (x <= xs[0]) return 2.15 * Math.pow(x, 1 / 3); if (x >= xs[xs.length - 1]) return 0;
  let lo = 0, hi = xs.length - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] > x) hi = m; else lo = m; }
  const f = (Math.log(x) - Math.log(xs[lo])) / (Math.log(xs[hi]) - Math.log(xs[lo])); return Fs[lo] + (Fs[hi] - Fs[lo]) * f; }
function numeric(K: number, B: number, nu: number): [number, number] {
  const p = SYN_P; let J = 0, A = 0, W = 0; const NA = 60, NG = 600, lg0 = Math.log(GAMMA_MIN), lg1 = Math.log(1e8), wB = (Q_E * B) / (M_E * C_CGS);
  for (let ia = 0; ia < NA; ia++) { const s = Math.sin(((ia + 0.5) / NA) * (Math.PI / 2)); let Pw = 0, Ia = 0;
    for (let ig = 0; ig < NG; ig++) { const lg = lg0 + ((lg1 - lg0) * (ig + 0.5)) / NG, g = Math.exp(lg), dg = (g * (lg1 - lg0)) / NG;
      const x = (2 * Math.PI * nu) / (1.5 * g * g * wB * s), P1 = ((Math.sqrt(3) * Q_E ** 3 * B * s) / (2 * Math.PI * M_E * C_CGS ** 2)) * F(x);
      Pw += K * Math.pow(g, -p) * P1 * dg; Ia += 2 * Math.PI * P1 * K * Math.pow(g, -p - 1) / (M_E * C_CGS ** 2) * dg; }
    J += (Pw / 2) * s; A += (((p + 2) * C_CGS ** 2) / (8 * Math.PI * nu * nu)) * Ia * s; W += s; }
  return [J / W, A / W];
}

describe("synchrotron coefficients (Rybicki & Lightman 6.36 / 6.53, pitch-averaged)", () => {
  it("lnGamma matches known values", () => {
    expect(lnGamma(5)).toBeCloseTo(Math.log(24), 10);
    expect(lnGamma(0.5)).toBeCloseTo(0.5 * Math.log(Math.PI), 10);
  });
  it("closed forms agree with direct numerical integration within 0.5 %", () => {
    const K = 1e6, B = 10, nu = 5.45e14;
    const [j, a] = synchCoeffs(K, B, nu), [jn, an] = numeric(K, B, nu);
    expect(Math.abs(j / jn - 1)).toBeLessThan(5e-3);
    expect(Math.abs(a / an - 1)).toBeLessThan(5e-3);
  }, 120000);
  it("scales as K B^((p+1)/2) nu^(-(p-1)/2) and K B^((p+2)/2) nu^(-(p+4)/2)", () => {
    const [j1, a1] = synchCoeffs(1e6, 10, 5e14), [j2, a2] = synchCoeffs(2e6, 20, 1e15), p = SYN_P;
    expect(j2 / j1).toBeCloseTo(2 * 2 ** ((p + 1) / 2) * 2 ** (-(p - 1) / 2), 9);
    expect(a2 / a1).toBeCloseTo(2 * 2 ** ((p + 2) / 2) * 2 ** (-(p + 4) / 2), 9);
  });
  it("below nu_min the low-frequency tail is continuous (X-ray binaries: B ~ 1e8 G)", () => {
    const B = 1e8, numin = Math.exp(synchConsts().lnNuMin0) * B;
    const [jl, al] = synchCoeffs(1e20, B, numin * (1 - 1e-9)), [jh, ah] = synchCoeffs(1e20, B, numin * (1 + 1e-9));
    expect(jl / jh).toBeCloseTo(1, 6); expect(al / ah).toBeCloseTo(1, 6);
    const [ja] = synchCoeffs(1e20, B, numin / 8), [jb] = synchCoeffs(1e20, B, numin / 64);
    expect(ja / jb).toBeCloseTo(2, 6); // nu^(1/3)
    for (const v of synchCoeffs(1e20, B, 5e14)) expect(Number.isFinite(v)).toBe(true);
  });
});
```

- [ ] **Step 2: Run** — `npx vitest run tests/synchrotron.test.ts` → FAIL (module missing).

- [ ] **Step 3: Implement** — `src/physics/synchrotron.ts` (first part):

```ts
// Synchrotron jet physics (spec docs/specs/2026-10-02-synchrotron-jet-design.md). CPU twin of the jet
// code in src/render/emission-shared.wgsl, plus what the shader cannot do (the energy-budget density
// scale). Gaussian cgs; lengths in r_g where noted.

export const Q_E = 4.80320471e-10, M_E = 9.1093837e-28, C_CGS = 2.99792458e10, G_CGS = 6.6743e-8, MSUN_G = 1.98847e33;
/** Electron power law N(gamma) = K gamma^-p for gamma >= GAMMA_MIN (spec 2.3). */
export const SYN_P = 2.4, GAMMA_MIN = 10;

export function lnGamma(x: number): number {
  const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lnGamma(1 - x);
  x -= 1; let s = c[0]; const t = x + g + 0.5;
  for (let i = 1; i < 9; i++) s += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(s);
}
/** <sin^x alpha> over an isotropic pitch-angle distribution. */
export const avgSinPow = (x: number) => (Math.sqrt(Math.PI) / 2) * Math.exp(lnGamma((x + 2) / 2) - lnGamma((x + 3) / 2));

/** ln of Cj, Ca in j = Cj K B^((p+1)/2) nu^(-(p-1)/2) (per sr, RL 6.36 with j = P_omega / 2) and
 *  alpha = Ca K B^((p+2)/2) nu^(-(p+4)/2) (RL 6.53, whose C is per unit ENERGY: C_E = K (m c^2)^(p-1)),
 *  both averaged over pitch angle; and ln of nu_min / B = 3 gmin^2 q / (4 pi m c). */
export function synchConsts(p = SYN_P, gmin = GAMMA_MIN) {
  const lnCj = 0.5 * Math.log(3) + 3 * Math.log(Q_E) - Math.log(4 * Math.PI * M_E * C_CGS ** 2 * (p + 1))
    + lnGamma(p / 4 + 19 / 12) + lnGamma(p / 4 - 1 / 12) + Math.log(avgSinPow((p + 1) / 2))
    - ((p - 1) / 2) * Math.log((2 * Math.PI * M_E * C_CGS) / (3 * Q_E));
  const lnCa = 0.5 * Math.log(3) + 3 * Math.log(Q_E) - Math.log(8 * Math.PI * M_E)
    + (p / 2) * Math.log((3 * Q_E) / (2 * Math.PI * M_E ** 3 * C_CGS ** 5))
    + lnGamma((3 * p + 2) / 12) + lnGamma((3 * p + 22) / 12) + Math.log(avgSinPow((p + 2) / 2))
    + (p - 1) * Math.log(M_E * C_CGS ** 2);
  const lnNuMin0 = Math.log((3 * gmin * gmin * Q_E) / (4 * Math.PI * M_E * C_CGS));
  return { lnCj, lnCa, lnNuMin0 };
}
const SC = synchConsts();
/** Plasma-frame emissivity j (erg s^-1 cm^-3 Hz^-1 sr^-1) and absorption alpha (cm^-1); below nu_min
 *  the low-frequency forms matched at nu_min (spec 2.3). Computed in logs (B up to ~1e9 G). */
export function synchCoeffs(K: number, B: number, nu: number): [number, number] {
  const lnB = Math.log(B), lnNu = Math.log(nu), lnNuMin = SC.lnNuMin0 + lnB, le = Math.max(lnNu, lnNuMin);
  let lj = SC.lnCj + Math.log(K) + ((SYN_P + 1) / 2) * lnB - ((SYN_P - 1) / 2) * le;
  let la = SC.lnCa + Math.log(K) + ((SYN_P + 2) / 2) * lnB - ((SYN_P + 4) / 2) * le;
  if (lnNu < lnNuMin) { lj += (lnNu - lnNuMin) / 3; la -= (5 / 3) * (lnNu - lnNuMin); }
  return [Math.exp(lj), Math.exp(la)];
}
```
(Measured while planning: closed form / numerical = 0.99952 for j and 0.99930 for α.)

- [ ] **Step 4: Run** — `npx vitest run tests/synchrotron.test.ts` → PASS (4). `npm test`, `npm run build` → green.

- [ ] **Step 5: Commit** — `git add src/physics/synchrotron.ts tests/synchrotron.test.ts && git commit -m "Add synchrotron emission and absorption coefficients, validated by direct integration"`

---

### Task 2: Jet model and energy budget (CPU)

**Files:**
- Modify: `src/physics/synchrotron.ts`, `tests/synchrotron.test.ts`

**Interfaces:**
- Consumes: Task 1; `metricUpper(r, th, a)`, `horizonOuter(a)` (kerr.ts); `mdotFromLambda(mSun, a, lambda)` (units.ts, kg/s);
  `funnelEdge(z)`, `wallProfile(rho, z)`, `lengthFalloff(z, zMax)`, `JET` (jet.ts).
- Produces:
  - `PHI_MAD = 50, KAPPA_BZ = 0.05, GAMMA_REF_Z = 280, GAMMA_SLOPE = 0.58, VIS_NU_LO, VIS_NU_HI`
  - `jetEnergetics(mSun, a, lambda): { rgCm; mdot (g/s); phi (G cm²); pBZ (erg/s); b0 (G) }`
  - `jetField(rho, z, a, b0): number` (G; rho, z in r_g)
  - `gammaProfile(z, g280): number`
  - `streamlineDir(r, th): [nR, nTh]`
  - `plasmaShift(s: Float64Array, a, gamma, nR, nTh): number` (ν_plasma/ν_obs; s = [t,r,θ,φ,p_t,p_r,p_θ,p_φ], camera-normalised p_t = 1)
  - `visLuminanceUnit(a, b0, rgCm, jetLength): number` (erg/s at k_scale = 1)
  - `jetUniforms(mSun, a, lambda, eps, jetLength): { jetB0; jetKScale; rgCm; pBZ }`
  - `slabStep(I: number, tau: number, j: number, alpha: number, ds: number): [I, tau]` (one band; CPU twin of `jetSlabJ`)
  - in `jet.ts` (not synchrotron.ts, to keep the import one-way synchrotron → jet): `GAMMA_REF_Z = 280`,
    `GAMMA_SLOPE = 0.58`, `gammaProfile(z, g280)`; synchrotron.ts re-exports them.

- [ ] **Step 1: Failing tests** — append to `tests/synchrotron.test.ts`:

```ts
import { jetEnergetics, gammaProfile, plasmaShift, streamlineDir, jetUniforms, visLuminanceUnit, slabStep } from "../src/physics/synchrotron";
import { screenToState } from "../src/physics/camera";
import { lambdaFromMdot } from "../src/physics/units";

describe("jet model and energy budget", () => {
  it("M87*: Blandford-Znajek power ~0.98 Mdot c^2 and inside 1e43-1e44 erg/s (spec 2.1)", () => {
    const E = jetEnergetics(6.5e9, 0.9, lambdaFromMdot(6.5e9, 0.9, 7.7e-4));
    expect(E.pBZ).toBeGreaterThan(1e43); expect(E.pBZ).toBeLessThan(1e44);       // measured 4.26e43
    expect(E.pBZ / (E.mdot * C_CGS ** 2)).toBeCloseTo(0.977, 2);
  });
  it("Gamma(z) = max(1, G280 (|z|/280)^0.58)", () => {
    expect(gammaProfile(280, 2)).toBe(2); expect(gammaProfile(-280, 2)).toBe(2);
    expect(gammaProfile(60, 2)).toBe(1);                                          // 0.82 -> clamped
    expect(gammaProfile(1000, 2)).toBeCloseTo(2 * Math.pow(1000 / 280, 0.58), 12);
  });
  it("plasma frequency shift: static at r = 1000 ~1 (gravitational), toward/away Doppler (Review Focus 5)", () => {
    const s = screenToState(0, 0, 0, Math.PI / 2, 1000), b = Math.sqrt(1 - 1 / 9);
    expect(plasmaShift(s, 0, 1, 1, 0)).toBeCloseTo(1 / Math.sqrt(1 - 2 / 1000), 4);
    expect(plasmaShift(s, 0, 3, 1, 0) / (3 * (1 - b))).toBeCloseTo(1, 2);       // toward camera: blueshift
    expect(plasmaShift(s, 0, 3, -1, 0) / (3 * (1 + b))).toBeCloseTo(1, 2);      // counter-jet: redshift, finite
  });
  it("streamlines are unit vectors, mirror-symmetric between lobes, more collimated than radial", () => {
    const [ur, ut] = streamlineDir(20, 0.2), [lr, lt] = streamlineDir(20, Math.PI - 0.2);
    expect(Math.hypot(ur, ut)).toBeCloseTo(1, 12);
    expect(lr).toBeCloseTo(ur, 12); expect(lt).toBeCloseTo(-ut, 12);
    expect(ut).toBeLessThan(0); // upper lobe bends toward the axis relative to radial
  });
  it("energy budget: visible luminance at the chosen k_scale equals eps * P_BZ", () => {
    for (const [m, a, l] of [[6.5e9, 0.9, lambdaFromMdot(6.5e9, 0.9, 7.7e-4)], [1e8, 0.9, 3.66e-4], [21.2, 0.998, 0.02]]) {
      const U = jetUniforms(m, a, l, 2e-3, 60);
      expect((visLuminanceUnit(a, U.jetB0, U.rgCm, 60) * U.jetKScale) / (2e-3 * U.pBZ)).toBeCloseTo(1, 9);
      expect(jetUniforms(m, a, l, 4e-3, 60).jetKScale / U.jetKScale).toBeCloseTo(2, 9);
    }
  });
  it("slab transfer: thin -> j ds, thick -> source function j/alpha, earlier light attenuated (spec 5)", () => {
    const [It] = slabStep(0, 0, 2, 1e-9, 3);
    expect(It).toBeCloseTo(6, 6);                                                 // optically thin
    const [Ik, tk] = slabStep(0, 0, 2, 1e3, 3);
    expect(Ik).toBeCloseTo(2 / 1e3, 12); expect(tk).toBe(3e3);                    // saturates at S = j/alpha
    let I = 0, tau = 0;
    for (let k = 0; k < 400; k++) [I, tau] = slabStep(I, tau, 2, 0.5, 0.05);      // 400 thin pieces = one slab
    expect(I).toBeCloseTo((2 / 0.5) * (1 - Math.exp(-10)), 6);                    // exact uniform-slab result
    const [Ib] = slabStep(5, 1, 2, 1, 1);                                         // gas behind tau = 1 dims by e^-1
    expect(Ib).toBeCloseTo(5 + 2 * (1 - Math.exp(-1)) * Math.exp(-1), 12);
  });
  it("extremes stay finite and positive (Review Focus 3)", () => {
    for (const [m, l, e] of [[1, 1e-10, 1e-5], [1e10, 1e-10, 1e-1], [1, 1, 1e-1], [1e10, 1, 1e-5]]) {
      const U = jetUniforms(m, 0.9, l, e, 60);
      for (const v of [U.jetB0, U.jetKScale, U.rgCm]) { expect(Number.isFinite(v)).toBe(true); expect(v).toBeGreaterThan(0); }
    }
  });
});
```
Run → FAIL (exports missing).

- [ ] **Step 2: Implement** — in `src/physics/jet.ts` add (after `lengthFalloff`):

```ts
/** Gamma(z) = max(1, G280 (|z| / 280 r_g)^0.58): M87's measured acceleration (Mertens et al. 2016,
 *  Park et al. 2019). WGSL twin: jetGammaAt in emission-shared.wgsl. */
export const GAMMA_REF_Z = 280, GAMMA_SLOPE = 0.58;
export const gammaProfile = (z: number, g280: number) => Math.max(1, g280 * Math.pow(Math.abs(z) / GAMMA_REF_Z, GAMMA_SLOPE));
```

Append to `src/physics/synchrotron.ts` (and add imports at the top:
`import { metricUpper, horizonOuter } from "./kerr"; import { mdotFromLambda } from "./units"; import { funnelEdge, wallProfile, lengthFalloff, JET } from "./jet";`
plus `export { gammaProfile, GAMMA_REF_Z, GAMMA_SLOPE } from "./jet";`):

```ts
/** Magnetically arrested horizon flux (Gaussian units, Tchekhovskoy et al. 2011) and BZ coefficient. */
export const PHI_MAD = 50, KAPPA_BZ = 0.05;
/** Visible band for the energy budget (400-750 nm). */
export const VIS_NU_LO = C_CGS / 750e-7, VIS_NU_HI = C_CGS / 400e-7;

/** r_g (cm), Mdot (g/s), horizon flux Phi = phi sqrt(Mdot c) r_g (G cm^2), BZ power (erg/s), b0 = Phi/(pi r_g^2) (G). */
export function jetEnergetics(mSun: number, a: number, lambda: number) {
  const rgCm = (G_CGS * mSun * MSUN_G) / C_CGS ** 2;
  const mdot = mdotFromLambda(mSun, a, lambda) * 1e3;
  const phi = PHI_MAD * Math.sqrt(mdot * C_CGS) * rgCm;
  const omegaH = (a * C_CGS) / (2 * horizonOuter(a) * rgCm);
  const pBZ = (KAPPA_BZ / (4 * Math.PI * C_CGS)) * phi * phi * omegaH * omegaH;
  return { rgCm, mdot, phi, pBZ, b0: phi / (Math.PI * rgCm * rgCm) };
}
/** |B| (G) at cylindrical (rho, z) in r_g: poloidal flux conservation in the funnel, B_p = b0 / rho_f^2, plus
 *  the force-free toroidal field B_phi = B_p rho Omega_F / c with Omega_F = Omega_H / 2 = a / (4 r_H). */
export function jetField(rho: number, z: number, a: number, b0: number): number {
  const rf = funnelEdge(z), bp = b0 / (rf * rf), w = (rho * a) / (4 * horizonOuter(a));
  return bp * Math.sqrt(1 + w * w);
}
/** Unit flow direction (n_r, n_theta) at BL (r, theta): outward along the self-similar funnel family
 *  rho = q rho_f(z) through the point. */
export function streamlineDir(r: number, th: number): [number, number] {
  const z = r * Math.cos(th), rho = r * Math.sin(th), az = Math.max(Math.abs(z), 1e-6);
  const vr = (rho / funnelEdge(z)) * (JET.slope / (2 * Math.sqrt(az))), vz = z >= 0 ? 1 : -1;
  const n = Math.hypot(vr, vz), ur = vr / n, uz = vz / n;
  return [ur * Math.sin(th) + uz * Math.cos(th), ur * Math.cos(th) - uz * Math.sin(th)];
}
/** nu_plasma / nu_observed = p.u for the camera-normalised, past-directed photon state s, with the plasma
 *  moving at Lorentz factor gamma along (nR, nTh) in the zero-angular-momentum observer's frame. */
export function plasmaShift(s: Float64Array, a: number, gamma: number, nR: number, nTh: number): number {
  const g = metricUpper(s[1], s[2], a);
  const alpha = Math.sqrt(-1 / g.tt), omega = g.tphi / g.tt, beta = Math.sqrt(Math.max(0, 1 - 1 / (gamma * gamma)));
  return gamma * ((s[4] + omega * s[7]) / alpha + beta * (nR * Math.sqrt(g.rr) * s[5] + nTh * Math.sqrt(g.thth) * s[6]));
}
/** Comoving, optically thin visible-band luminosity (erg/s) of the rendered jet (both lobes,
 *  zBase <= |z| <= jetLength, rho <= 1.2 rho_f) at k_scale = 1, i.e. K = B^2 * wall * falloff. Knots and
 *  turbulence are mean-one modulations (1 + amp (noise - 1/2) 2), so this is the time-averaged budget. */
export function visLuminanceUnit(a: number, b0: number, rgCm: number, jetLength: number): number {
  const NZ = 120, NR = 48, NNU = 16, l0 = Math.log(VIS_NU_LO), l1 = Math.log(VIS_NU_HI), zb = JET.zBase;
  let L = 0;
  for (let iz = 0; iz < NZ; iz++) {
    const dz = (jetLength - zb) / NZ, z = zb + dz * (iz + 0.5), rf = funnelEdge(z), dr = (1.2 * rf) / NR;
    for (let ir = 0; ir < NR; ir++) {
      const rho = dr * (ir + 0.5), shape = wallProfile(rho, z) * lengthFalloff(z, jetLength);
      if (shape <= 0) continue;
      const B = jetField(rho, z, a, b0), K = B * B * shape;
      let jv = 0;
      for (let k = 0; k < NNU; k++) { const nu = Math.exp(l0 + ((l1 - l0) * (k + 0.5)) / NNU); jv += synchCoeffs(K, B, nu)[0] * nu * ((l1 - l0) / NNU); }
      L += 2 * (2 * Math.PI * rho * dr * dz) * rgCm ** 3 * 4 * Math.PI * jv;
    }
  }
  return L;
}
/** The jet's three physics uniforms (spec 2.3): k_scale makes the visible luminance equal eps * P_BZ. */
export function jetUniforms(mSun: number, a: number, lambda: number, eps: number, jetLength: number) {
  const E = jetEnergetics(mSun, a, lambda);
  return { jetB0: E.b0, jetKScale: (eps * E.pBZ) / visLuminanceUnit(a, E.b0, E.rgCm, jetLength), rgCm: E.rgCm, pBZ: E.pBZ };
}
/** One band of one jet sample: exact solution across a uniform slab of path ds (cm) behind optical depth
 *  tau already accumulated from the camera. Twin of jetSlabJ in emission-shared.wgsl. */
export function slabStep(I: number, tau: number, j: number, alpha: number, ds: number): [number, number] {
  const dt = alpha * ds, fac = dt < 1e-4 ? 1 - 0.5 * dt : (1 - Math.exp(-dt)) / dt;
  return [I + j * ds * fac * Math.exp(-tau), Math.min(tau + dt, 1e30)];
}
```
(Measured while planning: M87\* P_BZ 4.260e43 erg/s at 0.977 Ṁc²; D static 1.00100, toward Γ = 3
0.17174 vs 0.17157 flat, away 5.834 vs 5.828; k_scale M87\* 0.1465, default 0.01098, Cyg X-1 0.880;
~20 ms per `jetUniforms`.)

- [ ] **Step 3: Run** — the file's tests PASS; `npm test`, `npm run build` green.

- [ ] **Step 4: Commit** — `git add src/physics/synchrotron.ts src/physics/jet.ts tests/synchrotron.test.ts && git commit -m "Add the jet field, flow, plasma shift, BZ power, slab transfer and energy-budget scale"`

---

### Task 3: Jet colour in the disk's units

**Files:**
- Modify: `src/physics/color.ts` (export `cieX`, `cieY`, `cieZ`), `src/physics/lookups.ts` (export `VIS_LREF`),
  `src/physics/synchrotron.ts`, `tests/synchrotron.test.ts`

**Interfaces:**
- Produces: `JET_BANDS_NM = [450, 550, 650]`; `jetBandMatrix(): number[][]` (3×3, row = R,G,B, column = band;
  linear sRGB in `blackbodyVisibleRGB / VIS_LREF` units per unit I_ν (cgs) in each band).

- [ ] **Step 1: Failing test** — append:

```ts
import { jetBandMatrix, JET_BANDS_NM } from "../src/physics/synchrotron";
import { blackbodyVisibleRGB, relLuminance } from "../src/physics/color";
import { VIS_LREF } from "../src/physics/lookups";

describe("jet colour in the disk's units (spec 2.4/2.5)", () => {
  it("a blackbody fed through the three bands has the disk path's luminance within 1 %", () => {
    const M = jetBandMatrix(), h = 6.62607015e-27, kB = 1.380649e-16;
    const Bnu = (T: number, nu: number) => (2 * h * nu ** 3) / C_CGS ** 2 / (Math.exp((h * nu) / (kB * T)) - 1);
    for (const T of [6000, 3e4, 1e7]) {
      const I = JET_BANDS_NM.map((nm) => Bnu(T, C_CGS / (nm * 1e-7)));
      const rgb = [0, 1, 2].map((r) => M[r][0] * I[0] + M[r][1] * I[1] + M[r][2] * I[2]) as [number, number, number];
      const ref = blackbodyVisibleRGB(T).map((v) => v / VIS_LREF) as [number, number, number];
      expect(Math.abs(relLuminance(rgb) / relLuminance(ref) - 1)).toBeLessThan(0.01); // measured 0.5-0.6 %
    }
  });
});
```
Run → FAIL.

- [ ] **Step 2: Implement** — in `color.ts` change `function cieX/cieY/cieZ` to `export function …`. In
`lookups.ts` rename the private `const LREF` to `export const VIS_LREF` (update its two uses). Append to
`synchrotron.ts` (import `cieX, cieY, cieZ` from `./color` and `VIS_LREF` from `./lookups`):

```ts
/** Colour bands (nm): 450, 550, 650; I_nu is taken piecewise constant over 360-500 / 500-600 / 600-830 nm. */
export const JET_BANDS_NM = [450, 550, 650] as const;
/** 3x3 matrix (row R,G,B; column band) from I_nu (cgs) per band to linear sRGB in the disk's units
 *  (blackbodyVisibleRGB / VIS_LREF). blackbodyVisibleRGB sums planck_m(lambda) = lambda_m^-5 / (e^x - 1) over
 *  5 nm bins; the cgs B_lambda = 2 h c^2 1e-10 planck_m, so an I_lambda = I_nu c / lambda^2 maps the same way. */
export function jetBandMatrix(): number[][] {
  const H_CGS = 6.62607015e-27, xyz = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let nm = 360; nm <= 830; nm += 5) {
    const b = nm < 500 ? 0 : nm < 600 ? 1 : 2, lcm = nm * 1e-7, f = C_CGS / (lcm * lcm) / (2 * H_CGS * C_CGS ** 2 * 1e-10);
    xyz[0][b] += cieX(nm) * f; xyz[1][b] += cieY(nm) * f; xyz[2][b] += cieZ(nm) * f;
  }
  const T = [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]];
  return [0, 1, 2].map((r) => [0, 1, 2].map((b) => (T[r][0] * xyz[0][b] + T[r][1] * xyz[1][b] + T[r][2] * xyz[2][b]) / VIS_LREF));
}
```
(Measured while planning: rows ≈ [[−57.86, 563.1, 4364], [−199.6, 5602, −151.7], [8198, −395.0, −69.56]].)

- [ ] **Step 3: Run / Step 4: Commit** — tests PASS, suite + build green;
`git add src/physics/color.ts src/physics/lookups.ts src/physics/synchrotron.ts tests/synchrotron.test.ts && git commit -m "Map the jet's three bands to the disk's colour units"`

---

### Task 4: Shared WGSL synchrotron + parity

**Files:**
- Modify: `src/render/emission-shared.wgsl`, `src/physics/jet.ts`, `tests/jet.test.ts`,
  `src/render/jet-parity.wgsl`, `src/test/parity.browser.ts`, `tests/synchrotron.test.ts`

**Interfaces:**
- Consumes: Tasks 1–3; WGSL `State`, `gUp`, `PI` (integrator-shared, prepended earlier).
- Produces (WGSL, sole copies):
  - consts `SYN_P, SYN_LNCJ, SYN_LNCA, SYN_LNNUMIN0, JET_LNNU: vec3<f32>, JET_BAND_M: mat3x3<f32>, GAMMA_REF_Z, GAMMA_SLOPE`
  - `jetGammaAt(z: f32, g280: f32) -> f32`, `streamlineDirJ(r: f32, th: f32) -> vec2<f32>`
  - `plasmaShiftJ(r: f32, th: f32, p: vec4<f32>, a: f32, gamma: f32) -> f32`
  - `jetShapeJ(r: f32, th: f32, t: f32, jetLength: f32, knotAmp: f32, g280: f32) -> f32` (0 outside the jet)
  - `struct SynchOut { j: vec3<f32>, a: vec3<f32> }`; `synchSampleJ(r, th, D, a, b0, kScale, shape) -> SynchOut`
    (j = (ν/ν′)³ j′ per band, cgs; a = α′ per band, cm⁻¹)
- Produces (TS): `jetShape(r, th, t, jetLength, knotAmp, g280): number` in jet.ts; `jetEmission`, `dopplerBoost` removed.

- [ ] **Step 1: WGSL-constant twin test (failing)** — append to `tests/synchrotron.test.ts`:

```ts
import { readFileSync } from "node:fs";
import { join } from "node:path";
const WGSL = readFileSync(join(__dirname, "../src/render/emission-shared.wgsl"), "utf8");
const constOf = (name: string) => { const m = WGSL.match(new RegExp(`const ${name}\\s*=\\s*([^;]+);`)); if (!m) throw new Error(`no ${name}`); return m[1]; };
// constructor arguments only: the text after the first "(" (skips the digits in vec3<f32> / mat3x3<f32>)
const args = (s: string) => s.slice(s.indexOf("(") + 1).split(",").map((v) => +v.replace(")", ""));
describe("emission-shared.wgsl synchrotron constants match the CPU twin", () => {
  it("coefficients, band frequencies, band matrix", () => {
    const sc = synchConsts();
    expect(+constOf("SYN_P")).toBe(SYN_P);
    expect(+constOf("SYN_LNCJ")).toBeCloseTo(sc.lnCj, 6);
    expect(+constOf("SYN_LNCA")).toBeCloseTo(sc.lnCa, 6);
    expect(+constOf("SYN_LNNUMIN0")).toBeCloseTo(sc.lnNuMin0, 6);
    const nu = args(constOf("JET_LNNU"));
    expect(nu.length).toBe(3);
    nu.forEach((v, b) => expect(v).toBeCloseTo(Math.log(C_CGS / (JET_BANDS_NM[b] * 1e-7)), 6));
    const M = jetBandMatrix(), m = args(constOf("JET_BAND_M"));
    expect(m.length).toBe(9);
    // WGSL mat3x3 is column-major: column b = band b.
    for (let b = 0; b < 3; b++) for (let r = 0; r < 3; r++) expect(Math.abs(m[b * 3 + r] / M[r][b] - 1)).toBeLessThan(1e-7);
  });
});
```
Run → FAIL (`no SYN_LNCJ`).

- [ ] **Step 2: WGSL** — in `emission-shared.wgsl`, append the block below after `jetEmissionCoreJ`.
The old core (`JET_PBEAM`, `boostJ`, `jetEmissionCoreJ`) stays until Task 5 switches the renderer and
deletes it, so the renderer keeps compiling and every gate runs in this task. (Constants computed
while planning from the Task 1–3 functions; the Step 1 test re-derives them.)

```wgsl
// --- Tier 2B synchrotron (spec 2026-10-02; twin: src/physics/synchrotron.ts). Constants for p = 2.4,
// gamma_min = 10; tests/synchrotron.test.ts checks every literal against the CPU twin. -------------
const SYN_P = 2.4;
const SYN_LNCJ = -42.13253082;
const SYN_LNCA = 28.35789673;
const SYN_LNNUMIN0 = 19.85549701;
const JET_LNNU = vec3<f32>(34.13261924, 33.93194855, 33.76489446);   // ln nu at 450, 550, 650 nm
// I_nu (cgs) per band -> linear sRGB in the disk's units (before lumNorm); column b = band b.
const JET_BAND_M = mat3x3<f32>(-57.85676090, -199.6091852, 8197.725079, 563.0803652, 5602.051454, -395.0105294, 4364.033919, -151.6783697, -69.56073163);
const GAMMA_REF_Z = 280.0; const GAMMA_SLOPE = 0.58;

fn jetGammaAt(z: f32, g280: f32) -> f32 { return max(1.0, g280 * pow(abs(z) / GAMMA_REF_Z, GAMMA_SLOPE)); }
// Unit flow direction (n_r, n_theta): outward along the funnel family rho = q rho_f(z).
fn streamlineDirJ(r: f32, th: f32) -> vec2<f32> {
  let z = r * cos(th); let rho = r * sin(th); let az = max(abs(z), 1e-6);
  let vr = (rho / funnelEdgeJ(z)) * (JET_SLOPE / (2.0 * sqrt(az)));
  let vz = select(-1.0, 1.0, z >= 0.0);
  let n = sqrt(vr * vr + vz * vz); let ur = vr / n; let uz = vz / n;
  return vec2<f32>(ur * sin(th) + uz * cos(th), ur * cos(th) - uz * sin(th));
}
// nu_plasma / nu_observed = p.u for the camera-normalised past-directed photon momentum p at (r, th);
// the plasma moves at Lorentz factor gamma along the streamline in the ZAMO frame (spec 2.2/2.4).
fn plasmaShiftJ(r: f32, th: f32, p: vec4<f32>, a: f32, gamma: f32) -> f32 {
  let g = gUp(r, th, a);
  let lapse = sqrt(-1.0 / g[0]); let omega = g[1] / g[0];
  let beta = sqrt(max(0.0, 1.0 - 1.0 / (gamma * gamma)));
  let n = streamlineDirJ(r, th);
  return gamma * ((p.x + omega * p.w) / lapse + beta * (n.x * sqrt(g[2]) * p.y + n.y * sqrt(g[3]) * p.z));
}
// Density modulation (wall x length falloff x knots x turbulence); 0 outside the emitting jet.
fn jetShapeJ(r: f32, th: f32, t: f32, jetLength: f32, knotAmp: f32, g280: f32) -> f32 {
  let z = r * cos(th); let az = abs(z);
  if (az < JET_ZBASE || az > jetLength) { return 0.0; }
  let rho = r * sin(th);
  let w = wallJ(rho, z);
  if (w <= 0.0) { return 0.0; }
  let turb = 1.0 + JET_TURB * (vnoiseE(log(1.0 + rho), JET_KZ * z) - 0.5) * 2.0;
  return max(0.0, w * lengthFalloffJ(z, jetLength) * knotsJ(z, t, jetGammaAt(z, g280), knotAmp) * turb);
}
struct SynchOut { j: vec3<f32>, a: vec3<f32> };
// Per band: j = (nu/nu')^3 j'(nu') (observed-frame weighted, cgs per sr) and alpha' (1/cm) at nu' = D nu.
fn synchSampleJ(r: f32, th: f32, D: f32, a: f32, b0: f32, kScale: f32, shape: f32) -> SynchOut {
  let z = r * cos(th); let rho = r * sin(th);
  let rf = funnelEdgeJ(z); let rH = 1.0 + sqrt(max(0.0, 1.0 - a * a)); let w = rho * a / (4.0 * rH);
  let lnB = log(b0 / (rf * rf)) + 0.5 * log(1.0 + w * w);
  let lnK = log(kScale) + 2.0 * lnB + log(shape);
  let lnNuMin = SYN_LNNUMIN0 + lnB;
  let lnD = log(D);
  var o: SynchOut;
  for (var b = 0; b < 3; b++) {
    let lnNu = JET_LNNU[b] + lnD;
    let le = max(lnNu, lnNuMin);
    var lj = SYN_LNCJ + lnK + 0.5 * (SYN_P + 1.0) * lnB - 0.5 * (SYN_P - 1.0) * le;
    var la = SYN_LNCA + lnK + 0.5 * (SYN_P + 2.0) * lnB - 0.5 * (SYN_P + 4.0) * le;
    if (lnNu < lnNuMin) { lj += (lnNu - lnNuMin) / 3.0; la -= (5.0 / 3.0) * (lnNu - lnNuMin); }
    o.j[b] = exp(lj - 3.0 * lnD);
    o.a[b] = exp(la);
  }
  return o;
}
// Light from the jet along one ray: I = observed I_nu per band (cgs), tau = optical depth from the camera.
struct JetOut { I: vec3<f32>, tau: vec3<f32> };
// One sample = a uniform slab of plasma-frame path ds (cm): exact solution, thin -> j ds, thick ->
// source function j / alpha, attenuated by what lies in front (twin: slabStep in synchrotron.ts).
fn jetSlabJ(acc: JetOut, j: vec3<f32>, alpha: vec3<f32>, ds: f32) -> JetOut {
  let dt = alpha * ds;
  let fac = select((vec3<f32>(1.0) - exp(-dt)) / max(dt, vec3<f32>(1e-30)), vec3<f32>(1.0) - 0.5 * dt, dt < vec3<f32>(1e-4));
  var o: JetOut;
  o.I = acc.I + j * ds * fac * exp(-acc.tau);
  o.tau = min(acc.tau + dt, vec3<f32>(1e30));
  return o;
}
```

- [ ] **Step 3: TS twin `jetShape` (tests first)** — in `tests/jet.test.ts`: delete the `jet beaming`
(`dopplerBoost`) describe; delete the "exactly 0 when jetStrength = 0" case (on/off is now the renderer's
switch, not an emission argument); rewrite the remaining `jetEmission(r, th, t, 5, 1, 60, 0.7)` calls as
`jetShape(r, th, t, 60, 0.7, 2)` (same positions and expectations), including the envelope property test
(`jetShape(r, th, t, 60, 0.7, 2) > 0` ⇒ `inJetEnvelope`); fix the import line. Add:

```ts
  it("jetShape is a mean-one modulation of the wall profile (energy budget assumes it, synchrotron.ts)", () => {
    // average over many times at a fixed point in the wall: the knots stream past and average out.
    // g280 = 8 so the flow moves here (Gamma(20) = 1.73; at g280 = 2 the plasma is still at Gamma = 1
    // below z ~ 84 and the knot pattern is static there - a consequence of M87's measured profile).
    const r = 20, th = Math.atan2(funnelEdge(20) * JET.qPeak, 20), z = r * Math.cos(th), rho = r * Math.sin(th);
    let m = 0; const N = 4000;
    for (let k = 0; k < N; k++) m += jetShape(r, th, k * 0.37, 60, 0.7, 8);
    const turb = 1 + JET.turbAmpJet * (vnoise(Math.log(1 + rho), JET.kz * z) - 0.5) * 2;
    expect(m / N / (wallProfile(rho, z) * lengthFalloff(z, 60) * turb)).toBeCloseTo(1, 1);
  });
```
(import `vnoise` from `../src/physics/emission` and `lengthFalloff`, `wallProfile`, `funnelEdge`, `JET`,
`jetShape` from `../src/physics/jet`.) Run `npx vitest run tests/jet.test.ts` → FAIL (`jetShape` missing).

Then in `jet.ts` replace `jetEmission` and `dopplerBoost` with:

```ts
/** Density modulation of the synchrotron jet (wall x length falloff x knots x turbulence); 0 outside the
 *  emitting region. Twin of jetShapeJ in emission-shared.wgsl. Knots ride the local flow, Gamma(z). */
export function jetShape(r: number, th: number, t: number, jetLength: number, knotAmp: number, g280: number): number {
  const z = r * Math.cos(th), az = Math.abs(z);
  if (az < JET.zBase || az > jetLength) return 0;
  const rho = r * Math.sin(th), w = wallProfile(rho, z);
  if (w <= 0) return 0;
  const turb = 1 + JET.turbAmpJet * (vnoise(Math.log(1 + rho), JET.kz * z) - 0.5) * 2;
  return Math.max(0, w * lengthFalloff(z, jetLength) * knots(z, t, gammaProfile(z, g280), knotAmp) * turb);
}
```
(`gammaProfile` is in jet.ts since Task 2, so there is no import cycle.) Remove `pBeam`, `gain`, `ceil`
from `JET` and the paragraph of its doc comment about `gain`/`ceil`; update the `inJetEnvelope` comment
("where jetShape can be non-zero"). Run → PASS.

- [ ] **Step 4: Parity harness** — `src/render/jet-parity.wgsl`:

Replace the whole file:

```wgsl
// Parity harness for the synchrotron jet: plasmaShiftJ, jetShapeJ, synchSampleJ and jetSlabJ come from
// emission-shared.wgsl (sole copies, also prepended to raytrace.wgsl), so ?parity checks the renderer's
// own bytes against src/physics/synchrotron.ts + jet.ts. Per case: x = (t, r, th, phi), p = photon
// momentum (covariant, camera-normalised), c = (a, g280, emission time, slab path ds in cm). Output:
// o0 = (D, shape, ln j450, ln j550), o1 = (ln j650, ln a450, ln a550, ln a650),
// o2 = (ln I450, tau450, ln I650, tau650) after two jetSlabJ steps of path ds from zero.
// The fixed physics constants below match parity.browser.ts.
struct JIn { x: vec4<f32>, p: vec4<f32>, c: vec4<f32> };
struct JOutP { o0: vec4<f32>, o1: vec4<f32>, o2: vec4<f32> };
@group(0) @binding(0) var<storage, read> inp: array<JIn>;
@group(0) @binding(1) var<storage, read_write> outp: array<JOutP>;
const P_JETLEN = 60.0; const P_KNOTS = 0.7; const P_B0 = 632.4; const P_KSCALE = 0.1465;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = inp[gid.x];
  let r = c.x.y; let th = c.x.z; let a = c.c.x;
  let D = plasmaShiftJ(r, th, c.p, a, jetGammaAt(r * cos(th), c.c.y));
  let shape = jetShapeJ(r, th, c.c.z, P_JETLEN, P_KNOTS, c.c.y);
  let s = synchSampleJ(r, th, D, a, P_B0, P_KSCALE, shape);
  var acc: JetOut; acc.I = vec3<f32>(0.0); acc.tau = vec3<f32>(0.0);
  acc = jetSlabJ(acc, s.j, s.a, c.c.w);
  acc = jetSlabJ(acc, s.j, s.a, c.c.w);
  outp[gid.x] = JOutP(vec4<f32>(D, shape, log(s.j.x), log(s.j.y)),
                      vec4<f32>(log(s.j.z), log(s.a.x), log(s.a.y), log(s.a.z)),
                      vec4<f32>(log(acc.I.x), acc.tau.x, log(acc.I.z), acc.tau.z));
}
```

In `parity.browser.ts` change the jet imports to
`import { jetShape } from "../physics/jet";` and
`import { plasmaShift, streamlineDir, gammaProfile, synchCoeffs, jetField, slabStep, JET_BANDS_NM, C_CGS } from "../physics/synchrotron";`
(`stepGeodesic`, `stepSize`, `H_TOL`, `H_TOL_FAR`, `screenToState` are already imported), and replace
the `// --- jet parity` block (through its `jcases.forEach`) with:

```ts
  // --- jet parity (CPU synchrotron.ts / jet.ts vs the SHIPPED emission-shared.wgsl) ---
  // Cases are real photon states inside the jet: rays walked with the CPU integrator from the camera
  // until the sample lies in the emitting region, so D sees realistic momenta (approaching and
  // receding lobes, two spins, two inclinations). ds is chosen per case so the slab steps span
  // thin (dtau << 1) and thick (dtau >> 1) samples.
  const J_B0 = 632.4, J_KS = 0.1465, J_LEN = 60, J_KNOTS = 0.7;
  type JCase = { s: Float64Array; a: number; g280: number; tEm: number; ds: number };
  const jcases: JCase[] = [];
  const rays: [number, number, number, number, number][] = [ // alpha, beta, a, incl (rad), g280
    [0.5, 4, 0.9, (17 * Math.PI) / 180, 2], [1.5, 8, 0.9, (17 * Math.PI) / 180, 4],
    [0.5, -4, 0.5, (60 * Math.PI) / 180, 2], [1.5, 8, 0.5, (60 * Math.PI) / 180, 6],
    [0.3, 6, 0.9, (60 * Math.PI) / 180, 2], [0.8, -6, 0.9, (17 * Math.PI) / 180, 3],
  ];
  for (const [al, be, a, inc, g280] of rays) {
    let s = screenToState(al, be, a, inc, 1000);
    const rh = 1 + Math.sqrt(1 - a * a);
    for (let k = 0; k < 20000 && jetShape(s[1], s[2], 1.7, J_LEN, J_KNOTS, g280) <= 0; k++) {
      const out = stepGeodesic(s, a, stepSize(s, rh, 40), s[1] > 60 ? H_TOL_FAR : H_TOL);
      if (!out.ok || out.s[1] <= rh * 1.005 || out.s[1] > 1200) break;
      s = out.s;
    }
    // A ray that misses the jet fails loudly; swap its (alpha, beta) for one that hits (keep >= 2 cases
    // per lobe: beta > 0 sees the upper lobe at these inclinations, beta < 0 the lower one).
    if (jetShape(s[1], s[2], 1.7, J_LEN, J_KNOTS, g280) <= 0) throw new Error(`jet parity: ray (${al}, ${be}) never entered the jet`);
    jcases.push({ s, a, g280, tEm: 1.7, ds: jcases.length % 2 ? 1e14 : 1e9 });
  }
  const J_IN = 12, J_OUT = 12;
  const jin = device.createBuffer({ size: jcases.length * J_IN * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const jarr = new Float32Array(jcases.length * J_IN);
  jcases.forEach((c, i) => jarr.set([...c.s.slice(0, 8), c.a, c.g280, c.tEm, c.ds], i * J_IN));
  device.queue.writeBuffer(jin, 0, jarr);
  const jout = device.createBuffer({ size: jcases.length * J_OUT * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const jread = device.createBuffer({ size: jcases.length * J_OUT * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const jmod = device.createShaderModule({ code: integratorSharedWGSL + emissionSharedWGSL + jetParityWGSL });
  const jpipe = device.createComputePipeline({ layout: "auto", compute: { module: jmod, entryPoint: "main" } });
  const jbind = device.createBindGroup({ layout: jpipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: jin } }, { binding: 1, resource: { buffer: jout } }] });
  const jenc = device.createCommandEncoder();
  const jcp = jenc.beginComputePass(); jcp.setPipeline(jpipe); jcp.setBindGroup(0, jbind); jcp.dispatchWorkgroups(jcases.length); jcp.end();
  jenc.copyBufferToBuffer(jout, 0, jread, 0, jcases.length * J_OUT * 4);
  device.queue.submit([jenc.finish()]);
  await jread.mapAsync(GPUMapMode.READ);
  const jgpu = new Float32Array(jread.getMappedRange().slice(0));
  const jetErrs: number[] = [];
  jcases.forEach((c, i) => {
    // the CPU side reads the same f32-rounded inputs the GPU got
    const s = Float64Array.from(jarr.subarray(i * J_IN, i * J_IN + 8));
    const r = s[1], th = s[2], z = r * Math.cos(th), rho = r * Math.sin(th);
    const D = plasmaShift(s, c.a, gammaProfile(z, c.g280), ...streamlineDir(r, th));
    const shape = jetShape(r, th, c.tEm, J_LEN, J_KNOTS, c.g280);
    const B = jetField(rho, z, c.a, J_B0), K = J_KS * B * B * shape;
    const ja = JET_BANDS_NM.map((nm) => synchCoeffs(K, B, (C_CGS / (nm * 1e-7)) * D));
    const slab = (b: number) => { let I = 0, tau = 0; const j = ja[b][0] / D ** 3;
      for (let k = 0; k < 2; k++) [I, tau] = slabStep(I, tau, j, ja[b][1], c.ds); return [Math.log(I), tau]; };
    const want = [D, shape, Math.log(ja[0][0] / D ** 3), Math.log(ja[1][0] / D ** 3),
      Math.log(ja[2][0] / D ** 3), Math.log(ja[0][1]), Math.log(ja[1][1]), Math.log(ja[2][1]), ...slab(0), ...slab(2)];
    let e = 0;
    for (let k = 0; k < J_OUT; k++) e = Math.max(e, Math.abs(jgpu[i * J_OUT + k] - want[k]) / (1 + Math.abs(want[k])));
    jetErrs.push(e); maxErr = Math.max(maxErr, e);
  });
  console.log("jet parity relErr per case", jetErrs.map((e) => e.toExponential(2)));
```
(The log comparisons keep the 10⁻³⁰…10³⁰ dynamic range of the coefficients meaningful in f32; the slab
τ values for the thick cases are large, so their relative error is what counts — the metric's `1 + |want|`
denominator gives exactly that.)

- [ ] **Step 5: Run** — `npx vitest run` → all PASS (the constant twin test now passes); `npm run build` green.
GPU gates (dev server :5173): `npm run verify:gpu` → `?parity` PASS including the 6 synchrotron cases (log the
per-case errors; expected ≲ 1e-4 from f32 — record the max in the ledger), `?golden` PASS **unchanged** (the
renderer does not call the new functions yet; a mismatch means the compiler rescheduled shared code — apply
the A/B rule), `?cachecheck` and app checks PASS.

- [ ] **Step 6: Commit** — `git add src/render/emission-shared.wgsl src/render/jet-parity.wgsl src/test/parity.browser.ts src/physics/jet.ts tests/jet.test.ts tests/synchrotron.test.ts && git commit -m "Shared WGSL synchrotron: plasma shift, density shape, coefficients, slab transfer; parity on the shipped bytes"`

---

### Task 5: Radiative transfer in the renderer

**Files:**
- Modify: `src/render/uniforms.ts`, `tests/uniforms.test.ts`, `src/render/raytrace.wgsl`,
  `src/render/emission-shared.wgsl` (delete the old core), `src/test/scenes.ts`, `src/test/shadow.browser.ts`,
  `scripts/bench.mjs`, `scripts/shot-delay.mjs`, `src/main.ts` (state, uniform literal, physics refresh only),
  `src/test/golden.json` (re-recorded)

**Interfaces:**
- Consumes: Task 2 `jetUniforms`; Task 4 WGSL `jetShapeJ`, `plasmaShiftJ`, `jetGammaAt`, `synchSampleJ`,
  `JetOut`, `jetSlabJ`, `JET_BAND_M`.
- Produces: `UniformValues.jetB0`, `.jetKScale`, `.rgCm` (floats 32–34; size 144 bytes); WGSL `U.jetB0/jetKScale/rgCm`;
  `jetStep(s, sNew, dl, acc: JetOut) -> JetOut`; `replayJet(bm, nJet) -> JetOut`; `TraceOut.jet: JetOut`;
  `storeComposite(idx, color, jet: JetOut)`. Semantics: `U.jetStrength` = jet on (1) / off (0); `U.jetGamma` = Γ₂₈₀.

- [ ] **Step 1: Uniforms (test first)** — `tests/uniforms.test.ts`: add `jetB0: 632.4, jetKScale: 0.1465, rgCm: 9.6e14`
to the first literal (and zeros to the second); assert `UNIFORM_SIZE` is **144** and floats at bytes 128, 132, 136.
Run → FAIL. Then `uniforms.ts`: header line `+ jetB0, jetKScale, rgCm (3, synchrotron jet) -> 27 floats`; fields
```ts
  /** Synchrotron jet (spec 2026-10-02): funnel field scale (G), energy-budget density scale, r_g (cm). */
  jetB0: number; jetKScale: number; rgCm: number;
```
`UNIFORM_FLOATS = 27`; `f[32] = u.jetB0; f[33] = u.jetKScale; f[34] = u.rgCm;`. WGSL struct: after
`lightDelay: f32,` add `jetB0: f32, jetKScale: f32, rgCm: f32,` (offsets 128/132/136, size 144).

- [ ] **Step 2: Callers** — compute the three values with `jetUniforms(massSun, a, lambda, eps, jetLength)`:
  - `scenes.ts`: `jetUniforms(CUSTOM_DEFAULT.massSun, s.a, CUSTOM_DEFAULT.lambda, 2e-3, 60)`, and set
    `jetGamma: 2` (Γ₂₈₀) in `sceneUniforms`.
  - `shadow.browser.ts`, `bench.mjs`, `scripts/shot-delay.mjs`: same with a = their spin; `jetGamma: 2`
    (`.mjs` scripts run in the page via `page.evaluate`/imports — compute the values with the same
    `jetUniforms` import the page already has, or, where the script builds a literal in Node, paste the
    values for its fixed scene printed once by vitest; the jet is off in shadow and shot-delay, so any
    finite positive values do there).
  - `main.ts`: `state.jetEff = 2e-3`, `state.jetGamma = 2` (was 5); `let jetU = jetUniforms(...)`, recomputed in
    `refreshPhysics` (`const refreshPhysics = () => { phys = computeReadouts(state, SPEED); jetU = jetUniforms(state.massSun,
    state.a, state.lambda, state.jetEff, state.jetLength); };`) — there is no jet-length slider (60 r_g is fixed),
    and spin, mass and accretion already call `refreshPhysics`; the uniform literal gets
    `jetB0: jetU.jetB0, jetKScale: jetU.jetKScale, rgCm: jetU.rgCm`. The Jet speed slider's range/label move in
    Task 6; until then it sets Γ₂₈₀ with its old range. (The UI that edits ε arrives in Task 6.)

- [ ] **Step 3: Shader transfer** — `raytrace.wgsl`:
  - delete `JET_GAIN`, `JET_CEIL`, `JET_TINT` and the `jetEmissionJ` wrapper; in `emission-shared.wgsl` delete
    `JET_PBEAM`, `boostJ` and `jetEmissionCoreJ` (no callers remain); update the comments that name
    `jetEmissionJ` (`jetTouches`, `inJetEnvelopeJ`) to `jetShapeJ`.
  - `jetStep`:
```wgsl
// Synchrotron emission and absorption along one integrator step (spec 2026-10-02 2.4): each sub-sample
// is a uniform slab of plasma-frame path ds' = r_g D dl / n, with D = nu' / nu_obs from the photon
// momentum (interpolated across the step like the position and time) and the local flow.
fn jetStep(s: State, sNew: State, dl: f32, accIn: JetOut) -> JetOut {
  let p0 = cartOf(s.x);
  let dvec = cartOf(sNew.x) - p0;                           // inward step (camera -> hole)
  if (dot(dvec, dvec) <= 1e-12) { return accIn; }
  if (jetChordMisses(p0, dvec)) { return accIn; }
  let n = jetSubCount(dl);
  var acc = accIn;
  for (var k = 0u; k < n; k++) {
    let q = jetSample(s, p0, dvec, k, n);
    let f = f32(k) / f32(n);
    let tS = select(s.x.x + (sNew.x.x - s.x.x) * f, s.x.x, k == 0u);
    let shape = jetShapeJ(q.x, q.y, emitTime(-tS - U.rObs), U.jetLength, U.jetKnots, U.jetGamma);
    if (shape > 0.0) {
      let D = plasmaShiftJ(q.x, q.y, mix(s.p, sNew.p, f), U.a, jetGammaAt(q.x * cos(q.y), U.jetGamma));
      if (D > 1e-6) {                                       // Review Focus 5: never divide by D -> 0
        let so = synchSampleJ(q.x, q.y, D, U.a, U.jetB0, U.jetKScale, shape);
        acc = jetSlabJ(acc, so.j, so.a, U.rgCm * D * dl / f32(n));
      }
    }
  }
  return acc;
}
```
  - `TraceOut`: `color: vec3<f32>, jet: JetOut,`. `traceRay`: replace `var jetAccum = vec3<f32>(0.0); …` with
    `var jet: JetOut; jet.I = vec3<f32>(0.0); jet.tau = vec3<f32>(0.0); // synchrotron light and optical depth along the ray`,
    `if (U.jetStrength > 0.0) { jet = jetStep(s, sNew, dl, jet); }`, and `out.jet = jet;`.
  - `replayJet(bm: State, nJet: u32) -> JetOut`: `var acc: JetOut; acc.I = vec3<f32>(0.0); acc.tau = vec3<f32>(0.0);`
    and `acc = jetStep(s, st.s, st.dl, acc);` in the loop.
  - `storeComposite(idx: u32, color: vec3<f32>, jet: JetOut)`:
```wgsl
  // Light from behind the jet (disk, sky) is absorbed: R, G, B by the 650 / 550 / 450 nm optical depths.
  // The jet's own light enters in the disk's units (band matrix) times the disk's lumNorm (spec 2.5);
  // clamped at 0 like blackbodyVisibleRGB (a pure power law can sit just outside the sRGB gamut).
  let raw = color * exp(-vec3<f32>(jet.tau.z, jet.tau.y, jet.tau.x)) + U.lumNorm * max(JET_BAND_M * jet.I, vec3<f32>(0.0));
```
    (the finite-range guard and EMA blend after it stay as they are). With the jet off, `jet` is all zeros and
    `raw = color * 1 + 0`: the jet-off image is the old one.
  - `shade`: `var jet: JetOut; jet.I = vec3<f32>(0.0); jet.tau = vec3<f32>(0.0);` and `jet = replayJet(...)` in the
    bookmark branch; `storeComposite(idx, color, jet)`. The cached colour is attenuated by the replayed τ inside
    `storeComposite`, exactly as the live trace (Review Focus 1).

- [ ] **Step 4: Local gates** — `npm test`, `npm run build` green.

- [ ] **Step 5: GPU gates** (dev server :5173):
  1. `npm run verify:gpu` **without** recording. Expected: `?parity` PASS, `?golden` FAIL on the jet scenes
     only — `jet-off` must still match `0e26f472` (Review Focus 4). If `jet-off` differs, stop and apply the
     2026-10-01 A/B rule (restore the old data flow in an experiment, show the difference is compiler
     rescheduling, ledger the ruling) before recording.
  2. `RECORD_GOLDEN=1 npm run verify:gpu`: golden recorded and PASS, `?cachecheck` PASS (cached = live; the jet
     scenes exercise the τ attenuation of cached colours — Review Focus 1), app checks PASS. Record the new
     hashes in the ledger.
  3. `node scripts/probe-scale.mjs` → PASS. (`probe-axis` still drives the old `#jet` slider; it runs in Task 6.)

- [ ] **Step 6: Commit** — `git add` the files above + `src/test/golden.json`;
`git commit -m "Synchrotron radiative transfer in the renderer and cache; absolute jet brightness"`.

---

### Task 6: Panel, presets, probes, screenshots

**Files:**
- Modify: `index.html`, `src/main.ts`, `src/physics/presets.ts`, `tests/presets.test.ts`, `scripts/probe-axis.mjs`, `scripts/verify-gpu.mjs`

**Interfaces:**
- Consumes: Task 5 state fields (`jetEff`, `jetGamma`), `refreshPhysics`.
- Produces: DOM `#jeton` (checkbox), `#jeteff` / `#jeteffv` (log10 ε slider), `#jg` relabelled Γ₂₈₀ (1.5–8, default 2);
  `Preset.jet: boolean` (replacing `jetStrength`); the old `#jet` slider and `#jetv` are removed.

- [ ] **Step 1: Presets (test first)** — in `tests/presets.test.ts` replace the jet-strength test with:
```ts
  it("jets: on for M87*, Cygnus X-1, GRS 1915+105; off for Sgr A* and Gargantua", () => {
    const on = Object.fromEntries(PRESETS.map((p) => [p.id, p.jet]));
    expect(on).toEqual({ m87: true, sgra: false, cygx1: true, grs1915: true, gargantua: false });
  });
```
Run → FAIL. In `presets.ts`: `jetStrength: number` → `jet: boolean` (with the doc comment "jet on/off;
brightness follows from the energy budget, spec 2026-10-02"); values per the test; M87\* caption: replace
the sentence "The jet is dimmed to ~10 % … (2021 data)." with "Its jet's brightness follows from energy
conservation: it radiates ε (≈ 0.2 % by default, from M87's optical nucleus vs its jet power) of its
Blandford–Znajek power as visible light. The EHT's ring is radio emission from a hot flow this renderer
does not model." Cyg X-1 / GRS captions: replace "The jet's brightness this close to the hole has not
been measured; it is shown at the model's default." with "Its jet uses the same energy budget; the jet's
brightness this close to the hole has not been measured." Remove the M87\* `0.1 / 29.8` comment. Tests PASS.

- [ ] **Step 2: Panel** — `index.html`: replace the "Jet" slider block with
```html
    <div class="ctrl">
      <label class="row" style="cursor:pointer"><span>Jet</span>
        <input id="jeton" type="checkbox" checked style="accent-color:var(--accent)"></label>
    </div>
    <div class="ctrl" title="Fraction of the jet's Blandford–Znajek power radiated as visible light">
      <div class="row"><label>Jet efficiency&nbsp;ε</label><span class="val"><b id="jeteffv">2.0e-3</b></span></div>
      <input id="jeteff" type="range" min="-5" max="-1" step="0.05" value="-2.7">
    </div>
```
and the Jet speed control: label `Jet speed&nbsp;Γ<sub>280</sub>`, `min="1.5" max="8" step="0.1" value="2"`,
readout `2.0`. `main.ts`: state `jetOn: true` (uniform `jetStrength: state.jetOn ? 1 : 0`); handlers:
```ts
  const jeton = $("jeton") as HTMLInputElement, jeteff = $("jeteff") as HTMLInputElement, jeteffv = $("jeteffv");
  const showEff = () => { jeteffv.textContent = state.jetEff.toExponential(1); };
  jeton.addEventListener("change", () => { state.jetOn = jeton.checked; reset(); });
  jeteff.addEventListener("input", () => { state.jetEff = 10 ** +jeteff.value; showEff(); physicsChanged(); });
```
remove `jet`/`jetv`/`showJet`; `applyPreset`: `state.jetOn = p.jet; jeton.checked = p.jet;` (instead of
the jet-strength lines); `DEFAULT_VIEW` gets `jet: true`; call `showEff()` at start-up. Build + tests green.

- [ ] **Step 3: Probes** — `scripts/probe-axis.mjs`: replace `await setSlider("jet", 0);` with
```js
  await page.evaluate(() => { const c = document.getElementById("jeton"); c.checked = false; c.dispatchEvent(new Event("change", { bubbles: true })); });
```
`scripts/verify-gpu.mjs`: no id references to `#jet` remain (check with a search).

- [ ] **Step 4: Gates and screenshots** — `npm run verify:gpu` (all PASS, presets lit and warning-free —
Review Focus 2), `node scripts/probe-axis.mjs`, `node scripts/probe-scale.mjs` PASS. `node scripts/shot-presets.mjs`;
look at every image and at a close-up of each jet. Measure and record, for the M87\*, Cygnus X-1 and
Default views, the jet's share of the rendered light (the jet-on minus jet-off accumulation, as in the
2026-10-01 jet-calibration measurement) — a prediction of the model, reported, not tuned.

- [ ] **Step 5: Commit** — `git add index.html src/main.ts src/physics/presets.ts tests/presets.test.ts scripts/probe-axis.mjs scripts/verify-gpu.mjs && git commit -m "Jet panel: on/off, radiative efficiency, Gamma at 280 r_g; presets and probes"`

---

### Task 7: Documentation

- [ ] README: `## Features` jet bullet → "Synchrotron jet — magnetically arrested field, M87's measured
acceleration, p = 2.4 electrons, emission and self-absorption transferred in general relativity, brightness
from energy conservation (ε × Blandford–Znajek power)"; a Status paragraph **Synchrotron jet (2026-10-02)**
with the physics choices and sources, the two validation numbers (closed forms vs integration; M87\* P_BZ),
the planning correction (RL 6.53 per unit energy), the energy-budget rationale (uncooled electrons would
exceed the jet power), the measured jet shares from Task 6, the gate results and the golden hashes; and two
consequences a viewer will notice: at the default Γ₂₈₀ = 2 the plasma inside the ±60 r_g view is still at
Γ = 1 (M87's profile reaches Γ > 1 only beyond ~84 r_g), so the knots there do not stream and there is no
beaming asymmetry between the lobes until Jet speed is raised; and what Task 6's screenshots show for the
X-ray binaries, whose visible band lies below ν_min (B ~ 10⁸ G), so their jets are in the low-frequency,
strongly self-absorbed regime. The README's Tier 2B caveat list is
rewritten: the jet is no longer phenomenological; what remains prescribed is the field and flow geometry.
- [ ] ROADMAP: tick "Synchrotron jet emission (2026-10-02)"; spec status "implemented on feat/synchrotron-jet".
- [ ] Commit `"Document the synchrotron jet"`.
