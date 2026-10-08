# MAD R-β electron heating for the hot flow — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hot flow's fixed electron temperature with magnetically-arrested R-β heating so M87\*'s 1.3 mm ring lands inside the EHT's 2σ, with one model for every hot flow.

**Architecture:** The physics lives in a CPU module (`src/physics/hot-flow.ts`) and its WGSL twin (`flowCoeffsJ` in `src/render/emission-shared.wgsl`), kept equal by `?parity` and by `tests/jet.test.ts`'s constant checks. The density scale n₀ and the hotspot amplitude A₀ are re-fitted by the existing calibration scripts. The gates (`?hotflow`, `?accuracy`, golden, the SWEEP tests) move to the new targets.

**Tech Stack:** TypeScript, WebGPU/WGSL, Vite (dev server on :5173), vitest, Playwright-driven `npm run verify:gpu`.

**Spec:** `docs/specs/2026-10-08-mad-rbeta-hotflow-design.md`

## Global Constraints

- Only the hot flow's electron temperature and field change. Density (n ∝ r^-1.1 e^(−z²/2ρ²)), velocity (Pu 50/50), the Mahadevan emissivity, Kirchhoff absorption, transfer, the visible view, the jet and the integrator do not.
- Constants: k T_i = m_p c²/(3r); midplane β_eq = 1; R_high = 160; R_low = 1; B²/8π = n_eq m_p c² r_S/(12 r β_eq) with n_eq = n₀ (r/2)^-1.1; β = 2 β_eq e^(−z²/2ρ²); T_e = T_i / R(β), R(β) = (R_high β² + R_low)/(1 + β²).
- M87\* EHT target 42 ± 3 µas, gated inside 2σ (36–48). Sgr A\* EHT 51.8 ± 2.3 is reported, not gated. If M87\* leaves 2σ at 96² (CPU) or 512² (GPU), stop and report; do not tune parameters.
- Ring robustness: |ring(unblurred) − ring(15 µas blur)| ≤ 3 µas for every gated ring.
- The machine runs low on memory: run one long job at a time (calibration, fixture builds, verify-gpu), and the dev server only while GPU checks need it.
- Commits: no Co-Authored-By trailer (user rule). Branch `feat/mad-rbeta`.
- All numbers written into code come from a script's output in that task, never typed from this plan's estimates.

## Review Focus

- **Pixels on the polar axis** (ρ < 1e-6): `flowBeta` must return 0 and `flowCoeffs` [0, 0] without NaN from z²/ρ². Pinned in Task 1 (axis test) and Task 2 (parity includes near-axis flow cases).
- **Far off the midplane** (e^(−g) underflows to 0): β → 0, R → R_low, finite T_e. WGSL `exp(-g)` for large g must not produce NaN. Pinned in Task 1 (large-z test) and Task 2 (a WGSL emission check at z/ρ = 20).
- **Custom hot-flow objects far from both calibrated masses** (e.g. Gargantua, the default view's 1e8 M☉ at λ 3.7e-4): they keep scaling n₀ from the nearest calibrated object and must render finite. Pinned in Task 5 (garg-mm accuracy scene) and Task 6 (app checks open the default and presets in 1.3 mm).
- **Bimodal radial profiles** that make the ring jump between two peaks: the robustness check must catch them on the CPU and the GPU. Pinned in Task 3 (sweep test) and Task 4 (`?hotflow`).
- **Hotspot frames**: the hotspot boosts the flow's electrons, so its added flux changes with T_e. A₀ and `HOTSPOT_TWIN` must be re-fitted before `?hotflow`'s hotspot rows are judged. Pinned in Task 3.

---

### Task 1: CPU physics — R-β electron temperature and the MAD field

**Files:**
- Modify: `src/physics/hot-flow.ts`
- Test: `tests/hot-flow.test.ts`

**Interfaces:**
- Produces: `HOTFLOW` gains `betaEq: 1, rHigh: 160, rLow: 1` and loses `T0`, `beta`. New exports: `flowDensityEq(r: number, n0: number): number`, `flowIonTemperature(r: number): number`, `flowBeta(r: number, th: number): number`, `flowRatio(beta: number): number`. Changed signatures: `flowTemperature(r: number, th: number): number`, `flowField(r: number, n0: number): number` (takes n₀, not the local density). `flowCoeffs(r, th, nu, n0)` is unchanged in signature.

- [ ] **Step 1: Write the failing tests.** In `tests/hot-flow.test.ts`, update the import line to:

```ts
import { HOTFLOW, mahadevanM, flowDensity, flowDensityEq, flowIonTemperature, flowBeta, flowRatio, flowTemperature, flowField, iscoEL, flowVelocity, flowCoeffs, isHotFlow, flowN0, HOTFLOW_N0 } from "../src/physics/hot-flow";
```

Replace the test `"Broderick profiles: density r^-1.1 with a Gaussian in z/rho, temperature r^-0.84, beta = 10 toroidal field"` with:

```ts
  it("Broderick density r^-1.1 with a Gaussian in z/rho; its midplane value is the field's reference", () => {
    expect(flowDensity(4, Math.PI / 2, 1e7)).toBeCloseTo(1e7 * 2 ** -1.1, 3);
    const th = Math.atan2(4, 3), r = 5; // rho = 4, z = 3
    expect(flowDensity(r, th, 1e7)).toBeCloseTo(1e7 * (r / 2) ** -1.1 * Math.exp(-9 / 32), 3);
    expect(flowDensity(10, 1e-9, 1e7)).toBe(0); // on the axis
    expect(flowDensityEq(r, 1e7)).toBeCloseTo(flowDensity(r, Math.PI / 2, 1e7), 6);
  });
  it("MAD R-beta electrons (spec 2026-10-08): virial ions, field from the midplane density, T_e = T_i / R(beta)", () => {
    const C = 2.99792458e10, MP = 1.67262192e-24, KB = 1.380649e-16;
    expect(HOTFLOW.betaEq).toBe(1); expect(HOTFLOW.rHigh).toBe(160); expect(HOTFLOW.rLow).toBe(1);
    expect(flowIonTemperature(3)).toBeCloseTo((MP * C * C) / (3 * KB * 3), -3);       // k T_i = m_p c^2 / (3 r)
    expect(flowIonTemperature(1) / 3.629398e12).toBeCloseTo(1, 6);
    // B^2 / 8 pi = n_eq m_p c^2 r_S / (12 r beta_eq), n_eq the midplane density
    const r = 6, n0 = 1e6, B = flowField(r, n0);
    expect((B * B) / (8 * Math.PI)).toBeCloseTo((flowDensityEq(r, n0) * MP * C * C * (2 / r)) / (12 * HOTFLOW.betaEq), 6);
    // beta is the ion plasma beta 8 pi n k T_i / B^2 = 2 beta_eq e^{-z^2 / 2 rho^2}
    for (const th of [Math.PI / 2, 1.2, 0.7]) {
      const n = flowDensity(r, th, n0), def = (8 * Math.PI * n * KB * flowIonTemperature(r)) / (B * B);
      expect(flowBeta(r, th) / def).toBeCloseTo(1, 9);
    }
    expect(flowBeta(r, Math.PI / 2)).toBeCloseTo(2 * HOTFLOW.betaEq, 12);
    expect(flowBeta(r, 1e-9)).toBe(0);                                                  // on the axis
    // R(beta): R_low where the field dominates, R_high where the gas does
    expect(flowRatio(0)).toBe(1); expect(flowRatio(1)).toBeCloseTo(80.5, 12); expect(flowRatio(1e4)).toBeCloseTo(160, 4);
    expect(flowTemperature(r, Math.PI / 2)).toBeCloseTo(flowIonTemperature(r) / flowRatio(2), -3);
    // far off the midplane: beta underflows to 0, electrons as hot as the ions, never NaN
    const thFar = Math.atan2(1, 20); // rho / z = 1 / 20
    expect(flowBeta(r, thFar)).toBe(0);
    expect(flowTemperature(r, thFar)).toBeCloseTo(flowIonTemperature(r), -3);
  });
```

In the test `"coefficients: Kirchhoff with the Rayleigh-Jeans source function; zero where there is no gas"`, change `flowTemperature(6)` to `flowTemperature(6, 1.3)`.

- [ ] **Step 2: Run the tests to verify they fail.**

Run: `npx vitest run tests/hot-flow.test.ts`
Expected: FAIL (`flowDensityEq` / `flowIonTemperature` / `flowBeta` / `flowRatio` are not exported).

- [ ] **Step 3: Implement.** In `src/physics/hot-flow.ts`:

Replace the header comment's first line block so it states the new model:

```ts
// Hot accretion flow for the 1.3 mm (EHT) view (spec 2026-10-04; electrons spec 2026-10-08): the semi-analytic RIAF of
// Broderick et al. 2011/2016 (n_e ~ r^-1.1 e^{-z^2/2 rho^2}) with magnetically-arrested R-beta electron heating
// (Moscibrodzka et al. 2016, as in the EHT's GRMHD libraries): virial ions, a field set by the midplane density at
// midplane beta 1, T_e = T_i / R(beta) with R_high 160, R_low 1. Sub-Keplerian dynamics of Pu, Akiyama & Asada 2016
// (Keplerian / zero-angular-momentum free fall mixed 50/50 in u^r and Omega; equatorial profiles of r, u^t from the
// local metric), thermal synchrotron (Mahadevan et al. 1996 isotropic fit) with Kirchhoff absorption. n0 calibrated
// to the measured 230 GHz flux (scripts/calibrate-hotflow.ts). WGSL twin in emission-shared.wgsl.
```

Replace the `HOTFLOW` constant:

```ts
export const HOTFLOW = { nu: 230e9, betaEq: 1, rHigh: 160, rLow: 1, rMax: 50, dl: 0.25, lambdaHot: 0.01, kTb: 6.1528e13 } as const;
```

Replace `flowDensity`, `flowTemperature` and `flowField` with:

```ts
/** Midplane density n0 (r / 2M)^-1.1: the Broderick profile without its Gaussian, and the MAD field's reference. */
export function flowDensityEq(r: number, n0: number): number { return n0 * (r / 2) ** -1.1; }
export function flowDensity(r: number, th: number, n0: number): number {
  const z = r * Math.cos(th), rho = r * Math.sin(th);
  if (rho < 1e-6) return 0;
  return flowDensityEq(r, n0) * Math.exp(-(z * z) / (2 * rho * rho));
}
/** Virial ion temperature: k T_i = m_p c^2 / (3 r), r in M. */
export function flowIonTemperature(r: number): number { return (MP * C * C) / (3 * KB * r); }
/** Ion plasma beta 8 pi n k T_i / B^2 with the field set by the midplane density: 2 beta_eq e^{-z^2 / 2 rho^2}. 0 on the axis. */
export function flowBeta(r: number, th: number): number {
  const z = r * Math.cos(th), rho = r * Math.sin(th);
  if (rho < 1e-6) return 0;
  return 2 * HOTFLOW.betaEq * Math.exp(-(z * z) / (2 * rho * rho));
}
/** T_i / T_e = (R_high beta^2 + R_low) / (1 + beta^2) (Moscibrodzka et al. 2016). */
export function flowRatio(beta: number): number { const b2 = beta * beta; return (HOTFLOW.rHigh * b2 + HOTFLOW.rLow) / (1 + b2); }
export function flowTemperature(r: number, th: number): number { return flowIonTemperature(r) / flowRatio(flowBeta(r, th)); }
/** Magnetically arrested field from the midplane density: B^2 / 8 pi = n_eq m_p c^2 r_S / (12 r beta_eq), r_S = 2 M. */
export function flowField(r: number, n0: number): number { return Math.sqrt((8 * Math.PI * flowDensityEq(r, n0) * MP * C * C * (2 / r)) / (12 * HOTFLOW.betaEq)); }
```

In `flowCoeffs`, change the line

```ts
  const T = flowTemperature(r), the = (KB * T) / (ME * C * C), B = flowField(r, n);
```

to

```ts
  const T = flowTemperature(r, th), the = (KB * T) / (ME * C * C), B = flowField(r, n0);
```

- [ ] **Step 4: Run the tests to verify they pass.**

Run: `npx vitest run tests/hot-flow.test.ts`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Run the whole suite and type check** (other modules may still use the old exports).

Run: `npx tsc --noEmit -p . && npx vitest run`
Expected: tsc exit 0. vitest: everything passes except, at most, `tests/jet.test.ts > hot-flow constants match hot-flow.ts ...` (fixed in Task 2). Any other failure: fix the caller (grep `flowTemperature(`, `flowField(`, `HOTFLOW.T0`, `HOTFLOW.beta`).

- [ ] **Step 6: Commit.**

```bash
git add src/physics/hot-flow.ts tests/hot-flow.test.ts
git commit -m "Hot flow: magnetically arrested R-beta electron heating on the CPU (virial ions, midplane-set field, R_high 160)"
```

---

### Task 2: WGSL twin of the new coefficients

**Files:**
- Modify: `src/render/emission-shared.wgsl:303-363` (hot-flow constants and `flowCoeffsJ`)
- Modify: `tests/jet.test.ts` (test `"hot-flow constants match hot-flow.ts and the cgs values they are logs of"`)

**Interfaces:**
- Consumes: Task 1's `HOTFLOW.betaEq/rHigh/rLow`, `flowCoeffs` (the parity reference).
- Produces: WGSL constants `HF_BETA_EQ`, `HF_R_HIGH`, `HF_R_LOW`, `HF_LN_TI`; `flowCoeffsJ(r, th, lnNu, n0) -> vec2<f32>` with the same signature.

- [ ] **Step 1: Write the failing test.** In `tests/jet.test.ts`, replace the body of the test `"hot-flow constants match hot-flow.ts and the cgs values they are logs of"` with:

```ts
    for (const [n, v] of [["HF_BETA_EQ", HOTFLOW.betaEq], ["HF_R_HIGH", HOTFLOW.rHigh], ["HF_R_LOW", HOTFLOW.rLow], ["HF_RMAX", 50], ["HF_KTB", HOTFLOW.kTb]] as [string, number][])
      expect(Number(wconst(n))).toBe(v);
    expect(WGSL_E).not.toMatch(/HF_T0|HF_BETA\b/);
    expect(Number(wconst("HF_LNNU"))).toBeCloseTo(Math.log(230e9), 7);
    const C = 2.99792458e10, QE = 4.80320471e-10, ME = 9.1093837e-28, MP = 1.67262192e-24, KB = 1.380649e-16;
    const logs: [string, number][] = [["HF_LN_QE2", Math.log(QE * QE)], ["HF_LN_THE", Math.log(KB / (ME * C * C))],
      ["HF_LN_B2", Math.log((8 * Math.PI * MP * C * C * 2) / 12)], ["HF_LN_RJ", Math.log((2 * KB) / (C * C))],
      ["HF_LN_2S3C", Math.log(2 * Math.sqrt(3) * C)], ["SYN_LNNUB0", Math.log(QE / (2 * Math.PI * ME * C))],
      ["HF_LN_TI", Math.log((MP * C * C) / (3 * KB))]];
    for (const [n, v] of logs) expect(Number(wconst(n))).toBeCloseTo(v, 7);
```

- [ ] **Step 2: Run it to verify it fails.**

Run: `npx vitest run tests/jet.test.ts -t "hot-flow constants"`
Expected: FAIL (`HF_BETA_EQ` not found).

- [ ] **Step 3: Implement.** In `src/render/emission-shared.wgsl`:

Replace the line

```wgsl
const HF_T0 = 1e11; const HF_BETA = 10.0; const HF_RMAX = 50.0; const HF_KTB = 6.1528e13;
```

with

```wgsl
// Electrons (spec 2026-10-08, twin of HOTFLOW): R-beta heating, R_high 160, R_low 1, midplane beta 1; ln(m_p c^2 / 3 k).
const HF_BETA_EQ = 1.0; const HF_R_HIGH = 160.0; const HF_R_LOW = 1.0; const HF_LN_TI = 28.92008804;
const HF_RMAX = 50.0; const HF_KTB = 6.1528e13;
```

Replace the body of `flowCoeffsJ` (from its first line through `return vec2<f32>(exp(lnJ), exp(lnA));`) with:

```wgsl
fn flowCoeffsJ(r: f32, th: f32, lnNu: f32, n0: f32) -> vec2<f32> {
  let z = r * cos(th); let rho = r * sin(th);
  if (rho < 1e-6) { return vec2<f32>(0.0); }
  let g = z * z / (2.0 * rho * rho);
  let lnNeq = log(n0) - 1.1 * log(0.5 * r);              // midplane density: the MAD field's reference
  let lnN = lnNeq - g;
  if (lnN < -40.0) { return vec2<f32>(0.0); }
  // R-beta (twin: flowTemperature): beta = 2 beta_eq e^-g (0 far off the plane: R = R_low), T_e = T_i / R(beta)
  let beta = 2.0 * HF_BETA_EQ * exp(-g); let b2 = beta * beta;
  let lnT = HF_LN_TI - log(r) - log((HF_R_HIGH * b2 + HF_R_LOW) / (1.0 + b2));
  let lnThe = HF_LN_THE + lnT;
  let lnB = 0.5 * (HF_LN_B2 - log(HF_BETA_EQ) + lnNeq - log(r));
  let lnX = log(2.0 / 3.0) + lnNu - (SYN_LNNUB0 + lnB) - 2.0 * lnThe;
  let lnJ = lnN + HF_LN_QE2 + lnNu - HF_LN_2S3C - 2.0 * lnThe + lnMahadevanJ(lnX);
  let lnA = lnJ - (HF_LN_RJ + 2.0 * lnNu + lnT);
  return vec2<f32>(exp(lnJ), exp(lnA));
}
```

Also update the comment block above the hot-flow constants (the line beginning `// Broderick et al. 2011 RIAF profiles`) to say "R-beta electrons (spec 2026-10-08)" in place of the fixed temperature.

- [ ] **Step 4: Run the tests to verify they pass.**

Run: `npx vitest run tests/jet.test.ts tests/hot-flow.test.ts tests/shader-twins.test.ts`
Expected: PASS.

- [ ] **Step 5: GPU parity, including a far-off-plane case.** In `src/test/parity.browser.ts` find the hot-flow case loop (search `for (const n0 of [5.03e5, 1.5e7]) hcases.push(`). Directly after that loop's closing brace, add one far-off-plane case per n₀ so `exp(-g)` underflow is exercised (z/ρ = 20 at r = 8):

```ts
    // far off the midplane (spec 2026-10-08 review focus): e^{-g} underflows, beta -> 0, R -> R_low; must stay finite
    for (const n0 of [5.03e5, 1.5e7]) { const th = Math.atan2(1, 20), r = 8, gu = metricUpper(r, th, a), L = 0.5;
      const R = Math.max(0, -gu.tt - 2 * gu.tphi * L - gu.phph * L * L);
      hcases.push({ s: Float64Array.from([0, r, th, 0, 1, -Math.sqrt(R / gu.rr), 0, L]), a, n0 }); }
```

(Adapt variable names `a`, `metricUpper`, `hcases` to the ones in scope at that point; read 20 lines around the insertion point first. If the existing cases build `s` differently, build this one the same way with r = 8 and θ = atan2(1, 20).)

Start the dev server (PowerShell, background): `Set-Location C:\Users\shoke\Documents\Claude\blackhole-sim; npx vite --port 5173 --strictPort`
Run: `node scripts/scratch-parity.mjs` (git-excluded helper; if missing, `npm run verify:gpu` and read the `?parity` line)
Expected: `PARITY PASS`, hot flow worst ≤ 1 of tolerance.

- [ ] **Step 6: Commit.**

```bash
git add src/render/emission-shared.wgsl tests/jet.test.ts src/test/parity.browser.ts
git commit -m "Hot flow WGSL twin: R-beta electrons and the midplane-set field; parity covers a far-off-plane case"
```

---

### Task 3: Recalibrate n₀ and A₀; CPU ring gate and robustness

**Files:**
- Modify: `scripts/calibrate-hotflow.ts` (print the 15 µas blur ring)
- Modify: `src/physics/hot-flow.ts` (`HOTFLOW_N0`)
- Modify: `src/physics/hot-flow-image.ts` (`HOTFLOW_TARGETS.*.cpuRingUas`, comments: M87\* gated, Sgr A\* reported)
- Modify: `tests/sweep-hotflow.test.ts`
- Modify: `src/physics/hotspot.ts` (`HOTSPOT.A0`, `HOTSPOT_TWIN`), `src/render/emission-shared.wgsl` (`HS_A0`)
- Modify: `src/test/parity.browser.ts` (the hard-coded n₀ values `5.03e5, 1.5e7` in the flow cases → the new `HOTFLOW_N0` values)

**Interfaces:**
- Consumes: Task 1's physics.
- Produces: `HOTFLOW_N0 = { sgra, m87 }` (new values), `HOTFLOW_TARGETS.{sgra,m87}.cpuRingUas` (new), `HOTSPOT.A0`, `HOTSPOT_TWIN` (new), constant `RING_ROBUST_UAS = 3` exported from `src/physics/hot-flow-image.ts`.

- [ ] **Step 1: Write the failing test.** Replace `tests/sweep-hotflow.test.ts` with:

```ts
import { describe, it, expect } from "vitest";
import { HOTFLOW_TARGETS, RING_ROBUST_UAS, traceFlowSamples, flowImage, imageFluxJy, ringDiameterUas } from "../src/physics/hot-flow-image";
import { flowN0 } from "../src/physics/hot-flow";
import { PRESETS } from "../src/physics/presets";
const RUN = process.env.SWEEP === "1";
// spec 2026-10-08: M87* gated inside the EHT's 2 sigma; Sgr A* reported; every ring robust to a 15 uas blur
describe.skipIf(!RUN)("hot flow calibration (96^2 CPU trace, ~1.5 min per object)", () => {
  for (const id of ["sgra", "m87"] as const) {
    it(`${id}: tabulated n0 reproduces the measured 230 GHz flux within 5 %; ring = CPU twin value, robust; M87* inside 2 sigma`, () => {
      const p = PRESETS.find((q) => q.id === id)!, T = HOTFLOW_TARGETS[id];
      const rg = 1.476625e5 * p.massSun, dist = T.distKpc * 3.0857e21, uasPerM = (rg / dist) * 206264.806e6;
      const smp = traceFlowSamples(p.a, p.inclDeg, 96, 13);
      const I = flowImage(smp, flowN0(p.massSun, p.lambda), rg);
      expect(Math.abs(imageFluxJy(I, 96, 13, rg, dist) / T.jy - 1)).toBeLessThan(0.05);
      const d = ringDiameterUas(I, 96, 13, uasPerM), d15 = ringDiameterUas(I, 96, 13, uasPerM, 15);
      console.log(`${id}: ring ${d.toFixed(1)} uas (15 uas blur ${d15.toFixed(1)}) vs EHT ${T.ringUas} +- ${T.ringErr}`);
      expect(Math.abs(d - T.cpuRingUas)).toBeLessThan(0.05);
      expect(Math.abs(d - d15)).toBeLessThanOrEqual(RING_ROBUST_UAS);
      if (id === "m87") expect(Math.abs(d - T.ringUas)).toBeLessThan(2 * T.ringErr);
    }, 600000);
  }
});
```

- [ ] **Step 2: Run it to verify it fails.**

Run: `SWEEP=1 npx vitest run tests/sweep-hotflow.test.ts`
Expected: FAIL to compile (`RING_ROBUST_UAS` not exported).

- [ ] **Step 3: Add the constant and the blur print.** In `src/physics/hot-flow-image.ts`, after `HOTFLOW_TARGETS`, add:

```ts
/** A gated ring must not move by more than this under a 15 uas blur (spec 2026-10-08: rejects bimodal profiles). */
export const RING_ROBUST_UAS = 3;
```

In `scripts/calibrate-hotflow.ts`, change the console line's `(20 uas blur: ${ringDiameterUas(I, 96, 13, u, 20).toFixed(1)})` to `(15 uas blur: ${ringDiameterUas(I, 96, 13, u, 15).toFixed(1)})`.

- [ ] **Step 4: Recalibrate n₀** (stop the dev server first to free memory).

Run: `npx vite-node scripts/calibrate-hotflow.ts` (~6 min)
Expected output shape: `sgra: n0 <X> cm^-3 -> 2.400 Jy; ring <D> uas (15 uas blur <D15>) ...` and the same for m87. Probe expectation (80²): Sgr A\* n₀ ≈ 1.4e7, ring ≈ 57; M87\* n₀ ≈ 1.1e5, ring ≈ 42.

**Decision gate:** if M87\*'s ring is outside 36–48 µas, or |ring − blur15| > 3 for either object, STOP. Report the numbers to the user (spec §1: report, do not tune).

- [ ] **Step 5: Write the calibrated values.** In `src/physics/hot-flow.ts` set `HOTFLOW_N0` to the printed n₀ values (3 significant figures, as today). In `src/physics/hot-flow-image.ts` set `cpuRingUas` for both to the printed unblurred rings (1 decimal). Rewrite the `HOTFLOW_TARGETS` comments: sgra `// EHT 2022 (Sgr A* Papers I, IV); reported, not gated (spec 2026-10-08)`, m87 `// EHT 2019 (Papers I, IV, VI); gated inside 2 sigma (spec 2026-10-08)`. In `src/test/parity.browser.ts` replace `for (const n0 of [5.03e5, 1.5e7])` (both occurrences, including Task 2's added loop) with `for (const n0 of [HOTFLOW_N0.m87, HOTFLOW_N0.sgra])` and add `HOTFLOW_N0` to that file's import from `../physics/hot-flow` (add the import line if none exists).

- [ ] **Step 6: Run the sweep test to verify it passes.**

Run: `SWEEP=1 npx vitest run tests/sweep-hotflow.test.ts`
Expected: PASS for both objects.

- [ ] **Step 7: Recalibrate the hotspot.**

Run: `npx vite-node scripts/calibrate-hotspot.ts` (~10 min)
Expected: `A0 <a> (... adds 0.3000 Jy ...)` and two `HOTSPOT_TWIN.delay = {...}` / `HOTSPOT_TWIN.instant = {...}` lines.
Write `A0` into `HOTSPOT.A0` (`src/physics/hotspot.ts`) and `HS_A0` (`src/render/emission-shared.wgsl`), and the two twin objects into `HOTSPOT_TWIN`. Update the `HOTSPOT.A0` comment to say "recalibrated 2026-10-08 for the R-beta electrons".

- [ ] **Step 8: Run the hotspot tests.**

Run: `npx vitest run tests/hotspot.test.ts && SWEEP=1 npx vitest run tests/sweep-hotspot.test.ts`
Expected: PASS (the WGSL `HS_A0` twin check and the 0.3 Jy check).

- [ ] **Step 9: Commit.**

```bash
git add scripts/calibrate-hotflow.ts src/physics/hot-flow.ts src/physics/hot-flow-image.ts tests/sweep-hotflow.test.ts src/physics/hotspot.ts src/render/emission-shared.wgsl src/test/parity.browser.ts
git commit -m "Recalibrate the hot flow for R-beta electrons: n0, CPU rings, hotspot A0 and twin; M87* gated inside 2 sigma, rings robust to a 15 uas blur"
```

---

### Task 4: `?hotflow` gate — M87\* inside 2σ, Sgr A\* reported, robustness on the GPU

**Files:**
- Modify: `src/physics/hot-flow-image.ts` (`ringDiameterUas`: separable blur, same result)
- Modify: `src/test/hotflow.browser.ts`
- Test: `tests/hot-flow.test.ts` (separable-blur equivalence)

**Interfaces:**
- Consumes: `RING_ROBUST_UAS`, `HOTFLOW_TARGETS` (Task 3).
- Produces: `HotFlowRow` gains `ringBlurUas: number`.

- [ ] **Step 1: Write the failing test** (the 512² GPU image makes the current 2-D blur ~10⁹ operations; a separable blur is the same Gaussian with the same per-axis edge normalisation, so its result must match the 2-D one). In `tests/hot-flow.test.ts`, inside `describe("ring diameter measure (hot-flow-image.ts)"`, add:

```ts
  it("the blur is separable: same ring as the direct 2-D Gaussian, edges included", () => {
    const N = 64, half = 13, I = new Float64Array(N * N);
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const x = -half + ((2 * half) / N) * (i + 0.5), y = -half + ((2 * half) / N) * (j + 0.5), b = Math.hypot(x, y);
      I[j * N + i] = Math.exp(-((b - 5) ** 2) / 0.5) + 0.3 * Math.exp(-((b - 9) ** 2) / 2) * (1 + 0.5 * Math.sin(3 * Math.atan2(y, x)));
    }
    const direct = (blur: number, uasPerM: number) => { // the previous 2-D implementation, inline
      const px = (2 * half) / N, sig = blur / 2.3548 / uasPerM, R = Math.ceil((3 * sig) / px), img = new Float64Array(N * N);
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { let v = 0, w = 0;
        for (let dj = -R; dj <= R; dj++) for (let di = -R; di <= R; di++) { const ii = i + di, jj = j + dj; if (ii < 0 || jj < 0 || ii >= N || jj >= N) continue;
          const ww = Math.exp(-((di * px) ** 2 + (dj * px) ** 2) / (2 * sig * sig)); v += ww * I[jj * N + ii]; w += ww; } img[j * N + i] = v / w; }
      return ringDiameterUas(img, N, half, uasPerM);
    };
    for (const u of [3.82, 5.18]) expect(ringDiameterUas(I, N, half, u, 15)).toBeCloseTo(direct(15, u), 9);
  });
```

- [ ] **Step 2: Run it.**

Run: `npx vitest run tests/hot-flow.test.ts -t separable`
Expected: PASS already (the current code is the 2-D blur). This pins the result before the refactor; it must still pass after Step 3.

- [ ] **Step 3: Make the blur separable.** In `ringDiameterUas`, replace the `if (blurUas > 0) { ... }` block with:

```ts
  if (blurUas > 0) { // separable Gaussian, normalised per axis over the pixels inside the frame (= the 2-D sum)
    const sig = blurUas / 2.3548 / uasPerM, R = Math.ceil((3 * sig) / px), k = new Float64Array(2 * R + 1);
    for (let d = -R; d <= R; d++) k[d + R] = Math.exp(-((d * px) ** 2) / (2 * sig * sig));
    const tmp = new Float64Array(N * N); img = new Float64Array(N * N);
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { let v = 0, w = 0;
      for (let d = -R; d <= R; d++) { const ii = i + d; if (ii < 0 || ii >= N) continue; v += k[d + R] * I[j * N + ii]; w += k[d + R]; } tmp[j * N + i] = v / w; }
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { let v = 0, w = 0;
      for (let d = -R; d <= R; d++) { const jj = j + d; if (jj < 0 || jj >= N) continue; v += k[d + R] * tmp[jj * N + i]; w += k[d + R]; } img[j * N + i] = v / w; }
  }
```

Run: `npx vitest run tests/hot-flow.test.ts`
Expected: PASS (including "separable").

- [ ] **Step 4: Update the gate.** In `src/test/hotflow.browser.ts`:
  - Replace the header comment's PASS sentence with: `PASS iff M87*'s ring is within 2 sigma of the EHT's 42 uas (spec 2026-10-08; Sgr A*'s EHT comparison is reported) and, for both objects, the flux is within 5 % of the measured value its n0 was fitted to, the ring within 0.5 uas of the CPU twin's, and the ring moves by at most RING_ROBUST_UAS under a 15 uas blur.`
  - Import `RING_ROBUST_UAS` from `../physics/hot-flow-image`.
  - `export interface HotFlowRow { name: string; jy: number; ringUas: number; ringBlurUas: number; peakTb: number }`
  - In `measure`, return `ringBlurUas: ringDiameterUas(I, N, HALF, geom(s.preset!).uasPerM, 15)` alongside `ringUas`.
  - In `runHotFlow`, replace the `twin`/`sgOk`/`m87Ok`/`ehtOk` lines and the first two result lines with:

```ts
  const twin = (x: HotFlowRow, t: typeof T | typeof M) => Math.abs(x.jy / t.jy - 1) < 0.05 && Math.abs(x.ringUas - t.cpuRingUas) < 0.5;
  const robust = (x: HotFlowRow) => Math.abs(x.ringUas - x.ringBlurUas) <= RING_ROBUST_UAS;
  const sgOk = twin(a, T) && robust(a), m87Ok = twin(b, M) && robust(b), ehtOk = Math.abs(b.ringUas - M.ringUas) < 2 * M.ringErr;
  const sgEht = Math.abs(a.ringUas - T.ringUas) / T.ringErr;
```

```ts
    `${row(a)} (twin: ${T.jy} Jy +- 5 %, CPU ring ${T.cpuRingUas} +- 0.5, blur 15 ${a.ringBlurUas.toFixed(1)} ${sgOk ? "ok" : "FAILED"}; EHT ${T.ringUas} +- ${T.ringErr}: ${sgEht.toFixed(1)} sigma, reported)`,
    `${row(b)} (twin: ${M.jy} Jy +- 5 %, CPU ring ${M.cpuRingUas} +- 0.5, blur 15 ${b.ringBlurUas.toFixed(1)} ${m87Ok ? "ok" : "FAILED"}; EHT ${M.ringUas} +- ${M.ringErr} ${ehtOk ? "ok" : "FAILED"})`,
```

- [ ] **Step 5: Run the GPU gate** (dev server on).

Run: `npm run verify:gpu` is long; for this task run only the route: open `http://localhost:5173/?hotflow` headless with the pattern of `scripts/scratch-acc.mjs` (copy it to `scripts/scratch-hotflow.mjs` if absent, changing the URL to `/?hotflow` and the wait regex to `/HOTFLOW (PASS|FAIL)/`), then `node scripts/scratch-hotflow.mjs`.
Expected: `HOTFLOW PASS`; M87\* ring within 0.5 µas of `cpuRingUas` and inside 36–48; both blur deltas ≤ 3; hotspot rows ok.
If M87\* is outside 2σ on the GPU: STOP and report (spec §5).

- [ ] **Step 6: Commit.**

```bash
git add src/physics/hot-flow-image.ts src/test/hotflow.browser.ts tests/hot-flow.test.ts
git commit -m "?hotflow: M87* gated inside the EHT's 2 sigma, Sgr A* reported, rings robust to a 15 uas blur (separable blur, same result)"
```

---

### Task 5: `?accuracy` for the new emission

**Files:**
- Modify: `scripts/build-accuracy-ref.ts` (rebuild selected scenes, keep the rest)
- Modify: `src/test/accuracy-ref.json` (regenerated mm rows)
- Modify: `src/test/accuracy.browser.ts` (frozen old-renderer bounds for the mm scenes)

**Interfaces:**
- Consumes: Task 3's `HOTFLOW_N0` (the builder's `flowN0`).
- Produces: `MM_OLD` bounds in `accuracy.browser.ts`.

- [ ] **Step 1: Let the builder rebuild only some scenes.** In `scripts/build-accuracy-ref.ts` change `import { writeFileSync } from "node:fs";` to `import { readFileSync, writeFileSync } from "node:fs";`, then directly before `for (const sc of ACC_SCENES) {` add:

```ts
// ONLY=sgra-mm,garg-mm rebuilds those scenes and keeps every other scene of the existing fixture as it is.
const ONLY = process.env.ONLY?.split(",");
const prev = ONLY ? JSON.parse(readFileSync("src/test/accuracy-ref.json", "utf8")) : null;
```

and as the first statement inside that loop:

```ts
  if (ONLY && !ONLY.includes(sc.name)) { (out.scenes as unknown[]).push(prev.scenes.find((s: { name: string }) => s.name === sc.name)); continue; }
```

- [ ] **Step 2: Rebuild the mm rows** (dev server off; ~45 min, run alone).

Run: `ONLY=sgra-mm,garg-mm npx vite-node scripts/build-accuracy-ref.ts`
Expected: `sgra-mm: 1600 rays, unconverged <small>` and `garg-mm: ...`, then `wrote src/test/accuracy-ref.json`. Check with `git diff --stat src/test/accuracy-ref.json` that the file changed, and with a short node one-liner that the geometry scenes (`default`, `face-on`, `edge-on`, `schwarzschild`) are byte-identical to `git show HEAD:src/test/accuracy-ref.json`'s.

- [ ] **Step 3: Freeze the old renderer's mm statistics.** In `src/test/accuracy.browser.ts`, after `const median = ...`, add:

```ts
// 1.3 mm scenes (spec 2026-10-08 MAD R-beta §4): the emission changed after the pre-Mino renderer was removed, so its
// recorded mm entries no longer describe this emission. Its statistics against the then-current reference (?accuracy
// 2026-10-08: pixels above floor and worst error / floor) are the bounds instead of a per-pixel comparison.
const MM_OLD: Record<string, { over: number; worst: number }> = { "sgra-mm": { over: 261, worst: 38.39 }, "garg-mm": { over: 235, worst: 19.27 } };
```

In `runAccuracy`'s per-pixel loop, replace

```ts
        if (eo && Number.isFinite(eo[q])) ratios[q].push({ k, n: en[q] / floor, o: eo[q] / floor });
```

with

```ts
        if (q === "I" && MM_OLD[s.name]) ratios[q].push({ k, n: en[q] / floor, o: NaN });
        else if (eo && Number.isFinite(eo[q])) ratios[q].push({ k, n: en[q] / floor, o: eo[q] / floor });
```

and in the per-quantity verdict loop, directly after `if (maxN === 0 && maxO === 0) continue; ...`, add (the `maxO` there is computed with `Math.max` over `x.o` and is NaN for these rows, which is why the frozen branch runs first):

```ts
      const frozen = q === "I" ? MM_OLD[s.name] : undefined;
      if (frozen) {
        if (overN > frozen.over) { worse[q]++; examples.push(`${q}: ${overN} px above floor > old ${frozen.over}`); }
        if (maxN > frozen.worst) { worse[q]++; examples.push(`${q}: worst ${maxN.toFixed(2)}x floor > old ${frozen.worst}x`); }
        verdicts.push(`${q} >floor ${overN}/${frozen.over} worst ${maxN.toFixed(2)}/${frozen.worst}x (old frozen)`);
        continue;
      }
```

Move the line `if (maxN === 0 && maxO === 0) continue;` so it reads `if (maxN === 0 && !(maxO > 0)) continue;` (NaN-safe).

- [ ] **Step 4: Run the gate** (dev server on).

Run: `node scripts/scratch-acc.mjs`
Expected: `ACCURACY PASS`; sgra-mm and garg-mm lines show `(old frozen)` with counts at or below 261 / 235 and worst at or below 38.39 / 19.27; the four geometry scenes unchanged from accuracy8 (0 worse).

- [ ] **Step 5: Type check, test, commit.**

Run: `npx tsc --noEmit -p . && npx vitest run tests/hot-flow.test.ts tests/jet.test.ts`
Expected: exit 0, PASS.

```bash
git add scripts/build-accuracy-ref.ts src/test/accuracy-ref.json src/test/accuracy.browser.ts
git commit -m "?accuracy: mm reference rows rebuilt for the R-beta emission; the old renderer's mm statistics become fixed bounds"
```

---

### Task 6: Golden, full gates, docs, merge

**Files:**
- Modify: `src/test/golden.json` (re-recorded), `README.md`, `docs/ROADMAP.md`, `docs/specs/2026-10-08-mad-rbeta-hotflow-design.md` (status line)

- [ ] **Step 1: Re-record golden and run every GPU gate** (dev server on, nothing else running).

Run: `RECORD_GOLDEN=1 npm run verify:gpu`
Expected: every line `✓ PASS`. In `src/test/golden.json` only `sgra-mm`, `m87-mm`, `sgra-mm-hotspot` change; the five visible hashes (`default 6ca46ee2, jet-off 158728de, edge-on 235adb17, face-on-jet 7d2b6e3f, delay 890c85cb`) are unchanged. If a visible hash changed, STOP: the visible view must not evaluate the hot flow.
Then run `npm run verify:gpu` once more without RECORD to confirm `GOLDEN PASS` against the recorded file.

- [ ] **Step 2: Unit tests and build.**

Run: `npx vitest run && npm run build`
Expected: all pass (record the count), build clean.

- [ ] **Step 3: README.** In the section **Hot flow and the 1.3 mm view**, replace the n₀/ring table rows with the Task 3 and Task 4 numbers (n₀, flux CPU/GPU, ring CPU/GPU, 15 µas blur, EHT), mark Sgr A\* "reported" and M87\* "gated, 2σ". Replace the limitation sentence beginning `Limitations: M87*'s ring is 2σ small` with the R-β limitation from spec §5. Add a new paragraph after the Faster integrator section:

```markdown
**MAD electron heating for the hot flow (feat/mad-rbeta, 2026-10-08; spec `docs/specs/2026-10-08-mad-rbeta-hotflow-design.md`):**
<What changed: R-β electrons (virial ions, midplane-set field at β 1, R_high 160, R_low 1), why (only where the
electrons are hot moved M87*'s ring: the probe table of spec §2 in one sentence), the results (M87* <ring> µas inside
42 ± 3, Sgr A* <ring> µas reported at <x>σ), the recalibrated n₀ and A₀, the gates, and the limitations of spec §5.>
```

Write that paragraph with the measured values in place of every `<...>`; no placeholder may remain. Update **Current gates** with this branch's results (test count, parity, golden mm hashes, hotflow numbers, accuracy).

- [ ] **Step 4: Roadmap and spec status.** In `docs/ROADMAP.md`, in the Hot flow item's follow-ups, replace "MAD compression for M87*'s ring" with "~~MAD compression for M87*'s ring~~ done 2026-10-08 (R-β electrons, README)". In the spec, set `**Status:** implemented (plan docs/plans/2026-10-08-mad-rbeta-hotflow.md, 2026-10-08)` and list any departures.

- [ ] **Step 5: Commit, merge, push** (standing rule: merge to main and push when gates pass; deploy only on request).

```bash
git add src/test/golden.json README.md docs/ROADMAP.md docs/specs/2026-10-08-mad-rbeta-hotflow-design.md
git commit -m "Golden re-recorded (mm only), README and roadmap: MAD R-beta electrons put M87*'s ring inside the EHT's 2 sigma"
git checkout main && git merge --ff-only feat/mad-rbeta && git push origin main feat/mad-rbeta
```

Expected: fast-forward; `origin/main` at the new head.
