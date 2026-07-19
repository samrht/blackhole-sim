# Tier 2B Relativistic Jet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bipolar, relativistically-beamed, limb-brightened, animated jet — an optically-thin emissive volume along the spin axis — to the existing Tier 1 + Tier 2A WebGPU Kerr renderer.

**Architecture:** The jet is integrated *along* each backward-traced geodesic (optically thin, additive) by piggybacking on the ray-march steps the integrator already takes — no new rays. A pure-TS reference (`src/physics/jet.ts`) is mirrored bit-closely by WGSL twins in `raytrace.wgsl`; a standalone `jet-parity.wgsl` proves CPU↔GPU agreement. The one-sided look comes from per-step relativistic Doppler beaming. With the jet off the render is bit-for-bit identical to Tier 2A.

**Tech Stack:** TypeScript, WebGPU/WGSL compute, Vite, Vitest, headless-WebGPU verification (`scripts/verify-gpu.mjs`).

## Global Constraints

- **Features-off invariant:** with `jetStrength = 0` the render MUST be bit-for-bit identical to Tier 2A (and Tier 1 with all features off). `jetEmission(...)` returns exactly `0.0` on the `jetStrength == 0` early-out, and the composite `color = terminationColor + jetStrength * jetAccum` reduces to `terminationColor` with no FP perturbation.
- **Append-only uniforms:** extend the uniform buffer only by appending; never reorder. Current size 80 bytes → new size 96 bytes (append 4× f32).
- **One noise basis:** the jet reuses the Tier 2A value noise (`vnoise`/`ihash` in `emission.ts`; `vnoiseE`/`ihashE` in `raytrace.wgsl`) — do NOT introduce a new hash.
- **Shared design constants (identical in `jet.ts` and every WGSL twin):**
  `rho0=0.6`, `slope=0.7`, `qPeak=0.8`, `wWall=0.22`, `zBase=2.0`, `kz=0.35`, `vKnot=6.0`, `pBeam=3.5`, `turbAmpJet=0.35`, `knotSeed=17.0`, `JET_GAIN=0.06`, `JET_CEIL=8.0`, `JET_TINT=(0.55,0.78,1.0)`.
- **Target hardware:** RTX 3050 laptop; must stay real-time interactive (~15–30 fps). Added per-step work sits behind the `jetStrength>0` early-out.
- **Platform:** dev/verify commands run in PowerShell; the dev server is `npm run dev` on port 5173; headless GPU checks are `BASE=http://localhost:5173 npm run verify:gpu`.
- **No Co-Authored-By trailer** in commits (user preference).

---

### Task 1: Jet physics core (pure TS) + unit tests

**Files:**
- Modify: `src/physics/emission.ts` (export `vnoise` so the jet reuses the one noise basis)
- Create: `src/physics/jet.ts`
- Test: `tests/jet.test.ts`

**Interfaces:**
- Consumes: `vnoise(x, y)` from `src/physics/emission.ts`.
- Produces:
  - `export const JET` — the shared constants object (see code).
  - `export function funnelEdge(z: number): number`
  - `export function wallProfile(rho: number, z: number): number`
  - `export function lengthFalloff(z: number, zMax: number): number`
  - `export function knots(z: number, t: number, timeScale: number, jetKnots: number): number`
  - `export function dopplerBoost(mu: number, gamma: number): number`
  - `export function jetEmission(r: number, th: number, t: number, timeScale: number, jetStrength: number, jetLength: number, jetKnots: number): number`

- [ ] **Step 1: Export `vnoise` from `emission.ts`**

In `src/physics/emission.ts`, change the `vnoise` declaration (currently line 24) from `function vnoise(` to `export function vnoise(`. Leave everything else unchanged.

```ts
export function vnoise(x: number, y: number): number {
```

- [ ] **Step 2: Write the failing test**

Create `tests/jet.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import {
  JET, funnelEdge, wallProfile, lengthFalloff, knots, dopplerBoost, jetEmission,
} from "../src/physics/jet";

describe("jet geometry", () => {
  it("funnel widens with height (parabolic)", () => {
    expect(funnelEdge(0)).toBeCloseTo(JET.rho0, 12);
    expect(funnelEdge(4)).toBeCloseTo(JET.rho0 + JET.slope * 2, 12); // sqrt(4)=2
    expect(funnelEdge(16)).toBeGreaterThan(funnelEdge(4));
  });

  it("wall profile peaks at q = qPeak (limb-brightened, hollow)", () => {
    const z = 9; const edge = funnelEdge(z);
    const atPeak = wallProfile(JET.qPeak * edge, z);
    const atAxis = wallProfile(0.0, z);
    const outside = wallProfile(1.3 * edge, z);
    expect(atPeak).toBeCloseTo(1.0, 6);   // gaussian peak == 1
    expect(atAxis).toBeLessThan(atPeak);  // dimmer on the axis (hollow tube)
    expect(outside).toBe(0);              // nothing beyond the wall
  });
});

describe("jet beaming", () => {
  it("Doppler boost is >1 approaching, collapses receding, monotonic in mu", () => {
    const g = 5;
    expect(dopplerBoost(0.9, g)).toBeGreaterThan(1);       // toward observer -> boosted
    expect(dopplerBoost(-0.9, g)).toBeLessThan(0.05);      // away -> counter-jet vanishes
    expect(dopplerBoost(0.9, g)).toBeGreaterThan(dopplerBoost(0.3, g));
    expect(dopplerBoost(0.3, g)).toBeGreaterThan(dopplerBoost(-0.3, g));
  });
});

describe("jet living emission field", () => {
  it("knots form a traveling wave (advancing t shifts the pattern)", () => {
    const a = knots(10, 0.0, 1, 0.7);
    const b = knots(10, 0.5, 1, 0.7);
    expect(a).not.toBeCloseTo(b, 6); // time changes the local knot brightness
  });

  it("emission is exactly 0 when jetStrength = 0 (features-off invariant)", () => {
    expect(jetEmission(6, 0.15, 3.2, 1, 0, 60, 0.7)).toBe(0);
  });

  it("emission is 0 outside the axial band and inside the funnel band it is positive", () => {
    const thAxis = 0.12;                 // near the pole -> inside a funnel
    const rIn = 8;
    expect(jetEmission(rIn, thAxis, 0, 1, 1, 60, 0.7)).toBeGreaterThan(0);
    expect(jetEmission(1.5, thAxis, 0, 1, 1, 60, 0.7)).toBe(0); // below zBase launch
    expect(jetEmission(400, thAxis, 0, 1, 1, 60, 0.7)).toBe(0); // beyond jetLength
    expect(jetEmission(8, Math.PI / 2, 0, 1, 1, 60, 0.7)).toBe(0); // equatorial: outside funnel
  });
});
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `npm test -- jet`
Expected: FAIL — `Cannot find module '../src/physics/jet'`.

- [ ] **Step 4: Write `src/physics/jet.ts`**

```ts
// Phenomenological relativistic jet (Tier 2B). Pure functions, no DOM/GPU.
// Mirrored in WGSL: raytrace.wgsl (render) and jet-parity.wgsl (parity test).
// Reuses the Tier 2A value-noise basis (emission.vnoise) so there is one shared noise impl.
import { vnoise } from "./emission";

/** Shared design constants. The WGSL twins hardcode these exact values. */
export const JET = {
  rho0: 0.6, slope: 0.7,      // funnel throat radius (M) and parabolic flare (M^1/2)
  qPeak: 0.8, wWall: 0.22,    // limb-brightening: wall peak position and width (in q units)
  zBase: 2.0,                 // launch height above the pole (M); below this = no jet
  kz: 0.35, vKnot: 6.0,       // knot spatial frequency and outward pattern speed (M / sim-s)
  pBeam: 3.5,                 // beaming exponent (3 + spectral index)
  turbAmpJet: 0.35,           // small cross-funnel churn
  knotSeed: 17.0,             // fixed 2nd-axis coordinate for the 1-D knot noise
  gain: 0.06,                 // per-dl emissivity -> radiance scale
  ceil: 8.0,                  // clamp on accumulated jet radiance (anti-blowout)
} as const;

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

/** Parabolic funnel wall radius at signed axial height z. */
export function funnelEdge(z: number): number {
  return JET.rho0 + JET.slope * Math.sqrt(Math.abs(z));
}

/** Limb-brightened wall profile: gaussian peaked at q = qPeak, zero beyond the wall. */
export function wallProfile(rho: number, z: number): number {
  const q = rho / funnelEdge(z);
  if (q > 1.2) return 0;
  const d = q - JET.qPeak;
  return Math.exp(-(d * d) / (2 * JET.wWall * JET.wWall));
}

/** Axial soft-gate (fade in above zBase, fade out near zMax) times 1/z-style falloff. */
export function lengthFalloff(z: number, zMax: number): number {
  const az = Math.abs(z);
  const fadeIn = smoothstep(JET.zBase, JET.zBase + 2, az);
  const fadeOut = 1 - smoothstep(zMax * 0.7, zMax, az);
  const decay = JET.zBase / Math.max(az, JET.zBase);
  return fadeIn * fadeOut * decay;
}

/** Traveling-wave knots: blobs of brightness marching outward as t advances. */
export function knots(z: number, t: number, timeScale: number, jetKnots: number): number {
  const phase = JET.kz * Math.abs(z) - JET.vKnot * t * timeScale;
  return 1 + jetKnots * (vnoise(phase, JET.knotSeed) - 0.5) * 2;
}

/** Relativistic Doppler boost of emissivity. mu = cos(angle) of emitter outflow toward observer. */
export function dopplerBoost(mu: number, gamma: number): number {
  const beta = Math.sqrt(Math.max(0, 1 - 1 / (gamma * gamma)));
  const delta = 1 / (gamma * (1 - beta * mu));
  return Math.pow(delta, JET.pBeam);
}

/** Scalar jet emissivity (no beaming). Exactly 0 when jetStrength=0, below zBase, beyond zMax,
 *  or outside the funnel wall. Beaming (dopplerBoost) is applied separately at the ray step. */
export function jetEmission(
  r: number, th: number, t: number, timeScale: number,
  jetStrength: number, jetLength: number, jetKnots: number,
): number {
  if (jetStrength === 0) return 0;
  const z = r * Math.cos(th);
  const az = Math.abs(z);
  if (az < JET.zBase || az > jetLength) return 0;
  const rho = r * Math.sin(th);
  const w = wallProfile(rho, z);
  if (w <= 0) return 0;
  const turb = 1 + JET.turbAmpJet * (vnoise(Math.log(1 + rho), JET.kz * z) - 0.5) * 2;
  return Math.max(0, w * lengthFalloff(z, jetLength) * knots(z, t, timeScale, jetKnots) * turb);
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `npm test -- jet`
Expected: PASS — all cases in `tests/jet.test.ts`.

- [ ] **Step 6: Commit**

```bash
git add src/physics/emission.ts src/physics/jet.ts tests/jet.test.ts
git commit -m "feat(physics): phenomenological relativistic jet emission + beaming (Tier 2B)"
```

---

### Task 2: Extend the uniform buffer (append jet fields)

**Files:**
- Modify: `src/render/uniforms.ts`
- Modify: `src/main.ts` (add jet fields to the render-loop `UniformValues` literal — jet OFF for now)
- Modify: `src/test/shadow.browser.ts` (add jet fields to its `UniformValues` literal — jet OFF)
- Test: `tests/uniforms.test.ts` (extend)

**Interfaces:**
- Consumes: existing `UniformValues`, `packUniforms`.
- Produces: `UniformValues` gains `jetStrength, jetGamma, jetLength, jetKnots: number`; `UNIFORM_FLOATS = 19`, `UNIFORM_SIZE = 96`; packed at `f[19..22]`.

- [ ] **Step 1: Update the existing test to expect 96 bytes + assert the jet offsets**

`tests/uniforms.test.ts` currently asserts `UNIFORM_SIZE === 80` and builds an inline `UniformValues` literal with no jet fields. Replace the whole test body so it (a) expects 96, (b) adds the four jet fields to the literal, and (c) asserts their byte offsets (f[19]=76, f[20]=80, f[21]=84, f[22]=88). Replace lines 5-21 (the single `it(...)` block) with:
```ts
  it("is 96 bytes and packs all fields (incl. Tier 2B jet) at the expected offsets", () => {
    expect(UNIFORM_SIZE).toBe(96);
    const u: UniformValues = {
      resW: 100, resH: 50, a: 0.9, incl: 1.2, rObs: 1000, fovScale: 14, rIn: 5, rOut: 40,
      Tpeak: 3e4, exposure: 1.6, time: 7, frame: 3, reset: 0, maxSteps: 1200,
      blend: 0.15, timeScale: 2, turbAmp: 0.6, breatheAmp: 0.1, nSpots: 4,
      jetStrength: 1.0, jetGamma: 5.0, jetLength: 60.0, jetKnots: 0.7,
    };
    const dv = new DataView(packUniforms(u));
    expect(dv.getFloat32(0, true)).toBeCloseTo(100);   // resW
    expect(dv.getFloat32(40, true)).toBeCloseTo(7);     // time (index 10)
    expect(dv.getUint32(44, true)).toBe(3);             // frame (index 11)
    expect(dv.getFloat32(56, true)).toBeCloseTo(0.15);  // blend (index 14)
    expect(dv.getUint32(72, true)).toBe(4);             // nSpots (index 18)
    expect(dv.getFloat32(76, true)).toBeCloseTo(1.0);   // jetStrength (index 19)
    expect(dv.getFloat32(80, true)).toBeCloseTo(5.0);   // jetGamma (index 20)
    expect(dv.getFloat32(84, true)).toBeCloseTo(60.0);  // jetLength (index 21)
    expect(dv.getFloat32(88, true)).toBeCloseTo(0.7);   // jetKnots (index 22)
  });
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npm test -- uniforms`
Expected: FAIL — `UNIFORM_SIZE` is 80 (not 96), and TS errors on the unknown `jetStrength/...` fields.

- [ ] **Step 3: Update `src/render/uniforms.ts`**

Replace the header comment, interface, size constants, and `packUniforms` body to append the four jet floats. The full file becomes:
```ts
// Layout MUST match the `Uniforms` struct in raytrace.wgsl (4-byte scalars, vec2 first).
// floats: resW,resH,a,incl,rObs,fovScale,rIn,rOut,Tpeak,exposure,time (11)
//         + blend,timeScale,turbAmp,breatheAmp (4)                     -> 15
//         + jetStrength,jetGamma,jetLength,jetKnots (4)                -> 19 floats
// uint:   frame,reset,maxSteps (3) + nSpots (1)                        -> 4 uints
export interface UniformValues {
  resW: number; resH: number; a: number; incl: number; rObs: number; fovScale: number;
  rIn: number; rOut: number; Tpeak: number; exposure: number; time: number;
  frame: number; reset: number; maxSteps: number;
  blend: number; timeScale: number; turbAmp: number; breatheAmp: number; nSpots: number;
  jetStrength: number; jetGamma: number; jetLength: number; jetKnots: number;
}
export const UNIFORM_FLOATS = 19, UNIFORM_UINTS = 4;
export const UNIFORM_SIZE = Math.ceil((UNIFORM_FLOATS + UNIFORM_UINTS) / 4) * 16; // -> 96 bytes

export function packUniforms(u: UniformValues): ArrayBuffer {
  const buf = new ArrayBuffer(UNIFORM_SIZE);
  const f = new Float32Array(buf), i = new Uint32Array(buf);
  f[0] = u.resW; f[1] = u.resH; f[2] = u.a; f[3] = u.incl;
  f[4] = u.rObs; f[5] = u.fovScale; f[6] = u.rIn; f[7] = u.rOut;
  f[8] = u.Tpeak; f[9] = u.exposure; f[10] = u.time;
  i[11] = u.frame; i[12] = u.reset; i[13] = u.maxSteps;
  f[14] = u.blend; f[15] = u.timeScale; f[16] = u.turbAmp; f[17] = u.breatheAmp;
  i[18] = u.nSpots;
  f[19] = u.jetStrength; f[20] = u.jetGamma; f[21] = u.jetLength; f[22] = u.jetKnots;
  return buf;
}
```

- [ ] **Step 4: Add jet-OFF fields to `src/main.ts`'s render-loop literal**

In `src/main.ts`, in the `loop` function's `const u: UniformValues = {...}` literal, append to the last line (currently `blend, timeScale: state.timeScale, turbAmp: state.turbAmp, breatheAmp: state.breatheAmp, nSpots: baseSpots.length,`):
```ts
        jetStrength: 0, jetGamma: 5, jetLength: 60, jetKnots: 0.7,
```
(Jet stays OFF — `jetStrength: 0` — until Task 5 wires the real state, so the render is unchanged.)

- [ ] **Step 5: Add jet-OFF fields to `src/test/shadow.browser.ts`'s literal**

In `src/test/shadow.browser.ts`, the `const u: UniformValues = {...}` literal (ends with `blend: 1, timeScale: 1, turbAmp: 0, breatheAmp: 0, nSpots: 0 }`): add the jet fields so it type-checks and keeps the jet off:
```ts
    blend: 1, timeScale: 1, turbAmp: 0, breatheAmp: 0, nSpots: 0,
    jetStrength: 0, jetGamma: 5, jetLength: 60, jetKnots: 0.7 };
```

- [ ] **Step 6: Run tests + build to verify green**

Run: `npm test -- uniforms`
Expected: PASS (byteLength 96, offsets correct).
Run: `npm run build`
Expected: succeeds (all `UniformValues` literals type-check).

- [ ] **Step 7: Commit**

```bash
git add src/render/uniforms.ts src/main.ts src/test/shadow.browser.ts tests/uniforms.test.ts
git commit -m "feat(render): append Tier 2B jet uniforms (jetStrength, jetGamma, jetLength, jetKnots)"
```

---

### Task 3: Wire jet emission into the compute shader (jet still OFF)

**Files:**
- Modify: `src/render/raytrace.wgsl`

**Interfaces:**
- Consumes: uniform fields `jetStrength, jetGamma, jetLength, jetKnots` (Task 2); existing `vnoiseE`, `smoothE`, `ihashE`, `PI`.
- Produces: WGSL twins `funnelEdgeJ`, `wallJ`, `lengthFalloffJ`, `knotsJ`, `boostJ`, `jetEmissionJ`; per-step jet accumulation + final additive composite. Because `main.ts` still sends `jetStrength=0` (Task 2), the rendered image is unchanged — this is the regression gate.

- [ ] **Step 1: Add the jet fields to the WGSL `Uniforms` struct**

In `src/render/raytrace.wgsl`, change the struct (lines 1-5) so the last line reads:
```wgsl
  blend: f32, timeScale: f32, turbAmp: f32, breatheAmp: f32, nSpots: u32,
  jetStrength: f32, jetGamma: f32, jetLength: f32, jetKnots: f32,
};
```

- [ ] **Step 2: Add the jet WGSL twins**

In `src/render/raytrace.wgsl`, immediately after `emissionFieldE(...)` (after line 144) add:
```wgsl
// --- Tier 2B jet (WGSL twin of src/physics/jet.ts) --------------------------------------------
const JET_QPEAK = 0.8;   const JET_WWALL = 0.22;
const JET_RHO0  = 0.6;   const JET_SLOPE = 0.7;
const JET_ZBASE = 2.0;   const JET_KZ    = 0.35;  const JET_VKNOT = 6.0;
const JET_PBEAM = 3.5;   const JET_TURB  = 0.35;  const JET_SEED  = 17.0;
const JET_GAIN  = 0.06;  const JET_CEIL  = 8.0;
const JET_TINT  = vec3<f32>(0.55, 0.78, 1.0);

fn smoothstepJ(a: f32, b: f32, x: f32) -> f32 {
  let t = clamp((x - a) / (b - a), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}
fn funnelEdgeJ(z: f32) -> f32 { return JET_RHO0 + JET_SLOPE * sqrt(abs(z)); }
fn wallJ(rho: f32, z: f32) -> f32 {
  let q = rho / funnelEdgeJ(z);
  if (q > 1.2) { return 0.0; }
  let d = q - JET_QPEAK;
  return exp(-(d * d) / (2.0 * JET_WWALL * JET_WWALL));
}
fn lengthFalloffJ(z: f32, zMax: f32) -> f32 {
  let az = abs(z);
  let fadeIn = smoothstepJ(JET_ZBASE, JET_ZBASE + 2.0, az);
  let fadeOut = 1.0 - smoothstepJ(zMax * 0.7, zMax, az);
  let decay = JET_ZBASE / max(az, JET_ZBASE);
  return fadeIn * fadeOut * decay;
}
fn knotsJ(z: f32, t: f32) -> f32 {
  let phase = JET_KZ * abs(z) - JET_VKNOT * t * U.timeScale;
  return 1.0 + U.jetKnots * (vnoiseE(phase, JET_SEED) - 0.5) * 2.0;
}
fn boostJ(mu: f32, gamma: f32) -> f32 {
  let beta = sqrt(max(0.0, 1.0 - 1.0 / (gamma * gamma)));
  let delta = 1.0 / (gamma * (1.0 - beta * mu));
  return pow(delta, JET_PBEAM);
}
// scalar emissivity (no beaming); exactly 0 when jet off / below zBase / beyond zMax / outside wall
fn jetEmissionJ(r: f32, th: f32, t: f32) -> f32 {
  if (U.jetStrength == 0.0) { return 0.0; }
  let z = r * cos(th);
  let az = abs(z);
  if (az < JET_ZBASE || az > U.jetLength) { return 0.0; }
  let rho = r * sin(th);
  let w = wallJ(rho, z);
  if (w <= 0.0) { return 0.0; }
  let turb = 1.0 + JET_TURB * (vnoiseE(log(1.0 + rho), JET_KZ * z) - 0.5) * 2.0;
  return max(0.0, w * lengthFalloffJ(z, U.jetLength) * knotsJ(z, t) * turb);
}
fn cartOf(x: vec4<f32>) -> vec3<f32> {
  let r = x.y; let th = x.z; let ph = x.w; let s = sin(th);
  return vec3<f32>(r * s * cos(ph), r * s * sin(ph), r * cos(th));
}
```

- [ ] **Step 3: Declare the jet accumulator before the march loop**

In `src/render/raytrace.wgsl`, just after `var color = vec3<f32>(0.0);` (line 169) add:
```wgsl
  var jetAccum = vec3<f32>(0.0); // optically-thin jet emission integrated along the ray
```

- [ ] **Step 4: Accumulate jet emission each step (before the disk break)**

In `src/render/raytrace.wgsl`, inside the `for` loop, immediately after `let sNew = rk4(s, a, dl);` (line 180) add:
```wgsl
    // Optically-thin jet: integrate emissivity * relativistic beaming along the ray. The disk
    // hit below still `break`s (opaque), so jet segments behind the disk/horizon are occluded.
    if (U.jetStrength > 0.0) {
      let jz = s.x.y * cos(s.x.z);
      let e = jetEmissionJ(s.x.y, s.x.z, U.time);
      if (e > 0.0) {
        let marchDir = normalize(cartOf(sNew.x) - cartOf(s.x)); // inward (camera -> hole)
        let axisSign = select(-1.0, 1.0, jz >= 0.0);
        let mu = -axisSign * marchDir.z;                        // emitter outflow toward observer
        jetAccum += JET_TINT * (e * JET_GAIN) * boostJ(mu, U.jetGamma) * dl;
      }
    }
```

- [ ] **Step 5: Composite the jet additively at the end**

In `src/render/raytrace.wgsl`, replace the final accumulation line (currently line 216, `accum[idx] = vec4<f32>(mix(accum[idx].rgb, color, U.blend), 1.0);`) with:
```wgsl
  // Additive optically-thin jet on top of whatever the ray terminated on (disk/starfield/shadow).
  let composited = color + U.jetStrength * min(jetAccum, vec3<f32>(JET_CEIL));
  accum[idx] = vec4<f32>(mix(accum[idx].rgb, composited, U.blend), 1.0);
```

- [ ] **Step 6: Verify features-off regression (jet still OFF via main.ts)**

Start the dev server if not running: `npm run dev` (serves on :5173).
Run: `BASE=http://localhost:5173 npm run verify:gpu`
Expected: `✓ PASS /?parity` (`maxRelErr=4.433e-8 over 8 cases`, unchanged) and `✓ PASS /?shadow` (`≈ 2.35 M`, unchanged). Because `jetStrength=0`, the interactive render is byte-identical to Tier 2A.

- [ ] **Step 7: Commit**

```bash
git add src/render/raytrace.wgsl
git commit -m "feat(render): jet emission twins + per-step beamed accumulation + composite (Tier 2B, off)"
```

---

### Task 4: GPU parity for the jet twins

**Files:**
- Create: `src/render/jet-parity.wgsl`
- Modify: `src/test/parity.browser.ts`

**Interfaces:**
- Consumes: `jetEmission`, `dopplerBoost`, `JET` from `src/physics/jet.ts`; the existing `device` in `runParity`.
- Produces: a third parity block; `runParity` still returns `{ maxErr, rows }` with `rows` increased by the jet cases.

- [ ] **Step 1: Create the standalone parity twin**

Create `src/render/jet-parity.wgsl` (self-contained: its own noise copy + jet functions with the fixed test parameters `jetStrength=1, jetLength=60, jetKnots=0.7, timeScale=1, gamma=5`):
```wgsl
// Standalone twin for CPU<->GPU parity of the Tier 2B jet. Input per case: vec4(r, th, t, mu).
// Output per case: vec4(emission, boost, 0, 0). Constants MUST match src/physics/jet.ts.
@group(0) @binding(0) var<storage, read> inp: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outp: array<vec4<f32>>;

const JET_QPEAK = 0.8; const JET_WWALL = 0.22; const JET_RHO0 = 0.6; const JET_SLOPE = 0.7;
const JET_ZBASE = 2.0; const JET_KZ = 0.35; const JET_VKNOT = 6.0; const JET_PBEAM = 3.5;
const JET_TURB = 0.35; const JET_SEED = 17.0;
const P_JETLEN = 60.0; const P_KNOTS = 0.7; const P_TS = 1.0; const P_GAMMA = 5.0;

fn ihashJ(ix: i32, iy: i32) -> f32 {
  var n = u32(ix) * 1973u + u32(iy) * 9277u;
  n = (n ^ (n >> 15u)) * 2246822519u;
  n = (n ^ (n >> 13u)) * 3266489917u;
  return f32(n & 0xffffffu) / f32(0xffffffu);
}
fn smoothJ(t: f32) -> f32 { return t * t * (3.0 - 2.0 * t); }
fn vnoiseJ(x: f32, y: f32) -> f32 {
  let ix = i32(floor(x)); let iy = i32(floor(y));
  let fx = smoothJ(x - floor(x)); let fy = smoothJ(y - floor(y));
  let a00 = ihashJ(ix, iy); let a10 = ihashJ(ix + 1, iy);
  let a01 = ihashJ(ix, iy + 1); let a11 = ihashJ(ix + 1, iy + 1);
  return (a00 * (1.0 - fx) + a10 * fx) * (1.0 - fy) + (a01 * (1.0 - fx) + a11 * fx) * fy;
}
fn ss(a: f32, b: f32, x: f32) -> f32 { let t = clamp((x - a) / (b - a), 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }
fn edge(z: f32) -> f32 { return JET_RHO0 + JET_SLOPE * sqrt(abs(z)); }
fn wall(rho: f32, z: f32) -> f32 {
  let q = rho / edge(z); if (q > 1.2) { return 0.0; }
  let d = q - JET_QPEAK; return exp(-(d * d) / (2.0 * JET_WWALL * JET_WWALL));
}
fn falloff(z: f32) -> f32 {
  let az = abs(z);
  return ss(JET_ZBASE, JET_ZBASE + 2.0, az) * (1.0 - ss(P_JETLEN * 0.7, P_JETLEN, az)) * (JET_ZBASE / max(az, JET_ZBASE));
}
fn kn(z: f32, t: f32) -> f32 { let ph = JET_KZ * abs(z) - JET_VKNOT * t * P_TS; return 1.0 + P_KNOTS * (vnoiseJ(ph, JET_SEED) - 0.5) * 2.0; }
fn emission(r: f32, th: f32, t: f32) -> f32 {
  let z = r * cos(th); let az = abs(z);
  if (az < JET_ZBASE || az > P_JETLEN) { return 0.0; }
  let rho = r * sin(th); let w = wall(rho, z); if (w <= 0.0) { return 0.0; }
  let turb = 1.0 + JET_TURB * (vnoiseJ(log(1.0 + rho), JET_KZ * z) - 0.5) * 2.0;
  return max(0.0, w * falloff(z) * kn(z, t) * turb);
}
fn boost(mu: f32) -> f32 {
  let beta = sqrt(max(0.0, 1.0 - 1.0 / (P_GAMMA * P_GAMMA)));
  let delta = 1.0 / (P_GAMMA * (1.0 - beta * mu));
  return pow(delta, JET_PBEAM);
}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = inp[gid.x];
  outp[gid.x] = vec4<f32>(emission(c.x, c.y, c.z), boost(c.w), 0.0, 0.0);
}
```

- [ ] **Step 2: Add the failing jet-parity block to `parity.browser.ts`**

In `src/test/parity.browser.ts`, add the import at the top:
```ts
import { jetEmission, dopplerBoost } from "../physics/jet";
import jetParityWGSL from "../render/jet-parity.wgsl?raw";
```
Then, immediately before the final `return { maxErr, rows: ... };`, insert a jet block that reuses the existing `device` (mirror the turbulence block's buffer dance; note the 16-byte input/output stride):
```ts
  // --- jet parity (CPU jet.ts vs GPU jet-parity.wgsl) ---
  const jcases = [
    { r: 8,  th: 0.12, t: 0.0, mu: 0.9 },
    { r: 14, th: 0.20, t: 1.3, mu: 0.3 },
    { r: 20, th: 0.10, t: 2.7, mu: -0.6 },
    { r: 6,  th: 0.30, t: 0.5, mu: -0.9 },
  ];
  const jin = device.createBuffer({ size: jcases.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const jarr = new Float32Array(jcases.length * 4);
  jcases.forEach((c, i) => { jarr.set([c.r, c.th, c.t, c.mu], i * 4); });
  device.queue.writeBuffer(jin, 0, jarr);
  const jout = device.createBuffer({ size: jcases.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const jread = device.createBuffer({ size: jcases.length * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const jmod = device.createShaderModule({ code: jetParityWGSL });
  const jpipe = device.createComputePipeline({ layout: "auto", compute: { module: jmod, entryPoint: "main" } });
  const jbind = device.createBindGroup({ layout: jpipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: jin } }, { binding: 1, resource: { buffer: jout } }] });
  const jenc = device.createCommandEncoder();
  const jcp = jenc.beginComputePass(); jcp.setPipeline(jpipe); jcp.setBindGroup(0, jbind); jcp.dispatchWorkgroups(jcases.length); jcp.end();
  jenc.copyBufferToBuffer(jout, 0, jread, 0, jcases.length * 16);
  device.queue.submit([jenc.finish()]);
  await jread.mapAsync(GPUMapMode.READ);
  const jgpu = new Float32Array(jread.getMappedRange().slice(0));
  jcases.forEach((c, i) => {
    const cpuE = jetEmission(c.r, c.th, c.t, 1, 1, 60, 0.7);
    const cpuB = dopplerBoost(c.mu, 5);
    maxErr = Math.max(maxErr, Math.abs(jgpu[i * 4 + 0] - cpuE) / (1 + Math.abs(cpuE)));
    maxErr = Math.max(maxErr, Math.abs(jgpu[i * 4 + 1] - cpuB) / (1 + Math.abs(cpuB)));
  });
```
Change the final return to include the jet rows:
```ts
  return { maxErr, rows: cases.length + tcases.length + jcases.length };
```
Note: the parity block dispatches `jcases.length` workgroups (the jet twin uses `@workgroup_size(1)` and indexes by `gid.x`), unlike the single-workgroup metric/turbulence twins.

- [ ] **Step 3: Run parity to verify jet cases pass**

Ensure the dev server is running (`npm run dev`).
Run: `BASE=http://localhost:5173 npm run verify:gpu`
Expected: `✓ PASS /?parity` with `over 12 cases` (4 metric + 4 turbulence + 4 jet) and `maxRelErr < 1e-3`.

- [ ] **Step 4: Commit**

```bash
git add src/render/jet-parity.wgsl src/test/parity.browser.ts
git commit -m "test(render): GPU-vs-CPU parity for the jet emission + beaming twins (Tier 2B)"
```

---

### Task 5: Turn the jet ON — drive it from `main.ts`

**Files:**
- Modify: `src/main.ts`

**Interfaces:**
- Consumes: uniform jet fields (Task 2); the shader jet path (Task 3).
- Produces: module-level `state` gains `jetStrength, jetGamma, jetLength, jetKnots`; the render loop feeds real jet values (jet visibly ON).

- [ ] **Step 1: Add jet fields to the interactive `state`**

In `src/main.ts`, extend the `const state = {...}` object (the Tier 2A line beginning `const state = { a: 0.9, incl: 72, exposure: 1.6, timeScale: 1.0, ...`) by appending the jet fields before the closing brace:
```ts
  const state = { a: 0.9, incl: 72, exposure: 1.6, timeScale: 1.0, turbAmp: 0.6, breatheAmp: 0.0, playing: true, flareScale: 1.0,
    jetStrength: 1.0, jetGamma: 5.0, jetLength: 60.0, jetKnots: 0.7 };
```

- [ ] **Step 2: Feed the real jet state into the uniform each frame**

In `src/main.ts`, in the `loop` function's `UniformValues` literal, replace the jet-OFF line added in Task 2 (`jetStrength: 0, jetGamma: 5, jetLength: 60, jetKnots: 0.7,`) with the live state:
```ts
        jetStrength: state.jetStrength, jetGamma: state.jetGamma,
        jetLength: state.jetLength, jetKnots: state.jetKnots,
```

- [ ] **Step 3: Build + manual visual check**

Run: `npm run build`
Expected: succeeds.
Run: `npm run dev`, open `http://localhost:5173/`. Expected (manual):
- A bright, blue-white jet arm rises from one pole; the opposite (counter-jet) arm is far dimmer or nearly invisible — the relativistic-beaming one-sidedness.
- The jet reads as an edge-brightened hollow funnel (limb-brightened rails), not a filled cone.
- Bright **knots propagate outward** along the jet over a few seconds.
- Dragging the canvas to change inclination swaps which arm dominates and the scene stays responsive.
- The disk still occludes the jet where the jet passes behind it.

- [ ] **Step 4: Verify the interactive still animates via headless capture (optional but recommended)**

With the dev server running:
Run: `BASE=http://localhost:5173 SHOT=jet.png npm run verify:gpu`
Expected: both routes still PASS (jet off in those routes); `jet.png` shows the one-sided jet over the disk.

- [ ] **Step 5: Commit**

```bash
git add src/main.ts
git commit -m "feat(app): enable the jet — feed live jet state into the renderer (Tier 2B)"
```

---

### Task 6: UI controls + README + final verification

**Files:**
- Modify: `index.html`
- Modify: `src/main.ts` (wire the new controls)
- Modify: `README.md`

**Interfaces:**
- Consumes: `state`, `$`, `reset` from Tier 2A; jet fields on `state` (Task 5).

- [ ] **Step 1: Add the control markup**

In `index.html`, after the Flares control block and before the Play/Pause button (i.e., after the `<div class="ctrl">…flare…</div>` block, before `<button id="playpause"...`), insert:
```html
    <div class="ctrl">
      <div class="row"><label>Jet</label><span class="val"><b id="jetv">1.0</b>×</span></div>
      <input id="jet" type="range" min="0" max="3" step="0.1" value="1">
    </div>
    <div class="ctrl">
      <div class="row"><label>Jet speed&nbsp;Γ</label><span class="val"><b id="jgv">5.0</b></span></div>
      <input id="jg" type="range" min="1.5" max="12" step="0.1" value="5">
    </div>
    <div class="ctrl">
      <div class="row"><label>Jet knots</label><span class="val"><b id="jkv">0.70</b></span></div>
      <input id="jk" type="range" min="0" max="1.5" step="0.05" value="0.7">
    </div>
```

- [ ] **Step 2: Wire the controls in `main.ts`**

In `src/main.ts`, after the Tier 2A control-wiring block (after the `playBtn.addEventListener(...)` block), add:
```ts
  const jet = $("jet") as HTMLInputElement, jg = $("jg") as HTMLInputElement, jk = $("jk") as HTMLInputElement;
  const jetv = $("jetv"), jgv = $("jgv"), jkv = $("jkv");

  jet.addEventListener("input", () => { state.jetStrength = +jet.value; jetv.textContent = state.jetStrength.toFixed(1); reset(); });
  jg.addEventListener("input", () => { state.jetGamma = +jg.value; jgv.textContent = state.jetGamma.toFixed(1); reset(); });
  jk.addEventListener("input", () => { state.jetKnots = +jk.value; jkv.textContent = state.jetKnots.toFixed(2); reset(); });
```

- [ ] **Step 3: Build + verify all controls**

Run: `npm run build`
Expected: succeeds.
Run: `npm run dev`, open `http://localhost:5173/` and confirm (manual):
- **Jet** slider: 0 removes the jet entirely (scene = Tier 2A); higher brightens it.
- **Jet speed Γ**: higher Γ → stronger beaming → counter-jet dims further / approaching jet tightens.
- **Jet knots**: 0 → smooth steady beam; higher → stronger propagating blobs.

- [ ] **Step 4: Full regression sweep**

Run: `npm test`
Expected: PASS — all suites including `jet`, `emission`, `uniforms`.
Run: `npm run build`
Expected: succeeds.
With the dev server running, open each route:
- `http://localhost:5173/?parity` → `PARITY PASS`, `maxRelErr < 1e-3`, `over 12 cases`.
- `http://localhost:5173/?shadow` → `SHADOW PASS (structural)` (unchanged; jet off in this route).

- [ ] **Step 5: Update the README status + commit**

In `README.md`, under `## Status`, add that Tier 2B (relativistic jet) shipped — a beamed, limb-brightened, animated bipolar jet — and note the next increment (or that the Tier 2 line is complete). Then:
```bash
git add index.html src/main.ts README.md
git commit -m "feat(ui): jet strength/speed/knots controls; document Tier 2B"
```

---

## Self-Review

**Spec coverage (§ = design doc sections):**
- §1 optically-thin, integrated-along-ray, no-new-rays → Task 3 Step 4 (per-step accumulation before the disk break), Step 5 (additive composite).
- §2 goals (bipolar funnel, limb-brightening, beaming one-sidedness, living knots, synchrotron palette, features-off invariant) → Task 1 (geometry/emission/beaming), Task 3 (twins + composite, `JET_TINT`), Task 5 (turn on), Global Constraints (invariant).
- §3 parabolic funnel + wall profile + axial gates → Task 1 `funnelEdge`/`wallProfile`/`lengthFalloff` (+ WGSL twins Task 3).
- §4 living emission field (lengthFalloff, knots, turb, clamp) → Task 1 `jetEmission`/`knots` (+ WGSL twin).
- §5 relativistic beaming (δ, `p_beam`, one-sided) → Task 1 `dopplerBoost`; Task 3 Step 4 (`mu` from marching direction + `boostJ`).
- §6 render integration (piggyback, composite, clamp, occlusion) → Task 3 Steps 3-5.
- §7 features-off invariant + gates → Task 1 (`=0` test), Task 3 Step 6 (`?parity`/`?shadow` unchanged), Task 6 Step 4.
- §8 uniforms append-only (4 f32, 96 bytes) → Task 2.
- §9 module plan (`jet.ts`, WGSL twin, `jet-parity.wgsl`, uniforms, main, index, tests) → Tasks 1-6.
- §10 testing (CPU unit, GPU parity, features-off, manual visual) → Task 1 (unit), Task 4 (parity), Task 3/6 (`?shadow`), Task 5 (visual).
- §11 risks (blowout clamp `JET_CEIL`, coarse upper-jet sampling accepted, shared noise, perf early-out) → Global Constraints + Task 3.

**Placeholder scan:** none — every code step contains complete code; every run step names the exact command and expected output.

**Type consistency:** `JET` constants object and the six exported jet functions have identical names/signatures across `jet.ts` (Task 1), the WGSL twins `funnelEdgeJ/wallJ/lengthFalloffJ/knotsJ/boostJ/jetEmissionJ` (Task 3), and `jet-parity.wgsl` (Task 4, with fixed test params). `UniformValues` field names `jetStrength/jetGamma/jetLength/jetKnots` are identical in `uniforms.ts`, `main.ts`, and `shadow.browser.ts`. Uniform struct field order in `raytrace.wgsl` matches `packUniforms` (f[19]=jetStrength, f[20]=jetGamma, f[21]=jetLength, f[22]=jetKnots). `runParity` return shape `{maxErr, rows}` preserved; `rows` grows to 12. `vnoise` is exported from `emission.ts` (Task 1 Step 1) and imported by `jet.ts`.
