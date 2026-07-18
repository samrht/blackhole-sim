# Tier 2A — Living Disk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the static Kerr accretion disk *move* — real-time differential rotation, azimuthal turbulence, and orbiting Doppler-beamed hot-spots — while staying interactive on an RTX 3050.

**Architecture:** Add a pure-TS phenomenological emission field `E(r, ψ)` sampled at the co-rotating pattern phase `ψ = φ_hit − Ω(r)·t·timeScale`, mirrored bit-closely in WGSL. It multiplies the *existing* beamed-blackbody disk radiance, so the current relativistic optics (geodesics, g-factor, LUTs) are untouched. Multi-frame progressive accumulation is generalized to a temporal exponential moving average (EMA) so an animating scene stays denoised; paused/static frames fall back to the exact Tier-1 running mean.

**Tech Stack:** TypeScript, WebGPU/WGSL compute, Vite, Vitest. Geometrized units (M = 1).

## Global Constraints

- **Geometrized units, M = 1** — all radii/times in units of M; `omegaKepler(r, a, prograde)` from `src/physics/orbits.ts` is the single source of Ω(r).
- **Features-off ≡ Tier 1, bit-for-bit.** With `turbAmp = 0`, `breatheAmp = 0`, `nSpots = 0`, `emissionField` MUST return exactly `1.0`, so the rendered image is identical to Tier 1. This is the strict regression gate (`?shadow` route).
- **Physics core stays pure.** `src/physics/*.ts` has zero DOM/GPU imports; new `emission.ts` follows this.
- **WGSL has no `#include`.** The repo already duplicates math between `raytrace.wgsl` and `parity.wgsl`; follow that pattern for the emission twins.
- **Uniform layout is append-only.** Add new fields to the END of the struct/packing so existing byte offsets (and the unchanged `present.wgsl`/`bloom.wgsl` structs, which are valid prefixes) keep working.
- **Integer hashing for the noise** so CPU (f64) and GPU (f32) agree: use `Math.imul(...) >>> 0` to mirror WGSL `u32` wraparound exactly; only interpolation arithmetic differs (well within parity tolerance `1e-3`).
- **Target ~15–30 fps** on the RTX 3050; `maxSteps` stays 1200 for the interactive loop.

## File Structure

- `src/physics/emission.ts` — **new.** Pure emission model: `patternPhase`, `turbulence`, `hotspotField`, `emissionField`, `HotSpot`, `T_BREATHE`.
- `tests/emission.test.ts` — **new.** Vitest unit tests for the above.
- `src/render/turb-parity.wgsl` — **new.** WGSL twin of `turbulence` for GPU parity.
- `src/render/uniforms.ts` — **modify.** Append `blend, timeScale, turbAmp, breatheAmp, nSpots`.
- `src/render/gpu.ts` — **modify.** Hot-spot storage buffer, binding 4, `uploadHotSpots`.
- `src/render/raytrace.wgsl` — **modify.** Emission WGSL twins, binding 4, apply `E` at disk hit, EMA accumulation.
- `src/render/present.wgsl` — **modify.** Drop the `/samples` divide (accum now holds normalized radiance).
- `src/test/parity.browser.ts` — **modify.** Add a turbulence GPU-vs-CPU parity pass.
- `src/test/shadow.browser.ts` — **modify.** Add features-off values to its `UniformValues` literal.
- `src/main.ts` — **modify.** Animation state, sim-time integration, blend selection, hot-spot upload.
- `index.html` — **modify.** Play/Pause + Motion/Turbulence/Flares controls.

---

### Task 1: Emission physics core (pure TS)

**Files:**
- Create: `src/physics/emission.ts`
- Test: `tests/emission.test.ts`

**Interfaces:**
- Consumes: `omegaKepler(r, a, prograde)` from `src/physics/orbits.ts`.
- Produces:
  - `interface HotSpot { r: number; psi: number; sigma: number; amp: number; }`
  - `const T_BREATHE = 2000`
  - `patternPhase(rHit, phiHit, t, timeScale, a): number`
  - `turbulence(logR, psi, octaves): number`
  - `hotspotField(rHit, psi, spots): number`
  - `emissionField(rHit, psi, t, turbAmp, breatheAmp, spots): number`

- [ ] **Step 1: Write the failing test**

`tests/emission.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { patternPhase, turbulence, hotspotField, emissionField, T_BREATHE, type HotSpot } from "../src/physics/emission";
import { omegaKepler } from "../src/physics/orbits";

const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * (1 + Math.abs(b));

describe("emission", () => {
  it("patternPhase = phi_hit at t=0, and advances by -Omega*t*timeScale", () => {
    expect(close(patternPhase(8, 1.3, 0, 1, 0.9), 1.3)).toBe(true);
    const om = omegaKepler(8, 0.9, true);
    expect(close(patternPhase(8, 1.3, 5, 2, 0.9), 1.3 - om * 5 * 2)).toBe(true);
  });

  it("inner annuli sweep faster than outer (differential rotation)", () => {
    const dInner = 0 - patternPhase(6, 0, 1, 1, 0.9);   // phase swept in unit time at r=6
    const dOuter = 0 - patternPhase(20, 0, 1, 1, 0.9);  // at r=20
    expect(dInner).toBeGreaterThan(dOuter);             // |Omega(6)| > |Omega(20)|
  });

  it("turbulence is deterministic and bounded to ~[0,1)", () => {
    const a = turbulence(Math.log(9), 1.0, 3);
    expect(turbulence(Math.log(9), 1.0, 3)).toBe(a);    // deterministic
    for (let p = 0; p < 6.28; p += 0.37) {
      const v = turbulence(Math.log(9), p, 3);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1.0);
    }
  });

  it("hotspotField peaks at the spot center and decays far away, and is periodic in psi", () => {
    const spots: HotSpot[] = [{ r: 10, psi: 1.0, sigma: 1.0, amp: 2.0 }];
    const peak = hotspotField(10, 1.0, spots);
    expect(close(peak, 2.0, 1e-6)).toBe(true);
    expect(hotspotField(10, 1.0 + 3, spots)).toBeLessThan(0.05);            // ~3 sigma away in arc
    expect(close(hotspotField(10, 1.0, spots), hotspotField(10, 1.0 + 2 * Math.PI, spots), 1e-6)).toBe(true);
  });

  it("emissionField reduces to exactly 1 when all features are off (Tier-1 regression gate)", () => {
    expect(emissionField(9, 2.0, 123.4, 0, 0, [])).toBe(1);
    expect(emissionField(15, -1.0, 5.0, 0, 0, [])).toBe(1);
  });

  it("emissionField is non-negative even with strong features", () => {
    const spots: HotSpot[] = [{ r: 8, psi: 0, sigma: 1.5, amp: 3 }];
    for (let p = 0; p < 6.28; p += 0.5) expect(emissionField(8, p, 10, 1.5, 0.9, spots)).toBeGreaterThanOrEqual(0);
  });

  it("T_BREATHE is the documented period constant", () => { expect(T_BREATHE).toBe(2000); });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- emission`
Expected: FAIL — cannot resolve `../src/physics/emission`.

- [ ] **Step 3: Write the implementation**

`src/physics/emission.ts`:
```ts
// Phenomenological time-varying disk emission (Tier 2A). Pure functions, no DOM/GPU.
// Mirrored in WGSL: raytrace.wgsl (render) and turb-parity.wgsl (parity test).
import { omegaKepler } from "./orbits";

export const T_BREATHE = 2000; // coordinate-time period (in M) of the optional slow "breathing"

export interface HotSpot { r: number; psi: number; sigma: number; amp: number; }

/** Co-rotating pattern phase. Matter at (r, phi) orbits at Omega(r), so a feature fixed in the
 *  co-rotating frame appears at psi = phi - Omega(r) * t * timeScale in the static observer frame. */
export function patternPhase(rHit: number, phiHit: number, t: number, timeScale: number, a: number): number {
  return phiHit - omegaKepler(rHit, a, true) * t * timeScale;
}

/** 32-bit integer cell hash -> [0,1]. Bit-identical to the WGSL twin: Math.imul / >>> 0 reproduce
 *  u32 multiply + logical shift exactly, so only the interpolation arithmetic differs f32 vs f64. */
function ihash(ix: number, iy: number): number {
  let n = (Math.imul(ix >>> 0, 1973) + Math.imul(iy >>> 0, 9277)) >>> 0;
  n = Math.imul(n ^ (n >>> 15), 2246822519) >>> 0;
  n = Math.imul(n ^ (n >>> 13), 3266489917) >>> 0;
  return (n & 0xffffff) / 0xffffff;
}
function smooth(t: number): number { return t * t * (3 - 2 * t); }
function vnoise(x: number, y: number): number {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = smooth(x - ix), fy = smooth(y - iy);
  const a00 = ihash(ix, iy), a10 = ihash(ix + 1, iy);
  const a01 = ihash(ix, iy + 1), a11 = ihash(ix + 1, iy + 1);
  return (a00 * (1 - fx) + a10 * fx) * (1 - fy) + (a01 * (1 - fx) + a11 * fx) * fy;
}
/** Multi-octave value noise in [0,~1); domain (logR, psi) so features shear with radius and phase. */
export function turbulence(logR: number, psi: number, octaves: number): number {
  let sum = 0, amp = 0.5, freq = 1;
  for (let o = 0; o < octaves; o++) { sum += amp * vnoise(logR * freq, psi * freq); amp *= 0.5; freq *= 2; }
  return sum;
}

const TWO_PI = 2 * Math.PI;
/** Sum of orbiting Gaussian hot-spots, each fixed in the co-rotating (r, psi) frame. */
export function hotspotField(rHit: number, psi: number, spots: HotSpot[]): number {
  let s = 0;
  for (const sp of spots) {
    const dr = rHit - sp.r;
    let dpsi = psi - sp.psi;
    dpsi -= TWO_PI * Math.round(dpsi / TWO_PI); // shortest angular separation
    const arc = sp.r * dpsi;                    // arc length along the ring
    s += sp.amp * Math.exp(-(dr * dr + arc * arc) / (2 * sp.sigma * sp.sigma));
  }
  return s;
}

/** Dimensionless emission multiplier. Exactly 1 when turbAmp=0, breatheAmp=0, no spots -> Tier-1. */
export function emissionField(
  rHit: number, psi: number, t: number,
  turbAmp: number, breatheAmp: number, spots: HotSpot[],
): number {
  const turb = 1 + turbAmp * (turbulence(Math.log(rHit), psi, 3) - 0.5) * 2;
  const breathe = 1 + breatheAmp * Math.sin(TWO_PI * t / T_BREATHE);
  return Math.max(0, turb * breathe + hotspotField(rHit, psi, spots));
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- emission`
Expected: PASS (all cases).

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `npm test`
Expected: PASS — existing physics suites plus the new `emission` suite.

- [ ] **Step 6: Commit**

```bash
git add src/physics/emission.ts tests/emission.test.ts
git commit -m "feat(physics): phenomenological time-varying disk emission field (Tier 2A)"
```

---

### Task 2: Extend the uniform buffer (append-only) + keep build green

**Files:**
- Modify: `src/render/uniforms.ts`
- Modify: `src/main.ts:107-111` (the `UniformValues` literal in the render loop)
- Modify: `src/test/shadow.browser.ts:27-28` (its `UniformValues` literal)
- Test: `tests/uniforms.test.ts` (new)

**Interfaces:**
- Produces: `UniformValues` gains `blend, timeScale, turbAmp, breatheAmp` (floats) and `nSpots` (uint). New `UNIFORM_SIZE = 80`. Field byte offsets: `blend@56, timeScale@60, turbAmp@64, breatheAmp@68, nSpots@72`.
- Consumed by: Task 3 (shader), Task 5 (main loop).

- [ ] **Step 1: Write the failing test**

`tests/uniforms.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { packUniforms, UNIFORM_SIZE, type UniformValues } from "../src/render/uniforms";

describe("uniforms packing", () => {
  it("is 80 bytes and packs new fields at the expected offsets", () => {
    expect(UNIFORM_SIZE).toBe(80);
    const u: UniformValues = {
      resW: 100, resH: 50, a: 0.9, incl: 1.2, rObs: 1000, fovScale: 14, rIn: 5, rOut: 40,
      Tpeak: 3e4, exposure: 1.6, time: 7, frame: 3, reset: 0, maxSteps: 1200,
      blend: 0.15, timeScale: 2, turbAmp: 0.6, breatheAmp: 0.1, nSpots: 4,
    };
    const dv = new DataView(packUniforms(u));
    expect(dv.getFloat32(0, true)).toBeCloseTo(100);   // resW
    expect(dv.getFloat32(40, true)).toBeCloseTo(7);     // time (index 10)
    expect(dv.getUint32(44, true)).toBe(3);             // frame (index 11)
    expect(dv.getFloat32(56, true)).toBeCloseTo(0.15);  // blend (index 14)
    expect(dv.getFloat32(60, true)).toBeCloseTo(2);     // timeScale (index 15)
    expect(dv.getFloat32(64, true)).toBeCloseTo(0.6);   // turbAmp (index 16)
    expect(dv.getFloat32(68, true)).toBeCloseTo(0.1);   // breatheAmp (index 17)
    expect(dv.getUint32(72, true)).toBe(4);             // nSpots (index 18)
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- uniforms`
Expected: FAIL — `UNIFORM_SIZE` is 64 / new fields missing from the type.

- [ ] **Step 3: Update `src/render/uniforms.ts`**

Replace the whole file with:
```ts
// Layout MUST match the `Uniforms` struct in raytrace.wgsl (4-byte scalars, vec2 first).
// floats: resW,resH,a,incl,rObs,fovScale,rIn,rOut,Tpeak,exposure,time (11)
//         + blend,timeScale,turbAmp,breatheAmp (4)                     -> 15 floats
// uint:   frame,reset,maxSteps (3) + nSpots (1)                        -> 4 uints
export interface UniformValues {
  resW: number; resH: number; a: number; incl: number; rObs: number; fovScale: number;
  rIn: number; rOut: number; Tpeak: number; exposure: number; time: number;
  frame: number; reset: number; maxSteps: number;
  blend: number; timeScale: number; turbAmp: number; breatheAmp: number; nSpots: number;
}
export const UNIFORM_FLOATS = 15, UNIFORM_UINTS = 4;
export const UNIFORM_SIZE = Math.ceil((UNIFORM_FLOATS + UNIFORM_UINTS) / 4) * 16; // -> 80 bytes

export function packUniforms(u: UniformValues): ArrayBuffer {
  const buf = new ArrayBuffer(UNIFORM_SIZE);
  const f = new Float32Array(buf), i = new Uint32Array(buf);
  f[0] = u.resW; f[1] = u.resH; f[2] = u.a; f[3] = u.incl;
  f[4] = u.rObs; f[5] = u.fovScale; f[6] = u.rIn; f[7] = u.rOut;
  f[8] = u.Tpeak; f[9] = u.exposure; f[10] = u.time;
  i[11] = u.frame; i[12] = u.reset; i[13] = u.maxSteps;
  f[14] = u.blend; f[15] = u.timeScale; f[16] = u.turbAmp; f[17] = u.breatheAmp;
  i[18] = u.nSpots;
  return buf;
}
```

- [ ] **Step 4: Keep the two existing `UniformValues` constructors compiling (features-off)**

In `src/main.ts`, the render-loop literal (currently lines 107-111) — add the five fields with features off and the EMA-compatible progressive blend. Replace the `const u: UniformValues = {...}` block with:
```ts
      const u: UniformValues = {
        resW: r.width, resH: r.height, a: state.a, incl: state.incl * Math.PI / 180,
        rObs: 1000, fovScale: 14, rIn, rOut, Tpeak: T_PEAK, exposure: state.exposure,
        time: now / 1000, frame: sample, reset: sample === 0 ? 1 : 0, maxSteps: 1200,
        blend: 1 / (sample + 1), timeScale: 1, turbAmp: 0, breatheAmp: 0, nSpots: 0,
      };
```
(Task 5 turns these into the live animation values; here they keep Tier-1 behavior.)

In `src/test/shadow.browser.ts`, extend its literal (currently lines 27-28) to:
```ts
  const u: UniformValues = { resW: r.width, resH: r.height, a, incl: Math.PI / 18, rObs: 1000,
    fovScale, rIn: rPh, rOut, Tpeak: 3.0e4, exposure: 0, time: 0, frame: 0, reset: 1, maxSteps: 8000,
    blend: 1, timeScale: 1, turbAmp: 0, breatheAmp: 0, nSpots: 0 };
```

- [ ] **Step 5: Run tests + typecheck build**

Run: `npm test -- uniforms && npm run build`
Expected: uniforms test PASS; `tsc` build succeeds (no type errors from the constructors).

- [ ] **Step 6: Commit**

```bash
git add src/render/uniforms.ts src/main.ts src/test/shadow.browser.ts tests/uniforms.test.ts
git commit -m "feat(render): append Tier 2A uniforms (blend, timeScale, turbAmp, breatheAmp, nSpots)"
```

---

### Task 3: Wire emission into the compute shader + EMA accumulation

**Files:**
- Modify: `src/render/gpu.ts` (add hot-spot buffer, binding 4, `uploadHotSpots`)
- Modify: `src/render/raytrace.wgsl` (struct, binding 4, emission twins, apply E, EMA)
- Modify: `src/render/present.wgsl` (drop the `/samples` divide)

**Interfaces:**
- Consumes: uniform fields from Task 2; `emissionField` semantics from Task 1.
- Produces: `Renderer.uploadHotSpots(spots: Float32Array)` — writes packed `(r, psi, sigma, amp)` vec4s to the hot-spot storage buffer (binding 4).

- [ ] **Step 1: Add the hot-spot buffer + binding to `gpu.ts`**

Add a field to the class (near `tempBuf!/colorBuf!`):
```ts
  spotBuf!: GPUBuffer;   // hot-spot params: array of vec4 (r, psi, sigma, amp)
```
In `init`, after the `colorBuf` placeholder line, create it (capacity 8 spots):
```ts
    this.spotBuf = this.device.createBuffer({ size: 8 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
```
Add the upload method (next to `uploadLUTs`):
```ts
  /** Upload packed hot-spot params (Float32Array of (r,psi,sigma,amp) per spot). */
  uploadHotSpots(spots: Float32Array) {
    this.device.queue.writeBuffer(this.spotBuf, 0, spots as Float32Array<ArrayBuffer>);
  }
```
In `rebind`, add binding 4 to the **compute** bind group (append to its `entries`):
```ts
      { binding: 3, resource: { buffer: this.colorBuf } },
      { binding: 4, resource: { buffer: this.spotBuf } }] });
```

- [ ] **Step 2: Extend the raytrace struct + add binding 4**

In `src/render/raytrace.wgsl`, replace the `struct Uniforms {...}` (lines 1-4) with:
```wgsl
struct Uniforms {
  res: vec2<f32>, a: f32, incl: f32, rObs: f32, fovScale: f32, rIn: f32, rOut: f32,
  Tpeak: f32, exposure: f32, time: f32, frame: u32, reset: u32, maxSteps: u32,
  blend: f32, timeScale: f32, turbAmp: f32, breatheAmp: f32, nSpots: u32,
};
```
After the existing bindings (line 8), add:
```wgsl
@group(0) @binding(4) var<storage, read> hotspots: array<vec4<f32>>; // (r, psi, sigma, amp)
```

- [ ] **Step 3: Add the emission WGSL twins**

In `src/render/raytrace.wgsl`, add these functions just above `@compute ... fn main` (after `starfield`):
```wgsl
// --- Tier 2A emission field (WGSL twin of src/physics/emission.ts) -----------------------------
fn ihashE(ix: i32, iy: i32) -> f32 {
  var n = u32(ix) * 1973u + u32(iy) * 9277u;
  n = (n ^ (n >> 15u)) * 2246822519u;
  n = (n ^ (n >> 13u)) * 3266489917u;
  return f32(n & 0xffffffu) / f32(0xffffffu);
}
fn smoothE(t: f32) -> f32 { return t * t * (3.0 - 2.0 * t); }
fn vnoiseE(x: f32, y: f32) -> f32 {
  let ix = i32(floor(x)); let iy = i32(floor(y));
  let fx = smoothE(x - floor(x)); let fy = smoothE(y - floor(y));
  let a00 = ihashE(ix, iy); let a10 = ihashE(ix + 1, iy);
  let a01 = ihashE(ix, iy + 1); let a11 = ihashE(ix + 1, iy + 1);
  return (a00 * (1.0 - fx) + a10 * fx) * (1.0 - fy) + (a01 * (1.0 - fx) + a11 * fx) * fy;
}
fn turbulenceE(logR: f32, psi: f32) -> f32 {
  var sum = 0.0; var amp = 0.5; var freq = 1.0;
  for (var o = 0u; o < 3u; o++) { sum += amp * vnoiseE(logR * freq, psi * freq); amp *= 0.5; freq *= 2.0; }
  return sum;
}
fn hotspotFieldE(rHit: f32, psi: f32) -> f32 {
  var s = 0.0;
  for (var k = 0u; k < U.nSpots; k++) {
    let sp = hotspots[k];
    let dr = rHit - sp.x;
    var dpsi = psi - sp.y;
    dpsi = dpsi - 2.0 * PI * round(dpsi / (2.0 * PI));
    let arc = sp.x * dpsi;
    s += sp.w * exp(-(dr * dr + arc * arc) / (2.0 * sp.z * sp.z));
  }
  return s;
}
fn emissionFieldE(rHit: f32, psi: f32) -> f32 {
  let turb = 1.0 + U.turbAmp * (turbulenceE(log(rHit), psi) - 0.5) * 2.0;
  let breathe = 1.0 + U.breatheAmp * sin(2.0 * PI * U.time / 2000.0);
  return max(0.0, turb * breathe + hotspotFieldE(rHit, psi));
}
```

- [ ] **Step 4: Apply the emission field at the disk hit**

In `src/render/raytrace.wgsl`, inside `if (rHit >= U.rIn && rHit <= U.rOut)` (currently lines 147-156), replace the body with:
```wgsl
        let Tn = sampleTemp(rHit);
        let Om = omegaKep(rHit, a);
        let gl = gLow(rHit, PI*0.5, a);
        let rad = -(gl[0] + 2.0*Om*gl[1] + Om*Om*gl[4]);
        let g = sqrt(max(0.0, rad)) / (1.0 - Om*xi); // Doppler + gravitational redshift factor
        let Tobs = U.Tpeak * g * Tn;                 // observed blackbody temperature
        let phiHit = mix(s.x.w, sNew.x.w, frac);     // azimuth of the emitting matter
        let psi = phiHit - Om * U.time * U.timeScale;// co-rotating pattern phase
        let E = emissionFieldE(rHit, psi);           // time-varying brightness (==1 when features off)
        color = sampleColor(Tobs) * pow(g * Tn, 4.0) * E;
        break;
```

- [ ] **Step 5: Switch accumulation to a temporal EMA**

In `src/render/raytrace.wgsl`, replace the final two lines (currently 171-172):
```wgsl
  let prev = select(accum[idx].rgb, vec3(0.0), U.reset == 1u);
  accum[idx] = vec4<f32>(prev + color, 1.0);
```
with:
```wgsl
  // Temporal EMA: blend = 1/(frame+1) reproduces the Tier-1 running mean when static; a fixed
  // blend (~0.15) tracks an animating scene. blend==1 (first frame after a reset) clears cleanly.
  accum[idx] = vec4<f32>(mix(accum[idx].rgb, color, U.blend), 1.0);
```

- [ ] **Step 6: Drop the sample-count divide in `present.wgsl`**

In `src/render/present.wgsl`, replace lines 28-30:
```wgsl
  let samples = f32(U.frame + 1u);

  var hdr = accum[idx].rgb / samples;
```
with:
```wgsl
  var hdr = accum[idx].rgb; // accum already holds normalized radiance (EMA / running mean)
```

- [ ] **Step 7: Build + regression-check the still image**

Run: `npm run build`
Expected: build succeeds.

Then verify Tier-1 parity of the static path (features off) still holds:
Run: `npm run dev` and open `http://localhost:5173/?shadow`
Expected: `SHADOW PASS (structural)` — centred shadow, ringed by disk, same apparent radius as before (features are off in this route, so the image must be unchanged).

Also confirm the metric parity is untouched:
Open `http://localhost:5173/?parity`
Expected: `PARITY PASS`.

- [ ] **Step 8: Commit**

```bash
git add src/render/gpu.ts src/render/raytrace.wgsl src/render/present.wgsl
git commit -m "feat(render): apply emission field at disk hit + EMA accumulation (Tier 2A)"
```

---

### Task 4: GPU parity for the turbulence twin

**Files:**
- Create: `src/render/turb-parity.wgsl`
- Modify: `src/test/parity.browser.ts` (add a second, turbulence, comparison pass)

**Interfaces:**
- Consumes: `turbulence(logR, psi, 3)` from `src/physics/emission.ts`.
- Produces: `runParity()` `maxErr` now also reflects turbulence CPU-vs-GPU error (still gated at `< 1e-3`).

- [ ] **Step 1: Create the parity shader**

`src/render/turb-parity.wgsl`:
```wgsl
// Parity shader: WGSL twin of turbulence() from src/physics/emission.ts. Inputs are (logR, psi).
fn ihashE(ix: i32, iy: i32) -> f32 {
  var n = u32(ix) * 1973u + u32(iy) * 9277u;
  n = (n ^ (n >> 15u)) * 2246822519u;
  n = (n ^ (n >> 13u)) * 3266489917u;
  return f32(n & 0xffffffu) / f32(0xffffffu);
}
fn smoothE(t: f32) -> f32 { return t * t * (3.0 - 2.0 * t); }
fn vnoiseE(x: f32, y: f32) -> f32 {
  let ix = i32(floor(x)); let iy = i32(floor(y));
  let fx = smoothE(x - floor(x)); let fy = smoothE(y - floor(y));
  let a00 = ihashE(ix, iy); let a10 = ihashE(ix + 1, iy);
  let a01 = ihashE(ix, iy + 1); let a11 = ihashE(ix + 1, iy + 1);
  return (a00 * (1.0 - fx) + a10 * fx) * (1.0 - fy) + (a01 * (1.0 - fx) + a11 * fx) * fy;
}
fn turbulenceE(logR: f32, psi: f32) -> f32 {
  var sum = 0.0; var amp = 0.5; var freq = 1.0;
  for (var o = 0u; o < 3u; o++) { sum += amp * vnoiseE(logR * freq, psi * freq); amp *= 0.5; freq *= 2.0; }
  return sum;
}
@group(0) @binding(0) var<storage, read> inp: array<vec2<f32>>;   // (logR, psi)
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = arrayLength(&inp);
  if (gid.x >= n) { return; }
  outp[gid.x] = turbulenceE(inp[gid.x].x, inp[gid.x].y);
}
```

- [ ] **Step 2: Extend `parity.browser.ts` to compare turbulence**

Add the import at the top:
```ts
import { turbulence } from "../physics/emission";
import turbParityWGSL from "../render/turb-parity.wgsl?raw";
```
Immediately before `return { maxErr, rows: cases.length };`, insert a second pass that reuses `device` and folds its error into `maxErr`:
```ts
  // --- turbulence parity (CPU emission.ts vs GPU turb-parity.wgsl) ---
  const tcases = [
    { logR: Math.log(6), psi: 0.4 }, { logR: Math.log(9), psi: 1.7 },
    { logR: Math.log(14), psi: 3.9 }, { logR: Math.log(22), psi: 5.2 },
  ];
  const tin = device.createBuffer({ size: tcases.length * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const tarr = new Float32Array(tcases.length * 2);
  tcases.forEach((c, i) => { tarr.set([c.logR, c.psi], i * 2); });
  device.queue.writeBuffer(tin, 0, tarr);
  const tout = device.createBuffer({ size: tcases.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const tread = device.createBuffer({ size: tcases.length * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const tmod = device.createShaderModule({ code: turbParityWGSL });
  const tpipe = device.createComputePipeline({ layout: "auto", compute: { module: tmod, entryPoint: "main" } });
  const tbind = device.createBindGroup({ layout: tpipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: tin } }, { binding: 1, resource: { buffer: tout } }] });
  const tenc = device.createCommandEncoder();
  const tcp = tenc.beginComputePass(); tcp.setPipeline(tpipe); tcp.setBindGroup(0, tbind); tcp.dispatchWorkgroups(1); tcp.end();
  tenc.copyBufferToBuffer(tout, 0, tread, 0, tcases.length * 4);
  device.queue.submit([tenc.finish()]);
  await tread.mapAsync(GPUMapMode.READ);
  const tgpu = new Float32Array(tread.getMappedRange().slice(0));
  tcases.forEach((c, i) => {
    const cpu = turbulence(c.logR, c.psi, 3);
    maxErr = Math.max(maxErr, Math.abs(tgpu[i] - cpu) / (1 + Math.abs(cpu)));
  });
  return { maxErr, rows: cases.length + tcases.length };
```
Delete the original `return { maxErr, rows: cases.length };` line so this new block's return is the only one.

- [ ] **Step 3: Build + run the parity route**

Run: `npm run build`
Expected: build succeeds.

Run: `npm run dev`, open `http://localhost:5173/?parity`
Expected: `PARITY PASS` with `maxRelErr` still `< 1e-3` over 8 cases (4 metric + 4 turbulence).

- [ ] **Step 4: Commit**

```bash
git add src/render/turb-parity.wgsl src/test/parity.browser.ts
git commit -m "test(render): GPU-vs-CPU parity for the turbulence emission twin"
```

---

### Task 5: Drive the animation from `main.ts`

**Files:**
- Modify: `src/main.ts`

**Interfaces:**
- Consumes: `HotSpot` from `src/physics/emission.ts`; `Renderer.uploadHotSpots` from Task 3; uniform fields from Task 2.
- Produces: live-animated render; module-level `state` gains `timeScale, turbAmp, breatheAmp, playing`; `flareScale` re-uploads spots.

- [ ] **Step 1: Add imports and animation state**

In `src/main.ts`, extend the physics import (line 3-4 area) and add a hot-spot set. After the existing `import { iscoRadius, photonOrbit } from "./physics/orbits";` line add:
```ts
import type { HotSpot } from "./physics/emission";
```
Inside the `else` interactive block, replace the `const state = {...}` line (currently line 36) with:
```ts
  const state = { a: 0.9, incl: 72, exposure: 1.6, timeScale: 1.0, turbAmp: 0.6, breatheAmp: 0.0, playing: true, flareScale: 1.0 };
  const SPEED = 20;        // coordinate-time M advanced per real second at timeScale = 1
  const EMA_BLEND = 0.15;  // trailing-window weight while animating
  let simTime = 0, lastNow = 0;
  const baseSpots: HotSpot[] = [
    { r: 8,  psi: 0.0, sigma: 1.2, amp: 1.8 },
    { r: 12, psi: 2.1, sigma: 1.6, amp: 1.2 },
    { r: 16, psi: 4.3, sigma: 2.0, amp: 0.9 },
  ];
  const packSpots = (scale: number) => {
    const f = new Float32Array(baseSpots.length * 4);
    baseSpots.forEach((s, i) => f.set([s.r, s.psi, s.sigma, s.amp * scale], i * 4));
    return f;
  };
```

- [ ] **Step 2: Upload the hot-spots once, after the LUTs are ready**

In `src/main.ts`, just after the initial `rebuildLUTs(); refreshReadouts();` calls (currently lines 97-98), add:
```ts
  r.uploadHotSpots(packSpots(state.flareScale));
```

- [ ] **Step 3: Integrate sim-time and choose the blend each frame**

Replace the whole `loop` function (currently lines 104-118) with:
```ts
  const TARGET_MS = 67; // ~15 fps cap so the tab stays responsive on modest GPUs
  let lastFrame = 0;
  function loop(now: number) {
    if (now - lastFrame >= TARGET_MS) {
      const dt = lastNow ? (now - lastNow) / 1000 : 0; lastNow = now;
      if (state.playing) simTime += dt * SPEED;   // advance coordinate time only while playing
      lastFrame = now;
      // Playing: fixed EMA (blend==1 on the reset frame to clear). Paused: progressive running mean.
      const blend = state.playing ? (sample === 0 ? 1 : EMA_BLEND) : 1 / (sample + 1);
      const u: UniformValues = {
        resW: r.width, resH: r.height, a: state.a, incl: state.incl * Math.PI / 180,
        rObs: 1000, fovScale: 14, rIn, rOut, Tpeak: T_PEAK, exposure: state.exposure,
        time: simTime, frame: sample, reset: sample === 0 ? 1 : 0, maxSteps: 1200,
        blend, timeScale: state.timeScale, turbAmp: state.turbAmp,
        breatheAmp: state.breatheAmp, nSpots: baseSpots.length,
      };
      r.frame(u);
      sample++;
      if ((sample & 7) === 0 || sample < 4) sppEl.textContent = String(sample);
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
```

- [ ] **Step 4: Build + manual visual check**

Run: `npm run build`
Expected: build succeeds.

Run: `npm run dev`, open `http://localhost:5173/`
Expected (manual):
- The disk visibly rotates; inner annuli sweep faster than outer.
- Bright hot-spots orbit and **brighten on the approaching (blue-shifted) side, dim on the receding side** (the existing g-factor beaming the moving features).
- Dragging the canvas still tilts the camera and the image stays responsive.

- [ ] **Step 5: Commit**

```bash
git add src/main.ts
git commit -m "feat(app): animate the disk — sim-time, EMA blend, orbiting hot-spots (Tier 2A)"
```

---

### Task 6: UI controls + final verification

**Files:**
- Modify: `index.html`
- Modify: `src/main.ts` (wire the new controls)

**Interfaces:**
- Consumes: `state`, `packSpots`, `r.uploadHotSpots`, `reset` from Task 5.

- [ ] **Step 1: Add the control markup**

In `index.html`, after the Exposure control block (currently lines 82-85, the `<div class="ctrl">…exp…</div>`) insert:
```html
    <div class="ctrl">
      <div class="row"><label>Motion</label><span class="val"><b id="tsv">1.0</b>×</span></div>
      <input id="ts" type="range" min="0" max="5" step="0.1" value="1">
    </div>
    <div class="ctrl">
      <div class="row"><label>Turbulence</label><span class="val"><b id="turbv">0.60</b></span></div>
      <input id="turb" type="range" min="0" max="1.5" step="0.05" value="0.6">
    </div>
    <div class="ctrl">
      <div class="row"><label>Flares</label><span class="val"><b id="flarev">1.0</b>×</span></div>
      <input id="flare" type="range" min="0" max="3" step="0.1" value="1">
    </div>
    <button id="playpause" style="width:100%;margin-top:6px;padding:9px;background:transparent;
      border:1px solid var(--hair);border-radius:9px;color:var(--accent);font:inherit;font-size:11px;
      letter-spacing:0.2em;text-transform:uppercase;cursor:pointer">Pause</button>
```
Also update the hint text (currently line 96) to:
```html
  <p id="hint">Drag to tilt · <b>live simulation</b></p>
```

- [ ] **Step 2: Wire the controls in `main.ts`**

In `src/main.ts`, after the existing `exp.addEventListener(...)` block (currently ends ~line 75), add:
```ts
  const ts = $("ts") as HTMLInputElement, turb = $("turb") as HTMLInputElement, flare = $("flare") as HTMLInputElement;
  const tsv = $("tsv"), turbv = $("turbv"), flarev = $("flarev"), playBtn = $("playpause") as HTMLButtonElement;

  ts.addEventListener("input", () => { state.timeScale = +ts.value; tsv.textContent = state.timeScale.toFixed(1); });
  turb.addEventListener("input", () => { state.turbAmp = +turb.value; turbv.textContent = state.turbAmp.toFixed(2); reset(); });
  flare.addEventListener("input", () => {
    state.flareScale = +flare.value; flarev.textContent = state.flareScale.toFixed(1);
    r.uploadHotSpots(packSpots(state.flareScale)); reset();
  });
  playBtn.addEventListener("click", () => {
    state.playing = !state.playing;
    playBtn.textContent = state.playing ? "Pause" : "Play";
    reset(); // clean restart (play) or fresh convergence to a still (pause)
  });
```
(`reset` is the existing `() => { sample = 0; }` from line 52; `$`, `packSpots`, and `state` are already in scope.)

- [ ] **Step 3: Build + verify all controls**

Run: `npm run build`
Expected: build succeeds.

Run: `npm run dev`, open `http://localhost:5173/` and confirm (manual):
- **Motion** slider changes rotation speed (0 = frozen positions, higher = faster).
- **Turbulence** slider changes the churn contrast; at 0 the disk is smooth.
- **Flares** slider brightens/dims the hot-spots; at 0 they vanish.
- **Pause** freezes the scene and the image re-converges to a clean still; **Play** resumes.

- [ ] **Step 4: Full regression sweep**

Run: `npm test`
Expected: PASS — all suites including `emission` and `uniforms`.

Run: `npm run build`
Expected: succeeds.

Open each validation route and confirm:
- `http://localhost:5173/?parity` → `PARITY PASS`, `maxRelErr < 1e-3`.
- `http://localhost:5173/?shadow` → `SHADOW PASS (structural)` (features-off path unchanged).

- [ ] **Step 5: Update the README status + commit**

In `README.md`, under `## Status`, note Tier 2A shipped (time evolution, turbulence, orbiting flares) and that the jet (Tier 2B) is next. Then:
```bash
git add index.html src/main.ts README.md
git commit -m "feat(ui): motion/turbulence/flare controls + play-pause; document Tier 2A"
```

---

## Self-Review

**Spec coverage (§ = design doc sections):**
- §1/§3.1 pattern phase & differential rotation → Task 1 (`patternPhase`, tests), applied in Task 3 (`psi` at disk hit).
- §3.2 emission field (turbulence, hot-spots, breathing) → Task 1 (`turbulence`/`hotspotField`/`emissionField`), Task 3 (WGSL twins + apply).
- §3.3 coupling to existing g-factor optics unchanged → Task 3 Step 4 (E multiplies existing `pow(g*Tn,4)`; geodesic/g-factor lines untouched).
- §4 render-model EMA (animate vs progressive vs reset) → Task 3 Step 5 (compute `mix`), Task 3 Step 6 (present divide removed), Task 5 Step 3 (blend selection).
- §5 module plan (`emission.ts`, WGSL twin, uniforms, gpu, main, index) → Tasks 1/2/3/5/6.
- §6 UI (play/pause, time-scale, flares, turbulence) → Task 6.
- §7 tests (CPU unit, GPU parity, features-off==Tier1 gate, manual visual) → Task 1 (unit), Task 4 (parity), Task 3 Step 7 + Task 6 Step 4 (`?shadow` regression), Task 5 Step 4 (visual).
- §8 out-of-scope (Σ-diffusion cut, jet deferred) → not implemented, correct.

**Placeholder scan:** none — every code step contains complete code; every run step names the exact command and expected result.

**Type consistency:** `HotSpot` fields `{r,psi,sigma,amp}` consistent across `emission.ts`, `main.ts` (`baseSpots`/`packSpots`), and the WGSL `sp.x/.y/.z/.w` mapping. `UniformValues` field names identical in `uniforms.ts`, `main.ts`, `shadow.browser.ts`. `uploadHotSpots(Float32Array)` signature matches its one caller. Uniform struct field order in `raytrace.wgsl` matches `packUniforms` index order (blend@14, timeScale@15, turbAmp@16, breatheAmp@17, nSpots@18).

**Note on `reset` uniform:** now vestigial in the shader (EMA `blend` subsumes it) but retained in the struct/interface to keep the append-only layout and avoid churn in `present.wgsl`/`bloom.wgsl`; `main.ts` still sets it harmlessly.
