# Physics Accuracy Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix six confirmed physics/math defects found in the 2026-07-21 audit, and correct the README's overstated jet claim.

**Architecture:** The most serious defect is that backward-traced rays use the wrong null direction at the camera, so the renderer integrates a photon *leaving* the camera rather than retracing the one that *arrives*. Fixing it needs a CPU-testable camera module first (the bug survived because nothing on the CPU side ever exercised the camera's initial conditions). The remaining fixes are one- or two-line corrections to the horizon threshold, the escaped-ray direction, the color LUT normalization, the turbulence bias, and the jet constants.

**Tech Stack:** TypeScript, WGSL, WebGPU, Vite (`?raw` imports), Vitest, playwright-core.

Audit findings are recorded in this plan; there is no separate spec document.

## Global Constraints

- **No `Co-Authored-By` trailer in any commit.** Verify after each commit: `git log -1 --format=%B | grep -ci "co-authored"` must print `0`.
- **Do NOT push to origin. Do NOT merge. Do NOT switch branches.** Work stays on `fix/physics-accuracy`.
- `?parity` must report maxRelErr **exactly 9.690e-7 over 36 cases** at every task boundary. It covers metric/orbit/g-factor/turbulence/jet/shadow-classifier math, none of which any of these fixes should alter — **except Task 5**, which changes turbulence in all three twins simultaneously and must keep them in agreement (the number must stay 9.690e-7; if it moves, the twins desynced).
- `?shadow` must stay PASS. Its apparent radius **may legitimately change in Task 2** — record the new value with its reason; do not force it back.
- GPU routes need the dev server on **:5173** (`npm run dev`). Check for a stale server on that port before trusting results — a leftover process from another session has caused false passes here before.
- `npm test` count rises as tasks add tests. Current baseline: **58**.
- Windows. Bash and PowerShell both available; Bash is fine for git.

## File Structure

| File | Responsibility |
|---|---|
| `src/physics/camera.ts` | **Create.** CPU twin of the camera: screen (α,β) → initial geodesic state. Makes the ray convention testable without a GPU. |
| `tests/camera.test.ts` | **Create.** Reciprocity round-trip + time-orientation tests. |
| `src/render/raytrace.wgsl` | **Modify.** Backward-ray IC (T2), horizon threshold (T3), escaped-ray direction (T4), turbulence normalizer (T5). |
| `src/physics/color.ts` | **Modify.** Luminance normalization instead of max-channel (T5). |
| `src/physics/emission.ts` | **Modify.** Turbulence octave normalizer (T5). |
| `src/render/turb-parity.wgsl` | **Modify.** Same turbulence normalizer, to keep parity (T5). |
| `src/physics/jet.ts` | **Modify.** Correct the desynced `gain`/`ceil` and the false invariant claim (T6). |
| `README.md` | **Modify.** Replace the Blandford–Znajek/synchrotron claim with what the jet actually is (T6). |

---

### Task 1: CPU camera module + reciprocity test

The backward-ray bug survived because the camera's initial conditions exist only in WGSL and nothing tests them. This task creates the CPU twin implementing the **correct** convention and a test that would have caught the bug.

**Files:**
- Create: `src/physics/camera.ts`
- Create: `tests/camera.test.ts`

**Interfaces:**
- Consumes: `nullRadialMomentum`, `rk4` from `src/physics/geodesic`; `metricUpper` from `src/physics/kerr`.
- Produces: `screenToState(alpha: number, beta: number, a: number, incl: number, rObs: number): Float64Array` — an 8-element state `[t, r, th, ph, pt, pr, pth, pphi]`. Task 2 mirrors this exact convention in WGSL.

- [ ] **Step 1: Write the failing test**

Create `tests/camera.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { screenToState } from "../src/physics/camera";
import { rk4, nullRadialMomentum, conserved } from "../src/physics/geodesic";
import { metricUpper } from "../src/physics/kerr";

const A = 0.9, INCL = (72 * Math.PI) / 180, ROBS = 1000;

/** integrate a backward ray inward; return the first equatorial crossing */
function diskHit(s0: Float64Array, a = A) {
  let s = s0.slice();
  const rh = 1 + Math.sqrt(1 - a * a);
  for (let i = 0; i < 300000; i++) {
    const dl = Math.min(0.5, Math.max(0.002, 0.02 * (s[1] - rh)));
    const nx = rk4(s, a, dl);
    if (!isFinite(nx[1]) || nx[1] <= rh * 1.005) return { fate: "hole" as const };
    if (nx[1] > ROBS * 1.2) return { fate: "sky" as const };
    const f0 = s[2] - Math.PI / 2, f1 = nx[2] - Math.PI / 2;
    if (f0 * f1 < 0) {
      const fr = f0 / (f0 - f1);
      return { fate: "disk" as const, r: s[1] + fr * (nx[1] - s[1]) };
    }
    s = nx;
  }
  return { fate: "budget" as const };
}

/** a PHYSICAL photon leaving the equator at rE with conserved (xi, eta), future-directed, outgoing */
function shootOut(rE: number, xi: number, eta: number) {
  const pth0 = -Math.sqrt(eta); // toward the northern hemisphere
  const pr = nullRadialMomentum(rE, Math.PI / 2, A, -1, xi, pth0);
  let s = new Float64Array([0, rE, Math.PI / 2, 0, -1, pr, pth0, xi]);
  for (let i = 0; i < 500000 && s[1] < ROBS; i++) {
    const nx = rk4(s, A, Math.min(0.5, Math.max(0.002, 0.02 * s[1])));
    if (!isFinite(nx[1]) || nx[1] < 2.4) return null;
    s = nx;
  }
  return s[1] >= ROBS ? s : null;
}

describe("camera initial conditions", () => {
  it("produces a null initial 4-momentum", () => {
    for (const [al, be] of [[-6, 1], [8, 4], [0, 0], [12, -7]]) {
      const H = conserved(screenToState(al, be, A, INCL, ROBS), A).H;
      expect(Math.abs(H)).toBeLessThan(1e-12);
    }
  });

  it("traces PAST-directed, retracing the photon that arrives (not one leaving)", () => {
    // dt/dl < 0: going back in coordinate time. dr/dl < 0: inward.
    // The buggy convention gave dt/dl = +1.002 here.
    for (const [al, be] of [[-6, 1], [8, 4], [12, -7]]) {
      const s = screenToState(al, be, A, INCL, ROBS);
      const g = metricUpper(s[1], s[2], A);
      expect(g.tt * s[4] + g.tphi * s[7]).toBeLessThan(0);
      expect(g.rr * s[5]).toBeLessThan(0);
    }
  });

  it("reciprocity: a photon shot from the disk traces back to its emitter", () => {
    // Ground truth independent of the camera code: shoot a real photon out from r=8, find the
    // one that arrives at the camera's latitude, convert to a pixel, then trace that pixel back.
    const rE = 8;
    let checked = 0;
    for (const xi of [2.8, -3.5, 1.5]) {
      let lo = 0.02, hi = 60;
      const arrive = (e: number) => { const s = shootOut(rE, xi, e); return s ? s[2] : NaN; };
      let alo = arrive(lo), ahi = arrive(hi);
      if (!isFinite(alo) || !isFinite(ahi) || (alo - INCL) * (ahi - INCL) > 0) continue;
      let eta = 0;
      for (let k = 0; k < 60; k++) {
        eta = 0.5 * (lo + hi);
        const am = arrive(eta);
        if (!isFinite(am)) break;
        if ((alo - INCL) * (am - INCL) <= 0) { hi = eta; ahi = am; } else { lo = eta; alo = am; }
      }
      const out = shootOut(rE, xi, eta);
      if (!out) continue;
      const hit = diskHit(screenToState(-xi / Math.sin(INCL), out[6], A, INCL, ROBS));
      expect(hit.fate).toBe("disk");
      expect((hit as { r: number }).r).toBeCloseTo(rE, 2);
      checked++;
    }
    expect(checked).toBeGreaterThanOrEqual(3); // guard: the loop must not silently skip everything
  }, 120_000);
});
```

- [ ] **Step 2: Run it to verify it fails**

```bash
npx vitest run tests/camera.test.ts
```

Expected: **FAIL** — `Failed to resolve import "../src/physics/camera"`.

- [ ] **Step 3: Write the implementation**

Create `src/physics/camera.ts`:

```ts
import { metricUpper } from "./kerr";
import { nullRadialMomentum } from "./geodesic";

/**
 * Screen impact parameters (alpha, beta) -> initial state for BACKWARD ray tracing.
 *
 * The photon we see arrives at the camera moving outward, forward in coordinate time. Retracing
 * it means following its worldline in reverse, i.e. integrating the fully negated 4-momentum:
 * every component flips, not just p_t. The resulting ray is PAST-directed (dt/dl < 0) and inward
 * (dr/dl < 0). Negating p_t alone leaves a future-directed ray falling away from the camera --
 * a different null geodesic entirely, which renders the image at inclination (pi - incl).
 *
 * The conserved ratio xi = L_z/E is unchanged by the negation (both flip), so the redshift
 * g-factor and the shadow classifier still take the same xi as before.
 */
export function screenToState(alpha: number, beta: number, a: number, incl: number, rObs: number): Float64Array {
  const xi = -alpha * Math.sin(incl);
  const pt = 1, pphi = -xi, pth = -beta;   // full negation of the arriving photon's momentum
  const pr = -nullRadialMomentum(rObs, incl, a, pt, pphi, pth); // inward
  return new Float64Array([0, rObs, incl, 0, pt, pr, pth, pphi]);
}

/** Carter constant from the Bardeen screen coordinates. Both xi and eta are conserved. */
export function screenToXiEta(alpha: number, beta: number, a: number, incl: number): [number, number] {
  const xi = -alpha * Math.sin(incl);
  const ci = Math.cos(incl), si = Math.sin(incl);
  return [xi, beta * beta + (xi * xi * ci * ci) / Math.max(si * si, 1e-8) - a * a * ci * ci];
}
```

Note `nullRadialMomentum(r, th, a, pt, pphi, pth)` — check the parameter order in `src/physics/geodesic.ts` and match it exactly; it is `(r, th, a, pt, pphi, pth)`.

- [ ] **Step 4: Run it to verify it passes**

```bash
npx vitest run tests/camera.test.ts
```

Expected: **3 passed.** If the reciprocity test fails, STOP and report — do not loosen `toBeCloseTo`.

- [ ] **Step 5: Full suite + commit**

```bash
npm test
npm run build
```

Expected: **61 passed** (58 + 3), clean build.

```bash
git add src/physics/camera.ts tests/camera.test.ts
git commit -m "Add CPU camera twin with a reciprocity test

Backward ray tracing must integrate the fully negated 4-momentum, not just
a negated p_t. The reciprocity test shoots a physical photon from the disk,
converts its arrival to a pixel, and requires tracing that pixel back to
return to the emitter -- which the shipped WGSL convention does not do."
git log -1 --format=%B | grep -ci "co-authored"
```

Expected: `0`

---

### Task 2: Fix the shader's backward-ray initial condition

**Files:**
- Modify: `src/render/raytrace.wgsl:246`, and the stale comment at `:258`

**Interfaces:**
- Consumes: the convention established by `screenToState` in Task 1.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Record the baseline**

Start `npm run dev` (confirm nothing stale already holds :5173), then:

```bash
npm run verify:gpu
```

Record the `?shadow` apparent radius and calibration. Also capture a screenshot for the visual comparison in Step 4:

```bash
SHOT=before.png npm run verify:gpu
```

- [ ] **Step 2: Apply the fix**

In `src/render/raytrace.wgsl`, change line 246 from:

```wgsl
  let pt = -1.0; let pphi = xi; let pth = beta; // sign of p_th set by image y
```

to:

```wgsl
  // Backward tracing follows the arriving photon's worldline in REVERSE, which negates the whole
  // 4-momentum -- not p_t alone. This ray is past-directed (dt/dl < 0) and inward (dr/dl < 0).
  // Negating only p_t leaves a future-directed ray falling away from the camera: a different
  // geodesic, which renders inclination (pi - incl). See src/physics/camera.ts and tests/camera.test.ts.
  // xi = L_z/E is unchanged by the negation, so the g-factor and classifier below still use xi.
  let pt = 1.0; let pphi = -xi; let pth = -beta;
```

Leave `pr = -sqrt(...)` at line 249 exactly as it is — it is already inward, and it must be recomputed from the new momenta (which it is, since `rest` at line 248 reads `pt`/`pphi`/`pth`).

Then correct the now-wrong comment at line 258. Replace:

```wgsl
    // dl > 0 with p_r < 0 integrates INWARD (matches the geodesic capture test; a negative dl
    // would march rays outward -> black screen).
```

with:

```wgsl
    // dl > 0 with p_r < 0 integrates INWARD along the reversed worldline.
```

- [ ] **Step 3: Verify the gates**

```bash
npm test
npm run build
npm run verify:gpu
```

- `npm test` must stay at **61** — Task 1's reciprocity test covers the CPU convention and is unaffected.
- `?parity` must stay **exactly 9.690e-7 over 36 cases**. It exercises metric/orbit/g-factor/turbulence/jet/classifier math, none of which this touches. If it moves, STOP and report.
- `?shadow` must stay **PASS**. Its radius **may change** — the shadow is close to symmetric so a large shift is not expected, but any change is legitimate here. Record the old and new values.

- [ ] **Step 4: Visual confirmation**

```bash
SHOT=after.png npm run verify:gpu
```

Compare `before.png` and `after.png`. Expected: the lensed image of the disk's far side should move from one side of the shadow to the other (it was rendering inclination 108° instead of 72°). The Doppler left/right asymmetry should NOT flip.

Report what you actually observe. If the two images are identical, the fix did not take effect — stop and investigate. Delete both PNGs before committing.

- [ ] **Step 5: Commit**

```bash
rm -f before.png after.png
git add src/render/raytrace.wgsl
git commit -m "Trace the photon that arrives, not one leaving the camera

raytrace.wgsl negated only p_t and integrated forward, giving a future-directed
ray falling away from the camera rather than the reversed worldline of the
arriving photon. Backward tracing negates the whole 4-momentum. The old
convention rendered a valid Kerr image at inclination (pi - incl): the disk's
lensed far side arced under the shadow instead of over it, and the inclination
slider was effectively inverted. Invisible at a=0 and at incl=90deg."
git log -1 --format=%B | grep -ci "co-authored"
```

Expected: `0`

---

### Task 3: Widen the horizon capture threshold

At a=0.9, `rh*1.001` leaves 1.44e-3 M of margin while the minimum step moves r by ~4.2e-3 M. RK4's intermediate stages therefore sample r < r_+, where Δ < 0 flips the sign of g^rr and g^tt, the signature inverts, and the state explodes. The blown-up r can land above the escape cutoff and paint starfield inside the shadow. This is the upstream cause of the NaN contamination the `usable` and `finite` guards currently defend against.

**Files:**
- Modify: `src/render/raytrace.wgsl` (the horizon capture test, currently `rh * 1.001`)

**Interfaces:** none.

- [ ] **Step 1: Locate the threshold**

```bash
grep -n "rh \* 1.001" src/render/raytrace.wgsl
```

Expected: one hit, the capture test `if (s.x.y <= rh * 1.001) { color = vec3(0.0); resolved = true; break; }`.

- [ ] **Step 2: Apply the fix**

Change `rh * 1.001` to `rh * 1.005`, and extend the comment:

```wgsl
    // captured -> shadow. The margin must exceed one integration step (dl_min = 0.002 moves r by
    // ~4.2e-3 M at a=0.9), otherwise RK4's intermediate stages sample r < r_+, where Delta < 0
    // flips the metric signature and the state explodes to garbage that can pass the escape test
    // and paint starfield inside the shadow.
    if (s.x.y <= rh * 1.005) { color = vec3(0.0); resolved = true; break; }
```

- [ ] **Step 3: Verify**

```bash
npm test
npm run build
npm run verify:gpu
```

Expected: 61 tests, clean build, `?parity` **exactly 9.690e-7 over 36 cases**, `?shadow` PASS.

The shadow radius may grow by a hair (the capture surface moved out by 0.4% of r_+ ≈ 0.006 M, far below one pixel at the default FOV). Record the value.

- [ ] **Step 4: Commit**

```bash
git add src/render/raytrace.wgsl
git commit -m "Widen the horizon capture margin past one integration step

rh*1.001 gave 1.44e-3 M of margin at a=0.9 while the minimum step moves r by
~4.2e-3 M, so RK4 stages sampled inside the horizon where the metric signature
inverts. The resulting garbage state could pass the escape test and paint sky
inside the shadow; it is also the upstream source of the non-finite values the
accumulation guards defend against."
git log -1 --format=%B | grep -ci "co-authored"
```

Expected: `0`

---

### Task 4: Sample the sky along the propagation direction

The escaped-ray branch builds its direction from the *position* angles at the cutoff, which differs from the true asymptotic direction by ~b/r_cut. At r_cut = 1200 M that is up to ~16 mrad — about 11 texels on a 4096-wide panorama, growing radially outward, reaching ~13% error in lensed sky position at the frame edge. Using the propagation direction `dx^mu/dl = g^{mu nu} p_nu` is ~1800x more accurate at the same cutoff.

**Files:**
- Modify: `src/render/raytrace.wgsl` — the in-loop escape branch and the post-loop `!resolved` fallback (both build `dir` the same way)

**Interfaces:** none.

- [ ] **Step 1: Add a direction helper**

Insert this function just before `@compute @workgroup_size(8,8) fn main(`:

```wgsl
// Asymptotic sky direction of an escaping ray. Built from the propagation direction
// dx^mu/dl = g^{mu nu} p_nu, NOT the position unit vector: at the r>1.2*rObs cutoff those differ
// by ~b/r (up to ~16 mrad, ~11 panorama texels), which displaces every background star radially.
// gUp indices: 0=tt, 1=tphi, 2=rr, 3=thth, 4=phph.
fn skyDir(s: State, a: f32) -> vec3<f32> {
  let r = s.x.y; let th = s.x.z; let ph = s.x.w;
  let g = gUp(r, th, a);
  let dr  = g[2] * s.y.y;
  let dth = g[3] * s.y.z;
  let dph = g[1] * s.y.x + g[4] * s.y.w;
  let st = sin(th); let ct = cos(th); let sp = sin(ph); let cp = cos(ph);
  return normalize(vec3<f32>(
    dr * st * cp + r * ct * cp * dth - r * st * sp * dph,
    dr * st * sp + r * ct * sp * dth + r * st * cp * dph,
    dr * ct - r * st * dth));
}
```

- [ ] **Step 2: Use it in the escape branch**

In the in-loop escape branch, replace:

```wgsl
      let th = s.x.z; let ph = s.x.w;
      let dir = normalize(vec3<f32>(sin(th)*cos(ph), sin(th)*sin(ph), cos(th)));
```

with:

```wgsl
      let dir = skyDir(s, a);
```

- [ ] **Step 3: Use it in the post-loop fallback**

In the `if (!resolved)` block, keep the `usable` finite-range guard exactly as it is (it reads `th`/`ph` from `s.x` and must continue to), and replace only the `dir` construction in the `else` branch:

```wgsl
      let dir = skyDir(s, a);
```

Leave `let th = s.x.z; let ph = s.x.w;` in place — the `usable` test still needs them. Do NOT remove that guard.

- [ ] **Step 4: Verify**

```bash
npm test
npm run build
npm run verify:gpu
```

Expected: 61 tests, clean build, `?parity` **exactly 9.690e-7 over 36 cases**, `?shadow` PASS.

Then confirm no non-finite pixels were introduced — `skyDir` divides by nothing but does normalize a vector that could in principle be zero-length. Capture a render and check it is clean:

```bash
SHOT=sky.png npm run verify:gpu
```

Inspect `sky.png`: the background should still show a lensed ring with no black speckle or NaN blocks. Delete it before committing.

- [ ] **Step 5: Commit**

```bash
rm -f sky.png
git add src/render/raytrace.wgsl
git commit -m "Sample the sky along the propagation direction, not the position vector

At the r>1.2*rObs cutoff the position unit vector differs from the asymptotic
direction by ~b/r -- up to ~16 mrad, roughly 11 texels on the 4k panorama,
growing radially to ~13% error in lensed sky position at the frame edge. The
momentum direction is converged to ~7e-3 mrad at the same cutoff. This also
makes the two 'asymptotic direction' comments true."
git log -1 --format=%B | grep -ci "co-authored"
```

Expected: `0`

---

### Task 5: Photometric fixes — color LUT luminance and turbulence bias

Two independent defects, both one-liners, sharing one visual verification.

**(a)** `blackbodyLinearSRGB` normalizes by max channel, so the returned color's luminance varies with T (0.960 at 6504 K, 0.489 at 40000 K). The shader multiplies by `(g*Tn)^4` assuming luminance is constant, so the hottest and most-beamed material renders up to ~2x too dim relative to cooler material, partly cancelling the g^4 asymmetry.

**(b)** `turbulence()` sums octaves 0.5+0.25+0.125, giving mean 0.434 and max 0.875, but callers recentre with `(turb - 0.5)`. Measured mean emission multiplier is 0.921 at the default `turbAmp = 0.6`, so raising turbulence dims the disk.

**Files:**
- Modify: `src/physics/color.ts:28-29`
- Modify: `src/physics/emission.ts` (`turbulence`)
- Modify: `src/render/raytrace.wgsl` (`turbulenceE`)
- Modify: `src/render/turb-parity.wgsl` (`turbulenceE`)
- Modify: `tests/color.test.ts` if it asserts max-channel normalization

**Interfaces:** none.

- [ ] **Step 1: Check whether an existing test pins the old behaviour**

```bash
grep -n "max\|1\.0\|toBe" tests/color.test.ts
```

If a test asserts that the max channel equals 1, it encodes the bug and must be updated in Step 4 to assert luminance = 1 instead. Report what you find.

- [ ] **Step 2: Fix the color normalization**

In `src/physics/color.ts`, change the docstring and the final lines. Replace:

```ts
/** Linear sRGB color of a blackbody at temperature T (K), chromaticity-preserving, max channel = 1. */
```

with:

```ts
/** Linear sRGB color of a blackbody at temperature T (K), chromaticity-preserving, luminance = 1. */
```

and replace:

```ts
  const m = Math.max(r, g, b) || 1;
  return [r / m, g / m, b / m];
```

with:

```ts
  // Normalize by relative luminance, NOT by max channel. The shader multiplies this by (g*Tn)^4
  // to realize the T_obs^4 law; if the LUT's luminance varied with T (max-channel normalization
  // gives 0.96 at 6504 K falling to 0.49 at 40000 K) that law would be silently scaled by a
  // factor sliding ~2x across the disk. Channels may exceed 1; the LUT is an f32 storage buffer.
  const Y = 0.2126 * r + 0.7152 * g + 0.0722 * b || 1;
  return [r / Y, g / Y, b / Y];
```

- [ ] **Step 3: Fix the turbulence bias in all three twins**

The octave amplitudes sum to 0.875, so dividing by that sum restores a mean near 0.5 and makes the documented `[0,1)` range true. Apply the **same** change in all three files or `?parity` will break.

`src/physics/emission.ts` — replace the body of `turbulence`:

```ts
/** Multi-octave value noise in [0,1); domain (logR, psi) so features shear with radius and phase. */
export function turbulence(logR: number, psi: number, octaves: number): number {
  let sum = 0, amp = 0.5, freq = 1, norm = 0;
  for (let o = 0; o < octaves; o++) { sum += amp * vnoise(logR * freq, psi * freq); norm += amp; amp *= 0.5; freq *= 2; }
  // Divide by the octave-amplitude sum: without it the mean is 0.434, not the 0.5 that callers
  // recentre against, so raising turbAmp systematically dims the disk (0.921x at the 0.6 default).
  return norm > 0 ? sum / norm : 0;
}
```

`src/render/raytrace.wgsl` — replace `turbulenceE`:

```wgsl
fn turbulenceE(logR: f32, psi: f32) -> f32 {
  var sum = 0.0; var amp = 0.5; var freq = 1.0; var norm = 0.0;
  for (var o = 0u; o < 3u; o++) { sum += amp * vnoiseE(logR * freq, psi * freq); norm += amp; amp *= 0.5; freq *= 2.0; }
  return sum / norm;
}
```

`src/render/turb-parity.wgsl` — replace `turbulenceE` with the identical body.

- [ ] **Step 4: Update or add tests**

If Step 1 found a test asserting max-channel = 1, change it to assert luminance = 1:

```ts
const Y = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
expect(Y).toBeCloseTo(1, 6);
```

Add to `tests/emission.test.ts` a test pinning the turbulence centre:

```ts
it("turbulence is centred near 0.5 so the emission multiplier is unbiased", () => {
  let s = 0; const N = 200000;
  for (let i = 0; i < N; i++) s += turbulence((i * 0.7331) % 50, (i * 1.4177) % 50, 3);
  const mean = s / N;
  expect(mean).toBeGreaterThan(0.47);
  expect(mean).toBeLessThan(0.53);
});
```

Make sure `turbulence` is imported in that file.

- [ ] **Step 5: Verify**

```bash
npm test
npm run build
npm run verify:gpu
```

- Test count rises by 1 (to **62**) plus any you modified.
- `?parity` must stay **exactly 9.690e-7 over 36 cases**. This is the real gate for Step 3: it compares CPU `turbulence` against `turb-parity.wgsl`. If it moves, the three twins are not identical — find the one you missed. Note `?parity` does NOT cover `raytrace.wgsl`'s copy, so re-read that one by eye against the other two.
- `?shadow` PASS.

- [ ] **Step 6: Visual check and exposure note**

```bash
SHOT=photometric.png npm run verify:gpu
```

Luminance normalization raises LUT values by 1/Y_old (roughly 1.04x to 2.05x), so the disk will be **brighter overall** and the hot inner region and approaching limb brighter still relative to the outer disk. Confirm the image is not blown out. If it clips badly, note the recommended `exposure` default change in your report but do **not** change it — that is a taste decision for the user. Delete the PNG.

- [ ] **Step 7: Commit**

```bash
rm -f photometric.png
git add src/physics/color.ts src/physics/emission.ts src/render/raytrace.wgsl src/render/turb-parity.wgsl tests/
git commit -m "Fix two photometric biases: LUT luminance and turbulence centre

The color LUT normalized by max channel, so its luminance slid from 0.96 at
6504 K to 0.49 at 40000 K while the shader multiplied by (g*Tn)^4 assuming it
was constant -- dimming the hottest, most-beamed material by up to ~2x and
partly cancelling the g^4 asymmetry. Separately, turbulence() summed octaves to
mean 0.434 while callers recentred at 0.5, so raising turbAmp dimmed the disk
(0.921x at the default). Normalizing by the octave-amplitude sum fixes it in
all three twins."
git log -1 --format=%B | grep -ci "co-authored"
```

Expected: `0`

---

### Task 6: Jet constant desync and README honesty

**Files:**
- Modify: `src/physics/jet.ts:6,15-16`
- Modify: `README.md` (the Tier 2B paragraph)

**Interfaces:** none.

- [ ] **Step 1: Confirm the desync**

```bash
grep -n "gain\|ceil" src/physics/jet.ts
grep -n "JET_GAIN\|JET_CEIL" src/render/raytrace.wgsl
grep -rn "JET.gain\|JET.ceil" src/ tests/
```

Expected: `jet.ts` says `gain: 0.06, ceil: 8.0`; `raytrace.wgsl` says `0.03` / `4.0`; the third grep returns **nothing** (they are dead constants). Confirm before editing.

- [ ] **Step 2: Correct the values and the false claim**

The shader's values are what actually ships and what the visuals were tuned against, so the TS constants adopt them. In `src/physics/jet.ts`, change line 6 from:

```ts
/** Shared design constants. The WGSL twins hardcode these exact values. */
```

to:

```ts
/**
 * Shared design constants. The WGSL twins hardcode these values; `?parity` covers the ones that
 * enter jetEmission/dopplerBoost, but NOT gain/ceil -- those are render-side scaling applied in
 * raytrace.wgsl and are recorded here for reference only. They drifted out of sync once already.
 */
```

and lines 15-16 from:

```ts
  gain: 0.06,                 // per-dl emissivity -> radiance scale
  ceil: 8.0,                  // clamp on accumulated jet radiance (anti-blowout)
```

to:

```ts
  gain: 0.03,                 // per-dl emissivity -> radiance scale (raytrace.wgsl JET_GAIN)
  ceil: 4.0,                  // clamp on accumulated jet radiance, anti-blowout (JET_CEIL)
```

- [ ] **Step 3: Correct the README**

The current Tier 2B paragraph claims "Blandford–Znajek funnel with synchrotron emission". The code contains no magnetic field, no Poynting flux, no dependence on spin or B (`jetEmission` does not take `a`), and no spectrum — "synchrotron" is a fixed RGB tint. The project's own design doc correctly lists both as out of scope; only the README overstates.

In `README.md`, find the sentence beginning "**Tier 2B — Relativistic Jet (shipped):**" and replace the parenthetical "(Blandford–Znajek funnel with synchrotron emission and propagating knots)" with an accurate description. The paragraph should state that the jet is **phenomenological**: a parabolic funnel wall with a limb-brightening profile, a 1/z emissivity falloff, and propagating knots, and that the one physically derived ingredient is relativistic Doppler beaming, delta = 1/(gamma(1 - beta*mu)). Keep the existing accurate statements about optically-thin additive integration and the sliders.

Write it as flowing prose consistent with the surrounding entries — do not bolt on a disclaimer block, and do not delete the section.

- [ ] **Step 4: Verify**

```bash
npm test
npm run build
npm run verify:gpu
```

Expected: unchanged from Task 5 — no code path changes here (the TS constants are dead, so editing their values cannot alter output).

- [ ] **Step 5: Commit**

```bash
git add src/physics/jet.ts README.md
git commit -m "Correct the desynced jet constants and the README's jet claim

jet.ts declared gain/ceil as 0.06/8.0 while raytrace.wgsl uses 0.03/4.0 --
both exactly 2x off -- under a header asserting the WGSL twins hardcode these
exact values. They are dead constants that parity cannot cover, so they are
now aligned to the shipped values and documented as reference-only.

The README described the jet as a Blandford-Znajek funnel with synchrotron
emission. There is no magnetic field, no Poynting flux, no spin or B
dependence, and no spectrum; the one real physical ingredient is Doppler
beaming. The design doc already scoped BZ and synchrotron out."
git log -1 --format=%B | grep -ci "co-authored"
```

Expected: `0`

---

## Done when

- `npm test` passes (62+, count depends on modified tests); `npm run build` clean.
- `npm run verify:gpu`: `?parity` **exactly 9.690e-7 over 36 cases**, `?shadow` PASS with its radius recorded (may differ from 4.51 M after Task 2 — that is expected and must be explained, not reverted).
- `tests/camera.test.ts` reciprocity passes: a photon shot from r=8 traces back to r=8.
- `grep -rn "Blandford" README.md` returns nothing.
- `grep -n "rh \* 1.001" src/render/raytrace.wgsl` returns nothing.
- All six commits free of `Co-Authored-By`. Nothing pushed; still on `fix/physics-accuracy`.
