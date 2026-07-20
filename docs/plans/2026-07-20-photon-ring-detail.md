# Photon-Ring Detail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the n=1 photon subring render as a distinct arc by fixing the step-budget bug that paints step-starved rays black, using the analytic Kerr critical curve to classify them.

**Architecture:** A new pure-TS `src/physics/shadow.ts` implements the Bardeen spherical-photon-orbit critical curve and a `classify(xi, eta, a)` capture test. `src/render/raytrace.wgsl` gains explicit outcome tracking and a coefficient-for-coefficient WGSL twin of that classifier, applied to rays that exhaust `maxSteps`. A "Detail" slider drives the already-existing `maxSteps` uniform. `?shadow` is upgraded to compare against analytic ground truth.

**Tech Stack:** TypeScript, Vitest, WebGPU/WGSL, Vite, playwright-core (headless GPU verification).

## Global Constraints

- **No `Co-Authored-By` trailer in any git commit.** (Standing user rule — verify with `git log -1 --format=%B` after each commit.)
- **Do not push to origin. Do not merge.** All work stays local on `feat/photon-ring` until the user explicitly directs otherwise.
- **`?parity` must stay bit-identical at maxRelErr = 9.690e-7.** It is math-only; if it moves, something is wrong.
- **The "all features off ⇒ bit-identical" invariant is explicitly waived for this feature** (spec §4). Correcting the artifact necessarily changes pixels near the shadow edge. Even at Detail=1200 the old image does not return, because the exhaustion classifier still applies.
- `UNIFORM_SIZE` stays **96 bytes**. `maxSteps` is already uint slot `i[13]`; no buffer-layout change.
- Geometric units, M = 1, matching `src/physics/kerr.ts`.
- Target hardware: RTX 3050 Laptop 4GB. The 15 fps cap (`TARGET_MS = 67`) absorbs over-budget frames.
- CPU/GPU twin discipline: any formula in both `shadow.ts` and `raytrace.wgsl` must match coefficient-for-coefficient, as `skymap.ts` does today.

## File Structure

| File | Responsibility |
|---|---|
| `src/physics/shadow.ts` (create) | Analytic Kerr critical curve: `criticalXiEta`, `photonShellRange`, `classify`, `shadowBoundary`. Pure TS, no DOM/GPU. |
| `tests/shadow.test.ts` (create) | Unit tests for the above against known analytic results. |
| `src/render/raytrace.wgsl` (modify) | Explicit `resolved` outcome flag; WGSL twin classifier; exhaustion branch. |
| `src/render/uniforms.ts` | **Unchanged** — `maxSteps` already exists at `i[13]`. |
| `src/main.ts` (modify) | `state.maxSteps`, Detail slider listener, pass through to uniform. |
| `index.html` (modify) | Detail slider markup. |
| `src/test/shadow.browser.ts` (modify) | Accept a `maxSteps` override; report analytic comparison. |
| `scripts/verify-gpu.mjs` (modify) | Sweep maxSteps and report the starvation-vs-calibration decomposition. |
| `README.md` (modify) | Document the feature, the waived invariant, and the 0.87 finding. |

---

### Task 1: Analytic critical curve (pure TS)

**Files:**
- Create: `src/physics/shadow.ts`
- Test: `tests/shadow.test.ts`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces:
  - `criticalXiEta(r: number, a: number): [number, number]`
  - `photonShellRange(a: number): [number, number]`
  - `classify(xi: number, eta: number, a: number): "captured" | "escaped"`
  - `shadowBoundary(a: number, incl: number, nSamples: number): [number, number][]` — celestial-plane (α, β) points
  - `A_EPS: number` (= 1e-4)

- [ ] **Step 1: Write the failing test**

Create `tests/shadow.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { criticalXiEta, photonShellRange, classify, shadowBoundary, A_EPS } from "../src/physics/shadow";

describe("Kerr critical curve", () => {
  it("photon shell endpoints match the equatorial photon orbits at a=0.9", () => {
    const [lo, hi] = photonShellRange(0.9);
    expect(lo).toBeCloseTo(1.5579, 3);
    expect(hi).toBeCloseTo(3.9103, 3);
  });

  it("eta vanishes at both shell endpoints", () => {
    const [lo, hi] = photonShellRange(0.9);
    expect(criticalXiEta(lo, 0.9)[1]).toBeCloseTo(0, 6);
    expect(criticalXiEta(hi, 0.9)[1]).toBeCloseTo(0, 6);
  });

  it("xi is strictly decreasing across the shell (precondition for bisection)", () => {
    const [lo, hi] = photonShellRange(0.9);
    let prev = Infinity;
    for (let i = 0; i <= 200; i++) {
      const r = lo + ((hi - lo) * i) / 200;
      const xi = criticalXiEta(r, 0.9)[0];
      expect(xi).toBeLessThan(prev);
      prev = xi;
    }
  });

  it("reduces to the Schwarzschild circle of radius sqrt(27) as a -> 0", () => {
    const [lo, hi] = photonShellRange(0);
    expect(lo).toBeCloseTo(3, 6);
    expect(hi).toBeCloseTo(3, 6);
    // at a=0 the shadow is a circle: eta + xi^2 = 27 on the critical curve
    const [xi, eta] = criticalXiEta(3, 0);
    expect(eta + xi * xi).toBeCloseTo(27, 4);
  });

  it("is continuous across the A_EPS branch cutover", () => {
    const below = criticalXiEta(3, A_EPS * 0.5);
    const above = criticalXiEta(3, A_EPS * 1.5);
    expect(below[1] + below[0] * below[0]).toBeCloseTo(above[1] + above[0] * above[0], 3);
  });

  it("classifies deep-interior rays as captured and distant rays as escaped", () => {
    // xi=0, eta=0 is a radial ray straight into the hole
    expect(classify(0, 0, 0.9)).toBe("captured");
    // a huge Carter constant means a large impact parameter -> escapes
    expect(classify(0, 400, 0.9)).toBe("escaped");
    // xi far outside the shell's xi range escapes
    expect(classify(50, 0, 0.9)).toBe("escaped");
  });

  it("brackets the Schwarzschild capture threshold at b = sqrt(27)", () => {
    // at a->0, pole-on, eta = b^2; capture iff b < sqrt(27)
    expect(classify(0, 27 * 0.98, 1e-6)).toBe("captured");
    expect(classify(0, 27 * 1.02, 1e-6)).toBe("escaped");
  });

  it("the a=0 boundary is a symmetric circle of radius sqrt(27)", () => {
    const pts = shadowBoundary(0, Math.PI / 2, 400);
    const al = pts.map((p) => p[0]);
    expect(Math.min(...al)).toBeCloseTo(-Math.sqrt(27), 6);
    expect(Math.max(...al)).toBeCloseTo(Math.sqrt(27), 6);
    for (const [x, y] of pts) expect(Math.hypot(x, y)).toBeCloseTo(Math.sqrt(27), 6);
  });

  it("the a=0.9 boundary is flattened on the prograde side (the Kerr signature)", () => {
    const al = shadowBoundary(0.9, Math.PI / 2, 400).map((p) => p[0]);
    const mn = Math.min(...al), mx = Math.max(...al);
    expect(mn).toBeCloseTo(-2.829, 2);
    expect(mx).toBeCloseTo(6.794, 2);
    // strongly asymmetric about alpha = 0, unlike Schwarzschild where this is exactly 0
    expect(Math.abs(mn + mx)).toBeGreaterThan(3.5);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run tests/shadow.test.ts`
Expected: FAIL — `Failed to resolve import "../src/physics/shadow"`.

- [ ] **Step 3: Write minimal implementation**

Create `src/physics/shadow.ts`:

```ts
// Analytic Kerr shadow boundary via the spherical-photon-orbit impact parameters (Bardeen 1973).
// Geometric units, M = 1, matching kerr.ts.
//
// Both xi and eta are 0/0 as a -> 0: the eta numerator factors as
//   4*Delta - r*(r-1)^2 = -r*(r-3)^2 + 4a^2
// so at a -> 0, r = 3 the limit is 27*4a^2/(4a^2) = 27 -- correct, but only as a limit. Naive
// evaluation divides by zero, so we branch to the exact Schwarzschild circle below A_EPS.
export const A_EPS = 1e-4;

const delta = (r: number, a: number) => r * r - 2 * r + a * a;

/** Impact parameters (xi, eta) of the spherical photon orbit at Boyer-Lindquist radius r. */
export function criticalXiEta(r: number, a: number): [number, number] {
  if (Math.abs(a) < A_EPS) {
    // Schwarzschild: the critical curve is the circle eta + xi^2 = 27, degenerate at r = 3.
    return [0, 27];
  }
  const d = delta(r, a);
  const xi = (r * r - a * a - r * d) / (a * (r - 1));
  const eta = (r * r * r * (4 * d - r * (r - 1) * (r - 1))) / (a * a * (r - 1) * (r - 1));
  return [xi, eta];
}

/** [prograde, retrograde] equatorial photon-orbit radii -- the endpoints of the photon shell. */
export function photonShellRange(a: number): [number, number] {
  const lo = 2 * (1 + Math.cos((2 / 3) * Math.acos(-a)));
  const hi = 2 * (1 + Math.cos((2 / 3) * Math.acos(a)));
  return [lo, hi];
}

/** Capture test for a null geodesic with conserved impact parameters (xi, eta).
 *
 *  xi_c(r) is strictly decreasing across the photon shell (asserted in tests), so we invert it by
 *  bisection to find the shell radius r* with xi_c(r*) = xi, then compare eta against eta_c(r*).
 *  Fixed iteration count: WGSL has no while-loops, and the twin must match this exactly. */
export function classify(xi: number, eta: number, a: number): "captured" | "escaped" {
  if (Math.abs(a) < A_EPS) {
    // Schwarzschild: capture iff the impact parameter b^2 = eta + xi^2 is inside b_crit^2 = 27.
    return eta + xi * xi < 27 ? "captured" : "escaped";
  }
  const [lo, hi] = photonShellRange(a);
  const xiHi = criticalXiEta(hi, a)[0]; // smallest xi (xi decreasing in r)
  const xiLo = criticalXiEta(lo, a)[0]; // largest xi
  if (xi <= xiHi || xi >= xiLo) return "escaped"; // no turning point in the shell
  let a0 = lo, b0 = hi;
  for (let k = 0; k < 24; k++) {
    const mid = 0.5 * (a0 + b0);
    if (criticalXiEta(mid, a)[0] > xi) a0 = mid;
    else b0 = mid;
  }
  const etaC = criticalXiEta(0.5 * (a0 + b0), a)[1];
  return eta < etaC ? "captured" : "escaped";
}

/** Shadow outline in celestial (alpha, beta) coordinates for a given spin and inclination.
 *  alpha = -xi/sin(i);  beta^2 = eta + a^2 cos^2(i) - xi^2 cot^2(i)  (both +/- beta branches). */
export function shadowBoundary(a: number, incl: number, nSamples: number): [number, number][] {
  const si = Math.sin(incl), ci = Math.cos(incl);
  const pts: [number, number][] = [];
  if (Math.abs(a) < A_EPS) {
    const b = Math.sqrt(27);
    for (let k = 0; k < nSamples; k++) {
      const t = (2 * Math.PI * k) / nSamples;
      pts.push([b * Math.cos(t), b * Math.sin(t)]);
    }
    return pts;
  }
  const [lo, hi] = photonShellRange(a);
  for (let k = 0; k <= nSamples; k++) {
    const r = lo + ((hi - lo) * k) / nSamples;
    const [xi, eta] = criticalXiEta(r, a);
    const b2 = eta + a * a * ci * ci - (xi * xi * ci * ci) / (si * si);
    if (b2 < 0) continue; // this shell radius is not visible at this inclination
    const al = -xi / si, be = Math.sqrt(b2);
    pts.push([al, be]);
    pts.push([al, -be]);
  }
  return pts;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run tests/shadow.test.ts`
Expected: PASS, 9 tests.

Then run the whole suite to confirm nothing regressed:

Run: `npm test`
Expected: PASS, 54 tests (45 existing + 9 new).

- [ ] **Step 5: Commit**

```bash
git add src/physics/shadow.ts tests/shadow.test.ts
git commit -m "Add analytic Kerr critical curve (Bardeen photon-shell impact parameters)"
git log -1 --format=%B | grep -ci "co-authored"   # must print 0
```

---

### Task 2: Quantify the starvation effect before changing the shader

This task is **measurement only** — it produces the evidence that the spec's §3.4 correction is right, and establishes the before-baseline. No production code changes.

**Files:**
- Modify: `src/test/shadow.browser.ts` (add an optional `maxSteps` override)
- Modify: `scripts/verify-gpu.mjs` (sweep and report)

**Interfaces:**
- Consumes: `measureShadow` from `src/test/shadow.browser.ts`.
- Produces: `measureShadow(canvas: HTMLCanvasElement, maxStepsOverride?: number)` — the override defaults to 8000, preserving current behaviour when omitted.

- [ ] **Step 1: Add the override parameter**

In `src/test/shadow.browser.ts`, change the signature and the uniform literal:

```ts
export async function measureShadow(canvas: HTMLCanvasElement, maxStepsOverride = 8000) {
```

and in the `UniformValues` literal replace `maxSteps: 8000,` with `maxSteps: maxStepsOverride,`.

- [ ] **Step 2: Expose the sweep through the validation route**

In `src/main.ts`, the `?shadow` branch currently reads:

```ts
  const { measureShadow } = await import("./test/shadow.browser");
  const res = await measureShadow(canvas);
```

Replace with:

```ts
  const { measureShadow } = await import("./test/shadow.browser");
  const stepsParam = new URLSearchParams(location.search).get("steps");
  const res = await measureShadow(canvas, stepsParam ? +stepsParam : 8000);
```

- [ ] **Step 3: Measure the sweep**

Start the dev server if not running (`npm run dev`), then run:

```bash
node -e '
const { chromium } = require("playwright-core");
(async () => {
  const b = await chromium.launch({ channel: "chrome", headless: true,
    args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist"] });
  const p = await b.newPage({ viewport: { width: 900, height: 700 } });
  for (const s of [1200, 2400, 4800, 8000]) {
    await p.goto(`http://localhost:5173/?shadow&steps=${s}`, { waitUntil: "load" });
    await p.waitForFunction(() => document.body.innerText.includes("SHADOW"), { timeout: 60000 });
    const t = await p.evaluate(() => document.body.innerText.replace(/\s+/g, " "));
    console.log(`steps=${String(s).padStart(5)}  ${t}`);
  }
  await b.close();
})();
'
```

Expected: four lines. **Prediction to test:** `apparent radius` is flat across all four (the `?shadow` route was never starved). If it *does* move, the spec's correction is itself wrong and that must be reported before proceeding.

- [ ] **Step 4: Record the finding**

Append the measured table to `docs/specs/2026-07-20-photon-ring-detail-design.md` under §3.4, replacing the word "predicted" with the actual observation. State plainly whether the prediction held.

- [ ] **Step 5: Commit**

```bash
git add src/test/shadow.browser.ts src/main.ts docs/specs/2026-07-20-photon-ring-detail-design.md
git commit -m "Measure shadow radius vs step budget; confirm ?shadow was never starved"
git log -1 --format=%B | grep -ci "co-authored"   # must print 0
```

---

### Task 3: WGSL classifier twin (dormant)

Add the classifier to the shader **without wiring it to anything**, so it can be verified in isolation before it changes any pixels.

**Files:**
- Modify: `src/render/raytrace.wgsl`

**Interfaces:**
- Consumes: the math from `src/physics/shadow.ts` (Task 1) — must match coefficient-for-coefficient.
- Produces: WGSL `fn classifyCaptured(xi: f32, eta: f32, a: f32) -> bool` (true = captured).

- [ ] **Step 1: Add the twin functions**

Insert into `src/render/raytrace.wgsl`, immediately before `@compute @workgroup_size(8,8) fn main`:

```wgsl
// ---- Analytic Kerr critical curve. Twin of src/physics/shadow.ts -- keep coefficients in sync. ----
const A_EPS = 1e-4;

fn criticalXi(r: f32, a: f32) -> f32 {
  let d = r*r - 2.0*r + a*a;
  return (r*r - a*a - r*d) / (a*(r - 1.0));
}
fn criticalEta(r: f32, a: f32) -> f32 {
  let d = r*r - 2.0*r + a*a;
  return (r*r*r * (4.0*d - r*(r - 1.0)*(r - 1.0))) / (a*a*(r - 1.0)*(r - 1.0));
}

// Capture test from the conserved impact parameters. Bisection count (24) must match shadow.ts.
fn classifyCaptured(xi: f32, eta: f32, a: f32) -> bool {
  if (abs(a) < A_EPS) { return eta + xi*xi < 27.0; }
  let lo = 2.0 * (1.0 + cos((2.0/3.0) * acos(-a)));
  let hi = 2.0 * (1.0 + cos((2.0/3.0) * acos(a)));
  let xiHi = criticalXi(hi, a);
  let xiLo = criticalXi(lo, a);
  if (xi <= xiHi || xi >= xiLo) { return false; }
  var a0 = lo; var b0 = hi;
  for (var k = 0; k < 24; k++) {
    let mid = 0.5 * (a0 + b0);
    if (criticalXi(mid, a) > xi) { a0 = mid; } else { b0 = mid; }
  }
  return eta < criticalEta(0.5*(a0 + b0), a);
}
```

- [ ] **Step 2: Verify the shader still compiles and nothing changed**

Run: `npm run build`
Expected: build succeeds.

Run: `BASE=http://localhost:5173 npm run verify:gpu`
Expected: `?parity` PASS at maxRelErr **9.690e-7** (unchanged), `?shadow` PASS, sky asset 200.

Note: `layout:"auto"` strips bindings not statically used by `main`, but this task adds no bindings, so nothing is stripped. The functions are dead code until Task 4; WGSL permits unused functions.

- [ ] **Step 3: Commit**

```bash
git add src/render/raytrace.wgsl
git commit -m "Add dormant WGSL critical-curve classifier (twin of physics/shadow.ts)"
git log -1 --format=%B | grep -ci "co-authored"   # must print 0
```

---

### Task 4: Wire the classifier to exhausted rays

**Files:**
- Modify: `src/render/raytrace.wgsl`

**Interfaces:**
- Consumes: `classifyCaptured(xi, eta, a) -> bool` (Task 3).
- Produces: no new exports; changes rendered output near the shadow edge.

- [ ] **Step 1: Compute the Carter constant at ray setup**

In `main`, immediately after the existing line `let xi = -alpha * sin(i);` (currently
`src/render/raytrace.wgsl:238`), add:

```wgsl
  // Carter constant from the Bardeen screen coordinates: eta = beta^2 + xi^2*cot^2(i) - a^2*cos^2(i).
  // Both xi and eta are conserved, so they classify a ray regardless of where integration stopped.
  let ci = cos(i); let si = sin(i);
  let eta = beta*beta + xi*xi*(ci*ci)/max(si*si, 1e-8) - a*a*ci*ci;
```

- [ ] **Step 2: Add the outcome flag**

Change the declaration (currently `src/render/raytrace.wgsl:249`) from:

```wgsl
  var color = vec3<f32>(0.0);
```

to:

```wgsl
  var color = vec3<f32>(0.0);
  var resolved = false; // set by each real termination; false => the step budget ran out
```

- [ ] **Step 3: Mark each real termination**

Three edits inside the integrator loop. Disk hit — change:

```wgsl
        color = sampleColor(Tobs) * pow(g * Tn, 4.0) * E;
        break;
```

to:

```wgsl
        color = sampleColor(Tobs) * pow(g * Tn, 4.0) * E;
        resolved = true;
        break;
```

Horizon capture — change:

```wgsl
    if (s.x.y <= rh * 1.001) { color = vec3(0.0); break; }   // captured -> shadow
```

to:

```wgsl
    if (s.x.y <= rh * 1.001) { color = vec3(0.0); resolved = true; break; }   // captured -> shadow
```

Escape — change:

```wgsl
      color = mix(starfield(dir), skySample(dir) * U.skyStrength, mixT);
      break;
```

to:

```wgsl
      color = mix(starfield(dir), skySample(dir) * U.skyStrength, mixT);
      resolved = true;
      break;
```

- [ ] **Step 4: Classify the leftovers**

Immediately after the closing brace of the integrator `for` loop, insert:

```wgsl
  // Budget exhausted without a real termination. Previously these rays kept color = vec3(0) and so
  // rendered as shadow -- a step-budget artifact that swallowed the n=1 photon subring. Classify
  // them from their conserved (xi, eta) instead: the sign of p_r at an arbitrary cutoff is
  // effectively random for a winding ray and would produce salt-and-pepper noise.
  if (!resolved) {
    if (classifyCaptured(xi, eta, a)) {
      color = vec3<f32>(0.0);
    } else {
      let th = s.x.z; let ph = s.x.w;
      let dir = normalize(vec3<f32>(sin(th)*cos(ph), sin(th)*sin(ph), cos(th)));
      let mixT = clamp(U.skyStrength, 0.0, 1.0);
      color = mix(starfield(dir), skySample(dir) * U.skyStrength, mixT);
    }
  }
```

- [ ] **Step 5: Verify**

Run: `npm run build`
Expected: build succeeds.

Run: `BASE=http://localhost:5173 npm run verify:gpu`
Expected: `?parity` PASS at **9.690e-7 exactly** (math-only, must not move). `?shadow` PASS. Sky 200.

Capture a visual before/after at the default view to confirm the dark region near the shadow edge changed:

```bash
node -e '
const { chromium } = require("playwright-core");
(async () => {
  const b = await chromium.launch({ channel: "chrome", headless: true,
    args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist"] });
  const p = await b.newPage({ viewport: { width: 1000, height: 680 } });
  await p.goto("http://localhost:5173/", { waitUntil: "load" });
  await p.waitForTimeout(8000);
  await p.screenshot({ path: "task4-after.png" });
  await b.close();
})();
'
```

Expected: renders without page errors; the black wedge below the shadow now shows lensed sky rather than flat black.

- [ ] **Step 6: Commit**

```bash
git add src/render/raytrace.wgsl
git commit -m "Classify step-exhausted rays by conserved (xi, eta) instead of painting them black"
git log -1 --format=%B | grep -ci "co-authored"   # must print 0
```

---

### Task 5: Detail slider

**Files:**
- Modify: `index.html`
- Modify: `src/main.ts`

**Interfaces:**
- Consumes: the existing `maxSteps` uniform field (`UniformValues.maxSteps`, uint slot `i[13]`).
- Produces: `state.maxSteps` (number, default 4800).

- [ ] **Step 1: Add the slider markup**

In `index.html`, immediately after the Sky control block (the `<div class="ctrl">` containing
`id="sky"`) and before the Play/Pause control, insert:

```html
    <div class="ctrl">
      <div class="row"><label>Detail</label><span class="val"><b id="detailv">4800</b> steps</span></div>
      <input id="detail" type="range" min="1200" max="9600" step="600" value="4800">
    </div>
```

- [ ] **Step 2: Add state and listener**

In `src/main.ts`, add `maxSteps: 4800` to the `state` object literal (currently line 37), so it ends:

```ts
jetKnots: 0.7, skyStrength: 1.0, maxSteps: 4800 };
```

Then immediately after the existing Sky listener block:

```ts
  const sky = $("sky") as HTMLInputElement, skyv = $("skyv");
  sky.addEventListener("input", () => { state.skyStrength = +sky.value; skyv.textContent = state.skyStrength.toFixed(2); reset(); });
```

add:

```ts
  const detail = $("detail") as HTMLInputElement, detailv = $("detailv");
  detail.addEventListener("input", () => { state.maxSteps = +detail.value; detailv.textContent = String(state.maxSteps); reset(); });
```

- [ ] **Step 3: Feed it to the uniform**

In the uniform literal in the render loop, change:

```ts
        time: simTime, frame: sample, reset: sample === 0 ? 1 : 0, maxSteps: 1200,
```

to:

```ts
        time: simTime, frame: sample, reset: sample === 0 ? 1 : 0, maxSteps: state.maxSteps,
```

- [ ] **Step 4: Verify**

Run: `npm run build`
Expected: build succeeds, no TypeScript errors.

Run: `npm test`
Expected: PASS, 54 tests.

Run: `BASE=http://localhost:5173 npm run verify:gpu`
Expected: parity 9.690e-7, shadow PASS, sky 200.

Manually confirm the slider moves the render: load `http://localhost:5173/`, drag Detail to 1200 and to 9600, and confirm the readout updates and the image re-converges.

- [ ] **Step 5: Commit**

```bash
git add index.html src/main.ts
git commit -m "Add Detail slider driving the maxSteps step budget (default 4800)"
git log -1 --format=%B | grep -ci "co-authored"   # must print 0
```

---

### Task 6: Analytic comparison in ?shadow, and docs

**Files:**
- Modify: `src/test/shadow.browser.ts`
- Modify: `scripts/verify-gpu.mjs`
- Modify: `README.md`

**Interfaces:**
- Consumes: `classify` and `A_EPS` from `src/physics/shadow.ts` (Task 1); `measureShadow(canvas, maxStepsOverride)` (Task 2).
- Produces: `measureShadow` return object gains `analyticRadiusM: number` and `calibration: number`.

- [ ] **Step 1: Compare against analytic ground truth**

In `src/test/shadow.browser.ts`, add the import:

```ts
import { classify } from "../physics/shadow";
```

Replace the `bCrit` line and the return object with:

```ts
  // Analytic ground truth. At a=0 the critical curve is the circle b = sqrt(27); we locate it by
  // bisecting `classify` in b^2 rather than hardcoding, so this stays honest if the model changes.
  let blo = 0, bhi = 20;
  for (let k = 0; k < 40; k++) {
    const bmid = 0.5 * (blo + bhi);
    if (classify(0, bmid * bmid, 0) === "captured") blo = bmid; else bhi = bmid;
  }
  const analyticRadiusM = 0.5 * (blo + bhi);
  const bCrit = Math.sqrt(27);                // = 5.196 M, the ideal Schwarzschild shadow radius
  const hasShadow = shadowPx > 4;
  const hasDisk = brightOnColumn > h * 0.1;
  const plausible = shadowRadiusM > 1.0 && shadowRadiusM < bCrit * 1.6;
  const ok = hasShadow && hasDisk && plausible;
  return {
    ok, shadowPx, hasShadow, hasDisk,
    shadowRadiusM: +shadowRadiusM.toFixed(2),
    bCritM: +bCrit.toFixed(2),
    analyticRadiusM: +analyticRadiusM.toFixed(3),
    scaleVsBcrit: +(shadowRadiusM / bCrit).toFixed(2),
    calibration: +(shadowRadiusM / analyticRadiusM).toFixed(3),
  };
```

- [ ] **Step 2: Replace the stale comment**

The docblock at the top of `src/test/shadow.browser.ts` claims "a roughly constant ~2x scale factor"
while the measured value is 0.87. Replace that sentence with the measured, explained account:

```ts
 *  The rendered radius differs from the analytic sqrt(27)*M by a constant camera-calibration factor
 *  (`calibration` in the return value, measured ~0.87). This is NOT a physics error and NOT a step-
 *  budget artifact -- this route runs at a high step budget and the value is flat across budgets
 *  (see docs/specs/2026-07-20-photon-ring-detail-design.md §3.4). It comes from the Tier-1 camera
 *  mapping screen coordinates to photon initial conditions heuristically (p_theta = beta,
 *  p_phi = -alpha*sin i) with no normalization, so `fovScale` is not calibrated in true M units.
 *  Recalibrating to a normalized Bardeen camera would rescale every image in the project and is
 *  tracked as a separate follow-up.
```

- [ ] **Step 3: Surface the new fields in the ?shadow page output**

The displayed string is built in `src/main.ts`, not in `shadow.browser.ts` — without this edit the
new fields are computed but never rendered. In `src/main.ts`, replace the `?shadow` template:

```ts
  document.body.innerHTML = `<pre style="color:${res.ok ? "#6f6" : "#f66"};font-size:18px;padding:20px">
SHADOW ${res.ok ? "PASS" : "FAIL"} (structural) — centred dark shadow=${res.hasShadow}, ringed by disk=${res.hasDisk}
apparent radius ≈ ${res.shadowRadiusM} M  (ideal sqrt(27) = ${res.bCritM} M; camera scale factor ≈ ${res.scaleVsBcrit})</pre>`;
```

with:

```ts
  document.body.innerHTML = `<pre style="color:${res.ok ? "#6f6" : "#f66"};font-size:18px;padding:20px">
SHADOW ${res.ok ? "PASS" : "FAIL"} (structural) — centred dark shadow=${res.hasShadow}, ringed by disk=${res.hasDisk}
apparent radius ≈ ${res.shadowRadiusM} M; analytic critical curve = ${res.analyticRadiusM} M
camera calibration factor = ${res.calibration} (not physics — see spec §3.4)</pre>`;
```

Then in `scripts/verify-gpu.mjs`, the console line truncates at 150 characters, which the longer
output now exceeds. Change:

```js
  console.log(`${ok ? "✓ PASS" : "✗ FAIL"}  ${path}\n        ${txt.slice(0, 150)}`);
```

to:

```js
  console.log(`${ok ? "✓ PASS" : "✗ FAIL"}  ${path}\n        ${txt.slice(0, 300)}`);
```

- [ ] **Step 4: Update the README**

In `README.md`, immediately after the "Lensed sky background (shipped)" paragraph, add:

```markdown
**Photon-ring detail (shipped):** rays that exhausted the integrator's step budget previously fell
out of the loop still holding their initial black colour, so step-starved rays rendered as shadow —
an artifact that swallowed the n=1 photon subring (at a=0.9 one winding near the prograde photon
orbit r≈1.56M costs ~3900 steps against a budget of 1200). Exhausted rays are now classified by
their conserved impact parameters (ξ, η) against the analytic Kerr critical curve, and a "Detail"
slider exposes the step budget (default 4800). Note this is the first feature that does **not**
preserve the project's bit-identical-when-off property: correcting the artifact necessarily changes
pixels near the shadow edge. `?parity` is math-only and is unchanged.

The `?shadow` diagnostic now compares the rendered boundary against the analytic critical curve.
The residual ~0.87 factor is camera calibration, not physics: the Tier-1 camera maps screen
coordinates to photon initial conditions without normalization, so `fovScale` is not in true M
units. A normalized Bardeen camera is a tracked follow-up.
```

Also update the Out-of-scope line to mention the camera recalibration follow-up.

- [ ] **Step 5: Verify everything**

Run: `npm test`
Expected: PASS, 54 tests.

Run: `npm run build`
Expected: succeeds.

Run: `BASE=http://localhost:5173 npm run verify:gpu`
Expected: parity **9.690e-7**, shadow PASS with `calibration` reported, sky 200.

- [ ] **Step 6: Commit**

```bash
git add src/test/shadow.browser.ts scripts/verify-gpu.mjs README.md
git commit -m "Compare rendered shadow against analytic critical curve; document the 0.87 as camera calibration"
git log -1 --format=%B | grep -ci "co-authored"   # must print 0
```

---

## Done criteria

- 54 tests pass (45 existing + 9 new in `tests/shadow.test.ts`).
- `?parity` reports maxRelErr **9.690e-7**, unchanged.
- `?shadow` PASSes and reports `analyticRadiusM` ≈ 5.196 and a `calibration` factor.
- The Detail slider changes the step budget live and the render re-converges.
- The black wedge below the shadow shows lensed sky instead of flat black.
- No commit contains a `Co-Authored-By` trailer.
- Nothing pushed to origin; `feat/photon-ring` not merged.
