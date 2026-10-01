# Light-Travel Delay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shade every pixel at its own emission time (scene time minus light-travel delay), with a
"Light delay" checkbox (on by default), so photon-ring subimages echo the disk tens of M later.

**Architecture:** The integrator already carries coordinate time `t` (`s.x.x`). The shader derives
`delay = −t_e − rObs` at each emission point and an emission time `U.time − U.lightDelay·delay`
that replaces `U.time` in the disk pattern phase, breathing and jet knots (per jet sample). The
geodesic cache keeps 16-byte entries by storing the delay in place of g and recomputing g in the
shade pass through a shared pixel→impact helper. Motion becomes playback speed in `main.ts`.

**Tech Stack:** TypeScript, WebGPU/WGSL, Vite, Vitest, Playwright (headless Chrome gates).

**Spec:** `docs/specs/2026-10-01-light-travel-delay-design.md`

## Global Constraints

- Gates green after every task: `npm test`, `npm run build`, `npm run verify:gpu` (dev server on
  :5173; any console warning fails a route), `node scripts/probe-axis.mjs`, `node scripts/probe-scale.mjs`.
- `?golden` must stay **bit-identical** to the current hashes (default b3d2e285, jet-off 0029a2dc,
  edge-on 4b383b5e, face-on-jet 8c049cb5): golden scenes run with `lightDelay: 0` and timeScale 1.
  Do NOT re-record golden in this plan.
- `?cachecheck` PASS (cached = live ≤ ~4e-7), including the new delay-on scene.
- `delay = −t_e − rObs`; `T_emit = U.time − U.lightDelay · delay`; `lightDelay` ∈ {0, 1}.
- Cache entries stay 16 bytes; the light-delay checkbox never rebuilds the cache.
- No Co-Authored-By trailer on commits in this repo.

## Review Focus

1. **Toggling Light delay while cached** must re-shade at once and never leave `cached` mode
   (delays are always stored). Pinned by Task 3's verify-gpu step.
2. **Motion slider** is playback speed: Motion 0 freezes the animation (no pattern drift), and
   changing Motion mid-play must not jump the pattern. Pinned by Task 3's verify-gpu step (Motion 0:
   two screenshots 1.5 s apart agree within 1 grey level).
3. **Delay with the jet on in cached mode**: jet replay must reproduce per-sample emission times.
   Pinned by Task 3's delay-on `?cachecheck` scene (SCENES[0] has the jet on).
4. **Delay off is today's renderer exactly** (comparison mode is honest). Pinned by `?golden`
   bit-identity in Task 2.
5. **Presets and Default view with delay on** stay lit and warning-free. Pinned by the existing
   verify-gpu preset loop, which runs with the default (delay on) after Task 3.

---

## File Structure

- Modify `src/render/uniforms.ts` (+ test) — `lightDelay` at float index 31 (byte 124).
- Modify `src/render/raytrace.wgsl` — struct field, `pixelImpact`, `emitTime`, `shadeDisk(…, tEmit)`,
  `emissionFieldE(…, tEmit)`, `knotsJ` without `U.timeScale`, `jetStep` per-sample time, DISK payload
  delay, `shade` recomputes g.
- Modify `src/test/scenes.ts`, `src/test/cachecheck.browser.ts` — `lightDelay` in scenes; delay-on scene.
- Modify `src/main.ts`, `index.html` — checkbox; `simTime` advanced by Motion.
- Modify `scripts/verify-gpu.mjs` — toggle-while-cached and Motion-0 checks. Create `scripts/shot-delay.mjs`.
- Create `tests/delay-physics.test.ts` — coordinate-time physics the feature relies on.
- Docs: `README.md`, `docs/ROADMAP.md`, spec status.

---

### Task 1: Coordinate-time physics checks (CPU twin)

**Files:**
- Create: `tests/delay-physics.test.ts`

**Interfaces:**
- Consumes: `screenToState(alpha, beta, a, incl, rObs)` (camera.ts), `stepGeodesic(s, a, dl, hTol, maxRetry)` (trace.ts). State layout `[t, r, θ, φ, p_t, p_r, p_θ, p_φ]`.
- Produces: nothing new (characterization tests of the integrator's `t`, which Tasks 2–3 rely on).

Spec §6 also lists "a far-side disk point is delayed more in its secondary image than its primary";
the second test below covers the same physics quantitatively (each extra half-orbit of a subimage
adds π√27 M), so it stands in for it (ledger this as a ruling).

These tests characterise existing behaviour, so they are expected to PASS on first run; their job is
to pin the physics before the shader starts depending on it. If either fails, stop: the integrator's
time is wrong and the feature's premise with it.

- [ ] **Step 1: Write the tests** — `tests/delay-physics.test.ts`

```ts
import { describe, it, expect } from "vitest";
import { screenToState } from "../src/physics/camera";
import { stepGeodesic } from "../src/physics/trace";

/** Fine-step integration (independent of the renderer's controller) until stop(s) is true. */
function integrate(s0: Float64Array, a: number, stop: (prev: Float64Array, next: Float64Array) => boolean) {
  let s = s0;
  for (let k = 0; k < 400000; k++) {
    const r = s[1], dl = r > 60 ? Math.min(2, 0.01 * r) : 0.005;
    const n = stepGeodesic(s, a, dl, 1e-9, 30).s;
    if (stop(s, n)) return [s, n] as const;
    s = n;
  }
  throw new Error("did not stop");
}

describe("light-travel time (coordinate t along the backward ray)", () => {
  it("a = 0 radial ray from r = 1000 to r = 10 takes r*(1000) - r*(10) = 999.652 M", () => {
    const rStar = (r: number) => r + 2 * Math.log(r / 2 - 1);
    const [p, n] = integrate(screenToState(0, 0, 0, Math.PI / 2, 1000), 0, (_, nx) => nx[1] <= 10);
    const f = (p[1] - 10) / (p[1] - n[1]), t10 = p[0] + f * (n[0] - p[0]);
    expect(t10).toBeLessThan(0); // past-directed: t decreases from the camera's t = 0
    expect(-t10).toBeCloseTo(rStar(1000) - rStar(10), 3);
  });
  it("a near-critical a = 0 ray crosses the equator every pi*sqrt(27) M near r = 3M (subimage delay)", () => {
    const b = Math.sqrt(27) * (1 + 1e-7), inc = (10 * Math.PI) / 180;
    let s = screenToState(0.3, Math.sqrt(b * b - 0.09), 0, inc, 1000);
    const cross: number[] = [];
    for (let k = 0; k < 400000 && cross.length < 5; k++) {
      const r = s[1], dl = r > 60 ? Math.min(2, 0.01 * r) : 0.005;
      const n = stepGeodesic(s, 0, dl, 1e-9, 30).s;
      if ((s[2] - Math.PI / 2) * (n[2] - Math.PI / 2) < 0 && n[1] < 4) {
        const f = (s[2] - Math.PI / 2) / (s[2] - n[2]); cross.push(s[0] + f * (n[0] - s[0]));
      }
      s = n; if (s[1] < 2.01 || s[1] > 1100) break;
    }
    expect(cross.length).toBeGreaterThanOrEqual(4);
    for (let i = 1; i < cross.length; i++) {
      expect(Math.abs(Math.abs(cross[i] - cross[i - 1]) / (Math.PI * Math.sqrt(27)) - 1)).toBeLessThan(0.03);
    }
  }, 120000);
});
```

- [ ] **Step 2: Run** — `npx vitest run tests/delay-physics.test.ts`
Expected: PASS (2 tests). Measured while writing the spec: crossings 16.325 / 16.324 / 16.326 M vs
π√27 = 16.324 M.

- [ ] **Step 3: Commit**

```bash
git add tests/delay-physics.test.ts
git commit -m "Pin the coordinate-time physics light-travel delay relies on"
```

---

### Task 2: Emission time in the shader (delay off = today, bit for bit)

**Files:**
- Modify: `src/render/uniforms.ts`, `tests/uniforms.test.ts`, `src/render/raytrace.wgsl`,
  `src/test/scenes.ts`, `src/main.ts` (uniform literal only), `src/test/shadow.browser.ts`, `scripts/bench.mjs`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `UniformValues.lightDelay: number` (required, packed at float index 31, byte 124); WGSL
  `U.lightDelay`; `pixelImpact(pix: vec2<u32>, jit: vec2<f32>) -> vec2<f32>`;
  `emitTime(delay: f32) -> f32`; `shadeDisk(rHit, phiHit, g, a, tEmit)`; DISK payload/entry
  `(rHit, phiHit, delay)`; `Scene.lightDelay?: number` (default 0) in scenes.ts.

- [ ] **Step 1: Failing uniforms test** — in `tests/uniforms.test.ts` add `lightDelay: 1,` to both
`UniformValues` literals (next to `lumNorm`) and in the first test:
```ts
    expect(dv.getFloat32(124, true)).toBeCloseTo(1);    // lightDelay (index 31)
```
Run `npx vitest run tests/uniforms.test.ts` → FAIL (reads 0).

- [ ] **Step 2: Pack it** — `src/render/uniforms.ts`: header comment line
`//         + lightDelay (1, 0/1 light-travel delay)                   -> 24 floats`; field
```ts
  /** 1 = shade at each pixel's emission time (light-travel delay), 0 = one global instant. */
  lightDelay: number;
```
after `lumNorm: number;`; `export const UNIFORM_FLOATS = 24, UNIFORM_UINTS = 8;` (size stays 128);
`f[31] = u.lightDelay;` after `f[30] = u.lumNorm;`. Run the test → PASS.

- [ ] **Step 3: Callers get `lightDelay`** (TypeScript will flag every literal):
  - `src/test/scenes.ts`: `Scene` gains `lightDelay?: number;`; in `sceneUniforms` add
    `lightDelay: s.lightDelay ?? 0,` next to `lumNorm`.
  - `src/test/shadow.browser.ts`: add `lightDelay: 0,` next to `lumNorm`.
  - `scripts/bench.mjs`: add `lightDelay: 1,` next to `lumNorm` (benchmark the default).
  - `src/main.ts` uniforms literal: add `lightDelay: 0,` for now (Task 3 wires the checkbox).
  Run `npm run build` → clean.

- [ ] **Step 4: Shader** — `src/render/raytrace.wgsl`:
  - struct: after `lumNorm: f32,` add `lightDelay: f32,`.
  - add near the top-level helpers (before `fn traceRay`):
```wgsl
// Pixel -> screen impact parameters (alpha, beta) in M with sub-pixel jitter. Shared by traceRay and
// the cache's shade pass (which recomputes g from them), so both see the same xi bit for bit.
fn pixelImpact(pix: vec2<u32>, jit: vec2<f32>) -> vec2<f32> {
  let aspect = U.res.x / U.res.y;
  let ndc = (vec2<f32>(f32(pix.x), f32(pix.y)) + 0.5 + jit) / U.res * 2.0 - 1.0;
  return vec2<f32>(ndc.x * U.fovScale * aspect, -ndc.y * U.fovScale);
}
// Light-travel delay (spec 2026-10-01): the backward ray starts at t = 0 and t decreases, so an
// emitter at coordinate time t_e is seen delay = -t_e - rObs later than a reference at the camera's
// distance (the constant rObs keeps values in tens of M). Emission time of what this pixel shows:
fn emitTime(delay: f32) -> f32 { return U.time - U.lightDelay * delay; }
```
  - `traceRay`: replace the four lines computing `aspect`, `ndc`, `alpha`, `beta` with
```wgsl
  let ab = pixelImpact(pix, jit);
  let alpha = ab.x; let beta = ab.y;
```
  - disk hit block: replace
```wgsl
        let g = diskG(rHit, xi, a);
        let phiHit = mix(s.x.w, sNew.x.w, frac);     // azimuth of the emitting matter
        color = shadeDisk(rHit, phiHit, g, a);
        out.kind = KIND_DISK; out.payload = vec3<f32>(rHit, phiHit, g);
```
    with
```wgsl
        let g = diskG(rHit, xi, a);
        let phiHit = mix(s.x.w, sNew.x.w, frac);     // azimuth of the emitting matter
        let delay = -mix(s.x.x, sNew.x.x, frac) - U.rObs;
        color = shadeDisk(rHit, phiHit, g, a, emitTime(delay));
        out.kind = KIND_DISK; out.payload = vec3<f32>(rHit, phiHit, delay);
```
    and update the `TraceOut` comment to `// DISK: (rHit, phiHit, delay); SKY: …`.
  - `shadeDisk`: signature `fn shadeDisk(rHit: f32, phiHit: f32, g: f32, a: f32, tEmit: f32) -> vec3<f32>`;
    `let psi = phiHit - Om * tEmit;                // co-rotating pattern phase at emission`;
    `let E = emissionFieldE(rHit, psi, tEmit);`.
  - `emissionFieldE(rHit: f32, psi: f32, tEmit: f32)`: breathing uses `tEmit` instead of `U.time`.
  - `knotsJ`: `let phase = JET_KZ * abs(z) - JET_VKNOT * t;` (time is already scene time; Motion is
    playback speed, Task 3).
  - `jetStep` loop: compute the sample's time with the same chord fraction as its position:
```wgsl
    let tS = select(s.x.x + (sNew.x.x - s.x.x) * (f32(k) / f32(n)), s.x.x, k == 0u);
    let e = jetEmissionJ(q.x, q.y, emitTime(-tS - U.rObs));
```
    (replacing `let e = jetEmissionJ(q.x, q.y, U.time);`).
  - `shade` entry point, DISK branch:
```wgsl
  if (kind == KIND_DISK) {
    // g is recomputed (the entry's third slot holds the delay): same helper and camera fragment as
    // traceRay, so xi and g match the live trace.
    let ab = pixelImpact(gid.xy, fixedJitter(U.setIndex));
    let g = diskG(e.p0, cameraXiEta(ab.x, ab.y, U.a, U.incl).x, U.a);
    color = shadeDisk(e.p0, e.p1, g, U.a, emitTime(e.p2));
  }
```
  - Entry comment: `struct Entry { word: u32, p0: f32, p1: f32, p2: f32 }; // DISK p = (rHit, phiHit, delay)`.
  - Search the file for any other `U.timeScale` read and remove the multiplication (there are two:
    pattern phase and knots). `U.timeScale` then has no readers (the field stays in the layout).

- [ ] **Step 5: Gates** — `npm test`, `npm run build`, then (dev server on :5173) `npm run verify:gpu`.
Expected: all PASS with `?golden` showing the UNCHANGED hashes above (bit-identical: lightDelay 0,
timeScale 1), `?cachecheck` PASS. A golden mismatch means the delay-off path is not today's
renderer — or that the GPU compiler fused the reworded time expressions differently (FMA
contraction of `phiHit - Om * tEmit` vs `phiHit - Om * U.time * U.timeScale`). Tell them apart with
an A/B pixel capture (temporarily store `readbackAccum()` per scene on `window` in
`golden.browser.ts`, capture on `main` and on this branch, compare): every pixel within 1e-6
relative = contraction → re-record golden with a ledgered ruling quoting the max difference;
anything larger is a bug → debug, do not re-record. Then `node scripts/probe-axis.mjs`, `node scripts/probe-scale.mjs` → PASS.

- [ ] **Step 6: Commit**

```bash
git add src/render/uniforms.ts tests/uniforms.test.ts src/render/raytrace.wgsl src/test/scenes.ts src/test/shadow.browser.ts scripts/bench.mjs src/main.ts
git commit -m "Shade at the emission time: light-travel delay in the shader and cache (off = today)"
```

---

### Task 3: Delay on — cache gate, checkbox, Motion as playback speed

**Files:**
- Modify: `src/test/cachecheck.browser.ts`, `src/main.ts`, `index.html`, `scripts/verify-gpu.mjs`
- Create: `scripts/shot-delay.mjs`

**Interfaces:**
- Consumes: Task 2 (`Scene.lightDelay`, `UniformValues.lightDelay`).
- Produces: DOM `#ldelay` (checkbox, checked); `state.lightDelay: boolean`; delay-on cachecheck scene "delay".

- [ ] **Step 1: Delay-on cache gate (test first)** — in `src/test/cachecheck.browser.ts` add to the
checks list, after `"overflow"`:
```ts
  { ...withSky(SCENES[0], "delay"), lightDelay: 1 },       // light-travel delay on (jet, spots, turbulence)
```
Run `npm run verify:gpu` (or the route `/?cachecheck`) → the `delay` line must be `ok`. It exercises
per-sample jet times in replay and the recomputed g; if it is BAD, the cache and live paths disagree:
debug Task 2's shade branch / jetStep, do not loosen the gate.

- [ ] **Step 2: verify-gpu checks (test first)** — in `scripts/verify-gpu.mjs`, inside the preset
block after the mass-nudge check (before `const preDiag = diagSince(dPre);`), add:
```js
// Light delay is shading-only: toggling it while cached must re-shade without a rebuild (Review Focus 1).
await page.selectOption("#preset", "default");
await waitMode("cached");
await page.waitForTimeout(800);
let delayLeft = false;
await page.evaluate(() => { const c = document.getElementById("ldelay"); c.checked = !c.checked; c.dispatchEvent(new Event("change", { bubbles: true })); });
for (let k = 0; k < 20; k++) { await page.waitForTimeout(75); if (!(await page.evaluate(() => document.getElementById("cmode").textContent)).startsWith("cached")) delayLeft = true; }
await page.evaluate(() => { const c = document.getElementById("ldelay"); c.checked = !c.checked; c.dispatchEvent(new Event("change", { bubbles: true })); });
preSteps.push(`delay toggle ${!delayLeft ? "ok" : "FAILED"} (stayed cached ${!delayLeft})`);
if (delayLeft) failed = true;
// Motion is playback speed: at 0 the scene freezes (Review Focus 2).
await page.evaluate(() => { const m = document.getElementById("ts"); m.value = "0"; m.dispatchEvent(new Event("input", { bubbles: true })); });
await page.waitForTimeout(1500);
const f0 = await meanBrightness(); await page.waitForTimeout(1500); const f1 = await meanBrightness();
const frozen = Math.abs(f1 - f0) < 1;
preSteps.push(`motion 0 ${frozen ? "ok" : "FAILED"} (${f0.toFixed(2)} -> ${f1.toFixed(2)})`);
if (!frozen) failed = true;
await page.evaluate(() => { const m = document.getElementById("ts"); m.value = "1"; m.dispatchEvent(new Event("input", { bubbles: true })); });
```
Run `ONLY_APP=1 node scripts/verify-gpu.mjs` → FAIL (no `#ldelay`: `Cannot read properties of null`).

- [ ] **Step 3: Checkbox** — `index.html`, after the Motion control:
```html
    <div class="ctrl">
      <label class="row" style="cursor:pointer"><span>Light delay</span>
        <input id="ldelay" type="checkbox" checked style="accent-color:var(--accent)"></label>
    </div>
```

- [ ] **Step 4: main.ts**
  - `state` gains `lightDelay: true`.
  - after the Motion (`ts`) handler:
```ts
  // Light-travel delay (spec 2026-10-01): shading-only, delays are always in the cache.
  const ldelay = $("ldelay") as HTMLInputElement;
  ldelay.addEventListener("change", () => { state.lightDelay = ldelay.checked; reset(); });
```
  - `simTime` advance: `if (state.playing) simTime += (dt / 1000) * SPEED * state.timeScale;`
    and update the `SPEED` comment to `// coordinate-time M per real second at Motion 1 (Motion = playback speed)`.
  - uniforms literal: `lightDelay: state.lightDelay ? 1 : 0,` (replacing Task 2's `lightDelay: 0,`).

- [ ] **Step 5: Gates** — `npm test`, `npm run build`, `npm run verify:gpu` → all PASS including the
`presets:` line with `delay toggle ok` and `motion 0 ok`, and `?cachecheck` with `ok delay`;
`?golden` hashes unchanged. Probes PASS.

- [ ] **Step 6: Echo screenshots** — create `scripts/shot-delay.mjs`:
```js
// Writes delay-off.png / delay-on.png: the default view with one bright hot spot, same scene time,
// light-travel delay off vs on (dev server on :5173). For visual review of the echoes.
import { chromium } from "playwright-core";
const BASE = process.env.BASE || "http://localhost:5173";
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1000, height: 700 } });
await page.goto(BASE + "/?bench", { waitUntil: "load" });
for (const ld of [0, 1]) {
  const png = await page.evaluate(async (ld) => {
    const { Renderer } = await import("/src/render/gpu.ts");
    const { buildTempLUT, buildVisibleLUT, lumNormFor } = await import("/src/physics/lookups.ts");
    const { iscoRadius } = await import("/src/physics/orbits.ts");
    document.body.innerHTML = ""; const c = document.createElement("canvas"); c.style.cssText = "width:960px;height:540px;display:block"; document.body.appendChild(c);
    const r = new Renderer(); await r.init(c);
    const a = 0.9, rIn = iscoRadius(a, true);
    r.uploadLUTs(buildTempLUT(a, true, rIn, 40, 512), buildVisibleLUT());
    r.uploadHotSpots(new Float32Array([7, 0, 0.8, 6])); r.rebind();
    for (let f = 0; f < 16; f++) r.frame({ resW: r.width, resH: r.height, outW: r.displayW, outH: r.displayH, a, incl: (60 * Math.PI) / 180, rObs: 1000,
      fovScale: 14, rIn, rOut: 40, Tpeak: 3e4, lumNorm: lumNormFor(3e4), lightDelay: ld, exposure: -1, time: 50, frame: f, reset: f === 0 ? 1 : 0,
      maxSteps: 4800, blend: 1 / (f + 1), timeScale: 1, turbAmp: 0, breatheAmp: 0, nSpots: 1, jetStrength: 0, jetGamma: 5, jetLength: 60, jetKnots: 0, skyStrength: 0 });
    await r.device.queue.onSubmittedWorkDone();
    return c.toDataURL("image/png");
  }, ld);
  (await import("node:fs")).writeFileSync(`delay-${ld ? "on" : "off"}.png`, Buffer.from(png.split(",")[1], "base64"));
  console.log(`• delay-${ld ? "on" : "off"}.png`);
}
await browser.close();
```
Add `delay-*.png` to `.gitignore`. Run `node scripts/shot-delay.mjs`; look at both images: with
delay ON the hot spot's lensed images near the shadow should sit at a visibly different azimuth
(an earlier position on its orbit) than with delay OFF; the primary image moves little. Save both
for the user's review. (If `toDataURL` returns a blank canvas because WebGPU presents
asynchronously, use `page.screenshot` of the canvas instead.)

- [ ] **Step 7: Commit**

```bash
git add src/test/cachecheck.browser.ts src/main.ts index.html scripts/verify-gpu.mjs scripts/shot-delay.mjs .gitignore
git commit -m "Light delay on by default: checkbox, delay-on cache gate, Motion as playback speed"
```

---

### Task 4: Documentation

**Files:**
- Modify: `README.md`, `docs/ROADMAP.md`, `docs/specs/2026-10-01-light-travel-delay-design.md`

- [ ] **Step 1: README** — `## Features`: add "Light-travel delay — each pixel shows its emitter at
the time the light left it, so photon-ring subimages echo the disk (toggle in the panel)"; add a
Status paragraph **Light-travel delay (2026-10-01)** with: the definition (delay = −t_e − rObs,
T_emit = time − delay), the two physics checks with their measured numbers (999.652 M radial;
π√27 = 16.32 M subimage spacing), the cache layout change (delay in the third slot, g recomputed),
Motion as playback speed, `?golden` unchanged (delay off = today, bit for bit), the delay-on
`?cachecheck` result, and what the echo screenshots showed. Update the "Current gates" line.
- [ ] **Step 2: ROADMAP** — tick "Light-travel delay (2026-10-01)".
- [ ] **Step 3: Spec status** — `**Status:** implemented on feat/light-travel-delay`.
- [ ] **Step 4: Commit**

```bash
git add README.md docs/ROADMAP.md docs/specs/2026-10-01-light-travel-delay-design.md
git commit -m "Document light-travel delay"
```
