# Geodesic Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Smooth full-resolution playback by tracing each pixel's geodesic once (per camera/geometry setting) and re-shading only the time-dependent parts every frame, with no change to the physics.

**Architecture:** `raytrace.wgsl` gains two compute entry points next to `main`: `build` (traces a slice of rows at a fixed jitter offset and records each pixel's disk hit / sky direction / shadow plus a jet *bookmark*) and `shade` (re-colours from the record and replays the bookmarked jet stretch with the real integrator). Shading maths is extracted into shared WGSL functions so live and cached frames run one copy. `main.ts` chooses Live or Cached per frame from a geometry key and a background build scheduler (pure TS in `src/render/cache-plan.ts`).

**Tech Stack:** TypeScript, WebGPU/WGSL, Vite, Vitest, playwright-core (headless Chrome) for GPU routes.

**Spec:** `docs/specs/2026-10-01-geodesic-cache-design.md`

## Global Constraints

- Branch `perf/geodesic-cache`. Commit after every task. Commit messages are a plain imperative sentence in this repo's style (e.g. "Add the geodesic cache build and shade passes") — **no** `feat:` prefixes and **no** `Co-Authored-By` trailer.
- No change to the physics: integrator, step control, `rObs`, constants (`H_TOL`, `K_FAR`, …), disk/jet/emission models are untouched.
- Every task ends green on: `npm test`, `npm run build`. Tasks that touch WGSL or `gpu.ts` also run `npm run verify:gpu` (dev server on :5173 in another terminal: `npm run dev`) and `node scripts/probe-axis.mjs`.
- Live path bit-identical to pre-refactor `main` (golden hashes, Task 2). If a golden hash changes, STOP and report the scene and max difference — never re-record to make it pass.
- Cache exactness (`?cachecheck`): non-jet pixels max relative difference ≤ 1e-5; jet-replay pixels ≤ 1e-3 for ≥ 99.99 % with max reported; `LIVE`-fallback pixels ≤ 1e-5. Failures are investigated, never absorbed by loosening.
- Fixed jitter sets (rotated grid): `(-0.125,-0.375) (0.375,-0.125) (0.125,0.375) (-0.375,0.125)`. `NSETS` 4 → 2 → 1 → Live fallback; total cache budget 512 MB; `BUILD_SLICES = 16`.
- Geometry key = `a, incl, fovScale, rObs, rIn, rOut, maxSteps, jetLength, displayW, displayH`. Nothing else invalidates the cache.
- Shell note (Windows): run git/npm from Bash; Vercel deploys are not part of this plan.

## Review Focus

1. **Window resize while cached** — buffers are reallocated; expected: a moment of Live, then Cached again, never garbage or a WebGPU validation error. Pinned by Task 7 Step 6 (verify-gpu resizes the page and waits for `cached` again with no page errors).
2. **Pause / resume** — paused must be today's progressive still at scale 1; resuming with a valid cache goes straight back to Cached without rebuilding. Pinned by Task 5 `chooseMode` tests and Task 7's scheduler-not-reset-on-pause wiring.
3. **Turning the jet on after the cache was built with it off** — jet appears immediately and exactly (bookmarks are geometric). Pinned by Task 6's `jet-toggle` cachecheck scene (built with `jetStrength 0`, shaded with `1`).
4. **Sky panorama landing after the cache is built** — sky must appear without a rebuild. Pinned by Task 5 (`skyStrength` not in the key) and Task 6's `default-sky` cachecheck scene (panorama loaded, shaded from `SKY` entries).
5. **Bookmark buffer overflow / small-memory adapters** — overflowing pixels must still render exactly (`LIVE` fallback), and a 4K canvas on a 128 MB-binding adapter must fall back, not crash. Pinned by Task 6's `overflow` cachecheck scene (capacity forced to 64) and Task 5's `planCache` tests.

---

### Task 1: Measure the jet envelope (decide before building)

**Files:**
- Modify: `src/physics/jet.ts` (add `inJetEnvelope`)
- Test: `tests/jet.test.ts` (envelope unit tests)
- Create: `tests/sweep-jetenvelope.test.ts` (measurement, skipped unless `SWEEP=1`)

**Interfaces:**
- Produces: `inJetEnvelope(r: number, th: number, jetLength: number): boolean` (TS twin of the WGSL function added in Task 4). The measured `BOOKMARK_FRAC` used in Task 5.

- [ ] **Step 1: Write the failing envelope tests** — append to `tests/jet.test.ts` (add `inJetEnvelope` to the existing import from `../src/physics/jet`):

```ts
describe("jet envelope (geodesic-cache bookmark region)", () => {
  it("contains every point where the jet can emit", () => {
    // jetEmission > 0 anywhere => inJetEnvelope true, over a grid and several times
    for (let r = 1.2; r < 80; r *= 1.07) {
      for (let th = 0.001; th < Math.PI; th += 0.013) {
        for (const t of [0, 3.3, 77]) {
          if (jetEmission(r, th, t, 1, 1, 60, 0.7) > 0) expect(inJetEnvelope(r, th, 60)).toBe(true);
        }
      }
    }
  });
  it("excludes below the launch height, beyond the length, and outside the wall", () => {
    expect(inJetEnvelope(1.5, 0.01, 60)).toBe(false);          // |z| < zBase
    expect(inJetEnvelope(70, 0.01, 60)).toBe(false);           // |z| > jetLength
    expect(inJetEnvelope(20, Math.PI / 2 - 0.2, 60)).toBe(false); // far outside the funnel
    expect(inJetEnvelope(20, 0.03, 60)).toBe(true);            // on the axis, inside
    expect(inJetEnvelope(20, Math.PI - 0.03, 60)).toBe(true);  // counter-jet
  });
  it("does not depend on jet strength (it takes none)", () => {
    expect(inJetEnvelope.length).toBe(3);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run tests/jet.test.ts`
Expected: FAIL — `inJetEnvelope` is not exported.

- [ ] **Step 3: Implement** — append to `src/physics/jet.ts`:

```ts
/** True where jetEmission can be non-zero for SOME jetStrength and time: zBase <= |z| <= jetLength
 *  and inside the funnel wall (q <= 1.2, wallProfile's cut). Purely geometric, so the geodesic
 *  cache's jet bookmark never depends on jetStrength (spec 2026-10-01 3.3).
 *  WGSL twin: inJetEnvelope in raytrace.wgsl. */
export function inJetEnvelope(r: number, th: number, jetLength: number): boolean {
  const z = r * Math.cos(th), az = Math.abs(z);
  if (az < JET.zBase || az > jetLength) return false;
  return (r * Math.sin(th)) / funnelEdge(z) <= 1.2;
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run tests/jet.test.ts`
Expected: PASS.

- [ ] **Step 5: Write the measurement** — create `tests/sweep-jetenvelope.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { screenToState } from "../src/physics/camera";
import { iscoRadius } from "../src/physics/orbits";
import { stepGeodesic, stepSize, traceRay, H_TOL, H_TOL_FAR } from "../src/physics/trace";
import { inJetEnvelope } from "../src/physics/jet";

/**
 * Measurement, not a test: how much of the screen the geodesic cache must bookmark for the jet, and
 * what share of a full trace the per-frame jet replay re-integrates (spec 2026-10-01 4, first risk).
 * Skipped unless SWEEP=1. The loop is a LOCAL COPY of traceRay that also records the first and last
 * step whose START state is inside the jet envelope; it first asserts it reproduces traceRay's fate
 * and step count on every ray, so the numbers are about the shipped integrator.
 *
 * Grid: 64 x 36 pixel centres over the shader's screen mapping (fovScale 14, aspect 16:9), maxSteps
 * 4800, rOut 40, rObs 1000, jetLength 60.
 */
const RUN = process.env.SWEEP === "1";
const ROBS = 1000, ROUT = 40, FOV = 14, MAX_STEPS = 4800, JET_LEN = 60, NX = 64, NY = 36;

function traceWithEnvelope(s0: Float64Array, a: number, rIn: number) {
  const rh = 1 + Math.sqrt(Math.max(0, 1 - a * a));
  let s = s0, first = -1, last = -1;
  for (let step = 1; step <= MAX_STEPS; step++) {
    if (inJetEnvelope(s[1], s[2], JET_LEN)) { if (first < 0) first = step; last = step; }
    const far = s[1] > ROUT * 1.5;
    const sN = stepGeodesic(s, a, stepSize(s, rh, ROUT), far ? H_TOL_FAR : H_TOL).s;
    const f0 = s[2] - Math.PI / 2, f1 = sN[2] - Math.PI / 2;
    if (f0 * f1 < 0 && Math.abs(sN[2] - s[2]) < 0.5) {
      const rHit = s[1] + (f0 / (f0 - f1)) * (sN[1] - s[1]);
      if (rHit >= rIn && rHit <= ROUT) return { fate: "disk", steps: step, first, last };
    }
    s = sN;
    if (s[1] <= rh * 1.005) return { fate: "captured", steps: step, first, last };
    if (s[1] > ROBS * 1.2) return { fate: "escaped", steps: step, first, last };
  }
  return { fate: "budget", steps: MAX_STEPS, first, last };
}

const VIEWS = [
  { name: "default a=0.9 i=72", a: 0.9, incl: 72 },
  { name: "edge-on a=0.99 i=85", a: 0.99, incl: 85 },
  { name: "face-on a=0.9 i=8", a: 0.9, incl: 8 },
];

describe.skipIf(!RUN)("jet envelope sweep", () => {
  it("reports bookmark fraction, mean nJet and replay share per view", () => {
    for (const v of VIEWS) {
      const incl = (v.incl * Math.PI) / 180, rIn = iscoRadius(v.a, true);
      let marked = 0, sumNJet = 0, sumSteps = 0;
      for (let iy = 0; iy < NY; iy++) for (let ix = 0; ix < NX; ix++) {
        const alpha = (((ix + 0.5) / NX) * 2 - 1) * FOV * (NX / NY);
        const beta = -(((iy + 0.5) / NY) * 2 - 1) * FOV;
        const s0 = screenToState(alpha, beta, v.a, incl, ROBS);
        const ref = traceRay(s0, v.a, { rIn, rOut: ROUT, rObs: ROBS, maxSteps: MAX_STEPS });
        const got = traceWithEnvelope(s0, v.a, rIn);
        expect(got.fate).toBe(ref.fate);
        expect(got.steps).toBe(ref.steps);
        sumSteps += got.steps;
        if (got.first >= 0) { marked++; sumNJet += got.last - got.first + 1; }
      }
      const n = NX * NY;
      console.log(`${v.name.padEnd(22)} bookmarked ${(100 * marked / n).toFixed(1)} %  ` +
        `mean nJet ${(marked ? sumNJet / marked : 0).toFixed(1)}  ` +
        `replay share ${(100 * sumNJet / sumSteps).toFixed(1)} % of all steps  (mean steps ${(sumSteps / n).toFixed(1)})`);
    }
  }, 600_000);
});
```

- [ ] **Step 6: Run the measurement**

Run: `SWEEP=1 npx vitest run tests/sweep-jetenvelope.test.ts`
Expected: PASS with three report lines. Paste them into the doc comment of the file under a `Measured 2026-10-01:` heading (same style as `tests/sweep-farstride.test.ts`).

- [ ] **Step 7: Decide (gate)**

- If the **default** view's replay share is **> 25 %**: STOP. Report the three lines to the user — Cached frames with the jet on cannot be more than ~4× faster than a full trace — and wait for a decision before Task 2.
- Otherwise compute `BOOKMARK_FRAC = max(0.05, min(1, ceil(1.5 × maxFractionAcrossViews × 100) / 100))` and note it in the doc comment; Task 5 uses it.

- [ ] **Step 8: Run all tests and commit**

Run: `npm test` → all pass (the sweep is skipped without `SWEEP=1`).

```bash
git add src/physics/jet.ts tests/jet.test.ts tests/sweep-jetenvelope.test.ts
git commit -m "Measure the jet envelope the geodesic cache will bookmark"
```

---

### Task 2: Golden hashes of the live pass (before any refactor)

**Files:**
- Modify: `src/render/gpu.ts` (accum `COPY_SRC`, `readbackAccum`, private `readback`)
- Create: `src/test/scenes.ts` (shared validation scenes), `src/test/golden.browser.ts`, `src/test/golden.json` (recorded)
- Modify: `src/main.ts` (route `?golden`), `scripts/verify-gpu.mjs` (golden check + `RECORD_GOLDEN=1`)

**Interfaces:**
- Produces: `Renderer.readbackAccum(): Promise<Float32Array>`; `SCENES`, `prepareScene(r, s): number` (returns `rIn`), `sceneUniforms(r, s, rIn, extra?): UniformValues` in `src/test/scenes.ts`; `fnv1a(words: Uint32Array): string`.

- [ ] **Step 1: Add accum readback to `Renderer`** — in `src/render/gpu.ts`, change the accum buffer creation in `resize()`:

```ts
    this.accumBuf = this.device.createBuffer({ size: this.displayW * this.displayH * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
```

and add these methods to the class (after `readbackPresented`):

```ts
  /** Copy `bytes` from the start of a COPY_SRC buffer to the CPU. Validation harnesses only. */
  private async readback(src: GPUBuffer, bytes: number): Promise<ArrayBuffer> {
    const buf = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, buf, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const out = buf.getMappedRange().slice(0);
    buf.unmap(); buf.destroy();
    return out;
  }
  /** Raw accum (internal width x height vec4<f32>) after the last submitted frame. */
  async readbackAccum(): Promise<Float32Array> {
    return new Float32Array(await this.readback(this.accumBuf, this.width * this.height * 16));
  }
```

- [ ] **Step 2: Create `src/test/scenes.ts`**

```ts
import type { Renderer } from "../render/gpu";
import type { UniformValues } from "../render/uniforms";
import { buildTempLUT, buildColorLUT } from "../physics/lookups";
import { iscoRadius } from "../physics/orbits";

/** Fixed scenes shared by the ?golden and ?cachecheck validation routes. */
export interface Scene { name: string; a: number; inclDeg: number; time: number; frame: number; jetStrength: number; skyStrength: number; }
export const SCENES: Scene[] = [
  { name: "default", a: 0.9, inclDeg: 72, time: 37, frame: 5, jetStrength: 1, skyStrength: 0 },
  { name: "jet-off", a: 0.5, inclDeg: 30, time: 0, frame: 0, jetStrength: 0, skyStrength: 0 },
  { name: "edge-on", a: 0.99, inclDeg: 85, time: 100, frame: 2, jetStrength: 1, skyStrength: 0 },
  { name: "face-on-jet", a: 0.9, inclDeg: 8, time: 12, frame: 1, jetStrength: 1, skyStrength: 0 },
];
const SPOTS = new Float32Array([8, 0, 1.2, 1.8, 12, 2.1, 1.6, 1.2, 16, 4.3, 2.0, 0.9]);

/** Upload the scene's LUTs and hot spots; returns rIn (the ISCO). */
export function prepareScene(r: Renderer, s: Scene): number {
  const rIn = iscoRadius(s.a, true);
  r.uploadLUTs(buildTempLUT(s.a, true, rIn, 40, 512), buildColorLUT(1000, 40000, 256));
  r.uploadHotSpots(SPOTS);
  r.rebind();
  return rIn;
}

export function sceneUniforms(r: Renderer, s: Scene, rIn: number, extra: Partial<UniformValues> = {}): UniformValues {
  return {
    resW: r.width, resH: r.height, outW: r.displayW, outH: r.displayH, a: s.a, incl: (s.inclDeg * Math.PI) / 180,
    rObs: 1000, fovScale: 14, rIn, rOut: 40, Tpeak: 3e4, exposure: 1.6, time: s.time, frame: s.frame, reset: 1,
    maxSteps: 4800, blend: 1, timeScale: 1, turbAmp: 0.6, breatheAmp: 0.2, nSpots: 3,
    jetStrength: s.jetStrength, jetGamma: 5, jetLength: 60, jetKnots: 0.7, skyStrength: s.skyStrength,
    ...extra,
  };
}

/** FNV-1a over 32-bit words -> 8 hex chars. Bit-exact image identity. */
export function fnv1a(words: Uint32Array): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < words.length; i++) {
    let w = words[i];
    for (let b = 0; b < 4; b++) { h ^= w & 0xff; h = Math.imul(h, 0x01000193) >>> 0; w >>>= 8; }
  }
  return h.toString(16).padStart(8, "0");
}
```

- [ ] **Step 3: Create `src/test/golden.browser.ts`**

First create the placeholder `src/test/golden.json` (Step 6 overwrites it):

```json
{"adapter":"","hashes":{}}
```

```ts
import { Renderer } from "../render/gpu";
import { SCENES, prepareScene, sceneUniforms, fnv1a } from "./scenes";
import goldenRaw from "./golden.json?raw";

export interface GoldenResult { adapter: string; hashes: Record<string, string>; }
/** Recorded by `RECORD_GOLDEN=1 npm run verify:gpu` on the pre-refactor shader. */
export const GOLDEN: GoldenResult = JSON.parse(goldenRaw);

/** PASS when every recorded scene hash matches; SKIP when recorded on a different adapter. */
export function judgeGolden(want: GoldenResult, got: GoldenResult): "PASS" | "FAIL" | "SKIP" {
  if (want.adapter !== got.adapter) return "SKIP";
  return Object.keys(want.hashes).every((k) => want.hashes[k] === got.hashes[k]) ? "PASS" : "FAIL";
}

/** Bit-exact identity of the LIVE pass (`main`) on fixed scenes at 320x180, scale 1, blend 1.
 *  Recorded once BEFORE the geodesic-cache refactor (plan 2026-10-01 Task 2); every later task must
 *  reproduce it. Hashes are per adapter: on a different GPU the route reports SKIP, not FAIL. */
export async function runGolden(canvas: HTMLCanvasElement) {
  canvas.style.width = "320px"; canvas.style.height = "180px";
  const r = new Renderer(); await r.init(canvas);
  const adapter = `${r.adapterInfo.vendor} ${r.adapterInfo.architecture} ${r.adapterInfo.description}`.trim();
  const hashes: Record<string, string> = {};
  for (const s of SCENES) {
    const rIn = prepareScene(r, s);
    r.frame(sceneUniforms(r, s, rIn));
    await r.device.queue.onSubmittedWorkDone();
    hashes[s.name] = fnv1a(new Uint32Array((await r.readbackAccum()).buffer));
  }
  return { adapter, hashes };
}
```

- [ ] **Step 4: Add the `?golden` route** — in `src/main.ts`, insert before the `else if (location.search.includes("bench"))` branch:

```ts
} else if (location.search.includes("golden")) {
  // Validation entry: bit-exact live-pass hashes (plan 2026-10-01 Task 2). `&record` prints JSON to commit.
  const { runGolden, judgeGolden, GOLDEN: want } = await import("./test/golden.browser");
  const got = await runGolden(canvas);
  if (location.search.includes("record")) {
    document.body.innerHTML = `<pre>GOLDEN RECORD</pre><pre id="json">${JSON.stringify(got)}</pre>`;
  } else {
    const lines = Object.keys(want.hashes).map((k) => `${k}: want ${want.hashes[k]} got ${got.hashes[k]}`);
    const verdict = judgeGolden(want, got);
    document.body.innerHTML = `<pre style="color:${verdict === "FAIL" ? "#f66" : "#6f6"};font-size:16px;padding:20px">GOLDEN ${verdict} (${got.adapter}; recorded on ${want.adapter})\n${lines.join("\n")}</pre>`;
  }
```

- [ ] **Step 5: Wire into `scripts/verify-gpu.mjs`** — add `import { writeFileSync } from "node:fs";` at the top, add a helper after `check`:

```js
async function checkAny(path, accepts) {
  await page.goto(BASE + path, { waitUntil: "load", timeout: 20000 });
  await page.waitForFunction((a) => a.some((e) => document.body.innerText.includes(e)), accepts, { timeout: 60000 }).catch(() => {});
  const txt = (await page.innerText("body")).replace(/\s+/g, " ").trim();
  const ok = accepts.some((e) => txt.includes(e));
  console.log(`${ok ? "✓ PASS" : "✗ FAIL"}  ${path}\n        ${txt.slice(0, 400)}`);
  if (!ok) failed = true;
}
```

and after `await check("/?shadow", "SHADOW PASS");`:

```js
if (process.env.RECORD_GOLDEN === "1") {
  await page.goto(BASE + "/?golden&record", { waitUntil: "load", timeout: 20000 });
  await page.waitForFunction(() => document.body.innerText.includes("GOLDEN RECORD"), null, { timeout: 60000 });
  writeFileSync("src/test/golden.json", (await page.innerText("#json")) + "\n");
  console.log("• recorded src/test/golden.json");
}
await checkAny("/?golden", ["GOLDEN PASS", "GOLDEN SKIP"]);
```

- [ ] **Step 6: Record the golden hashes on the UNCHANGED shader**

With `npm run dev` running: `RECORD_GOLDEN=1 npm run verify:gpu`
Expected: parity PASS, shadow PASS, `• recorded src/test/golden.json`, `✓ PASS /?golden … GOLDEN PASS`, sky PASS.
Then run `npm run verify:gpu` again (no env) → `GOLDEN PASS` (proves the hashes are deterministic run-to-run). If the second run FAILs, the live pass is not deterministic on this adapter: STOP and report.

- [ ] **Step 7: Tests, build, commit**

Run: `npm test && npm run build` → pass.

```bash
git add src/render/gpu.ts src/test/scenes.ts src/test/golden.browser.ts src/test/golden.json src/main.ts scripts/verify-gpu.mjs
git commit -m "Record golden hashes of the live trace before the geodesic-cache refactor"
```

Note for later tasks: `src/test/golden.json` is only rewritten by `RECORD_GOLDEN=1`. Never set it again after this step.

---

### Task 3: Uniforms for fixed jitter and row slices

**Files:**
- Modify: `src/render/uniforms.ts`, `tests/uniforms.test.ts`, `src/render/raytrace.wgsl` (struct + `fixedJitter` + `pixelJitter`)

**Interfaces:**
- Produces: optional `UniformValues` fields `jitterMode?: number` (0 hash, 1 fixed), `setIndex?: number`, `rowStart?: number`, `rowEnd?: number` (uint indices 26–29); `UNIFORM_SIZE = 128`. WGSL `fixedJitter(k: u32) -> vec2<f32>`, `pixelJitter(p: vec2<u32>) -> vec2<f32>`.

- [ ] **Step 1: Write the failing test** — in `tests/uniforms.test.ts` replace the size assertion and add a test:

```ts
    expect(UNIFORM_SIZE).toBe(128);
```

```ts
  it("packs the geodesic-cache fields after outW/outH and defaults them to 0", () => {
    const base: UniformValues = {
      resW: 1, resH: 1, a: 0, incl: 0, rObs: 1000, fovScale: 14, rIn: 6, rOut: 40, Tpeak: 3e4, exposure: 1,
      time: 0, frame: 0, reset: 0, maxSteps: 1, blend: 1, timeScale: 1, turbAmp: 0, breatheAmp: 0, nSpots: 0,
      jetStrength: 0, jetGamma: 5, jetLength: 60, jetKnots: 0, skyStrength: 0, outW: 1, outH: 1,
    };
    const d0 = new DataView(packUniforms(base));
    for (const off of [104, 108, 112, 116]) expect(d0.getUint32(off, true)).toBe(0);
    const d1 = new DataView(packUniforms({ ...base, jitterMode: 1, setIndex: 3, rowStart: 64, rowEnd: 128 }));
    expect(d1.getUint32(104, true)).toBe(1);   // jitterMode (index 26)
    expect(d1.getUint32(108, true)).toBe(3);   // setIndex (index 27)
    expect(d1.getUint32(112, true)).toBe(64);  // rowStart (index 28)
    expect(d1.getUint32(116, true)).toBe(128); // rowEnd (index 29)
  });
```

Also rename the first test's title to `"is 128 bytes and packs all fields (incl. display size) at the expected offsets"`.

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run tests/uniforms.test.ts` → FAIL (size 112, offsets 104+ are 0 for the second case).

- [ ] **Step 3: Implement in `src/render/uniforms.ts`** — update the layout comment, interface, constants and packer:

```ts
//         + outW,outH (2)                                               -> 22 floats
// uint:   frame,reset,maxSteps (3) + nSpots (1)                        -> 4 uints
//         + jitterMode,setIndex,rowStart,rowEnd (4, geodesic cache)    -> 8 uints
```

```ts
  outW: number; outH: number;
  /** Geodesic cache (spec 2026-10-01): 0 = per-frame hash jitter, 1 = fixed jitter set `setIndex`. */
  jitterMode?: number; setIndex?: number;
  /** Build pass row slice [rowStart, rowEnd). */
  rowStart?: number; rowEnd?: number;
}
export const UNIFORM_FLOATS = 22, UNIFORM_UINTS = 8;
export const UNIFORM_SIZE = Math.ceil((UNIFORM_FLOATS + UNIFORM_UINTS) / 4) * 16; // -> 128 bytes
```

and at the end of `packUniforms`, before `return buf;`:

```ts
  i[26] = u.jitterMode ?? 0; i[27] = u.setIndex ?? 0; i[28] = u.rowStart ?? 0; i[29] = u.rowEnd ?? 0;
```

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run tests/uniforms.test.ts` → PASS.

- [ ] **Step 5: Extend the WGSL struct and jitter** — in `src/render/raytrace.wgsl` change the struct's last line from `skyStrength: f32,` to:

```wgsl
  skyStrength: f32, outW: f32, outH: f32,
  jitterMode: u32, setIndex: u32, rowStart: u32, rowEnd: u32,
```

(`present.wgsl` and `bloom.wgsl` declare prefixes of this struct and need no change.) After `hash2`, add:

```wgsl
// Fixed rotated-grid jitter sets of the geodesic cache (spec 2026-10-01 3.2).
// Twin: JITTER in src/render/cache-plan.ts.
fn fixedJitter(k: u32) -> vec2<f32> {
  switch (k & 3u) {
    case 0u: { return vec2<f32>(-0.125, -0.375); }
    case 1u: { return vec2<f32>(0.375, -0.125); }
    case 2u: { return vec2<f32>(0.125, 0.375); }
    default: { return vec2<f32>(-0.375, 0.125); }
  }
}
fn pixelJitter(p: vec2<u32>) -> vec2<f32> {
  if (U.jitterMode == 1u) { return fixedJitter(U.setIndex); }
  return hash2(p, U.frame) - 0.5;
}
```

In `main`, replace `let jit = hash2(gid.xy, U.frame) - 0.5;` with `let jit = pixelJitter(gid.xy);`.

- [ ] **Step 6: Gates**

Run: `npm test && npm run build`, then (dev server up) `npm run verify:gpu` → parity, shadow, **GOLDEN PASS**, sky all pass; `node scripts/probe-axis.mjs` → passes as before.

- [ ] **Step 7: Commit**

```bash
git add src/render/uniforms.ts tests/uniforms.test.ts src/render/raytrace.wgsl
git commit -m "Add fixed jitter sets and row-slice uniforms for the geodesic cache"
```

---

### Task 4: Extract the trace into shared functions (live output unchanged)

**Files:**
- Modify: `src/render/raytrace.wgsl` (the `main` entry point and helpers after `skyDir`)

**Interfaces:**
- Produces (WGSL, used by Task 6):
  - `const KIND_SHADOW = 0u; KIND_DISK = 1u; KIND_SKY = 2u; KIND_LIVE = 3u;`
  - `fn skyColor(dir: vec3<f32>) -> vec3<f32>`
  - `fn diskG(rHit: f32, xi: f32, a: f32) -> f32`
  - `fn shadeDisk(rHit: f32, phiHit: f32, g: f32, a: f32) -> vec3<f32>`
  - `fn jetStep(s: State, sNew: State, dl: f32) -> vec3<f32>`
  - `fn inJetEnvelope(r: f32, th: f32) -> bool`
  - `struct TraceOut { color: vec3<f32>, jet: vec3<f32>, kind: u32, payload: vec3<f32>, hasBm: bool, bm: State, nJet: u32 }`
  - `fn traceRay(pix: vec2<u32>, jit: vec2<f32>, record: bool) -> TraceOut`
  - `fn storeComposite(idx: u32, color: vec3<f32>, jetAccum: vec3<f32>)`

- [ ] **Step 1: Replace everything from `@compute @workgroup_size(8,8) fn main` to the end of `raytrace.wgsl`** with the following. The arithmetic and its order are copied from the current `main`; only the structure changes.

```wgsl
// --- Shared pieces of the trace. `main` (live), and the geodesic cache's `build` and `shade` passes
// (spec 2026-10-01) call these, so a cached frame runs the same maths as a live one. -------------

const KIND_SHADOW = 0u; const KIND_DISK = 1u; const KIND_SKY = 2u; const KIND_LIVE = 3u;

// Background along an escaped ray's bent asymptotic direction: the baked panorama crossfaded over
// the procedural starfield by skyStrength (0 => procedural only).
fn skyColor(dir: vec3<f32>) -> vec3<f32> {
  let mixT = clamp(U.skyStrength, 0.0, 1.0);
  return mix(starfield(dir), skySample(dir) * U.skyStrength, mixT);
}

// Doppler + gravitational redshift factor of the disk matter at rHit seen along a ray with xi.
fn diskG(rHit: f32, xi: f32, a: f32) -> f32 {
  let Om = omegaKep(rHit, a);
  let gl = gLow(rHit, PI*0.5, a);
  let rad = -(gl[0] + 2.0*Om*gl[1] + Om*Om*gl[4]);
  return sqrt(max(0.0, rad)) / (1.0 - Om*xi);
}

// Observed disk colour at a hit: the only time dependence is the co-rotating pattern phase psi.
fn shadeDisk(rHit: f32, phiHit: f32, g: f32, a: f32) -> vec3<f32> {
  let Tn = sampleTemp(rHit);
  let Om = omegaKep(rHit, a);
  let Tobs = U.Tpeak * g * Tn;                 // observed blackbody temperature
  let psi = phiHit - Om * U.time * U.timeScale;// co-rotating pattern phase
  let E = emissionFieldE(rHit, psi);           // time-varying brightness (==1 when features off)
  return sampleColor(Tobs) * pow(g * Tn, 4.0) * E;
}

// Optically-thin jet radiance gathered over one step s -> sNew (zero outside the emitting region).
fn jetStep(s: State, sNew: State, dl: f32) -> vec3<f32> {
  let jz = s.x.y * cos(s.x.z);
  let e = jetEmissionJ(s.x.y, s.x.z, U.time);
  let dvec = cartOf(sNew.x) - cartOf(s.x);                  // inward step (camera -> hole)
  // guard normalize() against a zero-length step: a NaN mu here would poison the EMA accum
  // buffer permanently (mix(accum, NaN, blend) stays NaN). Effectively unreachable, cheap insurance.
  if (e > 0.0 && dot(dvec, dvec) > 1e-12) {
    let marchDir = normalize(dvec);
    let axisSign = select(-1.0, 1.0, jz >= 0.0);
    let mu = -axisSign * marchDir.z;                        // emitter outflow toward observer
    return JET_TINT * (e * JET_GAIN) * boostJ(mu, U.jetGamma) * dl;
  }
  return vec3<f32>(0.0);
}

// Where jetEmissionJ can be non-zero for SOME jetStrength/time (twin: inJetEnvelope in jet.ts).
// Geometric only, so the cache's bookmark never depends on jetStrength.
fn inJetEnvelope(r: f32, th: f32) -> bool {
  let z = r * cos(th);
  let az = abs(z);
  if (az < JET_ZBASE || az > U.jetLength) { return false; }
  return r * sin(th) / funnelEdgeJ(z) <= 1.2;
}

struct TraceOut {
  color: vec3<f32>, jet: vec3<f32>,
  kind: u32, payload: vec3<f32>,   // DISK: (rHit, phiHit, g); SKY: asymptotic direction; else 0
  hasBm: bool, bm: State, nJet: u32, // record only: first state inside the jet envelope, steps through the last
};

fn traceRay(pix: vec2<u32>, jit: vec2<f32>, record: bool) -> TraceOut {
  var out: TraceOut;
  out.kind = KIND_SHADOW; out.payload = vec3<f32>(0.0); out.hasBm = false; out.nJet = 0u;
  let a = U.a; let i = U.incl;

  // pixel -> impact parameters (alpha,beta) in units of M, with sub-pixel jitter for AA
  let aspect = U.res.x / U.res.y;
  let ndc = (vec2<f32>(f32(pix.x), f32(pix.y)) + 0.5 + jit) / U.res * 2.0 - 1.0;
  let alpha = ndc.x * U.fovScale * aspect;
  let beta  = -ndc.y * U.fovScale;
  // Bardeen impact parameters -> conserved (xi, eta). Sole copy lives in camera-shared.wgsl,
  // which gpu.ts prepends here and parity.browser.ts prepends to camera-parity.wgsl.
  let xe = cameraXiEta(alpha, beta, a, i);
  let xi = xe.x; let eta = xe.y;

  // initial state at (rObs, i, 0), E=1. Past-directed momentum -- see cameraMomenta().
  let r0 = U.rObs; let th0 = i;
  let gU = gUp(r0, th0, a);
  let p0 = cameraMomenta(xi, beta, gU[0], gU[1], gU[2], gU[3], gU[4]);
  var s = State(vec4<f32>(0.0, r0, th0, 0.0), p0);

  let rh = 1.0 + sqrt(max(0.0, 1.0 - a*a)); // horizon
  var color = vec3<f32>(0.0);
  var resolved = false; // set by each real termination; false => the step budget ran out
  var jetAccum = vec3<f32>(0.0); // optically-thin jet emission integrated along the ray
  var firstJ = 0u; var lastJ = 0u;

  for (var step = 0u; step < U.maxSteps; step++) {
    if (record && inJetEnvelope(s.x.y, s.x.z)) {
      if (!out.hasBm) { out.hasBm = true; out.bm = s; firstJ = step; }
      lastJ = step;
    }
    // dl > 0 with p_r < 0 integrates INWARD along the reversed worldline.
    let r = s.x.y;
    let far = r > U.rOut * 1.5; // same threshold as the far branch of stepSize: monitor OFF out there
    let st = stepGeodesic(s, a, stepSize(s, rh, U.rOut), select(H_TOL, H_TOL_FAR, far));
    // On retry exhaustion the smallest-step attempt is accepted and the ray proceeds; st.ok is
    // informational. Breaking to the (xi, eta) classifier here painted starfield over disk hits
    // for near-axis rays (it can only answer captured/escaped) -- a dark seam on the alpha = 0
    // column. A genuinely diverging ray still winds to budget exhaustion and reaches the
    // classifier below as before; a NaN state still ends in the `usable` guard.
    let dl = st.dl; let sNew = st.s;

    // Optically-thin jet: integrate emissivity * relativistic beaming along the ray. The disk
    // hit below still `break`s (opaque), so jet segments behind the disk/horizon are occluded.
    if (U.jetStrength > 0.0) { jetAccum += jetStep(s, sNew, dl); }

    // disk crossing: equatorial plane th = PI/2 (take the first hit -> optically-thick top surface).
    // A step that moved theta by more than 0.5 rad is not a plane crossing (a legitimate near-field
    // step moves theta by <= ~0.07 rad): it is a diverged state that reflectAxis's single-crossing
    // reduction cannot have made sense of, and interpolating a disk hit from it would be garbage.
    let f0 = s.x.z - PI*0.5; let f1 = sNew.x.z - PI*0.5;
    if (f0 * f1 < 0.0 && abs(sNew.x.z - s.x.z) < 0.5) {
      let frac = f0 / (f0 - f1);
      let rHit = mix(s.x.y, sNew.x.y, frac);
      if (rHit >= U.rIn && rHit <= U.rOut) {
        let g = diskG(rHit, xi, a);
        let phiHit = mix(s.x.w, sNew.x.w, frac);     // azimuth of the emitting matter
        color = shadeDisk(rHit, phiHit, g, a);
        out.kind = KIND_DISK; out.payload = vec3<f32>(rHit, phiHit, g);
        resolved = true;
        break;
      }
    }
    s = sNew;
    // captured -> shadow. The margin must exceed one integration step (dl_min = 0.002 moves r by
    // ~4.2e-3 M at a=0.9), otherwise RK4's intermediate stages sample r < r_+, where Delta < 0
    // flips the metric signature and the state explodes to garbage that can pass the escape test
    // and paint starfield inside the shadow.
    if (s.x.y <= rh * 1.005) { color = vec3(0.0); resolved = true; break; }
    if (s.x.y > r0 * 1.2) {
      // escaped: sample the background along the ray's (bent) asymptotic direction.
      // The deflected direction makes the starfield appear gravitationally lensed —
      // warped and magnified into a ring around the shadow.
      let dir = skyDir(s, a);
      color = skyColor(dir);
      out.kind = KIND_SKY; out.payload = dir;
      resolved = true;
      break;
    }
  }

  // Budget exhausted without a real termination. Previously these rays kept color = vec3(0) and so
  // rendered as shadow -- a step-budget artifact that swallowed the n=1 photon subring. Classify
  // them from their conserved (xi, eta) instead: the sign of p_r at an arbitrary cutoff is
  // effectively random for a winding ray and would produce salt-and-pepper noise.
  if (!resolved) {
    let th = s.x.z; let ph = s.x.w;
    // RK4 can diverge for rays near the critical impact parameter, leaving s.x non-finite when the
    // step budget runs out. Reading that state into `dir` would emit a NaN colour into the EMA
    // accumulator below, and bloom.wgsl's separable blur would then smear that single NaN pixel
    // across a whole neighbourhood. NaN comparisons are always false, so this range test rejects
    // non-finite th/ph without needing a bitcast/isnan helper, and we just treat the ray as captured.
    let usable = th > -1e6 && th < 1e6 && ph > -1e6 && ph < 1e6;
    if (classifyCaptured(xi, eta, a) || !usable) {
      color = vec3<f32>(0.0);
    } else {
      let dir = skyDir(s, a);
      color = skyColor(dir);
      out.kind = KIND_SKY; out.payload = dir;
    }
  }
  out.color = color; out.jet = jetAccum;
  if (out.hasBm) { out.nJet = lastJ - firstJ + 1u; }
  return out;
}

// Temporal EMA: blend = 1/(frame+1) reproduces the Tier-1 running mean when static; a fixed
// blend (~0.15) tracks an animating scene. blend==1 (first frame after a reset) clears cleanly.
// Additive optically-thin jet on top of whatever the ray terminated on (disk/starfield/shadow).
fn storeComposite(idx: u32, color: vec3<f32>, jetAccum: vec3<f32>) {
  let raw = color + U.jetStrength * min(jetAccum, vec3<f32>(JET_CEIL));
  // Single choke point: nothing non-finite may enter accum. The in-loop escape branch above reads
  // s.x without the `usable` guard, so a diverged RK4 ray (r = +inf compares true, th/ph NaN) can
  // still produce a NaN colour there. A NaN in accum is PERMANENT -- mix(NaN, ..) stays NaN for
  // every later frame -- and bloom.wgsl's separable blur amplifies that one pixel into a whole
  // block. NaN compares false to everything, so this range test rejects NaN and both infinities
  // without a bitcast, and is exactly inert for finite values.
  let finite = all(raw > vec3<f32>(-1e30)) && all(raw < vec3<f32>(1e30));
  let composited = select(vec3<f32>(0.0), raw, finite);
  accum[idx] = vec4<f32>(mix(accum[idx].rgb, composited, U.blend), 1.0);
}

@compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u32(U.res.x) || gid.y >= u32(U.res.y)) { return; }
  let idx = gid.y * u32(U.res.x) + gid.x;
  let t = traceRay(gid.xy, pixelJitter(gid.xy), false);
  storeComposite(idx, t.color, t.jet);
}
```

- [ ] **Step 2: Gates**

Run: `npm test && npm run build`; with the dev server: `npm run verify:gpu` → parity PASS, shadow PASS, **GOLDEN PASS**, sky PASS; `node scripts/probe-axis.mjs` → PASS.
If GOLDEN FAILs: the compiler re-associated something. Diff the scene's accum against the pre-refactor commit (check it out in a worktree, read back both with `readbackAccum`) and report the max difference to the user; do not re-record.

- [ ] **Step 3: Benchmark the live path did not slow down**

Run: `npm run bench` → note 1280×720 and 1920×1080 medians; compare with README's (≈ 72 / 165–186 ms on the RTX 3050). A change beyond run-to-run noise (± 10 %) is investigated before committing.

- [ ] **Step 4: Commit**

```bash
git add src/render/raytrace.wgsl
git commit -m "Extract the trace into shared functions; live output bit-identical"
```

---

### Task 5: Cache planning in TypeScript (key, memory, scheduler, mode)

**Files:**
- Create: `src/render/cache-plan.ts`
- Test: `tests/cache-plan.test.ts`

**Interfaces:**
- Produces:
  - `JITTER: ReadonlyArray<readonly [number, number]>`, `NSETS_MAX = 4`, `BUILD_SLICES = 16`, `BOOKMARK_FRAC` (Task 1 value), `ENTRY_BYTES = 16`, `BOOKMARK_BYTES = 48`, `CACHE_BUDGET_BYTES = 512 * 2**20`, `BM_NONE = 0x3fffffff`
  - `interface GeometryInputs { a; incl; fovScale; rObs; rIn; rOut; maxSteps; jetLength; displayW; displayH }` (all `number`), `geometryKey(g: GeometryInputs): string`
  - `interface CachePlan { nSets: number; entryBytes: number; bookmarkCapacity: number }`, `planCache(w: number, h: number, maxBinding: number, budget?: number): CachePlan`
  - `class BuildScheduler { constructor(nSets: number, height: number, slices?: number); completedSets: number; reset(nSets: number, height: number): void; next(): { set: number; rowStart: number; rowEnd: number } | null }`
  - `type Mode = "live" | "cached"`, `chooseMode(playing: boolean, completedSets: number, enabled: boolean): Mode`

- [ ] **Step 1: Write the failing tests** — create `tests/cache-plan.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import {
  JITTER, BUILD_SLICES, ENTRY_BYTES, BOOKMARK_BYTES, BOOKMARK_FRAC, CACHE_BUDGET_BYTES,
  geometryKey, planCache, BuildScheduler, chooseMode, type GeometryInputs,
} from "../src/render/cache-plan";

const G: GeometryInputs = { a: 0.9, incl: 72, fovScale: 14, rObs: 1000, rIn: 2.32, rOut: 40, maxSteps: 4800, jetLength: 60, displayW: 1920, displayH: 1080 };
const MB = 2 ** 20;

describe("jitter sets", () => {
  it("are 4 distinct offsets inside the pixel with zero mean", () => {
    expect(JITTER.length).toBe(4);
    expect(new Set(JITTER.map((j) => j.join())).size).toBe(4);
    for (const [x, y] of JITTER) { expect(Math.abs(x)).toBeLessThan(0.5); expect(Math.abs(y)).toBeLessThan(0.5); }
    expect(JITTER.reduce((s, j) => s + j[0], 0)).toBeCloseTo(0, 12);
    expect(JITTER.reduce((s, j) => s + j[1], 0)).toBeCloseTo(0, 12);
  });
});

describe("geometry key", () => {
  it("changes with every ray-path input", () => {
    const k = geometryKey(G);
    for (const f of Object.keys(G) as (keyof GeometryInputs)[]) {
      expect(geometryKey({ ...G, [f]: G[f] + 1 })).not.toBe(k);
    }
  });
  it("ignores shading-only inputs (time, exposure, jet strength, sky ...)", () => {
    const withShading = { ...G, time: 99, exposure: 3, jetStrength: 0, jetGamma: 9, jetKnots: 0.1, skyStrength: 0.5, turbAmp: 0 } as GeometryInputs;
    expect(geometryKey(withShading)).toBe(geometryKey(G));
  });
});

describe("planCache", () => {
  it("uses 4 sets at 1080p on a 128 MB binding", () => {
    const p = planCache(1920, 1080, 128 * MB);
    expect(p.nSets).toBe(4);
    expect(p.entryBytes).toBe(1920 * 1080 * ENTRY_BYTES);
    expect(p.bookmarkCapacity).toBe(Math.min(Math.ceil(BOOKMARK_FRAC * 1920 * 1080 * 4), Math.floor(128 * MB / BOOKMARK_BYTES)));
    expect(4 * p.entryBytes + p.bookmarkCapacity * BOOKMARK_BYTES).toBeLessThanOrEqual(CACHE_BUDGET_BYTES);
  });
  it("stays live when one set does not fit a binding (4K canvas, 128 MB)", () => {
    expect(planCache(3840, 2160, 128 * MB).nSets).toBe(0);
  });
  it("drops to fewer sets when the 512 MB budget would be exceeded", () => {
    const p = planCache(3840, 2160, 2048 * MB);
    expect([1, 2]).toContain(p.nSets);
    expect(p.nSets * p.entryBytes + p.bookmarkCapacity * BOOKMARK_BYTES).toBeLessThanOrEqual(CACHE_BUDGET_BYTES);
  });
  it("never plans a zero-capacity bookmark buffer when caching", () => {
    expect(planCache(8, 8, 128 * MB).bookmarkCapacity).toBeGreaterThanOrEqual(1);
  });
});

describe("BuildScheduler", () => {
  it("covers every row of every set exactly once, in BUILD_SLICES slices per set", () => {
    const h = 1080, s = new BuildScheduler(4, h);
    const seen = [new Uint8Array(h), new Uint8Array(h), new Uint8Array(h), new Uint8Array(h)];
    let n = 0;
    for (let x = s.next(); x; x = s.next()) { for (let y = x.rowStart; y < x.rowEnd; y++) seen[x.set][y]++; n++; }
    expect(n).toBe(4 * BUILD_SLICES);
    for (const a of seen) expect(a.every((v) => v === 1)).toBe(true);
    expect(s.completedSets).toBe(4);
    expect(s.next()).toBeNull();
  });
  it("marks a set complete on the call that hands out its last slice", () => {
    const s = new BuildScheduler(2, 32);
    for (let k = 0; k < BUILD_SLICES - 1; k++) s.next();
    expect(s.completedSets).toBe(0);
    s.next();
    expect(s.completedSets).toBe(1);
  });
  it("handles fewer rows than slices and resets", () => {
    const s = new BuildScheduler(1, 5);
    const got = []; for (let x = s.next(); x; x = s.next()) got.push(x);
    expect(got.map((g) => [g.rowStart, g.rowEnd])).toEqual([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]);
    s.reset(2, 10);
    expect(s.completedSets).toBe(0);
    expect(s.next()).toEqual({ set: 0, rowStart: 0, rowEnd: 1 });
  });
  it("never builds anything with 0 sets", () => {
    expect(new BuildScheduler(0, 100).next()).toBeNull();
  });
});

describe("chooseMode", () => {
  it("is cached only while playing, enabled, with a complete set", () => {
    expect(chooseMode(true, 1, true)).toBe("cached");
    expect(chooseMode(true, 0, true)).toBe("live");   // still building set 0
    expect(chooseMode(false, 4, true)).toBe("live");  // paused: progressive still
    expect(chooseMode(true, 4, false)).toBe("live");  // ?nocache / pinned scale
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run tests/cache-plan.test.ts` → FAIL (module not found).

- [ ] **Step 3: Implement `src/render/cache-plan.ts`** (set `BOOKMARK_FRAC` to the value decided in Task 1 Step 7):

```ts
// Geodesic cache planning (spec 2026-10-01): pure, testable decisions the Renderer and main loop act on.

/** Fixed rotated-grid sub-pixel offsets of the cached jitter sets. Twin: fixedJitter in raytrace.wgsl. */
export const JITTER: ReadonlyArray<readonly [number, number]> = [[-0.125, -0.375], [0.375, -0.125], [0.125, 0.375], [-0.375, 0.125]];
export const NSETS_MAX = 4;
export const BUILD_SLICES = 16;
/** Share of (pixels x sets) given a jet bookmark slot. 1.5x the largest share measured by
 *  tests/sweep-jetenvelope.test.ts (plan 2026-10-01 Task 1). Overflow falls back to LIVE pixels. */
export const BOOKMARK_FRAC = 0.25; // <- replace with the Task 1 value
export const ENTRY_BYTES = 16, BOOKMARK_BYTES = 48;
export const CACHE_BUDGET_BYTES = 512 * 2 ** 20;
/** Entry word = kind (2 bits) | bookmark index << 2; this index means "no bookmark". */
export const BM_NONE = 0x3fffffff;

export interface GeometryInputs {
  a: number; incl: number; fovScale: number; rObs: number; rIn: number; rOut: number;
  maxSteps: number; jetLength: number; displayW: number; displayH: number;
}
/** Everything a ray's path depends on. Shading-only inputs are deliberately absent. */
export function geometryKey(g: GeometryInputs): string {
  return [g.a, g.incl, g.fovScale, g.rObs, g.rIn, g.rOut, g.maxSteps, g.jetLength, g.displayW, g.displayH].join("|");
}

export interface CachePlan { nSets: number; entryBytes: number; bookmarkCapacity: number; }
/** Sets 4 -> 2 -> 1 within the binding limit and the total budget; nSets 0 = stay live. */
export function planCache(w: number, h: number, maxBinding: number, budget = CACHE_BUDGET_BYTES): CachePlan {
  const entryBytes = w * h * ENTRY_BYTES;
  if (entryBytes > maxBinding) return { nSets: 0, entryBytes, bookmarkCapacity: 0 };
  for (const nSets of [NSETS_MAX, 2, 1]) {
    const cap = Math.max(1, Math.min(Math.ceil(BOOKMARK_FRAC * w * h * nSets), Math.floor(maxBinding / BOOKMARK_BYTES), BM_NONE - 1));
    if (nSets * entryBytes + cap * BOOKMARK_BYTES <= budget) return { nSets, entryBytes, bookmarkCapacity: cap };
  }
  return { nSets: 0, entryBytes, bookmarkCapacity: 0 };
}

/** Hands out row slices set by set. A set counts as complete on the call that returns its last
 *  slice: that slice's build dispatch precedes the shade dispatch in the same compute pass, whose
 *  storage writes are visible to it, so the set can be shaded that same frame. */
export class BuildScheduler {
  completedSets = 0;
  private set = 0; private slice = 0;
  constructor(private nSets: number, private height: number, private slices = BUILD_SLICES) {}
  reset(nSets: number, height: number) {
    this.nSets = nSets; this.height = height; this.set = 0; this.slice = 0; this.completedSets = 0;
  }
  next(): { set: number; rowStart: number; rowEnd: number } | null {
    if (this.set >= this.nSets || this.height <= 0) return null;
    const rows = Math.ceil(this.height / this.slices);
    const rowStart = this.slice * rows, rowEnd = Math.min(this.height, rowStart + rows);
    const out = { set: this.set, rowStart, rowEnd };
    this.slice++;
    if (rowEnd >= this.height) { this.slice = 0; this.set++; this.completedSets = this.set; }
    return out;
  }
}

export type Mode = "live" | "cached";
export function chooseMode(playing: boolean, completedSets: number, enabled: boolean): Mode {
  return playing && enabled && completedSets > 0 ? "cached" : "live";
}
```

Note on the "fewer rows than slices" test: with `height = 5`, `rows = ceil(5/16) = 1`, so the set finishes after 5 slices (`rowEnd >= height`); with `height = 1080`, `rows = 68` and the 16th slice ends at 1080.

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run tests/cache-plan.test.ts` → PASS. Then `npm test && npm run build` → pass.

- [ ] **Step 5: Commit**

```bash
git add src/render/cache-plan.ts tests/cache-plan.test.ts
git commit -m "Plan the geodesic cache: geometry key, memory, build scheduler, mode"
```

---

### Task 6: Build and shade passes, renderer plumbing, and the exactness check

**Files:**
- Modify: `src/render/raytrace.wgsl` (bindings 7–9, `replayJet`, `build`, `shade`)
- Modify: `src/render/gpu.ts` (limits, explicit layout, cache buffers, `frame` options, readbacks)
- Create: `src/test/cachecheck.browser.ts`
- Modify: `src/main.ts` (route `?cachecheck`), `scripts/verify-gpu.mjs`

**Interfaces:**
- Consumes: Task 4's WGSL functions; Task 5's `planCache`, `BM_NONE`, `ENTRY_BYTES`, `BOOKMARK_BYTES`; Task 2's `scenes.ts`.
- Produces:
  - `interface FrameOpts { cachedSet?: number; build?: UniformValues & { setIndex: number; rowStart: number; rowEnd: number } }`
  - `Renderer.frame(u: UniformValues, opts?: FrameOpts): void` (unchanged behaviour with no opts)
  - `Renderer.cacheSets: number` (from the plan; 0 = live only), `Renderer.bookmarkCapacityOverride: number | null`
  - `Renderer.resetCache(): void` (zero the bookmark counter)
  - `Renderer.readbackEntries(set: number): Promise<Uint32Array>`, `Renderer.readbackBookmarks(): Promise<{ count: number; meanNJet: number }>`

- [ ] **Step 1: Add the cache bindings and passes to `raytrace.wgsl`** — after the existing `@group(0) @binding(6)` line add:

```wgsl
// Geodesic cache (spec 2026-10-01). One Entry per pixel per jitter set; bookmarks are sparse.
struct Entry { word: u32, p0: f32, p1: f32, p2: f32 };        // word = kind | bookmark index << 2
struct Bookmark { x: vec4<f32>, p: vec4<f32>, nJet: u32 };      // 48 bytes (vec4 alignment)
@group(0) @binding(7) var<storage, read_write> entries: array<Entry>;
@group(0) @binding(8) var<storage, read_write> bookmarks: array<Bookmark>;
@group(0) @binding(9) var<storage, read_write> bmCount: atomic<u32>;
const BM_NONE = 0x3fffffffu; // twin: BM_NONE in cache-plan.ts
```

and append at the end of the file:

```wgsl
// Re-integrate a bookmarked jet stretch: the same steps, in the same order, as traceRay took from
// the bookmark through the last step whose start state was inside the jet envelope.
fn replayJet(bm: State, nJet: u32) -> vec3<f32> {
  let a = U.a;
  let rh = 1.0 + sqrt(max(0.0, 1.0 - a*a));
  var s = bm; var acc = vec3<f32>(0.0);
  for (var k = 0u; k < nJet; k++) {
    let far = s.x.y > U.rOut * 1.5;
    let st = stepGeodesic(s, a, stepSize(s, rh, U.rOut), select(H_TOL, H_TOL_FAR, far));
    acc += jetStep(s, st.s, st.dl);
    s = st.s;
  }
  return acc;
}

// Trace rows [rowStart, rowEnd) of jitter set setIndex at full resolution and record them.
@compute @workgroup_size(8,8) fn build(@builtin(global_invocation_id) gid: vec3<u32>) {
  let y = U.rowStart + gid.y;
  if (gid.x >= u32(U.res.x) || y >= U.rowEnd || y >= u32(U.res.y)) { return; }
  let idx = y * u32(U.res.x) + gid.x;
  let t = traceRay(vec2<u32>(gid.x, y), fixedJitter(U.setIndex), true);
  var word = t.kind | (BM_NONE << 2u);
  if (t.hasBm) {
    let b = atomicAdd(&bmCount, 1u);
    if (b < arrayLength(&bookmarks)) {
      bookmarks[b] = Bookmark(t.bm.x, t.bm.p, t.nJet);
      word = t.kind | (b << 2u);
    } else {
      word = KIND_LIVE | (BM_NONE << 2u); // bookmark buffer full: shade traces this pixel in full
    }
  }
  entries[idx] = Entry(word, t.payload.x, t.payload.y, t.payload.z);
}

// Cached frame: re-colour from the record of jitter set setIndex; replay the jet stretch.
@compute @workgroup_size(8,8) fn shade(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u32(U.res.x) || gid.y >= u32(U.res.y)) { return; }
  let idx = gid.y * u32(U.res.x) + gid.x;
  let e = entries[idx];
  let kind = e.word & 3u;
  if (kind == KIND_LIVE) {
    let t = traceRay(gid.xy, fixedJitter(U.setIndex), false);
    storeComposite(idx, t.color, t.jet);
    return;
  }
  var color = vec3<f32>(0.0);
  if (kind == KIND_DISK) { color = shadeDisk(e.p0, e.p1, e.p2, U.a); }
  else if (kind == KIND_SKY) { color = skyColor(vec3<f32>(e.p0, e.p1, e.p2)); }
  var jet = vec3<f32>(0.0);
  let bi = e.word >> 2u;
  if (U.jetStrength > 0.0 && bi != BM_NONE) {
    let b = bookmarks[bi];
    jet = replayJet(State(b.x, b.p), b.nJet);
  }
  storeComposite(idx, color, jet);
}
```

- [ ] **Step 2: Renderer — limits, explicit layout, pipelines** — in `src/render/gpu.ts`:

Add imports:

```ts
import { planCache, BOOKMARK_BYTES, type CachePlan } from "./cache-plan";
```

Add fields:

```ts
  computeLayout!: GPUBindGroupLayout;
  buildPipe!: GPUComputePipeline; shadePipe!: GPUComputePipeline;
  buildUniformBuf!: GPUBuffer;
  entryBufs: GPUBuffer[] = []; bookmarkBuf!: GPUBuffer; bmCountBuf!: GPUBuffer;
  computeBinds: GPUBindGroup[] = []; buildBinds: GPUBindGroup[] = [];
  plan: CachePlan = { nSets: 0, entryBytes: 0, bookmarkCapacity: 0 };
  /** Validation only: force a tiny bookmark buffer to exercise the LIVE fallback. */
  bookmarkCapacityOverride: number | null = null;
  get cacheSets() { return this.plan.nSets; }
```

In `init`, replace `this.device = await adapter.requestDevice();` with:

```ts
    // The cache's per-set entry buffer is W*H*16 bytes; ask for the adapter's maximum binding so
    // large canvases still cache (planCache falls back to fewer sets / live within it).
    this.device = await adapter.requestDevice({ requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    } });
```

and before `this.resize(canvas);` add:

```ts
    this.buildUniformBuf = this.device.createBuffer({ size: UNIFORM_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bmCountBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
```

At the end of `resize()` (after `this.applyScale();`) add `this.allocCache();` and add the method:

```ts
  /** (Re)allocate the geodesic cache for the display size. Caller must rebind() (resize callers do). */
  private allocCache() {
    for (const b of this.entryBufs) b.destroy();
    this.bookmarkBuf?.destroy();
    this.plan = planCache(this.displayW, this.displayH, this.device.limits.maxStorageBufferBindingSize);
    const cap = this.bookmarkCapacityOverride ?? this.plan.bookmarkCapacity;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    // With nSets 0 one 16-byte placeholder keeps every bind group valid.
    this.entryBufs = Array.from({ length: Math.max(1, this.plan.nSets) }, () =>
      this.device.createBuffer({ size: this.plan.nSets ? this.plan.entryBytes : 16, usage }));
    this.bookmarkBuf = this.device.createBuffer({ size: Math.max(1, cap) * BOOKMARK_BYTES, usage });
    this.resetCache();
  }
  /** Start a rebuild: bookmark slots are handed out from 0 again. */
  resetCache() { this.device.queue.writeBuffer(this.bmCountBuf, 0, new Uint32Array([0])); }
```

Replace the compute pipeline creation in `buildPipelines()`:

```ts
    const st = (type: GPUBufferBindingType): GPUBindGroupLayoutEntry["buffer"] => ({ type });
    this.computeLayout = this.device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: st("uniform") },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: st("storage") },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: st("read-only-storage") },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: st("read-only-storage") },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: st("read-only-storage") },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: st("storage") },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: st("storage") },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: st("storage") }] });
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] });
    this.computePipe = this.device.createComputePipeline({ layout, compute: { module: cMod, entryPoint: "main" } });
    this.buildPipe = this.device.createComputePipeline({ layout, compute: { module: cMod, entryPoint: "build" } });
    this.shadePipe = this.device.createComputePipeline({ layout, compute: { module: cMod, entryPoint: "shade" } });
```

In `rebind()`, replace the `this.computeBind = …` statement with:

```ts
    const common = (ub: GPUBuffer, entryBuf: GPUBuffer): GPUBindGroupEntry[] => [
      { binding: 0, resource: { buffer: ub } },
      { binding: 1, resource: { buffer: this.accumBuf } },
      { binding: 2, resource: { buffer: this.tempBuf } },
      { binding: 3, resource: { buffer: this.colorBuf } },
      { binding: 4, resource: { buffer: this.spotBuf } },
      { binding: 5, resource: this.skyTex.createView() },
      { binding: 6, resource: this.skySampler },
      { binding: 7, resource: { buffer: entryBuf } },
      { binding: 8, resource: { buffer: this.bookmarkBuf } },
      { binding: 9, resource: { buffer: this.bmCountBuf } }];
    this.computeBinds = this.entryBufs.map((b) => this.device.createBindGroup({ layout: this.computeLayout, entries: common(this.uniformBuf, b) }));
    this.buildBinds = this.entryBufs.map((b) => this.device.createBindGroup({ layout: this.computeLayout, entries: common(this.buildUniformBuf, b) }));
```

and delete the `computeBind` field (now `computeBinds[0]`; `grep -rn computeBind src scripts` must then only show `computeBinds`). Init order matters: `init` creates `buildUniformBuf`/`bmCountBuf`, then `resize()` (→ `allocCache()`), then `buildPipelines()` (→ `rebind()`), so every buffer a bind group needs exists.

- [ ] **Step 3: Renderer — frame options and readbacks** — export the options type above the class:

```ts
/** Per-frame geodesic-cache work. `build` traces one row slice of a jitter set (full resolution,
 *  its own uniform buffer); `cachedSet` shades from that set instead of tracing (`main`). */
export interface FrameOpts { cachedSet?: number; build?: UniformValues & { setIndex: number; rowStart: number; rowEnd: number } }
```

Change `recordCompute` and `frame`:

```ts
  private recordCompute(enc: GPUCommandEncoder, opts: FrameOpts = {}) {
    const cp = enc.beginComputePass();
    if (opts.build) {
      // Before shade in the same pass: its entries writes are visible to a shade of the same set.
      const b = opts.build;
      cp.setPipeline(this.buildPipe); cp.setBindGroup(0, this.buildBinds[b.setIndex]);
      cp.dispatchWorkgroups(Math.ceil(this.displayW / 8), Math.ceil((b.rowEnd - b.rowStart) / 8));
    }
    if (opts.cachedSet !== undefined) {
      cp.setPipeline(this.shadePipe); cp.setBindGroup(0, this.computeBinds[opts.cachedSet]);
    } else {
      cp.setPipeline(this.computePipe); cp.setBindGroup(0, this.computeBinds[0]);
    }
    cp.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
    if (this.renderBloom) {
      cp.setPipeline(this.brightHPipe); cp.setBindGroup(0, this.brightHBind);
      cp.dispatchWorkgroups(Math.ceil(this.bw / 8), Math.ceil(this.bh / 8));
      cp.setPipeline(this.blurVPipe); cp.setBindGroup(0, this.blurVBind);
      cp.dispatchWorkgroups(Math.ceil(this.bw / 8), Math.ceil(this.bh / 8));
    }
    cp.end();
  }

  frame(u: UniformValues, opts: FrameOpts = {}) {
    this.device.queue.writeBuffer(this.uniformBuf, 0, packUniforms(u));
    if (opts.build) this.device.queue.writeBuffer(this.buildUniformBuf, 0, packUniforms({ ...opts.build, jitterMode: 1 }));
    const enc = this.device.createCommandEncoder();
    this.recordCompute(enc, opts);
```

(the rest of `frame` is unchanged). `readbackPresented` keeps calling `this.recordCompute(enc)`. Add next to `readbackAccum`:

```ts
  async readbackEntries(set: number): Promise<Uint32Array> {
    return new Uint32Array(await this.readback(this.entryBufs[set], this.displayW * this.displayH * 16));
  }
  /** Bookmarks handed out since resetCache() and their mean replay length. */
  async readbackBookmarks(): Promise<{ count: number; meanNJet: number }> {
    const handed = new Uint32Array(await this.readback(this.bmCountBuf, 4))[0];
    const count = Math.min(handed, this.bookmarkBuf.size / BOOKMARK_BYTES);
    if (!count) return { count: handed, meanNJet: 0 };
    const words = new Uint32Array(await this.readback(this.bookmarkBuf, count * BOOKMARK_BYTES));
    let sum = 0; for (let k = 0; k < count; k++) sum += words[k * 12 + 8];
    return { count: handed, meanNJet: sum / count };
  }
```

`bmCountBuf` needs `COPY_SRC`: create it with `GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC`.

- [ ] **Step 4: Gates before the new route**

Run: `npm test && npm run build`; `npm run verify:gpu` → parity, shadow, **GOLDEN PASS**, sky (the live pass must still be bit-identical with the new layout).

- [ ] **Step 5: Create `src/test/cachecheck.browser.ts`**

```ts
import { Renderer } from "../render/gpu";
import type { UniformValues } from "../render/uniforms";
import { BUILD_SLICES, BM_NONE, BuildScheduler } from "../render/cache-plan";
import { SCENES, prepareScene, sceneUniforms, type Scene } from "./scenes";

/** Exactness gate of the geodesic cache (spec 2026-10-01 3.6): for every jitter set and two times,
 *  a cached frame must equal a live-traced frame with the same fixed jitter. Raw accum floats,
 *  blend 1, scale 1, 480x270. */
type Check = Scene & { buildJet?: number; sky?: boolean; capacity?: number };
const CHECKS: Check[] = [
  { ...SCENES[0], name: "default-sky", skyStrength: 1, sky: true },
  { ...SCENES[2] },
  { ...SCENES[3] },
  { ...SCENES[0], name: "jet-toggle", buildJet: 0 },   // built with the jet off, shaded with it on
  { ...SCENES[0], name: "overflow", capacity: 64 },     // bookmark buffer forced tiny -> LIVE fallback
];
const TIMES = [0, 137.5];
const rel = (a: number, b: number) => Math.abs(a - b) / Math.max(Math.abs(a), 1e-3);

async function loadSky(r: Renderer) {
  const bmp = await createImageBitmap(await (await fetch("/sky/milkyway-4k.jpg")).blob());
  r.uploadSky(bmp); r.rebind();
}

export async function runCacheCheck(canvas: HTMLCanvasElement) {
  canvas.style.width = "480px"; canvas.style.height = "270px";
  const lines: string[] = []; let ok = true;
  for (const c of CHECKS) {
    const r = new Renderer();
    r.bookmarkCapacityOverride = c.capacity ?? null;
    await r.init(canvas);
    if (r.cacheSets === 0) { lines.push(`${c.name}: cache unavailable on this adapter`); ok = false; continue; }
    const rIn = prepareScene(r, c);
    if (c.sky) await loadSky(r);
    const base = sceneUniforms(r, c, rIn);
    // Build every set, slice by slice, exactly as the app does.
    r.resetCache();
    const sched = new BuildScheduler(r.cacheSets, r.displayH, BUILD_SLICES);
    for (let x = sched.next(); x; x = sched.next()) {
      r.frame(base, { build: { ...base, jetStrength: c.buildJet ?? c.jetStrength, resW: r.displayW, resH: r.displayH,
        setIndex: x.set, rowStart: x.rowStart, rowEnd: x.rowEnd } });
    }
    await r.device.queue.onSubmittedWorkDone();
    const bm = await r.readbackBookmarks();
    let worstNon = 0, worstLive = 0, nLive = 0; const jetDiffs: number[] = [];
    for (let k = 0; k < r.cacheSets; k++) {
      const entries = await r.readbackEntries(k);
      for (const time of TIMES) {
        const u: UniformValues = { ...base, time, setIndex: k, jitterMode: 1, blend: 1, reset: 1 };
        r.frame(u); await r.device.queue.onSubmittedWorkDone();
        const live = await r.readbackAccum();
        r.frame(u, { cachedSet: k }); await r.device.queue.onSubmittedWorkDone();
        const cached = await r.readbackAccum();
        for (let p = 0; p < r.width * r.height; p++) {
          const d = Math.max(rel(live[4 * p], cached[4 * p]), rel(live[4 * p + 1], cached[4 * p + 1]), rel(live[4 * p + 2], cached[4 * p + 2]));
          const word = entries[4 * p], kind = word & 3, bi = word >>> 2;
          if (kind === 3) { worstLive = Math.max(worstLive, d); nLive++; }
          else if (u.jetStrength > 0 && bi !== BM_NONE) jetDiffs.push(d);
          else worstNon = Math.max(worstNon, d);
        }
      }
    }
    jetDiffs.sort((x, y) => x - y);
    const p9999 = jetDiffs.length ? jetDiffs[Math.floor(0.9999 * (jetDiffs.length - 1))] : 0;
    const maxJet = jetDiffs.length ? jetDiffs[jetDiffs.length - 1] : 0;
    const pass = worstNon <= 1e-5 && worstLive <= 1e-5 && p9999 <= 1e-3 && (c.capacity === undefined || nLive > 0);
    ok &&= pass;
    lines.push(`${pass ? "ok  " : "BAD "} ${c.name}: sets ${r.cacheSets}, bookmarks ${bm.count} (mean nJet ${bm.meanNJet.toFixed(1)}), ` +
      `non-jet max ${worstNon.toExponential(2)}, jet p99.99 ${p9999.toExponential(2)} max ${maxJet.toExponential(2)} (n ${jetDiffs.length}), ` +
      `live-fallback ${nLive} max ${worstLive.toExponential(2)}`);
    r.device.destroy();
  }
  return { ok, lines };
}
```

- [ ] **Step 6: Add the `?cachecheck` route** — in `src/main.ts`, before the `golden` branch (so `?cachecheck` is matched first):

```ts
} else if (location.search.includes("cachecheck")) {
  // Validation entry: cached frames equal live frames (spec 2026-10-01 3.6).
  const { runCacheCheck } = await import("./test/cachecheck.browser");
  const res = await runCacheCheck(canvas);
  document.body.innerHTML = `<pre style="color:${res.ok ? "#6f6" : "#f66"};font-size:15px;padding:20px">CACHECHECK ${res.ok ? "PASS" : "FAIL"}\n${res.lines.join("\n")}</pre>`;
  console.log("cachecheck", res);
```

and in `scripts/verify-gpu.mjs` after the golden check:

```js
await checkAny("/?cachecheck", ["CACHECHECK PASS"]);
```

- [ ] **Step 7: Run the exactness gate**

Run (dev server up): `npm run verify:gpu`
Expected: parity, shadow, GOLDEN PASS, **CACHECHECK PASS** with 5 `ok` lines, sky PASS. The `overflow` line must show `live-fallback` > 0. If a line is `BAD`, investigate (e.g. a mismatch in non-jet pixels means `shade` and `traceRay` diverged in code — compare the extracted functions); do not change thresholds.
Then `node scripts/probe-axis.mjs` → PASS.

- [ ] **Step 8: Commit**

```bash
git add src/render/raytrace.wgsl src/render/gpu.ts src/test/cachecheck.browser.ts src/main.ts scripts/verify-gpu.mjs
git commit -m "Add the geodesic cache build and shade passes with an exactness check"
```

---

### Task 7: Use the cache in the app

**Files:**
- Modify: `src/main.ts` (mode choice, build slices, scale handling, readout), `index.html` (readout), `scripts/verify-gpu.mjs` (resize check)

**Interfaces:**
- Consumes: `geometryKey`, `BuildScheduler`, `chooseMode` (Task 5); `Renderer.frame(u, opts)`, `cacheSets`, `resetCache` (Task 6).

- [ ] **Step 1: Readout** — in `index.html`, after the FPS line (`<div class="ro"><span>FPS</span><b id="fps">—</b></div>`) add:

```html
      <div class="ro"><span>Cache</span><b id="cmode">live</b></div>
```

- [ ] **Step 2: Wiring in `src/main.ts`** — add the import:

```ts
import { geometryKey, BuildScheduler, chooseMode } from "./render/cache-plan";
```

After `const fpsEl = $("fps");` add:

```ts
  // Geodesic cache (spec 2026-10-01): playing with an unchanged camera re-shades cached geodesics
  // instead of re-tracing them. ?nocache (or a pinned ?scale) keeps the live path only.
  const cacheOn = !new URLSearchParams(location.search).has("nocache") && pinnedScale === null;
  const cmodeEl = $("cmode");
  const sched = new BuildScheduler(r.cacheSets, r.displayH);
  let geoKey = "", cachedFrame = 0, wasCached = false, liveScale = r.scale;
```

- [ ] **Step 3: Replace the body of `loop`** from `if (pinnedScale === null) {` through `r.frame(u);` with:

```ts
    const geo = geometryKey({ a: state.a, incl: state.incl, fovScale: 14, rObs: 1000, rIn, rOut,
      maxSteps: state.maxSteps, jetLength: state.jetLength, displayW: r.displayW, displayH: r.displayH });
    if (geo !== geoKey) { geoKey = geo; sched.reset(r.cacheSets, r.displayH); r.resetCache(); }
    // One background build slice per playing frame (the cache is only used while playing).
    const slice = state.playing && cacheOn ? sched.next() : null;
    const mode = chooseMode(state.playing, sched.completedSets, cacheOn);

    if (mode === "cached") {
      if (!wasCached) { liveScale = r.scale; r.setScale(1); showScale(); cachedFrame = 0; }
    } else {
      if (wasCached) { r.setScale(liveScale); ctl.reset(liveScale); reset(); showScale(); }
      if (pinnedScale === null) {
        if (state.playing) {
          // Controlled on GPU work time per frame, not the rAF delta: rAF is vsync-quantised (never
          // below 16.7 ms at 60 Hz, inside the 13-18 ms dead band), so after any slow spell a
          // rAF-driven controller could only ratchet down. NaN until the first frame completes.
          const ns = ctl.update(r.gpuMs, now);
          // Show the controller's new scale even when the internal size did not change (then no reset).
          if (ns !== null) { if (r.setScale(ns)) reset(); showScale(); }
        } else if (r.scale !== 1) {
          r.setScale(1); ctl.reset(1); reset(); showScale();
        }
      }
    }
    // Playing: fixed EMA (blend==1 on the reset frame to clear). Paused: progressive running mean.
    // A cached run starts with blend 1 too, which rewrites every pixel (no stale live content).
    const blend = mode === "cached" ? (cachedFrame === 0 ? 1 : EMA_BLEND)
      : state.playing ? (sample === 0 ? 1 : EMA_BLEND) : 1 / (sample + 1);
    const setIndex = mode === "cached" ? cachedFrame % sched.completedSets : 0;
    const u: UniformValues = {
      resW: r.width, resH: r.height, outW: r.displayW, outH: r.displayH, a: state.a, incl: state.incl * Math.PI / 180,
      rObs: 1000, fovScale: 14, rIn, rOut, Tpeak: T_PEAK, exposure: state.exposure,
      time: simTime, frame: sample, reset: sample === 0 ? 1 : 0, maxSteps: state.maxSteps,
      blend, timeScale: state.timeScale, turbAmp: state.turbAmp,
      breatheAmp: state.breatheAmp, nSpots: baseSpots.length,
      jetStrength: state.jetStrength, jetGamma: state.jetGamma,
      jetLength: state.jetLength, jetKnots: state.jetKnots,
      skyStrength: skyReady ? state.skyStrength : 0, setIndex,
    };
    r.frame(u, {
      cachedSet: mode === "cached" ? setIndex : undefined,
      build: slice ? { ...u, resW: r.displayW, resH: r.displayH, setIndex: slice.set, rowStart: slice.rowStart, rowEnd: slice.rowEnd } : undefined,
    });
    if (mode === "cached") cachedFrame++;
    wasCached = mode === "cached";
    cmodeEl.textContent = mode === "cached" ? `cached ${sched.completedSets}/${r.cacheSets}` : (r.cacheSets && cacheOn ? "live" : "off");
```

Keep the lines after it (`sample++; …; requestAnimationFrame(loop);`) unchanged. Note: when Cached the controller is idle and `r.scale` is 1; returning to Live restores the scale the controller had reached.

- [ ] **Step 4: Resize must rebuild bind groups after reallocating the cache** — the existing resize handler already calls `r.resize(canvas); r.rebind(); reset();`; the geometry key includes `displayW/H`, so the scheduler resets on the next frame. No code change; verified in Step 6.

- [ ] **Step 5: Manual check (dev server, real browser)**

Open `http://localhost:5173/`. Expected: Cache readout goes `live` → `cached 1/4` within ~1 s → `cached 4/4`; Render scale 100 %; FPS rises versus `http://localhost:5173/?nocache`. Drag the view: `live` while dragging, `cached` again ~1 s after release. Pause: `live`, image converges sharp; Play: `cached` immediately (no rebuild). Move Jet strength 0 ↔ 1 while cached: stays `cached n/4`, jet appears/disappears instantly.

- [ ] **Step 6: verify-gpu resize + mode check** — in `scripts/verify-gpu.mjs`, after the screenshot/sky check and before printing errors, add:

```js
// Geodesic cache in the live app: reaches `cached`, survives a resize, no page errors.
await page.goto(BASE + "/", { waitUntil: "load", timeout: 20000 });
const cached = () => page.waitForFunction(() => (document.getElementById("cmode")?.textContent || "").startsWith("cached"), null, { timeout: 120000 }).then(() => true, () => false);
let cacheOk = await cached();
await page.setViewportSize({ width: 820, height: 560 });
cacheOk = cacheOk && await cached();
console.log(`${cacheOk && !errors.length ? "✓ PASS" : "✗ FAIL"}  cache reaches 'cached' and survives a resize`);
if (!cacheOk || errors.length) failed = true;
```

Run: `npm test && npm run build && npm run verify:gpu` → all PASS. `node scripts/probe-axis.mjs` → PASS.

- [ ] **Step 7: Commit**

```bash
git add src/main.ts index.html scripts/verify-gpu.mjs
git commit -m "Play from the geodesic cache while the camera is still"
```

---

### Task 8: Measure, document, close out

**Files:**
- Modify: `scripts/bench.mjs`, `README.md` (Status), `docs/specs/2026-10-01-geodesic-cache-design.md` (Outcome)

- [ ] **Step 1: Extend the benchmark** — in `scripts/bench.mjs`, inside `bench(w, h, frames)` after the live timing loop and before `t.sort(...)`, add the cached measurement, and return it:

```js
    // Geodesic cache: build every set (timed), then cached frames with the jet on and off.
    const { BuildScheduler } = await import("/src/render/cache-plan.ts");
    let build = null, cachedOn = null, cachedOff = null, bm = null;
    if (r.cacheSets) {
      r.resetCache();
      const sched = new BuildScheduler(r.cacheSets, r.displayH);
      const b0 = performance.now();
      for (let x = sched.next(); x; x = sched.next()) {
        r.frame(u(1), { build: { ...u(1), resW: r.displayW, resH: r.displayH, setIndex: x.set, rowStart: x.rowStart, rowEnd: x.rowEnd } });
        await r.device.queue.onSubmittedWorkDone();
      }
      build = (performance.now() - b0) / r.cacheSets; // per set, including the live frames it rode on
      bm = await r.readbackBookmarks();
      const cachedMedian = async (jet) => {
        const ts = [];
        for (let f = 1; f <= frames; f++) {
          const uf = { ...u(f), jetStrength: jet, setIndex: f % r.cacheSets };
          const t0 = performance.now(); r.frame(uf, { cachedSet: f % r.cacheSets }); await r.device.queue.onSubmittedWorkDone();
          ts.push(performance.now() - t0);
        }
        ts.sort((x, y) => x - y); return ts[ts.length >> 1];
      };
      cachedOn = await cachedMedian(1); cachedOff = await cachedMedian(0);
    }
```

change the return to:

```js
    return { w: r.width, h: r.height, median: t[t.length >> 1], build, cachedOn, cachedOff, sets: r.cacheSets,
      bmFrac: bm ? bm.count / (r.displayW * r.displayH * r.cacheSets) : null, meanNJet: bm ? bm.meanNJet : null };
```

(move `t.sort(...)` above the new block and `r.device.destroy()` below it), and the report loop to:

```js
for (const r of out.rows) {
  console.log(`${`${r.w}x${r.h}`.padEnd(10)} live   median ${r.median.toFixed(1)} ms/frame (${(1000 / r.median).toFixed(1)} fps)`);
  if (r.sets) {
    console.log(`${"".padEnd(10)} cached median ${r.cachedOn.toFixed(1)} ms jet on (${(1000 / r.cachedOn).toFixed(1)} fps), ${r.cachedOff.toFixed(1)} ms jet off; ` +
      `build ${r.build.toFixed(0)} ms/set x ${r.sets}; bookmarks ${(100 * r.bmFrac).toFixed(1)} % mean nJet ${r.meanNJet.toFixed(1)}`);
  } else console.log(`${"".padEnd(10)} cache unavailable on this adapter`);
}
```

- [ ] **Step 2: Measure**

Run (dev server up): `npm run bench` twice back-to-back; use the second run (warm). Record both resolutions' live and cached numbers.

- [ ] **Step 3: Document** — in `README.md`'s Status section add a "Geodesic cache (2026-10-01)" paragraph: what it does in two sentences, the measured cached ms/frame and fps at 1280×720 and 1920×1080 (jet on/off) next to the live numbers, the build time, the bookmark share and mean nJet, the `?cachecheck` maxima (from the last verify run), and whether the 60 fps target was reached — as measured, not claimed. In the spec, fill an `## Outcome` section at the top with the same numbers and set **Status:** `implemented on perf/geodesic-cache`.

- [ ] **Step 4: Final gates**

Run: `npm test && npm run build`, `npm run verify:gpu` (parity, shadow, golden, cachecheck, sky, cache-in-app all PASS), `node scripts/probe-axis.mjs` (PASS).

- [ ] **Step 5: Commit**

```bash
git add scripts/bench.mjs README.md docs/specs/2026-10-01-geodesic-cache-design.md
git commit -m "Benchmark and document the geodesic cache"
```
