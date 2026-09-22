# Performance (Smooth First) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the animated default view smooth on an RTX laptop (target 60 fps while animating, full resolution when paused) without loosening any physics gate.

**Architecture:** A committed benchmark measures every change. The render loop loses its 15 fps cap and gains an internal-resolution scale driven by a small frame-time controller; the trace, accumulation and bloom run at the internal size and the present pass upsamples bilinearly. The integrator gets cheaper in both twins (CPU `src/physics/*.ts`, GPU `src/render/integrator-shared.wgsl`): longer sweep-chosen far-field strides, then closed-form metric derivatives replacing finite differences.

**Tech Stack:** TypeScript, WGSL, WebGPU, Vite (`?raw` imports), Vitest, playwright-core (headless Chrome for GPU scripts).

**Spec:** `docs/specs/2026-09-23-performance-design.md`

## Global Constraints

- **No `Co-Authored-By` trailer in any commit.** After each commit: `git log -1 --format=%B | grep -ci "co-authored"` must print `0`. This project rule overrides any default attribution guidance.
- Work on branch `perf/smooth-first`, created from `main` at `4ba0543`. **Do not push. Do not merge. Do not switch branches** (restoring individual files with `git checkout <rev> -- <files>` for a before/after proof is allowed; `git status` must be clean before and after).
- Baseline gates at `4ba0543` (must stay green; any number that legitimately moves is re-baselined **explicitly** in the commit message and README, never tuned to pass):
  - `npm test` → **70 passed, 2 skipped** (grows only by the tests this plan adds)
  - `npm run build` → clean (`tsc --noEmit` has `noUnusedLocals`)
  - `npm run verify:gpu` → `PARITY PASS — maxRelErr=2.663e-4 over 53 cases`, `SHADOW PASS … 3.89 M … 0.749`, sky `200`, exit 0
  - `node scripts/probe-axis.mjs` → `✓ PASS` at i = 72° and i = 8°, exit 0
- Never change `?parity`'s `1e-3` pass threshold (`src/main.ts`), `H_TOL`, `MAX_RETRY`, `POLE_S2`, `F_AXIS`, `DL_FAR_MIN` except where a task's measured sweep rule selects a new value.
- The CPU twin (`src/physics/trace.ts`, `geodesic.ts`, `kerr.ts`) and the GPU twin (`src/render/integrator-shared.wgsl`) must stay structurally identical: every integrator change goes into both, same order, same predicates, same constants.
- GPU scripts need the dev server on **:5173** (`npm run dev`, leave it running). If `curl http://localhost:5173/` doesn't answer 200, start it in the background.
- `verify-gpu.mjs` is blind to WGSL compile errors: Chrome reports them as console *warnings* and the output buffer stays zeroed, so a broken shader shows up as a nonsense number. To diagnose, write a scratch `scripts/_diag.mjs` (inside `scripts/` so `playwright-core` resolves; `import` syntax, the package is `"type": "module"`) with `page.on("console", m => console.log(m.type(), m.text()))`; delete it before committing. `meta` is a reserved WGSL word; check new identifiers.
- Sweeps are env-gated. PowerShell: `$env:SWEEP = "1"; npx vitest run tests/<file>; Remove-Item Env:SWEEP`. Bash: `SWEEP=1 npx vitest run tests/<file>`.
- Every task ends by running `npm run bench` (from Task 1 on) and appending its row to the README "Performance" table.

## Review Focus

1. **Tab hidden, then shown** — the first frame back has a multi-second delta; the controller must ignore it and not slam the scale to 50 %. (Task 3: `tests/scale.test.ts` "ignores stalls".)
2. **Window resized while scaled down** — buffers are reallocated at display size and the internal size is recomputed from the kept scale; no out-of-bounds reads, no garbage frame. (Task 3: `scripts/probe-scale.mjs` resizes at a pinned 50 %.)
3. **Pausing while scaled down** — must return to 100 % and converge to a sharp still. (Task 3: `probe-scale.mjs` pause check; `tests/scale.test.ts` `reset`.)
4. **Tiny or odd window sizes** (e.g. 301×157 at 50 %) — internal size ≥ 1, bloom dims rounded up, bilinear taps clamped; frame renders, no WGSL/validation warnings. (Task 3: `probe-scale.mjs` tiny-viewport case.)
5. **False "integrated GPU" warning on a discrete card** (NVIDIA, Intel Arc, AMD RX) — the warning must only fire for integrated parts. (Task 2: `tests/gpuinfo.test.ts`.)

## File Structure

| File | Responsibility |
|---|---|
| `scripts/bench.mjs` | **Create (T1).** Headless GPU benchmark: adapter + median ms/frame at 1280×720 and 1920×1080, scale 1.0. |
| `src/main.ts` | **Modify.** T1: idle `?bench` route. T2: GPU readout. T3: uncapped loop, scale controller, `?scale=` pin, uniform `outW/outH`. |
| `package.json` | **Modify (T1).** `"bench"` script. |
| `README.md` | **Modify (every task).** "Performance" table; gate numbers; T5 limitation text. |
| `src/render/gpuinfo.ts` | **Create (T2).** `GpuInfo`, `describeGpu`, `isIntegratedGpu`. Pure. |
| `tests/gpuinfo.test.ts` | **Create (T2).** |
| `index.html` | **Modify (T2).** GPU + render-scale readout cells, warning line. |
| `src/render/gpu.ts` | **Modify.** T2: keep `adapterInfo`. T3: `displayW/H`, `scale`, `setScale`, buffers at display size, readback at display size. |
| `src/render/scale.ts` | **Create (T3).** `ScaleController`. Pure. |
| `tests/scale.test.ts` | **Create (T3).** |
| `src/render/uniforms.ts`, `tests/uniforms.test.ts` | **Modify (T3).** `outW`, `outH`; 96 → 112 bytes. |
| `src/render/present.wgsl` | **Modify (T3).** Full `Uniforms` struct incl. `outW/outH`; bilinear upsample of accum from internal to display size. |
| `src/test/shadow.browser.ts` | **Modify (T3).** Pass `outW/outH`. |
| `scripts/probe-scale.mjs` | **Create (T3).** Browser check of resize / tiny viewport / pause at scaled resolutions. |
| `src/physics/trace.ts`, `src/render/integrator-shared.wgsl` | **Modify (T4, T5).** T4: `K_FAR`, `DL_FAR_MAX`. T5: drop `h` param. |
| `tests/sweep-farstride.test.ts` | **Create (T4).** Env-gated sweep that selects `K_FAR`, `DL_FAR_MAX`. |
| `tests/sweep-axiscap.test.ts` | **Modify (T4).** Its local `stepSizeLocal` far base uses `K_FAR`/`DL_FAR_MAX`. |
| `src/physics/kerr.ts` | **Modify (T5).** `metricUpperGrad`. |
| `src/physics/geodesic.ts` | **Modify (T5).** Analytic `rhs`; old one kept as `rhsFD`; `rk4` loses `h`. |
| `tests/metric-grad.test.ts` | **Create (T5).** Analytic vs Richardson reference. |
| `src/test/parity.browser.ts` | **Modify (T5).** Drop `GPU_FD_H`; rewrite the barrier-case comment. |
| `docs/specs/2026-09-23-performance-design.md` | **Modify (T5).** Status line + outcome. |

---

### Task 1: Benchmark script and idle bench route

**Files:**
- Create: `scripts/bench.mjs`
- Modify: `src/main.ts:9` (add a `?bench` branch), `package.json` (scripts), `README.md` (new "Performance" section)

**Interfaces:**
- Consumes: `Renderer` (`init`, `uploadLUTs`, `uploadHotSpots`, `rebind`, `frame`, `device`, `width`, `height`), `buildTempLUT`, `buildColorLUT`, `iscoRadius`.
- Produces: `npm run bench` printing lines of the exact form `adapter: <vendor> <architecture>` and `<W>x<H>  median <ms> ms/frame (<fps> fps)`. Later tasks run it and append README rows. It reads `r.displayW ?? r.width` / `r.displayH ?? r.height` for `outW/outH` so it keeps working after Task 3.

- [ ] **Step 1: Create the branch and record baseline gates**

```bash
git checkout -b perf/smooth-first
npm test 2>&1 | tail -4          # 70 passed, 2 skipped
npm run build                     # clean
npm run verify:gpu                # PARITY PASS 2.663e-4 over 53; SHADOW PASS 3.89 / 0.749; sky 200
node scripts/probe-axis.mjs       # PASS x2
```

If any number differs, STOP and report.

- [ ] **Step 2: Add the idle `?bench` route to `src/main.ts`**

Insert immediately before the final `} else {` that starts the interactive render (after the `?shadow` branch):

```ts
} else if (location.search.includes("bench")) {
  // Idle route for scripts/bench.mjs: no render loop, no validation pass. The benchmark imports the
  // Renderer through the dev server and drives it itself, so nothing else may compete for the GPU.
  document.body.innerHTML = `<pre style="color:#888;padding:20px">bench route (idle)</pre>`;
```

- [ ] **Step 3: Create `scripts/bench.mjs`**

```js
// GPU benchmark: drives the real Renderer (loaded through the dev server) in headless Chrome and
// reports the WebGPU adapter plus the median GPU time per frame of the default animated scene at
// full internal resolution. A report, not a gate -- timings vary across machines.
//
//   npm run dev        # in another terminal (:5173)
//   npm run bench
//
// --force_high_performance_gpu makes Chrome hand WebGPU the discrete GPU on hybrid laptops; without
// it Chrome returned the Intel iGPU for every powerPreference on the dev machine (2026-09-23).
// The sky panorama is not loaded (skyStrength 0), matching the procedural fallback.
import { chromium } from "playwright-core";

const BASE = process.env.BASE || "http://localhost:5173";
const browser = await chromium.launch({
  channel: "chrome", headless: true,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--force_high_performance_gpu"],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
const diags = [];
page.on("console", (m) => { if (m.type() === "warning" || m.type() === "error") diags.push(m.text()); });
page.on("pageerror", (e) => diags.push("PAGEERROR " + e.message));
await page.goto(BASE + "/?bench", { waitUntil: "load", timeout: 20000 });

const out = await page.evaluate(async () => {
  const ad = await navigator.gpu.requestAdapter();
  const info = (ad && ad.info) || {};
  const { Renderer } = await import("/src/render/gpu.ts");
  const { buildTempLUT, buildColorLUT } = await import("/src/physics/lookups.ts");
  const { iscoRadius } = await import("/src/physics/orbits.ts");
  async function bench(w, h, frames = 8) {
    document.body.innerHTML = "";
    const c = document.createElement("canvas");
    c.style.cssText = `width:${w}px;height:${h}px;display:block`;
    document.body.appendChild(c);
    const r = new Renderer(); await r.init(c);
    const a = 0.9, rIn = iscoRadius(a, true);
    r.uploadLUTs(buildTempLUT(a, true, rIn, 40, 512), buildColorLUT(1000, 40000, 256)); r.rebind();
    r.uploadHotSpots(new Float32Array([8, 0, 1.2, 1.8, 12, 2.1, 1.6, 1.2, 16, 4.3, 2.0, 0.9])); r.rebind();
    const u = (f) => ({
      resW: r.width, resH: r.height, outW: r.displayW ?? r.width, outH: r.displayH ?? r.height,
      a, incl: (72 * Math.PI) / 180, rObs: 1000, fovScale: 14, rIn, rOut: 40, Tpeak: 3e4, exposure: 1.6,
      time: f, frame: f, reset: f === 0 ? 1 : 0, maxSteps: 4800, blend: f === 0 ? 1 : 0.15, timeScale: 1,
      turbAmp: 0.6, breatheAmp: 0, nSpots: 3, jetStrength: 1, jetGamma: 5, jetLength: 60, jetKnots: 0.7,
      skyStrength: 0,
    });
    r.frame(u(0)); await r.device.queue.onSubmittedWorkDone(); // warm-up: pipeline compile
    const t = [];
    for (let f = 1; f <= frames; f++) {
      const t0 = performance.now(); r.frame(u(f)); await r.device.queue.onSubmittedWorkDone();
      t.push(performance.now() - t0);
    }
    t.sort((x, y) => x - y); r.device.destroy();
    return { w: r.width, h: r.height, median: t[t.length >> 1] };
  }
  return { adapter: `${info.vendor || "?"} ${info.architecture || "?"}`, rows: [await bench(1280, 720), await bench(1920, 1080)] };
});

console.log(`adapter: ${out.adapter}`);
for (const r of out.rows) console.log(`${`${r.w}x${r.h}`.padEnd(10)} median ${r.median.toFixed(1)} ms/frame (${(1000 / r.median).toFixed(1)} fps)`);
if (diags.length) console.log("console diagnostics:", diags.join(" | "));
await browser.close();
process.exit(diags.some((d) => d.startsWith("PAGEERROR")) ? 1 : 0);
```

- [ ] **Step 4: Add the npm script**

In `package.json` `"scripts"`, after `"verify:gpu"`: `"bench": "node scripts/bench.mjs"` (add the comma on the previous line).

- [ ] **Step 5: Run it**

```bash
npm run bench
```

Expected: `adapter: nvidia <arch>` on this machine and two timing lines (last measured by the controller on the RTX 3050: ~110 ms at 1280×720, ~263 ms at 1920×1080). If the adapter is Intel, the force flag didn't take — report it; do not continue with Intel numbers.

- [ ] **Step 6: README "Performance" section**

Append a new section before `## Status` in `README.md`:

```markdown
## Performance

`npm run bench` (dev server running) reports the WebGPU adapter and the median GPU time per frame of the
default animated scene at full internal resolution. It forces Chrome onto the discrete GPU
(`--force_high_performance_gpu`); in a normal browser, set Chrome to *High performance* in Windows
Settings → System → Display → Graphics, otherwise WebGPU may run on the integrated GPU (measured 5–8×
slower on the dev laptop).

| Change | Adapter | 1280×720 ms/frame | 1920×1080 ms/frame |
|---|---|---|---|
| Baseline (`4ba0543`) | <paste> | <paste> | <paste> |
```

Replace each `<paste>` with the numbers Step 5 printed.

- [ ] **Step 7: Gates and commit**

```bash
npm run build && npm test 2>&1 | tail -4 && npm run verify:gpu
git add scripts/bench.mjs src/main.ts package.json README.md
git commit -m "Add a GPU benchmark (npm run bench) and an idle ?bench route

Reports the WebGPU adapter and the median GPU ms/frame of the default
scene at 1280x720 and 1920x1080, full internal resolution. Baseline:
<adapter>, <ms> / <ms>."
git log -1 --format=%B | grep -ci "co-authored"   # 0
```

---

### Task 2: GPU readout and integrated-GPU warning

**Files:**
- Create: `src/render/gpuinfo.ts`, `tests/gpuinfo.test.ts`
- Modify: `src/render/gpu.ts` (`init`: store adapter info), `index.html` (readout cells + warning + CSS), `src/main.ts` (fill the readout)

**Interfaces:**
- Consumes: `GPUAdapter.info` (Chrome 121+: `vendor`, `architecture`, `description`, all strings, possibly empty).
- Produces: `export interface GpuInfo { vendor: string; architecture: string; description: string }`, `export function describeGpu(i: GpuInfo): string`, `export function isIntegratedGpu(i: GpuInfo): boolean`; `Renderer.adapterInfo: GpuInfo`; DOM ids `#gpu`, `#rscale`, `#gpuwarn` (Task 3 writes `#rscale`).

- [ ] **Step 1: Write the failing tests** — `tests/gpuinfo.test.ts`

```ts
import { describe, it, expect } from "vitest";
import { describeGpu, isIntegratedGpu, type GpuInfo } from "../src/render/gpuinfo";

const g = (vendor: string, architecture = "", description = ""): GpuInfo => ({ vendor, architecture, description });

describe("gpuinfo", () => {
  it("flags Intel integrated graphics (what Chrome returned on the dev laptop)", () => {
    expect(isIntegratedGpu(g("intel", "gen-12lp"))).toBe(true);
    expect(isIntegratedGpu(g("intel", "xe-lpg"))).toBe(true);
  });
  it("does not flag discrete GPUs", () => {
    expect(isIntegratedGpu(g("nvidia", "ampere"))).toBe(false);
    expect(isIntegratedGpu(g("nvidia", "blackwell"))).toBe(false);
    expect(isIntegratedGpu(g("intel", "xe-hpg"))).toBe(false);      // Arc A-series
    expect(isIntegratedGpu(g("intel", "xe2-hpg"))).toBe(false);     // Arc B-series
    expect(isIntegratedGpu(g("amd", "rdna-3", "AMD Radeon RX 7600"))).toBe(false);
  });
  it("flags AMD integrated only when the description says so", () => {
    expect(isIntegratedGpu(g("amd", "rdna-3", "AMD Radeon(TM) Graphics"))).toBe(true);
    expect(isIntegratedGpu(g("amd", "gcn-5", "AMD Radeon Vega 8 Graphics"))).toBe(true);
    expect(isIntegratedGpu(g("amd", "rdna-3", ""))).toBe(false);    // unknown: no false alarm
  });
  it("never flags unknown adapters", () => {
    expect(isIntegratedGpu(g(""))).toBe(false);
  });
  it("describes the adapter compactly", () => {
    expect(describeGpu(g("nvidia", "ampere"))).toBe("nvidia ampere");
    expect(describeGpu(g("", ""))).toBe("unknown");
  });
});
```

- [ ] **Step 2: Run to confirm failure**

Run: `npx vitest run tests/gpuinfo.test.ts` — Expected: FAIL (cannot resolve `../src/render/gpuinfo`).

- [ ] **Step 3: Implement `src/render/gpuinfo.ts`**

```ts
/** What WebGPU reports about the adapter it handed us (GPUAdapter.info; strings may be empty). */
export interface GpuInfo { vendor: string; architecture: string; description: string; }

export function describeGpu(i: GpuInfo): string {
  return `${i.vendor} ${i.architecture}`.trim() || "unknown";
}

/** True only when the adapter is recognisably an integrated GPU. On hybrid laptops Chrome can hand
 *  WebGPU the iGPU even when a discrete card is present (measured on the dev machine: Iris Xe for
 *  every powerPreference, 5-8x slower than the RTX). Unknown adapters are never flagged, so a
 *  missing description can only cost a missing warning, never a false one. */
export function isIntegratedGpu(i: GpuInfo): boolean {
  const v = i.vendor.toLowerCase(), arch = i.architecture.toLowerCase(), d = i.description.toLowerCase();
  if (v === "intel") return !arch.includes("hpg"); // Arc discrete parts report xe-hpg / xe2-hpg
  if (v === "amd" || v === "ati") return /radeon(\(tm\))? graphics|vega \d+ graphics/.test(d);
  return false;
}
```

- [ ] **Step 4: Run tests** — `npx vitest run tests/gpuinfo.test.ts` → PASS (5).

- [ ] **Step 5: Store the adapter info in `Renderer`**

In `src/render/gpu.ts` add `import type { GpuInfo } from "./gpuinfo";`, a field `adapterInfo: GpuInfo = { vendor: "", architecture: "", description: "" };` next to `renderBloom`, and in `init` right after the `if (!adapter) throw …` line:

```ts
    const info = (adapter as GPUAdapter & { info?: Partial<GpuInfo> }).info ?? {};
    this.adapterInfo = { vendor: info.vendor ?? "", architecture: info.architecture ?? "", description: info.description ?? "" };
```

- [ ] **Step 6: Readout markup** — `index.html`

In the `<style>` block after the `.ro b i{…}` rule add:

```css
  .ro.wide{grid-column:1/-1}
  #gpuwarn{margin:12px 0 0;font-size:11px;line-height:1.5;color:#ffb070}
```

Inside `<div id="readout">`, after the `Samples` cell, add:

```html
      <div class="ro wide"><span>GPU</span><b id="gpu">—</b></div>
      <div class="ro"><span>Render scale</span><b id="rscale">100%</b></div>
```

and immediately after the closing `</div>` of `#readout`:

```html
    <p id="gpuwarn" hidden>WebGPU is running on the integrated GPU. Set Chrome to <b>High performance</b> in Windows Settings → System → Display → Graphics, then restart Chrome.</p>
```

- [ ] **Step 7: Fill it from `src/main.ts`**

Add `import { describeGpu, isIntegratedGpu } from "./render/gpuinfo";` to the imports. In the interactive branch, right after the `try { await r.init(canvas); } catch …` block:

```ts
  $("gpu").textContent = describeGpu(r.adapterInfo);
  if (isIntegratedGpu(r.adapterInfo)) $("gpuwarn").hidden = false;
```

`$` is declared later in the file today (`const $ = (id: string) => …`); move that one line up to just after `r.init` succeeds so it is defined before use.

- [ ] **Step 8: Verify in the browser**

`npm run build`, `npm test` (→ 75 passed, 2 skipped), `npm run verify:gpu` (unchanged numbers). Then write a scratch `scripts/_diag.mjs` that opens `/` in headless Chrome with the verify-gpu flags plus `--force_high_performance_gpu`, waits 3 s, prints `#gpu` text and whether `#gpuwarn` is hidden; run it once **without** the force flag too (expect `intel gen-12lp` and the warning visible on this machine). Delete the scratch file.

- [ ] **Step 9: Bench, README row, commit**

Run `npm run bench`, add a row `GPU readout (no perf change expected)` to the README table.

```bash
git add src/render/gpuinfo.ts tests/gpuinfo.test.ts src/render/gpu.ts index.html src/main.ts README.md
git commit -m "Show which GPU WebGPU is using and warn on an integrated GPU

<paste the two _diag observations: forced -> nvidia, warning hidden;
default -> intel gen-12lp, warning shown>"
git log -1 --format=%B | grep -ci "co-authored"   # 0
```

---

### Task 3: Uncapped loop and adaptive internal resolution

**Files:**
- Create: `src/render/scale.ts`, `tests/scale.test.ts`, `scripts/probe-scale.mjs`
- Modify: `src/render/uniforms.ts`, `tests/uniforms.test.ts`, `src/render/present.wgsl`, `src/render/gpu.ts`, `src/main.ts`, `src/test/shadow.browser.ts`, `README.md`

**Interfaces:**
- Consumes: Task 2's `#rscale`; Task 1's bench.
- Produces: `UniformValues.outW`, `UniformValues.outH` (display size, floats 24–25); `UNIFORM_SIZE = 112`; `Renderer.displayW`, `Renderer.displayH`, `Renderer.scale`, `Renderer.setScale(s: number): boolean` (true when the internal size changed); `Renderer.width/height` now mean the **internal** size; `export class ScaleController { scale: number; update(dtMs: number, nowMs: number): number | null; reset(scale?: number): void }`; query param `?scale=<0.5..1>` pins the scale and disables the controller.

- [ ] **Step 1: Failing controller tests** — `tests/scale.test.ts`

```ts
import { describe, it, expect } from "vitest";
import { ScaleController } from "../src/render/scale";

/** Feed `n` frames of `dt` ms starting at time `t0`; returns the time after the last frame. */
function feed(c: ScaleController, dt: number, n: number, t0 = 0) {
  let t = t0;
  for (let i = 0; i < n; i++) { t += dt; c.update(dt, t); }
  return t;
}

describe("ScaleController", () => {
  it("starts at full resolution", () => { expect(new ScaleController().scale).toBe(1); });

  it("steps down under sustained slow frames and stops at 0.5", () => {
    const c = new ScaleController();
    const t = feed(c, 40, 30);                      // 1.2 s of 25 fps
    expect(c.scale).toBeLessThan(1);
    feed(c, 40, 400, t);                            // long enough to hit the floor
    expect(c.scale).toBe(0.5);
  });

  it("holds steady inside the dead band (60 Hz vsync = 16.7 ms)", () => {
    const c = new ScaleController();
    c.reset(0.7);
    feed(c, 16.7, 600);
    expect(c.scale).toBe(0.7);
  });

  it("climbs back to exactly 1.0 when frames are fast", () => {
    const c = new ScaleController();
    c.reset(0.5);
    feed(c, 8, 1000);
    expect(c.scale).toBe(1);
  });

  it("changes at most once per 500 ms", () => {
    const c = new ScaleController();
    const changes: number[] = [];
    let t = 0;
    for (let i = 0; i < 100; i++) { t += 40; if (c.update(40, t) !== null) changes.push(t); }
    for (let k = 1; k < changes.length; k++) expect(changes[k] - changes[k - 1]).toBeGreaterThanOrEqual(500);
    expect(changes.length).toBeGreaterThan(1);
  });

  it("ignores stalls (tab hidden, then shown) instead of collapsing the scale", () => {
    const c = new ScaleController();
    let t = feed(c, 16.7, 60);
    t += 5000; expect(c.update(5000, t)).toBeNull();
    feed(c, 16.7, 60, t);
    expect(c.scale).toBe(1);
  });

  it("ignores non-positive or NaN deltas", () => {
    const c = new ScaleController();
    expect(c.update(0, 100)).toBeNull();
    expect(c.update(-5, 200)).toBeNull();
    expect(c.update(NaN, 300)).toBeNull();
    expect(c.scale).toBe(1);
  });
});
```

Run `npx vitest run tests/scale.test.ts` → FAIL (module not found).

- [ ] **Step 2: Implement `src/render/scale.ts`**

```ts
/** Internal-resolution controller for the animated view ("smooth first", spec 3.3). Feed it each
 *  frame's wall-clock delta; it keeps a smoothed frame time and nudges the render scale down (x0.9)
 *  when frames are slower than SLOW_MS and up (x1.1) when faster than FAST_MS, within [MIN, MAX],
 *  at most once per HOLD_MS so the image does not pump. Deltas above MAX_DT_MS (tab switch, stall)
 *  are not a performance signal and are ignored. */
export const SCALE = { MIN: 0.5, MAX: 1, SLOW_MS: 18, FAST_MS: 13, DOWN: 0.9, UP: 1.1, HOLD_MS: 500, ALPHA: 0.2, MAX_DT_MS: 250 };

export class ScaleController {
  scale = 1;
  private ema = -1;
  private lastChange = -Infinity;

  /** Returns the new scale when it changed this frame, else null. */
  update(dtMs: number, nowMs: number): number | null {
    if (!(dtMs > 0) || dtMs > SCALE.MAX_DT_MS) return null;
    this.ema = this.ema < 0 ? dtMs : this.ema + SCALE.ALPHA * (dtMs - this.ema);
    if (nowMs - this.lastChange < SCALE.HOLD_MS) return null;
    let next = this.scale;
    if (this.ema > SCALE.SLOW_MS) next = this.scale * SCALE.DOWN;
    else if (this.ema < SCALE.FAST_MS) next = this.scale * SCALE.UP;
    next = Math.min(SCALE.MAX, Math.max(SCALE.MIN, next));
    if (next === this.scale) return null;
    this.scale = next;
    this.lastChange = nowMs;
    this.ema = -1; // re-measure at the new scale rather than reacting to the old one's history
    return next;
  }

  reset(scale = 1) { this.scale = scale; this.ema = -1; this.lastChange = -Infinity; }
}
```

Run `npx vitest run tests/scale.test.ts` → PASS (7).

- [ ] **Step 3: Uniforms gain `outW`, `outH`** — update the test first

In `tests/uniforms.test.ts`: rename the test to `"is 112 bytes and packs all fields (incl. display size) at the expected offsets"`, change `expect(UNIFORM_SIZE).toBe(96)` to `toBe(112)`, add `outW: 200, outH: 100,` to the object literal, and append:

```ts
    expect(dv.getFloat32(96, true)).toBeCloseTo(200);   // outW (index 24)
    expect(dv.getFloat32(100, true)).toBeCloseTo(100);  // outH (index 25)
```

Run it → FAIL. Then in `src/render/uniforms.ts`: add to the header comment `//         + outW,outH (2)                                               -> 22 floats`; add `outW: number; outH: number;` to `UniformValues`; set `export const UNIFORM_FLOATS = 22, UNIFORM_UINTS = 4;` (size → `ceil(26/4)*16 = 112`); in `packUniforms` add `f[24] = u.outW; f[25] = u.outH;`. Run → PASS.

- [ ] **Step 4: `present.wgsl` — full struct + bilinear upsample**

Replace the `Uniforms` struct with:

```wgsl
struct Uniforms {
  res: vec2<f32>, a: f32, incl: f32, rObs: f32, fovScale: f32, rIn: f32, rOut: f32,
  Tpeak: f32, exposure: f32, time: f32, frame: u32, reset: u32, maxSteps: u32,
  blend: f32, timeScale: f32, turbAmp: f32, breatheAmp: f32, nSpots: u32,
  jetStrength: f32, jetGamma: f32, jetLength: f32, jetKnots: f32,
  skyStrength: f32, outW: f32, outH: f32,
};
```

Replace the body of `fs` up to (not including) `hdr = hdr * exp2(U.exposure);` with:

```wgsl
  // U.res is the INTERNAL trace size; the framebuffer is (outW, outH). q is this fragment's centre
  // in internal-pixel units. At scale 1 the ratio is exactly 1, ip is an integer and f == 0, so
  // every tap below reduces to the old nearest read -- the full-resolution image is unchanged.
  let res = vec2<u32>(u32(U.res.x), u32(U.res.y));
  let q = fragCoord.xy * (U.res / vec2<f32>(U.outW, U.outH));
  let ip = clamp(q - 0.5, vec2<f32>(0.0), U.res - 1.0);
  let i0 = floor(ip);
  let f = ip - i0;
  let ax0 = u32(i0.x); let ay0 = u32(i0.y);
  let ax1 = min(ax0 + 1u, res.x - 1u); let ay1 = min(ay0 + 1u, res.y - 1u);
  var hdr = mix(mix(accum[ay0 * res.x + ax0].rgb, accum[ay0 * res.x + ax1].rgb, f.x),
                mix(accum[ay1 * res.x + ax0].rgb, accum[ay1 * res.x + ax1].rgb, f.x), f.y);

  // additive quarter-res bloom, bilinearly upsampled (nearest-tap would show 4x4 blocks)
  let bw = (res.x + 3u) / 4u;
  let bh = (res.y + 3u) / 4u;
  let bp = q / 4.0 - 0.5;
  let bi = floor(bp);
  let bf = bp - bi;
  let x0 = u32(clamp(bi.x, 0.0, f32(bw - 1u)));
  let y0 = u32(clamp(bi.y, 0.0, f32(bh - 1u)));
  let x1 = min(x0 + 1u, bw - 1u);
  let y1 = min(y0 + 1u, bh - 1u);
  let cb = mix(mix(bloom[y0 * bw + x0].rgb, bloom[y0 * bw + x1].rgb, bf.x),
               mix(bloom[y1 * bw + x0].rgb, bloom[y1 * bw + x1].rgb, bf.x), bf.y);
  hdr += cb * BLOOM;
```

and change the vignette's `let uv = vec2<f32>(fragCoord.x, fragCoord.y) / U.res - 0.5;` to `let uv = q / U.res - 0.5;`.

- [ ] **Step 5: `Renderer` — display vs internal size**

In `src/render/gpu.ts`:

- Fields: replace `width = 0; height = 0; bw = 0; bh = 0;` with
  ```ts
  displayW = 0; displayH = 0;          // framebuffer size (canvas pixels)
  scale = 1;                           // internal render scale in [0.5, 1]; see setScale
  width = 0; height = 0; bw = 0; bh = 0; // INTERNAL trace size and its quarter-res bloom size
  ```
- `resize(canvas)`: compute `displayW/displayH` from `clientWidth/Height * dpr` (same dpr rule), set `canvas.width/height` to the display size, configure the context, allocate `accumBuf` at `displayW * displayH * 16` bytes and both bloom buffers at `Math.ceil(displayW / 4) * Math.ceil(displayH / 4) * 16` bytes (the largest the internal size can reach), then call `this.applyScale()`.
- Add:
  ```ts
  /** Internal size from the display size and scale. Buffers are sized for scale 1 and indexed at the
   *  internal width, so a scale change never reallocates (and needs no rebind). */
  private applyScale() {
    this.width = Math.max(1, Math.round(this.displayW * this.scale));
    this.height = Math.max(1, Math.round(this.displayH * this.scale));
    this.bw = Math.ceil(this.width / 4); this.bh = Math.ceil(this.height / 4);
  }
  /** Returns true when the internal size changed (the caller must restart accumulation). */
  setScale(s: number): boolean {
    const v = Math.min(1, Math.max(0.5, s));
    if (v === this.scale) return false;
    const w = this.width, h = this.height;
    this.scale = v; this.applyScale();
    return this.width !== w || this.height !== h;
  }
  ```
- `readbackPresented`: the texture, `bytesPerRow`, the copy extent, the de-padding loop and the returned `w/h` use `displayW/displayH` instead of `width/height`.

- [ ] **Step 6: Callers pass `outW/outH`**

`src/test/shadow.browser.ts`: in the `UniformValues` literal add `outW: r.displayW, outH: r.displayH,` after `resH: r.height,`. `scripts/bench.mjs` already passes them (Task 1).

- [ ] **Step 7: Uncapped loop + controller in `src/main.ts`**

- Imports: `import { ScaleController } from "./render/scale";`.
- After the GPU readout lines from Task 2:
  ```ts
  // ?scale=0.5 .. 1 pins the internal render scale and disables the controller (debugging, probes).
  const scaleParam = new URLSearchParams(location.search).get("scale");
  const pinnedScale = scaleParam !== null && isFinite(+scaleParam) ? Math.min(1, Math.max(0.5, +scaleParam)) : null;
  if (pinnedScale !== null) r.setScale(pinnedScale);
  const ctl = new ScaleController();
  const rscaleEl = $("rscale");
  const showScale = () => { rscaleEl.textContent = `${Math.round(r.scale * 100)}%`; };
  showScale();
  ```
- Replace the whole block from `// On slower GPUs, cap accumulation …` through `requestAnimationFrame(loop);` (the call after the function) with:
  ```ts
  // Uncapped: render every animation frame. While animating, the ScaleController trades internal
  // resolution for frame rate (target ~60 fps, never below half resolution); paused, the scale snaps
  // back to 1 and the progressive running mean converges to a sharp still.
  function loop(now: number) {
    const dt = lastNow ? now - lastNow : 0; lastNow = now;
    if (state.playing) simTime += (dt / 1000) * SPEED;
    if (pinnedScale === null) {
      if (state.playing) {
        const ns = ctl.update(dt, now);
        if (ns !== null && r.setScale(ns)) { reset(); showScale(); }
      } else if (r.scale !== 1) {
        r.setScale(1); ctl.reset(1); reset(); showScale();
      }
    }
    // Playing: fixed EMA (blend==1 on the reset frame to clear). Paused: progressive running mean.
    const blend = state.playing ? (sample === 0 ? 1 : EMA_BLEND) : 1 / (sample + 1);
    const u: UniformValues = {
      resW: r.width, resH: r.height, outW: r.displayW, outH: r.displayH, a: state.a, incl: state.incl * Math.PI / 180,
      rObs: 1000, fovScale: 14, rIn, rOut, Tpeak: T_PEAK, exposure: state.exposure,
      time: simTime, frame: sample, reset: sample === 0 ? 1 : 0, maxSteps: state.maxSteps,
      blend, timeScale: state.timeScale, turbAmp: state.turbAmp,
      breatheAmp: state.breatheAmp, nSpots: baseSpots.length,
      jetStrength: state.jetStrength, jetGamma: state.jetGamma,
      jetLength: state.jetLength, jetKnots: state.jetKnots,
      skyStrength: skyReady ? state.skyStrength : 0,
    };
    r.frame(u);
    sample++;
    if ((sample & 7) === 0 || sample < 4) sppEl.textContent = String(sample);
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
  ```
  Delete the now-unused `TARGET_MS` and `lastFrame`. Keep `lastNow` (declared at the top).

- [ ] **Step 8: Gates**

`npm run build` (clean), `npm test` (→ 82 passed, 2 skipped: +7 scale), `npm run verify:gpu` (**must be unchanged**: 2.663e-4/53, 3.89/0.749 — validation routes run at scale 1 and the present pass is exact at scale 1), `node scripts/probe-axis.mjs` (PASS x2; its viewport renders the interactive view — if the headless controller lowered the scale, the probe's metrics still hold because they measure ratios; if a probe number changes materially, report it).

- [ ] **Step 9: `scripts/probe-scale.mjs`** (Review Focus 2, 3, 4)

```js
// Browser checks for the adaptive render scale: resize while scaled down, a tiny odd viewport, and
// pause restoring full resolution. Exits non-zero on failure. Needs the dev server on :5173.
import { chromium } from "playwright-core";

const BASE = process.env.BASE || "http://localhost:5173";
const browser = await chromium.launch({ channel: "chrome", headless: true,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--force_high_performance_gpu"] });
const page = await browser.newPage({ viewport: { width: 1000, height: 680 } });
const diags = [];
page.on("console", (m) => { if (m.type() === "warning" || m.type() === "error") diags.push(m.text()); });
page.on("pageerror", (e) => diags.push("PAGEERROR " + e.message));
let failed = false;
const check = (ok, msg) => { console.log(`${ok ? "✓ PASS" : "✗ FAIL"}  ${msg}`); if (!ok) failed = true; };

/** Mean RGB sum of the canvas region right of the panel (decoded in-page from a screenshot). */
async function meanLum() {
  const png = await page.screenshot({ type: "png" });
  return page.evaluate(async (b64) => {
    const img = new Image(); img.src = "data:image/png;base64," + b64; await img.decode();
    const cv = document.createElement("canvas"); cv.width = img.width; cv.height = img.height;
    const g = cv.getContext("2d"); g.drawImage(img, 0, 0);
    const x0 = Math.min(340, img.width - 1);
    const d = g.getImageData(x0, 0, img.width - x0, img.height).data;
    let s = 0; for (let i = 0; i < d.length; i += 4) s += d[i] + d[i + 1] + d[i + 2];
    return s / (d.length / 4);
  }, png.toString("base64"));
}
const rscale = () => page.textContent("#rscale");

// Pinned 50 %: resize twice (normal, then tiny and odd), frame must stay lit and warning-free.
await page.goto(BASE + "/?scale=0.5", { waitUntil: "load" });
await page.waitForTimeout(4000);
const ref = await meanLum();
check((await rscale()) === "50%" && ref > 20, `pinned 50%: readout ${await rscale()}, mean ${ref.toFixed(1)}`);
for (const [w, h] of [[700, 500], [301, 157]]) {
  await page.setViewportSize({ width: w, height: h });
  await page.waitForTimeout(3000);
  const m = await meanLum();
  check((await rscale()) === "50%" && m > 20, `resize to ${w}x${h} at 50%: mean ${m.toFixed(1)}`);
}
check(diags.length === 0, `no WebGPU warnings/errors after resizes${diags.length ? ": " + diags.join(" | ") : ""}`);

// Unpinned: headless is slow enough that the controller should drop below 100 %; pausing restores it.
await page.setViewportSize({ width: 1000, height: 680 });
await page.goto(BASE + "/", { waitUntil: "load" });
let dropped = false;
for (let i = 0; i < 30 && !dropped; i++) { await page.waitForTimeout(500); dropped = (await rscale()) !== "100%"; }
if (!dropped) console.log("• SKIP pause check: this GPU never needed to scale down at 1000x680");
else {
  await page.click("#playpause");
  await page.waitForTimeout(1500);
  check((await rscale()) === "100%", `pause restores full resolution (readout ${await rscale()})`);
}
await browser.close();
process.exit(failed ? 1 : 0);
```

Run `node scripts/probe-scale.mjs` → all PASS (the pause check may SKIP only if the GPU is fast enough — report which happened).

- [ ] **Step 10: See it**

Take screenshots of `/` (unpinned, animating, ~5 s in) and `/?scale=0.5` and look at them (the Read tool renders PNGs): the image must be the same scene, softer at 50 %, with no blocky seams, black borders or offset. Describe what you see in the report. Save them under `.superpowers/` (gitignored), not the repo root.

- [ ] **Step 11: Bench, README, commit**

`npm run bench` (scale 1 — measures only the loop/present changes; expect ≈ unchanged). Add a README row `Uncapped loop + adaptive scale (scale 1.0)` and one sentence under the table: "While animating, the render scale drops to as low as 50 % to hold ~60 fps; at the 1280×720 baseline cost that is roughly <baseline ms / 4> ms at 50 %." (compute from the baseline). Also update README's gate line for `npm test` to 82 + 2 skipped.

```bash
git add src/render/scale.ts tests/scale.test.ts src/render/uniforms.ts tests/uniforms.test.ts src/render/present.wgsl src/render/gpu.ts src/main.ts src/test/shadow.browser.ts scripts/probe-scale.mjs README.md
git commit -m "Uncap the render loop and adapt internal resolution while animating

Target ~60 fps by rendering at 50-100 % internal resolution (x0.9 / x1.1
steps outside a 13-18 ms dead band, at most one change per 500 ms, stalls
ignored); paused, snap to 100 % and converge. The present pass upsamples
bilinearly and is exact at scale 1, so ?parity / ?shadow are unchanged.
?scale= pins the scale. probe-scale.mjs checks resize, tiny viewports and
pause. Gates: <paste>."
git log -1 --format=%B | grep -ci "co-authored"   # 0
```

---

### Task 4: Far-field stride from a measured sweep

**Files:**
- Create: `tests/sweep-farstride.test.ts`
- Modify: `src/physics/trace.ts:40-53` (`stepSize` far base + two constants), `src/render/integrator-shared.wgsl:67-82` (same), `tests/sweep-axiscap.test.ts` (its local `stepSizeLocal` far base), `README.md`

**Interfaces:**
- Consumes: `stepGeodesic`, `traceRay`, `F_AXIS`, `DL_FAR_MIN`, `H_TOL`, `H_TOL_FAR`, `MAX_RETRY` from `trace.ts`; `screenToState`; `metricUpper`; `photonOrbit`.
- Produces: `export const K_FAR: number` and `export const DL_FAR_MAX: number` in `trace.ts` with identical WGSL twins; far base becomes `clamp(K_FAR * r, 0.6, DL_FAR_MAX)`.

- [ ] **Step 1: Introduce the constants at today's values (no behaviour change)**

In `trace.ts`, above `stepSize`:

```ts
/** Far-field stride: dl = K_FAR * r clamped to [0.6, DL_FAR_MAX] beyond rOut * 1.5, before the
 *  F_AXIS angular cap. Chosen from tests/sweep-farstride.test.ts (SWEEP=1). Twin constants in
 *  integrator-shared.wgsl. */
export const K_FAR = 0.04;
export const DL_FAR_MAX = 6;
```

and change `const base = Math.min(6, Math.max(0.6, 0.04 * r));` to `const base = Math.min(DL_FAR_MAX, Math.max(0.6, K_FAR * r));`. In `integrator-shared.wgsl` add, above `stepSize`:

```wgsl
// Far-field stride (see K_FAR / DL_FAR_MAX in trace.ts). Twin constants.
const K_FAR = 0.04;
const DL_FAR_MAX = 6.0;
```

and change `let base = clamp(0.04 * r, 0.6, 6.0);` to `let base = clamp(K_FAR * r, 0.6, DL_FAR_MAX);`. In `tests/sweep-axiscap.test.ts`, import `K_FAR, DL_FAR_MAX` and make its local `stepSizeLocal` far base use them the same way (its reproduction assertion against `traceRay` must keep holding when that sweep is re-run).

Run `npm test` (82 + 2 skipped) and `npm run verify:gpu` (unchanged numbers). Commit:

```bash
git add src/physics/trace.ts src/render/integrator-shared.wgsl tests/sweep-axiscap.test.ts
git commit -m "Name the far-field stride constants (K_FAR, DL_FAR_MAX) at their current values"
git log -1 --format=%B | grep -ci "co-authored"   # 0
```

- [ ] **Step 2: Write the sweep** — `tests/sweep-farstride.test.ts`

```ts
import { describe, it, expect } from "vitest";
import { screenToState } from "../src/physics/camera";
import { metricUpper } from "../src/physics/kerr";
import { photonOrbit } from "../src/physics/orbits";
import { stepGeodesic, traceRay, F_AXIS, DL_FAR_MIN, H_TOL, H_TOL_FAR, MAX_RETRY, K_FAR, DL_FAR_MAX } from "../src/physics/trace";

/**
 * Measurement, not a test: prints the K_FAR x DL_FAR_MAX table the far-field stride in trace.ts was
 * chosen from (spec 2026-09-23 3.4). Skipped unless SWEEP=1.
 *
 * The loop is a LOCAL COPY of traceRay with the far base parameterised; the test first asserts that
 * with the shipped constants it reproduces traceRay exactly on every ray, so the table is about the
 * shipped code. Reference = the shipped constants; it is itself checked against a finer stride
 * (K = 0.02, MAX = 3). Rules (applied in code, in order): 0 fate flips on every set; every disk hit
 * within 0.02 M of the reference; every escaped ray's sky direction (CPU twin of skyDir in
 * raytrace.wgsl, at the renderer's own exit point) within half a screen pixel of the reference,
 * 0.5 * 2 * fovScale / (720 * rObs) rad; then fewest mean steps; ties -> smaller K, then smaller MAX.
 */
const SWEEP = !!process.env.SWEEP;
const ROBS = 1000, ROUT = 40, FOV = 14, MAXSTEPS = 4800;
const HALF_PIXEL = 0.5 * (2 * FOV) / (720 * ROBS);

type Ray = { s0: Float64Array; a: number; rIn: number };
type Res = { fate: string; steps: number; rHit?: number; dir?: number[] };

function skyDirCPU(s: Float64Array, a: number): number[] {
  const r = s[1], th = s[2], ph = s[3];
  const g = metricUpper(r, th, a);
  const dr = g.rr * s[5], dth = g.thth * s[6], dph = g.tphi * s[4] + g.phph * s[7];
  const st = Math.sin(th), ct = Math.cos(th), sp = Math.sin(ph), cp = Math.cos(ph);
  const v = [dr * st * cp + r * ct * cp * dth - r * st * sp * dph,
             dr * st * sp + r * ct * sp * dth + r * st * cp * dph,
             dr * ct - r * st * dth];
  const n = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / n, v[1] / n, v[2] / n];
}

function stepSizeLocal(s: Float64Array, rh: number, K: number, MAXD: number): number {
  const r = s[1];
  if (r > ROUT * 1.5) {
    const base = Math.min(MAXD, Math.max(0.6, K * r));
    const pth = s[6];
    if (pth === 0) return base;
    const thD = Math.min(s[2], Math.PI - s[2]);
    return Math.min(base, Math.max(DL_FAR_MIN, F_AXIS * thD * r * r / Math.abs(pth)));
  }
  return Math.min(0.5, Math.max(0.002, 0.02 * (r - rh)));
}

function localTrace(ray: Ray, K: number, MAXD: number): Res {
  const { a, rIn } = ray, rh = 1 + Math.sqrt(Math.max(0, 1 - a * a));
  let s = ray.s0;
  for (let step = 1; step <= MAXSTEPS; step++) {
    const far = s[1] > ROUT * 1.5;
    const sN = stepGeodesic(s, a, stepSizeLocal(s, rh, K, MAXD), far ? H_TOL_FAR : H_TOL, MAX_RETRY).s;
    const f0 = s[2] - Math.PI / 2, f1 = sN[2] - Math.PI / 2;
    if (f0 * f1 < 0 && Math.abs(sN[2] - s[2]) < 0.5) {
      const frac = f0 / (f0 - f1), rHit = s[1] + frac * (sN[1] - s[1]);
      if (rHit >= rIn && rHit <= ROUT) return { fate: "disk", steps: step, rHit };
    }
    s = sN;
    if (s[1] <= rh * 1.005) return { fate: "captured", steps: step };
    if (s[1] > ROBS * 1.2) return { fate: "escaped", steps: step, dir: skyDirCPU(s, a) };
  }
  return { fate: "budget", steps: MAXSTEPS };
}

function sets(): { name: string; rays: Ray[] }[] {
  const grid = (a: number, inclDeg: number, n: number, rIn: number) => {
    const incl = (inclDeg * Math.PI) / 180, rays: Ray[] = [];
    for (let iy = 0; iy < n; iy++) for (let ix = 0; ix < n; ix++) {
      const alpha = (((ix + 0.5) / n) * 2 - 1) * FOV, beta = -(((iy + 0.5) / n) * 2 - 1) * FOV;
      rays.push({ s0: screenToState(alpha, beta, a, incl, ROBS), a, rIn });
    }
    return rays;
  };
  const band = (inclDeg: number, betas: number[], alphas: number[]) =>
    betas.flatMap((b) => alphas.map((al) => ({ s0: screenToState(al, b, 0, (inclDeg * Math.PI) / 180, ROBS), a: 0, rIn: 6 })));
  const lin = (lo: number, hi: number, n: number) => Array.from({ length: n }, (_, k) => lo + ((hi - lo) * k) / (n - 1));
  return [
    { name: "V1 a=0 i=8", rays: grid(0, 8, 32, photonOrbit(0, true)) },
    { name: "V2 a=0.9 i=72", rays: grid(0.9, 72, 32, photonOrbit(0.9, true)) },
    { name: "A axis band i=8", rays: band(8, lin(9, 14, 11), lin(0, 0.8, 11)) },
    { name: "B axis column i=1", rays: band(1, lin(2, 14, 13), lin(0, 1, 5)) },
  ];
}

const angle = (u: number[], v: number[]) => 2 * Math.asin(Math.min(1, Math.hypot(u[0] - v[0], u[1] - v[1], u[2] - v[2]) / 2));

describe.skipIf(!SWEEP)("far-field stride sweep (SWEEP=1)", () => {
  it("prints the table and the selection", () => {
    const S = sets();
    // 1) the local loop IS the shipped loop at the shipped constants
    for (const set of S) for (const ray of set.rays) {
      const loc = localTrace(ray, K_FAR, DL_FAR_MAX);
      const ship = traceRay(ray.s0, ray.a, { rIn: ray.rIn, rOut: ROUT, rObs: ROBS, maxSteps: MAXSTEPS });
      expect(loc.fate).toBe(ship.fate);
      expect(loc.steps).toBe(ship.steps);
      if (loc.fate === "disk") expect(loc.rHit).toBe(ship.rHit);
    }
    const ref = S.map((set) => set.rays.map((ray) => localTrace(ray, K_FAR, DL_FAR_MAX)));
    const score = (K: number, MAXD: number) => {
      let flips = 0, diskBad = 0, skyBad = 0, maxDisk = 0, maxSky = 0, steps = 0, n = 0;
      S.forEach((set, si) => set.rays.forEach((ray, ri) => {
        const r = localTrace(ray, K, MAXD), q = ref[si][ri];
        steps += r.steps; n++;
        if (r.fate !== q.fate) { flips++; return; }
        if (r.fate === "disk") { const e = Math.abs(r.rHit! - q.rHit!); maxDisk = Math.max(maxDisk, e); if (e > 0.02) diskBad++; }
        if (r.fate === "escaped") { const e = angle(r.dir!, q.dir!); maxSky = Math.max(maxSky, e); if (e > HALF_PIXEL) skyBad++; }
      }));
      return { K, MAXD, flips, diskBad, skyBad, maxDisk, maxSky, meanSteps: steps / n };
    };
    // 2) reference convergence
    const conv = score(0.02, 3);
    console.log(`reference check (K=0.02, MAX=3 vs shipped): flips ${conv.flips}, max disk ${conv.maxDisk.toExponential(2)} M, max sky ${conv.maxSky.toExponential(2)} rad (half pixel ${HALF_PIXEL.toExponential(2)})`);
    expect(conv.flips).toBe(0);
    expect(conv.maxDisk).toBeLessThan(0.02);
    expect(conv.maxSky).toBeLessThan(HALF_PIXEL);
    // 3) the table
    const base = score(K_FAR, DL_FAR_MAX).meanSteps;
    const rows = [];
    console.log("K_FAR  DL_MAX  flips  diskBad  skyBad  maxDisk(M)  maxSky(rad)  meanSteps  dSteps%");
    for (const K of [0.04, 0.08, 0.12, 0.2, 0.3]) for (const MAXD of [6, 20, 50, 150]) {
      const r = score(K, MAXD); rows.push(r);
      console.log(`${String(K).padEnd(6)} ${String(MAXD).padEnd(7)} ${String(r.flips).padEnd(6)} ${String(r.diskBad).padEnd(8)} ${String(r.skyBad).padEnd(7)} ${r.maxDisk.toExponential(2).padEnd(11)} ${r.maxSky.toExponential(2).padEnd(12)} ${r.meanSteps.toFixed(1).padEnd(10)} ${(100 * (r.meanSteps / base - 1)).toFixed(1)}`);
    }
    const ok = rows.filter((r) => r.flips === 0 && r.diskBad === 0 && r.skyBad === 0)
      .sort((x, y) => x.meanSteps - y.meanSteps || x.K - y.K || x.MAXD - y.MAXD);
    expect(ok.length).toBeGreaterThan(0); // the shipped row always survives (it is the reference)
    console.log(`SELECTED K_FAR = ${ok[0].K}, DL_FAR_MAX = ${ok[0].MAXD} (mean steps ${ok[0].meanSteps.toFixed(1)} vs ${base.toFixed(1)})`);
  }, 7_200_000);
});
```

- [ ] **Step 3: Run the sweep** (tens of minutes)

```powershell
$env:SWEEP = "1"; npx vitest run tests/sweep-farstride.test.ts; Remove-Item Env:SWEEP
```

Paste the reference-check line, the full table and the `SELECTED` line into the report. If the reference check fails, STOP and report — the table would be meaningless. Confirm `npm test` without SWEEP reports the new file as skipped (82 passed, 3 skipped).

- [ ] **Step 4: Apply the selection in both twins**

Set `K_FAR` / `DL_FAR_MAX` in `trace.ts` and `integrator-shared.wgsl` to the SELECTED values (WGSL: float literals, e.g. `50.0`). Rewrite the `trace.ts` doc comment's second sentence to state the chosen values, the rule, and the mean-steps change, and paste the table as a header comment in `tests/sweep-farstride.test.ts` (as `sweep-axiscap.test.ts` does). If SELECTED is the shipped pair, change nothing and say so in the commit.

- [ ] **Step 5: Gates**

`npm run build`; `npm test` (82 + 3 skipped); `npm run verify:gpu` — `?parity` may move (its "far"/"far-axis" cases compute the stride via the sentinel), `?shadow` may move slightly: record both; `node scripts/probe-axis.mjs` must PASS; `node scripts/probe-scale.mjs` must PASS. Look at the i = 72° and i = 8° probe PNGs: the sky and the disk must look unchanged (the sweep bounded star error to half a pixel).

- [ ] **Step 6: Bench, README, commit**

`npm run bench`; README row `Far-field stride K_FAR=<k>, DL_FAR_MAX=<m>`; update the README gate numbers if parity/shadow moved.

```bash
git add src/physics/trace.ts src/render/integrator-shared.wgsl tests/sweep-farstride.test.ts README.md
git commit -m "Lengthen far-field strides (K_FAR=<k>, DL_FAR_MAX=<m>) from a measured sweep

<reference-check line>
<table>
<SELECTED line>
Rules: 0 flips; disk hits within 0.02 M; escaped sky direction within
half a pixel (1.9e-5 rad); then fewest steps. Bench: <before> -> <after>
ms/frame at 1280x720. Gates: <parity line>, <shadow line>."
git log -1 --format=%B | grep -ci "co-authored"   # 0
```

---

### Task 5: Exact metric derivatives in both twins

**Files:**
- Create: `tests/metric-grad.test.ts`
- Modify: `src/physics/kerr.ts` (add `metricUpperGrad`), `src/physics/geodesic.ts` (analytic `rhs`, `rhsFD`, `rk4` without `h`), `src/physics/trace.ts` (`stepGeodesic` without `h`; drop the `FD_H` import), `src/render/integrator-shared.wgsl` (`MetricGrad`, `gUpGrad`, analytic `rhs`, no `pow`, remove unused `hquad`), `src/test/parity.browser.ts` (drop `GPU_FD_H`, rewrite the barrier comment), `README.md`, `docs/specs/2026-09-23-performance-design.md`

**Interfaces:**
- Consumes: `Metric`, `metricUpper`, `POLE_S2`, `M` from `kerr.ts`.
- Produces: `export interface MetricGrad { g: Metric; dr: Metric; dth: Metric }`, `export function metricUpperGrad(r: number, theta: number, a: number): MetricGrad`; `rhs(s, a)` (analytic), `rhsFD(s, a, h = FD_H)`, `rk4(s, a, dl)`; `stepGeodesic(s, a, dl0, hTol = H_TOL, maxRetry = MAX_RETRY)`; WGSL `struct MetricGrad { g: array<f32,5>, dr: array<f32,5>, dth: array<f32,5> }`, `fn gUpGrad(r: f32, th: f32, a: f32) -> MetricGrad` (indices 0=tt, 1=tphi, 2=rr, 3=thth, 4=phph).

- [ ] **Step 1: Failing tests** — `tests/metric-grad.test.ts`

```ts
import { describe, it, expect } from "vitest";
import { metricUpper, metricUpperGrad, horizonOuter, type Metric } from "../src/physics/kerr";
import { rhs } from "../src/physics/geodesic";

const KEYS: (keyof Metric)[] = ["tt", "tphi", "rr", "thth", "phph"];

/** Richardson-extrapolated central difference: O(h^4) truncation. */
function rich(f: (x: number) => number, x: number, h: number): number {
  const d = (hh: number) => (f(x + hh) - f(x - hh)) / (2 * hh);
  return (4 * d(h / 2) - d(h)) / 3;
}
const close = (got: number, want: number) => Math.abs(got - want) <= 1e-8 * (1 + Math.abs(want));

function states() {
  const out: { r: number; th: number; a: number }[] = [];
  for (const a of [0, 0.5, 0.998]) {
    const rh = horizonOuter(a);
    for (const r of [rh * 1.05, 2.5, 6, 40, 900]) {
      if (r <= rh * 1.01) continue;
      for (const th of [1e-3, 0.3, Math.PI / 2, 2.8, Math.PI - 2e-3]) out.push({ r, th, a });
    }
  }
  return out;
}
const hr = (r: number, a: number) => 1e-3 * (r - horizonOuter(a));
const ht = (th: number) => 1e-3 * Math.min(th, Math.PI - th);

describe("metricUpperGrad", () => {
  it("returns the same metric as metricUpper", () => {
    for (const { r, th, a } of states()) {
      const m = metricUpperGrad(r, th, a).g, ref = metricUpper(r, th, a);
      for (const k of KEYS) expect(close(m[k], ref[k])).toBe(true);
    }
  });
  it("matches a Richardson reference for d/dr and d/dtheta to 1e-8 relative", () => {
    let n = 0;
    for (const { r, th, a } of states()) {
      const d = metricUpperGrad(r, th, a);
      for (const k of KEYS) {
        const wantR = rich((x) => metricUpper(x, th, a)[k], r, hr(r, a));
        const wantT = rich((x) => metricUpper(r, x, a)[k], th, ht(th));
        if (!close(d.dr[k], wantR)) throw new Error(`d${k}/dr at r=${r} th=${th} a=${a}: ${d.dr[k]} vs ${wantR}`);
        if (!close(d.dth[k], wantT)) throw new Error(`d${k}/dth at r=${r} th=${th} a=${a}: ${d.dth[k]} vs ${wantT}`);
        n++;
      }
    }
    expect(n).toBeGreaterThan(300);
  });
});

describe("analytic rhs", () => {
  it("momentum forces equal -1/2 dH/dx from a Richardson reference", () => {
    const hq = (r: number, th: number, a: number, p: number[]) => {
      const g = metricUpper(r, th, a);
      return g.tt * p[0] ** 2 + 2 * g.tphi * p[0] * p[3] + g.rr * p[1] ** 2 + g.thth * p[2] ** 2 + g.phph * p[3] ** 2;
    };
    const p = [1, -0.7, 2.3, -3.1];
    for (const { r, th, a } of states()) {
      const d = rhs(new Float64Array([0, r, th, 0, ...p]), a);
      expect(close(d[5], -0.5 * rich((x) => hq(x, th, a, p), r, hr(r, a)))).toBe(true);
      expect(close(d[6], -0.5 * rich((x) => hq(r, x, a, p), th, ht(th)))).toBe(true);
      expect(d[4]).toBe(0);
      expect(d[7]).toBe(0);
    }
  });
});
```

Run `npx vitest run tests/metric-grad.test.ts` → FAIL (`metricUpperGrad` not exported).

- [ ] **Step 2: `metricUpperGrad` in `src/physics/kerr.ts`** (append)

```ts
export interface MetricGrad { g: Metric; dr: Metric; dth: Metric; }

/** Inverse metric and its exact r- and theta-derivatives (quotient rule, f' = (N' - f D') / D).
 *  Replaces finite differences in the integrator: one evaluation instead of four extra. Semantics
 *  match metricUpper exactly, including the POLE_S2 floor, whose theta-derivative is 0 below the
 *  floor (the floored denominator is constant there). Twin: gUpGrad in integrator-shared.wgsl. */
export function metricUpperGrad(r: number, theta: number, a: number): MetricGrad {
  const s = Math.sin(theta), c = Math.cos(theta), s2 = s * s, sc = s * c;
  const floored = s2 < POLE_S2, s2d = floored ? POLE_S2 : s2, ds2d = floored ? 0 : 2 * sc;
  const a2 = a * a, r2 = r * r, ra = r2 + a2;
  const Sig = r2 + a2 * c * c, dSigR = 2 * r, dSigT = -2 * a2 * sc;
  const D = r2 - 2 * M * r + a2, dDR = 2 * r - 2 * M;
  const A = ra * ra - a2 * D * s2, dAR = 4 * r * ra - a2 * dDR * s2, dAT = -2 * a2 * D * sc;
  const SD = Sig * D, dSDR = dSigR * D + Sig * dDR, dSDT = dSigT * D;
  const tt = -A / SD, tphi = -2 * M * a * r / SD, rr = D / Sig, thth = 1 / Sig;
  const N = D - a2 * s2, P = SD * s2d, phph = N / P;
  const dPR = dSDR * s2d, dPT = dSDT * s2d + SD * ds2d;
  return {
    g: { tt, tphi, rr, thth, phph },
    dr: { tt: (-dAR - tt * dSDR) / SD, tphi: (-2 * M * a - tphi * dSDR) / SD, rr: (dDR - rr * dSigR) / Sig,
          thth: -thth * dSigR / Sig, phph: (dDR - phph * dPR) / P },
    dth: { tt: (-dAT - tt * dSDT) / SD, tphi: -tphi * dSDT / SD, rr: -rr * dSigT / Sig,
           thth: -thth * dSigT / Sig, phph: (-2 * a2 * sc - phph * dPT) / P },
  };
}
```

- [ ] **Step 3: Analytic `rhs` in `src/physics/geodesic.ts`**

Import `metricUpperGrad` alongside `metricUpper`. Rename the existing `rhs` to `rhsFD` (unchanged body, keep the `h = FD_H` parameter) and rewrite the `FD_H` doc comment to: "Central-difference half-step of rhsFD, the finite-difference reference kept for tests; the integrator uses the analytic rhs." Add:

```ts
/** Hamilton's equations with exact forces: dp_mu/dl = -1/2 d(g^{ab} p_a p_b)/dx^mu, from
 *  metricUpperGrad. p_t and p_phi are Killing (their forces are exactly 0). */
export function rhs(s: Float64Array, a: number): Float64Array {
  const [, r, th, , pt, pr, pth, pphi] = s;
  const { g, dr: gr, dth: gt } = metricUpperGrad(r, th, a);
  const dQdr = gr.tt * pt * pt + 2 * gr.tphi * pt * pphi + gr.rr * pr * pr + gr.thth * pth * pth + gr.phph * pphi * pphi;
  const dQdth = gt.tt * pt * pt + 2 * gt.tphi * pt * pphi + gt.rr * pr * pr + gt.thth * pth * pth + gt.phph * pphi * pphi;
  return new Float64Array([g.tt * pt + g.tphi * pphi, g.rr * pr, g.thth * pth, g.tphi * pt + g.phph * pphi,
    0, -0.5 * dQdr, -0.5 * dQdth, 0]);
}
```

Change `rk4` to `export function rk4(s: Float64Array, a: number, dl: number): Float64Array` calling `rhs(…, a)` (no `h`). In `src/physics/trace.ts`: remove `FD_H` from the import, drop the `h` parameter from `stepGeodesic` (signature `(s, a, dl0, hTol = H_TOL, maxRetry = MAX_RETRY)`) and its two `rk4(s, a, dl, h)` calls become `rk4(s, a, dl)`.

Run `npx vitest run tests/metric-grad.test.ts` → PASS. Run `npm test` → every pre-existing test passes unchanged (85 passed: 82 + 3 new, 3 skipped — if any existing physics test fails, STOP and investigate; do not edit it).

- [ ] **Step 4: GPU twin in `src/render/integrator-shared.wgsl`**

- `bigA_`: `return (r*r+a*a)*(r*r+a*a) - a*a*delta_(r,a)*s*s;` (no `pow`).
- `gUp`: compute `sin`/`cos` once:
  ```wgsl
  fn gUp(r: f32, th: f32, a: f32) -> array<f32,5> {
    let sn = sin(th); let cs = cos(th); let s2 = sn*sn; let s2d = max(s2, POLE_S2);
    let a2 = a*a; let ra = r*r + a2;
    let Sig = r*r + a2*cs*cs; let d = r*r - 2.0*r + a2; let A = ra*ra - a2*d*s2;
    return array<f32,5>( -A/(Sig*d), -2.0*a*r/(Sig*d), d/Sig, 1.0/Sig, (d - a2*s2)/(Sig*d*s2d) );
  }
  ```
- Delete `fn hquad` (no remaining users after `rhs` changes; confirm with `grep -n "hquad(" src/render/*.wgsl`).
- Add, after `gLow`:
  ```wgsl
  // Inverse metric and its exact r- and theta-derivatives. Twin of metricUpperGrad in kerr.ts;
  // indices 0=tt, 1=tphi, 2=rr, 3=thth, 4=phph. The POLE_S2 floor's theta-derivative is 0 below it.
  struct MetricGrad { g: array<f32,5>, dr: array<f32,5>, dth: array<f32,5> };
  fn gUpGrad(r: f32, th: f32, a: f32) -> MetricGrad {
    let sn = sin(th); let cs = cos(th); let s2 = sn*sn; let sc = sn*cs;
    let floored = s2 < POLE_S2; let s2d = select(s2, POLE_S2, floored); let ds2d = select(2.0*sc, 0.0, floored);
    let a2 = a*a; let r2 = r*r; let ra = r2 + a2;
    let Sig = r2 + a2*cs*cs; let dSigR = 2.0*r; let dSigT = -2.0*a2*sc;
    let D = r2 - 2.0*r + a2; let dDR = 2.0*r - 2.0;
    let A = ra*ra - a2*D*s2; let dAR = 4.0*r*ra - a2*dDR*s2; let dAT = -2.0*a2*D*sc;
    let SD = Sig*D; let dSDR = dSigR*D + Sig*dDR; let dSDT = dSigT*D;
    let tt = -A/SD; let tphi = -2.0*a*r/SD; let rr = D/Sig; let thth = 1.0/Sig;
    let P = SD*s2d; let phph = (D - a2*s2)/P;
    let dPR = dSDR*s2d; let dPT = dSDT*s2d + SD*ds2d;
    return MetricGrad(
      array<f32,5>(tt, tphi, rr, thth, phph),
      array<f32,5>((-dAR - tt*dSDR)/SD, (-2.0*a - tphi*dSDR)/SD, (dDR - rr*dSigR)/Sig, -thth*dSigR/Sig, (dDR - phph*dPR)/P),
      array<f32,5>((-dAT - tt*dSDT)/SD, -tphi*dSDT/SD, -rr*dSigT/Sig, -thth*dSigT/Sig, (-2.0*a2*sc - phph*dPT)/P));
  }
  ```
- Replace `rhs`:
  ```wgsl
  // Hamilton's equations with exact forces (twin of rhs in geodesic.ts): one metric evaluation per
  // call instead of five finite-difference ones.
  fn rhs(s: State, a: f32) -> State {
    let m = gUpGrad(s.x.y, s.x.z, a); let g = m.g; let p = s.p;
    let dx = vec4<f32>(g[0]*p.x + g[1]*p.w, g[2]*p.y, g[3]*p.z, g[1]*p.x + g[4]*p.w);
    let dQdr  = m.dr[0]*p.x*p.x + 2.0*m.dr[1]*p.x*p.w + m.dr[2]*p.y*p.y + m.dr[3]*p.z*p.z + m.dr[4]*p.w*p.w;
    let dQdth = m.dth[0]*p.x*p.x + 2.0*m.dth[1]*p.x*p.w + m.dth[2]*p.y*p.y + m.dth[3]*p.z*p.z + m.dth[4]*p.w*p.w;
    return State(dx, vec4<f32>(0.0, -0.5*dQdr, -0.5*dQdth, 0.0));
  }
  ```
- Update the fragment's comments that still describe finite-difference forces (the monitor comment above `H_TOL`, the `H_TOL_FAR` comment's "f32 finite-difference force is pure noise" rationale — keep the exemption, restate the rationale as historical and note that Task 5 re-checks it below).

- [ ] **Step 5: Parity comparator** — `src/test/parity.browser.ts`

Delete `const GPU_FD_H = 1e-4;`, call `stepGeodesic(s32, c.a, dl0, Math.fround(c.hTol), MAX_RETRY)`, and rewrite the barrier-case comment (around line 271) to: both twins now use exact derivatives, so the comparator steps the same equations as the GPU with no `h`; record the barrier case's measured state error from Step 6.

- [ ] **Step 6: GPU gates and re-baseline**

`npm run build`; `npm test` (85 + 3 skipped); `npm run verify:gpu` — record the new `?parity` maxRelErr (expected to tighten; if it **loosens**, find out why with a scratch `_diag.mjs` printing per-case errors before recording anything) and the `?shadow` numbers; `node scripts/probe-axis.mjs` and `node scripts/probe-scale.mjs` must PASS. View `axis-i8.png`: report whether the ~6 lit pixels on the axis column inside the shadow (previously attributed to the FD half-step) are gone.

- [ ] **Step 7: Re-run the three sweeps and apply their rules**

```powershell
$env:SWEEP = "1"; npx vitest run tests/sweep-htol.test.ts; npx vitest run tests/sweep-axiscap.test.ts; npx vitest run tests/sweep-farstride.test.ts; Remove-Item Env:SWEEP
```

For each, apply the selection rule stated in that file (and in `trace.ts` comments) to the new table. If a rule now selects a different constant, change it in **both** twins, update the file's header table and the `trace.ts` comment, and re-run Step 6's gates. If it selects the same constant, record "confirmed under exact forces" in the header comment with the date. Also re-check the far-field exemption's premise: in the scratch `_diag.mjs`, print the GPU's `|ΔH|/scale` for the parity "far" case at `hTol = H_TOL` — if it is now below `H_TOL` (the FD noise that motivated `H_TOL_FAR` is gone), report it as a follow-up; do NOT change the exemption in this task.

- [ ] **Step 8: Docs**

README: bench row `Exact metric derivatives`; update the gate numbers (tests, parity, shadow); rewrite the limitation that attributes the in-shadow axis pixels to the FD half-step to match Step 6's observation; remove statements that the CPU and GPU use different FD steps. Spec: set **Status** to `implemented on perf/smooth-first` and add a 3–5 line "Outcome" section: bench before/after for each task, final gates, constants confirmed or changed.

- [ ] **Step 9: Commit**

```bash
git add src/physics/kerr.ts src/physics/geodesic.ts src/physics/trace.ts src/render/integrator-shared.wgsl src/test/parity.browser.ts tests/metric-grad.test.ts tests/sweep-*.test.ts README.md docs/specs/2026-09-23-performance-design.md
git commit -m "Integrate with exact metric derivatives instead of finite differences

metricUpperGrad / gUpGrad give the inverse metric and its r- and theta-
derivatives in one evaluation (was 1 + 4 per rhs), matching a Richardson
reference to 1e-8 including near-horizon and near-axis states and the
POLE_S2 floor. CPU and GPU now integrate identical equations; the parity
comparator's h = 1e-4 is gone. rhsFD stays as the test reference.
Gates: <parity>, <shadow>. Sweeps: <confirmed/changed per file>.
Bench: <before> -> <after> ms/frame at 1280x720."
git log -1 --format=%B | grep -ci "co-authored"   # 0
```

Do not merge. Report the full bench progression (baseline → each task) and the final gates.
