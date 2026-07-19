# Lensed Sky Background Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the procedural starfield behind the black hole with a real Milky-Way panorama, sampled along each escaped ray's already-lensed direction, so the warping (Einstein ring) reads photographically.

**Architecture:** A committed equirectangular JPG is decoded in the browser, uploaded as an sRGB GPU texture with a CPU-generated mip chain, and sampled in the raytrace compute shader with explicit LOD (compute has no fragment derivatives). A new `skyStrength` uniform + "Sky" slider crossfades from the procedural starfield (which stays as the offline fallback). No geodesic/disk/jet physics changes.

**Tech Stack:** TypeScript, WebGPU (WGSL compute), Vite, Vitest, playwright-core (asset tooling + headless GPU verify).

## Global Constraints

- No `Co-Authored-By` trailer in git commits (user rule).
- Target hardware: NVIDIA RTX 3050 Laptop (4 GB); keep resident memory modest (sky texture ≤ 4096×2048 RGBA + mips ≈ 43 MB).
- **Off-invariant:** with `skyStrength = 0` (or the asset failed to load), the escaped-ray output MUST be bit-for-bit identical to the current procedural void. `?parity` and `?shadow` MUST stay green.
- Append-only uniform layout: `skyStrength` occupies the existing padding slot `f[23]`; `UNIFORM_SIZE` stays **96 bytes**.
- Sky sampling lives in the render layer, not `src/physics/` (physics stays pure/DOM-free).
- Asset license: ESO/S. Brunier, CC BY 4.0 — attribution required in README + `public/sky/CREDIT.txt`.
- Match existing code style (terse, comment the *why*). Commit after every task.

---

### Task 1: Equirectangular sky-map helper (pure TS)

**Files:**
- Create: `src/render/skymap.ts`
- Test: `tests/skymap.test.ts`

**Interfaces:**
- Produces:
  - `SKY_TEXW: number` (= 4096) — panorama width, used for LOD.
  - `dirToEquirectUV(d: [number, number, number]): [number, number]` — unit direction → `[u, v]` in `[0,1]`.
  - `tiltDir(d: [number, number, number]): [number, number, number]` — applies the fixed galactic tilt `R_SKY`. WGSL twin `tiltDir` in `raytrace.wgsl` must match these coefficients.

- [ ] **Step 1: Write the failing test**

```ts
// tests/skymap.test.ts
import { describe, it, expect } from "vitest";
import { dirToEquirectUV, tiltDir, SKY_TEXW } from "../src/render/skymap";

describe("equirectangular sky map", () => {
  it("maps cardinal directions to the expected UV", () => {
    expect(SKY_TEXW).toBe(4096);
    const [u0, v0] = dirToEquirectUV([1, 0, 0]);
    expect(u0).toBeCloseTo(0.5); expect(v0).toBeCloseTo(0.5);      // +X → centre
    expect(dirToEquirectUV([0, 1, 0])[1]).toBeCloseTo(0.0);        // +Y → north pole (v=0)
    expect(dirToEquirectUV([0, -1, 0])[1]).toBeCloseTo(1.0);       // -Y → south pole (v=1)
    expect(dirToEquirectUV([0, 0, 1])[0]).toBeCloseTo(0.75);       // +Z
    expect(dirToEquirectUV([0, 0, -1])[0]).toBeCloseTo(0.25);      // -Z
  });

  it("wraps continuously across the ±Z seam behind -X", () => {
    const uHi = dirToEquirectUV([-1, 0, 1e-4])[0];   // just above the seam → u ≈ 1
    const uLo = dirToEquirectUV([-1, 0, -1e-4])[0];  // just below the seam → u ≈ 0
    expect(uHi).toBeGreaterThan(0.99);
    expect(uLo).toBeLessThan(0.01);
  });

  it("tiltDir is an orthonormal rotation (preserves length)", () => {
    const len = (d: number[]) => Math.hypot(d[0], d[1], d[2]);
    expect(len(tiltDir([1, 0, 0]))).toBeCloseTo(1);
    expect(len(tiltDir([0.3, -0.5, 0.81]))).toBeCloseTo(len([0.3, -0.5, 0.81]));
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- skymap`
Expected: FAIL — cannot resolve `../src/render/skymap`.

- [ ] **Step 3: Write minimal implementation**

```ts
// src/render/skymap.ts
// Equirectangular (lat-long) sampling of the sky panorama, plus a fixed galactic tilt so the
// Milky-Way band sits off-axis from the accretion disk. The WGSL twins (skySample / tiltDir in
// raytrace.wgsl) MUST mirror these formulas and coefficients.

export const SKY_TEXW = 4096;

// R_SKY = Rz(30°)·Rx(60°), applied as R·dir (row-major). Orthonormal (det = 1).
const R_SKY = [
  0.866025, -0.25,      0.433013,
  0.5,       0.433013, -0.75,
  0.0,       0.866025,  0.5,
] as const;

export function tiltDir(d: [number, number, number]): [number, number, number] {
  const m = R_SKY;
  return [
    m[0] * d[0] + m[1] * d[1] + m[2] * d[2],
    m[3] * d[0] + m[4] * d[1] + m[5] * d[2],
    m[6] * d[0] + m[7] * d[1] + m[8] * d[2],
  ];
}

export function dirToEquirectUV(d: [number, number, number]): [number, number] {
  const u = Math.atan2(d[2], d[0]) * (0.5 / Math.PI) + 0.5;
  const v = Math.acos(Math.max(-1, Math.min(1, d[1]))) * (1 / Math.PI);
  return [u, v];
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- skymap`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/render/skymap.ts tests/skymap.test.ts
git commit -m "feat(render): equirectangular sky-map helper + galactic tilt (unit-tested)"
```

---

### Task 2: Fetch + downscale the panorama asset

**Files:**
- Create: `scripts/fetch-sky.mjs`
- Create: `public/sky/milkyway-4k.jpg` (produced by the script)
- Create: `public/sky/CREDIT.txt`

**Interfaces:**
- Produces: a 4096×2048 equirectangular JPG served by Vite at `/sky/milkyway-4k.jpg`.

**Note on approach:** ESO's CDN does not send CORS headers, so we fetch the bytes in Node (not in-page) and hand them to the browser as a `data:` URL (same-origin, untainted) for canvas downscaling. This reuses the existing `playwright-core` dev dependency — no new packages.

- [ ] **Step 1: Write the fetch/downscale script**

```js
// scripts/fetch-sky.mjs
// Downloads the ESO/S. Brunier Milky-Way panorama (CC BY 4.0) and downscales it to a
// power-of-two 4096×2048 equirectangular JPG committed under public/sky/.
import { chromium } from "playwright-core";
import { writeFileSync, mkdirSync } from "fs";

const SRC = "https://cdn.eso.org/images/publicationjpg/eso0932a.jpg";
const OUT = "public/sky/milkyway-4k.jpg";
const W = 4096, H = 2048;

const resp = await fetch(SRC);
if (!resp.ok) throw new Error(`fetch failed: ${resp.status}`);
const srcB64 = Buffer.from(await resp.arrayBuffer()).toString("base64");

const browser = await chromium.launch({ channel: "chrome", headless: true,
  args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"] });
const page = await browser.newPage();
const { outB64, w, h } = await page.evaluate(async ({ b64, W, H }) => {
  const img = new Image();
  img.src = "data:image/jpeg;base64," + b64;
  await img.decode();
  const c = document.createElement("canvas"); c.width = W; c.height = H;
  const g = c.getContext("2d"); g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
  g.drawImage(img, 0, 0, W, H);
  return { outB64: c.toDataURL("image/jpeg", 0.85).split(",")[1], w: img.naturalWidth, h: img.naturalHeight };
}, { b64: srcB64, W, H });
await browser.close();

mkdirSync("public/sky", { recursive: true });
writeFileSync(OUT, Buffer.from(outB64, "base64"));
console.log(`source ${w}×${h} → ${OUT} (${W}×${H}), ${(Buffer.from(outB64, "base64").length / 1e6).toFixed(2)} MB`);
```

- [ ] **Step 2: Write the attribution file**

```
// public/sky/CREDIT.txt
milkyway-4k.jpg — "The Milky Way panorama" (eso0932a)
Credit: ESO/S. Brunier
License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
Source: https://www.eso.org/public/images/eso0932a/
Downscaled to 4096×2048 equirectangular via scripts/fetch-sky.mjs.
```

- [ ] **Step 3: Run the script to produce the asset**

Run: `node scripts/fetch-sky.mjs`
Expected: prints `source 9104×4552 → public/sky/milkyway-4k.jpg (4096×2048), ~2-3 MB` (source dims may differ; output must be 4096×2048).

- [ ] **Step 4: Verify the asset exists and is 2:1**

Run: `node -e "const b=require('fs').readFileSync('public/sky/milkyway-4k.jpg');console.log('bytes',b.length)"`
Expected: a non-trivial byte count (> 500 000). Visually confirm the file opens as a starry panorama.

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch-sky.mjs public/sky/milkyway-4k.jpg public/sky/CREDIT.txt
git commit -m "feat(asset): fetch+downscale ESO Milky-Way panorama (CC BY 4.0)"
```

---

### Task 3: Add the `skyStrength` uniform

**Files:**
- Modify: `src/render/uniforms.ts`
- Modify: `tests/uniforms.test.ts`
- Modify: `src/main.ts` (uniform literal only — a hardcoded `0` so it compiles; made live in Task 5)
- Modify: `src/test/shadow.browser.ts` (uniform literal — `skyStrength: 0`, forcing sky off in the shadow route)

**Interfaces:**
- Produces: `UniformValues.skyStrength: number`, packed at byte offset **92** (float index 23). `UNIFORM_FLOATS` = 20, `UNIFORM_SIZE` unchanged at 96.

- [ ] **Step 1: Update the failing test**

In `tests/uniforms.test.ts`, add `skyStrength` to the `u` literal and assert its offset. Change the object (line 11 area) to include it and add the assertion after the `jetKnots` line:

```ts
      jetStrength: 1.0, jetGamma: 5.0, jetLength: 60.0, jetKnots: 0.7,
      skyStrength: 0.6,
    };
    const dv = new DataView(packUniforms(u));
    // ...existing assertions...
    expect(dv.getFloat32(88, true)).toBeCloseTo(0.7);   // jetKnots (index 22)
    expect(dv.getFloat32(92, true)).toBeCloseTo(0.6);   // skyStrength (index 23)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- uniforms`
Expected: FAIL — TS error (`skyStrength` missing on `UniformValues`) / offset 92 reads 0.

- [ ] **Step 3: Implement the uniform field**

In `src/render/uniforms.ts`:
- Update the header comment `-> 19 floats` to `-> 20 floats` and note `+ skyStrength (1) -> 20`.
- Add `skyStrength: number;` to the `UniformValues` interface (after `jetKnots`).
- Change `export const UNIFORM_FLOATS = 19` to `20`.
- In `packUniforms`, after `f[22] = u.jetKnots;` add:

```ts
  f[23] = u.skyStrength;
```

- [ ] **Step 4: Keep call sites compiling**

In `src/main.ts`, in the `UniformValues` literal inside `loop()` (after `jetLength: state.jetLength, jetKnots: state.jetKnots,`) add:

```ts
        skyStrength: 0,
```

In `src/test/shadow.browser.ts`, in the `u` literal (after `jetStrength: 0, jetGamma: 5, jetLength: 60, jetKnots: 0.7 }`) add `skyStrength: 0` — the shadow route must render with the sky off:

```ts
    jetStrength: 0, jetGamma: 5, jetLength: 60, jetKnots: 0.7, skyStrength: 0 };
```

- [ ] **Step 5: Run tests + typecheck**

Run: `npm test -- uniforms && npm run build`
Expected: uniforms test PASS; build succeeds (no TS errors).

- [ ] **Step 6: Commit**

```bash
git add src/render/uniforms.ts tests/uniforms.test.ts src/main.ts src/test/shadow.browser.ts
git commit -m "feat(render): add skyStrength uniform (f[23], size stays 96)"
```

---

### Task 4: GPU sky texture + shader bindings (dormant)

**Files:**
- Modify: `src/render/gpu.ts` (texture/sampler fields, placeholder in `init`, `uploadSky`, bind group)
- Modify: `src/render/raytrace.wgsl` (Uniforms field, texture/sampler bindings, `tiltDir`/`skySample`, escaped-ray branch)

**Interfaces:**
- Consumes: `skyStrength` uniform (Task 3).
- Produces: `Renderer.uploadSky(bitmap: ImageBitmap): void` — uploads a 4096×2048 sRGB texture with a full mip chain. Caller must then call `rebind()`.

**Why dormant:** `main.ts` still feeds `skyStrength: 0` (Task 3), so this task adds all plumbing with **zero visual change** and the binding stays statically used (auto pipeline layout keeps it because `skySample` is always called). Task 5 turns it on.

- [ ] **Step 1: Add the Uniforms field + bindings + sampler in the shader**

In `src/render/raytrace.wgsl`, add `skyStrength: f32,` to the `Uniforms` struct (after `jetKnots: f32,`). After the existing `@binding(4)` line add:

```wgsl
@group(0) @binding(5) var skyTex: texture_2d<f32>;
@group(0) @binding(6) var skySamp: sampler;
```

Immediately after the `starfield` function, add:

```wgsl
// --- Baked sky panorama (equirectangular; twin of src/render/skymap.ts) ------------------------
const SKY_TEXW = 4096.0;
fn tiltDir(d: vec3<f32>) -> vec3<f32> {   // R_SKY = Rz(30°)·Rx(60°), must match skymap.ts
  return vec3<f32>(
    0.866025 * d.x - 0.25 * d.y + 0.433013 * d.z,
    0.5 * d.x + 0.433013 * d.y - 0.75 * d.z,
    0.866025 * d.y + 0.5 * d.z);
}
fn skySample(dir: vec3<f32>) -> vec3<f32> {
  let d = tiltDir(dir);
  let u = atan2(d.z, d.x) * (0.5 / PI) + 0.5;
  let v = acos(clamp(d.y, -1.0, 1.0)) * (1.0 / PI);
  // Compute shaders have no implicit derivatives, so pick LOD analytically from the far-field
  // angular footprint of one pixel. Mip chain + temporal AA absorb residual minification aliasing.
  let anglePerPixel = 2.0 * U.fovScale / U.res.y / U.rObs;
  let lod = max(0.0, log2(SKY_TEXW * anglePerPixel / (2.0 * PI)));
  return textureSampleLevel(skyTex, skySamp, vec2<f32>(u, v), lod).rgb;
}
```

Replace the escaped-ray body (the `color = starfield(dir);` line inside the `if (s.x.y > r0 * 1.2)` block) with:

```wgsl
      // Baked panorama crossfaded over the procedural starfield by skyStrength (0 => unchanged).
      let mixT = clamp(U.skyStrength, 0.0, 1.0);
      color = mix(starfield(dir), skySample(dir) * U.skyStrength, mixT);
```

- [ ] **Step 2: Add texture/sampler resources in the renderer**

In `src/render/gpu.ts`:
- Add fields near the other buffers: `skyTex!: GPUTexture; skySampler!: GPUSampler;`
- In `init`, after the `spotBuf` creation, add a 1×1 placeholder + sampler:

```ts
    this.skyTex = this.device.createTexture({ size: [1, 1], format: "rgba8unorm-srgb",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    this.skySampler = this.device.createSampler({ magFilter: "linear", minFilter: "linear",
      mipmapFilter: "linear", addressModeU: "repeat", addressModeV: "clamp-to-edge" });
```

- In `rebind`, add two entries to the `computeBind` entries array (after the `binding: 4` entry):

```ts
      { binding: 5, resource: this.skyTex.createView() },
      { binding: 6, resource: this.skySampler }] });
```
(Move the closing `] });` so these are inside the `computeBind` entries.)

- [ ] **Step 3: Implement `uploadSky` with a CPU-generated mip chain**

Add this method to `Renderer` (after `uploadHotSpots`):

```ts
  /** Upload the equirectangular sky panorama as an sRGB texture with a full mip chain. Mips are
   *  generated on the CPU via canvas downscales (WebGPU has no built-in generateMipmaps), which
   *  keeps the lensed/minified sky from aliasing. Caller must call rebind() afterward. */
  uploadSky(bitmap: ImageBitmap) {
    const W = 4096, H = 2048;
    const mips = Math.floor(Math.log2(Math.max(W, H))) + 1;
    this.skyTex = this.device.createTexture({ size: [W, H], mipLevelCount: mips, format: "rgba8unorm-srgb",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    const canvas = document.createElement("canvas");
    const g = canvas.getContext("2d")!;
    for (let lvl = 0; lvl < mips; lvl++) {
      const lw = Math.max(1, W >> lvl), lh = Math.max(1, H >> lvl);
      canvas.width = lw; canvas.height = lh;
      g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
      g.drawImage(bitmap, 0, 0, lw, lh);
      this.device.queue.copyExternalImageToTexture({ source: canvas }, { texture: this.skyTex, mipLevel: lvl }, [lw, lh]);
    }
  }
```

- [ ] **Step 4: Verify no visual change + validation routes still pass**

Run (dev server on 5173): `BASE=http://localhost:5173 npm run verify:gpu`
Expected: `?parity` PASS (maxRelErr ~9.69e-7), `?shadow` PASS. The `render.png` looks identical to before (skyStrength still 0 everywhere).

- [ ] **Step 5: Commit**

```bash
git add src/render/gpu.ts src/render/raytrace.wgsl
git commit -m "feat(render): sky texture + sampler bindings + uploadSky mip chain (dormant at skyStrength=0)"
```

---

### Task 5: Turn the sky on — load path + UI

**Files:**
- Modify: `index.html` (Sky slider)
- Modify: `src/main.ts` (state field, slider wiring, uniform feed, fetch → uploadSky)

**Interfaces:**
- Consumes: `Renderer.uploadSky` (Task 4), `skyStrength` uniform (Task 3), asset at `/sky/milkyway-4k.jpg` (Task 2).

- [ ] **Step 1: Add the Sky slider to the panel**

In `index.html`, after the `Jet knots` control block (the `<div class="ctrl">` ending `id="jk"`), and before the `<button id="playpause"...>`, add:

```html
    <div class="ctrl">
      <div class="row"><label>Sky</label><span class="val"><b id="skyv">0.6</b>×</span></div>
      <input id="sky" type="range" min="0" max="2" step="0.05" value="0.6">
    </div>
```

- [ ] **Step 2: Add state + slider listener + uniform feed in main.ts**

In `src/main.ts`:
- Add `skyStrength: 0.6` to the `state` object literal (after `jetKnots: 0.7`).
- Change the uniform literal `skyStrength: 0,` (added in Task 3) to `skyStrength: state.skyStrength,`.
- After the jet slider listeners (the `jk.addEventListener` block), add:

```ts
  const sky = $("sky") as HTMLInputElement, skyv = $("skyv");
  sky.addEventListener("input", () => { state.skyStrength = +sky.value; skyv.textContent = state.skyStrength.toFixed(2); reset(); });
```

- [ ] **Step 3: Add the fetch → uploadSky load path**

In `src/main.ts`, after `r.uploadHotSpots(packSpots(state.flareScale));` (near the end of setup, before the render loop), add:

```ts
  // Load the baked sky panorama asynchronously; on failure keep the procedural starfield fallback.
  fetch("/sky/milkyway-4k.jpg").then((res) => res.blob()).then(createImageBitmap)
    .then((bmp) => { r.uploadSky(bmp); r.rebind(); reset(); })
    .catch(() => { /* offline / decode error — procedural starfield stays */ });
```

- [ ] **Step 4: Visual verification at three inclinations**

Create a throwaway capture script `scripts/_skyshot.mjs` (delete after):

```js
import { chromium } from "playwright-core";
const b = await chromium.launch({ channel: "chrome", headless: true,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist"] });
const p = await b.newPage({ viewport: { width: 1000, height: 680 } });
await p.goto("http://localhost:5173/", { waitUntil: "load" });
await p.waitForTimeout(1200);
const set = (id, v) => p.evaluate(({ id, v }) => { const e = document.getElementById(id); e.value = String(v); e.dispatchEvent(new Event("input", { bubbles: true })); }, { id, v });
for (const incl of [17, 45, 85]) { await set("incl", incl); await p.waitForTimeout(4000); await p.screenshot({ path: `sky-i${incl}.png` }); console.log("sky-i" + incl); }
await b.close();
```

Run: `node scripts/_skyshot.mjs`
Expected: `sky-i17.png` shows the Milky-Way band lensing into an Einstein ring around the shadow; `sky-i45`/`sky-i85` show the panorama warped around the hole; no pole pinching or seam through the disk. Then `rm scripts/_skyshot.mjs sky-i*.png`.

- [ ] **Step 5: Confirm the off-invariant + validation routes**

Set the Sky slider to 0 in the browser (or confirm via capture) → the render matches the pre-sky procedural void.
Run: `BASE=http://localhost:5173 npm run verify:gpu`
Expected: `?parity` and `?shadow` PASS (shadow route forces `skyStrength: 0`).

- [ ] **Step 6: Commit**

```bash
git add index.html src/main.ts
git commit -m "feat(ui): Sky slider + async panorama load (procedural fallback on failure)"
```

---

### Task 6: Automated sky check + docs

**Files:**
- Modify: `scripts/verify-gpu.mjs` (assert the sky asset is served + no errors with sky on)
- Modify: `README.md` (attribution + status)

**Interfaces:**
- Consumes: the full feature (Tasks 1–5).

- [ ] **Step 1: Assert the sky asset loads in verify-gpu**

In `scripts/verify-gpu.mjs`, before `await browser.close();`, add a check that the panorama request succeeds during the interactive render:

```js
// Sky panorama should be fetched and served (200) during the interactive render.
const skyResp = await page.request.get(BASE + "/sky/milkyway-4k.jpg");
const skyOk = skyResp.ok();
console.log(`${skyOk ? "✓ PASS" : "✗ FAIL"}  /sky/milkyway-4k.jpg  (${skyResp.status()})`);
if (!skyOk) failed = true;
```

- [ ] **Step 2: Run the full verification**

Run: `BASE=http://localhost:5173 npm run verify:gpu`
Expected: three ✓ PASS lines (parity, shadow, sky asset), `render.png` saved, no page errors.

- [ ] **Step 3: Update the README**

In `README.md`:
- In **Features**, change the lensed-starfield "out of scope" note. Under **Status**, after the Visual polish paragraph, add:

```markdown
**Lensed sky background (shipped):** the procedural starfield is replaced by a real Milky-Way panorama (ESO/S. Brunier, CC BY 4.0) sampled along each escaped ray's gravitationally-bent direction, so the background warps into a clear Einstein ring around the shadow. A "Sky" slider crossfades its brightness; at 0 (or if the asset fails to load) the render falls back to the original procedural void, so `?parity` and `?shadow` are unchanged.
```
- Change the final "Out of scope" line to drop "lensed starfield background" (now shipped), keeping Tier 3 (full GRMHD) out of scope.
- Add an **Attribution** note near the bottom: `Milky-Way panorama: ESO/S. Brunier, CC BY 4.0 (eso0932a).`

- [ ] **Step 4: Commit**

```bash
git add scripts/verify-gpu.mjs README.md
git commit -m "test+docs: verify sky asset serves; document lensed sky background"
```

---

## Notes for the implementer

- **Auto pipeline layout:** `computePipe` uses `layout: "auto"`, which drops bindings not statically used by `main`. `skySample` is always called in the escaped-ray branch, so bindings 5/6 survive even at `skyStrength = 0`. Do not guard `skySample` behind an `if (U.skyStrength > 0)` — that could let the compiler strip the binding and break `rebind()`.
- **sRGB:** the texture is `rgba8unorm-srgb`, so `textureSampleLevel` returns **linear** radiance — do not de-gamma in the shader.
- **`copyExternalImageToTexture`** requires the destination to carry `RENDER_ATTACHMENT` usage; keep it on `skyTex`.
- **Dev server:** if not already running, `npm run dev` (port 5173) before any `verify:gpu` / capture step; pass `BASE=http://localhost:5173`.
- **Do not push to origin** unless the user asks (all prior tiers were merged locally only).
