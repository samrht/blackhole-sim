# Shadow-Classifier Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `?parity` exercise the same bytes the renderer compiles for the Kerr critical-curve classifier, so a future edit cannot silently desync `src/physics/shadow.ts` from the WGSL twin and quietly move the rendered shadow edge.

**Architecture:** Extract the self-contained classifier block from `raytrace.wgsl` into `src/render/shadow-shared.wgsl`. Both `gpu.ts` (renderer) and `parity.browser.ts` (test harness) prepend that fragment as a plain string to their own shader. Shaders already load via Vite `?raw` imports, so this needs no preprocessor or build tooling. The new `shadow-parity.wgsl` is a thin entry point containing **no copy of the math**.

**Tech Stack:** TypeScript, WGSL, WebGPU, Vite (`?raw` imports), Vitest, playwright-core (headless Chrome for GPU routes).

Spec: `docs/specs/2026-07-21-shadow-classifier-parity-design.md`

## Global Constraints

- **No `Co-Authored-By` trailer in any commit.** Verify after each commit with `git log -1 --format=%B | grep -ci "co-authored"` — must print `0`.
- **Do NOT push to origin. Do NOT merge. Do NOT switch branches.** Work stays on `feat/shadow-parity`.
- `?parity` must report maxRelErr **exactly 9.690e-7**. This is a hard gate — if it differs by even one digit, STOP and report rather than adjusting anything.
- `?shadow` must stay PASS with apparent radius **≈ 4.51 M** and calibration **0.868**.
- `npm test` must stay at **58 passing**. This work adds no CPU-side unit tests.
- **No rendered pixel may change.** This is a test-coverage change only.
- Prepend (never append) the shared fragment, so declarations always precede use.
- GPU routes need the dev server running on **:5173** first (`npm run dev`). `npm run verify:gpu` works without a `BASE=` override.

## File Structure

| File | Responsibility |
|---|---|
| `src/render/shadow-shared.wgsl` | **Create.** Sole copy of the WGSL classifier: `A_EPS`, `criticalXi`, `criticalEta`, `classifyCaptured`. Prepended by both consumers. |
| `src/render/raytrace.wgsl` | **Modify.** Remove lines 226–252 (the classifier block); the functions now arrive via the prepended fragment. |
| `src/render/gpu.ts` | **Modify.** Import the fragment and prepend it when building the raytrace shader module. |
| `src/render/shadow-parity.wgsl` | **Create.** ~12-line entry point: reads a case, calls `classifyCaptured`, writes 0.0/1.0. No math. |
| `src/test/parity.browser.ts` | **Modify.** Build the classification cases, dispatch, compare against `classify()` from `shadow.ts`, fold into `maxErr`. |
| `README.md` | **Modify.** One-line update to the `?parity` route description. |

---

### Task 1: Extract the classifier into a shared fragment

Pure refactor. The deliverable is "the renderer behaves identically while the classifier lives in one file." There are no new tests here — the existing gates ARE the test, and they must be run before and after to prove nothing moved.

**Files:**
- Create: `src/render/shadow-shared.wgsl`
- Modify: `src/render/raytrace.wgsl:226-252` (delete the block)
- Modify: `src/render/gpu.ts:2-4` (import), `src/render/gpu.ts:91` (prepend)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `shadow-shared.wgsl` exporting WGSL symbols `A_EPS` (const), `criticalXi(r: f32, a: f32) -> f32`, `criticalEta(r: f32, a: f32) -> f32`, `classifyCaptured(xi: f32, eta: f32, a: f32) -> bool`. Task 2 relies on `classifyCaptured` with exactly that signature.

- [ ] **Step 1: Record the baseline**

Start the dev server in one terminal, leave it running for the whole task:

```bash
npm run dev
```

In another terminal:

```bash
npm run verify:gpu
```

Expected — copy these numbers down, they are the comparison for Step 6:

```
✓ PASS  /?parity   PARITY PASS — maxRelErr=9.690e-7 over 12 cases
✓ PASS  /?shadow   SHADOW PASS (structural) — centred dark shadow=true, ringed by disk=true
                   apparent radius ≈ 4.51 M; analytic critical curve = 5.196 M
                   camera calibration factor = 0.868
✓ PASS  /sky/milkyway-4k.jpg  (200)
```

- [ ] **Step 2: Create the shared fragment**

Create `src/render/shadow-shared.wgsl` with exactly this content. The function bodies are copied **verbatim** from `raytrace.wgsl:226-252` — do not reformat, reorder, or "improve" them; byte-identical math is what makes the no-pixel-change claim true.

```wgsl
// ---- Analytic Kerr critical curve. Twin of src/physics/shadow.ts -- keep coefficients in sync. ----
//
// This is the SOLE WGSL copy of the classifier. It is prepended as a plain string to
// raytrace.wgsl (by gpu.ts) and to shadow-parity.wgsl (by parity.browser.ts), so the ?parity
// route exercises the same bytes the renderer compiles. Do not inline a second copy anywhere:
// that is precisely the failure mode this file exists to prevent.
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

- [ ] **Step 3: Delete the block from raytrace.wgsl**

Delete lines 226–252 of `src/render/raytrace.wgsl` — everything from the comment line

```wgsl
// ---- Analytic Kerr critical curve. Twin of src/physics/shadow.ts -- keep coefficients in sync. ----
```

through the closing brace of `classifyCaptured`, inclusive. The line immediately before the deletion is the closing `}` of the function ending at line 224; the line immediately after is the blank line preceding `@compute @workgroup_size(8,8) fn main(...)`. Leave exactly one blank line between them.

Do **not** touch anything else in the file — in particular leave the `usable` guard and the `accum` write-site sanitize alone.

- [ ] **Step 4: Verify the shader is now broken**

This step proves the fragment is genuinely load-bearing rather than dead weight.

```bash
npm run verify:gpu
```

Expected: **FAIL.** The browser console will report a WGSL compile error along the lines of `unresolved identifier 'classifyCaptured'` (exact wording varies by driver). If this instead PASSES, the deletion did not take effect — stop and investigate before continuing.

- [ ] **Step 5: Wire the fragment into gpu.ts**

In `src/render/gpu.ts`, add the import alongside the existing shader imports (currently lines 2–4):

```ts
import shadowSharedWGSL from "./shadow-shared.wgsl?raw";
```

Then change line 91 from:

```ts
    const cMod = this.device.createShaderModule({ code: raytraceWGSL });
```

to:

```ts
    // Prepended, not appended: declarations must precede use. shadow-shared.wgsl is the sole copy
    // of the critical-curve classifier and is shared verbatim with the ?parity route.
    const cMod = this.device.createShaderModule({ code: shadowSharedWGSL + raytraceWGSL });
```

- [ ] **Step 6: Verify nothing moved**

```bash
npm run verify:gpu
```

Expected: identical to the Step 1 baseline — `maxRelErr=9.690e-7 over 12 cases`, shadow radius `≈ 4.51 M`, calibration `0.868`, sky 200.

If `maxRelErr` is anything other than `9.690e-7`, or the radius is not `4.51`, STOP and report. Do not proceed and do not adjust tolerances.

Then confirm the CPU suite and typecheck are unaffected:

```bash
npm test
npm run build
```

Expected: `58 passed`, and a clean build.

- [ ] **Step 7: Commit**

```bash
git add src/render/shadow-shared.wgsl src/render/raytrace.wgsl src/render/gpu.ts
git commit -m "Extract the Kerr critical-curve classifier into a shared WGSL fragment

raytrace.wgsl and the parity harness can now compile the same bytes rather
than two hand-synced copies. Pure refactor: ?parity unchanged at 9.690e-7,
?shadow radius unchanged at 4.51M."
```

Verify no trailer:

```bash
git log -1 --format=%B | grep -ci "co-authored"
```

Expected output: `0`

---

### Task 2: Add the classification case to ?parity

**Files:**
- Create: `src/render/shadow-parity.wgsl`
- Modify: `src/test/parity.browser.ts` (imports at top; new block before the `return`; the `rows` total)
- Modify: `README.md` (the `?parity` bullet under "Validation routes")

**Interfaces:**
- Consumes: `classifyCaptured(xi: f32, eta: f32, a: f32) -> bool` from `shadow-shared.wgsl` (Task 1); `classify(xi, eta, a): "captured" | "escaped"`, `criticalXiEta(r, a): [number, number]`, and `photonShellRange(a): [number, number]` from `src/physics/shadow.ts`.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Write the failing test — the parity block**

In `src/test/parity.browser.ts`, add to the imports at the top of the file:

```ts
import { classify, criticalXiEta, photonShellRange } from "../physics/shadow";
import shadowSharedWGSL from "../render/shadow-shared.wgsl?raw";
import shadowParityWGSL from "../render/shadow-parity.wgsl?raw";
```

Then insert this block immediately **before** the final `return { maxErr, rows: ... };` line:

```ts
  // --- shadow-classifier parity (CPU shadow.ts vs the SHIPPED classifier in shadow-shared.wgsl) ---
  // Unlike the blocks above, this does NOT compare against a separate copy of the math: the same
  // shadow-shared.wgsl that gpu.ts prepends to raytrace.wgsl is prepended here, so a desync
  // between the renderer and shadow.ts cannot hide. Cases straddle the critical curve at +/-1% in
  // eta, which is where a formula drift would first flip a pixel.
  const scases: { xi: number; eta: number; a: number }[] = [];
  for (const a of [0.9, 0.998]) {
    const [slo, shi] = photonShellRange(a);
    for (let i = 1; i <= 5; i++) {
      const r = slo + ((shi - slo) * i) / 6; // strictly inside the photon shell
      const [xiC, etaC] = criticalXiEta(r, a);
      scases.push({ xi: xiC, eta: 0.99 * etaC, a }); // just inside -> captured
      scases.push({ xi: xiC, eta: 1.01 * etaC, a }); // just outside -> escaped
    }
  }
  // The shell sampling never reaches these two branches, so cover them explicitly:
  scases.push({ xi: 0, eta: 26.5, a: 0 }, { xi: 0, eta: 27.5, a: 0 });      // |a| < A_EPS short-circuit
  scases.push({ xi: 1000, eta: 1, a: 0.9 }, { xi: -1000, eta: 1, a: 0.9 }); // out-of-bracket early return
  const sin_ = device.createBuffer({ size: scases.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const sarr = new Float32Array(scases.length * 4);
  scases.forEach((c, i) => { sarr.set([c.xi, c.eta, c.a, 0], i * 4); });
  device.queue.writeBuffer(sin_, 0, sarr);
  const sout = device.createBuffer({ size: scases.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const sread = device.createBuffer({ size: scases.length * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const smod = device.createShaderModule({ code: shadowSharedWGSL + shadowParityWGSL });
  const spipe = device.createComputePipeline({ layout: "auto", compute: { module: smod, entryPoint: "main" } });
  const sbind = device.createBindGroup({ layout: spipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: sin_ } }, { binding: 1, resource: { buffer: sout } }] });
  const senc = device.createCommandEncoder();
  const scp = senc.beginComputePass(); scp.setPipeline(spipe); scp.setBindGroup(0, sbind); scp.dispatchWorkgroups(scases.length); scp.end();
  senc.copyBufferToBuffer(sout, 0, sread, 0, scases.length * 16);
  device.queue.submit([senc.finish()]);
  await sread.mapAsync(GPUMapMode.READ);
  const sgpu = new Float32Array(sread.getMappedRange().slice(0));
  scases.forEach((c, i) => {
    // Absolute difference, not relative: the value is already a 0/1 flag, so a mismatch scores 1.0
    // -- three orders of magnitude above the 1e-3 pass threshold -- and agreement scores exactly 0,
    // leaving the documented 9.690e-7 gate untouched.
    const cpu = classify(c.xi, c.eta, c.a) === "captured" ? 1 : 0;
    maxErr = Math.max(maxErr, Math.abs(sgpu[i * 4 + 0] - cpu));
  });
```

Finally, change the return statement to include the new cases:

```ts
  return { maxErr, rows: cases.length + tcases.length + jcases.length + scases.length };
```

- [ ] **Step 2: Run it to verify it fails**

```bash
npm run verify:gpu
```

Expected: **FAIL.** `shadow-parity.wgsl` does not exist yet, so Vite cannot resolve the import — the page will error with something like `Failed to resolve import "../render/shadow-parity.wgsl?raw"`.

- [ ] **Step 3: Create the parity entry point**

Create `src/render/shadow-parity.wgsl`:

```wgsl
// Entry point for CPU<->GPU parity of the Kerr critical-curve classifier.
// Input per case: vec4(xi, eta, a, 0). Output per case: vec4(captured ? 1 : 0, 0, 0, 0).
//
// This file deliberately contains NO copy of the classifier math. shadow-shared.wgsl is prepended
// to it by parity.browser.ts, exactly as it is prepended to raytrace.wgsl by gpu.ts -- so this
// route verifies the code the renderer actually runs, not a hand-synced duplicate of it.
@group(0) @binding(0) var<storage, read> inp: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outp: array<vec4<f32>>;

@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = inp[gid.x];
  outp[gid.x] = vec4<f32>(select(0.0, 1.0, classifyCaptured(c.x, c.y, c.z)), 0.0, 0.0, 0.0);
}
```

- [ ] **Step 4: Prove the fragment is genuinely shared, not duplicated**

This is the step that validates the entire design, so do not skip it. Temporarily change the module line in `src/test/parity.browser.ts` from:

```ts
  const smod = device.createShaderModule({ code: shadowSharedWGSL + shadowParityWGSL });
```

to:

```ts
  const smod = device.createShaderModule({ code: shadowParityWGSL });
```

Run `npm run verify:gpu`. Expected: **FAIL** with a WGSL compile error naming `classifyCaptured` as unresolved. That failure is the proof that `shadow-parity.wgsl` holds no local copy of the math.

If it PASSES, a duplicate copy has crept in somewhere — stop and report.

Now revert the line back to `shadowSharedWGSL + shadowParityWGSL`.

- [ ] **Step 5: Run it to verify it passes**

```bash
npm run verify:gpu
```

Expected:

```
✓ PASS  /?parity   PARITY PASS — maxRelErr=9.690e-7 over 36 cases
```

Two things must both hold:
- `maxRelErr` is **exactly 9.690e-7** — unchanged, because agreement contributes 0.
- the case count rose from **12 to 36** (12 existing + 24 new), proving the new cases actually ran.

`?shadow` must still report radius `≈ 4.51 M` and calibration `0.868`.

If `maxRelErr` moved at all, the twins disagree — that is a real finding. STOP and report it rather than adjusting the cases or the comparison.

- [ ] **Step 6: Confirm the gate can actually fail**

A test that cannot fail is not a test. Temporarily edit `src/render/shadow-shared.wgsl` and change the bisection loop bound from `k < 24` to `k < 3`:

```wgsl
  for (var k = 0; k < 3; k++) {
```

Run `npm run verify:gpu`. Expected: **`?parity` FAILS** with a large `maxRelErr` (order 1.0, since at least one boundary case flips).

Revert the change back to `k < 24` and re-run to confirm PASS at `9.690e-7`.

- [ ] **Step 7: Update the README**

In `README.md`, under "Validation routes", change:

```markdown
- `?parity` — CPU↔GPU parity check for the core physics math
```

to:

```markdown
- `?parity` — CPU↔GPU parity check for the core physics math, including the shadow-edge classifier compiled from the same shared WGSL fragment the renderer uses
```

- [ ] **Step 8: Full verification**

```bash
npm test
npm run build
npm run verify:gpu
```

Expected: `58 passed`; clean build; all three GPU checks PASS with `maxRelErr=9.690e-7 over 36 cases` and shadow radius `≈ 4.51 M`.

- [ ] **Step 9: Commit**

```bash
git add src/render/shadow-parity.wgsl src/test/parity.browser.ts README.md
git commit -m "Cover the shadow-edge classifier in ?parity

Adds 24 boolean classification cases bracketing the critical curve at +/-1%
in eta at a=0.9 and a=0.998, plus the a~0 short-circuit and out-of-bracket
branches. Compares against shadow.ts through the same shared fragment the
renderer compiles, so a CPU/GPU desync now fails a gate instead of silently
moving the shadow edge. maxRelErr unchanged at 9.690e-7."
```

Verify no trailer:

```bash
git log -1 --format=%B | grep -ci "co-authored"
```

Expected output: `0`

---

## Done when

- `npm run verify:gpu` reports `maxRelErr=9.690e-7 over 36 cases`, `?shadow` PASS at ≈4.51 M / 0.868, sky 200.
- `npm test` reports 58 passing; `npm run build` is clean.
- `classifyCaptured` exists in exactly one WGSL file. Confirm with:
  ```bash
  grep -rln "fn classifyCaptured" src/
  ```
  Expected: exactly one path, `src/render/shadow-shared.wgsl`.
- Both commits are free of `Co-Authored-By`. Nothing pushed; still on `feat/shadow-parity`.
