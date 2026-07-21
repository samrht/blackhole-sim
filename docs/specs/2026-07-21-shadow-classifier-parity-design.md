# Shadow-classifier parity — design

Date: 2026-07-21
Status: approved
Follows: `docs/specs/2026-07-20-photon-ring-detail-design.md`

## 1. Problem

The Photon-ring detail feature added an analytic Kerr critical-curve classifier as a CPU/GPU twin:
`src/physics/shadow.ts` (the verified original) and a hand-transcribed WGSL port at
`src/render/raytrace.wgsl:226-252`. The twins agree today — the final whole-branch review verified
them coefficient-for-coefficient and ran 234 boundary assertions with zero mismatches.

Nothing enforces that they keep agreeing. `src/render/parity.wgsl` never references
`criticalXi`, `criticalEta`, or `classifyCaptured`, so ~26 lines of physics that directly determine
the rendered shadow edge are guarded only by human diff-reading. The failure mode is a subtly wrong
shadow boundary — no existing gate would catch it.

### 1.1 The deeper problem: parity tests copies, not shipped code

Investigation while designing this fix found that the existing parity harness does not test the
shader that actually renders. Every WGSL function is duplicated:

- `raytrace.wgsl:175-190` contains the jet code; `jet-parity.wgsl` contains a second copy.
- `parity.wgsl` states it "mirrors the metric/orbit/g-factor helpers from raytrace.wgsl" — again a
  separate copy.

So `?parity` proves *`jet-parity.wgsl` agrees with `jet.ts`*. It proves nothing about
`raytrace.wgsl`. A desync between `raytrace.wgsl` and its parity twin is invisible to every gate in
the project.

This matters directly: naively following the existing pattern for the classifier would compare a
fresh third copy against `shadow.ts` while the rendered shadow edge continued to come from
`raytrace.wgsl`. That closes nothing and adds false confidence — worse than the current honest gap.

## 2. Goal and non-goals

**Goal:** make `?parity` exercise the *same bytes* the renderer compiles for the shadow classifier,
so any future edit that desyncs the twins fails a gate.

**Non-goals:**
- Retrofitting the jet, metric, and turbulence parity cases to the shared-fragment pattern. This
  spec establishes the pattern on the smallest, newest case; the others are a tracked follow-up.
- Changing any rendered pixel. This work is purely a test-coverage change.
- Changing the documented `?parity` gate constant of 9.690e-7 (see §4).

## 3. Design

### 3.1 Shared fragment

Extract `raytrace.wgsl:226-252` verbatim into a new `src/render/shadow-shared.wgsl`. The block is
fully self-contained: it depends only on WGSL built-ins (`cos`, `acos`, `abs`) and references no
uniforms or other functions. It contains `A_EPS`, `criticalXi`, `criticalEta`, and
`classifyCaptured`.

Both consumers prepend it as a plain string:

- `gpu.ts`: `createShaderModule({ code: shadowSharedWGSL + raytraceWGSL })`
- `parity.browser.ts`: `createShaderModule({ code: shadowSharedWGSL + shadowParityWGSL })`

Shaders are already loaded via Vite `?raw` imports, so this needs no preprocessor, include
directive, or build tooling. Prepending rather than appending guarantees declaration-before-use
regardless of WGSL forward-reference rules.

`src/render/shadow-parity.wgsl` is a thin entry point (~12 lines) that reads a case, calls
`classifyCaptured`, and writes the result. **It contains no copy of the math.**

### 3.2 Data flow

Input per case: `vec4(xi, eta, a, 0)`.
Output per case: `vec4(f32(captured), 0, 0, 0)` — the bool encoded as 0.0/1.0, because storage
buffers cannot carry `bool`.

The harness computes `classify(xi, eta, a) === "captured" ? 1 : 0` on the CPU and folds
`|gpu − cpu|` into the existing `maxErr` accumulator, matching how the other parity blocks report.

### 3.3 Test cases

Mirror the CPU-side test that already passes 238 assertions (`tests/shadow.test.ts`): sample radii
across the photon shell at a=0.9 and a=0.998, take the exact `(ξ_c, η_c)` from `criticalXiEta`, and
emit two cases per sampled radius:

- `(ξ_c, 0.99·η_c)` → expect captured
- `(ξ_c, 1.01·η_c)` → expect escaped

Plus coverage of the two other branches, which the shell sampling does not reach:

- the `|a| < A_EPS` short-circuit: `η + ξ²` either side of 27
- the out-of-bracket early return: `ξ` outside `[xiHi, xiLo]`

Target roughly 16–24 cases, not the CPU test's 238. `?parity` runs on every `verify:gpu` and
dispatches a single workgroup; sensitivity here comes from tight bracketing around the critical
curve, not from volume.

### 3.4 Why boolean-only comparison

Only the captured/escaped outcome is compared — not the intermediate `criticalXi` / `criticalEta`
float values.

Comparing the floats would catch formula drift marginally earlier, before it flips a classification.
But those values carry genuine f32 error: the final review measured `criticalEta` relative error
reaching ~1.1e-4 near the shell inner edge at high spin. Folding that into `maxErr` would move the
reported gate from 9.690e-7 to ~1e-4, requiring updates to the README, the photon-ring spec, and
saved memory — a large blast radius for a small gain.

Boolean comparison with cases bracketed at ±1% of η_c is sensitive to any drift that affects
pixels, which is the property that matters, and contributes exactly 0 to `maxErr` when the twins
agree.

## 4. Invariants

- `?parity` must continue to report maxRelErr **exactly 9.690e-7**. A shadow mismatch contributes
  error 1.0, far above the 1e-3 pass threshold, so a desync fails loudly; agreement contributes 0
  and leaves the constant untouched.
- `?shadow` must remain PASS with the rendered radius unchanged at ~4.51M.
- No rendered pixel may change. The extracted fragment is byte-identical to the code it replaces,
  and prepending only affects declaration order, not semantics.
- `npm test` must stay at 58 passing (this work adds no CPU-side unit tests).
- No `Co-Authored-By` trailer in any commit. Nothing pushed to origin.

## 5. Risks

**Spurious mismatch from f32 bracketing.** If `criticalXi` at a bracket endpoint differed between
f32 and f64 enough to flip a bracketing decision, a legitimately-agreeing implementation could
report a false failure. Evidence this is not a live risk: the review measured f32 bisection
converging to 2.38e-7 with zero stalled iterations, and minimum η_c at interior sampling was 0.695
(a=0.9) / 1.534 (a=0.998), so ±1% sits far above float noise.

If a case does prove flaky, the correct fix is to move it further from the shell endpoint — **not**
to loosen the comparison. A boolean comparison has no tolerance to loosen, which is deliberate.

**Line-number churn.** Prepending shifts reported line numbers in WGSL compile errors for
`raytrace.wgsl` by the fragment's length. Accepted; the alternative (an include-directive
preprocessor that preserves offsets) is disproportionate machinery for this gain.

## 6. Follow-ups (out of scope)

1. Apply the shared-fragment pattern to the jet, metric, and turbulence parity cases, so every
   `?parity` block tests shipped code rather than a copy.
2. Analytic Christoffels or f64/compensated integration to raise the n=1 accuracy ceiling.
3. Normalized Bardeen camera to retire the 0.868 calibration factor.
