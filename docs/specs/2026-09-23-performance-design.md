# Performance — design

**Date:** 2026-09-23
**Status:** implemented on perf/smooth-first
**Branch:** `perf/smooth-first` (created from `main` at `77c1bd9`)

## Outcome

- Bench (nvidia ampere, 1280×720 / 1920×1080 ms/frame; compare only back-to-back pairs, the laptop drifts thermally): baseline 110.0 / 262.3; Task 2 GPU readout 122.1 / 322.2 (noise, no GPU change); Task 3 uncapped loop at scale 1.0 146.0 / 356.8 → 147.5 / 369.9 back-to-back (noise; the gain is the adaptive scale, down to 50 % while animating); Task 4 far stride: no change (kept); Task 5 exact derivatives `b4e3f08` 110.6 / 277.9 and 117.3 / 315.0 → 71.8 / 164.5 and 72.6 / 185.5 back-to-back (≈ −35…−38 % at 1280×720, ≈ −41 % at 1920×1080).
- Success criterion, as measured (RTX 3050 Laptop; the RTX 5050 was not available): 60 fps is the controller's *target*, not a result. Full resolution ≈ 72 ms at 1280×720 and ≈ 165–186 ms at 1920×1080, so at the 50 % floor ≈ 18 ms (~55 fps) and ≈ 41–46 ms (~22–24 fps). An FPS readout next to Render scale shows what a machine achieves.
- Controller signal (final review I1): §3.3's rAF-delta EMA is replaced by the GPU busy time per frame (`Renderer.gpuMs`, from `onSubmittedWorkDone`, excluding time queued behind earlier frames). rAF deltas are vsync-quantised (16.7 ms at 60 Hz sits in the 13–18 ms dead band), so the scale could only ratchet down while animating. Thresholds unchanged.
- §3.4's sweep rules were superseded during implementation: the sweep used a converged monitored reference and the rule "no ray worse than today by more than 0.02 M disk / half a pixel sky" (header of `tests/sweep-farstride.test.ts`).
- Final gates: `npm test` 87 passed, 3 skipped; build clean; `?parity` PASS 6.789e-5 over 53 cases (was 2.663e-4); `?shadow` PASS 3.95 M / 0.761 (was 3.89 / 0.749: a lit axis pixel inside the shadow no longer stops the centre-column scan); sky 200; probe-axis PASS ×2; probe-scale all PASS (pause check ran).
- Constants: K_FAR 0.04 / DL_FAR_MAX 6 kept by the Task 4 sweep; H_TOL 1e-3, MAX_RETRY 8, F_AXIS 0.1, DL_FAR_MIN 0.05, K_FAR / DL_FAR_MAX all confirmed under exact forces by re-running the three sweeps.
- Follow-up: the GPU's |ΔH|/scale on the far-field parity step is now 2.7e-8 (was 2.4e-3 under FD), so the far-field monitor exemption's premise is gone; converging the near-axis far-field passage (r 60–150) should come before any far-stride saving.

## 1. Problem, as measured

Measured on 2026-09-23 with a throwaway headless benchmark that drove the real `Renderer` from the dev
server (RTX 3050 Laptop + Iris Xe; the user's own machine is an RTX 5050 Laptop, 8 GB):

| Cause | Evidence |
|---|---|
| Chrome may give WebGPU the integrated GPU | `requestAdapter()` returned `intel / gen-12lp` for every `powerPreference`; 550–890 ms/frame at 1280×720 on Iris Xe vs ~110 ms on the RTX with `--force_high_performance_gpu`. The user confirms they already run on the RTX. |
| Full re-trace of every pixel every frame | 1280×720 → 110 ms, 1920×1080 → 263 ms on the RTX; cost is linear in pixels (DPR capped at 1.5). |
| Over half the steps are in flat space | CPU-twin census of the default view (a = 0.9, i = 72°, 128×72 rays): mean 333 steps/ray, of which 180 are at r > 1.5·rOut walking in from rObs = 1000 (far stride capped at 6 M). Escaped rays: 365 of 615. |
| Each step is expensive | Finite-difference forces: every `rhs` evaluates the inverse metric 5× (1 + 4 for ∂/∂r, ∂/∂θ); RK4 = 4 `rhs` + 2 monitor evaluations ≈ 22 metric evaluations per step, each with sin, cos, `pow`, five divisions. |
| 15 fps cap | `main.ts` `TARGET_MS = 67`. |

Measured NOT to matter: bloom (~0 ms), the jet (~2 %), the step budget (1200 → 9600 steps: 526 → 564 ms),
monitor retries (~0 per ray), warp divergence (92 % of lanes busy).

## 2. Goal and success criteria

The animated default view is smooth on the user's RTX 5050 laptop at their normal window size, with no
loss of physical correctness.

- **Smooth first** (user's choice): target 60 fps while animating by lowering internal render resolution
  down to half; snap back to full resolution and converge to a sharp still when paused.
- Every change keeps these green: `npm test`, `npm run build`, `npm run verify:gpu` (`?parity`,
  `?shadow`, sky), `node scripts/probe-axis.mjs`. Any number that legitimately moves is re-baselined
  **explicitly and recorded** (README Status + commit message), never tuned to pass.
- A committed benchmark reports before/after GPU ms/frame for every task. Gains are reported, not claimed.

## 3. Design

### 3.1 Benchmark (`scripts/bench.mjs`, `npm run bench`)

Headless Chrome via playwright-core (same launch flags as `verify-gpu.mjs`, plus
`--force_high_performance_gpu` so the dGPU is measured on hybrid laptops). Loads the real `Renderer`
through the dev server, renders the default scene at a fixed 1280×720, **scale 1.0**, and reports: adapter
vendor/architecture, median GPU ms/frame over ≥ 6 frames after a warm-up (timed with
`queue.onSubmittedWorkDone()`), and the same at 1920×1080. It is a report, not a pass/fail gate — timings
vary across machines. Requires the dev server on :5173, like the other GPU scripts.

### 3.2 GPU readout

A readout line in the panel: adapter vendor + architecture (from `GPUAdapter.info`) and the current
internal render scale. If the vendor is `intel`, or the architecture/description identifies an
integrated AMD part, show a short warning: WebGPU is on the integrated GPU — set Chrome to *High
performance* in Windows Settings → System → Display → Graphics and restart Chrome. Purely informational;
never blocks rendering. `Renderer` exposes the adapter info it received.

### 3.3 Uncapped loop + adaptive internal resolution

- **Loop:** remove `TARGET_MS`; render on every `requestAnimationFrame`.
- **Internal resolution:** `Renderer` gets a `scale` in [0.5, 1.0]. The trace, the accumulation buffer
  and bloom run at `internalW × internalH = round(displayW·scale) × round(displayH·scale)`; the present
  pass upsamples the accumulation buffer bilinearly to the display size (the bloom upsample already does
  this). `accumBuf` and the bloom buffers are allocated once at display size (scale 1.0) and indexed at
  the internal width, so a scale change never reallocates.
- **Uniforms:** `res` becomes the internal size (raytrace and bloom are unchanged in meaning: they already
  work in `U.res`). Two floats are appended, `outW, outH` (display size), used only by `present.wgsl` for
  the upsample and the vignette. `UNIFORM_SIZE` grows 96 → 112 bytes; `uniforms.ts`, its test and every
  WGSL `Uniforms` struct that needs the new fields are updated together.
- **Controller (animating only):** smoothed frame time from rAF deltas (EMA; *implemented as the GPU busy
  time per frame instead, see Outcome*). If > 18 ms, scale ×0.9; if
  < 13 ms, scale ×1.1; clamped to [0.5, 1.0]; at most one change per 500 ms (hysteresis so the image does
  not pump). A scale change is a reset (`blend = 1`), exactly as a slider change is today.
- **Paused:** scale snaps to 1.0 and the existing progressive running mean converges to a sharp still.
- **Validation routes** (`?parity`, `?shadow`) and `readbackPresented` always run at scale 1.0, so this
  change cannot move their numbers.

### 3.4 Far-field stride

> **Superseded during implementation** (controller ruling, SDD ledger): the shipped controller is not a
> converged reference near the axis and an absolute half-pixel sky rule is unsatisfiable for near-critical
> rays; the sweep instead used a converged monitored reference and the rule "no ray worse than today by
> more than 0.02 M disk / half a pixel sky" — see the header of `tests/sweep-farstride.test.ts`.

`stepSize`'s far branch (`r > 1.5·rOut`) is today `clamp(0.04·r, 0.6, 6.0)`, then the angular axis cap
(`F_AXIS = 0.1`, `DL_FAR_MIN = 0.05`). It becomes `clamp(K_FAR·r, 0.6, DL_FAR_MAX)` followed by the same
angular cap, in both twins. `K_FAR` and `DL_FAR_MAX` are chosen by a measured sweep
(`tests/sweep-farstride.test.ts`, `SWEEP=1` pattern) over the existing sweep views plus the axis-cap sets,
against the shipped controller as reference (plus a converged check of that reference), with these
rules, applied mechanically:

1. zero fate flips on every set;
2. disk hits within 0.02 M of the reference;
3. escaped rays: sky direction (`skyDir` twin on the CPU) within half a screen pixel of the reference,
   i.e. `< 0.5 · 2·fovScale / (H·rObs)` rad at H = 720 (≈ 1.9e-5 rad);
4. among survivors, fewest mean steps; ties → smaller values.

Candidates: `K_FAR ∈ {0.04, 0.08, 0.12, 0.2, 0.3}`, `DL_FAR_MAX ∈ {6, 20, 50, 150}`. The table goes in the
commit message. The angular cap and `H_TOL_FAR` exemption are unchanged. The parity "far" and
"far-axis" cases exercise the new branch through the dl0 sentinel.

### 3.5 Exact derivatives

`rhs()` in both twins replaces the central finite differences of `hquad` with closed-form
∂(g^μν p_μ p_ν)/∂r and ∂/∂θ built from the analytic derivatives of Σ, Δ, A and the five inverse-metric
components. Each metric evaluation computes sin θ, cos θ, Σ, Δ once (no `pow`: `(r²+a²)²` as a product).
Semantics that must be preserved exactly:

- the `POLE_S2` floor: where `sin²θ < POLE_S2` the floored term's θ-derivative uses the floored
  denominator (derivative of `max(s2, POLE_S2)` is 0 below the floor), matching what the FD scheme
  approximated;
- `p_t` and `p_φ` derivatives remain exactly 0 (Killing).

Verification:

- A CPU test compares analytic `rhs` against a high-accuracy f64 reference (Richardson-extrapolated
  central differences) over a grid of (r, θ, a, momenta) that includes near-horizon, near-axis and
  equatorial states: agreement to 1e-8 relative. The existing FD `rhs` stays in `geodesic.ts` as
  `rhsFD` for this test and for the h-parameter parity history; the integrator uses the analytic one.
- All existing physics tests (`geodesic`, `camera`, `trace`, `shadow`, …) must pass unchanged.
- `?parity`: the GPU and CPU now integrate the same equations, so the barrier-case CPU comparator no
  longer needs `h = 1e-4`; the comparator's `h` parameter is removed and the maxRelErr is re-baselined
  and recorded. Expected to tighten, not loosen; a loosening is investigated before it is recorded.
- The H_TOL sweep (`tests/sweep-htol.test.ts`) and the axis-cap sweep (`tests/sweep-axiscap.test.ts`)
  are re-run. If their mechanical selection rules pick different constants under exact forces, the
  constants change in both twins, with the tables in the commit. Otherwise they are recorded as confirmed.
- `?shadow`, the axis probe and the i = 8° / i = 72° renders are re-checked; the README limitation about
  the in-shadow axis pixels (attributed to FD-h) is updated to whatever is now observed.

### 3.6 Order

1. Benchmark → 2. GPU readout → 3. Uncapped loop + adaptive scale → 4. Far-field stride → 5. Exact
derivatives. Each is independently reviewable and records `npm run bench` before/after.

## 4. Risks

- **Scale changes read as flicker.** Mitigated by hysteresis and the ×0.9/×1.1 steps; if still visible,
  widen the dead band rather than add temporal reconstruction.
- **Far stride vs star positions.** Rule 3 bounds the escaped-direction error to half a pixel; if no
  candidate beats the shipped stride under it, the shipped stride stays and that is reported.
- **Analytic derivatives in f32** near the horizon (Δ → 0) may cancel differently from FD. The parity
  near-horizon case and the shadow gate cover it; any regression is investigated, not absorbed.
- **Benchmark noise.** Medians over ≥ 6 frames after warm-up; report, don't gate.

## 5. Out of scope

Temporal/checkerboard reconstruction; Carter first-order equations; moving the camera closer; an
approximate flat-space jump; the jet-model axis line; the true critical-curve shadow gate; the GPU
r-relative FD step (moot once FD is gone).
