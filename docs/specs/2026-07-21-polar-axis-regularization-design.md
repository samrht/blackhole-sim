# Polar-axis regularization — design

**Date:** 2026-07-21
**Status:** approved, pending implementation plan
**Branch:** `fix/polar-axis` (to be created from `main`)

## 1. The defect

The Boyer–Lindquist polar axis carries a coordinate singularity: `g^φφ` contains a `1/sin²θ`
factor that diverges as θ → 0 or π. Since Tier 1 the renderer has suppressed this by flooring the
denominator:

```
const POLE_S2 = 1e-3;                       // src/render/raytrace.wgsl:26, src/physics/kerr.ts:9
let s2 = sin(th)*sin(th); let s2d = max(s2, POLE_S2);
```

`√POLE_S2 ≈ 0.0316 rad ≈ 1.81°`, so inside a cone of that half-angle around each pole the metric is
deliberately wrong.

The README described this as "removing a thin black meridian seam and hard central cap". That was
false and has since been corrected: the floor does not remove the artifact, it *converts* it into a
solid black cone. Rendering at a = 0 with the sky and jet off shows a black band running from the
top of the shadow to the frame edge at i = 72°, widening into a broad wedge that cuts clean through
the photon ring at i = 8°.

### Why it renders black rather than merely distorted

Inside the cone `s2d` is pinned to the constant `POLE_S2`, so `∂/∂θ` of the `g^φφ p_φ²` term in the
Hamiltonian loses its `1/sin²θ` contribution entirely. Forces in `rhs()` come from finite
differences of `hquad()`, so this is not a small perturbation — **the centrifugal barrier vanishes**.
Rays that should be turned back by it instead continue inward, reach `r ≤ rh·1.005`, and are
classified as captured. Hence black, not smeared.

### What the correct behaviour is

Under Carter separation the θ-motion obeys

```
p_θ² = Θ(θ) = η + a²cos²θ − ξ²cot²θ
```

As θ → 0, `cot²θ → ∞`. Unless ξ = 0 exactly, Θ goes negative, which is forbidden. So:

- **ξ ≠ 0 rays have a genuine centrifugal turning point at θ_min > 0 and never reach the axis.**
- **ξ = 0 rays (a measure-zero set) do reach the axis and pass straight through it.**

The Θ expression above is written in the usual E = 1 normalization. **Do not re-derive the sign and
normalization conventions — take them from `src/render/camera-shared.wgsl` and
`src/physics/camera.ts`, which are now the single source of truth and are parity-gated.** This
project shipped an inclination-flipping bug for its entire history because the momentum sign
convention was re-derived inconsistently in one place; that is a live hazard, not a hypothetical.

"Correct axis behaviour" therefore means resolving the turning point properly, not making the axis
traversable. The present code lets rays cross a barrier that physically should have stopped them.

## 2. Success criterion

Physically correct near-axis behaviour, verifiable by an automated gate — not merely an artifact
that has been made too small to see. Performance is explicitly not a constraint for this change.

## 3. Design

### 3.1 The floor becomes a NaN guard, not a physics cap

`POLE_S2` drops from `1e-3` to `1e-12` in both twins (`src/physics/kerr.ts`, `src/render/raytrace.wgsl`).

It is not retained as a fudge. It exists solely to prevent `inf · 0 = NaN` when an exactly-on-axis
ray with `p_φ = 0` evaluates `g^φφ p_φ²`. At `1e-12` the affected cone has half-angle `1e-6` rad
≈ 6e-5°, thousands of times below one pixel at any field of view this renderer supports, versus
1.81° today. Everywhere a ray can physically feel the barrier, the barrier is now exact.

### 3.2 Overshoot prevented by constraint-monitored adaptive stepping

Restoring the true forces is unsafe under the current step controller. `dl` is sized on `r` alone
(`clamp(0.02*(r - rh), 0.002, 0.5)`), so a ray approaching its θ turning point takes a step far too
large for the local dynamics, receives an enormous `1/sin²θ` kick, and departs on a garbage
trajectory. This is precisely the divergence the floor was originally introduced to hide.

Rather than deriving θ_min in closed form and clamping to it, use the null constraint as an error
estimator. For a null geodesic

```
H = ½ g^μν p_μ p_ν = 0
```

exactly, for all affine parameter. `hquad()` already computes this quantity. After each RK4 step,
if `|H|` exceeds a tolerance, halve `dl` and redo the step, up to a bounded retry count.

Properties:

- Self-correcting; requires no turning-point algebra and no special-casing of the axis.
- Active on every step, so it tightens global integration accuracy, not only near the pole.
- Affordable because performance is not a constraint here.

The retry cap must be bounded so a pathological ray cannot spin forever. On exhausting retries the
ray proceeds with the smallest permitted step and is marked `resolved = false`, which routes it to
the existing conserved-quantity `(ξ, η)` classifier rather than letting a constraint-violating
trajectory be accepted as if it were trustworthy. Confirm that routing against the current loop
before relying on it — do not assume it.

**The |H| tolerance and the retry cap are to be determined empirically during implementation, not
guessed.** The plan must state the method: sweep candidate values, report the resulting step-count
and gate impact, and pick from measured data. A tolerance chosen to make a test pass is not
acceptable.

### 3.3 Axis crossing handled analytically for the rays that genuinely cross

If θ still lands ≤ 0 or ≥ π — reachable only when ξ ≈ 0 — continue the geodesic through the
coordinate singularity exactly rather than clamping:

```
θ → −θ      (or 2π − θ at the south pole)
φ → φ + π
p_θ → −p_θ
```

This is the exact analytic continuation, not an approximation. `ξ = p_φ` is unchanged, as it must
be.

## 4. Verification

### 4.1 New automated tests

1. **Conservation across a near-axis pass.** Integrate a ray whose turning point lies close to the
   axis; assert ξ and η are preserved to tolerance, and that `|H|` stays within the constraint
   bound for the whole trajectory.
2. **The barrier turns rays back.** A ray aimed near the pole with ξ ≠ 0 must reach a θ minimum and
   return, and must NOT terminate as captured. This is the direct regression test for the defect:
   it fails on today's code.
3. **Exact crossing.** A ξ = 0 ray must pass through the axis and emerge with φ shifted by π and
   the sign of p_θ flipped.

### 4.2 Existing gates

`gUp` is parity-covered, so changing `POLE_S2` will likely move `?parity`'s maxRelErr from its
current 9.690e-7 over 45 cases. This is a legitimate re-baseline, not a regression — but the
tolerance must be **re-baselined explicitly and recorded**, never loosened to accommodate a
disagreement that has not been understood. Both twins must change identically or parity will fail,
which is the intended safety property.

`?shadow` may shift its 4.49 M / 0.864 calibration slightly. Record the new values.

### 4.3 Visual confirmation

Re-render at a = 0 with sky and jet off, at i = 72° and i = 8°, and confirm the black wedge is
gone. Note that `src/main.ts` reads **only** the `steps` query parameter — inclination, spin, sky
and jet cannot be set by query string, and a probe script that tries to will silently capture four
identical default renders. Drive the DOM sliders (`#spin`, `#incl`, `#sky`, `#jet`) and assert the
values were actually applied.

## 5. Risks and things that will move

- **Step counts change everywhere.** The constraint check is global by explicit decision, so
  photon-ring behaviour — which is step-budget-limited — will shift alongside the axis fix. Some of
  that may read as improvement and some as regression, and the two changes will be entangled. This
  was accepted deliberately in preference to a near-axis-only retry.
- **`?parity` and `?shadow` numbers move.** Expected; see 4.2.
- **Not bit-identical.** The project's "features off ⇒ bit-identical" property was already waived
  by Photon-ring detail. This change also alters pixels, necessarily.

## 6. Explicitly out of scope

- Switching to Carter's separated first-order equations (`Σ dr/dλ = ±√R`, `Σ dθ/dλ = ±√Θ`). This
  would be exact, would remove finite-difference forces entirely, and would lift the n = 1 subring
  accuracy ceiling — but it is an integrator rewrite touching every twin and gate, with fiddly
  turning-point sign bookkeeping. Bundling it here would mean that when the gates move, we could
  not tell which change moved them. Tracked as a separate follow-up with its own spec.
- Kerr–Schild or Cartesian coordinates.
- Retiring the `?shadow` camera calibration factor.
