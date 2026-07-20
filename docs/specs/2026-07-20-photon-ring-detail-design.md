# Photon-Ring Detail — Design

**Status:** approved (brainstorm) — pending spec review
**Date:** 2026-07-20
**Depends on:** Tier 1, Tier 2A (Living Disk), Tier 2B (Relativistic Jet), Lensed Sky — all shipped on `main` (`6eed4df`).

## 1. Goal

Make the **n=1 photon subring** — the higher-order lensed image that stacks just outside the shadow —
render as a distinct, sharp arc, and remove a correctness bug that is currently swallowing it.

Non-goals: resolving n=2 (see §2), any change to disk/jet/sky physics, any zoom or FOV control.

## 2. The bug

The integrator loop is capped at `U.maxSteps` (hardcoded **1200**). All three real terminations —
disk hit, horizon capture, escape — `break` out. But when a ray simply **runs out of steps**, the
loop falls through with `color` still at its `vec3(0.0)` initializer.

**Step-starved rays therefore render black and blend into the shadow.** Today's shadow edge is
partly a step-budget artifact, not geometry.

How much budget a winding actually costs, at a=0.9 (`rh = 1.436`, step policy
`dl = clamp(0.02·(r − rh), 0.002, 0.5)`):

| Photon-shell radius | `dl` | steps / winding |
|---|---|---|
| r ≈ 1.557 (prograde edge, `2{1+cos[⅔·arccos(−a)]}`) | 0.0025 | ~3900 |
| r ≈ 3.910 (retrograde edge) | 0.049 | ~500 |

Travelling in from `rObs = 1000` to the strong field costs only ~280 steps, so far-field striding is
already efficient and is *not* the problem. The remaining ~920 steps buy well under half a winding
at the prograde edge of the photon shell.

**Consequence for scope:** raising the budget to 4800 fully resolves the retrograde side of the
shell and most of n=1, but the innermost prograde subring remains budget-limited. This is accepted
in v1 and must not be described as "fully fixed". n=2 is out of reach at any budget here for a
separate reason: successive subrings are thinner by ~e^(−2π) ≈ 1/535, so at the default
`fovScale = 14` n=2 is far below one pixel. Seeing it needs a narrow-FOV zoom, which is out of scope.

## 3. Components

### 3.1 `src/physics/shadow.ts` (new, pure TS)

Analytic Kerr shadow boundary via the spherical-photon-orbit impact parameters (Bardeen), M=1:

```
Δ(r)  = r² − 2r + a²
ξ(r)  = [ (r² − a²) − r·Δ(r) ] / [ a(r − 1) ]
η(r)  = r³[ 4Δ(r) − r(r − 1)² ] / [ a²(r − 1)² ]
```

Sweeping r across the photon shell traces the exact shadow outline for any spin/inclination. The
shell endpoints are the equatorial photon orbits, where η = 0:
`r_ph = 2{1 + cos[⅔·arccos(∓a)]}` (− for prograde, + for retrograde).

Exports:
- `criticalXiEta(r, a)` → `[ξ, η]`
- `photonShellRange(a)` → `[r_min, r_max]`
- `shadowBoundary(a, incl, nSamples)` → array of celestial-plane `(α, β)` points
- `shadowRadiusSchwarzschild()` → `√27`

**Numerical hazard (must be handled explicitly):** both ξ and η are 0/0 as a→0. The η numerator
factors as `4Δ − r(r−1)² = −r(r−3)² + 4a²`, so at a→0, r=3 the limit is `27·4a²/(4a²) = 27` —
correct, but only as a limit; naive evaluation divides by zero. Implement an explicit
`|a| < A_EPS` branch (A_EPS = 1e-4) returning the exact Schwarzschild circle of radius √27, and
assert continuity across the cutover in tests.

Unit tests (`tests/shadow.test.ts`):
- a→0 reduces to a circle of radius √27 (within 1e-6)
- continuity of ξ, η across the A_EPS branch cutover
- `photonShellRange(0.9)` ≈ [1.5579, 3.9103] (verified numerically)
- η = 0 at both shell endpoints (holds to ~1e-15)
- ξ_c is strictly decreasing across the shell, range ≈ [−6.832, 2.844] at a=0.9 — this is the
  precondition for the §3.2 bisection and must be asserted, not assumed
- a=0.9 boundary is flattened on the prograde side (asymmetric in α about 0)

### 3.2 Shader: explicit outcome + principled exhaustion

Replace the implicit fallthrough with an explicit outcome, and classify unresolved rays using the
**conserved** impact parameters rather than instantaneous state:

```wgsl
var resolved = false;
// disk hit / horizon capture / escape each set `color` and `resolved = true`, then break
// after the loop:
if (!resolved) {
  // classify this ray's (xi, eta) against the analytic critical curve
  //   inside  -> captured -> black
  //   outside -> escaping -> sample sky along the current direction
}
```

Classification must use (ξ, η), **not** the sign of `p_r`: a winding ray oscillates in r, so `p_r`'s
sign at an arbitrary cutoff is effectively random and would produce salt-and-pepper noise across the
near-critical annulus. ξ and η are constants of motion, so the classification is deterministic and
independent of where the budget ran out.

**The inside/outside test, stated explicitly.** ξ_c(r) is monotonic over the photon shell, so the
curve can be inverted in ξ:

1. If ξ lies outside `[ξ_c(r_max), ξ_c(r_min)]`, the ray has no turning point in the shell → **escapes**.
2. Otherwise bisect r over `photonShellRange(a)` to find r\* with `ξ_c(r\*) = ξ` (fixed 24 iterations;
   no `while` loops in WGSL, and 24 halvings of the shell width is far below f32 resolution).
3. **Captured iff η < η_c(r\*)**, else escapes.

Exported from `shadow.ts` as `classify(xi, eta, a) → "captured" | "escaped"` and mirrored in WGSL.
The bisection is the only loop added to the shader and runs once per unresolved ray, not per step.

The WGSL classifier is a coefficient-for-coefficient twin of `shadow.ts`, following the existing
`skymap.ts` ↔ `raytrace.wgsl` twin pattern.

For a ray classified as escaping, the sky is sampled along the current instantaneous direction. This
is approximate — a chaotic near-critical ray's true exit direction is unpredictable — but such rays
are a measure-zero set and the alternative (black) is strictly worse.

### 3.3 Detail slider

`maxSteps` is **already** a uniform (uint slot `i[13]`), so there is no buffer-layout change and
`UNIFORM_SIZE` stays 96.

- `state.maxSteps`, default **4800**
- UI slider `id="detail"`, range 1200–9600, step 600, readout in steps
- `reset()` on input

### 3.4 `?shadow` upgraded to an analytic comparison

The current check asserts "centred dark shadow" + "ringed by disk" and reports apparent radius
≈ 4.51M against ideal √27 ≈ 5.2M, attributing the gap to a **camera scale factor ≈ 0.87** that has
never been justified. The new check measures the rendered boundary and compares it to
`shadow.ts` at the same spin/inclination.

**Scope of the automated check:** `?shadow` stays at **a=0, pole-on**, where the analytic boundary is
exactly a circle of radius √27 and "apparent radius" is unambiguous. The general a=0.9 outline is
exercised in `tests/shadow.test.ts` only, where it can be compared point-by-point without needing to
reduce a non-circular boundary to a single number.

**Recorded prediction (so it can be wrong on the record):** most of the 0.87 is expected to be
genuine camera projection — `rObs = 1000` is finite rather than asymptotic, and `fovScale = 14` sets
the plate scale — with a smaller contribution from step starvation inflating the dark region. If the
corrected radius lands near 5.2 with no residual scale factor, then the 0.87 was entirely this bug
and the old check was rationalizing an artifact.

Deliverable: measure apparent radius at maxSteps 1200 vs 4800, report both against the analytic
value, and document the decomposition in the README.

## 4. Invariants

- `?parity` is math-only and **must stay bit-identical at 9.690e-7**.
- The "all features off ⇒ unchanged" invariant **cannot hold for this feature** and is explicitly
  waived. Correcting the artifact necessarily changes pixels near the shadow edge; that is the
  point. This is the first feature in the project to break that property, and the README must say so.
- Setting the Detail slider to 1200 reproduces the old step budget but **not** the old image, because
  the exhaustion classifier still applies. This is intended.

## 5. Performance

Measured relative scaling (headless, sublinear because `maxSteps` is a *cap* — rays that escape,
get captured, or hit the disk break early, so only the thin near-critical annulus spends the full
budget):

| maxSteps | relative frame time |
|---|---|
| 1200 | baseline |
| 2400 | +18% |
| 4800 | **+44%** |
| 9600 | +123% |

Absolute headless numbers were ~2.3 fps at baseline and are **not** representative — headless runs a
software/weak-Vulkan path, not the RTX 3050. Only the *shape* of the curve transfers, and it does so
because it follows from the early-out structure rather than from any hardware detail.

The existing 15 fps cap (`TARGET_MS = 67`) means an over-budget frame degrades to a lower framerate
rather than breaking, and the Detail slider lets the user dial cost down on weaker hardware.

## 6. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| SIMD divergence: one slow lane stalls a whole workgroup | Near-critical pixels are spatially clustered in a thin annulus, so most workgroups are uniformly fast; measured cost is sublinear. Two-pass compaction is the escape hatch if it bites. |
| a→0 singularity in ξ, η | Explicit A_EPS branch + continuity test. |
| 4800 still starves the innermost prograde subring | Accepted and documented in v1; Detail slider goes to 9600. |
| Classifier twin drifts from `shadow.ts` | Same CPU/GPU twin discipline as `skymap.ts`; parity-style spot check. |
| Perf regression on the 3050 | Detail slider; 15 fps cap degrades gracefully. |

## 7. Out of scope

- n=2 / n=3 subrings and any narrow-FOV zoom control.
- Two-pass compaction with an append buffer and indirect dispatch (§6 escape hatch only).
- Adaptive per-pixel step budgets driven by proximity to the critical curve.
- Any change to the disk, jet, or sky models.
