# Tier 2B — Relativistic Jet: Beamed, Limb-Brightened, Living Outflow

**Date:** 2026-07-19
**Status:** Design (brainstorming deliverable) — awaiting user review
**Scope:** Sub-project B of the Tier 2 upgrade. Adds a bipolar relativistic jet — a
volumetric, optically-thin emissive outflow along the spin axis — to the existing Tier 1 +
Tier 2A renderer.
**Depends on:** Tier 1 (complete & verified) and Tier 2A (Living Disk, merged). Physics spec:
`docs/specs/2026-05-30-relativistic-blackhole-accretion-design.md`; Tier 2A design:
`docs/specs/2026-07-19-tier2a-living-disk-design.md`.

---

## 1. Motivation & the core insight

Tier 1/2A render the accretion **disk** as an opaque equatorial surface: each ray marches
the Kerr geodesic inward and, on its *first* crossing of the θ = π/2 plane, takes the disk
color and **breaks**. Everything is a surface hit-test.

A jet is fundamentally different: it is a **3-D volume** of emitting plasma along the poles,
not a surface. Two consequences drive this whole design:

1. **The jet is optically thin and must be *integrated along the ray*, not hit once.** As a
   backward-traced ray travels from the camera toward the hole, it accumulates jet emission
   at every step where it lies inside the funnel, then terminates on the disk / horizon /
   starfield as before. The jet glows *in front of* whatever the ray finally hits and is
   naturally occluded by the opaque disk and the horizon behind it.
2. **We do not trace new rays.** The geodesic integrator already visits ~hundreds–thousands
   of points per ray. Jet sampling piggybacks on those existing steps: one funnel test plus
   one emission evaluation per step. Cost on the target RTX 3050 stays negligible.

The signature visual — the jet looking **one-sided** (bright approaching arm, near-invisible
counter-arm, as in M87) — is not painted in; it falls out of **relativistic Doppler beaming**
of plasma streaming outward near light-speed, evaluated per step from the photon direction
against the jet axis.

## 2. Goals / non-goals

**Goals**
- A **bipolar, collimated outflow** along the black-hole spin axis (θ ≈ 0 and θ ≈ π),
  rendered as an optically-thin emissive volume accumulated along each ray.
- A **parabolic, limb-brightened funnel**: emission peaked near the funnel *wall* (hollow
  edge-brightened rails, the realistic M87 look), collimated at the base and slowly flaring.
- **Relativistic beaming** producing the physical one-sided appearance: approaching jet
  Doppler-boosted bright, counter-jet dimmed toward invisibility, dependent on viewing angle.
- A **living jet**: brightness **knots propagate outward** along the axis over time, driven
  by the same sim-time clock as the Tier 2A disk.
- **Synchrotron-ish cool blue-white** palette, visually distinct from the warm thermal disk.
- Remains **real-time interactive** on the RTX 3050 (~15–30 fps); camera drag and all
  parameter tweaks stay live while the scene animates.
- **Hard invariant:** with the jet off (`jetStrength = 0`) the render is **bit-for-bit
  identical to Tier 2A** (and therefore to Tier 1 with all features off).

**Non-goals (this tier)**
- True synchrotron spectrum / spectral-index frequency dependence, and polarization.
- Explicit magnetic-field geometry or a Blandford–Znajek force-free solution (the jet is
  phenomenological, mirroring the Tier 2A decision to drop literal Σ-diffusion).
- Jet precession, instabilities (kink/helical), or self-absorption.
- Counter-jet lensing subtleties beyond whatever falls out of the existing geodesic.
- Tier 3 (full GRMHD, multi-GPU / offline).

## 3. Geometry — the parabolic funnel

Work in cylindrical coordinates derived from Boyer–Lindquist `(r, θ)` at each step:

- Axial height (signed, along the spin axis): `z = r · cos θ`
- Cylindrical radius: `ρ = r · sin θ`

The emitting region is the union of two funnels (`z > 0` top, `z < 0` bottom), symmetric in
geometry (the *brightness* asymmetry comes entirely from beaming, §5, not the shape).

**Funnel wall radius vs. height** (parabolic collimation):
```
rho_edge(z) = rho0 + slope * sqrt(|z|)
```
with `rho0 ≈ 0.6 M` (narrow throat) and `slope ≈ 0.7 M^{1/2}` (gentle flare). This is
collimated at the base and widens sub-linearly — the iconic paraboloidal profile.

**Axial extent:** emission is gated to `z_base ≤ |z| ≤ z_max` with soft fades:
- `z_base ≈ 2 M` — launch just above the horizon (r₊ = 1.44 M at a = 0.9); a `smoothstep`
  fade-in over `[z_base, z_base + 2]` avoids a hard edge at the throat.
- `z_max ≈ jetLength` (default 60 M) — a `smoothstep` fade-out over the top decade.

**Wall (limb-brightening) profile.** Rather than filling the funnel, emission peaks at the
wall. With normalized cross-funnel coordinate `q = ρ / rho_edge(z)`:
```
wall(q) = exp( -((q - q_peak)^2) / (2 * w_wall^2) )     // q_peak ≈ 0.8, w_wall ≈ 0.22
```
Outside `q > 1` (beyond the wall) emission is ~0; near the axis (`q → 0`) it is dim. This
yields the hollow-tube, edge-brightened-rails appearance. (A single design constant switch
`q_peak → 0` degrades gracefully to a filled cone if we ever want it.)

## 4. Living jet — the emission field

The scalar jet emissivity at a sampled point is:
```
E_jet(rho, z, t) = wall(q) * lengthFalloff(z) * knots(z, t) * (1 + turbAmp_jet * (noise - 0.5)*2)
```
- **`lengthFalloff(z)`** — overall dimming with height so the jet fades outward, e.g.
  `lengthFalloff = z_base / max(|z|, z_base)` (∝ 1/z), combined with the axial soft-fade gates.
- **`knots(z, t)`** — the traveling wave that makes blobs march outward:
  ```
  knots(z, t) = 1 + jetKnots * (vnoise(kz*|z| - v_knot * t * timeScale, seed) - 0.5) * 2
  ```
  `kz ≈ 0.35` (knot spacing), `v_knot ≈ 6` (pattern speed in M per sim-second at
  `timeScale = 1`). Reuses the **Tier 2A `vnoise` basis** (same integer-hash value noise) so
  there is one noise implementation shared CPU↔GPU.
- **`turbAmp_jet`** — small fixed cross-funnel churn (design constant, not a UI knob) so the
  rails shimmer rather than reading as glassy.

Emissivity is clamped `≥ 0`. When `jetStrength = 0` the entire funnel branch is skipped and
`E_jet` never contributes (see §7).

## 5. Relativistic beaming — the one-sided look

The plasma has a bulk outflow 4-velocity directed **radially outward along ±z** with speed
`β_jet` (Lorentz factor `Γ = jetGamma`, default ≈ 5, β = √(1 − 1/Γ²)).

**Doppler factor.** For a photon whose local propagation direction has axial cosine
`mu = n̂ · ẑ` (from the geodesic momentum at the step; the top funnel uses `+ẑ`, the bottom
`−ẑ`, i.e. the sign of `z`), the relativistic Doppler factor of the *approaching-vs-receding*
emitter is
```
delta = 1 / ( Gamma * (1 - beta * mu_toward_observer) )
```
where `mu_toward_observer` accounts for the fact that a backward-traced ray direction is the
reverse of the photon's travel direction toward the camera. Observed emissivity is boosted by
```
boost = delta ^ p_beam       // p_beam = 3 + alpha ≈ 3.5  (moving, optically-thin synchrotron)
```
This is the whole reason the jet is one-sided: at a general inclination one pole tilts toward
the camera (`δ > 1`, boosted) and the other away (`δ < 1`, `δ^{3.5}` drives it toward zero).
At the default 72° inclination one arm dominates strongly.

**Gravitational piece.** The step already carries the metric; the emission additionally
inherits the gravitational shift through the same geodesic that lenses/redshifts the disk.
For this phenomenological tier we fold gravity in via the local energy factor already
available from the metric (a `sqrt(-g_tt)`-style lapse) rather than a full transfer solution;
the dominant, characteristic effect is the special-relativistic beaming above.

**"Physical one-sided" (chosen):** full `p_beam` exponent, no counter-jet brightness floor —
the counter-jet realistically fades out.

## 6. Rendering integration — piggyback accumulation

In `raytrace.wgsl`'s existing per-step loop, **before** the disk-crossing `break`, add an
accumulation at the current point (using the same adaptive `dl` the integrator chose):

```
// optically-thin jet emission along the ray
let jetE = jetEmission(r, th, U.time);      // 0 when jetStrength==0 or outside funnel
jetAccum += jetTint * jetE * beaming * dl;  // jetTint = cool blue-white
```

The ray then continues exactly as today; the disk hit still `break`s (opaque). At loop end:
```
color = terminationColor + jetStrength * jetAccum;   // additive emissive volume
```
- Jet segments **in front of** the termination point (disk/horizon/escape) are integrated →
  visible glow.
- Jet segments **beyond** it are never reached → correctly occluded by the opaque disk and by
  the horizon.
- `jetAccum` is clamped to a sane ceiling before compositing to avoid single-ray blowout; the
  existing ACES tonemap + bloom then handle the HDR glow (bloom already reads the composited
  `accum`, no change needed).

**Sampling-rate note:** the integrator's far-field step grows large (`dl` up to ~6 M), which
under-samples the upper jet. The `lengthFalloff` and axial fade make the upper jet dim and
smooth, so coarse quadrature there is visually acceptable; the bright, structured base sits in
the fine-step strong-field region. No extra steps are added.

## 7. The features-off invariant & how it is enforced

`jetStrength = 0` must reproduce Tier 2A **bit-for-bit**. Enforcement:
- `jetEmission(...)` returns **exactly 0.0** when `jetStrength == 0` (early-out before any
  funnel math), and `color = terminationColor + jetStrength * jetAccum` reduces to
  `terminationColor` — the Tier 2A pixel path — with no FP perturbation.
- The regression gates are the existing headless routes with the jet off: `?parity`
  (CPU↔GPU metric/orbit/g-factor + turbulence, unchanged) and `?shadow` (unchanged radius),
  plus a new "jet off ⇒ identical" assertion.

## 8. Uniforms & parameters (append-only)

Extend the uniform buffer **append-only**, exactly as Tier 2A did (current size 80 bytes):
- `jetStrength: f32` — master brightness / on-off (**0 ⇒ Tier 2A**). Default 1.0.
- `jetGamma: f32` — bulk Lorentz factor Γ (beaming strength). Default 5.0.
- `jetLength: f32` — `z_max` in M. Default 60.
- `jetKnots: f32` — knot animation amplitude. Default 0.7.

Packed after `nSpots` at the next free offsets; new size 96 bytes (4 × f32). WGSL `Uniforms`
struct gains the four fields in the same order; `packUniforms` and `UniformValues` extended to
match. All other constants (`rho0`, `slope`, `q_peak`, `w_wall`, `kz`, `v_knot`, `p_beam`,
`jetTint`, `turbAmp_jet`, `z_base`) are design constants shared between `jet.ts` and the WGSL
twin — not uniforms.

## 9. Module plan (mirrors Tier 2A)

- **`src/physics/jet.ts`** (new, pure TS): `funnelEdge(z)`, `wallProfile(rho, z)`,
  `knots(z, t, timeScale)`, `dopplerBoost(mu, gamma)`, `jetEmission(r, th, t, ...)` — the CPU
  reference, unit-tested, with the features-off early-out returning `0`.
- **`src/render/raytrace.wgsl`**: WGSL twins (`funnelEdgeJ`/`wallJ`/`knotsJ`/`boostJ`/
  `jetEmissionJ`), the per-step accumulation, and the final additive composite.
- **`src/render/uniforms.ts`**: append the four fields; update `UNIFORM_*` sizes and `pack`.
- **`src/main.ts`**: jet state (`jetStrength, jetGamma, jetLength, jetKnots`), feed into the
  uniform each frame; the jet also advances on the existing `simTime`.
- **`index.html` + `src/main.ts`**: **Jet** (strength), **Speed** (Γ), **Knots** sliders.
- **`src/physics/jet.test.ts`** (new): unit tests incl. the features-off (`=0`) gate,
  funnel-membership, limb-brightening peak location, monotonic beaming vs. μ.
- **`src/test/parity.browser.ts`**: add jet CPU↔GPU parity cases (funnel edge, wall profile,
  knots, Doppler boost) alongside the existing metric/turbulence rows.

## 10. Testing strategy

- **CPU unit tests** (`jet.test.ts`): geometry (inside/outside funnel, wall peak at
  `q ≈ q_peak`), knots traveling-wave sign, Doppler monotonicity (`δ` increases as `μ →` toward
  observer; `δ^p` collapses the counter-jet), and **`jetEmission = 0` when `jetStrength = 0`**.
- **GPU parity** (`?parity`): CPU↔GPU agreement of the jet twins to the existing ~1e-3
  tolerance, added as new rows so the harness covers geometry + emission + beaming.
- **Features-off regression** (`?shadow`, `?parity` with jet off): unchanged numbers ⇒ Tier 2A
  preserved bit-for-bit.
- **Manual visual** (dev server): one bright beamed jet arm with a near-invisible counter-arm
  at 72°; knots visibly propagate outward; limb-brightened rails; disk correctly occludes the
  jet behind it; dragging inclination swaps which arm dominates; Jet = 0 looks exactly like
  Tier 2A. A two-frame diff confirms outward knot motion.

## 11. Risks & mitigations

- **Blowout from `δ^{3.5}` on near-axis rays** → clamp `jetAccum` and cap `δ` before the power;
  tune `jetTint` magnitude so the boosted arm sits within tonemap range.
- **Under-sampled upper jet from coarse far-field `dl`** → accepted; upper jet is dim/smooth by
  `lengthFalloff`. If it aliases, gate jet accumulation to `dl ≤ dl_jetMax` near the base only.
- **CPU/GPU divergence** in the shared value-noise → reuse the Tier 2A `vnoise`/`ihash`
  verbatim (already parity-verified) rather than a new hash.
- **Performance regression** → the added per-step work is a handful of ops behind an early-out;
  verify fps with `verify:gpu` and the interactive view stays responsive.
