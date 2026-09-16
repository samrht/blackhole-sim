# blackhole-sim

A physically accurate, real-time renderer of a Kerr (spinning) black hole's accretion disk, running entirely in the browser on WebGPU.

This isn't a stylized visualization — it's a real general-relativistic ray tracer. Every pixel backward-integrates a null geodesic through curved Kerr spacetime, terminating on the event horizon (shadow), the accretion disk (emission), or escaping to the background. Disk brightness and color come from real physics: Novikov–Thorne relativistic disk flux, combined gravitational + Doppler redshift, and blackbody emission mapped through actual CIE color-matching functions into sRGB — not a hand-tuned color ramp.

## Features

- Exact Kerr metric (Boyer–Lindquist), horizons, ergosphere, frame dragging
- Hamiltonian geodesic integration (adaptive RK4) with conserved quantities (E, L_z, Carter constant)
- ISCO, photon orbit, and marginally bound radii for arbitrary spin
- Novikov–Thorne / Page–Thorne relativistic disk flux (correct zero-torque inner boundary — flux peaks just outside the ISCO, not at it)
- Combined gravitational + Doppler redshift (g-factor), producing physically correct approaching/receding disk asymmetry
- Blackbody emission → CIE XYZ → linear sRGB (Wyman et al. color-matching fit) — color comes from real temperature, not an artist's gradient
- HDR accumulation with ACES tonemapping and progressive anti-aliasing
- Interactive controls: spin (a*), inclination, exposure — drag the canvas to tilt the camera live

## Requirements

- A WebGPU-capable browser: **Chrome/Edge 113+** or **Safari 18+**
- A dedicated GPU is strongly recommended. This was scoped to Tier 1 of a 3-tier physics spec specifically to run on modest hardware (target: NVIDIA RTX 3050 Laptop, 4GB), with progressive frame accumulation capped around 15fps to stay responsive — full GR ray tracing is inherently expensive per pixel, so performance on integrated graphics may still be limited.

## Quickstart

```bash
npm install
npm run dev       # http://localhost:5173
npm test          # run the physics unit test suite (Vitest)
npm run build     # production build
```

Validation routes (append to the dev URL):
- `?parity` — CPU↔GPU parity check for the core physics math, including the shadow-edge classifier, the camera mapping and the constraint-monitored integrator step, all compiled from the same shared WGSL fragments the renderer uses (52 cases; the metric cases run the shipped `integrator-shared.wgsl` bytes too)
- `?shadow` — structural check that the Schwarzschild (a = 0) render has a centred dark region ringed by disk, with a plausible radius; the number it records is the lensed inner edge of its emitter, not the critical curve (see Status), and is a regression gate rather than a √27 M assertion
- `node scripts/probe-axis.mjs` — visual check of the polar-axis fix: drives the sliders to a = 0, sky and jet off, at i = 72° and 8°, and asserts the column through the pole is no darker than a column beside it (fails on the pre-fix shader); writes `axis-i72.png` / `axis-i8.png`

## Architecture

The physics core (`src/physics/`) is pure TypeScript with zero DOM/GPU dependencies, unit-tested in isolation against known analytic results (Schwarzschild horizon = 2M, ISCO = 6M/M/9M for a=0/extremal-prograde/extremal-retrograde, etc.). Its outputs are baked into 1D lookup tables (temperature and color) uploaded as GPU textures. A WebGPU compute shader (`src/render/raytrace.wgsl`) mirrors the core's math, tracing one geodesic per pixel and accumulating HDR radiance; a present pass resolves the accumulation buffer with exposure, ACES tonemapping, and gamma correction.

Full physics specification and derivations: `docs/specs/2026-05-30-relativistic-blackhole-accretion-design.md`

## Key references

Kerr (1963); Carter (1968); Bardeen, Press & Teukolsky (1972); Bardeen (1973); Novikov & Thorne (1973); Page & Thorne (1974); James, von Tunzelmann, Franklin & Thorne (2015, the Interstellar/Gargantua paper).

## Attribution

Milky-Way panorama: ESO/S. Brunier, CC BY 4.0 (eso0932a).

## Status

Tier 1 (single-GPU, real-time image) complete and verified.

**Tier 2A — Living Disk (shipped):** the accretion disk now evolves in real time. Differential rotation carries a co-rotating turbulence pattern (procedural value-noise over log-radius and pattern phase), orbiting Gaussian hot-spots are Doppler-beamed by the existing g-factor (brightening on the approaching side, dimming on the receding side), and a temporal EMA replaces the static progressive average so motion is smooth while a paused scene still re-converges to a clean still. Motion / Turbulence / Flares sliders and a Play/Pause control drive it live. With all features off the render is bit-for-bit identical to Tier 1 (`?parity` and `?shadow` unchanged).

**Tier 2B — Relativistic Jet (shipped):** a beamed, limb-brightened, animated bipolar jet. This is **phenomenological, not an MHD or emission calculation**: the funnel is an assumed parabolic surface (ρ = 0.6 + 0.7·√|z|), brightness is a Gaussian profile peaked near the wall, and the knots are advected value noise. There is no electromagnetic field, no Blandford–Znajek power extraction, and no synchrotron emissivity computed from a field and electron distribution — the shape is chosen to *resemble* published BZ-funnel images. The one piece with real physics content is the beaming: radiance is boosted by the Doppler factor to the power 3 + α (α = 0.5), the beaming exponent for a *discrete moving blob* radiating a p = 2 power-law synchrotron spectrum. Note a steady continuous jet would take 2 + α; the code applies 3.5 uniformly to a steady funnel with knots superposed, so this is the blob exponent used somewhat beyond its remit. The tint is a fixed RGB constant, not a computed spectrum. The jet is integrated along rays before disk accumulation, optically thin with additive composite. Jet / Jet speed Γ / Jet knots sliders control the strength, relativistic beaming γ-factor, and knot amplitude live. With the jet off, the render is bit-for-bit identical to Tier 2A (`?parity` and `?shadow` unchanged).

**Visual polish:** the default jet brightness/ceiling were retuned so the relativistically beamed jet reads with internal structure at low (pole-on) inclination instead of clipping to a solid white funnel.

**Polar axis (fix/polar-axis):** the Boyer–Lindquist polar axis (where g^φφ ∝ 1/sin²θ diverges) is handled physically rather than capped. `POLE_S2` is now a 1e-12 NaN guard in both twins. It was a 1e-3 floor, which did not make near-axis rays "terminate as captured" as this section used to claim: inside the 1.8° cone the floor removed the centrifugal barrier, θ ran through 0 to large negative values, the equatorial-crossing test never fired (the plane sits at θ = −π/2 on that branch), and the ray tunnelled through the axis, escaped, and rendered as background — the black wedge from each pole. The integrator now halves its step whenever the per-step drift of the null constraint H = ½ g^μν p_μ p_ν exceeds `H_TOL` = 1e-3 *relative to the magnitude of the cancelled terms* Σ|terms| (near the capture margin g^tt ~ 1e2, so an absolute tolerance would sit at f32 noise), up to `MAX_RETRY` = 8 halvings. Both constants come from a measured sweep (`tests/sweep-htol.test.ts`, run with `SWEEP=1`): 1e-3, 1e-4 and 1e-5 all pass the near-axis accuracy criterion (disk-hit radius within 0.02 M of a converged reference; the survivors measured 0.0097 M), 1e-2 fails it at 0.34 M, 1e-6 is below the f32 floor; the retry cap was never approached in f64, so 8 is headroom, not a tuned value. Mean steps per ray are unchanged (338.4 over 2048 sampled rays; retries 10 → 61). Two rulings made during implementation differ from the spec and the plan: the monitor is **off in the far field** (r > 1.5·rOut, the existing far-field step branch, `H_TOL_FAR` = 1e30) because the f32 finite-difference force at r ≈ 1000 is noise (ulp 6e-5 against an FD half-step of 1e-4), so monitoring there halved steps on noise at +35 % cost for no accuracy — the spec's "global" monitor is therefore near-field only; and a ray that exhausts the retries **proceeds with the smallest-step attempt** (the spec's literal wording) rather than breaking to the (ξ, η) classifier as the plan had it, because the classifier can only answer captured/escaped and painted a 1-px starfield seam over the disk hits on the α = 0 column (rays with |α| ~ 1e-3 exhaust the halvings — α = 0 and 0.01 did not under the old build, 1e-3 did; `tests/trace.test.ts` now checks α ∈ {0, 1e-4, 1e-3, 1e-2} all hit the disk within 0.02 M of α = 0). The rare ξ = 0 rays that genuinely reach the axis are continued analytically (θ → −θ, or 2π − θ at the south pole; φ → φ + π; p_θ → −p_θ), which is exact. The step controller, monitored step and axis crossing are gated CPU↔GPU by `?parity` on the shipped shader bytes (seven integrator cases: far field, strong-field equatorial, near horizon, near axis, a forced retry the harness refuses to let be vacuous, a barrier step, an axis reflection; `H_TOL` and `MAX_RETRY` are compared too). This changes pixels near the axis and, because the near field is monitored everywhere, marginally elsewhere; headless fps went 0.54 → 0.78 with the far-field exemption. `scripts/probe-axis.mjs` is the visual gate for the wedge (it counts void-class pixels, RGB sum < 80, on the axis column above the shadow; fails on the old shader at both inclinations, passes here) — it does not see the dimmer near-axis residuals described under limitation (a) below.

Known limitations of the fix, stated plainly: (a) a single f32 barrier step is only ~1e-2 accurate in p_θ (finite-difference force precision, not shipped-bytes drift — the parity "barrier" case compares retries and the ok flag but not the state, and documents the measured 2 % divergence), so the GPU's near-axis *trajectory* is gated indirectly — by the CPU sweep; the probe gates only the wedge (void-class pixels), not this band — not by the state parity. It is visible in `axis-i8.png`: at i = 8° with the jet off there is a dark, speckled streak on either side of the axis column above the shadow (column means ~55 % of their neighbours at x = 498 and 501, tapering out by ~row 130; static across captures; the exact axis columns are clean), plus a few lit pixels on the axis column inside the shadow. The interpretation — rays passing within ~1e-2 rad of the axis without exhausting the retries, so their f32 barrier step is accepted with its ~1e-2 error — is inferred from the f64 twin, not measured on the GPU. The in-shadow pixels are about six on columns 499–501 of a 1000-px frame, lit (RGB sums 20–570 at fixed rows, i.e. β ≈ 1.1–4.5 M, where the f64 twin captures every ray; on the old shader those rays tunnelled and read dark). At i = 72° neither is measurable (column means flat to ±1). (b) Far-field GPU forces are finite-difference noise — pre-existing since Tier 1, now documented and exempted from the monitor; the far parity case compares positions only. (c) With the jet on, the α = 0 column shows a thin bright line because on-axis rays now traverse the funnel lengthwise and the phenomenological wall profile has a small non-zero core emissivity; that is a jet-model follow-up, not an integrator seam (with the jet off the α = 0 column profile at i = 72° is monotone).

**Lensed sky background (shipped):** the procedural starfield is replaced by a real Milky-Way panorama (ESO/S. Brunier, CC BY 4.0) sampled along each escaped ray's gravitationally-bent direction, so the background warps around the shadow into a lensed ring (most pronounced near edge-on inclination). A "Sky" slider crossfades its brightness; at 0 (or if the asset fails to load) the render falls back to the original procedural void, so `?parity` and `?shadow` are unchanged. The default strength was subsequently retuned from 0.6 to 1.0 so the lensed background reads at the default 72° view without washing the sky into a flat haze at near pole-on inclinations.

**Photon-ring detail (shipped):** rays that exhausted the integrator's step budget previously fell
out of the loop still holding their initial black colour, so step-starved rays rendered as shadow —
an artifact that swallowed the n=1 photon subring (at a=0.9 one winding near the prograde photon
orbit r≈1.56M costs ~3900 steps against a budget of 1200). Exhausted rays are now classified by
their conserved impact parameters (ξ, η) against the analytic Kerr critical curve, and a "Detail"
slider exposes the step budget (default 4800). This does not fully resolve the ring: per-winding
step cost varies roughly 8x across the photon shell (at a=0.9, ~3900 steps at the prograde edge
r≈1.56M vs. ~500 at the retrograde edge r≈3.91M), so a budget of 4800 resolves the retrograde side
completely but leaves the innermost prograde part of the n=1 subring still budget-limited. n=2
subrings are not visible and cannot be at the default field of view — successive subrings are
thinner by a factor of roughly e^(−2π) ≈ 1/535, well under one pixel at the default `fovScale`;
seeing one would require a narrow-FOV zoom, which is out of scope — and a zoom alone would not
suffice, since n=2 is also bounded by integrator accuracy: the geodesic forces come from finite
differences (`h=1e-4` in raytrace.wgsl) rather than analytic Christoffels, giving ~1e-3 relative
force error per step in f32, and non-symplectic RK4 lets that drift secularly while a near-critical
spherical orbit amplifies it by the same e^(2π) ≈ 535 per winding, so the error reaches O(1) after
roughly 2–3 windings. Note this is the first feature
that does **not** preserve the project's bit-identical-when-off property: correcting the artifact
necessarily changes pixels near the shadow edge. `?parity` is math-only and is unchanged.

**What `?shadow` measures — a correction.** Earlier revisions of this section said the `?shadow`
route's rendered radius differed from the analytic critical curve by a "~0.87 camera-calibration
factor, not physics". That was a misdiagnosis. The route lights a flat emitter from the photon
orbit (rIn = 3 M, inside the capture region) outward and scans the centre column; verified on the
f64 CPU twin (a = 0, i = 10°, α = 0), rays with 4.2 ≤ β < 5.196 — η < 27, captured-class — cross
the equatorial plane at r = 3.05–4.02 M and register as disk hits *before* they reach the horizon
(on the β < 0 side even |β| = 3.9 hits at r = 3.3 M). So the dark span the route measures is the
lensed silhouette of the emitter's inner edge, not the critical curve, and the ratio it prints is
not a camera constant. The old 0.864 also contained the polar-axis tunnelling artefact (the centre
column is exactly the ξ = 0 column, whose far-side crossings the old floor hid); with the axis
handled physically the route reads **3.89 M, ratio 0.749** to the analytic 5.196 M, and that is
the self-consistent number. The route remains a valid structural smoke test (centred dark region,
ringed by disk, plausible radius) and its recorded numbers are a regression gate. A true
critical-curve gate — an emitter that stops outside the capture region, or a (ξ, η)
classification image — is a follow-up. (For what it is worth, the interactive view at a = 0,
i = 8° puts the shadow's top edge at 5.2 M by the same vertical scale, i.e. at √27; the camera
was never the problem.)

**Current gates (fix/polar-axis):** `npm test` 69 passed, 1 skipped (the sweep, run with
`SWEEP=1`); `?parity` PASS, maxRelErr 2.663e-4 over 52 cases — re-baselined from 9.690e-7 over 45
because the seven new integrator cases dominate (a full f32 RK4 step carries more error than a
metric component; the forced-retry case's p_r sets the maximum) and the 1e-3 threshold is
unchanged; `?shadow` PASS, 3.89 M / 0.749 (was 4.49 / 0.864, see above); `scripts/probe-axis.mjs`
PASS at i = 72° and 8°.

Out of scope: Tier 3 (full GRMHD, multi-GPU/offline); a critical-curve gate for `?shadow`
(follow-up, see above); Kerr–Schild or Cartesian-momentum handoff near the axis (no longer
needed for the wedge; would only address limitation (a) above).
