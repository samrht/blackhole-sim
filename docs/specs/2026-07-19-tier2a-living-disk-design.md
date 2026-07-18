# Tier 2A — Living Disk: Time Evolution, Turbulence & Orbiting Flares

**Date:** 2026-07-19
**Status:** Design (brainstorming deliverable) — awaiting user review
**Scope:** Sub-project A of the Tier 2 upgrade. Adds real-time animation, azimuthal disk
structure, and orbiting relativistic hot-spots to the existing Tier 1 renderer.
**Depends on:** Tier 1 (complete & verified). Physics spec:
`docs/specs/2026-05-30-relativistic-blackhole-accretion-design.md` (§3, §5, §6, §8).

---

## 1. Motivation & the core insight

Tier 1 renders a **static, axisymmetric** image. The disk emission in `raytrace.wgsl`
depends only on the hit radius `rHit` (via the temperature LUT), never on azimuth `φ`.

**Consequence that drives this whole design:** a perfectly axisymmetric disk looks
*identical* under rotation. Simply advancing time and rotating the disk produces **no
visible motion**. To make the disk appear to move, the emission must carry **azimuthal
structure** (turbulence + discrete features) that is then advected at the local orbital
angular velocity `Ω(r)`.

Therefore "make it move" and "flares & hot-spots" are one coupled feature, delivered
together here. Sub-project B (jet) and the dropped Σ-diffusion are out of scope (§8).

## 2. Goals / non-goals

**Goals**
- The disk visibly **rotates** at the physically correct differential rate — inner annuli
  sweep faster than outer, per `Ω(r) = ±√M / (r^{3/2} ± a√M)` (spec §3).
- Azimuthal **turbulence**: multi-octave procedural brightness modulation sheared by
  differential rotation, so the disk reads as churning gas, not a painted ring.
- A small number of **orbiting hot-spots** (Sgr A*-style flares) advected at local `Ω`,
  automatically Doppler-beamed by the *existing* g-factor (brighter on the approaching
  side, dimmer receding).
- Optional slow, large-scale **brightness "breathing"** to suggest secular disk change
  (stands in for the cut Σ-diffusion; purely phenomenological).
- Remains **real-time interactive** on the target RTX 3050: ~15–30 fps, camera drag and
  parameter tweaks stay live while the scene animates.
- Play/pause and time-scale control so a still frame can still be inspected.

**Non-goals**
- Literal viscous Σ-diffusion PDE (cut — invisible on real-time timescales; see §8).
- The jet (Sub-project B, separate spec).
- Absorption / full radiative transfer (Tier 3).
- Physically-calibrated flare statistics — hot-spots are phenomenological set-dressing
  consistent with the g-factor physics, not a fit to observed light curves.

## 3. Physics model

### 3.1 Pattern phase (advection)
When a ray crosses the disk plane it yields `(rHit, φ_hit)` — the Boyer–Lindquist azimuth
of the emitting matter. Disk gas on a circular geodesic orbits at coordinate angular
velocity `Ω(rHit)` (prograde). The *co-rotating pattern phase* is

```
ψ(rHit, φ_hit, t) = φ_hit − Ω(rHit) · (t · timeScale)
```

Any emission feature is a function of `(rHit, ψ)`. Because `Ω` decreases with radius, a
radial spoke shears into a trailing spiral over time — the correct differential-rotation
signature. `t` is coordinate time in units of `M`; `timeScale` is a UI-controlled
dimensionless multiplier (default gives a visually pleasant ~seconds-per-inner-orbit).

> Note on light travel time: a fully correct treatment uses the photon's arrival (retarded)
> time per hit. Tier 2A uses the observer coordinate time `t` uniformly across the image —
> a standard, visually faithful simplification for a thin disk at these radii. Documented
> here so it is a known, deliberate approximation, not an oversight.

### 3.2 Emission field `E(rHit, ψ)`
A dimensionless brightness multiplier applied to the existing thermal emission. Composed of:

1. **Turbulence** — 2–3 octaves of value/simplex noise over `(log rHit, ψ)`, giving
   filament/eddy structure that shears with `ψ`. Contrast controlled by a `turbAmp`
   parameter; mean of the field ≈ 1 so total flux is roughly conserved.
2. **Hot-spots** — `N` (≈2–4) Gaussian bright blobs, each with a fixed `(r_k, ψ_k, σ_k,
   amp_k)`. Rendered brightness at a hit is `Σ_k amp_k · exp(−[(rHit−r_k)² + (r_k·Δψ)²] /
   2σ_k²)` where `Δψ` is the angular separation on the ring. Because each blob sits at a
   fixed `ψ` in the co-rotating frame, it orbits in the observer frame at `Ω(r_k)`.
3. **Breathing (optional)** — a slow global factor `1 + breatheAmp · sin(2π t / T_breathe)`
   for secular variation. Default `breatheAmp = 0`.

Final emitted (pre-redshift) brightness scale: `E = turb(rHit,ψ) · breathe(t) + hotspots(rHit,ψ)`.

### 3.3 Coupling to existing relativistic optics (unchanged)
The g-factor path is **untouched**. `E` multiplies the beamed blackbody radiance already
computed at the hit:

```
color = sampleColor(Tobs) · pow(g·Tn, 4) · E
```

So hot-spots and turbulent filaments are Doppler-beamed and gravitationally redshifted for
free — the physics that makes the approaching side brighter applies to the features too.
No change to geodesic integration, horizon capture, or the temperature LUT.

## 4. Rendering-model change: static accumulation → temporal EMA

Tier 1 converges by **summing** samples into `accum` forever and resetting on input
(`accum[idx] = prev + color`, divided by sample count at present time). A moving scene
can never converge this way. We switch to a **temporal exponential moving average**:

```
accum[idx].rgb = mix(accum[idx].rgb, color, blend)   // blend ≈ 0.15 when animating
```

- **Animating (playing):** `blend ∈ [0.1, 0.25]` — denoises over a short trailing window
  while tracking motion. The present pass divides by 1 (EMA already normalized), or we
  store normalized radiance directly.
- **Paused / static (Tier 1 behavior):** fall back to progressive averaging so a still
  frame converges to full Tier 1 quality (`blend = 1/(frame+1)` reproduces the running
  mean). This preserves the existing `?shadow`/`?parity` still-image quality.
- **Hard reset** (camera drag, spin/inclination change, resize): clear `accum`, restart —
  same triggers as today.

This means the same buffer serves both modes; only the `blend` weight differs. New uniform
`blend: f32` replaces the implicit `1/(frame+1)` in the present pass.

## 5. Module & interface plan

### New: `src/physics/emission.ts` (pure TS, unit-tested — mirrors the CPU/GPU pattern of the physics core)
```ts
// Deterministic value-noise turbulence over (logR, psi). Hashable, matches WGSL bit-for-bit
// closely enough for the parity harness tolerance.
export function turbulence(logR: number, psi: number, octaves: number): number;

// One hot-spot's contribution at a disk hit, in the co-rotating frame.
export interface HotSpot { r: number; psi: number; sigma: number; amp: number; }
export function hotspotField(rHit: number, psi: number, spots: HotSpot[]): number;

// Full emission multiplier (turbulence * breathe + hotspots), clamped >= 0.
export function emissionField(rHit: number, psi: number, t: number, p: EmissionParams): number;

// Pattern phase psi = phi - Omega(r) * (t * timeScale).  Omega from orbits.ts.
export function patternPhase(rHit: number, phiHit: number, tScaled: number, a: number): number;
```
`omegaKepler` already exists in `orbits.ts` (reused, not duplicated).

### New: `src/render/emission.wgsl` (or inlined into `raytrace.wgsl`)
WGSL twins of `turbulence`, `hotspotField`, `patternPhase`, `emissionField`. Hot-spot
parameters uploaded as a small uniform/storage array. Must match the TS within the parity
harness tolerance (extends the existing `?parity` route with emission-field rows).

### Modified files
- `raytrace.wgsl` — capture `φ_hit` at the crossing; compute `ψ`; multiply `color` by
  `emissionField`; swap accumulation for EMA. New uniforms: `time` (already present),
  `blend`, `timeScale`, `turbAmp`, `breatheAmp`, plus hot-spot array binding.
- `present.wgsl` — normalize by EMA (divide by 1) instead of by sample count in EMA mode.
- `uniforms.ts` — add `blend`, `timeScale`, `turbAmp`, `breatheAmp`; add hot-spot buffer.
  Bump `UNIFORM_FLOATS` and the packing (mind the 16-byte alignment already handled).
- `gpu.ts` — create/bind the hot-spot storage buffer; expose an upload method.
- `main.ts` — animation state (`playing`, `timeScale`); feed real elapsed time into `time`;
  choose `blend` per mode; wire new UI controls; keep hard-reset triggers.
- `index.html` — UI: play/pause button, time-scale slider, flare-intensity slider,
  turbulence-amount slider (breathing optional/advanced).

### Unchanged
`kerr.ts`, `geodesic.ts`, `redshift.ts`, `disk.ts`, `color.ts`, `lookups.ts`, and the
geodesic/g-factor core of `raytrace.wgsl`.

## 6. UI additions
- **Play / Pause** — toggles animation; paused falls back to progressive convergence.
- **Time scale** — how fast coordinate time advances (log slider).
- **Flare intensity** — hot-spot amplitude (0 = off).
- **Turbulence** — `turbAmp` contrast of the churning field.
- Existing spin / inclination / exposure controls and camera-drag unchanged.

## 7. Verification & testing

**Unit (Vitest, CPU — extends existing suite):**
- `emission.test.ts`:
  - `patternPhase` reduces to `φ_hit` at `t=0`; advances by `−Ω(r)Δt`; inner radius phase
    advances faster than outer (differential rotation sign check).
  - `turbulence` deterministic, bounded, mean ≈ expected over a ψ sweep.
  - `hotspotField` peaks at the spot center, decays to ~0 beyond a few σ, is periodic in ψ.
  - `emissionField ≥ 0` and reduces to `1` when turbAmp=0, no spots, breatheAmp=0
    (i.e. Tier 1 emission recovered exactly → guarantees no regression when features off).

**GPU parity (`?parity` route):** add emission-field rows — CPU `emissionField` vs WGSL
twin at sampled `(r, ψ, t)` within existing tolerance (`maxErr < 1e-3` structure; noise
may need a looser, documented tolerance).

**Structural / visual:**
- `?shadow` route still PASSes (features-off path must equal Tier 1) — this is the key
  regression gate: with turbAmp=0 and no spots, output must be bit-comparable to Tier 1.
- Manual: with features on, confirm (a) disk visibly rotates, inner faster; (b) a hot-spot
  brightens as it sweeps to the approaching side and dims receding (g-factor coupling);
  (c) camera drag and spin changes still hard-reset and stay responsive; (d) frame rate
  stays in the ~15–30 fps band on target hardware.

## 8. Out of scope (explicit)
- **Literal Σ-diffusion PDE — cut.** Viscous timescale `~ t_orb/(α·(H/r)²)` is thousands
  of orbits; invisible in real time without absurd acceleration, and only a slow global
  brightness redistribution even then. The optional "breathing" factor stands in for any
  sense of secular change. Decision recorded during brainstorming.
- **Jet (Blandford–Znajek, synchrotron)** — Sub-project B, its own spec/plan next.
- **Retarded-time per-hit light-travel correction** — deliberate uniform-`t` approximation
  (§3.1).
- **Absorption / GRRT, evolving GRMHD** — Tier 3.

## 9. Risks & mitigations
- **Noise looks like static, not gas.** Mitigate: shear the noise domain by ψ and radius,
  low octave count, animate slowly; tune `turbAmp` default conservatively.
- **EMA smearing / ghosting of fast inner features.** Mitigate: `blend` toward the higher
  end (~0.25) when playing; expose if needed. Inner disk moves fast — accept some motion
  blur as physically plausible.
- **Frame budget.** Emission field is a handful of noise octaves per disk hit (one hit per
  ray) — cheap relative to geodesic integration. Hot-spots are `N`≈4 Gaussians. Low risk;
  measure on target GPU during implementation.
- **Parity tolerance for noise.** Bit-exact hashing across TS/WGSL is fragile; document a
  looser emission tolerance and keep the *features-off equals Tier 1* test as the strict gate.
