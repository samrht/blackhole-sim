# Physical disk turbulence (MRI) — design

**Date:** 2026-10-03
**Status:** approved in conversation; awaiting written-spec review
**Roadmap:** `docs/ROADMAP.md`, Physics features. Replaces the item "Evolving alpha-disk (surface-density
diffusion)": a thin disk's viscous time, t_visc ≈ [α (h/r)²]⁻¹ orbital times ≈ 3 × 10⁶ M at r = 10 M (α = 0.1,
h/r = 0.01), takes ~9 hours to play even at the fastest Motion (100 M/s), so surface-density evolution is static on
screen at every mass. What changes on screen is turbulence, on the orbital time — and that is what is still
decorative.

## 1. Intent (agreed)

- Replace the decorative disk noise with turbulence whose **size, lifetime, shape and statistics are the
  measured properties of magnetorotational (MRI) turbulence** in thin disks.
- **Amplitude from observation:** the disk's intrinsic integrated light flickers at the rms measured for thin,
  thermally dominated disks (2 %; Schnittman, Krolik & Hawley 2006's ray-traced simulation gives ≲ 2 % above 10 Hz,
  and soft-state X-ray binaries show a few per cent). The slider becomes "Disk flicker (rms)", 0–10 %, default 2 %.
- **Energy conservation:** turbulence redistributes the disk's light; the time-averaged luminosity stays the
  Novikov–Thorne value.
- Persistent orbiting hot spots ("Flares") and the slow "breathing" stay available, **off by default**, labelled as
  illustrative (not MRI turbulence). The light-delay echo demo can switch a hot spot on.
- Not in scope: viscous surface-density evolution (see above), vertical structure, a disk corona, the jet's own
  turbulence noise (unchanged).

## 2. Physics

### 2.1 Measured properties (the targets)

| Property | Value | Source |
|---|---|---|
| Structure size | δ(ln r) ≈ 0.3, δφ ≈ 25°, self-similar over 3–25 M | Schnittman, Krolik & Hawley 2006 (SKH06) |
| Lifetime | exponential distribution, mean 0.3 T_orb(r) | SKH06 (eq. 36) |
| Shape | trailing spirals, Δφ ≈ 0.9π ln r in a snapshot | SKH06 (eq. 38); global thin-disk runs agree (Beckwith, Armitage & Simon 2011) |
| Statistics | multiplicative fluctuations → lognormal | Hogg & Reynolds 2016 |

The spiral is not an independent input: Keplerian shear acting on a feature for one lifetime winds it by
(3/2) Ω × 0.3 T_orb = 0.9π radians per e-fold of radius. A model with the right size, lifetime and shear reproduces
it; the current model's features never die, so their winding grows without limit as the animation runs (the disk
turns into ever-tighter stripes) — the defect this design removes.

### 2.2 The field

- **Unit Gaussian lattice field** on (η = ln r, φ): independent standard-normal values at lattice points (hashed
  uniform pairs through Box–Muller), interpolated with smoothstep weights w and renormalised by √(Σ w²), so the value
  at every point is exactly unit-variance Gaussian. Two octaves (the second at half the cell size, weight ½,
  renormalised), 2π-periodic in φ (integer cell count around the ring, as the 2026-10-01 seam fix).
- **Generations:** each radius runs a clock τ = t / T_c(r) with T_c ∝ T_orb(r) = 2π/Ω_K(r, a) (Kerr Keplerian).
  Generation j is born at t = (j − 1) T_c, fades in over one cell, fades out over the next, and is advected by
  Keplerian rotation from its birth: it is sampled at φ_j = φ − Ω_K (t − (j − 1) T_c), with its own seed. In cell
  k = ⌊τ⌋ with phase f = τ − k: g = cos(πf/2) g_k + sin(πf/2) g_{k+1} (unit variance; continuous across cells and
  across radius). Features therefore only ever wind forward (trailing), by at most two cells of shear.
- **Calibration of the clock and lattice:** T_c is set so the field's integral autocorrelation time following
  the flow equals 0.3 T_orb; the lattice cell sizes so its autocorrelation e-folding lengths are δη = 0.3 and
  δφ = 25°. Both constants are measured on the CPU twin and pinned by tests.
- **Emission:** the Novikov–Thorne surface brightness times **exp(σ g − σ²/2)** — lognormal with mean exactly 1.
  Then × (breathing) + (hot spots) as today (both default 0).
- **σ from the flicker target:** the intrinsic (face-on, before relativistic beaming) rms of the disk's integrated
  light is ≈ κ σ for small σ; κ is measured on the CPU twin (a = 0.9, default disk) and stored as a constant;
  σ = flicker / κ. Relativistic beaming then raises the visible flicker with inclination, as SKH06 found.

## 3. Interfaces

- `src/physics/emission.ts`: the Gaussian lattice field, `turbulenceAt(r, φ, t, a)` → g, the constants (sizes, clock,
  κ), `emissionField(r, φ, t, a, σ, breatheAmp, spots)`; hot spots keep their co-rotating pattern phase. The jet's
  `vnoise` is unchanged.
- `src/render/emission-shared.wgsl`: the sole WGSL copy; `raytrace.wgsl`'s disk shading passes (r, φ_hit, t_emit, a).
  The geodesic cache already stores r, φ and the delay: shading-only, no rebuild.
- `src/render/turb-parity.wgsl` + `?parity`: the new field, CPU vs GPU.
- Uniform `turbAmp` carries σ (no layout change).
- Panel: "Disk flicker (rms)" 0–10 % (default 2 %); Flares default 0; Flares/Breathing labelled illustrative.

## 4. Error handling

- Box–Muller input guarded away from 0; the field is finite everywhere; exp(σ g − σ²/2) with σ ≤ the slider's
  maximum stays far from f32 overflow. The `accum` choke point remains.

## 5. Testing

- Field statistics: mean 0, variance 1, kurtosis 3 (±5 %); emission factor mean 1 (±1e-3); 2π-periodic in φ.
- Measured on the CPU twin: correlation lengths 0.3 (η) and 25° (φ) within 10 %; integral lifetime following the
  flow 0.3 T_orb within 10 %; emergent snapshot spiral tilt dφ/dη ≈ 0.9π within 20 %; statistics after 1000 orbits
  equal those after 1 (no runaway winding).
- Calibration: intrinsic integrated rms at the default slider 2.0 % ± 0.2 %.
- GPU: `?parity` on the new field; `?golden` re-recorded (every scene uses turbulence); `?cachecheck`; app checks;
  probes; screenshots.
