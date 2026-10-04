# Hot accretion flow and the 1.3 mm (EHT) view — design

**Date:** 2026-10-04
**Status:** design approved in conversation; this document awaits review
**Why:** the app draws every object as a thin Novikov–Thorne disk in visible light. Sgr A* and M87* are hot,
geometrically thick, radiatively inefficient flows (RIAFs) whose horizon-scale emission is millimetre synchrotron —
what the Event Horizon Telescope images. (It is also the prerequisite the shelved eruption flares need.)

## 1. Intent (agreed)

- A **band switch**: "Visible" (today's renderer, unchanged) and "1.3 mm (EHT)".
- In the mm view, objects **below 1 % of Eddington** (the hot-flow regime) show a **hot flow** instead of the thin disk;
  thin-disk objects show their disk's (faint, Rayleigh–Jeans) mm emission; the jet is shown at 230 GHz where the model
  holds there (§2.5). The visible view does not change for any object.
- **Approach A:** a published semi-analytic RIAF, its density normalisation **calibrated to the measured 230 GHz flux**,
  its **ring diameter a prediction** checked against EHT.
- False colour as the EHT shows it: brightness temperature in an afmhot-style map; the panel shows the image's flux in
  Jy for objects with a known distance.

## 2. Physics

### 2.1 Flow model (Broderick et al. 2011a, 2016; dynamics from Pu, Akiyama & Asada 2016)

r_S = 2M; ρ, z cylindrical; r spherical (Boyer–Lindquist).

- Thermal electrons: n_e = n₀ (r/r_S)^−1.1 exp(−z²/2ρ²).
- Electron temperature: T_e = T₀ (r/r_S)^−0.84, T₀ = 10¹¹ K (the prototype showed the ring insensitive to T₀
  between 1 and 1.6 × 10¹¹ K once n₀ is recalibrated).
- Field: toroidal, β = 10 equipartition with the ions: B²/8π = n_e m_p c² r_S / (12 r β).
- **Velocity (Pu et al. 2016 eqs. 1–3, α = β_Ω = 0.5):** u^r = u^r_K + 0.5 (u^r_ff − u^r_K),
  Ω = Ω_K + 0.5 (Ω_ff − Ω_K), with the Keplerian profiles (circular outside the ISCO; inside, the equatorial plunge on
  the ISCO's energy and angular momentum) and the zero-angular-momentum free fall (u_t = −1, u_φ = 0; Ω_ff = ω), all
  as functions of r (equatorial metric, "R = r"); u^t from the local metric,
  (u^t)² = (1 + g_rr (u^r)²) / K₀, K₀ = −(g_tt + 2Ω g_tφ + Ω² g_φφ); no emission where K₀ ≤ 0. (Without the
  sub-Keplerian dynamics the predicted ring swung from 44 to 73 µas with spin; with them it is 48–55 µas — §5.)
- Not modelled: the non-thermal electron population (needed only below ~43 GHz), polarisation, scattering by the
  interstellar medium, time variability of the flow.

### 2.2 Emission and transfer at 230 GHz

- Thermal synchrotron, isotropic fit (Mahadevan et al. 1996 / Leung et al. 2011):
  j_ν = n_e e² ν M(X) / (2√3 c θ_e²), X = 2ν/(3 ν_c θ_e²), ν_c = eB/(2π m_e c), θ_e = k T_e/(m_e c²),
  M(X) = 4.0505 X^−1/6 (1 + 0.40 X^−1/4 + 0.5316 X^−1/2) exp(−1.8899 X^1/3).
- Absorption by Kirchhoff's law with the Rayleigh–Jeans source function, α_ν = j_ν / (2ν² k T_e / c²).
- Plasma-frame frequency ν' = D ν, D = u^t p_t + u^r p_r + u^φ p_φ (camera-normalised, past-directed covariant p, as
  the jet's `plasmaShiftJ`); per sample a uniform slab of plasma-frame path r_g D dλ:
  I += (j'/D³) ds (1 − e^−Δτ)/Δτ · e^−τ, τ += α' ds (the jet's `jetSlabJ`).
- Hot-flow region: r < 50 M, sampled every 0.25 M along the ray (the jet's quadrature step); no disk termination.

### 2.3 Calibration and scaling

- n₀ per preset so the traced 230 GHz flux equals the measurement: **Sgr A*** 2.4 Jy at 8.2 kpc (EHT 2022);
  **M87*** 0.5 Jy (compact flux, EHT 2019) at 16.8 Mpc. Prototype (Sgr A*, 96²): n₀ = 1.5 × 10⁷ cm⁻³ — the published
  RIAF value (~10⁷).
- Custom objects below 1 % Eddington: n₀ scaled from Sgr A*'s as density ∝ Ṁ/(r_g² c), i.e. n₀ ∝ λ/M. The M87*
  calibration is compared with that scaling and reported.
- Display: brightness temperature T_b = I_ν c²/(2 ν² k) — independent of distance, so custom objects need none;
  the Jy readout appears only for presets with a distance.

### 2.4 Validation targets (predictions, not fitted)

| Object | EHT ring diameter | Prototype (96², unblurred / 20 µas blur) |
|---|---|---|
| Sgr A* (preset a = 0.94, i = 30°) | 51.8 ± 2.3 µas (EHT 2022) | 48.2 / 46.0 µas |
| Sgr A* at a = 0.1, i = 60° (Broderick 2016 fit) | — | 55.0 / 55.0 µas |
| M87* (preset a = 0.9, i = 17°) | 42 ± 3 µas (EHT 2019) | 35.6 / 27.3 µas (n₀ = 5.0 × 10⁵ cm⁻³; θ_g = 3.82 µas/M, EHT 3.8) |

**Known limitation (measured while designing):** at M87*'s nearly face-on 17° the flow between the observer and the
horizon fills the shadow and pulls the brightness peak inward: −2.1σ unblurred, ~−5σ blurred. MAD flows (EHT's
favoured M87* model) compress the flow height near the horizon (McKinney et al. 2012, noted by Pu et al. 2016), which
this analytic model omits. Documented in the app's M87* caption and the README; Sgr A* is the validation gate.


### 2.5 The jet at 230 GHz (added during Task 3)

The jet's emissivity (cooled power law, synchrotron kernel above γ = 10, table x = ν′/ν_B from 1e-4 to 1e10) holds
where ν′ is above the gyrofrequency. At 230 GHz that is true for M87* (base field ≤ 257 G: x ≈ 320 at the base,
16–9e4 over the visible jet and D = 0.05–20). It is not for the X-ray binaries: Cyg X-1's base field ~1.6e8 G and GRS
1915+105's ~9e8 G give x ≈ 5e-4 and 9e-5 at the base (down to 2.6e-5 and 4.4e-6), below every electron's fundamental
(x = 1/γ) and below the table, where the kernel's x^(1/3) tail would invent the emission. Their observed mm jets come
from ~10⁴ r_g and beyond, outside this 60 M frame. Rule: in the mm view the jet is drawn only when 230 GHz exceeds the
gyrofrequency of the strongest field in the jet (x > 1 at D = 1 at the base wall); otherwise it is off and the caption
says why. Applies to custom objects by the same test.
## 3. Rendering and code

- `src/physics/hot-flow.ts` (CPU twin): flow profiles, velocity field, D, j/α, n₀ table and scaling, the hot-flow
  regime test (λ < 0.01).
- `scripts/calibrate-hotflow.ts`: traces Sgr A* and M87* at 96² with the CPU integrator, solves n₀ for the flux,
  measures the ring (azimuthal-mean peak, unblurred and 20 µas-blurred), prints the table.
- `src/render/emission-shared.wgsl`: the WGSL twin (sole copy, prepended to the renderer and `?parity`).
- `src/render/raytrace.wgsl`: band switch; in mm mode hot-flow objects integrate the flow (no disk termination),
  thin-disk objects shade the disk at 230 GHz (Rayleigh–Jeans blackbody at g T); the jet at 230 GHz (its band loop
  evaluated at one frequency). `present.wgsl`: mm display (T_b → afmhot × exposure).
- Geodesic cache: the band and the regime enter the geometry key (disk termination changes the path set). In mm
  hot-flow mode the build pass stores each pixel's integrated steady-flow intensity; cached frames display it (the jet
  replays as now).
- Uniforms: band, hot-flow flag, n₀ (cm⁻³), r_g already present; the block grows (layout test updated).
- UI: the band control; a "Flux (230 GHz)" readout for presets with distance; preset captions updated (the M87* and
  Sgr A* captions currently say the hot flow is not modelled).

## 4. On screen

Switching Sgr A* or M87* to "1.3 mm (EHT)" shows the asymmetric glowing ring of the EHT images, brighter on the side
where the flow approaches, with the shadow inside; M87*'s jet base appears faintly at 230 GHz. Thin-disk objects in
the mm view are nearly dark (their disks emit little at 1.3 mm); Cyg X-1's and GRS 1915+105's jets are not drawn
there (§2.5).

## 5. Tests and gates

**CPU (`npm test`):** M(X) against Mahadevan's tabulated values; Kirchhoff consistency; Pu velocity limits
(α = β = 1 → Keplerian, 0 → free fall), u normalised (u·u = −1 where K₀ > 0), D = 1 for a static emitter far away;
density/temperature/field formulas; the regime threshold; n₀ scaling ∝ λ/M. **Calibration gate** (SWEEP-style,
48²): traced flux reproduces 2.4 Jy (Sgr A*) and 0.5 Jy (M87*) at the tabulated n₀ within 5 %, and the ring
diameter of Sgr A* (unblurred) lies within 2σ of EHT's 51.8 ± 2.3 µas; the M87* ring is reported against
42 ± 3 µas, not gated (§2.4 limitation).

**GPU (`npm run verify:gpu`):** `?parity` cases for the flow functions and the per-sample slab (a mutation fails);
golden: two new mm scenes (Sgr A*, M87*), visible scenes unchanged (or the known compiler rescheduling, proven with
the stubbed-kernel check); `?cachecheck` gains an mm scene (cached = live); app check: band switch at Sgr A* shows a
ring with the flux readout within 5 % of 2.4 Jy, no console warnings; performance of live mm frames measured and
reported (bench).
