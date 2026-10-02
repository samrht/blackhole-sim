# Synchrotron jet (Tier 2) — design

**Date:** 2026-10-02
**Status:** implemented on `feat/synchrotron-jet` (plan `docs/plans/2026-10-02-synchrotron-jet.md`)
**Roadmap item:** `docs/ROADMAP.md`, "Physics features: Synchrotron jet emission"

## 1. Intent (agreed)

- Replace the hand-made jet (Gaussian wall, fixed blue tint, `JET_GAIN`, `JET_CEIL`, flat-space
  δ^3.5 "blob" beaming) with **synchrotron emission from a magnetic field and a power-law electron
  population, transferred along the existing Kerr geodesics in general relativity**.
- **Absolute brightness** tied to mass, spin and accretion rate, in the **same physical units as
  the disk**, so the jet-vs-disk ratio comes out of the physics. Per-preset jet calibration
  (M87\* 0.1/29.8) is removed.
- **Energy conservation sets the brightness.** The jet's visible-band synchrotron luminosity within
  the rendered region equals **ε × P_BZ** (its Blandford–Znajek power), with the **Jet radiative
  efficiency ε** slider (log 10⁻⁵–10⁻¹, default **2 × 10⁻³**) anchored to M87: optical nucleus
  νL_ν ≈ 10⁴¹ erg/s (synchrotron; Chiaberge et al. 1999, Prieto et al. 2016) against a jet power of
  10⁴³–10⁴⁴ erg/s (Stawarz et al. 2006; Rieger & Aharonian 2012). (Revised 2026-10-02: with uncooled
  power-law electrons, an equipartition or σ-based density makes the jet radiate more than its total
  power — the optical-emitting electrons cool in ~0.1 s against minutes of flow time — so the absolute
  density is set by the energy budget instead.)
- Not in scope: GRMHD (field and flow are prescribed, not simulated), polarisation, a time-dependent
  field, inverse-Compton, radio wavelengths (the renderer stays visible-band).

## 2. Physics

All in Gaussian cgs units; r_g = GM/c² (cm), lengths in the shader in units of r_g.

### 2.1 Magnetic field (magnetically arrested disk, force-free jet)

- **Horizon flux:** Φ = φ · √(Ṁ c) · r_g with **φ = 50** (Gaussian units; ≈ 15 in the
  Heaviside–Lorentz units of the EHT papers): the magnetically arrested saturation value
  (Tchekhovskoy, Narayan & McKinney 2011). Ṁ from `units.ts` `mdotFromLambda(M, a, λ)` (the Mass and
  Accretion sliders and presets).
- **Along the funnel** (the existing parabolic funnel ρ_f(z) = 0.6 + 0.7 √|z| r_g, which matches
  M87's parabolic jet): poloidal field by flux conservation **B_p = Φ / (π ρ_f²)**, and the wound-up
  toroidal field of a rotating force-free jet **B_φ = B_p · ρ Ω_F / c** with field-line angular
  velocity **Ω_F = Ω_H / 2** (Ω_H = a c / (2 r_H)); **B = √(B_p² + B_φ²)**.
- **Validation (a test):** the implied Blandford–Znajek power P = (κ/4πc) Φ² Ω_H² with κ = 0.05
  gives ≈ 0.98 Ṁc² at a = 0.9, and for the M87\* preset (Ṁ = 7.7 × 10⁻⁴ M_☉/yr) **≈ 4.3 × 10⁴³
  erg/s**, inside the published M87 jet-power range 10⁴³–10⁴⁴ erg/s.

### 2.2 Flow

- Outward along the funnel's streamlines (the self-similar parabola family through the point),
  measured by the local zero-angular-momentum observer, with Lorentz factor from M87's measured
  acceleration **Γ(z) = max(1, Γ₂₈₀ · (|z| / 280 r_g)^0.58)** (Mertens et al. 2016; Park et al.
  2019: Γ ≈ 2 at ~280 r_g, Γ ∝ z^0.56–0.58 below ~10³ r_s). The **Jet speed** slider sets Γ₂₈₀
  (default **2**, was a constant Γ = 5); within the ±60 r_g view the flow is mildly relativistic.
  No toroidal flow velocity (stated simplification).
- Knots ride the flow at the local β(z) (the 2026-10-01 fix, now with the profile).

### 2.3 Electrons

- Power law N(γ) = K γ^−p for γ ≥ γ_min, **p = 2.4** (optical spectral index α = (p−1)/2 = 0.7,
  within M87's measured 0.6–0.9, Perlman et al. 2001), **γ_min = 10** (stated assumption).
- **Density:** shape n ∝ B² (constant magnetization along the funnel) × wall profile × length
  falloff × knots × turbulence; absolute scale **K = k_scale · B² · (shape)**, with k_scale chosen on the
  CPU so that the optically thin, comoving visible-band luminosity of the rendered jet (both lobes,
  z_base ≤ |z| ≤ jetLength, ρ ≤ 1.2 ρ_f; 4π∫∫ j_ν dν dV, ν over 400–750 nm) equals ε · P_BZ. Recomputed
  whenever mass, spin, accretion, ε or the jet length change.
- **Below the power law's range:** the closed-form coefficients assume ν′ ≫ ν_min = 3 γ_min² q B /
  (4π m_e c). For ν′ < ν_min (X-ray binaries: B ~ 10⁸ G puts visible light below it) the
  low-frequency forms are used, matched at ν_min: j ∝ (ν′/ν_min)^(1/3), α ∝ (ν′/ν_min)^(−5/3).
- **Expected outcome:** reported, not targeted. (An earlier estimate in this spec's history used an
  absorption coefficient off by (m_e c²)^(p−1) — Rybicki & Lightman's 6.53 is per unit energy — and is
  void; the corrected closed forms agree with direct numerical integration of the single-electron
  spectrum to 0.05 % (j) and 0.07 % (α).)

### 2.4 Radiation and transfer

- Pitch-angle-averaged synchrotron emissivity j′_ν′ and absorption α′_ν′ for a power law
  (Rybicki & Lightman 1979, eqs. 6.36 and 6.53, averaged over an isotropic pitch-angle
  distribution), evaluated in the plasma frame at ν′ = ν / g.
- **Exact frequency shift:** g = ν_obs / ν_emit from the photon momentum the integrator already
  carries and the plasma four-velocity (the disk's `diskG` does the same for the disk).
- **Covariant transfer** of the invariant 𝓘 = I_ν/ν³ along the backward ray:
  `d𝓘 = (j′/ν′³) ds′ · e^(−τ)` and `τ += α′ ds′`, ds′ = r_g · |p·u| · dλ (the plasma-frame path
  length of an affine step), accumulated per jet quadrature sample (`JET_DL` chord samples, as now).
  The disk (and sky) behind the jet are attenuated by e^(−τ).
- **Colour:** I_ν at three visible frequencies (λ = 450, 550, 650 nm), each taken as constant over its
  band (360–500 / 500–600 / 600–830 nm), converted to linear sRGB with the same CIE colour-matching sum as
  the disk: a constant 3×3 matrix built on the CPU. (Revised in planning from a LUT over spectral slope:
  the matrix needs no power-law assumption, so it also holds where self-absorption bends the spectrum;
  a blackbody fed through it reproduces the disk path's luminance to 0.6 %.)

### 2.5 Common absolute scale with the disk

The disk is already shaded as visible-band radiance at T_obs divided by the visible luminance at
T_peak (`lumNorm`). That ratio is unchanged; the jet's absolute I_ν (cgs) is mapped into the **same
units** as the disk's blackbody radiance (the band matrix includes the cgs Planck prefactor 2hc² and
the disk's reference luminance), then multiplied by the **same** `lumNorm`. Exposure keeps its meaning; jet and disk brightness are
directly comparable. `JET_TINT`, `JET_GAIN` and `JET_CEIL` are deleted.

## 3. Interfaces and data flow

- `src/physics/synchrotron.ts` (new, CPU twin + validation): emissivity/absorption coefficients,
  field and flow profiles, BZ power, the energy-budget k_scale.
- `src/physics/jet.ts`: funnel/wall/falloff/knots stay; `jetEmission` and `dopplerBoost` are
  replaced by the synchrotron twin; `JET.vKnot`-era leftovers removed.
- `src/render/emission-shared.wgsl`: the jet core becomes the synchrotron sample
  `(j′, α′) at (r, θ, t)` + plasma four-velocity; still the sole copy, parity-tested.
- `src/render/raytrace.wgsl`: `jetStep` returns (𝓘 RGB, Δτ); `traceRay` attenuates the hit by
  e^(−τ); the cache's jet replay returns τ too, and `shade` attenuates the cached disk/sky colour.
- **Uniforms:** the 128-byte buffer is full (32/32 slots), so it grows to 144 bytes (one more vec4)
  for the jet's physical constants computed on the CPU each time mass, spin, accretion or the
  sliders change: Φ-derived field scale, the energy-budget density scale k_scale, Γ₂₈₀ and r_g in cm. `bloom.wgsl`/`present.wgsl` keep their shorter struct prefixes.
- **Panel:** a **Jet** checkbox (on/off) plus **Jet radiative efficiency ε** replacing the "Jet"
  strength slider (log, 10⁻⁵–10⁻¹, default 2 × 10⁻³); "Jet speed Γ" → **Jet speed (Γ at 280 r_g)** (1.5–8, default 2); presets set
  jet on (M87\*, Cyg X-1, GRS 1915+105) or jet off (Sgr A\*, Gargantua). Captions drop
  the M87\* calibration sentence; M87\*'s notes that the EHT ring is radio emission from a hot flow not
  modelled here, and that the jet's brightness follows from ε × its jet power.

## 4. Error handling

- B, K, j, α are finite for every slider value (Ṁ > 0); the e^(−τ) factor is in [0, 1]; τ is
  capped so its exponential never underflows to NaN.
- The `accum` NaN choke point remains the last guard; `verify:gpu` fails on any console warning.

## 5. Testing

- **Unit (CPU twin):** the closed-form power-law coefficients (j and α) against a direct numerical
  integration of the single-electron synchrotron spectrum F(x) = x∫K_{5/3} over the power law and the
  pitch-angle distribution, within 1 %; the BZ power for the M87\* preset in 10⁴³–10⁴⁴ erg/s; Γ(z) profile values;
  the energy budget (the CPU volume integral of the visible luminosity at the chosen k_scale equals ε · P_BZ to 1e-6); an optically thick slab saturating at the source
  function S_ν = j/α; invariance check (g = 1 → transfer equals the flat-space result).
- **Parity:** the synchrotron sample (`emission-shared.wgsl`) against the CPU twin, several cases.
- **GPU:** `?golden` re-recorded on purpose (jet scenes change). The jet-off scene must stay
  bit-identical, or — if the compiler reschedules shared code — within the 2026-10-01 A/B rule
  (differences explained by an experiment that restores the old data flow); `?cachecheck` exact including τ attenuation of cached colours; app checks and
  probes PASS; a jet screenshot set per preset for review; the M87\* jet-to-disk visible-light ratio
  reported (no target: it is a prediction of the model).
