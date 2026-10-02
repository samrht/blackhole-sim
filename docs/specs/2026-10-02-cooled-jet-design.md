# Cooled synchrotron jet: full-spectrum energy budget — design

**Date:** 2026-10-02
**Status:** implemented on `feat/cooled-jet` (plan `docs/plans/2026-10-02-cooled-jet.md`)
**Follows:** `docs/specs/2026-10-02-synchrotron-jet-design.md` (shipped, main e950f65), whose final review found
that the visible-band energy budget is energetically impossible for the X-ray binaries.

## 1. Intent (agreed)

- **Energy conservation over the whole spectrum.** The jet's electrons are given a fraction **η** of the
  jet's Blandford–Znajek power and radiate it (synchrotron, all frequencies). Visible light is whatever
  share of that the physics puts in the visible band — a prediction for every object, never a target.
  This replaces "visible luminosity = ε × P_BZ", which in Cygnus X-1 and GRS 1915+105 demanded electrons
  radiating 0.6× and 6× P_BZ below ν_min alone.
- **Radiative cooling, computed.** Every electron in view radiates its energy far faster than the flow
  crosses the view (measured with the shipped model: the energy that cools in one flow time is γ ≈ 10⁻³
  for the default view, 0.03–0.5 for M87*, 10⁻⁶–10⁻⁴ for the X-ray binaries). The electron population is
  the exact cooled solution, not an assumed power law.
- **Exact emission from slow electrons.** The X-ray binaries' visible light comes from electrons with
  γ ≈ 0.4–3 (the cyclotron regime), where the γ ≫ 1 closed forms fail. Emission and absorption are the
  exact cyclo-synchrotron result, precomputed into a lookup table.
- **Electron spectrum fitted to M87's measured core**, used for every object (agreed choice
  "fit the whole core SED"): synchrotron peak at ~10^11.5–10^12 Hz, average index α ≈ 1.1 from infrared to
  X-rays (Prieto et al. 2016; broad-band nucleus modelling, arXiv 2406.17200), α ≈ 1.5 from optical to UV
  (Perlman et al. 2011).
- **η anchored to M87's optical nucleus** (νL_ν ≈ 10⁴¹ erg/s at 550 nm, the observation the ε default
  came from), then applied unchanged to every object.
- Not in scope: inverse-Compton cooling (a stated simplification: synchrotron losses only), pair
  production, polarisation, a time-dependent or ordered field (the field orientation is averaged as random,
  as now), GRMHD.

## 2. Physics

Gaussian cgs; σ_T = 6.6524587 × 10⁻²⁵ cm²; ν_B = qB / (2π m_e c) (cyclotron frequency); u = γβ.

### 2.1 Injection

- Electrons are accelerated in place with the injection spectrum Q(γ) = Q₀ g(γ):
  g = γ^−2.2 for 40 ≤ γ ≤ 1200, g = 1200^0.8 γ^−3.0 for γ > 1200, g = 0 below 40.
  In fast cooling the observed index is p_inj / 2, so these give α = 1.1 below and α = 1.5 above the
  break; γ_min = 40 and γ_br = 1200 put the M87* preset's spectral peak at ~10^11.75 Hz and its break in
  the optical (ν = 1.5 γ² ν_B with B ≈ 85 G at z ≈ 10 r_g). Fixed by M87, used for every object.
- Injected power density (plasma frame) **q′ = q₀ · B² · shape**, with the existing shape (wall × length
  falloff × knots × turbulence): dissipation follows the magnetic energy density, as the shipped density
  did. Q₀ = q′ / (m_e c² I_g), I_g = ∫ (γ − 1) g dγ (finite: p = 3 above the break).

### 2.2 Cooling (exact, synchrotron losses)

- Loss rate for any γ, averaged over pitch angle: dγ/dt = −k u², k = σ_T B² / (6π m_e c) (the Larmor
  generalisation; exact from the cyclotron to the synchrotron limit).
- **Cooling depth** s = k t′, with t′ the plasma's proper time since leaving the jet base. With the shipped
  proper-speed profile Γβ = U₀ (|z| / 280)^0.58, U₀ = √(Γ₂₈₀² − 1):
  t′ = (r_g / c) · 280^0.58 / U₀ · (|z|^0.42 − z_b^0.42) / 0.42, z_b = 2 r_g (JET.zBase).
- **Population** for injection at a constant rate for a time t′ (exact): an electron cools from G to γ in
  time (arccoth γ − arccoth G) / k, so
  N(γ) = (1 / (k u²)) ∫_γ^{G(γ, s)} Q(γ′) dγ′, G = coth(arccoth γ − s) if arccoth γ > s, else ∞.
  Fast cooling (s ≫ 1) gives N ∝ γ^−(p+1) above γ_min and ∝ u^−2 below it; slow cooling (s ≪ 1) gives
  N ≈ Q t′. The dimensionless N̂(γ; s) = N k / Q₀ depends on s only.

### 2.3 Emission and absorption

- Single-electron spectrum: the **exact cyclo-synchrotron emissivity** (Bessel series over harmonics;
  Melrose 1980; Leung, Gammie & Noble 2011), integrated over emission direction and averaged over pitch
  angle: P_ν(γ) = (q³ B / m_e c²) p̂(ν/ν_B, γ). For γ above a seam **γ_s = 10** the pitch-averaged
  synchrotron kernel F(x) is used instead. (Revised in planning: measured, the kernel's power-weighted
  difference from the exact sum is 14 % at γ = 3, 5.1 % at 5, 1.3 % at 10, falling as 1/γ²; 0.1 % would
  need γ_s ≈ 35, ~40× the build cost, and summing harmonics as a continuum was measured 4–10 % off.)
- Isotropic electrons in a random field: j_ν = (1/4π) ∫ N P_ν dγ and (Ghisellini & Svensson 1991, exact
  for any γ) α_ν = −(1 / (8π m_e ν²)) ∫ P_ν γ u ∂/∂γ [N / (γ u)] dγ.
- Both reduce to j′ = C_j · q₀ · shape · B · Ĵ(x, s) and α′ = C_α · q₀ · shape · B^−1 · x^−2 · Â(x, s),
  with x = ν′/ν_B and the dimensionless Ĵ, Â depending only on (x, s). (The plan derives C_j, C_α and
  checks them against the closed forms.)
- **Table** `public/synch-table.bin`: ln Ĵ and ln Â on a grid of ln x (10⁻⁴ to 10¹⁰) × ln s (10⁻³–40),
  built offline by `scripts/build-synch-table.mjs` from `src/physics/cyclosynch.ts`. Outside the grid:
  above x = 10¹⁰ the exact asymptote of the fast-cooled population above γ_br (a pure γ^−4 law:
  j ∝ ν^−1.5, α ∝ ν^−4, i.e. Ĵ ∝ x^−1.5 and Â ∝ x^−2); below x = 10⁻⁴ nothing in the slider range samples
  (the largest field, ~3 × 10¹⁰ G, with D ≥ 0.1 gives x ≳ 10⁻³; a test pins this), and the first column is
  used; in s, linear in ln s beyond either end (N̂ ∝ s as s → 0; s-independent as s → ∞ apart from the
  coldest electrons). The shader samples it as a texture with manual bilinear interpolation in (ln x, ln s); the
  CPU twin reads the same file.

### 2.4 Energy budget and anchor

- **η = injected electron power / P_BZ**, with the injected power counted as energy at infinity:
  L_inj = Σ_lobes ∫ α_lapse² · Γ · q′ · √(g_rr g_θθ g_φφ) dr dθ dφ over the rendered region (one lapse
  factor for the energy's redshift, one for the time dilation; Γ for the plasma's density in the
  zero-angular-momentum frame — revised in planning from a single lapse factor)
  (z_b ≤ |z| ≤ jetLength, ρ ≤ 1.2 ρ_f); the shape's knots and turbulence are mean-one, so their time
  average drops out. q₀ = η P_BZ / (that integral at q₀ = 1). What is radiated is ≤ what is injected by
  construction (equal in fast cooling), so energy is conserved for every object and slider value.
- **P_BZ with the high-spin correction** (Tchekhovskoy, Narayan & McKinney 2010):
  P_BZ = (κ / 4πc) Φ² Ω_H² f(Ω_H), f = 1 + 1.38 ω² − 9.2 ω⁴, ω = Ω_H r_g / c = a / (2 r_H)
  (+4.7 % at a = 0.9, −14 % at 0.998); κ = 0.05, Φ as shipped. M87* becomes ≈ 4.46 × 10⁴³ erg/s.
- **Anchor:** the default η is solved once so the M87* preset's optically thin, comoving νL_ν at 550 nm
  (both lobes, rendered region, the same α_lapse² Γ √γ weighting) equals 10⁴¹ erg/s. The HST nucleus aperture (~0.1″ ≈ 2 × 10⁴ r_g) contains
  more jet than the ±60 r_g view, so the anchored η is an upper bound; stated in the README.
- **Slider:** "Jet efficiency η" (log 10⁻⁴–1, default = the anchor), replacing ε.

### 2.5 Unchanged

Transfer along the ray (slab steps, e^−τ attenuation of the disk and sky, cached replay), the plasma
frequency shift D, the three bands and the band matrix into the disk's units, the field, the flow profile,
the funnel geometry, the panel's Jet checkbox and Jet speed, the presets' on/off choices.

## 3. Interfaces

- `src/physics/cyclosynch.ts` (new): exact single-electron spectrum, cooling, N̂(γ; s), Ĵ / Â, table
  build and CPU lookup.
- `scripts/build-synch-table.mjs` (new) → `public/synch-table.bin` (committed; header with grid ranges and
  the physics constants it was built for, so a stale table is detected by a test).
- `src/physics/synchrotron.ts`: injection constants, P_BZ correction, the new budget and anchor;
  `jetUniforms` returns `jetQ0` in place of `jetKScale` (uniform buffer stays 144 bytes); the closed-form
  coefficient path and `visLuminanceUnit` are replaced.
- `src/render/emission-shared.wgsl`: `synchSampleJ` reads the table (texture + manual bilinear) and
  computes s from (z, B, U.jetGamma, r_g); the shipped transfer code is untouched. The table texture is
  declared by each consumer (renderer, jet-parity harness): the compute stage already uses 7 of the 8
  guaranteed storage buffers, so a texture rather than a buffer.
- `src/render/gpu.ts`: fetches the table at init, creates the texture, binds it.
- Panel/presets: ε → η; captions revised once the measured result is known (the X-ray-binary
  "overestimate" sentence goes if the energy budget now holds for them).

## 4. Error handling

- Every table value finite (logs of positive numbers; zero emission below the fundamental represented by
  a floor, not −∞); s clamped to a finite ln range before the lookup; U₀ > 0 guarded (Jet speed ≥ 1.5).
- k_scale's old spin-0 guard becomes q₀ = 0 (P_BZ = 0): jetStep returns early, as now.
- `verify:gpu` fails on any console warning; the NaN choke point stays.

## 5. Testing

- **Single electron:** the emission spectrum summed over harmonics and integrated over frequency equals
  the Larmor power (4/3) σ_T c U_B u² within 0.5 % for u = 0.1–3 (measured 0.03 %); the seam at γ_s = 10
  differs from the synchrotron kernel by ≤ 1.5 % power-weighted.
- **Kirchhoff:** for a thermal (Maxwell–Jüttner) population, j_ν / α_ν = B_ν(T) within 1 % at
  kT = 0.1, 1, 10 m_e c² — validates the absorption independently of the emission.
- **Cooling:** N̂ solves the continuity equation (numerical check); calorimetry — the fast-cooled
  population radiates, over all frequencies, the injected power within 1 %; s → 0 gives N̂ = s · g.
- **Table:** lookup vs direct computation at random points within 0.1 %; continuity at the grid edges and
  the asymptotes; for M87* at 550 nm the table equals the closed-form power-law coefficients for the cooled
  spectrum within 1 %; header matches the constants in code.
- **Budget:** L_inj = η P_BZ to 1e-9; the anchor gives M87* νL_ν(550 nm) = 10⁴¹ erg/s; P_BZ(M87*) in
  10⁴³–10⁴⁴; f(Ω_H) values; extremes finite and non-negative.
- **GPU:** `?parity` on the table lookup and coefficients against the CPU twin with an absolute tolerance on
  logs (the review's minor finding); `?golden` re-recorded with the jet-off A/B rule; `?cachecheck` exact,
  with a new optically thick (X-ray-binary-class) scene; app checks, probes, a before/after `bench.mjs`
  delta, a screenshot per preset, and the jet ÷ disk light per object, reported, not tuned.
