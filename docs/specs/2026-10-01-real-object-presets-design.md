# Real-object presets with physical units — design

**Date:** 2026-10-01
**Status:** approved in conversation; awaiting written-spec review
**Roadmap item:** `docs/ROADMAP.md`, "Next: physics on what the code already has"

## 1. Intent (agreed)

- **Purpose: accurate and educational.** Real black holes rendered with their measured spin, viewing
  angle and mass, physically derived disk temperature, readouts in physical units, and an honest
  caption wherever our model differs from the real object.
- **Objects:** M87\*, Sagittarius A\*, Cygnus X-1, GRS 1915+105, and Interstellar's Gargantua
  (fictional, with published parameters).
- **Temperature:** physical, from mass and accretion rate, with a Mass and an Accretion slider to
  explore.
- **Brightness:** visible-band radiance everywhere (what a camera would record), replacing today's
  bolometric T⁴ law in every view, including the default one.
- Not in scope: product/UX items (export, shareable links, mobile; `docs/ROADMAP.md`), absolute
  luminosity (brightness stays exposure-relative), new accretion-flow models (RIAF/ADAF), the
  Interstellar film's "Doppler off" rendering mode.

## 2. Physics

### 2.1 Units and constants (`src/physics/units.ts`, new)

SI constants (G, c, σ_SB, M_sun, year). Gravitational radius r_g = G M / c², time unit
t_g = G M / c³. Eddington luminosity L_Edd = 1.2572 × 10³¹ W × (M / M_sun) (electron scattering,
hydrogen). Accretion parameter λ is defined through the thin-disk luminosity:
**Ṁ = λ L_Edd / (η(a) c²)** with η(a) the existing `efficiency(a)` (1 − E_ISCO), so λ = 1 is a thin
disk radiating at the Eddington limit. Conversions: λ ↔ Ṁ in M_sun / yr.

### 2.2 Peak temperature

The existing `pageThorneFluxShape(r, a)` is the Page–Thorne flux for Ṁ = M = 1 in geometric units,
so the physical flux from one face is

  F(r) = Ṁ c² · shape(r) / (4π r_g²),  T(r) = (F / σ_SB)^{1/4},

and **T_peak = T at the radius maximising shape** (the same maximiser `temperatureShape` already
normalises by, so the rendered profile T(r) = T_peak · temperatureShape(r) is unchanged). Measured
with the code as it stands (2026-10-01): 10 M_sun, a = 0, λ = 1 → 7.0 × 10⁶ K (textbook order 10⁷ K);
large-r Newtonian check shape · r³ / (1 − √(6/r)) = 1.47 at r = 10⁴ (3/2 expected; the Simpson
integral is coarse that far out, and T_peak is evaluated at r ≈ 2–10).

### 2.3 Visible-band brightness

For a blackbody emitter the observed specific intensity is a blackbody at T_obs = g · T (I_ν / ν³
is invariant along the ray), so the colour **and brightness** a camera records is the visible-band
radiance of a blackbody at T_obs. Today the shader uses `colorLUT(T_obs)` normalised to luminance 1
times (g · T/T_peak)⁴ — the bolometric law. New:

  pixel = VisRad(T_obs) / VisLum(T_peak) · E

- **VisRad(T)**: linear-sRGB radiance of a blackbody at T integrated against the CIE colour-matching
  functions already in `color.ts` (same integral, **not** luminance-normalised).
- **VisLum(T_peak)**: its luminance at the disk's rest-frame peak temperature, a new CPU-side uniform
  (`lumNorm` = 1 / VisLum(T_peak)), so exposure means the same thing for every preset and a 300 K
  disk is not 10²⁰ times dimmer than a 10⁷ K one on screen.
- **E**: the existing emission field (turbulence, hot spots, breathing), unchanged.
- The LUT is **log-spaced over 100 K – 10⁹ K, 1024 entries**, stored relative to VisLum(10⁴ K) so the
  f32 values stay in range (Wien-tail entries underflow harmlessly towards 0: a 250 K disk is
  invisible, which is correct). `sampleColor` changes from linear to log indexing.

### 2.4 Physical readouts (panel)

Mass (M_sun); peak disk temperature (K); horizon r₊, ISCO and photon orbit in km (r × r_g); orbital
period at the ISCO, 2π (r^{3/2} + a) t_g, auto-scaled to ms / s / min / h / days; **playback scale**
"1 s on screen ≈ X real time" from the animation's M-per-second rate and t_g. The existing values in
units of M stay alongside.

## 3. Presets (`src/physics/presets.ts`, new)

A data table; each entry has name, M, a, inclination, λ, jet strength, caption and sources.

| Object | M (M_sun) | a | i | λ (Ṁ) | Jet | T_peak (thin disk) |
|---|---|---|---|---|---|---|
| M87\* | 6.5 × 10⁹ | 0.9 | 17° | 8.3 × 10⁻⁶ (Ṁ ≈ 7.7 × 10⁻⁴ M_sun/yr) | on | ≈ 4,100 K |
| Sgr A\* | 4.3 × 10⁶ | 0.94 | 30° | 1.9 × 10⁻⁷ (Ṁ ≈ 10⁻⁸ M_sun/yr) | off | ≈ 11,000 K |
| Cygnus X-1 | 21.2 | 0.998 | 27° | 0.02 | on | ≈ 6.1 × 10⁶ K |
| GRS 1915+105 | 12.4 | 0.98 | 60° | 0.3 | on | ≈ 1.1 × 10⁷ K |
| Gargantua | 1.0 × 10⁸ | 0.6 | 85° | 1.7 × 10⁻⁶ (T_peak ≈ 5,800 K) | off | ≈ 5,800 K |
| Custom (default) | 1.0 × 10⁸ | 0.9 | 72° | 3.66 × 10⁻⁴ | on | ≈ 30,000 K (today's colour) |

The T_peak column is computed by §2.2 from the stored values (M87\*: Ṁ at the log-midpoint of
(3–20) × 10⁻⁴ M_sun/yr; λ values are derived in a test from the cited Ṁ, not hand-entered).

**Captions (shown under the selector; sources in the code):**
- **M87\*** — EHT 2019: M = 6.5 × 10⁹ M_sun (Paper VI); viewing angle 17° from the jet (Mertens et
  al. 2016, Walker et al. 2018); Ṁ (3–20) × 10⁻⁴ M_sun/yr (EHT Paper VIII). Spin is not measured;
  0.9 is a common model value. *Caveat: M87\*'s real flow is a hot, thick, radio-bright flow, not
  the thin disk drawn here; the colour is what a thin disk at this accretion rate would emit.*
- **Sgr A\*** — mass 4.3 × 10⁶ M_sun (GRAVITY); a = 0.94 and i = 30° are EHT 2022 Paper V's
  "best-bet" model values; Ṁ ~ 10⁻⁸ M_sun/yr. *Same thin-disk caveat; spin and angle are model
  preferences, not measurements.*
- **Cygnus X-1** — M = 21.2 ± 2.2 M_sun and a\* > 0.9985 (Miller-Jones et al. 2021); i ≈ 27° (Orosz et al. 2011);
  λ ≈ 0.02 (hard state). *Spin sits at the slider's 0.998 cap (the Thorne limit) — the measured value
  is higher. The real disk peaks in X-rays; the visible colour shown is its blue-white tail.*
- **GRS 1915+105** — M = 12.4 M_sun, jet inclination 60° ± 5°, a ≈ 0.98 (Reid et al. 2014);
  λ ≈ 0.3. *Same X-ray caveat.*
- **Gargantua (fictional)** — M = 10⁸ M_sun (Thorne, *The Science of Interstellar*); the film's disk
  was rendered at a = 0.6, chosen for the visuals (James, von Tunzelmann, Franklin & Thorne 2015);
  an "anemic" disk about as hot as the Sun's surface. The 85° viewing angle is our choice, to
  resemble the film's near-edge-on shots (no published value). *The film removed Doppler colour and
  brightness shifts; this render keeps them, which is why one side is brighter here.*

Selecting a preset sets spin, inclination, mass, accretion and jet strength (1.0 where the table
says on, 0 where off; other sliders keep their values). Moving spin, inclination, mass or accretion afterwards switches the selector to
**Custom** (the caption hides). Inclinations in the table are within the 1–89° slider; spins within
0–0.998.

## 4. Architecture and data flow

- `src/physics/units.ts` — constants, r_g, t_g, L_Edd, λ ↔ Ṁ, `peakTemperature(M, a, λ)`,
  `formatDuration(seconds)`, `formatLength(m)`. Pure, unit-tested.
- `src/physics/presets.ts` — `PRESETS` table + `CUSTOM_DEFAULT`; pure data, unit-tested.
- `src/physics/color.ts` — new `blackbodyVisibleRadiance(T)` (un-normalised); the existing
  luminance-normalised function stays for its tests and callers.
- `src/physics/lookups.ts` — `buildColorLUT` becomes the log-spaced visible-radiance table
  (signature changes to the range and size it needs; callers updated).
- `src/render/uniforms.ts` — new `lumNorm` field (and its test); `Tpeak` is now computed, not a
  constant.
- `src/render/raytrace.wgsl` — `sampleColor` log indexing; `shadeDisk` returns
  `sampleColor(T_obs) * U.lumNorm * E` (no `pow(g * Tn, 4)`).
- `src/main.ts` + `index.html` — Object selector and caption, Mass and Accretion sliders (log
  scales), new readouts; `T_PEAK` constant removed. Mass/accretion are **shading-only**: they change
  T_peak and lumNorm, never the geometry key, so the geodesic cache does not rebuild; a preset that
  changes spin or inclination rebuilds as any spin/inclination change does.
- Test scenes (`src/test/scenes.ts`) keep T_peak = 3 × 10⁴ K and compute lumNorm the same way.

## 5. Error handling

- Log sliders clamp to their ranges; λ and M are always > 0, so T_peak > 0 and lumNorm is finite
  (VisLum(T_peak) > 0 for T_peak ≥ 100 K; T_peak below 100 K is clamped to the LUT floor).
- Formatting falls back to scientific notation outside the unit ladder.

## 6. Testing

- **Unit (vitest):** r_g / t_g for 1 M_sun (1477 m, 4.93 µs); L_Edd; η(0) = 0.0572, η(0.998) ≈ 0.32;
  T_peak(10 M_sun, a = 0, λ = 1) within 10 % of 7.0 × 10⁶ K and ∝ λ^{1/4}, ∝ M^{-1/4}; λ ↔ Ṁ
  round trip; every preset's values inside the slider ranges and its T_peak inside the table's
  ±10 %; M87\* and Sgr A\* λ derived from the cited Ṁ; Gargantua's T_peak within 5 % of 5,800 K;
  VisRad luminance strictly increasing in T; VisRad(T) / VisLum(T) equal to the existing normalised
  colour (same chromaticity); log-LUT lookup within 0.5 % of direct evaluation; formatter cases.
- **GPU:** `?golden` re-recorded on purpose (brightness law changes); `?cachecheck` PASS (mass and
  accretion are shading-only); `?parity` / `?shadow` / probes unchanged (`?parity` has no colour case:
  `?shadow` uses its own flat LUT, which moves to the new format and log indexing). Visual check of each preset (screenshot set saved to the
  ledger) and of the default view; if the default view is markedly dimmer or brighter under the
  visible-band law, the default exposure is retuned and the value recorded.
- `verify:gpu` now fails on any console warning, so a WGSL slip in this change cannot pass silently.
