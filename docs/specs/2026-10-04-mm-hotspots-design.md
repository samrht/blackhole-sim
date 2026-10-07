# Hotspot flares in the 1.3 mm view — design

**Date:** 2026-10-04
**Status:** implemented (plan `docs/plans/2026-10-07-mm-hotspots.md`, 2026-10-07). Departures from this design:
(1) the live-window half-width is measured, `HOTSPOT.pad` = 100 M (max light-travel delay through the blob region
77.9 M for Sgr A*, 68.8 M for Gargantua), equal to the 100 M written in §3 but now backed by a gated sweep; (2) in mm,
`main.ts` sends `fluxVar = 0` while no hotspot can be in view (in mm it drives only the hotspots), so hotspot-free live
frames skip the hotspot code exactly; (3) a dev-only `window.__bhSetTime` hook lets the app check jump to a hotspot.
Calibrated A₀ = 9.22.
**Builds on:** `docs/specs/2026-10-04-hot-flow-mm-design.md` (the hot flow and the 1.3 mm view),
`docs/specs/2026-10-03-flux-statistics-design.md` (the eruption history it reuses). Revives the intent of the shelved
`feat/eruption-flares` (disk hotspots, invisible against the thin disk) on the hot flow, where they are visible.

## 1. Intent (agreed)

- **Recurring hotspot flares** in the 1.3 mm view: a bright blob is born at each horizon-flux eruption, orbits the
  hole one to three times and fades, as ALMA saw after Sgr A*'s X-ray flare of 2017 April 11 (Wielgus et al. 2022).
- **All hot-flow objects**, anchored on Sgr A*'s measurement and scaled in units of M with brightness relative to the
  local flow; other objects' hotspots are captioned as a model prediction.
- **Volumetric and exact**: the hotspot is integrated with the flow along each ray (Doppler, lensing, light-travel
  delay, absorption). Frames with a hotspot alive are **traced live**; the steady-flow cache serves the rest
  (user's choice after the prototype, §2.5).
- The visible view, and its illustrative "Flares" spots, are unchanged.

## 2. Physics

### 2.1 Measured anchor (Wielgus et al. 2022, A&A 665, L6)

| Property | Value | Section |
|---|---|---|
| Orbital radius (Keplerian) | 10.0–11.2 r_g (a = 0), 9.8–11.0 (a = 1); fiducial 11 r_g | §3.2, §3.3 |
| Period | 74 ± 6 min (polarization loop) | §2.3 |
| Inclination, sense | ~20°, clockwise on the sky; hints of prograde motion | abstract |
| Size | Gaussian, diameter ~6 r_g | §3.3 |
| Hotspot flux at 229 GHz | ℐ_hsp ≲ 0.3 Jy against ℐ_sha ≈ 2.4 Jy | §3.1 |
| Duration | one clear loop (~68 min) inside a ~2 h "loopy" period | §2.3 |

Near-infrared orbits (GRAVITY Collaboration 2018, A&A 618 L10): "approximately six to ten times the gravitational
radius", periods ~45 (±15) min, clockwise, near face-on.

### 2.2 Hotspot k

From the flux history (`flux-history.ts`; eruption k at t_k, depth δ_k; slider s = Flux variability):

- **Orbit radius** r_c,k = 8 + 4 u_k r_g, u_k ∈ [0, 1) a hash of k (spans the GRAVITY and ALMA radii; uniform within
  the range is an assumption). **Starting azimuth** φ₀,k = 2π v_k (second hash).
- **Orbit**: Keplerian, prograde, equatorial: φ_c(t) = φ₀,k + Ω_c (t − t_k), Ω_c = 1 / (r_c^{3/2} + a). The blob is
  rigid (moves at Ω_c); real hotspots shear, but over 1–3 orbits this is what the ALMA fit uses.
- **Shape**: Gaussian G(d) = exp(−d² / 2σ²), σ = 6 / 2.3548 = 2.548 M (the ~6 r_g diameter taken as FWHM), d the
  Euclidean distance in Boyer–Lindquist pseudo-Cartesian coordinates (x = r sinθ cosφ, …) to the centre
  (r_c, π/2, φ_c).
- **Light curve** with τ = t − t_k and P = 2π / Ω_c: L(τ) = smoothstep(0, 0.1 P, τ) · exp(−τ / P) ·
  (1 − smoothstep(2.5 P, 3 P, τ)); zero for τ < 0 and τ ≥ 3 P. 3 P ≤ 784 M at r_c = 12, a = 0 < the 1000 M minimum
  eruption spacing, so at most one hotspot is alive at any emission time.
- **Amplitude** A_k = A₀ · s δ_k / δ̄ (δ̄ = FLUX.dbar, the mean depth); s = 0 means no eruptions and no hotspots.

### 2.3 Emission

Inside the blob the flow's emitting electrons are boosted by the factor A_k L(τ) G(d) at the flow's own temperature
and field: the hotspot adds j′_h = A_k L G · j′_flow(r, θ; ν D_h) and α′_h = A_k L G · α′_flow(r, θ; ν D_h), radiating
with its own Doppler factor D_h = u^t (p_t + Ω_c p_φ), u^t = (−(g_tt + 2Ω_c g_tφ + Ω_c² g_φφ))^{−1/2} at the sample
(no contribution where that is not timelike). Per sample both emitters share one slab (§2.5): the observed
contribution (j′_f / D_f³) ds_f + (j′_h / D_h³) ds_h with ds = r_g D dl and optical depth α′_f ds_f + α′_h ds_h, the
time of each sample its own emission time (light-travel delay, or one global instant with the Light-delay toggle
off, as for the disk and jet).

### 2.4 Calibration

A₀ is fitted once by `scripts/calibrate-hotspot.ts`: with the CPU twin (96², Sgr A* preset, r_c = 10 r_g, δ_k = δ̄,
s = 1, the hotspot at the peak of L and its emission-time dependence frozen), the flux the hotspot adds, averaged over
four orbital phases, equals 0.3 Jy. Prototype (2026-10-04): A₀ ≈ 8.5 (A = 10 → +0.351 Jy; +1.3e10 K over the
~0.8e10 K flow at the hotspot's pixels; ring peak 4.0e10 K: clearly visible, fainter than the ring).

### 2.5 Why live tracing (prototype findings)

- The hotspot is partly opaque (optical depth ~0.3 through it): adding its light over a steady flow image is wrong by
  8 % in flux and 44 % of the peak in the worst pixel (A = 10); flow and hotspot must be integrated together.
- A bookmark replay of the hotspot's region (r ≲ 20 M near the plane) would re-integrate most of each ray (~255 of
  ~410 steps) and bookmark nearly every pixel, beyond the 512 MB cache budget at 1080p. Live tracing costs the same
  per frame and is exact by construction.

## 3. Rendering and code

- `src/physics/hotspot.ts` (CPU twin): `hotspotAt(t, s)` → { alive, k, rc, phiC, amp } (amp = A_k L); `hotspotBoost`
  (the G-weighted factor at a point); `hotspotShift` (D_h); `hotspotAliveWindow(t, s, pad)` for the mode switch;
  constants `HOTSPOT = { rMin: 8, rSpan: 4, sigma: 2.548, rise: 0.1, cutFrom: 2.5, cutTo: 3, A0 }`.
- `src/physics/hot-flow-image.ts`: the CPU image gains an optional hotspot (samples carry φ and t) for calibration
  and validation.
- `src/render/emission-shared.wgsl` (sole copy): `HS_*` constants, `hotspotStateJ(epoch, rel, s, a)` → vec4(rc, φ_c,
  amp, alive) on the f32-safe epoch split (`splitPeriodJ`, the eruption-k logic of `fluxDeficitJ`),
  `hotspotShiftJ`. `raytrace.wgsl` `flowStep`: per sample, when a hotspot is alive at its emission time and the sample
  passes a cheap |r − r_c| < 4σ, |z| < 4σ test, add the hotspot's coefficients to the slab.
- No new uniforms (s = `fluxVar`, the clock, `lightDelay` exist).
- `src/main.ts`: in mm hot-flow mode, `hotspotAliveWindow(simTime, s, 100 M)` forces live tracing (`chooseMode`
  gains the flag); the cache is neither cleared nor rebuilt, so cached frames resume when the hotspot fades.
- UI: no new controls. Sgr A*'s caption cites ALMA's hotspot; other hot-flow captions call theirs a model prediction;
  the Flux-variability tooltip says it also drives the 1.3 mm hotspots.

## 4. On screen

Sgr A* at 1.3 mm: every ~1500 M (~8.5 h of Sgr A* time; seconds at high Motion) a blob appears at ~8–12 r_g, orbits
in ~150–270 M (~50–90 min for Sgr A*), brighter where it approaches, with its lensed secondary image in the photon ring, and fades
within three orbits. M87*, Gargantua and dense custom flows show the same in units of M.

## 5. Tests and gates

- **CPU (`npm test`):** schedule (one hotspot per eruption, radius range, at most one alive, none at s = 0),
  light-curve shape (rise, e-fold, zero at 3P), Keplerian period, D_h for a static far observer limit, boost profile
  (G = 1 at the centre, FWHM 6 M), mode rule (live iff mm hot flow and a hotspot in the padded window).
  **Calibration gate** (SWEEP=1): the tabulated A₀ reproduces 0.3 Jy within 5 %.
- **GPU (`npm run verify:gpu`):** `?parity` hotspot cases (state at late epochs, boost, D_h; a mutation fails);
  `?hotflow` renders Sgr A* at a hotspot peak: added flux within 5 % of the CPU twin, centroid within 0.5 M;
  `?golden` adds `sgra-mm-hotspot` (visible and existing mm hashes unchanged); `?cachecheck` unchanged (cached frames
  are never shown with a hotspot alive); app check: at a hotspot time the mode reads live and the image differs from
  the steady flow; bench: live mm with a hotspot.
- Visible golden hashes must stay bit-identical (or the proven compiler-rescheduling exception).

## 6. Limitations

- Rigid blob (no shear), thermal electrons boosted at the flow's temperature (real hotspots are heated and partly
  non-thermal, and strongly polarized: this renderer has no polarization).
- The radius range and A₀ come from Sgr A*; elsewhere they are predictions.
- Hotspot frames are live: they render at the live rate (the scale controller lowers resolution while animating).
