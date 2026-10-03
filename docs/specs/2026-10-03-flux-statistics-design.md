# Horizon-flux statistics done right — design

**Date:** 2026-10-03
**Status:** design approved in conversation 2026-10-03; this document awaits review
**Follows:** `docs/specs/2026-10-03-jet-flux-knots-design.md` (follow-up 1 of 3: "spin-dependent flux statistics").

## 1. Why (two calibration errors in the shipped jet knots)

1. **Wrong variable.** The jet power is P ∝ Φ² (Blandford–Znajek; equivalently η Ṁ c² with η ∝ φ_BH² and
   φ_BH = Φ/√Ṁ). The shipped generator was calibrated to the variability of the *normalised* MAD parameter φ_BH
   (Narayan et al. 2022 Table 3: 0.209 at a = 0.9). The same paper's Figure 9 gives the variability of the
   absolute horizon flux Φ separately (red stars), and it is about half: ≈ 0.098 at a = 0.9.
2. **Wrong statistic.** Narayan et al.'s σ/μ is a *modulation index*: the rms within 1000 t_g windows divided by
   the window mean, averaged over 50 windows (their eq. 13), and they report similar values for 500 and 2000 t_g
   windows. The shipped calibration matched the rms of the whole series. Window-insensitivity also means most of
   the variance is fast (≲ 500 t_g), which a pure sawtooth with ~1500 M eruptions cannot have; their Figure 2
   (φ_BH(t), a = ±0.7) is visibly dominated by fast flicker with occasional deeper dips.

## 2. Physics

### 2.1 Targets

| Quantity | Value | Source |
|---|---|---|
| Φ modulation index (1000 t_g windows) vs prograde spin | trend line 0.067 (a = 0) → 0.098 (a = 0.9), linear in a; a > 0.9 uses 0.098 | Narayan et al. 2022 Fig. 9 (red stars; the authors' dashed trend line; points 0.067, 0.079, 0.098, 0.081, 0.098 at a = 0, 0.3, 0.5, 0.7, 0.9) |
| Window insensitivity | 500 and 2000 t_g windows give "similar" results; made precise here as M₅₀₀ / M₂₀₀₀ ≥ 0.80 | Narayan et al. 2022 §3.3 |
| Eruption amplitude | δφ = 2 σ_φ (the series rms) | arXiv 2510.25842 §IV.3 |
| Eruption shape and recurrence | unchanged: ~1500 M recurrence (gaps 1000–2000 M), exponential drop with 500 M e-folding, linear refill | arXiv 2510.25842; Ripperda et al. 2022 |

The app's spin slider is prograde only (0–0.998), so only prograde values are used.

### 2.2 Model

  Φ(t) / Φ_sat = (1 − d(t)) · (1 + ε · n(t))

- **d(t):** the shipped eruption sawtooth (same hash, same jitter 1/3, depth spread ±25 %, 500 M e-fold, linear
  refill), with mean depth δ̄(a).
- **n(t):** flicker — unit-variance Gaussian red noise: hashed standard normals on a 1-D lattice with cell
  τ_n = 50 M (own salt), smoothstep-interpolated between neighbours and renormalised by √(w₀² + w₁²) (exactly
  N(0, 1) at every t, as the disk turbulence), then **clipped to [−3, 3]** (0.27 % of the time; keeps the widest
  jet inside the existing cache envelope, §3).
- **Calibration per spin (δ̄, ε):** two conditions at s = 1 over 10⁶ M: (i) the modulation index with 1000 M
  windows equals the trend-line target; (ii) δ̄ = 2 × the series' own σ/μ (self-consistent). τ_n = 50 M is the
  longest cell for which M₅₀₀/M₂₀₀₀ ≥ 0.80 holds (prototype: 0.83 at 25 M, 0.805 at 50 M, 0.755 at 100 M).
  Prototype at a = 0.9: δ̄ = 0.214, ε = 0.075, M₅₀₀/M₁₀₀₀/M₂₀₀₀ = 0.083/0.098/0.104; at a = 0: δ̄ = 0.146,
  ε = 0.053.
- **Table:** (δ̄, ε, ⟨d⟩, ⟨d²⟩, ⟨d³⟩) at a = 0, 0.3, 0.6, 0.9, computed by `scripts/calibrate-flux.ts`,
  hardcoded in both twins, interpolated linearly in a (clamped to [0, 0.9]).
- **Slider** s ("Flux variability", 0–1.4, default 1): f = (1 − s d(t)) (1 + s ε n(t)) / ⟨…⟩. s = 0 is exactly the
  steady jet.
- **Mean and moments** (n independent of d, symmetric): with c₂ = E[clip(n)²] (computed once):
  μ = 1 − s⟨d⟩; ⟨f²⟩ = ⟨(1 − s d)²⟩ (1 + s² ε² c₂) / μ²; ⟨f³⟩ = ⟨(1 − s d)³⟩ (1 + 3 s² ε² c₂) / μ³.
  These feed the existing mean-power cubic (`meanInjection`), so the time-averaged injected power stays η P_BZ.

### 2.3 Jet response

Unchanged from the jet-knots spec: width × √f, density × f, at the plasma's launch time; filaments unchanged.

## 3. Code

- `src/physics/flux-history.ts`: `FLUX` gains the flicker constants (cell 50, clip 3, salt); `FLUX_SPIN` table;
  `fluxParams(a)` (interpolated δ̄, ε, ⟨dⁿ⟩); `fluxFlicker(t)`; `fluxDeficit(t, dbar)` (unchanged algorithm);
  `fluxRatio(t, s, a)`; `fluxMoment(n, s, a)`; `modulationIndex(series, window)`; `measureFlux` replaced by a
  calibration helper that returns the modulation indices, series σ/μ and moments.
- `scripts/calibrate-flux.ts`: solves (δ̄, ε) per table spin and prints the table.
- `src/render/emission-shared.wgsl`: `fluxRatioJ(epoch, rel, s, a)` adds the flicker (time split in 50 M cells by
  the existing `splitPeriodJ`) and the spin-interpolated table; `jetShapeJ` passes its `a` through.
- `src/physics/synchrotron.ts`: `meanInjection(…, fluxVar, a)` uses `fluxMoment(n, s, a)`.
- `src/physics/jet.ts`: `jetShape` passes `a`; `JET_ENV_Q` stays 1.51 if the §5 envelope test holds at every
  spin (prototype: f_max ≈ (1 + 1.4 · 0.075 · 3) / (1 − 1.4 · 0.11) ≈ 1.55 → √ ≈ 1.25 → 1.2 × 1.25 ≈ 1.50,
  just inside 1.51; if the calibrated table pushes it over, JET_ENV_Q is raised and BOOKMARK_FRAC re-checked).
- `src/test/parity.browser.ts` + `src/render/flux-parity.wgsl`: flux cases gain the spin.
- Docs: a correction paragraph in the README jet-knots section; ROADMAP follow-up 1 closed.

## 4. On screen

Fine knots stream up the jet continuously (flicker cells of 50 M carried at β ≈ 0.2–0.6: ~10–30 M apart), with a
shallower eruption front (~21 % instead of ~51 % at a = 0.9) every ~75 s at default Motion.

## 5. Tests and gates

**CPU (`npm test`):**
- Per table spin, over 10⁶ M at s = 1: 1000 M modulation index = target ± 0.002; M₅₀₀/M₂₀₀₀ ≥ 0.80;
  δ̄ = 2 σ/μ ± 1 %; hardcoded moments match the generator (1e-4 relative).
- Flicker: deterministic; variance of clip(n) = c₂ ± 0.01 over 10⁶ M; autocorrelation at 100 M lag < 0.05;
  |n| ≤ 3.
- Closed-form ⟨f⟩ = 1, ⟨f²⟩, ⟨f³⟩ vs brute force at s ∈ {0.5, 1, 1.4} and a ∈ {0, 0.45, 0.9, 0.998}.
- Envelope: no f exceeds (JET_ENV_Q / 1.2)² at any spin, slider value or time; existing envelope sweep extended
  over spins.
- s = 0: f ≡ 1; spin interpolation continuous and clamped.
- Existing jet, propagation and energy tests updated for the spin argument; WGSL constants match the CPU table.

**GPU (`npm run verify:gpu`):** `?parity` flux cases at spins 0, 0.6, 0.9, 0.998 and epochs to 2048 × 8000 M;
mutation of the flicker cell (50 → 51) and of one table entry must fail; `?golden` re-recorded (jet-off via the
known jet-free-kernel check); `?cachecheck` 0.00e+0; app checks pass.

**Visual:** frames across an eruption at the default view; knot spacing measured along the jet.
