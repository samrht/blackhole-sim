# Jet knots from horizon-flux variability — design

**Date:** 2026-10-03
**Status:** approved 2026-10-03; §2.3, §2.4, §2.5 corrected while planning (see "Corrections from planning"); implemented on feat/jet-flux-knots (plan `docs/plans/2026-10-03-jet-flux-knots.md`)
**Roadmap:** `docs/ROADMAP.md` queue item #3, "the jet's own churn and knots". Both are decorative value noise
today: the knots are 1-D noise sliding outward at the local flow speed, and the cross-funnel "churn" is a noise
pattern frozen in space (it never moves at all).

## 1. Intent (agreed)

- The knots become what MAD simulations say the jet's structure near the black hole is: the **imprint of the
  horizon magnetic-flux history** φ_BH(t), launched at the base and carried outward by the plasma (internal-shock
  driving). The user chose internal shocks; the honest refinement found while researching is that shells collide
  mostly beyond the 60 M view, so inside the view what is seen is the *driver* of the shocks: the varying power and
  width launched from the base.
- Timescales, shape and amplitude of φ_BH(t) come from published GRMHD simulations (§2.1). Nothing is tuned by eye.
- The churn becomes **filaments frozen into the moving plasma**: carried outward at the flow speed and twisting with
  the field-line rotation. The motion is physical; the strength is not measured and is labelled illustrative.
- **Energy conservation:** the time-averaged jet power stays η·P_BZ, so the M87 η calibration
  (cooled-jet spec) still holds.
- Not in scope: shell collisions/shock compression beyond the view, spin dependence of the flux statistics
  (one calibration, a = 0.9, for all spins; follow-up), coupling the eruptions to the disk (Ṁ drops, hot spots),
  flow streamlines that bend with the width change.

## 2. Physics

### 2.1 Measured properties (the targets)

| Property | Value | Source |
|---|---|---|
| Eruption recurrence | 1000–2500 M, ~1500 M typical | arXiv 2510.25842 ("Demystifying flux eruptions"), §III.2, eq. 42 |
| Drop shape | quasi-exponential decay, e-folding 500 M (converged, plasmoid-mediated reconnection rate 0.01c), lasting a few hundred M | Ripperda et al. 2022 (arXiv 2109.15115), Fig. 4E, Appendix C |
| Rebuild | slow: flux re-advected with the gas between eruptions ("slow rise, fast decay") | Ripperda et al. 2022; review arXiv 2407.15929, Fig. 3 |
| Swing | σ/μ of φ_BH = 0.209 at a = 0.9 (MAD); other simulations 0.11 (2510.25842 thick disk) | Narayan et al. 2022 (arXiv 2108.12380), Table 3 |
| Jet response | P_jet ∝ φ_BH²; flux drop then rise → jet narrows/truncates then widens: a thin–thick pattern that travels with the plasma to several hundred r_g; jet light lags φ_BH by ~50 M | Tsunetoe, Narayan & Ricarte 2024 (arXiv 2411.08116), §III, §IV.1 |

The 2σ eruption-depth rule of 2510.25842 (§IV.3) describes their small-amplitude eruptions and is not used: with
the sawtooth below, a 21 % rms swing requires a mean drop of ≈ 51 %, which at a 500 M e-folding lasts ≈ 360 M —
Ripperda's "a few hundred M". The two measured numbers (swing, e-folding) are consistent; the 2σ rule is not.

### 2.2 Flux history φ(t)

A deterministic, seeded sawtooth in units of the saturated flux φ_sat:

- **Eruption times** t_k = T (k + 0.5 + j u_k), T = 1500 M, j = 1/3, u_k ∈ [−0.5, 0.5) a hash of k. Gaps
  t_{k+1} − t_k = T (1 + j (u_{k+1} − u_k)) lie in [1000, 2000] M, inside the measured 1000–2500 M.
- **Depth** δ_k = δ̄ (1 + 0.5 v_k), v_k ∈ [−0.5, 0.5) a second hash of k (so δ_k ∈ [0.75, 1.25] δ̄). δ̄ is
  calibrated (§2.6) so the series rms is σ/μ = 0.209 at slider 1; the prototype gives δ̄ = 0.509, ⟨d⟩ = 0.262,
  ⟨d²⟩ = 0.0925 (u, v from a 32-bit integer hash; the plan's implementation re-derives the constants).
- **Drop** over D_k = −τ_d ln(1 − δ_k), τ_d = 500 M: the deficit d(t) = 1 − e^{−(t−t_k)/τ_d} for
  t_k ≤ t < t_k + D_k (reaching δ_k; D_k ≤ 510 M < the shortest gap).
- **Rebuild** linear: d falls from δ_k to 0 between t_k + D_k and t_{k+1}.
- **Slider** s = "Flux variability" ∈ [0, 1.4], default 1: φ(t) = max(0.05, 1 − s·d(t)). The shape (D_k, gaps) is
  fixed at its s = 1 values; s scales the deficit linearly, so s = 1 is the calibrated simulation and other values
  are labelled as a scaled swing.
- **Mean normalisation:** f(t) = φ(t)/φ̄(s), φ̄(s) = 1 − s⟨d⟩. Two constants ⟨d⟩, ⟨d²⟩ are measured once from the
  generator (exact integration over 10⁶ M) and hardcoded in both twins; ⟨f²⟩(s) = (1 − 2s⟨d⟩ + s²⟨d²⟩)/φ̄². (The
  0.05 floor is never reached: max δ_k · s_max = 1.25 · 0.509 · 1.4 = 0.89 < 0.95; a §5 test enforces it, and the
  slider max is lowered if the re-derived δ̄ breaks it.)
- **Evaluation at any time:** k₀ = floor(t/T); eruption k ∈ {k₀−1, k₀, k₀+1} with the largest t_k ≤ t is the
  current cycle. Time is passed as clock epoch + remainder (as the turbulence does): epoch − k·T is exact in f32
  (both integers < 2²⁴), so precision holds after hours of play.

### 2.3 Launch time and travel time τ(z)

Plasma at height |z| left the base at t_launch = t_emit − τ(|z|), τ(z) = ∫₀^z dz'/β(z'), with the existing flow
law Γβ = A z^p, A = √(G₂₈₀² − 1)/280^p, p = 0.58. Substituting x = A^{1/p} z:

  τ(z) = A^{−1/p} F(A^{1/p} z),  F(x) = ∫₀^x √(1 + x'^{2p}) x'^{−p} dx'

so one universal function F serves every Jet speed. GPU: F(x) = x^{1−p}/(1−p) · G(x) with G a 64-entry table on
log x ∈ [10⁻⁴, 10²] (G → 1 as x → 0; linear extrapolation in x above 10², where F ≈ x), stored as WGSL constants in
`emission-shared.wgsl`, linearly interpolated in log x. CPU twin: exact adaptive integral. Required: relative error
< 10⁻³ over z ∈ [0, 1000] M for G₂₈₀ ∈ [1.5, 8]. τ(60 M) ≈ 166 M at G₂₈₀ = 2 (82 M at 5, 246 M at 1.5).

Light-travel delay needs nothing new: each jet sample already has its own emission time.

### 2.4 Jet response

At a jet sample with launch-time flux ratio f = φ(t_launch)/φ̄:

- **Width:** funnel edge ρ_f(z) → ρ_f(z)·√f. Flux conservation (Φ ∝ B_p ρ²) with the jet edge in pressure balance
  with its surroundings (fixed B at fixed external pressure) gives ρ ∝ Φ^{1/2}.
- **Field strength:** unchanged (same pressure balance), so the cooling table, colours and absorption apply as
  they are.
- **Density per volume:** × f (the shape factor that multiplies injected power and absorption). Power per unit
  length ∝ f · (√f)² = f² ∝ φ², as P_jet ∝ φ_BH².
- **Mean power:** `jetQ0` is divided by ⟨f²⟩(s) (§2.2), so the time-averaged injected power is η·P_BZ at every
  slider value.
- Flow directions stay on the steady streamlines (q = ρ/ρ_f(z) of the unscaled funnel): the width change (√f ≈ 0.70–1.16 at s = 1)
  bends them by a few degrees, below what is visible.

### 2.5 Filaments (replacing the churn)

A noise field constant along each plasma parcel's path:

- **Along the jet:** t_launch.
- **Across:** the field-line label q = ρ/(ρ_f(z)√f).
- **Around:** the co-moving azimuth φ_c = φ − Ω_F (τ(z) − z). Plasma on a field line rotating at Ω_F moves
  azimuthally at Ω = Ω_F (1 − β) (v_φ = Ω_F ρ + v_p B_φ/B_p with B_φ/B_p ≈ −Ω_F ρ/c); integrated along the path,
  ∫Ω dt = Ω_F ∫(1/β − 1) dz = Ω_F (τ(z) − z). Ω_F = Ω_H/2 = a/(4 r₊) (Blandford–Znajek).
- Factor 1 + 0.35 (n − 0.5)·2, mean one (the current churn amplitude), n a smooth value noise with cells of 25 M in
  t_launch, 8 around, 3 across the wall (illustrative; stated as such in the UI tooltip and README).

### 2.6 Calibration

δ̄ is solved once (bisection, CPU) so the generator's φ(t) has σ/μ = 0.209 at s = 1 over 10⁶ M; δ̄, ⟨d⟩, ⟨d²⟩ are
recorded as constants with the script/test that reproduces them.

## 3. Code

- **`src/render/emission-shared.wgsl`** (sole GPU copy; CPU twins in `src/physics/jet.ts`):
  - `fluxHistoryJ(epoch, tRel, s) -> f32` — f = φ/φ̄ at launch time epoch + tRel.
  - `launchDelayJ(z, g280) -> f32` — τ(|z|) from the G table.
  - `filamentsJ(q, phiC, epoch, tRel) -> f32` — replaces the churn.
  - `jetShapeJ` gains φ and splits time as epoch + remainder; `knotsJ` and `JET_KZ`/`JET_SEED` knot use removed.
- **`src/render/raytrace.wgsl`**: jet sample call passes φ and (U.timeEpoch, emitRel(...)).
- **Uniforms:** no new slots. `jetKnots` → `fluxVar` (s); `jetQ0` includes 1/⟨f²⟩(s); `jetUniforms` gains s.
- **Cache safety:** `jetTouches` / `inJetEnvelope` widen the envelope by the largest width factor √(f_max) = 1/√φ̄(s_max) ≈ 1.26 at the
  slider maximum, so a ray cached as missing the jet cannot be crossed by a widened jet.
- **UI:** "Jet knots" → "Flux variability" (0–1.4, default 1.0); tooltip: 1× = simulated φ_BH swing at spin 0.9
  (Narayan+2022); filaments illustrative. Presets' knot values become flux values (all 1.0).
- **Docs:** README physics section, ROADMAP item #3 closed with follow-ups.

## 4. What changes on screen

Default view (G₂₈₀ = 2, 60 M jet, 20 M/s): at each eruption (~every 75 s) a dimming, narrowing front climbs the
jet in ~8 s, followed by slow re-brightening and re-widening; filaments stream outward and twist (period
2π/Ω_F ≈ 40 M ≈ 2 s at the base for a = 0.9, slowing as the flow accelerates since Ω = Ω_F(1 − β)).

## 5. Tests and gates

**CPU (`npm test`):**
- Generator over 10⁶ M: mean gap 1500 ± 50 M, all gaps in [1000, 2000]; drop segments exponential with e-fold
  500 M; σ/μ = 0.209 ± 0.01 at s = 1; φ ≥ 0.05 never clamped for s ≤ max; deterministic.
- Epoch split: f at epoch 2048·n + remainder equals an f64 reference at large n.
- τ table vs exact integral: relative error < 10⁻³ (range in §2.3).
- Propagation: f(t_launch) at the base reappears at z after τ(z); the filament value is constant along a parcel
  path (z(t), φ(t)) integrated with the flow law and Ω_F(1 − β).
- Energy: time-averaged power per unit length at s ∈ {0, 0.5, 1, 1.4} equals the s = 0 value within 1 %.
- Envelope: jetShape is zero outside the widened inJetEnvelope, swept over f and positions.
- Shader-twin guard (tests/shader-twins.test.ts) still passes.

**GPU (`npm run verify:gpu`):**
- `?parity`: new cases for fluxHistoryJ, launchDelayJ, filamentsJ, jetShapeJ from the shared fragment; a
  mutation of each must fail.
- `?golden`: re-recorded by design; jet-off scenes bit-identical or ~1 ulp (the known compiler-rescheduling
  effect at jet-code changes, proven by a jet-free kernel).
- `?cachecheck`: cached = live with the jet widening and narrowing.
- `?shadow` unchanged; build; probe scripts; every preset warning-free.

**Visual check:** headless frame capture across one eruption at the default view and M87*; frames attached to
the report.

## Corrections from planning (2026-10-03)

1. **τ(z) by quadrature, not a table (§2.3).** τ(z) = ∫_{z_base}^{z} dz'/β(z') — measured from the jet base
   z_base = 2 M where emission starts (as `flowTime` already does), not from 0, where the flow law's β → 0 adds an
   unphysical ~48 M. Substituting v = z^{1−p}: τ = [A (1 − p)]⁻¹ ∫ √(1 + A² v^{2p/(1−p)}) dv over
   [z_base^{1−p}, |z|^{1−p}], integrand smooth; **6-point Gauss–Legendre** on that interval agrees with the exact
   integral to 2.9 × 10⁻⁵ (G₂₈₀ 1.5–8, z ≤ 1000 M), 35× better than the planned 128-entry table (2.8 × 10⁻⁴) and with
   no constant array. CPU twin: the same quadrature; the test compares both with a 20 000-point reference.
   τ(60 M) = 166 M at G₂₈₀ = 2.
2. **Mean-power normalisation (§2.4).** The toroidal field B_φ = B_p ρ Ω_F grows with radius, so a widened jet
   (field lines at larger ρ) injects slightly more than f² per unit length: injected power is
   P(f) ∝ f² (A + C f) (B_p unchanged, B_φ² ∝ f at fixed field-line label). `jetQ0` is therefore
   η·P_BZ / ⟨P(f)⟩: P is evaluated at f = 0.5, 1, 1.5, 2 (the existing volume sum with the wall scaled by √f), the
   exact cubic through them is averaged with the generator's moments ⟨f²⟩, ⟨f³⟩ (constants ⟨d⟩, ⟨d²⟩, ⟨d³⟩). The
   §5 energy test is unchanged: the time average equals the steady value within 1 %.
3. **Filament twist (§2.5).** With τ measured from z_base, φ_c = φ − Ω_F (τ(|z|) − (|z| − z_base)).
4. **Files (§3).** The generator lives in a new `src/physics/flux-history.ts` (jet.ts imports it); its WGSL twin is
   in `emission-shared.wgsl` as specified. Filament noise is a 3-D value noise built on the turbulence's integer
   hash (`hash4` / `hash4T`), periodic in azimuth.
