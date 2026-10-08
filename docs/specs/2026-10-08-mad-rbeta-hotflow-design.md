# Magnetically arrested electron heating for the hot flow (M87*'s ring) — design

**Date:** 2026-10-08
**Status:** implemented (plan `docs/plans/2026-10-08-mad-rbeta-hotflow.md`, 2026-10-08). Results: M87\* 41.8 µas
(GPU and CPU twin), gated inside 2σ and robust to the blur; Sgr A\* 57.3 µas, reported (+2.4σ). Departures, each in the
plan's ledger: (1) ring robustness is required of the EHT-gated ring only (user decision: Sgr A\*'s ring is
flat-topped, ~51–61 µas); (2) the CPU twin rings are measured on the GPU's 28 M frame at 192² (`TWIN_GRID`; on the 96²,
26 M calibration grid Sgr A\*'s flat-topped peak moved by up to 8 µas), and Sgr A\*'s twin tolerance is 3 µas, not 0.5;
(3) the hotspot amplitude is A₀ 692 (bisection bracket widened: the cold midplane electrons need ~600×); (4) `?accuracy`
judges intensity where the reference or the renderer reaches 1e-9 of the scene peak (the GPU's density cutoff), holds the
mm scenes to this renderer's own statistics with headroom (final review: the old renderer's were 30-60x looser), and 16 % / 25 % of the rebuilt mm reference
rays, nearly all faint, no longer converge to 1e-6 and go unscored.
**Builds on:** `docs/specs/2026-10-04-hot-flow-mm-design.md` (the hot flow and the 1.3 mm view),
`docs/specs/2026-10-04-mm-hotspots-design.md` (hotspots borrow the flow's electrons).

## 1. Intent (agreed)

- **Fix M87\*'s 1.3 mm ring.** The hot flow renders it at 35.6 µas against the EHT's 42 ± 3 (Paper VI): 2.1σ small,
  recorded as a limitation since 2026-10-04.
- **Done means gated:** M87\*'s ring inside the EHT's 2σ (36–48 µas) with a published, physically motivated change.
  If the honest physics does not land there, report it instead of tuning (user's rule).
- **One model for every hot flow** (user's choice): the same physics for Sgr A\*, M87\* and any Custom hot flow. Where
  it moves Sgr A\* past its own 2σ, Sgr A\*'s EHT comparison is reported, not gated.
- The visible view, the jet, the disk and the integrator are unchanged.

## 2. What the measurements showed (CPU twin, n₀ refitted to the measured flux in each case)

Probe `scripts/scratch-ringsens.ts` (git-excluded), 80²–96², ring = peak of the radial profile as in `?hotflow`.

| Change | M87\* ring | Sgr A\* ring |
|---|---|---|
| today's RIAF (Broderick, Pu 50/50, T_e ∝ r^-0.84, β 10) | 35.6 µas | 48.2 µas |
| GRMHD-calibrated MAD velocity (arXiv 2607.21852: α = β = 0.95, w = 1) | 35.5 | 48.0 |
| pure free fall | 37.4 | 53.5 |
| density slope r^-1.5 / thickness H 0.5 | 35.4 / 35.5 | 48.2 / 47.7 |
| flatter T_e (r^-0.5, r^-0.25) | 36.0 / 36.3 | — |
| **R-β, R_high 160, R_low 1, midplane β 1** | **42.0** (15 µas blur 42.0) | **57.1** (56.7) |
| R-β, R_high 80, β 1 | 41.9 | 57.3 |
| R-β, R_high 160, β 10 | 42.2 | 59.8 |
| R-β, R_high 20, β 1 / β 10 | 35.4 / 35.5 | 46.2 / 52.2 (bimodal) |

Velocity, density and the temperature slope barely move the ring. The photon ring of M87\* at a = 0.9 seen face-on is
~9.9 θ_g ≈ 38 µas; the EHT's α = D/θ_g ≈ 11 needs the 230 GHz light concentrated outside it, which today's T_e (hottest
at the horizon) does not give. Where the electrons are hot is the lever, and the R-β prescription of the EHT's own
GRMHD libraries (cold electrons where gas pressure dominates, hot where the field does) moves it there.

## 3. Physics

Everything below replaces `flowTemperature` and `flowField` in the hot flow only. Density (Broderick et al. 2011:
n ∝ r^-1.1 e^(−z²/2ρ²)), velocity (Pu et al. 2016, 50/50), the thermal synchrotron emissivity (Mahadevan et al. 1996
fit), Kirchhoff absorption and the transfer are unchanged.

- **Ions, virial:** k T_i = m_p c² / (3 r) (r in M), i.e. T_i = 3.63 × 10¹² K / r.
- **Field, magnetically arrested:** set by the midplane density n_eq = n₀ (r/2)^-1.1, not the local one, with
  Broderick's normalisation at midplane plasma β_eq = 1 (MAD; today β = 10 everywhere):
  B²/8π = n_eq m_p c² r_S / (12 r β_eq), r_S = 2M.
- **Local β** (ion pressure over magnetic): β = 8π n k T_i / B² = 2 β_eq (n / n_eq) = 2 β_eq e^(−z²/2ρ²). It falls off
  the midplane toward the funnel.
- **Electrons:** T_e = T_i / R(β), R(β) = R_high β²/(1 + β²) + R_low/(1 + β²), **R_high = 160, R_low = 1** (the EHT's
  GRMHD libraries span R_high 1–160; 160 is the top of that range, the coldest disk electrons). Cold midplane electrons
  (R = 128 at the midplane, where β = 2 β_eq = 2; R → 160 for β ≫ 1), hot funnel-wall electrons (R → 1).
- Constants live in `HOTFLOW` (CPU) and `emission-shared.wgsl` (GPU) as twins (`tests/jet.test.ts` checks the twinned
  constants). The WGSL keeps its log-space form (ln T_e, ln B) so f32 stays in range.

**Calibration (scripts, as today):**
- n₀ refitted to the measured 230 GHz flux, Sgr A\* 2.4 Jy and M87\* 0.5 Jy (`scripts/calibrate-hotflow.ts`, 96²).
  The probe gave n₀ ≈ 1.4 × 10⁷ (Sgr A\*) and 1.1 × 10⁵ cm⁻³ (M87\*). Other hot-flow objects keep scaling n₀ as λ / M
  from the calibrated object nearest in mass.
- Hotspot amplitude A₀ refitted to ALMA's +0.3 Jy for Sgr A\* (`scripts/calibrate-hotspot.ts`), since hotspots boost
  the flow's own electrons at its temperature.
- `HOTFLOW_TARGETS.cpuRingUas` updated to the CPU twin's new rings.

## 4. Gates

- **`?hotflow`:** flux ±5 % (both); ring within ±0.5 µas of the CPU twin (both); **M87\* inside the EHT's 2σ (gated)**;
  Sgr A\*'s EHT comparison reported (no longer gated) with the README stating it; the hotspot twin as today.
- **Ring robustness (new, both objects, CPU and GPU):** |ring(unblurred) − ring(15 µas blur)| ≤ 3 µas. It fails the
  bimodal profiles the probe saw at R_high 20 (Sgr A\* 52.2 vs 47.6), so a gated ring is a real ring.
- **`?parity`:** the flow cases recomputed against the new CPU coefficients (same tolerances).
- **`?golden`:** mm hashes re-recorded (sgra-mm, m87-mm, sgra-mm-hotspot); the five visible hashes unchanged
  (bit-identical: the visible view never evaluates the hot flow).
- **`?cachecheck`** exact.
- **`?accuracy`:** its 1.3 mm rows (`accuracy-ref.json`) encode the old emission and are rebuilt with the new
  coefficients (`scripts/build-accuracy-ref.ts`, mm scenes only). The pre-Mino renderer that recorded
  `accuracy-old.json` no longer exists, so its mm entries cannot be re-recorded under the new emission: for the two mm
  scenes the gate keeps the old renderer's recorded statistics as fixed bounds (pixels above floor 261 / 235, worst
  ratio 38.4 / 19.3 for sgra-mm / garg-mm, measured 2026-10-08) and drops the per-pixel old comparison there. Geometry
  scenes are unchanged.
- `npm test`, `npm run build`, every app check.

## 5. Risks and limitations (stated in the README)

- If at 96² or on the GPU M87\*'s ring leaves 2σ, it is reported, not tuned.
- Sgr A\* moves from 48 µas (−1.6σ) to about 57 µas (+2.3σ, 0.7 µas past the edge). Both objects come out at
  α ≈ 11.0, as GRMHD images do; the EHT's Sgr A\* ring is α ≈ 10 for the preset's mass and distance.
- R-β with virial ions and a midplane-set field is a semi-analytic stand-in for GRMHD: no turbulence, no flux
  eruptions in the field, no funnel geometry beyond the density's Gaussian.
- The dense hot flows (default view, λ 3.7e-4) change appearance: cold midplane electrons change their opacity.
- Velocity is unchanged (the GRMHD-calibrated MAD profile moved neither ring); a follow-up, not this change.

## 6. Out of scope

Velocity field changes, positron or non-thermal electrons, polarization, a jet at 1.3 mm (decision of 2026-10-04
stands: this is the hot flow's own electrons, not the optical jet model), the visible view.

## References

- EHT Collaboration 2019, M87\* Papers V–VI (R-β GRMHD libraries; ring 42 ± 3 µas, α ≈ 11).
- EHT Collaboration 2022, Sgr A\* Papers I, IV, V (ring 51.8 ± 2.3 µas; MAD favoured).
- Mościbrodzka, Falcke & Shiokawa 2016 (R-β electron heating).
- Broderick et al. 2011/2016 (RIAF density, β normalisation); Pu, Akiyama & Asada 2016 (velocity).
- Saurabh, Wielgus et al. 2025, arXiv 2508.11760 (semi-analytic M87\* RIAF with R-β; α ≈ 11 target).
- arXiv 2607.21852 (GRMHD-calibrated sub-Keplerian velocity; tested, not adopted).
