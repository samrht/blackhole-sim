# Eruption flares: flux tubes on the disk — design

**Date:** 2026-10-04 (session of 2026-10-03)
**Status:** SHELVED 2026-10-04 (see "Why shelved" at the end); branch feat/eruption-flares kept for reference, not merged
**Follows:** jet-knots follow-up 2 ("coupling eruptions to the disk"); builds on
`docs/specs/2026-10-03-flux-statistics-design.md` (the eruption history it reuses).

## 1. Intent (agreed)

- Each horizon-flux eruption ejects a **flux tube** into the disk that is heated by reconnection and shines as an
  orbiting **hot spot** — the Sgr A* near-infrared flare picture (Dexter et al. 2020; Porth et al. 2021; Ripperda et
  al. 2022; GRAVITY 2018–2023 observe orbiting hot spots). The jet's eruption front and its disk flare come from
  the same event.
- **Rendered on the disk plane** (user's choice): each tube is a patch where rays hit the disk, adding optically
  thin synchrotron light; rays passing just above or below the disk near a tube miss it (stated limitation).
- **Keplerian orbits** (user's choice: what GRAVITY observes; simulations of thick MAD flows find ~½ the local,
  already sub-Keplerian speed — the tension is noted, not modelled).
- **Energy anchored to observation:** a default Sgr A* tube radiates a typical NIR flare's energy, 10³⁸ erg
  (a typical flare: ≈ 4 mJy at 1.6 µm for ≈ 30 min; Yusef-Zadeh et al. 2006; GRAVITY 2020 flux distribution).
- Replaces the illustrative "Flares" (fixed Gaussian hot spots, `nSpots`, the hot-spot storage buffer).
- Why not "inner-disk dimming": no thin-disk simulation measures the disk's light change during eruptions
  (arXiv 2510.25842's thin MAD reports flux transport only), so its size and depth would be invented. The
  accretion-rate correlations of Narayan et al. 2022 are for φ_BH = Φ/√Ṁ, mixed by construction.

## 2. Physics

### 2.1 Measured properties (targets)

| Property | Value | Source |
|---|---|---|
| One tube per eruption, quasi-periodic (1000–2000 M) | — | Porth et al. 2021 §2.2 |
| Circularisation radius | 5–40 r_g (thick MAD); thin MAD flux penetrates to ~30 r_g | Porth 2021 abstract; arXiv 2510.25842 §III.2 |
| Size | mostly ≲ 2 M, up to 7 M; Δr/r 0.1–0.3 | Porth 2021 §2.5–2.6 |
| Lifetime | observed ≤ ~2 orbits (dissolved by expansion and Kelvin–Helmholtz) | Porth 2021 Discussion |
| Tube magnetic energy (Sgr A*) | 3×10³⁶ – 3×10³⁸ erg (prograde) | Porth 2021 §2.5 |
| Emission | thermal negligible; reconnection-accelerated non-thermal electrons needed | Porth 2021; Ripperda 2022 |
| Typical Sgr A* NIR flare | ≈ 4 mJy at 1.6 µm, ≈ 30 min, ≈ 10³⁸ erg | Yusef-Zadeh et al. 2006 |

### 2.2 Tube k

From the flux history (`flux-history.ts`, slider s = Flux variability): eruption time t_k, depth δ_k s, drop
duration D_k = −τ_d ln(1 − δ_k) (the s = 1 shape, as for the jet).

- **Orbit radius** r_c,k = max(5 + 25 u_k, r_ISCO + 2R_k) with u_k ∈ [0, 1) a hash of k (uniform in r is an
  assumption; the sources give the range only). **Starting azimuth** φ₀,k = 2π v_k.
- **Size** R_k = 0.2 r_c,k (round patch on the disk plane, Gaussian profile exp(−d²/2R²) in the in-plane distance d).
- **Path:** for 0 ≤ τ = t − t_k < D_k the tube spirals out, r(τ) = r_in + (r_c − r_in) τ/D_k, azimuth
  φ(τ) = φ₀ + ∫₀^τ Ω_K(r) dτ' (6-point Gauss, as the jet's travel time); afterwards it orbits at r_c with
  φ = φ(D_k) + Ω_K(r_c)(τ − D_k), Ω_K = 1/(r^{3/2} + a).
- **Light curve** A(τ): linear rise τ/D_k during the spiral; then exp(−τ_c/(P/2)) · (1 − smoothstep(1.5P, 2P, τ_c))
  with τ_c = τ − D_k and P = 2π/Ω_K(r_c) (fades by half an orbit's e-fold, gone after 2 orbits).
- At most three tubes are alive at once (life < D_k + 2P ≤ 2500 M; eruptions ≥ 1000 M apart): a disk pixel
  evaluates eruptions k−2…k for its emission time.

### 2.3 Energy and emission

- **Released energy** E_k = δ_k s · E_mag with E_mag = Φ²/(8π² r_g) (the horizon field's energy in a volume
  π r_g³; Φ from `jetEnergetics`).
- **Radiated energy** ζ E_k · f (f = the **Eruption flares** slider, 0–3, default 1); ζ is fixed once so the default
  Sgr A* preset's mean tube (δ̄) radiates 10³⁸ erg at f = 1 (anchor test, §5).
- **Field** B_k: the released energy in the tube volume V = πR²·2R: B_k = √(8π E_k / V).
- **Emission:** fast-cooling synchrotron exactly as the jet (cooled-jet spec): injected power density
  P(τ) = ζ E_k f A(τ)/∫A dτ / V, electrons radiating it through the existing cyclo-synchrotron table at field B_k
  and cooling depth from B_k² × age (q set from P/B² with the jet's injection constant). Column: optically thin
  slab of plasma-frame path 2R through the tube; plasma moves with the disk (Keplerian), so the photon shift is the
  disk's g at that point; three bands → the jet's band matrix → the disk's units (× lumNorm), added to the disk
  pixel. Light-travel delay: the disk pixel's own emission time.

## 3. Code

- `src/physics/eruption-spots.ts` (new, CPU twin): `tubeOf(k, s, a)` (t_k, r_c, R, φ₀, D_k, E_k),
  `tubeAt(k, t, s, a)` (r, φ, A), `tubeColumn(...)` (per-band intensity), `ERUPTION_FLARE` constants, ζ.
- `src/render/emission-shared.wgsl`: twins; called from the disk shading (`shadeDisk`), shading-only — the
  geodesic cache re-shades tubes like the disk turbulence (no new geometry, no bookmarks).
- `src/render/uniforms.ts`, `raytrace.wgsl`, `present.wgsl`: drop `nSpots`; add `flareStrength` (f) and
  `flareEUnit` (ζ E_mag in erg for the current object); the block grows 144 → 160 bytes.
- `src/render/gpu.ts`: drop the hot-spot buffer and `uploadHotSpots`; `src/physics/emission.ts`: drop
  `HotSpot`/`hotspotField` and their tests.
- `src/main.ts`, `index.html`: "Flares (illustrative)" → "Eruption flares" (0–3×, default 1×, tooltip with the
  anchor); `scripts/shot-delay.mjs` and `scripts/bench.mjs`, `src/test/scenes.ts` use the tubes.
- Docs: README section; ROADMAP follow-up 2 done.

## 4. On screen

Every ~1500 M (≈ 75 s at default Motion) a patch brightens near the inner disk, spirals out to 5–30 r_g over
~360 M, then orbits for up to two orbits while fading — Doppler-brightened on the approaching side, lensed into the
photon ring, and (light delay on) echoed. Its brightness relative to the disk follows the energy anchor, so it is
prominent for Sgr A* (a faint, hot-flow source in reality; here a thin disk at the preset's accretion rate) and
faint against bright disks.

## 5. Tests and gates

**CPU (`npm test`):** births at eruption times; r_c ∈ [5, 30] (and ≥ r_ISCO + 2R); R = 0.2 r_c; spiral reaches r_c
at D_k with continuous azimuth; Keplerian angle afterwards; A continuous, zero after 2P; ≤ 3 alive; f = 0 or s = 0
→ no emission; outside every patch the disk shading is unchanged; **anchor:** the default Sgr A* mean tube's
radiated energy (∫ over time of the column-integrated luminosity, all bands of the table's spectrum) = 10³⁸ erg
± 5 % at f = 1; energy scales ∝ δ_k s f Φ²/r_g.

**GPU (`npm run verify:gpu`):** `?parity` cases for the tube functions (several k, times, spins 0.3 / 0.9, epochs to
2048 × 8000 M) and the column intensity; a mutation (e.g. R = 0.2 → 0.21 r_c) fails; `?golden` re-recorded;
`?cachecheck` 0.00e+0; new app check: Sgr A* preset with flares on shows a tube patch brighter than its
surroundings during a tube's life.

**Visual:** Sgr A* frames across a tube's life (spiral, orbit, fade) and the light-delay echo pair.

## Corrections from planning (2026-10-03)

1. **Tube field by flux conservation (§2.3).** The tube carries the ejected flux δ_k s Φ through its cross-section:
   B_k = δ_k s Φ / (π R_k² r_g²) — instead of the released horizon energy spread over the tube volume. The tube's own
   magnetic energy is E_k = B_k² V / 8π = (δ_k s)² Φ² / (4π² R_k r_g) (V = 2π R_k³ r_g³), and the tube **radiates
   ζ E_k f** (ζ: fraction of its field energy radiated).
2. **ζ from the anchor in closed form.** Over tubes, ⟨(1 + 0.5 v)²⟩ = 1 + 1/48 and ⟨1/r_c⟩ = ln 6 / 25 (uniform r_c on
   [5, 30]), so ⟨E_k⟩ = δ̄² (1 + 1/48) Φ² (ln 6 / 25) / (0.8 π² r_g); ζ = 10³⁸ erg / ⟨E_k⟩ at the default Sgr A* preset
   (s = 1) = **1.06 × 10⁻²** — about 1 % of the tube's field energy, Ripperda et al. 2022's own order-of-magnitude
   estimate for the radiated fraction of reconnection power. Computed at run time from the preset (`flareZeta()`).
3. **The anchor is on injected energy.** Measured with the synchrotron table (Sgr A*): radiated/injected = 1.00 at
   r_c = 5 (B ≈ 630 G), 0.80–0.93 at 17.5 (52 G, ages 300–1000 M), 0.44–0.65 at 30 (18 G): outer tubes are not fully
   fast-cooling within their life. The docs report this; the anchor test checks the injected energy, a second test
   bounds radiated/injected ≥ 0.4.
4. **Tube energies vs Porth et al.** Sgr A* tubes here hold 4 × 10³⁹ – 2.6 × 10⁴⁰ erg of field, 10–100× Porth's
   simulated tubes (≤ 3 × 10³⁸ erg); mostly this app's larger Sgr A* flux (Ṁ = 10⁻⁸ M☉/yr). Stated in the docs.
5. **Uniforms.** Slot 18 (`nSpots`, u32) becomes `flareStrength` (f32, f); new `flarePhi` (Φ, G cm²) and
   `flareZeta` (ζ) at floats 36–37; the block becomes 160 bytes.

## Why shelved (2026-10-04)

Implemented through the renderer (CPU model, WGSL twin, `?parity` 108 cases at 2.5e-5 in ln I, mutation-checked), then
measured: the brightest tube's visible luminance relative to the disk's peak, per preset — M87* 3.2e-4, Sgr A* 3.2e-4,
Gargantua 2.1e-3 (invisible), Cygnus X-1 225×, GRS 1915+105 167× (implausibly bright: the Sgr A* anchor scaled as
Φ²/r_g to ~10⁷ K disks that emit little visible light). The observed flare phenomenon (NIR flares 3–4× the quiescent
hot-flow emission of Sgr A*) belongs to hot, radiatively inefficient flows; this app's thin Novikov–Thorne disk is
orders of magnitude brighter in the visible than Sgr A*'s real flow, so physically anchored flares cannot be seen
against it. Coupling eruptions to the disk needs a hot-flow (RIAF) emission model first. The user chose to shelve;
main keeps the illustrative Flares.
