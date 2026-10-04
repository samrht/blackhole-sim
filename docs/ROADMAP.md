# Roadmap

What is next, in the order agreed on 2026-10-01. Shipped work lives in `docs/specs` and `docs/plans`;
this file is the queue. Each item gets its own spec (and plan) before code, as every tier has.

## Now

- [x] **`verify-gpu` fails on WGSL compile errors** (2026-10-01). Any console warning/error fails the
  route that logged it (favicon 404 excepted), and the first one ends the route's wait, so a broken
  shader fails in seconds with the WGSL message instead of as a strange number or a 10-minute
  timeout. `probe-axis` fails on diagnostics too (`probe-scale` already did).

## Next: physics on what the code already has

- [x] **Real-object presets with physical units** (2026-10-01). M87\* (M = 6.5e9 M_sun, a* ~ 0.9) and a stellar
  black hole such as Cygnus X-1 (~15 M_sun), from the master parameter table in
  `docs/specs/2026-05-30-relativistic-blackhole-accretion-design.md`: spin, inclination, disk
  temperature from mass and accretion rate, and readouts in physical units (horizon size, ISCO in
  km, orbital period at the ISCO in seconds/days). Uses physics already in the code; adds no new
  approximation.

## Correctness and gates (follow-ups recorded in the README)

- [x] **A true critical-curve gate for `?shadow`** (2026-10-03): a (xi, eta) classification image — every
  pixel traced with no emitter must match the analytic capture test (a = 0, 0.9, 0.998); 0 mismatches,
  area ratio 1.0000 (README `?shadow`).
- [x] **Shared-fragment parity for the jet and turbulence cases** (2026-10-02): `emission-shared.wgsl`
  is the sole copy, prepended by the renderer and ?parity; golden bit-identical; a changed jet
  constant now fails ?parity (2.0e-3); `tests/shader-twins.test.ts` guards against new copies.
- [x] **Vertical seam above the shadow** (2026-10-01): turbulence noise made 2π-periodic in ψ
  (hit azimuths of rays passing either side of the hole differ by 2π); the brightness step across
  the centre column above the shadow went 10.1 % -> 1.9 % (typical column step 3.5 %).
- [x] **Jet core line** (2026-10-01): already gone since the angular step caps bound the near field;
  the jet's own light is smooth to < 0.2 % across the α = 0 column at 72° and 8°. No jet change.

## Physics features (original Tier 2, not started)

- [x] **Synchrotron jet emission** (2026-10-02): magnetically arrested field, M87's measured
  acceleration, p = 2.4 electrons, GR transfer with self-absorption, brightness from ε × the
  Blandford–Znajek power (README "Synchrotron jet").
- [x] **Cooled jet** (2026-10-02): full-spectrum energy budget — electrons given η × P_BZ, cooled exactly by
  their own radiation, exact cyclo-synchrotron coefficients from a precomputed table, η fixed by M87's
  optical nucleus; the X-ray-binary jets now conserve energy (README "Cooled jet").
- [x] **MRI disk turbulence** (2026-10-03): measured sizes, lifetime, spirals and lognormal statistics; amplitude
  from the observed 2 % flicker (README "MRI disk turbulence"). Replaces "Evolving alpha-disk": the viscous
  evolution is static on screen (t_visc ≈ 3e6 M at 10 M).
- [x] **Light-travel delay** (2026-10-01) (the model has none: disk and jet are seen at one coordinate time).
- [x] **Jet knots from horizon-flux variability** (2026-10-03): the decorative knot noise and the static churn are
  replaced by the imprint of a MAD's horizon-flux history (eruptions every ~1500 M, 500 M drop e-folding, 20.9 %
  swing), launched at the base and carried by the plasma, plus filaments frozen into the moving, rotating plasma
  (README "Jet knots from horizon-flux variability"). Follow-ups: ~~spin-dependent flux statistics~~ done 2026-10-03 (absolute flux Φ, windowed
  modulation index — spin-independent per the source — eruptions × fast flicker; README "Correction: horizon-flux statistics"); ~~shell collisions (internal shocks proper)~~ SHELVED 2026-10-04: they form at hundreds of
  r_g, while the frame spans ±14 M of impact parameter (the 60 M jet already leaves it at z ≈ 15 M for 72°, ≈ 50 M
  for 17°), and the model has no launch-speed variability to make shells collide (the flux drives power and width,
  not speed; no MAD-jet measurement of it found) — would need a camera zoom-out and a sourced speed model; ~~coupling the eruptions
  to the disk~~ SHELVED 2026-10-04: physically anchored eruption flares (flux tubes, Sgr A* NIR-flare energy) are
  ~3e-4 of this app's thin-disk visible light for M87* and Sgr A* (invisible) and 100-200x too bright for X-ray
  binaries; the observed flares belong to hot flows, which the app does not model (spec and code on the unmerged
  branch feat/eruption-flares).

## Open questions from the presets work

- [x] **Jet brightness vs the disk** (2026-10-01; superseded 2026-10-02: jet brightness now follows from the
  energy budget, ε × the Blandford–Znajek power): presets calibrate the jet where an observation
  bounds it. M87\*: EHT horizon-scale jet base <= ~10 % of the ring -> jet strength 0.1 / 29.8
  (measured jet/disk light ratio at strength 1). Cyg X-1 / GRS 1915+105: unmeasured, model default,
  said in their captions. The default view keeps strength 1 (not a real object).

## Product / UX (later, by agreement)

- [x] **Screenshot export** (2026-10-03): **Save PNG** under Pause saves exactly the presented frame at the canvas's
  full internal resolution (README "Screenshot export"). A size picker (e.g. 4K offscreen) was considered and left out.
- [x] **Video / clip export** (2026-10-03): **Record** / **Stop** records the view as it plays (MP4 H.264, WebM
  fallback) at the canvas's full internal size (README "Clip export"). An offline fixed-step renderer for
  always-smooth clips was considered and left out.
- [x] **Shareable links** (2026-10-03): the address bar always holds the view (`#p=m87&x=0.5&play=0`) and **Copy link**
  copies it (README "Shareable links").
- [x] **Hot flow and the 1.3 mm view** (2026-10-04; README "Hot flow and the 1.3 mm view"): RIAF calibrated to the
  measured 230 GHz flux, Sgr A*'s ring 48 µas vs the EHT's 51.8 ± 2.3, no jet at 1.3 mm. Follow-ups: live mm frames
  are 2.2x slower than visible (hoist the ISCO orbit out of the per-sample velocity; skip the bloom pass in mm, whose
  output present ignores); a cooling-aware T_e for dense
  hot flows (λ ≳ 1e-4); MAD compression for M87*'s ring; eruption flares (feat/eruption-flares) can now be revisited
  against the hot flow.
- [ ] **Mobile layout**: control panel as a collapsible sheet and a phone-sized default render
  scale (camera drag already works on touch: it uses pointer events).

(Already shipped, not on this list: the panel names the WebGPU adapter and warns when it is an
integrated GPU, `src/render/gpuinfo.ts`.)

## Out of scope

Tier 3 (full GRMHD, multi-GPU/offline rendering).
