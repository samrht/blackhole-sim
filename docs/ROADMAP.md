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

- [ ] **A true critical-curve gate for `?shadow`**: the route measures the emitter's lensed inner
  edge, not the critical curve (an emitter that stops outside the capture region, or a (xi, eta)
  classification image).
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
- [ ] **Evolving alpha-disk** (surface-density diffusion) in place of static Novikov-Thorne plus
  decorative turbulence.
- [x] **Light-travel delay** (2026-10-01) (the model has none: disk and jet are seen at one coordinate time).

## Open questions from the presets work

- [x] **Jet brightness vs the disk** (2026-10-01; superseded 2026-10-02: jet brightness now follows from the
  energy budget, ε × the Blandford–Znajek power): presets calibrate the jet where an observation
  bounds it. M87\*: EHT horizon-scale jet base <= ~10 % of the ring -> jet strength 0.1 / 29.8
  (measured jet/disk light ratio at strength 1). Cyg X-1 / GRS 1915+105: unmeasured, model default,
  said in their captions. The default view keeps strength 1 (not a real object).

## Product / UX (later, by agreement)

- [ ] **Screenshot export** at full internal resolution (a still from the converged accumulation).
- [ ] **Video / clip export** of the animated view (MediaRecorder on the canvas stream).
- [ ] **Shareable camera links**: spin, inclination, sliders and preset encoded in the URL.
- [ ] **Mobile layout**: control panel as a collapsible sheet and a phone-sized default render
  scale (camera drag already works on touch: it uses pointer events).

(Already shipped, not on this list: the panel names the WebGPU adapter and warns when it is an
integrated GPU, `src/render/gpuinfo.ts`.)

## Out of scope

Tier 3 (full GRMHD, multi-GPU/offline rendering).
