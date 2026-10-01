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
- [ ] **Shared-fragment parity for the jet and turbulence cases**: `?parity` still tests copies of
  that math (`jet-parity.wgsl`, `turb-parity.wgsl`); camera, shadow and integrator already verify
  the shipped bytes.
- [ ] **Jet core line**: with the jet on, the alpha = 0 column shows a thin bright line because the
  phenomenological wall profile has a small non-zero core emissivity (jet model, not integrator).

## Physics features (original Tier 2, not started)

- [ ] **Synchrotron jet emission** from a field and electron distribution (the jet is
  phenomenological today; see the README's Tier 2B caveats).
- [ ] **Evolving alpha-disk** (surface-density diffusion) in place of static Novikov-Thorne plus
  decorative turbulence.
- [ ] **Light-travel delay** (the model has none: disk and jet are seen at one coordinate time).

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
