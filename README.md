# blackhole-sim

A physically accurate, real-time renderer of a Kerr (spinning) black hole's accretion disk, running entirely in the browser on WebGPU.

This isn't a stylized visualization — it's a real general-relativistic ray tracer. Every pixel backward-integrates a null geodesic through curved Kerr spacetime, terminating on the event horizon (shadow), the accretion disk (emission), or escaping to the background. Disk brightness and color come from real physics: Novikov–Thorne relativistic disk flux, combined gravitational + Doppler redshift, and blackbody emission mapped through actual CIE color-matching functions into sRGB — not a hand-tuned color ramp.

## Features

- Exact Kerr metric (Boyer–Lindquist), horizons, ergosphere, frame dragging
- Hamiltonian geodesic integration (adaptive RK4) with conserved quantities (E, L_z, Carter constant)
- ISCO, photon orbit, and marginally bound radii for arbitrary spin
- Novikov–Thorne / Page–Thorne relativistic disk flux (correct zero-torque inner boundary — flux peaks just outside the ISCO, not at it)
- Combined gravitational + Doppler redshift (g-factor), producing physically correct approaching/receding disk asymmetry
- Blackbody emission → CIE XYZ → linear sRGB (Wyman et al. color-matching fit) — color comes from real temperature, not an artist's gradient
- HDR accumulation with ACES tonemapping and progressive anti-aliasing
- Interactive controls: spin (a*), inclination, exposure — drag the canvas to tilt the camera live

## Requirements

- A WebGPU-capable browser: **Chrome/Edge 113+** or **Safari 18+**
- A dedicated GPU is strongly recommended. This was scoped to Tier 1 of a 3-tier physics spec specifically to run on modest hardware (target: NVIDIA RTX 3050 Laptop, 4GB), with progressive frame accumulation capped around 15fps to stay responsive — full GR ray tracing is inherently expensive per pixel, so performance on integrated graphics may still be limited.

## Quickstart

```bash
npm install
npm run dev       # http://localhost:5173
npm test          # run the physics unit test suite (Vitest)
npm run build     # production build
```

Validation routes (append to the dev URL):
- `?parity` — CPU↔GPU parity check for the core physics math
- `?shadow` — structural check that the Schwarzschild shadow radius matches the analytic value (√27 M)

## Architecture

The physics core (`src/physics/`) is pure TypeScript with zero DOM/GPU dependencies, unit-tested in isolation against known analytic results (Schwarzschild horizon = 2M, ISCO = 6M/M/9M for a=0/extremal-prograde/extremal-retrograde, etc.). Its outputs are baked into 1D lookup tables (temperature and color) uploaded as GPU textures. A WebGPU compute shader (`src/render/raytrace.wgsl`) mirrors the core's math, tracing one geodesic per pixel and accumulating HDR radiance; a present pass resolves the accumulation buffer with exposure, ACES tonemapping, and gamma correction.

Full physics specification and derivations: `docs/specs/2026-05-30-relativistic-blackhole-accretion-design.md`

## Key references

Kerr (1963); Carter (1968); Bardeen, Press & Teukolsky (1972); Bardeen (1973); Novikov & Thorne (1973); Page & Thorne (1974); James, von Tunzelmann, Franklin & Thorne (2015, the Interstellar/Gargantua paper).

## Status

Tier 1 (single-GPU, real-time image) complete and verified.

**Tier 2A — Living Disk (shipped):** the accretion disk now evolves in real time. Differential rotation carries a co-rotating turbulence pattern (procedural value-noise over log-radius and pattern phase), orbiting Gaussian hot-spots are Doppler-beamed by the existing g-factor (brightening on the approaching side, dimming on the receding side), and a temporal EMA replaces the static progressive average so motion is smooth while a paused scene still re-converges to a clean still. Motion / Turbulence / Flares sliders and a Play/Pause control drive it live. With all features off the render is bit-for-bit identical to Tier 1 (`?parity` and `?shadow` unchanged).

Next: **Tier 2B — relativistic jet** (Blandford–Znajek funnel / synchrotron). Out of scope: lensed starfield background, Tier 3 (full GRMHD, multi-GPU/offline).
