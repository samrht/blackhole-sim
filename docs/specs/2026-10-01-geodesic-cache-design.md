# Geodesic cache — design

**Date:** 2026-10-01
**Status:** approved design, not yet planned
**Branch:** `perf/geodesic-cache` (created from `main` at `d95e4a1`)

## 1. Problem, as measured

After the smooth-first round (`docs/specs/2026-09-23-performance-design.md`), a full trace costs
≈ 72 ms at 1280×720 and ≈ 165–186 ms at 1920×1080 on the RTX 3050 Laptop. While animating, the scale
controller drops to half resolution: ≈ 55 fps windowed, ≈ 22–24 fps fullscreen.

Every animated frame re-integrates every geodesic, but while playing with a still camera the geodesics do
not change. In `raytrace.wgsl` the time `U.time` enters only *after* the path is known:

- **disk:** `psi = phiHit - Om * time * timeScale` and `emissionFieldE(rHit, psi)` (turbulence, breathing,
  hot spots) — evaluated at the hit point;
- **jet:** `knotsJ(z, t)` inside `jetEmissionJ`, evaluated at every step inside the jet;
- **sky / shadow:** time-independent.

The path itself depends only on the **geometry inputs**: `a`, `incl`, `fovScale`, `rObs`, `rIn`, `rOut`,
`maxSteps`, `jetLength` (it bounds the jet envelope, §3.3), and the internal resolution. There is no
light-travel delay in the model (emission uses observer time), so caching the path is exact.

## 2. Goal and success criteria

Smooth playback at full resolution, fullscreen 1080p included, with **no change to the physics or its
accuracy**.

- **Target:** 60 fps at 1920×1080, scale 1.0, default scene (jet on) on the RTX 3050 Laptop while playing
  with a still camera. A target, not a claim: the benchmark reports what is reached, and README Status
  records it.
- **Exactness:** a cached frame equals a live-traced frame with the same jitter offset and time
  (`?cachecheck`, §3.6). Non-jet pixels: max relative difference ≤ 1e-5. Jet-replay pixels: ≤ 1e-3 for
  ≥ 99.99 % of them, with the max reported. A failure is investigated, never absorbed by loosening.
- **Unchanged:** the live path (dragging, geometry sliders, paused progressive still) behaves and renders
  as today; the refactored `main` entry point is bit-identical to the pre-refactor one (golden readback,
  §3.6).
- Every task keeps green: `npm test`, `npm run build`, `npm run verify:gpu` (`?parity`, `?shadow`, sky,
  plus the new `?cachecheck`), `node scripts/probe-axis.mjs`.

## 3. Design

### 3.1 Modes

`Renderer` runs in one of two modes per frame:

- **Live** — today's `main` trace pass with hash jitter, EMA / progressive mean, adaptive scale. Used while
  paused, while the cache is invalid or still building its first set, and whenever the geometry key
  changes (dragging, spin/inclination/step-budget/jet-length sliders, resize).
- **Cached** — used while playing once ≥ 1 jitter set is complete. Internal scale is pinned to 1.0 and the
  scale controller is idle. Each frame runs the `shade` pass (§3.4) instead of `main`, then the unchanged
  bloom + present.

The **geometry key** is the tuple in §1 plus `displayW × displayH`. `main.ts` compares it every frame; any
change invalidates the cache (back to Live, rebuild starts). Shading-only inputs — `time`, `exposure`,
`Tpeak`, `timeScale`, `turbAmp`, `breatheAmp`, hot spots / flare scale, `jetStrength`, `jetGamma`,
`jetKnots`, `skyStrength`, sky upload — never invalidate it. A panel readout shows `live` or
`cached n/4` next to Render scale.

### 3.2 Fixed jitter sets

The cache holds `NSETS = 4` jitter sets per pixel at fixed sub-pixel offsets (rotated grid):
`(-0.125,-0.375) (0.375,-0.125) (0.125,0.375) (-0.375,0.125)`. Cached frame `n` shades set
`n mod completedSets` and blends with the existing playing EMA (`EMA_BLEND = 0.15`), giving 4-sample
anti-aliasing while playing. The live path keeps `hash2` jitter, so the paused still keeps unlimited
progressive AA.

### 3.3 Build pass (`build` entry point)

Traces full-resolution rays for one jitter set over a slice of rows (`rowStart..rowEnd` uniforms),
`BUILD_SLICES = 16` slices per set, one slice per frame while playing, alongside the Live frame. When set 0
completes the display switches to Cached; sets 1–3 keep building one slice per frame (Cached frames are
cheap). Build is paused while paused. Expected: ≈ 12 ms extra per frame at 1080p for ≈ 16 frames, then
Cached. `BUILD_SLICES` is re-measured by the benchmark.

Per pixel per set it writes one **entry** (16 bytes: `u32` kind | index, 3 × `f32`):

| kind | payload |
|---|---|
| `DISK` | `rHit`, `phiHit`, `g` (Doppler + gravitational factor, already includes ξ) |
| `SKY` | asymptotic `skyDir` (octahedral-encoded, 2 × f32) — from the in-loop escape **or** the budget-exhausted classifier |
| `SHADOW` | — (captured, horizon, or non-finite state) |
| `LIVE` | — (bookmark buffer full: this pixel is fully traced in `shade`) |

plus a 30-bit **bookmark index** (or none) for rays that touch the jet envelope.

**Jet envelope.** `inJetEnvelope(r, θ) := JET_ZBASE ≤ |z| ≤ jetLength && ρ / funnelEdgeJ(z) ≤ 1.2` —
exactly the region where `jetEmissionJ` can be non-zero. It is evaluated on the step's start state, as
the jet accumulation already is, and does **not** depend on `jetStrength`, so turning the jet on or off
never rebuilds.

**Bookmark.** For a ray whose start state is inside the envelope on some step, record the state `s` at
the first such step and `nJet` = number of steps from it through the last such step (contiguous, so the
path between jet visits is re-integrated too). Bookmarks (`x: vec4, p: vec4, nJet: u32`, 48 bytes
padded) are allocated with `atomicAdd` from a sparse buffer of capacity `BOOKMARK_FRAC × entries`
(initial 0.25, set from the measurement in the plan's first task). On overflow the entry kind becomes
`LIVE`; correctness is kept, only speed is lost.

### 3.4 Shade pass (`shade` entry point)

Per pixel, read the entry of the current set:

- `DISK` → `shadeDisk(rHit, phiHit, g)`: `Tn = sampleTemp(rHit)`, `Om = omegaKep(rHit, a)`,
  `Tobs = Tpeak·g·Tn`, `psi = phiHit − Om·time·timeScale`, `E = emissionFieldE(rHit, psi)`,
  `color = sampleColor(Tobs)·(g·Tn)^4·E`.
- `SKY` → `skyColor(dir)` (starfield / panorama crossfade by `skyStrength`).
- `SHADOW` → 0.
- `LIVE` → the full `traceRay` (identical to the live pass, fixed jitter of this set).
- If a bookmark exists and `jetStrength > 0`: replay `nJet` steps from it with `stepGeodesic` +
  `jetStep`, accumulating `jetAccum` exactly as the trace does.

Then the existing composite (`color + jetStrength·min(jetAccum, JET_CEIL)`), NaN choke point and EMA write.

**One copy of the maths.** `shadeDisk`, `skyColor` and `jetStep` are extracted from `main`'s loop into
shared functions called by `main`, `build` and `shade`, the same way the integrator, camera and shadow
code are already shared. `main` keeps its exact operation order (golden check, §3.6).

### 3.5 Buffers and limits

- Entries: one buffer per set (`W·H·16` bytes; 33 MB at 1920×1080), each under the default 128 MB binding
  limit; `shade` binds the active set's buffer (4 bind groups).
- Bookmarks: one buffer + an atomic counter, reset when a rebuild starts.
- Live and Cached use separate accum buffers (Live at the scaled size, Cached at full size), so switching
  mode never shows a cleared buffer; the first Cached frame uses `blend = 1`.
- Larger displays: if a set's entries exceed the adapter's `maxStorageBufferBindingSize` (requested at the
  adapter maximum in `requestDevice`), or the cache would exceed 512 MB total, use `NSETS = 2`; if one set
  still does not fit, stay Live. Logged, never fatal.
- New uniforms: `jitterMode` (0 hash, 1 fixed set), `setIndex`, `rowStart`, `rowEnd`. `uniforms.ts` and its
  test are updated.

### 3.6 Validation

- **Golden live check** (first task, before any refactor): read back `main`'s raw accum floats for fixed
  scenes (default, jet off, high spin edge-on) at fixed `frame`/`time`; store hashes. After the refactor
  the live pass must reproduce them bit-for-bit.
- **`?cachecheck` route** (added to `verify:gpu`): build all sets; for each set `k` and
  `time ∈ {0, 137.5}`, render Live with `jitterMode = 1, setIndex = k, blend = 1` and Cached with the same
  inputs; compare raw accum floats (new `Renderer.readbackAccum()`) against §2's thresholds; report max
  and 99.99th-percentile differences split into non-jet / jet-replay / `LIVE`-fallback pixels.
- **Unit tests:** geometry-key invalidation (which inputs do and do not invalidate), jitter offsets, build
  slicing / scheduler, memory sizing and the `NSETS` fallback, octahedral encode/decode round-trip.
- **Existing gates** unchanged and green.

### 3.7 Benchmark

`npm run bench` adds: Cached ms/frame at 1280×720 and 1920×1080 with jet on and off, total build time
per set, measured bookmark fraction and mean `nJet`. Reported, not gated.

## 4. Risks

- **Jet replay cost.** Replay re-integrates the near-field stretch; if jet pixels are a large share of the
  screen, Cached gains shrink with the jet on. The plan's first task measures the jet pixel fraction and
  mean `nJet` before building anything; a poor result is reported to the user before continuing.
- **Replay rounding.** `shade` and `build` compile separately, so replayed states may differ in the last
  bits; near the critical curve that can grow. `?cachecheck` quantifies it; the jet contribution of
  near-critical rays is small.
- **Memory on large displays.** Bounded by §3.5's fallbacks.
- **Build hitch.** Bounded by slicing; re-tuned from the benchmark if visible.

## 5. Out of scope

Light-travel (emission-time) delay — the cache makes it cheap to add later; the flat-space skip (option
#3); speeding up the live path while dragging (temporal reprojection); changing `rObs`, the integrator,
step control, or any physics constant.
