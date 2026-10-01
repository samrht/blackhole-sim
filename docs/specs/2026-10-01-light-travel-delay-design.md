# Light-travel delay — design

**Date:** 2026-10-01
**Status:** implemented on `feat/light-travel-delay` (plan `docs/plans/2026-10-01-light-travel-delay.md`)
**Roadmap item:** `docs/ROADMAP.md`, "Physics features: Light-travel delay"

## 1. Intent (agreed)

- Every pixel shows its emitter **as it was when the light left it**, not one global instant: the
  far side of the disk, and the higher-order images that loop around the hole (the photon-ring
  subimages), are seen tens of M earlier than the near side. Visible payoff: a hot spot appears in
  the primary image and then **echoes** in the thin rings near the shadow.
- A **"Light delay" checkbox, on by default** (the accurate setting); off reproduces today's
  same-instant view exactly, for comparison.
- The **Motion slider becomes pure playback speed** (how fast scene time runs), so delays, which are
  physical coordinate time, stay correct at any Motion setting.
- Not in scope: time-dependent sky (the panorama is static), light-travel effects on the camera's
  own motion (it is static), relativistic aberration changes (already in the geodesics).

## 2. Physics

The integrator already carries Boyer–Lindquist coordinate time t in the state (`s.x.x`; CPU twin
`s[0]`). The camera starts at t = 0 and the backward-traced ray's t decreases (past-directed
momentum), so at an emission point t_e < 0 is minus the light-travel time.

- **Relative delay:** `delay = −t_e − rObs` (M). The constant `rObs` removes the common ~1000 M
  travel time, so values are tens of M and f32-friendly; a constant offset only shifts the whole
  animation in time.
- **Emission time:** `T_emit = U.time − U.lightDelay · delay`, with `U.time` the scene's coordinate
  time and `U.lightDelay` ∈ {0, 1} from the checkbox.
- **Uses:** the disk's co-rotating pattern phase `ψ = φ − Ω(r) · T_emit` (turbulence, hot spots),
  the breathing term `sin(2π T_emit / T_BREATHE)`, and the jet's knot phase. For the jet, each
  quadrature sample takes its own t by linear interpolation between the step's end states (the
  same chord interpolation as its position).
- **Motion as playback speed:** `main.ts` advances `simTime += dt · SPEED · timeScale`; the shader no
  longer multiplies time by `U.timeScale` (the field stays in the uniform layout, unused). At
  timeScale = 1 this is arithmetically identical to today.
- **Checks on the physics:** (1) for a = 0 a radial ray (α = β = 0) from r = 1000 to r = 10 takes
  exactly t = r\*(1000) − r\*(10), r\* = r + 2M ln(r/2M − 1) (= 999.65 M); (2) a near-critical ray
  at a = 0 winding near r = 3M crosses the equatorial plane every half orbit, Δt → π√27 M ≈ 16.3 M
  (the delay between successive photon-ring subimages, Gralla & Lupsasca 2020).

## 3. Geodesic cache

Entries stay 16 bytes. A DISK entry stores `(word, rHit, φHit, delay)` instead of
`(word, rHit, φHit, g)`: the shade pass **recomputes g** with `diskG(rHit, ξ, a)`, where ξ comes from
the pixel and its fixed jitter offset through the same shared code as the live trace (a new helper
returning the pixel's impact parameters, used by both `traceRay` and `shade`), so cached and live
shading stay equal. SKY entries are unchanged. Jet bookmarks already store the full state including
t, so `replayJet` reproduces per-sample emission times automatically. The checkbox is shading-only
(delays are always stored): flipping it never rebuilds the cache.

## 4. Architecture and data flow

- `src/render/raytrace.wgsl` — `pixelImpact(pix, jit) -> vec2` (α, β) shared by `traceRay` and
  `shade`; `shadeDisk(rHit, phiHit, g, a, tEmit)`; `emissionFieldE(rHit, psi, tEmit)` (breathing);
  `jetEmissionJ(r, th, tEmit)` fed per sample; DISK entry payload `(rHit, phiHit, delay)`;
  `TraceOut` carries the hit delay.
- `src/render/uniforms.ts` — new `lightDelay` field (f32, index 31; the 128-byte buffer has room).
- `src/main.ts` + `index.html` — "Light delay" checkbox (on), `simTime` advanced by Motion.
- `src/test/scenes.ts` — golden scenes use `lightDelay: 0` (toggle-off path must equal today);
  `?cachecheck` gains a delay-on scene with hot spots and turbulence.
- `src/physics/trace.ts` — the CPU twin already returns t; tests read it.

## 5. Error handling

Delay is finite wherever a hit is recorded (t is finite on accepted states; the `accum` NaN choke
point already guards the output). Budget-exhausted rays classified by (ξ, η) carry no emission time.

## 6. Testing

- **Unit (vitest, CPU twin):** the radial Schwarzschild light time within 1e-3 M; successive
  equatorial crossings of a near-critical a = 0 ray separated by π√27 M ± 3 % near r = 3M; delay of a
  far-side disk point larger in its secondary image than in its primary; uniforms pack
  `lightDelay` at byte 124.
- **GPU:** `?golden` must stay **bit-identical** to the current hashes (scenes run with delay off and
  timeScale 1); `?cachecheck` PASS including the new delay-on scene (cached = live ≤ ~4e-7);
  `?parity`, `?shadow`, app checks, probes PASS.
- **Visual:** an echo screenshot pair (one hot spot, delay on vs off) saved for review.
