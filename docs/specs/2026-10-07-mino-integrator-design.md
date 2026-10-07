# Faster geodesic integrator (Carter equations in Mino time) — design

**Date:** 2026-10-07
**Status:** design approved in conversation (sections 1–4); this document awaits review
**Goal (agreed):** live frames as fast as possible in both views, with **nothing scientific changed**. The metric, the
geodesic equations, every emission model, constant and calibration stay as they are. Only the numerical method that solves
the same Kerr null geodesics changes, and no ray may become less accurate.

## 1. The rule

- **Science unchanged.** Disk, jet, hot flow, hotspot and sky models are byte-identical. So are their constants, and the
  calibrations (n₀, A₀, η, flux statistics). The CPU references that produced those calibrations (`trace.ts`'s Hamiltonian
  integrator, `hot-flow-image.ts`, `scripts/calibrate-*.ts`) stay as they are. They are reference data.
- **Numerics no worse.** Compared with a fully converged reference:
  - no ray changes fate (shadow / disk / sky);
  - every ray's error is ≤ today's error on every measured quantity (§5), up to a stated single-precision floor.
  
  Pixels then move at the numerical-error level, and the golden images are re-recorded only after this is shown.
- **Every existing gate keeps passing:** `?parity`, `?shadow` (critical curve, 0 mismatches), `?hotflow` (flux, ring,
  hotspot twins), `?cachecheck` (exactly 0), and the app checks.

## 2. Measure first (go / no-go before any renderer change)

**(a) GPU cost split.** A scratch variant of the renderer (not shipped) times live frames at 720p on the RTX 3050 Laptop
for two scenes: the default visible view (a 0.9, 72°, jet on) and Sgr A* at 1.3 mm. Three runs per scene:
- full;
- emission stubbed: disk, jet, flow and hotspot coefficients replaced by constants, so the geometry is unchanged;
- geometry stubbed: rays stop after a fixed short path, with the emission code intact.

The result is each part's share of the frame, and so the ceiling an integrator change can reach. It is reported before
step (b).

**(b) CPU prototype.** `src/physics/trace-mino.ts` (§3, §4) runs on the existing ray sets (§5) against the converged
reference. It reports:
- fate;
- every §5 error, next to today's integrator (`trace.ts` `stepGeodesic` + `stepSize`) on the same rays;
- steps per ray.

**(c) Per-step cost.** A WGSL microbenchmark times N steps of today's `stepGeodesic` (RK4 + monitor) against N steps of
the new step, on the same GPU.

**Go** if: no fate flip; no ray worse (§5); and (steps per ray × cost per step) ≤ ½ of today's, measured over the 1.3 mm
flow rays and the visible rays separately. **No-go:** report the numbers and fall back to error-controlled stepping of
today's equations (the user's route 3), which needs its own design.

## 3. The integrator

The same null geodesics in Carter's separated form (Carter 1968), in Mino time λ with dλ = dl / Σ (l affine,
Σ = r² + a²u², Δ = r² − 2r + a², M = 1).

- **Constants per ray** (exact, from the camera as now):
  - p_t = 1, the past-directed, camera-normalised convention of `screenToState` / `cameraMomenta`;
  - ξ = −p_φ;
  - η = Carter's constant (`screenToXiEta`).
  
  P(r) = r² + a² − aξ, R(r) = P² − Δ (η + (ξ − a)²), U(u) = η(1 − u²) + a²u²(1 − u²) − ξ²u², u = cos θ.
- **State:** (t, r, u, φ, l, r′, u′), with ′ = d/dλ. Evolution:
  - r″ = R′(r)/2 = 2r P(r) − (r − 1)(η + (ξ − a)²)
  - u″ = U′(u)/2 = u (a² − η − ξ²) − 2a² u³
  - t′ = −[ (r² + a²) P / Δ + a (ξ − a (1 − u²)) ]  (Σ × g^tμ p_μ for this p; t decreases along the backward ray)
  - φ′ = −[ a P / Δ − a + ξ / (1 − u²) ]
  - l′ = Σ = r² + a²u²
  
  The r and u equations are polynomials with no singularity at the horizon or the poles. t, φ and l are quadratures
  that never feed back into r or u. The first integrals r′² = R and u′² = U hold at the initial point and are not
  enforced afterwards; the sweep measures their drift.
- **Initial state from today's camera State** (x = (0, r_obs, θ₀, 0), p): r′ = Δ p_r, u = cos θ₀,
  u′ = −sin θ₀ p_θ, l = 0.
- **Conversion back** (what every consumer reads; `State` stays the interface):
  - x = (t, r, acos u, φ);
  - p = (1, r′/Δ, p_θ, −ξ), with p_θ = −u′ / sin θ.
  - Where sin θ < 1e-4, p_θ = sign(−u′) √max(Θ, 0) with Θ = U / (1 − u²) taken from its polynomial form. Never 0/0.
  - dl for a step is Δl.
- **Dense output:** each step keeps both endpoint states and their derivatives. A cubic Hermite interpolant in λ gives
  (t, r, u, φ, l) anywhere inside the step. Jet, flow and hotspot samples take their positions from it, at the same
  affine spacing as today (`JET_DL` = 0.25 M, the flow's r/8 rule beyond 8 M), instead of today's straight Cartesian
  chord. The equatorial crossing (disk hit) is the root of the u cubic, not a linear interpolation.

## 4. Step control

- **Dormand–Prince 5(4)** (embedded, FSAL) on (r, r′, u, u′, t, φ, l). Seven cheap stages per step against RK4's four
  metric-plus-gradient stages and two constraint checks.
- **Error norm:** max over components of |err_i| / (atol_i + rtol |y_i|).
  - r and r′ are measured relative;
  - u and u′ absolute;
  - t, φ and l relative to their per-step change.
  
  The tolerance (one scalar `MINO_TOL` and fixed component weights) is the loosest value at which the §5 sweep passes
  with a 2× margin. It is never below the single-precision floor of 1e-6. Standard step-size update (safety 0.9, growth
  limits 0.2–5). A rejected step is redone; the attempts are bounded.
- **Guards:**
  - a step may not change u by more than U_MAX_DU, a fraction of the u-oscillation's amplitude that the sweep sets, so a
    step cannot hide two plane crossings;
  - the first step size comes from today's `stepSize` converted to λ (dλ = dl / Σ).
- **Termination unchanged:** r ≤ 1.005 r₊ (captured), r > 1.2 r_obs (escaped; the sky direction from the converted
  State as now), the 1.3 mm early exit at r > 50 M outgoing, the `maxSteps` budget (4800), and the analytic classifier
  for rays that exhaust it.
- **Determinism:** the steps depend only on the ray's inputs, so the cache's bookmark replay takes exactly the live
  trace's steps.
- **Removed once unused:** the Hamiltonian constraint monitor (`hquadScaled`, retries), `angularCap` (F_AXIS, F_PHI),
  `reflectAxis`, and the far/near `stepSize` branches. `gUp`, `gLow` and `gUpGrad` stay wherever emission code and
  `?parity` use them.

## 5. Accuracy sweep (the gate that sets the tolerance)

`tests/sweep-mino.test.ts` (`SWEEP=1`; the `HOLDOUT=1` split as in the far-monitor sweep).

- **Ray sets:**
  - default 72° and a = 0.9 (disk and jet);
  - face-on 8° and edge-on 85° (a = 0.99);
  - near-axis rays (the polar-axis set);
  - a band of near-critical rays within 0.02 M of the critical curve at a = 0, 0.9 and 0.998;
  - the 1.3 mm flow rays for Sgr A* (0.94, 30°) and Gargantua (0.6, 85°).
- **Reference:** today's Hamiltonian equations (`trace.ts`) at a step 100× finer, accepted where it agrees with a 200×
  run. This is the far-monitor sweep's converged-reference method, and an independent formulation that cross-checks the
  new one. Rays the reference cannot converge are listed and excluded, as before.
- **Measured per ray** (new and today's, both against the reference):
  - fate;
  - disk-hit r and φ;
  - arrival time (light delay);
  - escape direction, in on-screen pixels through the lensing Jacobian (as before);
  - for the flow rays, the integrated 230 GHz intensity with the hot-flow coefficients.
- **Pass:**
  - zero fate flips;
  - for every ray and quantity, error_new ≤ max(error_today, floor), with floors 1e-5 relative (r, t, intensity),
    1e-5 rad (φ) and 0.01 px (sky);
  - the first-integral drift |r′² − R| / P² and |u′² − U| stays below 1e-4.

## 6. Code layout

- `src/physics/trace-mino.ts` (CPU twin): `minoInit(s, a, xi, eta)`, `minoRhs`, `minoStep` (DP5(4) with the error norm
  and guards), `minoDense(s0, s1, theta)` (Hermite), `minoToState`, `minoTrace` (the ray loop used by the sweep).
- `src/render/integrator-shared.wgsl` (sole WGSL copy): the same functions, with `MinoState` (two vec4 plus the per-ray
  ξ, η).
- `src/render/raytrace.wgsl`:
  - `traceRay` steps in Mino time;
  - `jetStep`, `flowStep` and `jetTouches` take their sample points from `minoDense` instead of the chord;
  - the disk crossing uses the u-cubic root;
  - a `Bookmark` stores the Mino state in the same 48 bytes (ξ and η are recomputed from the pixel), and `replayJet`
    steps from it.
- `src/test/parity.browser.ts` + `integrator-parity.wgsl`: Mino cases (one step and dense output against the CPU twin at
  spins 0 / 0.9 / 0.998, near the horizon, near the axis, near turning points) replace the old integrator cases. A
  mutated constant must fail.
- `tests/shader-twins.test.ts` keeps guarding single copies.

## 7. Gates and measurements

- `npm test`, `npm run build`.
- `?parity` PASS (new cases, mutation check).
- `?shadow` 0 mismatches at a = 0, 0.9, 0.998.
- `?hotflow`: Sgr A* and M87* flux ±5 % and ring ±0.5 µas of the CPU twin; Sgr A* within the EHT 2σ; hotspot twins
  within 5 % and 0.5 M.
- `?cachecheck` exactly 0, every scene.
- `?golden` re-recorded only after §5 passes (every hash is expected to move).
- Every app check.
- Bench (interleaved with `main`, minima): visible and 1.3 mm, live and cached, 720p and 1080p, plus the mm hotspot row.

## 8. Order of work

1. §2 measurements (a, b, c), then the go/no-go report.
2. CPU integrator + sweep, which set `MINO_TOL` and `U_MAX_DU`.
3. WGSL twin + parity.
4. Renderer swap (ray loop, dense-output sampling, disk crossing, bookmarks).
5. Gates, bench, README, final review, merge and push. Deploy on request.

## 9. Limitations and risks

- Single precision: the Mino-time polynomials have large terms far from the hole (R ~ r⁴ at r_obs = 1000). The error
  norm and the f32 floor must be checked there (parity cases at r = 1000 and 60).
- φ and t grow through 1/Δ near the horizon. That is a coordinate effect (as today); the 1.005 r₊ stop keeps them finite.
- If the cost split shows emission dominating the 1.3 mm frame, the integrator alone cannot make that view much faster.
  The report says so, and any emission-side work would be a separate design under the same science rule.
