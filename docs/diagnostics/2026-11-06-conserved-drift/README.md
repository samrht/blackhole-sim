# Conserved-quantity drift: a = 0.998, i = 85 deg (2026-11-06)

Per ray, by the end of the ray: `|Q_end - Q_start| / |Q_start| + |H_end|`, on the renderer's Mino-time integrator (DP5(4), f32, tolerance 1e-5, 1280x720, RTX). Debug route `?drift`; re-run with the commands at the end. `report.json` in each folder has every number below and more.

## What the picture shows, and what it means

| Frame | File |
|---|---|
| The number as specified: the shadow is a black block | `disk/total.png` |
| Same rays, with f32 rounding removed from the measure | `disk/pTotal.png`, `nodisk/pTotal.png` |
| Q alone: arcs and chevrons | `mid/pRelQ.png`, against `mid/phase.png` |
| 280x onto the critical curve | `zoom-top/pRelQ.png` |

**1. The horizon cluster is real on the map, and it is not integrator drift.** As specified, captured rays read 0.5 to 14. Median |H_end| of captured rays, disk frame (no-disk frame in brackets), each tier removing one source:

| Trajectory | Measure | Median \|H_end\| | What it isolates |
|---|---|---|---|
| f32 (GPU) | f32 | 0.58 (0.94) | the number as specified |
| f64 | f32 | 0.66 (0.97) | evaluating H in f32 alone: it reproduces the row above |
| f32 (GPU) | f64 | 9.6e-4 (2.0e-3) | the drift the GPU's state really has |
| f64, rounded to f32 | f64 | 2.5e-4 (4.6e-4) | merely storing the state in f32 |
| f64 | f64 | 6.9e-5 (1.8e-4) | integrator truncation alone |

About 99.8 % of the signal is the measure: at r = 1.068, Delta = r^2 - 2r + a^2 cancels to ~7e-4 and carries ~2e-4 relative error into terms of order 1e4.

**2. What is left near the horizon is a coordinate factor.** H = [(r^4 / Delta) W + T] / (2 Sigma), with W = w'^2 - R~(w) the radial first integral. On captured rays H / [r^4 / (2 Sigma Delta) x W] has median 1.000, p99 1.053, and the factor is ~818 (p99 ~1100). W itself (no-disk frame) is 2.4e-6 at the horizon against 1.2e-6 in the sky (medians; the p99s differ more, 3.6e-5 against 2.8e-6). For |w'| > 1 W sits at 1.8e-7 to 2.7e-7 of w'^2, about 2 f32 eps of its own terms, which is why the retrograde side of the shadow (large |w'|) is darker.

**3. No clustering on the photon ring.** Down to d < 1e-4 M from the critical curve (280x zoom, 1.4e-4 M per pixel, up to 201 steps), escaping rays' |dQ| / |Q0| by distance from the curve, inward to outward: 2.3e-6, 3.1e-6, 1.7e-6, 2.1e-6, 1.8e-6, 9.7e-7, 2.1e-6 (zoom-top, medians). Flat and non-monotonic; the maximum is 7e-6, below the wide frame's. With the opaque disk most near-critical rays stop before winding at all (disk frame: at most 76 steps).

**4. The Q map's structure follows where the ray ends in its polar oscillation.** Arcs and chevrons in `mid/pRelQ.png` sit on the contours and cusps of `mid/phase.png`, and those contours nest toward the critical curve: that is the ring-like feature. Sky rays with |Q0| >= 1: median |dQ| / |Q0| ~3.8e-6 for |theta'| / sqrt|Q0| below 0.6, falling to 6.3e-7 at an equator crossing. **Mechanism open:** the first guess, dQ = 2 theta' d(theta'), predicts the opposite sign.

**5. Rounding, not truncation, sets the drift everywhere.** Medians, GPU state measured in f64 against f64 trajectory:

| Rays (disk frame) | \|dQ\| / \|Q0\| f32 | f64 | \|H_end\| f32 | f64 |
|---|---|---|---|---|
| disk | 7.0e-7 | 6.2e-8 | 4.1e-7 | 3.4e-8 |
| escaped | 9.3e-7 | 1.2e-7 | 6.9e-7 | 5.2e-8 |
| captured | 6.4e-7 | 3.1e-8 | 9.6e-4 | 6.9e-5 |

**6. Where a ray can actually be seen, it is clean.** With the measure's rounding removed, on the disk frame: disk rays median 1.1e-6, p99 8.4e-6, sky 1.6e-6 / 5.4e-6. Worst visible pixels: the disk's inner edge (r_end 1.2 to 1.5; ISCO 1.237), p99 1.5e-4, max 2.6e-4, the same 1/Delta factor. Captured pixels, the worst of all, are black: their colour does not depend on their end state.

## Traps in the number itself

- **|Q0| in the denominator.** The dark sliver on beta ~ 0 inside the shadow (`mid/pRelQ.png`) is where Q0 passes through 0: worst |dQ| / |Q0| 5.7e-4 at |Q0| < 1e-3 (58 pixels), where |dQ| is ~1e-8, the smallest in the frame. The bow-tie along beta ~ 0 is the same denominator at larger scale. Above |Q0| = 0.1 the median |dQ| / |Q0| falls with |Q0|: 8.5e-6 at 0.1 to 1, 4.7e-7 above 100 (no-disk frame).
- **H in Boyer-Lindquist form** grows as 1/Delta at the horizon for any fixed error: it is not comparable between r = 1000 and r = 1.07.
- **f32 evaluation.** Near the horizon the number as specified measures the f32 cancellation in Delta. Measure a stored f32 state in f64 (`pTotal`).

## Limits of this result

- **The step controller watches these same integrals** (`minoDriftNorm`, tolerance 1e-8 of their terms or the rounding floor), so low drift in f64 is partly by construction. In f32 the rounding floor mostly switches that check off (measured in `../2026-11-06-accuracy-limit/`, section 2), so the GPU numbers here are less affected. Either way, low drift does not show that a ray landed in the right place.
- One spin and inclination, one jitter set, one GPU. f64 tiers sample every 8th pixel (14,400 rays). End-of-ray values only: an excursion that returns is not seen.
- 1.4e-4 M per pixel does not separate the highest photon subrings; f32 eta rounding (~2e-6 at eta ~ 27) also limits how critical a GPU ray can be.
- No ray ran out of steps (budget 4800); step medians 25 to 78.

## Self-checks (all frames)

Ray fates against the renderer's `audit` entry: 0 mismatches of 921,600. f64 re-trace fates: 0 mismatches. WGSL measure against its CPU twin in f32: worst 0.16 of tolerance over 360 states. Start-of-ray |H|: at most 2.7e-7. `raytrace.wgsl`, `integrator-shared.wgsl`, `gpu.ts`, `uniforms.ts` and the other shared shaders are byte-identical to main; `?golden` PASS, 8 of 8 scenes bit-identical on the recording adapter (Intel; it reports SKIP on the RTX); the app checks on the normal page pass.

## Re-run

```
npx vite
node scripts/drift-render.mjs disk "a=0.998&i=85"
node scripts/drift-render.mjs nodisk "a=0.998&i=85&disk=0"
node scripts/drift-render.mjs mid "a=0.998&i=85&disk=0&fov=5&px=2.4"
node scripts/drift-render.mjs zoom-top "a=0.998&i=85&disk=0&fov=0.05&px=2.0986&py=5.1927"
node scripts/drift-render.mjs zoom-prograde "a=0.998&i=85&disk=0&fov=0.05&px=-1.9395&py=2.4032"
```

Or open `http://localhost:5173/?drift&a=0.998&i=85` for the view picker and a per-pixel readout on hover. Only the images listed above are committed; the script regenerates the rest (`relQ`, `absH`, `pH`, `pW`, `absdQ`, `wRes`, `hUlps`, `qUlps`, `fate`).
