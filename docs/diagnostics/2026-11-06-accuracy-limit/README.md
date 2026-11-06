# What limits accuracy on a GPU ray tracer: f32 rounding or integrator order? (2026-11-06)

Follow-up to `../2026-11-06-conserved-drift/`, which found that conserved-quantity drift is mostly f32 evaluation error and a coordinate factor. This experiment scores what a ray actually delivers instead: where it lands on screen.

![Median on-screen error against force evaluations](limit-cost.svg)

## Answer

**For ordinary rays, f32 rounding sets the limit, and it builds up during integration.** Integrator order only sets where that wall sits, by about 2.5x between 3rd and 5th order. At the shipped tolerance f32 rays are 14x to 21x less accurate than the same integrator in f64.

**The exception is rays passing almost over the pole.** There the shipped tolerance itself is too loose (truncation, plus f32 making it worse), and errors reach whole pixels.

**In absolute terms the ordinary error is tiny:** at the shipped tolerance, the 99th percentile is 5e-4 to 9e-3 of a 720p pixel, depending on the view and ray type.

## Method

The same 4,304 rays per view go through every arm: a 64 x 36 grid, 1,700 rays log-spaced from 1e-4 to 1 M off the critical curve, and 300 rays with |alpha| from 1e-5 to 0.03 (near the pole). All alpha and beta values are exactly representable in f32, so an arm's error is its own and not the rounding of the pixel coordinate.

Score: on-screen displacement in pixels of the app's 720p view, through the reference's own Jacobian at each ray, plus fate flips. Disk rays report the hit point, sky rays their direction. Medians are over the ray type shown.

| Arm | Runs on | Purpose |
|---|---|---|
| DP5(4), Mino equations, tol 1e-3 to 1e-7 (f64 to 1e-9) | GPU f32, CPU f64 | the shipped integrator |
| BS3(2), same equations, error norm and landing, tol 1e-3 to 1e-7 | GPU f32, CPU f64 | the clean order test |
| RK4, affine Hamiltonian, linear disk landing, step scale 1 to 1/8 | GPU f32, CPU f64 | the pre-Mino renderer |
| f64 at tol 1e-9 with f32 ray constants | CPU | rounding entering with xi, eta, a, i |
| f64 at tol 1e-9, result rounded to f32 | CPU | what the output format alone costs |
| f32 end state, sky direction computed in f64 | GPU + CPU | rounding in the direction calculation |

## Results

### 1. f32 hits a wall; f64 does not

Median px, disk rays at a = 0.998, i = 85 (attempts per ray in brackets):

| Tolerance | 1e-3 | 1e-4 | **1e-5 (shipped)** | 1e-6 | 1e-7 |
|---|---|---|---|---|---|
| DP5 f32 | 1.3e-3 (30) | 9.4e-4 (34) | **3.7e-4 (36)** | 2.0e-4 (45) | 2.1e-4 (67) |
| DP5 f64 | 1.4e-3 (27) | 1.6e-4 (33) | **2.1e-5 (43)** | 2.9e-6 (57) | 5.1e-7 (74) |
| BS3 f32 | 5.8e-3 (72) | 3.6e-3 (84) | 2.7e-3 (104) | 5.7e-4 (207) | 5.6e-4 (455) |
| BS3 f64 | 5.6e-3 (72) | 9.8e-4 (117) | 1.7e-4 (197) | 2.9e-5 (343) | 4.9e-6 (613) |

f32 and f64 agree at 1e-3, where truncation dominates. DP5 f64 keeps improving to 1e-8 px at tol 1e-9 (whole frame); f32 stops at 1e-6.

### 2. Across all four views

| View (rays) | DP5 f32 shipped | DP5 f64 shipped | f32 / f64 | DP5 f32 floor | BS3 f32 floor | Floor ratio |
|---|---|---|---|---|---|---|
| a = 0.998, i = 85 (disk) | 3.7e-4 | 2.1e-5 | 18x | 2.0e-4 | 5.6e-4 | 2.8x |
| a = 0.998, i = 85, no disk (sky) | 8.1e-4 | 4.1e-5 | 20x | 2.9e-4 | 7.1e-4 | 2.4x |
| a = 0.9, i = 72 (disk) | 9.5e-5 | 4.5e-6 | 21x | 6.4e-5 | 1.6e-4 | 2.5x |
| a = 0, i = 30 (disk) | 1.4e-4 | 9.8e-6 | 14x | 6.2e-5 | 1.7e-4 | 2.7x |

Floor: the f32 median at tol 1e-7. BS3 reaches its floor at 5.1x to 6.3x as many accepted steps as DP5 and 2.2x to 2.7x the force evaluations.

### 3. Where the f32 error comes from

Median px at the shipped tolerance. Each row removes everything except one source:

| Source | a = 0.998 disk | a = 0.998 sky | a = 0.9 disk | a = 0 disk |
|---|---|---|---|---|
| Truncation (f64, tol 1e-5) | 2.1e-5 | 1.4e-4 | 4.5e-6 | 9.8e-6 |
| f32 ray constants only | 4.2e-5 | 1.3e-4 | 1.3e-5 | 6.0e-6 |
| f32 output format only | 4.4e-5 | 9.4e-6 | 9.7e-6 | 1.5e-5 |
| **f32 as rendered** | **3.7e-4** | **3.0e-3** | **9.5e-5** | **1.4e-4** |

Even added linearly, those three fall 3.5x to 4.5x short of the rendered error on disk rays and about 11x short on sky rays (the a = 0.998 sky column is the disk frame's 980 sky rays). Computing the sky direction in f64 from the same f32 state barely helps (3.0e-3 to 2.4e-3). By elimination the rest is rounding in the state during integration. That is inferred, not measured directly.

At the f32 floor (tol 1e-7) the gap is smaller: the ray constants and output format, added linearly, reach about a third to two fifths of the floor on disk rays (2.4x to 3x short), and about a sixth on the no-disk sky rays. State rounding is the largest single source, not the only one.

### 4. Order and landing, from the older RK4

- **Sky rays, f64:** error falls by about 14x to 16x per step halving (4.2e-4, 3.1e-5, 1.9e-6, 1.2e-7): true 4th order.
- **Disk rays, f64:** only about 4x per halving (6.9e-3, 1.7e-3, 4.3e-4, 1.1e-4): 2nd order, set by the linear disk landing rather than by RK4.
- **Sky rays, f32:** error *rises* as steps shrink (7.3e-4, 9.2e-4, 1.3e-3, 1.8e-3), which looks like rounding accumulating with step count.

RK4 differs from DP5 in variables and landing as well as order, so the clean order result is BS3 against DP5.

### 5. The exception: rays that graze the pole

On the near-axis rays (|alpha| < 0.03) at the shipped tolerance:

| | f32, p99 / max | f64, p99 / max | f32 at tol 1e-7, max |
|---|---|---|---|
| a = 0.998 disk | 4.1 / 5.5 px | 0.88 / 1.0 px | 5.8e-3 px |
| a = 0.9 disk | 1.1 / 1.6 px | 0.35 / 0.48 px | 1.5e-3 px |

Here tightening the tolerance helps enormously, so this is not the f32 wall. The damage is concentrated at |alpha| < 1e-3. At a = 0.998 the f32 maximum falls from 5.5 px there to 0.15 px for 1e-3 to 1e-2.

- **Reach in the app:** at 1x zoom the nearest jittered ray to the axis is |alpha| = 4.9e-3 M, outside the worst band, so this should not be visible by default. At 16x zoom that distance falls to 3e-4 M, inside it, and a 1x pixel is 16 zoomed pixels. This is a prediction from the numbers; **it has not been looked for on screen.**
- **Reliability:** the largest values are not trustworthy as pixel figures. The check that the Jacobian holds fails for the 80 px (Schwarzschild) and 69 px (a = 0.998, no disk) rays, so those are only "large". They also differ between GPUs (80 px vs 13 px maximum). The table's rows pass that check well enough (worst ratio 0.35 for the a = 0.998 disk rays above 3 px).

### 6. What tightening the shipped tolerance would buy

| View | Median gain at tol 1e-6 | Extra attempts |
|---|---|---|
| a = 0.998 disk | 1.9x | +25 % |
| a = 0.998 no disk | 2.3x | +36 % |
| a = 0.9 disk | 1.5x | +21 % |
| a = 0 disk | 2.1x | +37 % |

Tol 1e-7 adds nothing for ordinary rays but is what fixes the near-axis rays, at +50 % to +240 % attempts. Nothing in the renderer was changed.

## Open questions

- **Why f32 at the shipped tolerance sits 1.5x to 3x above its own floor**, when f64 truncation there is at least 14x below it. Tolerance is reaching f32 error through something other than truncation. Candidate, untested: rounding in the stage sums growing with step size. f32 also takes fewer steps than f64 at the same tolerance (Schwarzschild, first run: 45 against 58).
- **Why a lower order raises the wall.** Between methods the floor ratio matches the square root of the step ratio (BS3 / DP5 accepted steps 5.1x to 6.3x, root 2.3x to 2.5x, floor 2.4x to 2.8x), which suggests a random walk. But within a method the floor stays flat as steps grow (DP5 36 to 55 steps, BS3 202 to 437 at a = 0.998), so that cannot be the whole story.
- **The state-rounding claim is by elimination.** The direct test: compensated summation of the state, which should lower the wall if this is right. That is the first "precision trick".

## Limits

- Four views, 4,304 rays each, two GPUs (RTX and Intel iGPU). DP5 medians on ordinary rays agree to about 1.4x. Sky-ray p99s as rendered differ by up to 5x (Schwarzschild, 4.2e-3 against 2.1e-2), while the same rays with the direction computed in f64 agree, which points at the iGPU's built-in sin and cos in that calculation (not proven). Near-axis maxima do not agree.
- Geometry only. A displacement is not a colour error; emission gradients decide what it does to a pixel.
- f32 RK4 uses the shipped monitor at every step scale, f64 RK4 a monitor tightened with the step. A tightened monitor in f32 sits below rounding noise and only rejects (first run: 8x the attempts and many fate flips), so that run was discarded.
- Pixel-coordinate rounding is excluded by design. It adds to the f32 arms in the real renderer.

## Self-checks

- **Reference:** DP5 f64 at tol 1e-11. Self-convergence against 1e-12: median 2e-10 px, max 2.7e-8 px, no flips. Against the independent Hamiltonian code on 173 rays per view: median 9e-10 px on the sky-only view, 5e-5 px on the views with a disk (the independent code's own linear landing), no flips; 1 to 9 of those rays did not converge in that code. So below 5e-5 px on disk views, the reference rests on self-convergence alone.
- **Method order** (`tests/mino-order.test.ts`): measured local error goes as h^4 for BS3 and h^6 for DP5, and the default stays bit-identical to DP5.
- **Fate flips**, DP5 f32: 0 of 4,304 at the shipped tolerance on both GPUs in all four views; at most 2 at tighter tolerances.
- **GPU loop against CPU:** f32 and f64 agree at tol 1e-3, where truncation dominates.
- **Renderer untouched:** every renderer shader and `gpu.ts` byte-identical to main; `?golden` PASS, 8 of 8 on the recording adapter. `npm test`: 310 passed, 14 skipped. The new `setMinoMethod` switch and `minoTraceRay` are CPU-only.

## Re-run

```
npx vite-node scripts/limit-cpu.ts edge-998       # also edge-998-nodisk, default-9, schw: 3 to 6 min each, side by side
npx vite                                          # dev server for the GPU arms
node scripts/limit-run.mjs edge-998               # add "igpu" for the integrated GPU: about 15 s each
node scripts/limit-plot.mjs                       # redraws limit-cost.svg
```

`results/` holds every arm's statistics and the per-ray near-axis scan. `data/` (rays, references, Jacobians; 3.9 MB) is regenerated by `limit-cpu.ts` and not committed.
