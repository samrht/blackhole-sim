import { describe, it, expect } from "vitest";
import { screenToState } from "../src/physics/camera";
import { traceRay, type Fate } from "../src/physics/trace";
import { photonOrbit } from "../src/physics/orbits";

/**
 * Measurement, not a test: prints the H_TOL x MAX_RETRY table the constants in trace.ts were
 * chosen from. Skipped unless SWEEP=1 (PowerShell: $env:SWEEP=1; npx vitest run tests/sweep-htol.test.ts).
 * Two views: the spec's worst case (a = 0, i = 8 deg, broad axis wedge) and the default UI view
 * (a = 0.9, i = 72 deg). 32x32 rays per view at fovScale 14, the interactive default.
 *
 * The committed table (trace.ts) was measured before the far-field exemption: traceRay now applies
 * the hTol override only for r < 1.5 * rOut and uses H_TOL_FAR beyond it, so a re-run's retries and
 * meanSteps columns may differ from the committed numbers. The selection is unaffected -- the
 * near-axis accuracy criterion and the f32 floor both live in the monitored near field.
 *
 * Confirmed under exact forces (metricUpperGrad, 2026-09-23; retry cap columns 4/8/12 identical):
 *   hTol  meanSteps  retries  exhausted  budget  disk/captured/escaped  |rHit-truth| M
 *   1e-2  338.7      10       0          0       1913/87/48             0.3381  (fails accuracy)
 *   1e-3  338.7      53       0          0       1913/87/48             0.0098  <- ship (largest survivor)
 *   1e-4  338.7      238      0          0       1913/87/48             0.0098
 *   1e-5  338.9      1000     0          0       1913/87/48             0.0033
 *   1e-6  339.3      3154     0          0       1913/87/48             0.0001  (below the f32 floor)
 * Converged reference rHit 4.86391 M. Same selection: H_TOL = 1e-3, MAX_RETRY = 8.
 */
const SWEEP = !!process.env.SWEEP;
const ROBS = 1000, FOV = 14, N = 32, STEPS = 4800;
const VIEWS = [{ a: 0, incl: (8 * Math.PI) / 180 }, { a: 0.9, incl: (72 * Math.PI) / 180 }];

function sweepOne(hTol: number, maxRetry: number) {
  let steps = 0, retries = 0, exhausted = 0, rays = 0;
  const fates: Record<Fate, number> = { disk: 0, captured: 0, escaped: 0, budget: 0 };
  for (const v of VIEWS) {
    const rIn = photonOrbit(v.a, true);
    for (let iy = 0; iy < N; iy++) for (let ix = 0; ix < N; ix++) {
      const alpha = (((ix + 0.5) / N) * 2 - 1) * FOV, beta = -(((iy + 0.5) / N) * 2 - 1) * FOV;
      const r = traceRay(screenToState(alpha, beta, v.a, v.incl, ROBS), v.a,
        { rIn, rOut: 40, rObs: ROBS, maxSteps: STEPS, hTol, maxRetry });
      steps += r.steps; retries += r.retries; exhausted += r.exhausted; rays++; fates[r.fate]++;
    }
  }
  return { meanSteps: steps / rays, retries, exhausted, fates };
}

/** Disk-hit radius of the spec's reference near-axis ray (theta_min ~ 1.2e-3 rad). */
function nearAxisHit(hTol: number, maxRetry: number) {
  const r = traceRay(screenToState(0.05, 6, 0, (8 * Math.PI) / 180, ROBS), 0,
    { rIn: 3, rOut: 40, rObs: ROBS, hTol, maxRetry });
  return r.fate === "disk" ? r.rHit! : NaN;
}

describe.skipIf(!SWEEP)("H_TOL / MAX_RETRY sweep (SWEEP=1)", () => {
  it("prints the table", () => {
    // Converged reference: two successively tighter settings must agree to 1e-4 M or the
    // "truth" is not converged and the table is meaningless.
    const truthA = nearAxisHit(1e-8, 24), truthB = nearAxisHit(1e-9, 28);
    expect(Math.abs(truthA - truthB)).toBeLessThan(1e-4);
    console.log(`near-axis reference rHit (converged) = ${truthA.toFixed(5)} M`);
    console.log("hTol      retry  meanSteps  retries  exhausted  budget  disk  captured  escaped  |rHit-truth| M");
    for (const hTol of [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]) {
      for (const maxRetry of [4, 8, 12]) {
        const s = sweepOne(hTol, maxRetry);
        const err = Math.abs(nearAxisHit(hTol, maxRetry) - truthA);
        console.log(`${hTol.toExponential(0).padEnd(9)} ${String(maxRetry).padEnd(6)} ${s.meanSteps.toFixed(1).padEnd(10)} ${String(s.retries).padEnd(8)} ${String(s.exhausted).padEnd(10)} ${String(s.fates.budget).padEnd(7)} ${String(s.fates.disk).padEnd(5)} ${String(s.fates.captured).padEnd(9)} ${String(s.fates.escaped).padEnd(8)} ${err.toFixed(4)}`);
      }
    }
  }, 1_800_000);
});
