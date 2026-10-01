import { describe, it, expect } from "vitest";
import { screenToState } from "../src/physics/camera";
import { iscoRadius } from "../src/physics/orbits";
import { stepGeodesic, stepSize, traceRay, H_TOL, H_TOL_FAR } from "../src/physics/trace";
import { inJetEnvelope } from "../src/physics/jet";

/**
 * Measurement, not a test: how much of the screen the geodesic cache must bookmark for the jet, and
 * what share of a full trace the per-frame jet replay re-integrates (spec 2026-10-01 4, first risk).
 * Skipped unless SWEEP=1. The loop is a LOCAL COPY of traceRay that also records the first and last
 * step whose START state is inside the jet envelope; it first asserts it reproduces traceRay's fate
 * and step count on every ray, so the numbers are about the shipped integrator.
 *
 * Grid: 64 x 36 pixel centres over the shader's screen mapping (fovScale 14, aspect 16:9), maxSteps
 * 4800, rOut 40, rObs 1000, jetLength 60.
 *
 * Measured 2026-10-01 (f64):
 *   default a=0.9 i=72     bookmarked 6.4 %  mean nJet 37.8  replay share 0.7 % of all steps  (mean steps 333.8)
 *   edge-on a=0.99 i=85    bookmarked 10.7 %  mean nJet 34.6  replay share 0.8 % of all steps  (mean steps 473.4)
 *   face-on a=0.9 i=8      bookmarked 13.6 %  mean nJet 71.7  replay share 3.2 % of all steps  (mean steps 309.1)
 * Gate (plan Task 1 Step 7): default replay share 0.7 % <= 25 % -> proceed.
 * BOOKMARK_FRAC = max(0.05, ceil(1.5 x 13.6) / 100) = 0.21.
 */
const RUN = process.env.SWEEP === "1";
const ROBS = 1000, ROUT = 40, FOV = 14, MAX_STEPS = 4800, JET_LEN = 60, NX = 64, NY = 36;

function traceWithEnvelope(s0: Float64Array, a: number, rIn: number) {
  const rh = 1 + Math.sqrt(Math.max(0, 1 - a * a));
  let s = s0, first = -1, last = -1;
  for (let step = 1; step <= MAX_STEPS; step++) {
    if (inJetEnvelope(s[1], s[2], JET_LEN)) { if (first < 0) first = step; last = step; }
    const far = s[1] > ROUT * 1.5;
    const sN = stepGeodesic(s, a, stepSize(s, rh, ROUT), far ? H_TOL_FAR : H_TOL).s;
    const f0 = s[2] - Math.PI / 2, f1 = sN[2] - Math.PI / 2;
    if (f0 * f1 < 0 && Math.abs(sN[2] - s[2]) < 0.5) {
      const rHit = s[1] + (f0 / (f0 - f1)) * (sN[1] - s[1]);
      if (rHit >= rIn && rHit <= ROUT) return { fate: "disk", steps: step, first, last };
    }
    s = sN;
    if (s[1] <= rh * 1.005) return { fate: "captured", steps: step, first, last };
    if (s[1] > ROBS * 1.2) return { fate: "escaped", steps: step, first, last };
  }
  return { fate: "budget", steps: MAX_STEPS, first, last };
}

const VIEWS = [
  { name: "default a=0.9 i=72", a: 0.9, incl: 72 },
  { name: "edge-on a=0.99 i=85", a: 0.99, incl: 85 },
  { name: "face-on a=0.9 i=8", a: 0.9, incl: 8 },
];

describe.skipIf(!RUN)("jet envelope sweep", () => {
  it("reports bookmark fraction, mean nJet and replay share per view", () => {
    for (const v of VIEWS) {
      const incl = (v.incl * Math.PI) / 180, rIn = iscoRadius(v.a, true);
      let marked = 0, sumNJet = 0, sumSteps = 0;
      for (let iy = 0; iy < NY; iy++) for (let ix = 0; ix < NX; ix++) {
        const alpha = (((ix + 0.5) / NX) * 2 - 1) * FOV * (NX / NY);
        const beta = -(((iy + 0.5) / NY) * 2 - 1) * FOV;
        const s0 = screenToState(alpha, beta, v.a, incl, ROBS);
        const ref = traceRay(s0, v.a, { rIn, rOut: ROUT, rObs: ROBS, maxSteps: MAX_STEPS });
        const got = traceWithEnvelope(s0, v.a, rIn);
        expect(got.fate).toBe(ref.fate);
        expect(got.steps).toBe(ref.steps);
        sumSteps += got.steps;
        if (got.first >= 0) { marked++; sumNJet += got.last - got.first + 1; }
      }
      const n = NX * NY;
      console.log(`${v.name.padEnd(22)} bookmarked ${(100 * marked / n).toFixed(1)} %  ` +
        `mean nJet ${(marked ? sumNJet / marked : 0).toFixed(1)}  ` +
        `replay share ${(100 * sumNJet / sumSteps).toFixed(1)} % of all steps  (mean steps ${(sumSteps / n).toFixed(1)})`);
    }
  }, 600_000);
});
