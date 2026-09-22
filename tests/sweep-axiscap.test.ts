import { describe, it, expect } from "vitest";
import { screenToState } from "../src/physics/camera";
import { stepGeodesic, traceRay, F_AXIS, DL_FAR_MIN, H_TOL, H_TOL_FAR, MAX_RETRY, K_FAR, DL_FAR_MAX, type Fate } from "../src/physics/trace";

/**
 * Measurement, not a test: prints the F_AXIS x DL_FAR_MIN table the far-field angular step cap in
 * trace.ts was chosen from. Skipped unless SWEEP=1 (PowerShell: $env:SWEEP=1; npx vitest run
 * tests/sweep-axiscap.test.ts).
 *
 * The loop below is a LOCAL COPY of traceRay/stepSize (not an option on the shipped code) so that
 * (a) the converged reference can be monitored everywhere -- the shipped far-field exemption
 * H_TOL_FAR is exactly what the cap compensates for, so the reference must not have it -- and
 * (b) the cap constants can be varied without touching the twins. The "shipped" row (cap off) is
 * the pre-cap controller; the test asserts the local loop with the shipped constants reproduces
 * traceRay exactly on every ray, so the table is about the shipped code, not a look-alike.
 *
 * Sets (a = 0, rIn = 6, rOut = 40, rObs = 1000):
 *   A: i = 8 deg, beta in [9, 14] x alpha in [0, 0.8] M on an 11 x 11 grid (121 rays) -- the band
 *      beside the axis column above the shadow where every ray crosses the axis at r > 60, inside
 *      the exempt zone;
 *   B: i = 1 deg, beta in [2, 14] x alpha in [0, 1] on a 13 x 5 grid (65 rays) -- the axis column
 *      of a nearly pole-on view, which crosses the axis in the far field for every beta.
 * Reference: hTol = 1e-7, maxRetry = 24, monitored everywhere, cap 0.05/0.02 (the tightest
 * candidate); checked against a second reference (1e-8, 28, cap 0.02/0.01) for convergence. The
 * review's reference (1e-6, 20, monitored everywhere, no cap) was tried first and is NOT converged:
 * it differs from (1e-7, 24) by 0.0196 M on set A (beta = 14, alpha = 0.32) -- the relative monitor
 * cannot reject a far-field barrier step whose inflated momenta inflate the scale, so monitoring
 * alone does not converge there; the four tighter variants agree to <= 3e-3 M on every ray.
 * A ray is "wrong" when its fate differs from the reference or |rHit - ref| > 0.02 M (half a pixel
 * at ~24 px/M); a "flip" is a fate difference.
 *
 * Selection rule (controller ruling, final review C1): 0 flips on both sets, then the fewest wrong
 * rays, subject to <= +15 % mean steps on set A relative to the shipped (uncapped) controller.
 *
 * Table measured 2026-09-17 (f64; reference convergence 2.0e-3 M; reference fates A: 121 disk,
 * B: 40 disk / 20 captured / 5 escaped):
 *   F_AXIS/DL_MIN      A.flips  A.wrong  A.steps  A.dSteps%   B.flips  B.wrong  B.steps  B.dSteps%
 *   uncapped (pre-C1)  16       98       318.4    0.0         1        10       471.9    0.0
 *   0.5/0.05           1        97       326.1    2.4         1        10       476.0    0.9
 *   0.5/0.02           1        97       326.5    2.5         1        10       476.7    1.0
 *   0.25/0.05          1        87       332.4    4.4         1        10       469.3    -0.6
 *   0.25/0.02          1        87       333.1    4.6         1        10       470.9    -0.2
 *   0.1/0.05  <- ship  0        45       356.7    12.0        0        5        493.3    4.5
 *   0.1/0.02           0        45       358.4    12.6        0        5        496.9    5.3
 *   0.05/0.05          1        35       405.2    27.3        0        4        546.0    15.7
 *   0.05/0.02          1        37       408.1    28.2        0        4        553.5    17.3
 *   monitored-far      0        95       325.5    2.2         1        10       471.9    0.0
 * Only F_AXIS = 0.1 has 0 flips on both sets; the two DL_FAR_MIN values tie on wrong rays (45 + 5)
 * and 0.05 is the cheaper, so F_AXIS = 0.1, DL_FAR_MIN = 0.05. The informational "monitored-far"
 * row (H_TOL = 1e-3 everywhere, no cap; not shippable on the GPU, see H_TOL_FAR) has 0 flips but
 * 95 wrong: the RELATIVE monitor does not bound the accuracy of a far-field barrier step (inflated
 * momenta inflate its scale), so the cap, not the monitor, is what buys accuracy there. The
 * remaining 45 wrong rays at 0.1/0.05 are radius errors > 0.02 M (half a pixel) with the right
 * fate; a tighter cap trades them for steps (0.05: 35 wrong at +27 %).
 *
 * Confirmed under exact forces (metricUpperGrad, 2026-09-23): the table is identical to the one
 * above to the printed precision except A.steps 0.25/0.05 332.5 (was 332.4), B.steps 0.25/0.05
 * 469.2, 0.1/0.05 493.2, 0.1/0.02 496.8, 0.05/0.02 553.4 (each 0.1 lower); flips and wrong counts
 * unchanged, reference convergence 1.18e-3 M (was 2.0e-3), reference fates unchanged. Same
 * selection: F_AXIS = 0.1, DL_FAR_MIN = 0.05.
 */
const SWEEP = !!process.env.SWEEP;
const ROBS = 1000, RIN = 6, ROUT = 40;

type Opts = { hTol: number; maxRetry: number; monitorFar: boolean; fAxis: number; dlFarMin: number; maxSteps?: number };
type Res = { fate: Fate; rHit: number; steps: number };

/** Local copy of stepSize (trace.ts) with the cap constants as parameters; fAxis = Infinity disables the cap. */
function stepSizeLocal(s: Float64Array, rh: number, rOut: number, fAxis: number, dlFarMin: number): number {
  const r = s[1];
  if (r > rOut * 1.5) {
    const base = Math.min(DL_FAR_MAX, Math.max(0.6, K_FAR * r));
    const pth = s[6];
    if (pth === 0 || fAxis === Infinity) return base;
    const thD = Math.min(s[2], Math.PI - s[2]);
    return Math.min(base, Math.max(dlFarMin, fAxis * thD * r * r / Math.abs(pth)));
  }
  return Math.min(0.5, Math.max(0.002, 0.02 * (r - rh)));
}

/** Local copy of traceRay (trace.ts): same termination order and the same |delta theta| disk guard. */
function traceLocal(s0: Float64Array, a: number, o: Opts): Res {
  const rh = 1 + Math.sqrt(Math.max(0, 1 - a * a));
  const maxSteps = o.maxSteps ?? 20000; // traceRay's default; the tight references need more
  let s = s0;
  for (let step = 1; step <= maxSteps; step++) {
    const far = s[1] > ROUT * 1.5;
    const hTol = far && !o.monitorFar ? H_TOL_FAR : o.hTol;
    const out = stepGeodesic(s, a, stepSizeLocal(s, rh, ROUT, o.fAxis, o.dlFarMin), hTol, o.maxRetry);
    const sN = out.s;
    const f0 = s[2] - Math.PI / 2, f1 = sN[2] - Math.PI / 2;
    if (f0 * f1 < 0 && Math.abs(sN[2] - s[2]) < 0.5) {
      const frac = f0 / (f0 - f1);
      const rHit = s[1] + frac * (sN[1] - s[1]);
      if (rHit >= RIN && rHit <= ROUT) return { fate: "disk", rHit, steps: step };
    }
    s = sN;
    if (s[1] <= rh * 1.005) return { fate: "captured", rHit: NaN, steps: step };
    if (s[1] > ROBS * 1.2) return { fate: "escaped", rHit: NaN, steps: step };
  }
  return { fate: "budget", rHit: NaN, steps: maxSteps };
}

function grid(incl: number, betas: number[], alphas: number[]): Float64Array[] {
  const out: Float64Array[] = [];
  for (const beta of betas) for (const alpha of alphas) out.push(screenToState(alpha, beta, 0, incl, ROBS));
  return out;
}
const lin = (lo: number, hi: number, n: number) => Array.from({ length: n }, (_, k) => lo + ((hi - lo) * k) / (n - 1));
const SET_A = grid((8 * Math.PI) / 180, lin(9, 14, 11), lin(0, 0.8, 11));
const SET_B = grid((1 * Math.PI) / 180, lin(2, 14, 13), lin(0, 1, 5));

function score(rays: Float64Array[], ref: Res[], o: Opts) {
  let flips = 0, wrong = 0, steps = 0;
  rays.forEach((s0, k) => {
    const r = traceLocal(s0, 0, o);
    steps += r.steps;
    const flip = r.fate !== ref[k].fate;
    if (flip) flips++;
    if (flip || (r.fate === "disk" && Math.abs(r.rHit - ref[k].rHit) > 0.02)) wrong++;
  });
  return { flips, wrong, meanSteps: steps / rays.length };
}

describe.skipIf(!SWEEP)("F_AXIS / DL_FAR_MIN sweep (SWEEP=1)", () => {
  it("prints the table", () => {
    const refOpts: Opts = { hTol: 1e-7, maxRetry: 24, monitorFar: true, fAxis: 0.05, dlFarMin: 0.02, maxSteps: 40000 };
    const refA = SET_A.map((s) => traceLocal(s, 0, refOpts)), refB = SET_B.map((s) => traceLocal(s, 0, refOpts));
    // Convergence: a tighter reference must agree on every fate and to a quarter of the 0.02 M
    // "wrong" threshold on every disk hit.
    const ref2: Opts = { hTol: 1e-8, maxRetry: 28, monitorFar: true, fAxis: 0.02, dlFarMin: 0.01, maxSteps: 40000 };
    let maxConv = 0;
    for (const [set, ref] of [[SET_A, refA], [SET_B, refB]] as const) {
      set.forEach((s, k) => {
        const r = traceLocal(s, 0, ref2);
        expect(r.fate).toBe(ref[k].fate);
        if (r.fate === "disk") maxConv = Math.max(maxConv, Math.abs(r.rHit - ref[k].rHit));
      });
    }
    expect(maxConv).toBeLessThan(5e-3);
    console.log(`reference convergence: max |rHit(1e-7,24,cap.05/.02) - rHit(1e-8,28,cap.02/.01)| = ${maxConv.toExponential(2)} M`);
    const fates = (rs: Res[]) => rs.reduce((m, r) => ((m[r.fate] = (m[r.fate] ?? 0) + 1), m), {} as Record<string, number>);
    console.log("reference fates A", JSON.stringify(fates(refA)), "B", JSON.stringify(fates(refB)));

    // The local loop with the shipped constants must BE the shipped traceRay, ray for ray.
    const shippedOpts: Opts = { hTol: H_TOL, maxRetry: MAX_RETRY, monitorFar: false, fAxis: F_AXIS, dlFarMin: DL_FAR_MIN };
    for (const s of [...SET_A, ...SET_B]) {
      const l = traceLocal(s, 0, shippedOpts), t = traceRay(s, 0, { rIn: RIN, rOut: ROUT, rObs: ROBS });
      expect(l.fate).toBe(t.fate);
      expect(l.steps).toBe(t.steps);
      if (l.fate === "disk") expect(l.rHit).toBe(t.rHit!);
    }

    const rows: { label: string; o: Partial<Opts> }[] = [{ label: "uncapped (pre-C1)", o: { fAxis: Infinity, dlFarMin: 0 } }];
    for (const fAxis of [0.5, 0.25, 0.1, 0.05]) for (const dlFarMin of [0.05, 0.02]) rows.push({ label: `${fAxis}/${dlFarMin}`, o: { fAxis, dlFarMin } });
    rows.push({ label: "monitored-far", o: { fAxis: Infinity, dlFarMin: 0, monitorFar: true } }); // informational only
    const base: Opts = { hTol: H_TOL, maxRetry: MAX_RETRY, monitorFar: false, fAxis: Infinity, dlFarMin: 0 };
    const baseA = score(SET_A, refA, base), baseB = score(SET_B, refB, base);
    console.log("F_AXIS/DL_MIN      A.flips  A.wrong  A.steps  A.dSteps%   B.flips  B.wrong  B.steps  B.dSteps%");
    for (const row of rows) {
      const o: Opts = { ...base, ...row.o };
      const A = score(SET_A, refA, o), B = score(SET_B, refB, o);
      const dA = (100 * (A.meanSteps / baseA.meanSteps - 1)).toFixed(1), dB = (100 * (B.meanSteps / baseB.meanSteps - 1)).toFixed(1);
      console.log(`${row.label.padEnd(18)} ${String(A.flips).padEnd(8)} ${String(A.wrong).padEnd(8)} ${A.meanSteps.toFixed(1).padEnd(8)} ${dA.padEnd(11)} ${String(B.flips).padEnd(8)} ${String(B.wrong).padEnd(8)} ${B.meanSteps.toFixed(1).padEnd(8)} ${dB}`);
    }
  }, 1_800_000);
});
