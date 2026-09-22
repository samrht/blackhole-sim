import { describe, it, expect } from "vitest";
import { screenToState } from "../src/physics/camera";
import { metricUpper } from "../src/physics/kerr";
import { photonOrbit } from "../src/physics/orbits";
import { stepGeodesic, traceRay, F_AXIS, DL_FAR_MIN, H_TOL, H_TOL_FAR, MAX_RETRY, K_FAR, DL_FAR_MAX } from "../src/physics/trace";

/**
 * Measurement, not a test: prints the K_FAR x DL_FAR_MAX table the far-field stride in trace.ts was
 * chosen from (spec 2026-09-23 3.4). Skipped unless SWEEP=1.
 *
 * The loop is a LOCAL COPY of traceRay with the far base, the angular cap, the tolerances and the
 * retry cap parameterised; the test first asserts that with the shipped constants it reproduces
 * traceRay exactly on every ray, so the table is about the shipped code.
 *
 * Sets: V1 = the a = 0, i = 8 deg view and V2 = the a = 0.9, i = 72 deg view on 32 x 32 grids over
 * +-14 M, plus the two near-axis sets of sweep-axiscap.test.ts (A: 121-ray band at i = 8 deg,
 * B: 65-ray axis column at i = 1 deg). 2234 rays.
 *
 * Reference (controller ruling, 2026-09-23). The first cut used the SHIPPED stride as the reference
 * and checked it against a finer stride (K = 0.02, MAX = 3) under the shipped controller; that
 * check failed -- "flips 0, max disk 7.16e+0 M, max sky 1.67e-2 rad (half pixel 1.94e-5)" -- because
 * the shipped controller is not converged near the axis (set A: 54 of 121 disk hits moved > 0.02 M,
 * up to 7.16 M; still 48 wrong at 0.01/1.5; set B: 0.69 M, 1.7e-2 rad) and near-critical escaped
 * rays in V2 disagree by ~1e-3 rad between any two strides, so an absolute half-pixel sky rule
 * against it is unreachable. Now: REF1 = local loop monitored everywhere, hTol 1e-7, maxRetry 24,
 * far base K 0.01 / MAX 1.5, angular cap 0.05 / 0.02; checked against REF2 = 1e-8, 28, 0.005 / 0.75,
 * cap 0.02 / 0.01. A ray is UNCONVERGED (excluded from scoring) if REF1 and REF2 differ in fate, by
 * > 5e-3 M in disk radius, or by > half a pixel in sky direction; > 25 % unconverged in any set
 * would block the sweep.
 *
 * Rule: "no ray worse than today by more than half a pixel". errShipped = the shipped pair's
 * error against REF1 per scored ray (shipped controller: H_TOL near, H_TOL_FAR far, shipped cap).
 * A candidate passes iff on every scored ray: no fate flip where the shipped pair has none; disk
 * |rHit - ref| <= errShipped + 0.02 M; sky angle <= errShipped + HALF_PIXEL (1.94e-5 rad). Then
 * fewest mean steps (over ALL rays -- it is the cost); ties -> smaller K, then smaller MAX.
 *
 * Measured 2026-09-23 (f64, 1.1 min):
 *   reference V1 a=0 i=8: unconv 0/1024 (0.0 %), ref1-vs-ref2 max disk 5.56e-4 M, max sky 0 rad, fates {"disk":966,"captured":58}
 *   reference V2 a=0.9 i=72: unconv 0/1024 (0.0 %), ref1-vs-ref2 max disk 1.91e-3 M, max sky 1.55e-6 rad, fates {"disk":947,"captured":29,"escaped":48}
 *   reference A axis band i=8: unconv 0/121 (0.0 %), ref1-vs-ref2 max disk 3.00e-4 M, max sky 0 rad, fates {"disk":121}
 *   reference B axis column i=1: unconv 3/65 (4.6 %), ref1-vs-ref2 max disk 9.80e-4 M, max sky 7.43e-5 rad, fates {"captured":20,"escaped":5,"disk":40}
 *   K_FAR  DL_MAX  newFlips  diskWorse  skyWorse  maxDiskDelta(M)  maxSkyDelta(rad)  meanSteps  dSteps%
 *   0.04   6       0         0          0         0.00e+0          0.00e+0           344.1      0.0
 *   0.04   20      0         48         3         1.68e+1          7.72e-4           254.4      -26.1
 *   0.04   50      0         51         3         1.55e+1          8.93e-3           246.2      -28.5
 *   0.04   150     0         51         3         1.55e+1          8.93e-3           246.2      -28.5
 *   0.08   6       0         56         3         1.53e+1          2.72e-3           335.0      -2.6
 *   0.08   20      0         75         1         1.49e+1          8.96e-3           232.2      -32.5
 *   0.08   50      0         73         3         1.42e+1          8.61e-3           212.8      -38.2
 *   0.08   150     0         72         3         1.98e+1          3.13e-3           211.3      -38.6
 *   0.12   6       0         56         2         1.53e+1          2.72e-3           336.3      -2.3
 *   0.12   20      3         80         3         1.94e+1          8.96e-3           226.4      -34.2
 *   0.12   50      2         78         3         2.39e+1          2.77e-1           204.1      -40.7
 *   0.12   150     4         76         3         1.56e+1          3.35e-3           190.9      -44.5
 *   0.2    6       0         56         2         1.53e+1          2.72e-3           336.3      -2.3
 *   0.2    20      7         101        3         2.77e+1          8.96e-3           215.8      -37.3
 *   0.2    50      6         81         49        2.19e+1          5.21e-1           190.8      -44.5
 *   0.2    150     7         91         47        2.80e+1          1.08e-1           176.0      -48.9
 *   0.3    6       0         56         2         1.53e+1          2.72e-3           336.3      -2.3
 *   0.3    20      10        102        49        2.46e+1          8.96e-3           205.5      -40.3
 *   0.3    50      7         131        48        2.63e+1          5.21e-1           187.7      -45.5
 *   0.3    150     14        133        48        2.52e+1          1.08e-1           172.5      -49.9
 *   SELECTED K_FAR = 0.04, DL_FAR_MAX = 6 (mean steps 344.1 vs 344.1)
 * No longer stride survives: every candidate makes some rays worse by > half a pixel, and not only
 * on the axis sets -- per set (scratch breakdown) 0.04/20 has 6 V1 disk rays and 2 V2 sky rays
 * worse, 29 on A, 13 + 1 on B; 0.08/6 has 6 V1, 2 + 2 V2, 48 A. The shipped stride stays; the far
 * field is not where the default view's cost can be cut without visible error.
 */
const SWEEP = !!process.env.SWEEP;
const ROBS = 1000, ROUT = 40, FOV = 14, MAXSTEPS = 4800, REF_MAXSTEPS = 400000;
const GRID_N = 32;
const HALF_PIXEL = 0.5 * (2 * FOV) / (720 * ROBS);

type Ray = { s0: Float64Array; a: number; rIn: number };
type Res = { fate: string; steps: number; rHit?: number; dir?: number[] };
/** Integrator configuration: far base (K, MAXD), angular cap (fAxis, dlFarMin), near/far monitor. */
type Cfg = { K: number; MAXD: number; fAxis: number; dlFarMin: number; hNear: number; hFar: number; maxRetry: number; maxSteps: number };

const shippedCfg = (K: number, MAXD: number): Cfg =>
  ({ K, MAXD, fAxis: F_AXIS, dlFarMin: DL_FAR_MIN, hNear: H_TOL, hFar: H_TOL_FAR, maxRetry: MAX_RETRY, maxSteps: MAXSTEPS });
/** Reference: monitored everywhere, fine far base, tight angular cap. */
const REF1: Cfg = { K: 0.01, MAXD: 1.5, fAxis: 0.05, dlFarMin: 0.02, hNear: 1e-7, hFar: 1e-7, maxRetry: 24, maxSteps: REF_MAXSTEPS };
/** Convergence check on the reference: everything tightened again. */
const REF2: Cfg = { K: 0.005, MAXD: 0.75, fAxis: 0.02, dlFarMin: 0.01, hNear: 1e-8, hFar: 1e-8, maxRetry: 28, maxSteps: REF_MAXSTEPS };

function skyDirCPU(s: Float64Array, a: number): number[] {
  const r = s[1], th = s[2], ph = s[3];
  const g = metricUpper(r, th, a);
  const dr = g.rr * s[5], dth = g.thth * s[6], dph = g.tphi * s[4] + g.phph * s[7];
  const st = Math.sin(th), ct = Math.cos(th), sp = Math.sin(ph), cp = Math.cos(ph);
  const v = [dr * st * cp + r * ct * cp * dth - r * st * sp * dph,
             dr * st * sp + r * ct * sp * dth + r * st * cp * dph,
             dr * ct - r * st * dth];
  const n = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / n, v[1] / n, v[2] / n];
}

function stepSizeLocal(s: Float64Array, rh: number, c: Cfg): number {
  const r = s[1];
  if (r > ROUT * 1.5) {
    const base = Math.min(c.MAXD, Math.max(0.6, c.K * r));
    const pth = s[6];
    if (pth === 0) return base;
    const thD = Math.min(s[2], Math.PI - s[2]);
    return Math.min(base, Math.max(c.dlFarMin, c.fAxis * thD * r * r / Math.abs(pth)));
  }
  return Math.min(0.5, Math.max(0.002, 0.02 * (r - rh)));
}

function localTrace(ray: Ray, c: Cfg): Res {
  const { a, rIn } = ray, rh = 1 + Math.sqrt(Math.max(0, 1 - a * a));
  let s = ray.s0;
  for (let step = 1; step <= c.maxSteps; step++) {
    const far = s[1] > ROUT * 1.5;
    const sN = stepGeodesic(s, a, stepSizeLocal(s, rh, c), far ? c.hFar : c.hNear, c.maxRetry).s;
    const f0 = s[2] - Math.PI / 2, f1 = sN[2] - Math.PI / 2;
    if (f0 * f1 < 0 && Math.abs(sN[2] - s[2]) < 0.5) {
      const frac = f0 / (f0 - f1), rHit = s[1] + frac * (sN[1] - s[1]);
      if (rHit >= rIn && rHit <= ROUT) return { fate: "disk", steps: step, rHit };
    }
    s = sN;
    if (s[1] <= rh * 1.005) return { fate: "captured", steps: step };
    if (s[1] > ROBS * 1.2) return { fate: "escaped", steps: step, dir: skyDirCPU(s, a) };
  }
  return { fate: "budget", steps: c.maxSteps };
}

function sets(): { name: string; rays: Ray[] }[] {
  const grid = (a: number, inclDeg: number, n: number, rIn: number) => {
    const incl = (inclDeg * Math.PI) / 180, rays: Ray[] = [];
    for (let iy = 0; iy < n; iy++) for (let ix = 0; ix < n; ix++) {
      const alpha = (((ix + 0.5) / n) * 2 - 1) * FOV, beta = -(((iy + 0.5) / n) * 2 - 1) * FOV;
      rays.push({ s0: screenToState(alpha, beta, a, incl, ROBS), a, rIn });
    }
    return rays;
  };
  const band = (inclDeg: number, betas: number[], alphas: number[]) =>
    betas.flatMap((b) => alphas.map((al) => ({ s0: screenToState(al, b, 0, (inclDeg * Math.PI) / 180, ROBS), a: 0, rIn: 6 })));
  const lin = (lo: number, hi: number, n: number) => Array.from({ length: n }, (_, k) => lo + ((hi - lo) * k) / (n - 1));
  return [
    { name: "V1 a=0 i=8", rays: grid(0, 8, GRID_N, photonOrbit(0, true)) },
    { name: "V2 a=0.9 i=72", rays: grid(0.9, 72, GRID_N, photonOrbit(0.9, true)) },
    { name: "A axis band i=8", rays: band(8, lin(9, 14, 11), lin(0, 0.8, 11)) },
    { name: "B axis column i=1", rays: band(1, lin(2, 14, 13), lin(0, 1, 5)) },
  ];
}

const angle = (u: number[], v: number[]) => 2 * Math.asin(Math.min(1, Math.hypot(u[0] - v[0], u[1] - v[1], u[2] - v[2]) / 2));

/** Error of r against the reference q (same fate required): |drHit| for disk, sky angle for escaped, 0 otherwise. */
const err = (r: Res, q: Res) => (r.fate === "disk" ? Math.abs(r.rHit! - q.rHit!) : r.fate === "escaped" ? angle(r.dir!, q.dir!) : 0);

describe.skipIf(!SWEEP)("far-field stride sweep (SWEEP=1)", () => {
  it("prints the table and the selection", () => {
    const S = sets();
    const t0 = Date.now();
    // 1) the local loop IS the shipped loop at the shipped constants
    for (const set of S) for (const ray of set.rays) {
      const loc = localTrace(ray, shippedCfg(K_FAR, DL_FAR_MAX));
      const ship = traceRay(ray.s0, ray.a, { rIn: ray.rIn, rOut: ROUT, rObs: ROBS, maxSteps: MAXSTEPS });
      expect(loc.fate).toBe(ship.fate);
      expect(loc.steps).toBe(ship.steps);
      if (loc.fate === "disk") expect(loc.rHit).toBe(ship.rHit);
    }
    // 2) the reference and its convergence: rays on which REF1 and REF2 disagree are UNCONVERGED
    //    and excluded from scoring.
    const ref = S.map((set) => set.rays.map((ray) => localTrace(ray, REF1)));
    const ref2 = S.map((set) => set.rays.map((ray) => localTrace(ray, REF2)));
    const scored = S.map((set, si) => set.rays.map((_, ri) => {
      const p = ref[si][ri], q = ref2[si][ri];
      if (p.fate !== q.fate || p.fate === "budget") return false;
      if (p.fate === "disk") return Math.abs(p.rHit! - q.rHit!) <= 5e-3;
      if (p.fate === "escaped") return angle(p.dir!, q.dir!) <= HALF_PIXEL;
      return true;
    }));
    let blocked = false;
    S.forEach((set, si) => {
      const n = set.rays.length, un = scored[si].filter((x) => !x).length;
      let maxD = 0, maxS = 0; const fates: Record<string, number> = {};
      set.rays.forEach((_, ri) => {
        const p = ref[si][ri], q = ref2[si][ri];
        fates[p.fate] = (fates[p.fate] || 0) + 1;
        if (p.fate === q.fate && p.fate === "disk") maxD = Math.max(maxD, Math.abs(p.rHit! - q.rHit!));
        if (p.fate === q.fate && p.fate === "escaped") maxS = Math.max(maxS, angle(p.dir!, q.dir!));
      });
      console.log(`reference ${set.name}: unconv ${un}/${n} (${((100 * un) / n).toFixed(1)} %), ref1-vs-ref2 max disk ${maxD.toExponential(2)} M, max sky ${maxS.toExponential(2)} rad, fates ${JSON.stringify(fates)}`);
      if (un > 0.25 * n) blocked = true;
    });
    console.log(`references done in ${((Date.now() - t0) / 60000).toFixed(1)} min`);
    expect(blocked).toBe(false);
    // 3) shipped errors per scored ray (under the shipped controller at the shipped pair)
    const ship = S.map((set) => set.rays.map((ray) => localTrace(ray, shippedCfg(K_FAR, DL_FAR_MAX))));
    const score = (K: number, MAXD: number) => {
      let newFlips = 0, diskWorse = 0, skyWorse = 0, maxDiskDelta = 0, maxSkyDelta = 0, steps = 0, n = 0;
      S.forEach((set, si) => set.rays.forEach((ray, ri) => {
        const r = localTrace(ray, shippedCfg(K, MAXD)), q = ref[si][ri], sh = ship[si][ri];
        steps += r.steps; n++; // mean steps over ALL rays: it is the cost, converged or not
        if (!scored[si][ri]) return;
        if (sh.fate !== q.fate) return; // already flipped today: nothing to be worse than
        if (r.fate !== q.fate) { newFlips++; return; }
        const d = err(r, q) - err(sh, q);
        if (r.fate === "disk") { maxDiskDelta = Math.max(maxDiskDelta, d); if (d > 0.02) diskWorse++; }
        if (r.fate === "escaped") { maxSkyDelta = Math.max(maxSkyDelta, d); if (d > HALF_PIXEL) skyWorse++; }
      }));
      return { K, MAXD, newFlips, diskWorse, skyWorse, maxDiskDelta, maxSkyDelta, meanSteps: steps / n };
    };
    const base = score(K_FAR, DL_FAR_MAX).meanSteps;
    const rows = [];
    console.log("K_FAR  DL_MAX  newFlips  diskWorse  skyWorse  maxDiskDelta(M)  maxSkyDelta(rad)  meanSteps  dSteps%");
    for (const K of [0.04, 0.08, 0.12, 0.2, 0.3]) for (const MAXD of [6, 20, 50, 150]) {
      const r = score(K, MAXD); rows.push(r);
      console.log(`${String(K).padEnd(6)} ${String(MAXD).padEnd(7)} ${String(r.newFlips).padEnd(9)} ${String(r.diskWorse).padEnd(10)} ${String(r.skyWorse).padEnd(9)} ${r.maxDiskDelta.toExponential(2).padEnd(16)} ${r.maxSkyDelta.toExponential(2).padEnd(17)} ${r.meanSteps.toFixed(1).padEnd(10)} ${(100 * (r.meanSteps / base - 1)).toFixed(1)}`);
    }
    const ok = rows.filter((r) => r.newFlips === 0 && r.diskWorse === 0 && r.skyWorse === 0)
      .sort((x, y) => x.meanSteps - y.meanSteps || x.K - y.K || x.MAXD - y.MAXD);
    expect(ok.length).toBeGreaterThan(0); // the shipped row always survives (every delta is 0)
    console.log(`SELECTED K_FAR = ${ok[0].K}, DL_FAR_MAX = ${ok[0].MAXD} (mean steps ${ok[0].meanSteps.toFixed(1)} vs ${base.toFixed(1)})`);
    console.log(`total ${((Date.now() - t0) / 60000).toFixed(1)} min`);
  }, 7_200_000);
});
