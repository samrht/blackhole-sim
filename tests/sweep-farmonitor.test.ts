import { describe, it, expect } from "vitest";
import { screenToState } from "../src/physics/camera";
import { metricUpper } from "../src/physics/kerr";
import { photonOrbit } from "../src/physics/orbits";
import { stepGeodesic, traceRay, F_AXIS, F_PHI, DL_FAR_MIN, H_TOL, H_TOL_FAR, MAX_RETRY, K_FAR, DL_FAR_MAX } from "../src/physics/trace";

/**
 * Measurement, not a test: the table the 2026-10-01 step controller was chosen from (far field
 * monitored at H_TOL_FAR, far stride K_FAR / DL_FAR_MAX, azimuthal cap F_PHI, angular caps in the
 * near field too). Skipped unless SWEEP=1; HOLDOUT=1 swaps in five views the choice never saw.
 *
 * The loop is a LOCAL COPY of traceRay with every controller knob parameterised; the test first
 * asserts that the shipped configuration reproduces traceRay exactly on every ray.
 *
 * Reference: REF1 (monitored everywhere at 1e-7, far base 0.01 / 1.5, tight caps in both fields),
 * checked against REF2 (all tightened again). A ray where they disagree in fate, by > 5e-3 M in disk
 * radius or by > half a pixel in sky direction is UNCONVERGED and not scored.
 *
 * Scoring per ray, against REF1:
 *   - disk: |rHit - ref|; "bad" if > 0.02 M (half a pixel at the interactive scale).
 *   - sky: the error as an ON-SCREEN displacement. REF1 traced at +1 px in alpha and in beta gives
 *     the local lensing Jacobian J (rad per pixel); the direction error is solved against J in least
 *     squares -> pixels. An angle threshold is meaningless near the critical curve, where the sky
 *     map stretches by orders of magnitude (the old rule flagged rays whose error was 0.02 px).
 *     "bad" if > 0.5 px.
 *   - WORSE than OLD (the pre-2026-10-01 controller): a fate flip OLD did not have; disk error >
 *     max(OLD's, FARREF's) + 0.02 M, where FARREF is OLD's near field with REF1's far field (on two
 *     near-critical rays of the default view OLD's error was lower than a converged far field's:
 *     two near-field errors cancelling by luck, which a far-field change cannot be blamed for
 *     losing); sky error > OLD's + 0.5 px.
 * Rule: the shipped configuration has 0 new flips and 0 worse rays. Cost = mean steps + retries.
 *
 * Sets: V1 a=0 i=8 and V2 a=0.9 i=72 (32x32 over +-14 M), A (axis band i=8, 121 rays crossing the
 * axis in the far field), B (axis column i=1, 65), N (88 rays crossing the axis INSIDE r = 60:
 * i = 8/30/72, alpha 0..0.3 beside the axis column). Holdout: a=0.5 i=30, a=0.99 i=85, a=0.9 i=3,
 * a=0.7 i=55 (24x24) and a=0.9 i=72 at 31x31 (offset grid).
 *
 * Measured 2026-10-01 (f64; 0.7 min, holdout 1.0 min). References: every set 0 unconverged.
 *   config                            newFlips  diskBad>0.02M  skyBad>0.5px  worse  maxDisk(M)  maxSky(px)  cost
 *   OLD 0.04/6, far unmonitored       0         77             3             0      1.29e+1     1.005       347.9
 *   + far monitor 1e-5                0         30             3             0      1.59e+0     1.005       348.1
 *   + far stride 0.08/50              1         36             5             18     3.62e-1     1.618       217.1
 *   + F_PHI 0.1                       1         30             4             13     3.62e-1     1.618       217.8
 *   + near-field caps = SHIPPED       0         0              0             0      7.18e-3     0.151       220.4
 *   shipped but far unmonitored       0         0              0             0      7.18e-3     0.151       220.4
 *   shipped but far stride 0.08/20    0         0              0             0      7.09e-3     0.071       240.0
 *   shipped but far stride 0.1/50     0         0              0             0      7.10e-3     0.206       210.2
 *   shipped but far stride 0.12/50    0         0              0             0      7.32e-3     0.206       212.4
 *   shipped but far stride 0.08/150   0         0              0             0      7.12e-3     0.321       219.1
 * Holdout (HOLDOUT=1):
 *   OLD 0.04/6, far unmonitored       0         4              0             0      2.71e-2     0.190       353.1
 *   + near-field caps = SHIPPED       0         0              0             0      2.10e-3     0.002       217.2
 *   (every other row: 0 flips, 0 worse, maxSky <= 0.194 px)
 * Reading: the old controller's error was near-field near-axis error (set N, the V2 near-critical
 * sky rays, the axis line), which the far-stride sweep of 2026-09-23 kept tripping over; the
 * angular caps in the near branch remove it, after which the far stride and the far monitor barely
 * matter (unmonitored = monitored: the monitor is a safety net). 0.08 / 50 shipped: -37 % cost,
 * half the worst sky ray of 0.1 / 50.
 */
const SWEEP = !!process.env.SWEEP, HOLDOUT = !!process.env.HOLDOUT;
const ROBS = 1000, ROUT = 40, FOV = 14, MAXSTEPS = 4800, REF_MAXSTEPS = 400000, GRID_N = 32;
const PX = (2 * FOV) / 720;                       // one pixel in screen units (M) at 720 px tall
const HALF_PIXEL = 0.5 * PX / ROBS;               // in rad, for the reference convergence check

type Ray = { s0: Float64Array; a: number; rIn: number; al: number; be: number; incl: number };
type Res = { fate: string; steps: number; retries: number; rHit?: number; dir?: number[] };
/** fPhi 0 = no azimuthal cap; nearCaps = angular caps also bound the near branch. */
type Cfg = { K: number; MAXD: number; fAxis: number; fPhi: number; dlFarMin: number; nearCaps: boolean; hNear: number; hFar: number; maxRetry: number; maxSteps: number };

const OLD: Cfg = { K: 0.04, MAXD: 6, fAxis: 0.1, fPhi: 0, dlFarMin: 0.05, nearCaps: false, hNear: 1e-3, hFar: 1e30, maxRetry: 8, maxSteps: MAXSTEPS };
const SHIP: Cfg = { K: K_FAR, MAXD: DL_FAR_MAX, fAxis: F_AXIS, fPhi: F_PHI, dlFarMin: DL_FAR_MIN, nearCaps: true, hNear: H_TOL, hFar: H_TOL_FAR, maxRetry: MAX_RETRY, maxSteps: MAXSTEPS };
const REF1: Cfg = { K: 0.01, MAXD: 1.5, fAxis: 0.02, fPhi: 0.02, dlFarMin: 0.01, nearCaps: true, hNear: 1e-7, hFar: 1e-7, maxRetry: 24, maxSteps: REF_MAXSTEPS };
const REF2: Cfg = { K: 0.005, MAXD: 0.75, fAxis: 0.01, fPhi: 0.01, dlFarMin: 0.005, nearCaps: true, hNear: 1e-8, hFar: 1e-8, maxRetry: 28, maxSteps: REF_MAXSTEPS };
const FARREF: Cfg = { ...REF1, hNear: OLD.hNear, maxRetry: OLD.maxRetry, nearCaps: false, fAxis: 0.02 };

function skyDirCPU(s: Float64Array, a: number): number[] {
  const r = s[1], th = s[2], ph = s[3];
  const g = metricUpper(r, th, a);
  const dr = g.rr * s[5], dth = g.thth * s[6], dph = g.tphi * s[4] + g.phph * s[7];
  const st = Math.sin(th), ct = Math.cos(th), sp = Math.sin(ph), cp = Math.cos(ph);
  const v = [dr * st * cp + r * ct * cp * dth - r * st * sp * dph, dr * st * sp + r * ct * sp * dth + r * st * cp * dph, dr * ct - r * st * dth];
  const n = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / n, v[1] / n, v[2] / n];
}

/** Same structure and operation order as angularCap/stepSize in trace.ts. */
function capLocal(s: Float64Array, dl0: number, floor: number, c: Cfg): number {
  const r = s[1];
  let dl = dl0;
  if (s[6] !== 0) dl = Math.min(dl, Math.max(floor, c.fAxis * Math.min(s[2], Math.PI - s[2]) * r * r / Math.abs(s[6])));
  if (c.fPhi > 0 && s[7] !== 0) { const sn = Math.sin(s[2]); dl = Math.min(dl, Math.max(floor, c.fPhi * r * r * sn * sn / Math.abs(s[7]))); }
  return dl;
}
function stepSizeLocal(s: Float64Array, rh: number, c: Cfg): number {
  const r = s[1];
  if (r > ROUT * 1.5) return capLocal(s, Math.min(c.MAXD, Math.max(0.6, c.K * r)), c.dlFarMin, c);
  const dn = Math.min(0.5, Math.max(0.002, 0.02 * (r - rh)));
  return c.nearCaps ? capLocal(s, dn, 0.002, c) : dn;
}

function localTrace(ray: Ray, c: Cfg): Res {
  const { a, rIn } = ray, rh = 1 + Math.sqrt(Math.max(0, 1 - a * a));
  let s = ray.s0, retries = 0;
  for (let step = 1; step <= c.maxSteps; step++) {
    const far = s[1] > ROUT * 1.5;
    const o = stepGeodesic(s, a, stepSizeLocal(s, rh, c), far ? c.hFar : c.hNear, c.maxRetry);
    retries += o.retries; const sN = o.s;
    const f0 = s[2] - Math.PI / 2, f1 = sN[2] - Math.PI / 2;
    if (f0 * f1 < 0 && Math.abs(sN[2] - s[2]) < 0.5) {
      const frac = f0 / (f0 - f1), rHit = s[1] + frac * (sN[1] - s[1]);
      if (rHit >= rIn && rHit <= ROUT) return { fate: "disk", steps: step, retries, rHit };
    }
    s = sN;
    if (s[1] <= rh * 1.005) return { fate: "captured", steps: step, retries };
    if (s[1] > ROBS * 1.2) return { fate: "escaped", steps: step, retries, dir: skyDirCPU(s, a) };
  }
  return { fate: "budget", steps: c.maxSteps, retries };
}

const lin = (lo: number, hi: number, n: number) => Array.from({ length: n }, (_, k) => lo + ((hi - lo) * k) / (n - 1));
const ray = (al: number, be: number, a: number, inclDeg: number, rIn: number): Ray => {
  const incl = (inclDeg * Math.PI) / 180;
  return { s0: screenToState(al, be, a, incl, ROBS), a, rIn, al, be, incl };
};
const grid = (a: number, inclDeg: number, n: number) => {
  const rays: Ray[] = [];
  for (let iy = 0; iy < n; iy++) for (let ix = 0; ix < n; ix++)
    rays.push(ray((((ix + 0.5) / n) * 2 - 1) * FOV, -(((iy + 0.5) / n) * 2 - 1) * FOV, a, inclDeg, photonOrbit(a, true)));
  return rays;
};
const band = (inclDeg: number, betas: number[], alphas: number[], rIn: number) => betas.flatMap((b) => alphas.map((al) => ray(al, b, 0, inclDeg, rIn)));
function sets(): { name: string; rays: Ray[] }[] {
  if (HOLDOUT) return [["H1 a=0.5 i=30", 0.5, 30, 24], ["H2 a=0.99 i=85", 0.99, 85, 24], ["H3 a=0.9 i=3", 0.9, 3, 24], ["H4 a=0.7 i=55", 0.7, 55, 24], ["H5 a=0.9 i=72", 0.9, 72, 31]]
    .map(([name, a, i, n]) => ({ name: name as string, rays: grid(a as number, i as number, n as number) }));
  const NA = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3];
  return [
    { name: "V1 a=0 i=8", rays: grid(0, 8, GRID_N) },
    { name: "V2 a=0.9 i=72", rays: grid(0.9, 72, GRID_N) },
    { name: "A axis band i=8", rays: band(8, lin(9, 14, 11), lin(0, 0.8, 11), 6) },
    { name: "B axis column i=1", rays: band(1, lin(2, 14, 13), lin(0, 1, 5), 6) },
    { name: "N near-field axis", rays: [...band(8, [3, 4, 5, 6, 7, 8], NA, 3), ...band(30, [5, 8, 12, 16], NA, 3), ...band(72, [6, 8, 12], NA, 3)] },
  ];
}

const angle = (u: number[], v: number[]) => 2 * Math.asin(Math.min(1, Math.hypot(u[0] - v[0], u[1] - v[1], u[2] - v[2]) / 2));

describe.skipIf(!SWEEP)("far-field monitor / stride / caps sweep (SWEEP=1)", () => {
  it("prints the table; the shipped controller has no new flips and no worse rays", () => {
    const t0 = Date.now();
    const S = sets();
    // 1) the local loop IS the shipped loop
    for (const set of S) for (const r of set.rays) {
      const loc = localTrace(r, SHIP), ship = traceRay(r.s0, r.a, { rIn: r.rIn, rOut: ROUT, rObs: ROBS, maxSteps: MAXSTEPS });
      expect(loc.fate).toBe(ship.fate);
      expect(loc.steps).toBe(ship.steps);
      if (loc.fate === "disk") expect(loc.rHit).toBe(ship.rHit);
    }
    // 2) reference and its convergence
    const ref = S.map((set) => set.rays.map((r) => localTrace(r, REF1)));
    const ref2 = S.map((set) => set.rays.map((r) => localTrace(r, REF2)));
    const scored = S.map((set, si) => set.rays.map((_, ri) => {
      const p = ref[si][ri], q = ref2[si][ri];
      if (p.fate !== q.fate || p.fate === "budget") return false;
      if (p.fate === "disk") return Math.abs(p.rHit! - q.rHit!) <= 5e-3;
      if (p.fate === "escaped") return angle(p.dir!, q.dir!) <= HALF_PIXEL;
      return true;
    }));
    S.forEach((set, si) => {
      const un = scored[si].filter((x) => !x).length;
      console.log(`reference ${set.name}: ${set.rays.length} rays, unconverged ${un}`);
      expect(un).toBeLessThan(0.25 * set.rays.length);
    });
    // 3) the lensing Jacobian of every scored escaped ray (REF1 at +1 px in alpha and beta)
    const jac = new Map<string, number[][]>();
    S.forEach((set, si) => set.rays.forEach((r, ri) => {
      const q = ref[si][ri]; if (!scored[si][ri] || q.fate !== "escaped") return;
      const d1 = localTrace(ray(r.al + PX, r.be, r.a, (r.incl * 180) / Math.PI, r.rIn), REF1);
      const d2 = localTrace(ray(r.al, r.be + PX, r.a, (r.incl * 180) / Math.PI, r.rIn), REF1);
      if (d1.fate === "escaped" && d2.fate === "escaped") jac.set(`${si}#${ri}`, [q.dir!.map((v, k) => d1.dir![k] - v), q.dir!.map((v, k) => d2.dir![k] - v)]);
    }));
    const dot = (u: number[], v: number[]) => u.reduce((t, x, k) => t + x * v[k], 0);
    const pxErr = (key: string, d: number[], q: number[]) => {
      const J = jac.get(key); if (!J) return NaN; // fate edge one pixel away: not scored in pixels
      const e = d.map((v, k) => v - q[k]);
      const a11 = dot(J[0], J[0]), a12 = dot(J[0], J[1]), a22 = dot(J[1], J[1]), b1 = dot(J[0], e), b2 = dot(J[1], e);
      const det = a11 * a22 - a12 * a12;
      return Math.hypot((a22 * b1 - a12 * b2) / det, (a11 * b2 - a12 * b1) / det);
    };
    const old = S.map((set) => set.rays.map((r) => localTrace(r, OLD)));
    const farRef = S.map((set) => set.rays.map((r) => localTrace(r, FARREF)));
    const score = (c: Cfg) => {
      let newFlips = 0, diskBad = 0, skyBad = 0, worse = 0, cost = 0, n = 0, maxDisk = 0, maxPx = 0;
      S.forEach((set, si) => set.rays.forEach((r, ri) => {
        const x = localTrace(r, c), q = ref[si][ri], o = old[si][ri], f = farRef[si][ri];
        cost += x.steps + x.retries; n++;
        if (!scored[si][ri]) return;
        if (x.fate !== q.fate) { if (o.fate === q.fate) newFlips++; return; }
        if (x.fate === "disk") {
          const e = Math.abs(x.rHit! - q.rHit!); maxDisk = Math.max(maxDisk, e); if (e > 0.02) diskBad++;
          const eo = o.fate === "disk" ? Math.abs(o.rHit! - q.rHit!) : Infinity, ef = f.fate === "disk" ? Math.abs(f.rHit! - q.rHit!) : 0;
          if (e > Math.max(eo, ef) + 0.02) worse++;
        }
        if (x.fate === "escaped") {
          const key = `${si}#${ri}`, e = pxErr(key, x.dir!, q.dir!); if (isNaN(e)) return;
          maxPx = Math.max(maxPx, e); if (e > 0.5) skyBad++;
          const eo = o.fate === "escaped" ? pxErr(key, o.dir!, q.dir!) : Infinity;
          if (e > eo + 0.5) worse++;
        }
      }));
      return { newFlips, diskBad, skyBad, worse, maxDisk, maxPx, cost: cost / n };
    };
    const rows: [string, Cfg][] = [
      ["OLD 0.04/6, far unmonitored", OLD],
      ["+ far monitor 1e-5", { ...OLD, hFar: 1e-5 }],
      ["+ far stride 0.08/50", { ...OLD, hFar: 1e-5, K: 0.08, MAXD: 50 }],
      ["+ F_PHI 0.1", { ...OLD, hFar: 1e-5, K: 0.08, MAXD: 50, fPhi: 0.1 }],
      ["+ near-field caps = SHIPPED", SHIP],
      ["shipped but far unmonitored", { ...SHIP, hFar: 1e30 }],
      ["shipped but far stride 0.08/20", { ...SHIP, MAXD: 20 }],
      ["shipped but far stride 0.1/50", { ...SHIP, K: 0.1 }],
      ["shipped but far stride 0.12/50", { ...SHIP, K: 0.12 }],
      ["shipped but far stride 0.08/150", { ...SHIP, MAXD: 150 }],
    ];
    console.log("config                            newFlips  diskBad>0.02M  skyBad>0.5px  worse  maxDisk(M)  maxSky(px)  cost(steps+retries)");
    let shipped = score(SHIP);
    for (const [name, c] of rows) {
      const r = c === SHIP ? shipped : score(c);
      console.log(`${name.padEnd(34)}${String(r.newFlips).padEnd(10)}${String(r.diskBad).padEnd(15)}${String(r.skyBad).padEnd(14)}${String(r.worse).padEnd(7)}${r.maxDisk.toExponential(2).padEnd(12)}${r.maxPx.toFixed(3).padEnd(12)}${r.cost.toFixed(1)}`);
    }
    expect(shipped.newFlips).toBe(0);
    expect(shipped.worse).toBe(0);
    shipped = score(SHIP);
    console.log(`total ${((Date.now() - t0) / 60000).toFixed(1)} min`);
  }, 7_200_000);
});
