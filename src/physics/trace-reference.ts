// Converged reference traces for accuracy gates (spec 2026-10-07 mino integrator §5): today's Hamiltonian equations
// (trace.ts stepGeodesic) with every step and monitor tightened, so the result is converged to ~1e-6 relative. Shared by
// tests/sweep-mino.test.ts, scripts/build-accuracy-ref.ts and the ?accuracy route. Geometry and the 1.3 mm flow
// intensity only; no new physics (flowShift / flowCoeffs are hot-flow.ts's).
import { metricUpper } from "./kerr";
import { rk4 } from "./geodesic";
import { stepGeodesic, F_AXIS, F_PHI, DL_FAR_MIN, H_TOL, H_TOL_FAR, MAX_RETRY, K_FAR, DL_FAR_MAX } from "./trace";
import { flowShift, flowCoeffs, HOTFLOW } from "./hot-flow";

export const REF_ROBS = 1000, REF_ROUT = 40;
export type RefCfg = { K: number; MAXD: number; fAxis: number; fPhi: number; dlFarMin: number; nearCaps: boolean; hNear: number; hFar: number; maxRetry: number; maxSteps: number; dlCap?: number; scale?: number; landEsc?: boolean };
/** Today's shipped controller (trace.ts traceRay's), parameterised. */
export const SHIP: RefCfg = { K: K_FAR, MAXD: DL_FAR_MAX, fAxis: F_AXIS, fPhi: F_PHI, dlFarMin: DL_FAR_MIN, nearCaps: true, hNear: H_TOL, hFar: H_TOL_FAR, maxRetry: MAX_RETRY, maxSteps: 4800 };
// The far-monitor sweep's REF1/REF2 with every base step also scaled down (0.1x / 0.05x; RK4 error ~ h^4) and the
// monitors at 1e-9 / 1e-10: converged to ~1e-6 relative, a tenth of the scoring floors. landEsc: the sky direction is read ON
// the escape sphere r = 1.2 r_obs (the renderer's cutoff; the direction still turns by ~M b / r^2 beyond it, 0.02 px over
// one 50 M shipped far step), so references and the Mino renderer (minoSphere) compare the same quantity.
export const REF1: RefCfg = { K: 0.01, MAXD: 1.5, fAxis: 0.02, fPhi: 0.02, dlFarMin: 0.01, nearCaps: true, hNear: 1e-9, hFar: 1e-9, maxRetry: 24, maxSteps: 400000, scale: 0.1, landEsc: true };
export const REF2: RefCfg = { K: 0.005, MAXD: 0.75, fAxis: 0.01, fPhi: 0.01, dlFarMin: 0.005, nearCaps: true, hNear: 1e-10, hFar: 1e-10, maxRetry: 28, maxSteps: 400000, scale: 0.05, landEsc: true };

export function skyDirCPU(s: Float64Array, a: number): number[] {
  const r = s[1], th = s[2], ph = s[3];
  const g = metricUpper(r, th, a);
  const dr = g.rr * s[5], dth = g.thth * s[6], dph = g.tphi * s[4] + g.phph * s[7];
  const st = Math.sin(th), ct = Math.cos(th), sp = Math.sin(ph), cp = Math.cos(ph);
  const v = [dr * st * cp + r * ct * cp * dth - r * st * sp * dph, dr * st * sp + r * ct * sp * dth + r * st * cp * dph, dr * ct - r * st * dth];
  const n = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / n, v[1] / n, v[2] / n];
}
function capLocal(s: Float64Array, dl0: number, floor: number, c: RefCfg): number {
  const r = s[1];
  let dl = dl0;
  if (s[6] !== 0) dl = Math.min(dl, Math.max(floor, c.fAxis * Math.min(s[2], Math.PI - s[2]) * r * r / Math.abs(s[6])));
  if (c.fPhi > 0 && s[7] !== 0) { const sn = Math.sin(s[2]); dl = Math.min(dl, Math.max(floor, c.fPhi * r * r * sn * sn / Math.abs(s[7]))); }
  return dl;
}
/** trace.ts stepSize with the knobs exposed (equal to it at SHIP). */
export function refStepSize(s: Float64Array, rh: number, c: RefCfg): number {
  const r = s[1], k = c.scale ?? 1;
  if (r > REF_ROUT * 1.5) return capLocal(s, k * Math.min(c.MAXD, Math.max(0.6, c.K * r)), k * c.dlFarMin, c);
  const dn = k * Math.min(0.5, Math.max(0.002, 0.02 * (r - rh)));
  const d = c.nearCaps ? capLocal(s, dn, k * 0.002, c) : dn;
  return c.dlCap ? Math.min(d, c.dlCap) : d;
}
const cart = (r: number, th: number, ph: number) => [r * Math.sin(th) * Math.cos(ph), r * Math.sin(th) * Math.sin(ph), r * Math.cos(th)];
/** Closest approach to the origin of the chord p0 -> p1 (raytrace.wgsl chordMisses). */
export const chordMin = (p0: number[], p1: number[]) => {
  const d = p1.map((v, i) => v - p0[i]), dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  const tc = dd > 0 ? Math.max(0, Math.min(1, -(p0[0] * d[0] + p0[1] * d[1] + p0[2] * d[2]) / dd)) : 0;
  return Math.hypot(p0[0] + d[0] * tc, p0[1] + d[1] * tc, p0[2] + d[2] * tc);
};
export type FlowObj = { n0: number; rg: number };
/** One slab of the hot flow at a sample (the renderer's flowStep, scalar): [dI, dTau] before exp(-tau). */
export function flowSlab(r: number, th: number, p: number[], a: number, rh: number, dl: number, fl: FlowObj): [number, number] {
  if (r >= HOTFLOW.rMax || r <= rh * 1.01) return [0, 0];
  const D = flowShift(Float64Array.from([0, r, th, 0, p[0], p[1], p[2], p[3]]), a);
  if (D === null || !(D > 1e-6)) return [0, 0];
  const [j, al] = flowCoeffs(r, th, D * HOTFLOW.nu, fl.n0); if (!(j > 0)) return [0, 0];
  const ds = fl.rg * D * dl;
  return [(j / D ** 3) * ds, al * ds];
}
/** Slab transfer (the renderer's): I += dI (1 - e^-dTau)/dTau e^-tau; tau += dTau. acc = [I, tau]. */
export function flowAccum(acc: number[], dI: number, dT: number) {
  if (dI === 0 && dT === 0) return;
  const fac = dT < 1e-4 ? 1 - 0.5 * dT : (1 - Math.exp(-dT)) / dT;
  acc[0] += dI * fac * Math.exp(-acc[1]); acc[1] += dT;
}
export type RefRes = { fate: string; steps: number; retries: number; rHit?: number; phiHit?: number; tHit?: number; dir?: number[]; I?: number };
/** Hamiltonian trace in the render loop's termination order (the far-monitor sweep's local loop), plus phiHit / tHit; for
 *  flow rays (fl set: no disk termination, the mm exit at r > 50 outgoing) the intensity sampled on each step's chord,
 *  n per step = ceil(dl / flowDl) when flowDl > 0, else the GPU's flowStep rule (OLD replica). */
export function refTrace(s0: Float64Array, a: number, rIn: number, c: RefCfg, fl?: FlowObj, flowDl = 0): RefRes {
  const rh = 1 + Math.sqrt(Math.max(0, 1 - a * a)), I = [0, 0];
  let s = s0, retries = 0;
  for (let step = 1; step <= c.maxSteps; step++) {
    const far = s[1] > REF_ROUT * 1.5;
    const o = stepGeodesic(s, a, refStepSize(s, rh, c), far ? c.hFar : c.hNear, c.maxRetry);
    retries += o.retries; const sN = o.s;
    if (fl) {
      const p0 = cart(s[1], s[2], s[3]), p1 = cart(sN[1], sN[2], sN[3]);
      if (chordMin(p0, p1) <= HOTFLOW.rMax) { // the GPU's chordMisses(p0, dvec, HF_RMAX) test
        const n = flowDl > 0 ? Math.max(1, Math.ceil(o.dl / flowDl)) : Math.min(32, Math.max(1, Math.ceil(o.dl / (0.25 * Math.max(1, Math.min(s[1], sN[1]) / 8)))));
        for (let k = 0; k < n; k++) {
          const f = k / n; let r: number, th: number;
          if (k === 0) { r = s[1]; th = s[2]; }
          else { const q = p0.map((v, i) => v + (p1[i] - v) * f); r = Math.hypot(q[0], q[1], q[2]); th = Math.acos(Math.max(-1, Math.min(1, q[2] / r))); }
          const p = [4, 5, 6, 7].map((i) => s[i] + (sN[i] - s[i]) * f);
          const [dI, dT] = flowSlab(r, th, p, a, rh, o.dl / n, fl); flowAccum(I, dI, dT);
        }
      }
    } else {
      const f0 = s[2] - Math.PI / 2, f1 = sN[2] - Math.PI / 2;
      if (f0 * f1 < 0 && Math.abs(sN[2] - s[2]) < 0.5) {
        const frac = f0 / (f0 - f1), rHit = s[1] + frac * (sN[1] - s[1]);
        if (rHit >= rIn && rHit <= REF_ROUT) return { fate: "disk", steps: step, retries, rHit, phiHit: s[3] + frac * (sN[3] - s[3]), tHit: s[0] + frac * (sN[0] - s[0]) };
      }
    }
    const rPrev = s[1], sPrev = s; s = sN;
    if (s[1] <= rh * 1.005) return { fate: "captured", steps: step, retries, I: I[0] };
    if (fl && s[1] > HOTFLOW.rMax && s[1] > rPrev) return { fate: "escaped", steps: step, retries, I: I[0] };
    if (s[1] > REF_ROBS * 1.2) {
      if (c.landEsc && !fl) { // bisect a single RK4 step from the last state inside onto r = 1.2 r_obs (steps here are <= 1.5 M)
        let lo = 0, hi = o.dl;
        for (let k = 0; k < 60; k++) { const m = 0.5 * (lo + hi); if (rk4(sPrev, a, m)[1] < REF_ROBS * 1.2) lo = m; else hi = m; }
        s = rk4(sPrev, a, 0.5 * (lo + hi));
      }
      return { fate: "escaped", steps: step, retries, dir: skyDirCPU(s, a), I: I[0] };
    }
  }
  return { fate: "budget", steps: c.maxSteps, retries, I: I[0] };
}
/** REF1 checked against REF2 (flow rays: steps capped at 0.05 / 0.025 M, intensity sampled every 0.02 / 0.01 M). converged:
 *  same fate (not budget) and agreement to a tenth of the scoring floors. */
export function convergedRef(s0: Float64Array, a: number, rIn: number, fl?: FlowObj): { ref: RefRes; converged: boolean } {
  const ref = refTrace(s0, a, rIn, fl ? { ...REF1, dlCap: 0.05 } : REF1, fl, fl ? 0.02 : 0);
  const q = refTrace(s0, a, rIn, fl ? { ...REF2, dlCap: 0.025 } : REF2, fl, fl ? 0.01 : 0);
  let converged = ref.fate === q.fate && ref.fate !== "budget";
  if (converged && fl) converged = Math.abs(ref.I! - q.I!) <= 1e-6 * Math.max(Math.abs(q.I!), 1e-300) || (ref.I === 0 && q.I === 0);
  else if (converged && ref.fate === "disk") converged = Math.abs(ref.rHit! - q.rHit!) <= 1e-6 * q.rHit! && wrapAngle(ref.phiHit! - q.phiHit!) <= 1e-6 && Math.abs(ref.tHit! - q.tHit!) <= 1e-6 * Math.abs(q.tHit!);
  else if (converged && ref.fate === "escaped") converged = angleBetween(ref.dir!, q.dir!) <= 0.001 * (2 * 14 / 720) / REF_ROBS; // 0.001 px
  return { ref, converged };
}
export const wrapAngle = (x: number) => Math.abs(Math.atan2(Math.sin(x), Math.cos(x)));
export const angleBetween = (u: number[], v: number[]) => 2 * Math.asin(Math.min(1, Math.hypot(u[0] - v[0], u[1] - v[1], u[2] - v[2]) / 2));
/** On-screen pixel error of direction d against reference q, through the local lensing Jacobian J (rad per px; rows =
 *  d(dir)/d(alpha), d(dir)/d(beta)), least squares. NaN without J. */
export function pxError(J: number[][] | undefined, d: number[], q: number[]): number {
  if (!J) return NaN;
  const dot = (u: number[], v: number[]) => u.reduce((t, x, k) => t + x * v[k], 0);
  const e = d.map((v, k) => v - q[k]);
  const a11 = dot(J[0], J[0]), a12 = dot(J[0], J[1]), a22 = dot(J[1], J[1]), b1 = dot(J[0], e), b2 = dot(J[1], e);
  const det = a11 * a22 - a12 * a12;
  return Math.hypot((a22 * b1 - a12 * b2) / det, (a11 * b2 - a12 * b1) / det);
}

// ---- Flow intensity along a recorded path (plan Task 2 ruling: the renderer's 0.25 M left-sample rule has its own ~1e-3
// quadrature error, so the path's accuracy is measured with a fine second-order rule, and the rule-level result against
// the exact integral with the rule's own resolution as the floor). -------------------------------------------------
/** One straight sub-segment of a path: affine l0..l1, Cartesian ends, covariant momenta at the ends. */
export type PathSeg = { l0: number; l1: number; x0: number[]; x1: number[]; p0: number[]; p1: number[] };
const polar = (x: number[]) => { const r = Math.hypot(x[0], x[1], x[2]); return [r, Math.acos(Math.max(-1, Math.min(1, x[2] / r)))]; };
/** Midpoint rule over every sub-segment (second order; the path must be fine, <= 0.005 M). */
export function flowMid(path: PathSeg[], a: number, fl: FlowObj): number {
  const rh = 1 + Math.sqrt(Math.max(0, 1 - a * a)), acc = [0, 0];
  for (const g of path) {
    const xm = g.x0.map((v, i) => 0.5 * (v + g.x1[i])), pm = g.p0.map((v, i) => 0.5 * (v + g.p1[i])), [r, th] = polar(xm);
    const [dI, dT] = flowSlab(r, th, pm, a, rh, g.l1 - g.l0, fl); flowAccum(acc, dI, dT);
  }
  return acc[0];
}
/** The renderer's rule along a path: left samples every 0.25 max(1, r/8) M of affine length, weight = that spacing. */
export function flowRuleAlong(path: PathSeg[], a: number, fl: FlowObj): number {
  const rh = 1 + Math.sqrt(Math.max(0, 1 - a * a)), acc = [0, 0];
  let lNext = path.length ? path[0].l0 : 0;
  for (const g of path) {
    if (lNext < g.l0) lNext = g.l0; // a gap (outside the flow): restart at the segment
    while (lNext < g.l1) {
      const f = (lNext - g.l0) / Math.max(g.l1 - g.l0, 1e-300);
      const x = g.x0.map((v, i) => v + (g.x1[i] - v) * f), p = g.p0.map((v, i) => v + (g.p1[i] - v) * f), [r, th] = polar(x);
      const sp = 0.25 * Math.max(1, r / 8);
      const [dI, dT] = flowSlab(r, th, p, a, rh, sp, fl); flowAccum(acc, dI, dT);
      lNext += sp;
    }
  }
  return acc[0];
}
/** Hamiltonian trace of a flow ray (the mm loop's terminations) recording its path as sub-segments of <= pathDl inside
 *  r < 51 (the chord of each step: fine steps make it the curve; OLD's coarse steps make it what the GPU samples). */
export function refFlowPath(s0: Float64Array, a: number, c: RefCfg, pathDl: number): { fate: string; path: PathSeg[] } {
  const rh = 1 + Math.sqrt(Math.max(0, 1 - a * a)), path: PathSeg[] = [];
  let s = s0, l = 0;
  for (let step = 1; step <= c.maxSteps; step++) {
    const far = s[1] > REF_ROUT * 1.5;
    const o = stepGeodesic(s, a, refStepSize(s, rh, c), far ? c.hFar : c.hNear, c.maxRetry), sN = o.s;
    const x0 = cart(s[1], s[2], s[3]), x1 = cart(sN[1], sN[2], sN[3]);
    if (chordMin(x0, x1) <= HOTFLOW.rMax + 1) {
      const m = Math.max(1, Math.ceil(o.dl / pathDl));
      for (let k = 0; k < m; k++) {
        const f0 = k / m, f1 = (k + 1) / m, lerp = (u: number[], v: number[], f: number) => u.map((q, i) => q + (v[i] - q) * f);
        path.push({ l0: l + o.dl * f0, l1: l + o.dl * f1, x0: lerp(x0, x1, f0), x1: lerp(x0, x1, f1),
          p0: lerp([s[4], s[5], s[6], s[7]], [sN[4], sN[5], sN[6], sN[7]], f0), p1: lerp([s[4], s[5], s[6], s[7]], [sN[4], sN[5], sN[6], sN[7]], f1) });
      }
    }
    l += o.dl; const rPrev = s[1]; s = sN;
    if (s[1] <= rh * 1.005) return { fate: "captured", path };
    if (s[1] > HOTFLOW.rMax && s[1] > rPrev) return { fate: "escaped", path };
  }
  return { fate: "budget", path };
}
/** Converged exact flow intensity of a ray (REF1 steps capped 0.01 M, path 0.005 M; checked against REF2 at half both),
 *  and the renderer rule's resolution q = |rule along the exact path - exact|. */
export function flowReference(s0: Float64Array, a: number, fl: FlowObj): { fate: string; I: number; q: number; converged: boolean } {
  const p1 = refFlowPath(s0, a, { ...REF1, dlCap: 0.01 }, 0.005), p2 = refFlowPath(s0, a, { ...REF2, dlCap: 0.005 }, 0.0025);
  const I = flowMid(p1.path, a, fl), I2 = flowMid(p2.path, a, fl), q = Math.abs(flowRuleAlong(p1.path, a, fl) - I);
  const converged = p1.fate === p2.fate && p1.fate !== "budget" && (Math.abs(I - I2) <= 1e-6 * Math.max(Math.abs(I2), 1e-300) || (I === 0 && I2 === 0));
  return { fate: p1.fate, I, q, converged };
}
