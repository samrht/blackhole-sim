// Carter's separated null-geodesic equations in Mino time (spec 2026-10-07 mino integrator). Same Kerr geodesics as
// trace.ts (which stays the reference); M = 1, camera-normalised past-directed p_t = 1, xi = -p_phi, eta = Carter's
// constant. y = [t, w = 1/r, u = cos(theta), phi, l (affine), w', u'] with ' = d/d(lambda), d(lambda) = dl / Sigma.
//
// Radial variable w = 1/r, not r: the second-order form conserves (first integral) - (potential) only up to a constant
// that integration error sets. In r that constant is fixed in the far field, where R(r) ~ r^4 ~ 1e12 at the camera, and
// it swamps R near the hole (measured: 0.6 % of r' by r = 100, a disk ray captured). In w the potential is
// R~(w) = w^4 R(1/w) = (1 + (a^2 - a xi) w^2)^2 - K w^2 (1 - 2w + a^2 w^2), of order 1 from the camera to the horizon.
// WGSL twin: the Mino block of integrator-shared.wgsl.
import type { Fate } from "./trace";

export const MINO_TOL = 1e-5;     // DP5(4) error-norm tolerance, set by tests/sweep-mino.test.ts (SWEEP=1)
export const MINO_UFRAC = 0.25;   // step <= MINO_UFRAC x the u-oscillation half-period bound (no hidden double crossing)
export const MINO_MAX_REJECT = 10;

export interface MinoRay { a: number; xi: number; eta: number; K: number; A1: number; omMax: number }
export function minoRay(a: number, xi: number, eta: number): MinoRay {
  // |d(u'')/du| <= (eta + xi^2 - a^2) + 6 a^2: a bound on the u-oscillation's angular frequency
  const om2 = Math.max(0, eta + xi * xi - a * a) + 6 * a * a;
  return { a, xi, eta, K: eta + (xi - a) * (xi - a), A1: a * a - a * xi, omMax: Math.sqrt(om2) };
}
/** R~(w) = w^4 R(1/w): (dw/dlambda)^2 along the ray. */
export function minoRw(w: number, c: MinoRay): number { const Q = 1 + c.A1 * w * w; return Q * Q - c.K * w * w * (1 - 2 * w + c.a * c.a * w * w); }
/** U(u) = (du/dlambda)^2 along the ray. */
export function minoU(u: number, c: MinoRay): number { return c.eta * (1 - u * u) + c.a * c.a * u * u * (1 - u * u) - c.xi * c.xi * u * u; }
/** Camera at (r0, th0) with screen beta: w' = +sqrt(R~) (r decreasing), u' = sin(th0) beta (= -sin th p_theta, p_theta = -beta). */
export function minoInit(r0: number, th0: number, beta: number, c: MinoRay): Float64Array {
  const w0 = 1 / r0;
  return new Float64Array([0, w0, Math.cos(th0), 0, 0, Math.sqrt(Math.max(0, minoRw(w0, c))), Math.sin(th0) * beta]);
}
export function minoRhs(y: Float64Array, c: MinoRay): Float64Array {
  const w = y[1], u = y[2], a = c.a, xi = c.xi, a2 = a * a, w2 = w * w;
  const Q = 1 + c.A1 * w2;                 // P w^2
  const Dw = 1 - 2 * w + a2 * w2;          // Delta w^2
  const s2 = Math.max((1 - u) * (1 + u), 1e-30); // sin^2 theta, (1-u)(1+u) keeps its precision near the poles (f32 twin)
  return new Float64Array([
    -((1 + a2 * w2) * Q / (w2 * Dw) + a * (xi - a * s2)), // t' = -[(r^2 + a^2) P / Delta + a (xi - a sin^2)]
    y[5],                                                 // w'
    y[6],                                                 // u'
    -(a * Q / Dw - a + xi / s2),                          // phi' = -[a P / Delta - a + xi / sin^2]
    1 / w2 + a2 * u * u,                                  // l' = Sigma
    2 * c.A1 * w * Q - c.K * (w - 3 * w2 + 2 * a2 * w2 * w), // w'' = R~'(w) / 2
    u * (a2 - c.eta - xi * xi) - 2 * a2 * u * u * u,      // u'' = U'(u) / 2
  ]);
}
// Dormand-Prince 5(4) (FSAL). A: stage rows 2..7 (row 7 = the 5th-order weights); E = B - B* (error weights).
const A = [[], [1 / 5], [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9], [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
  [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656], [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]];
const E = [71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40];
/** pole: for an exact xi = 0 ray, the fraction of the step where it passed through the pole (u' changed sign), else -1;
 *  y's phi then already includes the pole's pi (minoDense applies it after that fraction only). */
export interface MinoStepOut { y: Float64Array; f0: Float64Array; f1: Float64Array; h: number; hNext: number; attempts: number; en: number; pole: number }
/** Error scale per component: t, phi, l relative to the step's change (floor 1); w relative (= r relative); w' against
 *  max(|w'|, w); u absolute; u' against max(1, |u'|). */
function errNorm(y0: Float64Array, y1: Float64Array, err: Float64Array, tol: number): number {
  const sc = [Math.max(1, Math.abs(y1[0] - y0[0])), Math.max(y0[1], y1[1]), 1, Math.max(1, Math.abs(y1[3] - y0[3])),
    Math.max(1, Math.abs(y1[4] - y0[4])), Math.max(Math.abs(y1[5]), y1[1]), Math.max(1, Math.abs(y1[6]))];
  let m = 0; for (let j = 0; j < 7; j++) m = Math.max(m, Math.abs(err[j]) / (tol * sc[j]));
  return m;
}
/** First-integral residuals (w'^2 - R~(w), u'^2 - U(u)) and the sizes of their terms. Exactly 0 along a true ray; a
 *  step that changes them aims the ray at slightly different constants of motion, which the photon orbit amplifies
 *  exponentially (~23x per half orbit), so their per-step change is part of the error norm (MINO_CTOL x tol). */
export function minoConstraint(y: Float64Array, c: MinoRay): [number, number, number, number] {
  const w = y[1], u = y[2], w2 = w * w, u2 = u * u, s2 = (1 - u) * (1 + u), Q = 1 + c.A1 * w2, kw = c.K * w2 * (1 - 2 * w + c.a * c.a * w2);
  const Uterms = [c.eta * s2, c.a * c.a * u2 * s2, -c.xi * c.xi * u2];
  return [y[5] * y[5] - (Q * Q - kw), y[5] * y[5] + Q * Q + Math.abs(kw), y[6] * y[6] - (Uterms[0] + Uterms[1] + Uterms[2]),
    y[6] * y[6] + Math.abs(Uterms[0]) + Math.abs(Uterms[1]) + Math.abs(Uterms[2])];
}
export const MINO_CTOL = Number((globalThis as unknown as { process?: { env: Record<string, string> } }).process?.env.MINO_CTOL_EXP ?? 0.01);
/** One DP5(4) attempt of size h from y (f0 = minoRhs(y)): the 5th-order result, its derivative (FSAL) and the error norm. */
export function minoTry(y: Float64Array, h: number, c: MinoRay, f0: Float64Array, tol = MINO_TOL): { y1: Float64Array; f1: Float64Array; en: number } {
  const k: Float64Array[] = [f0];
  for (let s = 1; s < 7; s++) {
    const ys = y.slice();
    for (let j = 0; j < 7; j++) { let acc = 0; for (let q = 0; q < s; q++) acc += A[s][q] * k[q][j]; ys[j] = y[j] + h * acc; }
    k.push(minoRhs(ys, c));
  }
  // row 7 of A is the 5th-order solution, so stage 7's argument IS y1 and k[6] = f(y1) (FSAL)
  const y1 = y.slice(); for (let j = 0; j < 7; j++) { let acc = 0; for (let q = 0; q < 6; q++) acc += A[6][q] * k[q][j]; y1[j] = y[j] + h * acc; }
  const err = new Float64Array(7); for (let j = 0; j < 7; j++) { let acc = 0; for (let q = 0; q < 7; q++) acc += E[q] * k[q][j]; err[j] = h * acc; }
  const c0 = minoConstraint(y, c), c1 = minoConstraint(y1, c);
  const ec = Math.max(Math.abs(c1[0] - c0[0]) / (MINO_CTOL * tol * c1[1]), Math.abs(c1[2] - c0[2]) / (MINO_CTOL * tol * Math.max(c1[3], 1e-30)));
  return { y1, f1: k[6], en: Math.max(errNorm(y, y1, err, tol), ec) };
}
/** One accepted DP5(4) step from y with proposed size h0 (capped at the u-oscillation guard). Rejected attempts shrink h
 *  and retry; after MINO_MAX_REJECT the last attempt is accepted (the ray proceeds, as today's monitor does). f0In: the
 *  FSAL derivative at y from the previous step (recomputed when absent; bitwise the same value either way). */
export function minoStep(y: Float64Array, h0: number, c: MinoRay, tol = MINO_TOL, uFrac = MINO_UFRAC, f0In?: Float64Array): MinoStepOut {
  const hMax = c.omMax > 0 ? (uFrac * Math.PI) / c.omMax : Infinity;
  const f0 = f0In ?? minoRhs(y, c);
  let h = Math.min(h0, hMax);
  for (let att = 1; ; att++) {
    const { y1, f1, en } = minoTry(y, h, c, f0, tol);
    // step-size factor, kept finite: 5 for a zero error, 0.2 for a non-finite one (twin of the WGSL select chain)
    const fac = en === 0 ? 5 : Number.isFinite(en) ? 0.9 * Math.pow(en, -0.2) : 0.2;
    if (en <= 1 || att > MINO_MAX_REJECT) {
      // xi = 0: U(u) = (1 - u^2)(eta + a^2 u^2) vanishes only at the poles, so a sign change of u' is a passage through the
      // pole, where the ray continues on the far side: phi + pi (trace.ts reflectAxis does the same). For xi != 0 the
      // swing comes out of phi' = ... + xi / sin^2 itself.
      let pole = -1;
      if (c.xi === 0 && y[6] * y1[6] < 0) {
        let lo = 0, hi = 1;
        for (let k = 0; k < 40; k++) { const m = 0.5 * (lo + hi); if (minoDense(y, y1, f0, f1, h, m)[6] * y[6] > 0) lo = m; else hi = m; }
        pole = 0.5 * (lo + hi); y1[3] += Math.PI;
      }
      return { y: y1, f0, f1, h, hNext: Math.min(hMax, h * Math.min(5, Math.max(0.2, fac))), attempts: att, en, pole };
    }
    h *= Math.max(0.2, Math.min(0.9, fac));
  }
}
const herm = (th: number) => { const t2 = th * th, t3 = t2 * th; return [2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + th, -2 * t3 + 3 * t2, t3 - t2]; };
/** Cubic Hermite in lambda for every component (values y0, y1, derivatives f0, f1) at fraction th of the step. pole: the
 *  step's pole passage (MinoStepOut.pole); y1's phi carries its pi, which applies after that fraction only. */
export function minoDense(y0: Float64Array, y1: Float64Array, f0: Float64Array, f1: Float64Array, h: number, th: number, pole = -1): Float64Array {
  if (th === 0) return y0.slice();
  const [h00, h10, h01, h11] = herm(th), o = new Float64Array(7);
  for (let j = 0; j < 7; j++) o[j] = h00 * y0[j] + h10 * h * f0[j] + h01 * y1[j] + h11 * h * f1[j];
  if (pole >= 0) o[3] += th > pole ? 0 : -h01 * Math.PI; // undo y1's pi before the passage (h01 = 1 at th = 1)
  if (pole >= 0 && th > pole) o[3] += Math.PI * (1 - h01);
  return o;
}
/** Fraction of the step where u = 0 (the equatorial plane), or -1 if u does not change sign across the step. */
export function minoCrossing(y0: Float64Array, y1: Float64Array, f0: Float64Array, f1: Float64Array, h: number): number {
  if (!(y0[2] * y1[2] < 0 || (y1[2] === 0 && y0[2] !== 0))) return -1;
  let lo = 0, hi = 1; const ulo = y0[2];
  for (let k = 0; k < 40; k++) { const m = 0.5 * (lo + hi), um = minoDense(y0, y1, f0, f1, h, m)[2]; if (um * ulo > 0) lo = m; else hi = m; }
  return 0.5 * (lo + hi);
}
/** Today's State (t, r, theta, phi, p_t, p_r, p_theta, p_phi): p_r = r'/Delta = -w' / (1 - 2w + a^2 w^2). Near the
 *  axis p_theta comes from Theta, never 0/0. */
export function minoToState(y: Float64Array, c: MinoRay): Float64Array {
  const w = y[1], u = Math.max(-1, Math.min(1, y[2])), s2 = (1 - u) * (1 + u), Dw = 1 - 2 * w + c.a * c.a * w * w;
  let pth: number;
  if (s2 > 1e-8) pth = -y[6] / Math.sqrt(s2);
  else { const Th = c.eta + c.a * c.a * u * u - (c.xi * c.xi * u * u) / Math.max(s2, 1e-12); pth = -Math.sign(y[6]) * Math.sqrt(Math.max(Th, 0)); }
  // theta = atan2(sin, cos): acos(u) loses precision near the axis in f32 (the WGSL twin), so both twins use this
  return new Float64Array([y[0], 1 / w, Math.atan2(Math.sqrt(s2), u), y[3], 1, -y[5] / Dw, pth, -c.xi]);
}
export interface MinoTraceResult { fate: Fate; steps: number; attempts: number; rHit?: number; phiHit?: number; tHit?: number; s: Float64Array; drift: number }
/** The render loop's termination order: disk crossing (u = 0 root of the dense output), capture, escape, budget. drift is
 *  the largest |w'^2 - R~(w)| seen (the first integral, of order 1 along the whole ray). */
export function minoTrace(al: number, be: number, a: number, inclDeg: number, o: { rIn: number; rOut: number; rObs: number; maxSteps?: number; tol?: number; uFrac?: number }): MinoTraceResult {
  const incl = (inclDeg * Math.PI) / 180, xi = -al * Math.sin(incl), ci = Math.cos(incl), si = Math.sin(incl);
  const c = minoRay(a, xi, be * be + (xi * xi * ci * ci) / Math.max(si * si, 1e-8) - a * a * ci * ci);
  const rh = 1 + Math.sqrt(Math.max(0, 1 - a * a)), maxSteps = o.maxSteps ?? 4800;
  let y = minoInit(o.rObs, incl, be, c), h = 50 / (o.rObs * o.rObs), f: Float64Array | undefined, attempts = 0, drift = 0;
  for (let step = 1; step <= maxSteps; step++) {
    const st = minoStep(y, h, c, o.tol ?? MINO_TOL, o.uFrac ?? MINO_UFRAC, f);
    attempts += st.attempts;
    drift = Math.max(drift, Math.abs(st.y[5] * st.y[5] - minoRw(st.y[1], c)));
    const th = minoCrossing(y, st.y, st.f0, st.f1, st.h);
    if (th >= 0) {
      const d = minoDense(y, st.y, st.f0, st.f1, st.h, th, st.pole), rHit = 1 / d[1];
      if (rHit >= o.rIn && rHit <= o.rOut) return { fate: "disk", steps: step, attempts, rHit, phiHit: d[3], tHit: d[0], s: minoToState(d, c), drift };
    }
    y = st.y; h = st.hNext; f = st.f1;
    // tests in w (a long outgoing step may carry w past 0, where 1/w would read as captured)
    if (y[1] >= 1 / (rh * 1.005)) return { fate: "captured", steps: step, attempts, s: minoToState(y, c), drift };
    if (y[1] < 1 / (o.rObs * 1.2)) return { fate: "escaped", steps: step, attempts, s: minoToState(y, c), drift };
  }
  return { fate: "budget", steps: maxSteps, attempts, s: minoToState(y, c), drift };
}
