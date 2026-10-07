// Carter's separated null-geodesic equations in Mino time (spec 2026-10-07 mino integrator). Same Kerr geodesics as
// trace.ts (which stays the reference); M = 1, camera-normalised past-directed p_t = 1, xi = -p_phi, eta = Carter's
// constant. y = [t, w = 1/r, v, phi, l (affine), w', v', sigma] with ' = d/d(lambda), d(lambda) = dl / Sigma: v is the polar
// angle measured from the NEARER pole (theta = v for sigma = +1, pi - v for sigma = -1).
//
// Radial variable w = 1/r, not r: the second-order form conserves (first integral) - (potential) only up to a constant
// that integration error sets. In r that constant is fixed in the far field, where R(r) ~ r^4 ~ 1e12 at the camera, and
// it swamps R near the hole (measured: 0.6 % of r' by r = 100, a disk ray captured). In w the potential is
// R~(w) = w^4 R(1/w) = (1 + (a^2 - a xi) w^2)^2 - K w^2 (1 - 2w + a^2 w^2), of order 1 from the camera to the horizon.
// Polar variable theta itself (theta'^2 = Theta(theta) = eta + a^2 cos^2 - xi^2 cot^2, theta' = p_theta), not
// u = cos(theta): near a pole the azimuthal speed xi / sin^2 needs sin^2 to RELATIVE precision, which u cannot carry
// (a 6e-8 error in u near the pole was 1e-3 rad of phi, 3 px of sky at i = 1 deg in f64, and far worse in f32). theta
// may pass through a pole (xi = 0 exactly); minoToState folds it back (theta -> -theta, phi + pi), as the geometry does.
// Hemisphere: theta itself near the SOUTH pole has only absolute precision (f32: 2.4e-7 at theta ~ 3.14, i.e. 4e-4 of
// pi - theta at 7e-4: 2e-3 rad of phi on a GPU ray that grazed it). The equations are identical under theta -> pi - theta
// (Theta, sin^2, cos^2 are even about the equator), so a step that starts past the equator first mirrors the state
// (v -> pi - v, v' -> -v', sigma -> -sigma): v stays in [0, ~pi/2] with relative precision at both poles.
// WGSL twin: the Mino block of integrator-shared.wgsl (with camera-shared.wgsl's precise sinCosP).
import type { Fate } from "./trace";

export const MINO_TOL = 1e-5;     // DP5(4) error-norm tolerance, set by tests/sweep-mino.test.ts (SWEEP=1)
export const MINO_UFRAC = 0.25;   // step <= MINO_UFRAC x the polar-oscillation half-period bound (no hidden double crossing)
export const MINO_MAX_REJECT = 10;
/** First-integral drift tolerance relative to tol (set by tests/sweep-mino.test.ts; WGSL twin MINO_CTOL). */
export const MINO_CTOL = 1e-3;
export const MINO_LAND_ITERS = 2;  // Newton iterations of the plane landing step (minoLand)
export const MINO_LAND_BAND = 0.01; // relative widening of [rIn, rOut] before landing (minoPlane)

export interface MinoRay { a: number; xi: number; eta: number; K: number; A1: number; omMax: number }
export function minoRay(a: number, xi: number, eta: number): MinoRay {
  // A bound on the polar oscillation's angular frequency near the equator (sqrt of eta + xi^2 - a^2 plus 6 a^2)
  const om2 = Math.max(0, eta + xi * xi - a * a) + 6 * a * a;
  return { a, xi, eta, K: eta + (xi - a) * (xi - a), A1: a * a - a * xi, omMax: Math.sqrt(om2) };
}
/** R~(w) = w^4 R(1/w): (dw/dlambda)^2 along the ray. */
export function minoRw(w: number, c: MinoRay): number { const Q = 1 + c.A1 * w * w; return Q * Q - c.K * w * w * (1 - 2 * w + c.a * c.a * w * w); }
/** Theta(theta) = (dtheta/dlambda)^2 along the ray. */
export function minoTheta(th: number, c: MinoRay): number { const s = Math.sin(th), co = Math.cos(th); return c.eta + c.a * c.a * co * co - (c.xi * c.xi * co * co) / (s * s); }
/** Camera at (r0, th0) with screen beta: w' = +sqrt(R~) (r decreasing), theta' = p_theta = -beta. */
export function minoInit(r0: number, th0: number, beta: number, c: MinoRay): Float64Array {
  const w0 = 1 / r0;
  return minoHemi(new Float64Array([0, w0, th0, 0, 0, Math.sqrt(Math.max(0, minoRw(w0, c))), -beta, 1]));
}
/** Mirror a state that is past the equator into the other hemisphere's frame (identity otherwise); exact. */
// threshold = the f32 value of pi / 2 (the WGSL's 0.5 * PI), so both twins mirror the same f32 state; either frame is exact
const MINO_HALF_PI = Math.fround(Math.PI / 2);
export function minoHemi(y: Float64Array): Float64Array {
  if (y[2] <= MINO_HALF_PI) return y;
  const o = y.slice(); o[2] = Math.PI - y[2]; o[6] = -y[6]; o[7] = -y[7]; return o;
}
export function minoRhs(y: Float64Array, c: MinoRay): Float64Array {
  const w = y[1], a = c.a, xi = c.xi, a2 = a * a, w2 = w * w;
  const s = Math.sin(y[2]), co = Math.cos(y[2]), s2 = Math.max(s * s, 1e-30);
  const Q = 1 + c.A1 * w2;                 // P w^2
  const Dw = 1 - 2 * w + a2 * w2;          // Delta w^2
  return new Float64Array([
    -((1 + a2 * w2) * Q / (w2 * Dw) + a * (xi - a * s2)), // t' = -[(r^2 + a^2) P / Delta + a (xi - a sin^2)]
    y[5],                                                 // w'
    y[6],                                                 // theta'
    -(a * Q / Dw - a + xi / s2),                          // phi' = -[a P / Delta - a + xi / sin^2]
    1 / w2 + a2 * co * co,                                // l' = Sigma
    2 * c.A1 * w * Q - c.K * (w - 3 * w2 + 2 * a2 * w2 * w), // w'' = R~'(w) / 2
    co * (xi * xi / (s2 * s) - a2 * s),                   // v'' = Theta'(v) / 2 = -a^2 sin cos + xi^2 cos / sin^3 (either hemisphere)
    0,                                                    // sigma is constant within a step
  ]);
}
// Dormand-Prince 5(4) (FSAL). A: stage rows 2..7 (row 7 = the 5th-order weights); E = B - B* (error weights).
const A = [[], [1 / 5], [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9], [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
  [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656], [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]];
const E = [71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40];
/** y0, f0: the step's start state and derivative AS USED (mirrored into the nearer hemisphere if it started past the
 *  equator): dense output and the plane crossing of this step take (y0, y, f0, f1). */
export interface MinoStepOut { y0: Float64Array; y: Float64Array; f0: Float64Array; f1: Float64Array; h: number; hNext: number; attempts: number; en: number }
/** Error scale per component: t, l relative to the step's change (floor 1); w relative (= r relative); theta and phi
 *  absolute, theta also relative to sin(theta) near a pole (the azimuthal speed goes as 1 / sin^2); w' against
 *  max(|w'|, w); theta' against max(1, |theta'|). */
function errNorm(y0: Float64Array, y1: Float64Array, err: Float64Array, tol: number): number {
  const sth = Math.min(Math.abs(Math.sin(y0[2])), Math.abs(Math.sin(y1[2])));
  const sc = [Math.max(1, Math.abs(y1[0] - y0[0])), Math.max(y0[1], y1[1]), Math.min(1, Math.max(sth, 1e-4)), 1,
    Math.max(1, Math.abs(y1[4] - y0[4])), Math.max(Math.abs(y1[5]), y1[1]), Math.max(1, Math.abs(y1[6]))];
  let m = 0; for (let j = 0; j < 7; j++) m = Math.max(m, Math.abs(err[j]) / (tol * sc[j]));
  return m;
}
/** Change of the two first integrals over a step, (w'^2 - R~(w)) and (theta'^2 - Theta(theta)), with the size of their
 *  terms and the rounding noise of the change. Exactly 0 along a true ray; a step that changes them aims the ray at
 *  slightly different constants of motion, which the photon orbit amplifies exponentially (~23x per half orbit), so the
 *  error norm includes them. Computed from differences (w1 - w0, sin(theta1 - theta0) factored out), so rounding scales
 *  with the step, not with the terms. The noise term (the stored values' eps each, times the local slopes; f1 =
 *  minoRhs(y1): dR~/dw = 2 w'', dTheta/dtheta = 2 theta'') keeps a step from being rejected for rounding alone. */
export function minoDrift(y0: Float64Array, y1: Float64Array, c: MinoRay, f1: Float64Array, eps: number): [number, number, number, number, number, number] {
  const w0 = y0[1], w1 = y1[1], a2 = c.a * c.a;
  // R~(w) = 1 + c2 w^2 + c3 w^3 + c4 w^4
  const c2 = 2 * c.A1 - c.K, c3 = 2 * c.K, c4 = c.A1 * c.A1 - c.K * a2;
  const sw = w1 + w0, dR = (w1 - w0) * (c2 * sw + c3 * (w1 * w1 + w1 * w0 + w0 * w0) + c4 * sw * (w1 * w1 + w0 * w0));
  const dW = (y1[5] - y0[5]) * (y1[5] + y0[5]) - dR;
  // Theta(t1) - Theta(t0) = sin(t1 - t0) sin(t1 + t0) (xi^2 / (s0^2 s1^2) - a^2)
  const s0 = Math.sin(y0[2]), s1 = Math.sin(y1[2]), co1 = Math.cos(y1[2]);
  const dTh = Math.sin(y1[2] - y0[2]) * Math.sin(y1[2] + y0[2]) * ((c.xi * c.xi) / Math.max(s0 * s0 * s1 * s1, 1e-60) - a2);
  const dV = (y1[6] - y0[6]) * (y1[6] + y0[6]) - dTh;
  const w2 = w1 * w1, Q = 1 + c.A1 * w2;
  const scaleW = y1[5] * y1[5] + Q * Q + Math.abs(c.K * w2 * (1 - 2 * w1 + a2 * w2));
  const scaleT = y1[6] * y1[6] + Math.abs(c.eta) + a2 * co1 * co1 + (c.xi * c.xi * co1 * co1) / Math.max(s1 * s1, 1e-30);
  const noiseW = 4 * eps * (2 * Math.abs(f1[5]) * w1 + Math.abs(y1[5]) * (Math.abs(y1[5]) + Math.abs(y0[5])));
  // theta carries RELATIVE precision (eps |theta|): near a pole that is what keeps the check alive (max(|theta|, 1) here
  // overstated the noise ~1500x at theta ~ 7e-4 and switched the check off where the barrier needs it)
  const noiseT = 4 * eps * (2 * Math.abs(f1[6]) * Math.abs(y1[2]) + Math.abs(y1[6]) * (Math.abs(y1[6]) + Math.abs(y0[6])));
  return [Math.abs(dW), scaleW, noiseW, Math.abs(dV), scaleT, noiseT];
}
/** One DP5(4) attempt of size h from y (f0 = minoRhs(y)): the 5th-order result, its derivative (FSAL) and the error norm.
 *  eps: machine epsilon of the arithmetic being modelled (2^-52 here; 2^-23 when ?parity predicts the f32 GPU). */
export function minoTry(y: Float64Array, h: number, c: MinoRay, f0: Float64Array, tol = MINO_TOL, ctol = MINO_CTOL, eps = 2 ** -52): { y1: Float64Array; f1: Float64Array; en: number } {
  const k: Float64Array[] = [f0];
  for (let s = 1; s < 7; s++) {
    const ys = y.slice();
    for (let j = 0; j < 7; j++) { let acc = 0; for (let q = 0; q < s; q++) acc += A[s][q] * k[q][j]; ys[j] = y[j] + h * acc; }
    k.push(minoRhs(ys, c));
  }
  // row 7 of A is the 5th-order solution, so stage 7's argument IS y1 and k[6] = f(y1) (FSAL)
  const y1 = y.slice(); for (let j = 0; j < 7; j++) { let acc = 0; for (let q = 0; q < 6; q++) acc += A[6][q] * k[q][j]; y1[j] = y[j] + h * acc; }
  const err = new Float64Array(7); for (let j = 0; j < 7; j++) { let acc = 0; for (let q = 0; q < 7; q++) acc += E[q] * k[q][j]; err[j] = h * acc; }
  const d = minoDrift(y, y1, c, k[6], eps);
  const ec = Math.max(d[0] / Math.max(ctol * tol * d[1], d[2]), d[3] / Math.max(ctol * tol * d[4], d[5]));
  return { y1, f1: k[6], en: Math.max(errNorm(y, y1, err, tol), ec) };
}
/** One accepted DP5(4) step from y with proposed size h0 (capped at the polar-oscillation guard). Rejected attempts shrink
 *  h and retry; after MINO_MAX_REJECT the last attempt is accepted (the ray proceeds, as today's monitor does). f0In: the
 *  FSAL derivative at y from the previous step (recomputed when absent; bitwise the same value either way). */
export function minoStep(y: Float64Array, h0: number, c: MinoRay, tol = MINO_TOL, uFrac = MINO_UFRAC, f0In?: Float64Array, ctol = MINO_CTOL, eps = 2 ** -52): MinoStepOut {
  const hMax = c.omMax > 0 ? (uFrac * Math.PI) / c.omMax : Infinity;
  // mirror into the nearer hemisphere first (recomputing f there); bitwise deterministic, so cache replays match
  const yIn = y; y = minoHemi(y);
  const f0 = y === yIn && f0In ? f0In : minoRhs(y, c);
  let h = Math.min(h0, hMax);
  for (let att = 1; ; att++) {
    const { y1, f1, en } = minoTry(y, h, c, f0, tol, ctol, eps);
    // step-size factor, kept finite: 5 for a zero error, 0.2 for a non-finite one (twin of the WGSL select chain)
    const fac = en === 0 ? 5 : Number.isFinite(en) ? 0.9 * Math.pow(en, -0.2) : 0.2;
    if (en <= 1 || att > MINO_MAX_REJECT) return { y0: y, y: y1, f0, f1, h, hNext: Math.min(hMax, h * Math.min(5, Math.max(0.2, fac))), attempts: att, en };
    h *= Math.max(0.2, Math.min(0.9, fac));
  }
}
const herm = (th: number) => { const t2 = th * th, t3 = t2 * th; return [2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + th, -2 * t3 + 3 * t2, t3 - t2]; };
/** Cubic Hermite in lambda for every component (values y0, y1, derivatives f0, f1) at fraction th of the step. */
export function minoDense(y0: Float64Array, y1: Float64Array, f0: Float64Array, f1: Float64Array, h: number, th: number): Float64Array {
  if (th === 0) return y0.slice();
  const [h00, h10, h01, h11] = herm(th), o = new Float64Array(8);
  for (let j = 0; j < 7; j++) o[j] = h00 * y0[j] + h10 * h * f0[j] + h01 * y1[j] + h11 * h * f1[j];
  o[7] = y0[7];
  return o;
}
/** Fraction of the step where cos(theta) = 0 (the equatorial plane), or -1 if it does not change sign across the step. */
export function minoCrossing(y0: Float64Array, y1: Float64Array, f0: Float64Array, f1: Float64Array, h: number): number {
  const c0 = Math.cos(y0[2]), c1 = Math.cos(y1[2]);
  if (!(c0 * c1 < 0 || (c1 === 0 && c0 !== 0))) return -1;
  let lo = 0, hi = 1;
  for (let k = 0; k < 40; k++) { const m = 0.5 * (lo + hi), cm = Math.cos(minoDense(y0, y1, f0, f1, h, m)[2]); if (cm * c0 > 0) lo = m; else hi = m; }
  return 0.5 * (lo + hi);
}
/** The state ON the equatorial plane within the step (y0, f0, h) whose dense crossing is at fraction th: a DP5 step of
 *  size s from y0, s refined by safeguarded Newton on cos(theta) (two iterations; bisection inside the sign bracket when
 *  Newton leaves it, e.g. a grazing theta' ~ 0). The cubic Hermite crossing is only 4th order: its interpolation error
 *  (~1e-4 rad of phi on near-critical rays, 1e-2 M of t on far ones) exceeds the step's own; the landing step carries the
 *  step's 5th-order accuracy to the plane. Twin: WGSL minoLand. */
export function minoLand(y0: Float64Array, f0: Float64Array, h: number, th: number, c: MinoRay): Float64Array {
  const c0 = Math.cos(y0[2]);
  let lo = 0, hi = h, s = th * h;
  for (let k = 0; k < MINO_LAND_ITERS; k++) {
    const { y1: d, f1: fd } = minoTry(y0, s, c, f0), g = Math.cos(d[2]);
    if (g * c0 > 0) lo = s; else hi = s;
    const sn = s - g / (-Math.sin(d[2]) * fd[2]);
    s = sn > lo && sn < hi ? sn : 0.5 * (lo + hi);
  }
  return minoTry(y0, s, c, f0).y1;
}
/** Disk test of one step: the landed plane state when the step crosses the plane with the dense crossing radius inside
 *  [rIn, rOut] widened by MINO_LAND_BAND (landing moves r by ~1e-5 relative at most), else undefined. The caller tests
 *  the landed radius against [rIn, rOut]. */
export function minoPlane(st: MinoStepOut, c: MinoRay, rIn: number, rOut: number): Float64Array | undefined {
  const th = minoCrossing(st.y0, st.y, st.f0, st.f1, st.h);
  if (th < 0) return undefined;
  const r = 1 / minoDense(st.y0, st.y, st.f0, st.f1, st.h, th)[1];
  if (!(r >= rIn * (1 - MINO_LAND_BAND) && r <= rOut * (1 + MINO_LAND_BAND))) return undefined;
  return minoLand(st.y0, st.f0, st.h, th, c);
}
/** Today's State (t, r, theta, phi, p_t, p_r, p_theta, p_phi): p_r = r'/Delta = -w' / (1 - 2w + a^2 w^2), p_theta =
 *  theta'. theta is folded into [0, pi] (through a pole: theta -> -theta or 2 pi - theta, phi + pi, p_theta -> -p_theta;
 *  the same point and direction). */
export function minoToState(y: Float64Array, c: MinoRay): Float64Array {
  const w = y[1], Dw = 1 - 2 * w + c.a * c.a * w * w, south = y[7] < 0;
  let th = (south ? Math.PI - y[2] : y[2]) % (2 * Math.PI), ph = y[3], pth = south ? -y[6] : y[6];
  if (th < 0) th += 2 * Math.PI;
  if (th > Math.PI) { th = 2 * Math.PI - th; ph += Math.PI; pth = -pth; }
  return new Float64Array([y[0], 1 / w, th, ph, 1, -y[5] / Dw, pth, -c.xi]);
}
export interface MinoTraceResult { fate: Fate; steps: number; attempts: number; rHit?: number; phiHit?: number; tHit?: number; s: Float64Array; drift: number }
/** The render loop's termination order: disk crossing (cos theta = 0 root of the dense output), capture, escape, budget.
 *  drift is the largest |w'^2 - R~(w)| seen (the first integral, of order 1 along the whole ray). */
export function minoTrace(al: number, be: number, a: number, inclDeg: number, o: { rIn: number; rOut: number; rObs: number; maxSteps?: number; tol?: number; uFrac?: number; ctol?: number }): MinoTraceResult {
  const incl = (inclDeg * Math.PI) / 180, xi = -al * Math.sin(incl), ci = Math.cos(incl), si = Math.sin(incl);
  const c = minoRay(a, xi, be * be + (xi * xi * ci * ci) / Math.max(si * si, 1e-8) - a * a * ci * ci);
  const rh = 1 + Math.sqrt(Math.max(0, 1 - a * a)), maxSteps = o.maxSteps ?? 4800;
  let y = minoInit(o.rObs, incl, be, c), h = 50 / (o.rObs * o.rObs), f: Float64Array | undefined, attempts = 0, drift = 0;
  for (let step = 1; step <= maxSteps; step++) {
    const st = minoStep(y, h, c, o.tol ?? MINO_TOL, o.uFrac ?? MINO_UFRAC, f, o.ctol ?? MINO_CTOL);
    attempts += st.attempts;
    drift = Math.max(drift, Math.abs(st.y[5] * st.y[5] - minoRw(st.y[1], c)));
    const d = minoPlane(st, c, o.rIn, o.rOut);
    if (d) {
      const rHit = 1 / d[1];
      if (rHit >= o.rIn && rHit <= o.rOut) { const s = minoToState(d, c); return { fate: "disk", steps: step, attempts, rHit, phiHit: s[3], tHit: d[0], s, drift }; }
    }
    y = st.y; h = st.hNext; f = st.f1;
    // tests in w (a long outgoing step may carry w past 0, where 1/w would read as captured)
    if (y[1] >= 1 / (rh * 1.005)) return { fate: "captured", steps: step, attempts, s: minoToState(y, c), drift };
    if (y[1] < 1 / (o.rObs * 1.2)) return { fate: "escaped", steps: step, attempts, s: minoToState(y, c), drift };
  }
  return { fate: "budget", steps: maxSteps, attempts, s: minoToState(y, c), drift };
}
