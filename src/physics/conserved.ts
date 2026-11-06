// Conserved quantities of a Mino state, measured from the evolved state alone (the ?drift diagnostic). WGSL twin:
// src/render/drift.wgsl. y = [tau, w = 1/r, v, phi, ell, w', v', sigma] (trace-mino.ts): v is theta from the nearer pole,
// and p_t = 1, p_phi = -xi, p_r = -w' / (1 - 2w + a^2 w^2), p_theta = +-v'.
//
// xi and eta are parameters of Carter's equations, not state, so "Q at the end" must be rebuilt from (v, v'): Q = v'^2 +
// cos^2 v (xi^2 / sin^2 v - a^2). Being second order in Mino time, the integrator does not hold Q or H fixed by
// construction; the state can leave both. With W = w'^2 - R~(w) and T = Q - eta,
//     H = [ (r^4 / Delta) W + T ] / (2 Sigma),
// so near the horizon H carries the radial residual amplified by r^4 / (2 Sigma Delta): a Boyer-Lindquist factor, not
// integration error growing. Rounding of the f32 state and of H's own cancelling terms goes through the same factor.
import { sinCosP } from "../render/sincos";
import { minoRay, minoRw, type MinoRay } from "./trace-mino";

type Round = (v: number) => number;
const id: Round = (v) => v;
// f32: the shader's sinCosP, op by op. f64: the true functions (sinCosP is a single-precision polynomial, ~1e-8 off in f64).
const trig = (x: number, fr: Round): [number, number] => (fr === id ? [Math.sin(x), Math.cos(x)] : sinCosP(x, true));

/** Carter's Q from (v, v') and its size (sum of |terms|, the scale of its rounding). fr = Math.fround evaluates it as f32. */
export function minoQ(y: ArrayLike<number>, c: MinoRay, fr: Round = id): { q: number; qScale: number } {
  const [s, co] = trig(fr(y[2]), fr);
  const s2 = Math.max(fr(s * s), 1e-30), c2 = fr(co * co), vp = fr(y[6]);
  const t0 = fr(vp * vp), t1 = fr(c2 * fr(fr(c.xi * c.xi) / s2)), t2 = fr(c2 * fr(c.a * c.a));
  return { q: fr(fr(t0 + t1) - t2), qScale: fr(fr(t0 + t1) + t2) };
}

/** H = 1/2 g^{mu nu} p_mu p_nu (0 for light), the size of its cancelling terms, and the radial residual w'^2 - R~(w). */
export function minoHQ(y: ArrayLike<number>, c: MinoRay, fr: Round = id): { h: number; hScale: number; wRes: number } {
  const a = c.a, a2 = fr(a * a), w = fr(y[1]), wp = fr(y[5]), vp = fr(y[6]), xi = c.xi;
  const [s, co] = trig(fr(y[2]), fr);
  const s2 = Math.max(fr(s * s), 1e-30), r = fr(1 / w), r2 = fr(r * r), ra = fr(r2 + a2);
  const Sig = fr(r2 + fr(a2 * fr(co * co))), D = fr(fr(r2 - fr(2 * r)) + a2), SD = fr(Sig * D);
  const A = fr(fr(ra * ra) - fr(fr(a2 * D) * s2));
  const pr = fr(-wp / fr(fr(1 - fr(2 * w)) + fr(a2 * fr(w * w))));
  const t0 = fr(-A / SD), t1 = fr(fr(fr(fr(4 * a) * r) * xi) / SD), t2 = fr(fr(fr(D / Sig) * pr) * pr), t3 = fr(fr(vp * vp) / Sig);
  const t4 = fr(fr(fr(D - fr(a2 * s2)) / fr(SD * s2)) * fr(xi * xi));
  const sum = fr(fr(fr(fr(t0 + t1) + t2) + t3) + t4);
  const abs = fr(fr(fr(fr(Math.abs(t0) + Math.abs(t1)) + Math.abs(t2)) + Math.abs(t3)) + Math.abs(t4));
  return { h: fr(0.5 * sum), hScale: fr(0.5 * abs), wRes: fr(fr(wp * wp) - minoRw(w, c)) };
}

/** The diagnostic's number: |Q_end - Q_start| / |Q_start| + |H_end|. Infinity at Q_start = 0, where relative drift has no meaning. */
export function driftMetric(q0: number, q1: number, h1: number): number {
  return Math.abs(q1 - q0) / Math.abs(q0) + Math.abs(h1);
}

/** The drift of a stored f32 trajectory with f32 rounding of the MEASURE removed: the same numbers (end state, start state, the ray's own f32 xi and eta), evaluated in f64. */
export function preciseDrift(end: { w: number; wp: number; v: number; vp: number }, start: { v: number; vp: number }, ray: { a: number; xi: number; eta: number; r0: number }): { q0: number; q1: number; h1: number; wRes: number; total: number } {
  const c = minoRay(ray.a, ray.xi, ray.eta, ray.r0), y = Float64Array.from([0, end.w, end.v, 0, 0, end.wp, end.vp, 1]);
  const q0 = minoQ(Float64Array.from([0, 0, start.v, 0, 0, 0, start.vp, 1]), c).q, q1 = minoQ(y, c).q, H = minoHQ(y, c);
  return { q0, q1, h1: H.h, wRes: H.wRes, total: driftMetric(q0, q1, H.h) };
}
