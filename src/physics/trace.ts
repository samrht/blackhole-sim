import { metricUpper } from "./kerr";
import { rk4 } from "./geodesic";

/**
 * CPU twin of the render loop's INTEGRATION logic (src/render/integrator-shared.wgsl is the sole
 * WGSL copy; ?parity compares stepGeodesic between the two). Geometry only: no emission, jet or
 * colour. Keep the control flow of stepGeodesic/traceRay structurally identical to the shader.
 */

/** Null-constraint tolerance, RELATIVE to the magnitude of the Hamiltonian's terms (see
 *  hquadScaled). Chosen from tests/sweep-htol.test.ts (SWEEP=1): 1e-3, 1e-4, 1e-5 all survived
 *  (untrusted = 0 on both views, near-axis rHit within 0.02 M of the converged reference); 1e-2
 *  failed accuracy and 1e-6 fell below the f32 floor; picked the largest (cheapest) survivor.
 *  Twin constant in integrator-shared.wgsl. */
export const H_TOL = 1e-3;
/** Maximum number of step halvings before a step is accepted as untrusted. Chosen from the same
 *  sweep: at H_TOL = 1e-3, maxRetry = 4, 8 and 12 all behaved identically (the cap was never
 *  approached); shipped one size up from the smallest surviving value (4) for margin.
 *  Twin constant in integrator-shared.wgsl. */
export const MAX_RETRY = 8;
/** Far-field exemption: beyond rOut * 1.5 (the far branch of stepSize) the monitor is OFF. The f32
 *  finite-difference force is pure noise at r ~ 1e3 (ulp 6e-5 vs the FD half-step 1e-4; measured
 *  |dH|/scale 2.4e-3 on the GPU vs 2e-11 in f64 for the same step), so halving on dH there costs
 *  steps for nothing while curvature ~M/r^3 is negligible. Still a NaN guard: abs(NaN) <= x is
 *  false. Twin constant in integrator-shared.wgsl. */
export const H_TOL_FAR = 1e30;

/** Baseline step length: fine in the strong-field/disk region, long strides through the near-flat
 *  far field. Twin of stepSize() in integrator-shared.wgsl. */
export function stepSize(r: number, rh: number, rOut: number): number {
  if (r > rOut * 1.5) return Math.min(6, Math.max(0.6, 0.04 * r));
  return Math.min(0.5, Math.max(0.002, 0.02 * (r - rh)));
}

/** g^{mu nu} p_mu p_nu (= 2H, exactly 0 for a null geodesic) together with the sum of |terms|.
 *  The constraint drift is judged relative to that sum because near the capture margin
 *  g^tt ~ -A/(Sigma Delta) reaches ~1e2 and H becomes a cancellation of large terms; an absolute
 *  tolerance would sit at the f32 noise floor there. With p_t = 1 and |g^tt| >= 1 outside the
 *  horizon the scale is always >= 1, so this is never looser than an absolute tolerance. */
export function hquadScaled(r: number, th: number, a: number, pt: number, pr: number, pth: number, pphi: number): [number, number] {
  const g = metricUpper(r, th, a);
  const t0 = g.tt * pt * pt, t1 = 2 * g.tphi * pt * pphi, t2 = g.rr * pr * pr, t3 = g.thth * pth * pth, t4 = g.phph * pphi * pphi;
  return [t0 + t1 + t2 + t3 + t4, Math.abs(t0) + Math.abs(t1) + Math.abs(t2) + Math.abs(t3) + Math.abs(t4)];
}

/** Exact analytic continuation through the Boyer-Lindquist polar axis: theta -> -theta (or
 *  2pi - theta at the south pole), phi -> phi + pi, p_theta -> -p_theta. p_phi is unchanged.
 *  Reachable only when a step carries theta past 0 or pi; identity otherwise. */
export function reflectAxis(s: Float64Array): Float64Array {
  const th = s[2];
  if (th >= 0 && th <= Math.PI) return s;
  const o = s.slice();
  o[2] = th < 0 ? -th : 2 * Math.PI - th;
  o[3] = s[3] + Math.PI;
  o[6] = -s[6];
  return o;
}

export interface StepOut { s: Float64Array; ok: boolean; retries: number; dl: number; }

/** One constraint-monitored RK4 step. The step is accepted when the per-step change in the null
 *  constraint is within tolerance; otherwise dl is halved and the step redone, up to maxRetry
 *  halvings. The final attempt is returned either way, with ok = false if it still failed, so the
 *  caller can route the ray to the conserved-quantity classifier instead of trusting it.
 *  NaN drift compares false against the tolerance, so a diverged step is never accepted. */
export function stepGeodesic(s: Float64Array, a: number, dl0: number, hTol = H_TOL, maxRetry = MAX_RETRY): StepOut {
  const [h0] = hquadScaled(s[1], s[2], a, s[4], s[5], s[6], s[7]);
  let dl = dl0;
  for (let k = 0; k < maxRetry; k++) {
    const sN = rk4(s, a, dl);
    const [h1, scale] = hquadScaled(sN[1], sN[2], a, sN[4], sN[5], sN[6], sN[7]);
    if (Math.abs(h1 - h0) <= hTol * scale) return { s: reflectAxis(sN), ok: true, retries: k, dl };
    dl *= 0.5;
  }
  const sN = rk4(s, a, dl);
  const [h1, scale] = hquadScaled(sN[1], sN[2], a, sN[4], sN[5], sN[6], sN[7]);
  const ok = Math.abs(h1 - h0) <= hTol * scale;
  return { s: reflectAxis(sN), ok, retries: maxRetry, dl };
}

export type Fate = "disk" | "captured" | "escaped" | "budget" | "untrusted";
export interface TraceOpts { rIn: number; rOut: number; rObs: number; maxSteps?: number; hTol?: number; maxRetry?: number; }
export interface TraceResult { fate: Fate; s: Float64Array; steps: number; retries: number; thMin: number; rHit?: number; phiHit?: number; }

/** Backward ray trace with the render loop's termination order: disk crossing, then capture, then
 *  escape. thMin is the smallest sampled angular distance from either pole along the trajectory. */
export function traceRay(s0: Float64Array, a: number, o: TraceOpts): TraceResult {
  const rh = 1 + Math.sqrt(Math.max(0, 1 - a * a));
  const maxSteps = o.maxSteps ?? 20000;
  let s = s0, retries = 0, thMin = Math.min(s0[2], Math.PI - s0[2]);
  for (let step = 1; step <= maxSteps; step++) {
    const far = s[1] > o.rOut * 1.5; // same threshold as the far branch of stepSize
    const out = stepGeodesic(s, a, stepSize(s[1], rh, o.rOut), far ? H_TOL_FAR : o.hTol, o.maxRetry);
    retries += out.retries;
    if (!out.ok) return { fate: "untrusted", s: out.s, steps: step, retries, thMin };
    const sN = out.s;
    const f0 = s[2] - Math.PI / 2, f1 = sN[2] - Math.PI / 2;
    if (f0 * f1 < 0) {
      const frac = f0 / (f0 - f1);
      const rHit = s[1] + frac * (sN[1] - s[1]);
      if (rHit >= o.rIn && rHit <= o.rOut) {
        return { fate: "disk", s: sN, steps: step, retries, thMin, rHit, phiHit: s[3] + frac * (sN[3] - s[3]) };
      }
    }
    s = sN;
    thMin = Math.min(thMin, s[2], Math.PI - s[2]);
    if (s[1] <= rh * 1.005) return { fate: "captured", s, steps: step, retries, thMin };
    if (s[1] > o.rObs * 1.2) return { fate: "escaped", s, steps: step, retries, thMin };
  }
  return { fate: "budget", s, steps: maxSteps, retries, thMin };
}
