import { metricUpper } from "./kerr";
import { rk4 } from "./geodesic";

/**
 * CPU twin of the render loop's INTEGRATION logic (src/render/integrator-shared.wgsl is the sole
 * WGSL copy; ?parity compares stepGeodesic between the two). Geometry only: no emission, jet or
 * colour. Keep the control flow of stepGeodesic/traceRay structurally identical to the shader.
 */

/** Null-constraint tolerance, RELATIVE to the magnitude of the Hamiltonian's terms (see
 *  hquadScaled). Chosen from tests/sweep-htol.test.ts (SWEEP=1): 1e-3, 1e-4, 1e-5 all survived
 *  (exhausted = 0 on both views, near-axis rHit within 0.02 M of the converged reference); 1e-2
 *  failed accuracy and 1e-6 fell below the f32 floor; picked the largest (cheapest) survivor.
 *  Confirmed under exact forces (2026-09-23): same table shape, same selection. Twin constant in integrator-shared.wgsl. */
export const H_TOL = 1e-3;
/** Maximum number of step halvings. On exhaustion the smallest-step attempt is accepted anyway and
 *  the ray proceeds (ok = false is informational; traceRay counts it in `exhausted`). Chosen from
 *  the same sweep: at H_TOL = 1e-3, maxRetry = 4, 8 and 12 all behaved identically (the cap was
 *  never approached); shipped one size up from the smallest surviving value (4) for margin.
 *  Twin constant in integrator-shared.wgsl. */
export const MAX_RETRY = 8;
/** Far-field monitor tolerance, beyond rOut * 1.5 (the far branch of stepSize). Was 1e30 (monitor
 *  OFF): under finite-difference forces the f32 force was pure noise at r ~ 1e3 (|dH|/scale 2.4e-3
 *  on the GPU vs 2e-11 in f64 for the same step). Under exact forces the GPU's |dH|/scale is 2.7e-8
 *  on the parity "far" step, so the far field is now monitored as a safety net, at 1e-5 (370x that
 *  floor). With the angular caps in both fields it never fires on any ray of
 *  tests/sweep-farmonitor.test.ts (SWEEP=1, identical to unmonitored); without the near-field
 *  caps it was what held the near-axis far-field passage (77 -> 27 disk rays off). Twin constant
 *  in integrator-shared.wgsl. */
export const H_TOL_FAR = 1e-5;
/** Far-field angular step cap. The far field was unmonitored when this was added, so nothing but
 *  the step controller bounded a stride there -- and a ray aimed at screen beta crosses the axis at
 *  r ~ beta / sin(i), i.e. INSIDE the exempt zone whenever beta > 1.5 * rOut * sin(i) (beta > 8.3 M
 *  at i = 8 deg). A dl = 3 stride at theta ~ 3e-3 carried theta to -9.2 and p_theta to -6e4
 *  (H ~ 1e10): the ray escaped and painted a dark streak beside the axis column above the shadow.
 *  The cap limits the far stride to F_AXIS of the affine distance to the axis at the current
 *  angular rate (theta_d * Sigma / |p_theta|, since dtheta/dl = p_theta / Sigma), floored at
 *  DL_FAR_MIN. Chosen from tests/sweep-axiscap.test.ts (SWEEP=1) against a converged reference
 *  monitored everywhere; see the table in that file (confirmed under exact forces, 2026-09-23). Twin constants in integrator-shared.wgsl. */
export const F_AXIS = 0.1;
export const DL_FAR_MIN = 0.05;

/** Far-field azimuthal step cap: dl <= F_PHI * r^2 sin^2(th) / |p_phi| (floored at DL_FAR_MIN), i.e.
 *  at most F_PHI rad of azimuth per step (dphi/dl ~ p_phi / (r^2 sin^2 th) out there; frame dragging
 *  ~ 2a/r^3 is negligible). A ray with p_phi != 0 swings through ~pi of azimuth where it turns in
 *  theta beside the axis -- there p_theta ~ 0, so the F_AXIS cap is inactive -- and at longer far
 *  strides that swing was under-resolved (sky rays 1-1.4 px off at i = 1 deg). angularCap applies
 *  it in the near field too. Twin constant in integrator-shared.wgsl. */
export const F_PHI = 0.1;

/** Far-field stride: dl = K_FAR * r clamped to [0.6, DL_FAR_MAX] beyond rOut * 1.5, before the
 *  angular caps. 0.08 / 50 (was 0.04 / 6), chosen by tests/sweep-farmonitor.test.ts (SWEEP=1)
 *  against a converged reference: no fate flips and no ray worse than the old controller (disk
 *  +0.02 M; sky +0.5 px of ON-SCREEN displacement through the local lensing Jacobian) on 2338
 *  sweep rays and 3265 held-out rays; mean steps + retries 348 -> 220 (-37 %, held-out 353 -> 217).
 *  The old controller's longer-stride failures were near-field near-axis error (fixed by the
 *  angular caps in the near branch), not far-stride error. 0.1 / 50 and 0.12 / 50 also pass at
 *  -40 % with a worse worst sky ray (0.21 px vs 0.15 px); 0.08 / 50 shipped for margin. Twin
 *  constants in integrator-shared.wgsl. */
export const K_FAR = 0.08;
export const DL_FAR_MAX = 50;

/** Angular step caps, applied to both branches of stepSize: at most F_AXIS of the affine distance
 *  to the axis at the current polar rate (dtheta/dl = p_theta / Sigma ~ p_theta / r^2) and at most
 *  F_PHI rad of azimuth per step (dphi/dl ~ p_phi / (r^2 sin^2 th)), each floored at `floor`.
 *  r^2 stands in for Sigma and the exact g^phiphi: step heuristics, identical in both twins.
 *  In the near field (2026-10-01) they resolve the near-axis passages inside r = 1.5 rOut, which
 *  H_TOL alone left up to 0.3 M off (the axis-line residual): 0 of 104 near-field axis-band rays
 *  off by > 0.02 M (tests/sweep-farmonitor.test.ts, set N), at ~+1 % mean steps. Twin of
 *  angularCap() in integrator-shared.wgsl. */
export function angularCap(s: Float64Array, dl0: number, floor: number): number {
  const r = s[1];
  let dl = dl0;
  const pth = s[6];
  if (pth !== 0) { // no polar motion: nothing to cap (and no division by zero)
    const thD = Math.min(s[2], Math.PI - s[2]);
    dl = Math.min(dl, Math.max(floor, F_AXIS * thD * r * r / Math.abs(pth)));
  }
  const pph = s[7];
  if (pph !== 0) {
    const sn = Math.sin(s[2]);
    dl = Math.min(dl, Math.max(floor, F_PHI * r * r * sn * sn / Math.abs(pph)));
  }
  return dl;
}

/** Baseline step length: fine in the strong-field/disk region, long strides through the near-flat
 *  far field; both capped by angularCap. Twin of stepSize() in integrator-shared.wgsl. */
export function stepSize(s: Float64Array, rh: number, rOut: number): number {
  const r = s[1];
  if (r > rOut * 1.5) return angularCap(s, Math.min(DL_FAR_MAX, Math.max(0.6, K_FAR * r)), DL_FAR_MIN);
  return angularCap(s, Math.min(0.5, Math.max(0.002, 0.02 * (r - rh))), 0.002);
}

/** g^{mu nu} p_mu p_nu (= 2H, exactly 0 for a null geodesic) together with the sum of |terms|.
 *  The constraint drift is judged relative to that sum because near the capture margin
 *  g^tt ~ -A/(Sigma Delta) reaches ~1e2 and H becomes a cancellation of large terms; an absolute
 *  tolerance would sit at the f32 noise floor there. With p_t = 1 and |g^tt| >= 1 outside the
 *  horizon the scale is always >= 1, so this is never looser than an absolute tolerance. Being
 *  RELATIVE, it cannot reject a step whose garbage momenta inflate the scale along with the drift
 *  -- which is why the step controller (stepSize's F_AXIS cap), not the monitor, must bound
 *  near-axis strides. */
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
 *  halvings. The final attempt is returned either way, with ok = false if it still failed; the
 *  caller proceeds with it regardless (a near-axis ray that exhausts the budget is still an
 *  axis-crosser by continuity, and a diverging one winds to budget exhaustion as before) and only
 *  records the exhaustion. NaN drift compares false against the tolerance, so it is never ok. */
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

export type Fate = "disk" | "captured" | "escaped" | "budget";
export interface TraceOpts { rIn: number; rOut: number; rObs: number; maxSteps?: number; hTol?: number; maxRetry?: number; }
/** exhausted = number of steps whose monitor came back ok = false (the smallest-step attempt was
 *  accepted and the ray proceeded; see stepGeodesic). maxAbsH = max |H| = |g^{mu nu} p_mu p_nu| / 2
 *  over every sampled state (spec 4.1: the null constraint is bounded along the WHOLE trajectory,
 *  not just at the end); diagnostic, not in the shader. */
export interface TraceResult { fate: Fate; s: Float64Array; steps: number; retries: number; exhausted: number; thMin: number; maxAbsH: number; rHit?: number; phiHit?: number; }

/** Backward ray trace with the render loop's termination order: disk crossing, then capture, then
 *  escape. thMin is the smallest sampled angular distance from either pole along the trajectory. */
export function traceRay(s0: Float64Array, a: number, o: TraceOpts): TraceResult {
  const rh = 1 + Math.sqrt(Math.max(0, 1 - a * a));
  const maxSteps = o.maxSteps ?? 20000;
  const absH = (x: Float64Array) => 0.5 * Math.abs(hquadScaled(x[1], x[2], a, x[4], x[5], x[6], x[7])[0]);
  let s = s0, retries = 0, exhausted = 0, thMin = Math.min(s0[2], Math.PI - s0[2]), maxAbsH = absH(s0);
  for (let step = 1; step <= maxSteps; step++) {
    const far = s[1] > o.rOut * 1.5; // same threshold as the far branch of stepSize
    const out = stepGeodesic(s, a, stepSize(s, rh, o.rOut), far ? H_TOL_FAR : o.hTol, o.maxRetry);
    retries += out.retries;
    // Retry exhaustion does not end the ray (twin of raytrace.wgsl): the smallest-step attempt is
    // accepted. Ending it at the (xi, eta) classifier painted starfield over the disk hits of
    // near-axis rays (the classifier can only answer captured/escaped) -- the alpha = 0 seam.
    if (!out.ok) exhausted++;
    const sN = out.s;
    maxAbsH = Math.max(maxAbsH, absH(sN));
    // A step that moved theta by more than 0.5 rad is not a plane crossing (a legitimate near-field
    // step moves theta by <= ~0.07 rad): it is a diverged state that reflectAxis's single-crossing
    // reduction cannot have made sense of, and interpolating a disk hit from it would be garbage.
    const f0 = s[2] - Math.PI / 2, f1 = sN[2] - Math.PI / 2;
    if (f0 * f1 < 0 && Math.abs(sN[2] - s[2]) < 0.5) {
      const frac = f0 / (f0 - f1);
      const rHit = s[1] + frac * (sN[1] - s[1]);
      if (rHit >= o.rIn && rHit <= o.rOut) {
        return { fate: "disk", s: sN, steps: step, retries, exhausted, thMin, maxAbsH, rHit, phiHit: s[3] + frac * (sN[3] - s[3]) };
      }
    }
    s = sN;
    thMin = Math.min(thMin, s[2], Math.PI - s[2]);
    if (s[1] <= rh * 1.005) return { fate: "captured", s, steps: step, retries, exhausted, thMin, maxAbsH };
    if (s[1] > o.rObs * 1.2) return { fate: "escaped", s, steps: step, retries, exhausted, thMin, maxAbsH };
  }
  return { fate: "budget", s, steps: maxSteps, retries, exhausted, thMin, maxAbsH };
}
