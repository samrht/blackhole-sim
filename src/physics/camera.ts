import { nullRadialMomentum } from "./geodesic";

/**
 * Screen impact parameters (alpha, beta) -> initial state for BACKWARD ray tracing.
 *
 * GATED: the GPU twin is src/render/camera-shared.wgsl, which is the SOLE WGSL copy -- gpu.ts
 * prepends it to raytrace.wgsl and parity.browser.ts prepends it to camera-parity.wgsl, so the
 * ?parity route compiles the exact bytes the renderer runs. Changing the convention in either
 * file without the other now FAILS ?parity. Do not inline a second copy of this math anywhere.
 *
 * The photon we see arrives at the camera moving outward, forward in coordinate time. Retracing
 * it means following its worldline in reverse, i.e. integrating the fully negated 4-momentum:
 * every component flips, not just p_t. The resulting ray is PAST-directed (dt/dl < 0) and inward
 * (dr/dl < 0). Negating p_t alone leaves a future-directed ray falling away from the camera --
 * a different null geodesic entirely, which renders the image at inclination (pi - incl).
 *
 * The conserved ratio xi = L_z/E is unchanged by the negation (both flip), so the redshift
 * g-factor and the shadow classifier still take the same xi as before.
 */
export function screenToState(alpha: number, beta: number, a: number, incl: number, rObs: number): Float64Array {
  const xi = -alpha * Math.sin(incl);
  const pt = 1, pphi = -xi, pth = -beta;   // full negation of the arriving photon's momentum
  const pr = -nullRadialMomentum(rObs, incl, a, pt, pphi, pth); // inward
  return new Float64Array([0, rObs, incl, 0, pt, pr, pth, pphi]);
}

/** Carter constant from the Bardeen screen coordinates. Both xi and eta are conserved. */
export function screenToXiEta(alpha: number, beta: number, a: number, incl: number): [number, number] {
  const xi = -alpha * Math.sin(incl);
  const ci = Math.cos(incl), si = Math.sin(incl);
  return [xi, beta * beta + (xi * xi * ci * ci) / Math.max(si * si, 1e-8) - a * a * ci * ci];
}
