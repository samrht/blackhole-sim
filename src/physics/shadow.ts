// Analytic Kerr shadow boundary via the spherical-photon-orbit impact parameters (Bardeen 1973).
// Geometric units, M = 1, matching kerr.ts.
//
// Both xi and eta are 0/0 as a -> 0: the eta numerator factors as
//   4*Delta - r*(r-1)^2 = -r*(r-3)^2 + 4a^2
// so at a -> 0, r = 3 the limit is 27*4a^2/(4a^2) = 27 -- correct, but only as a limit. Naive
// evaluation divides by zero, so we branch to the exact Schwarzschild circle below A_EPS.
export const A_EPS = 1e-4;

const delta = (r: number, a: number) => r * r - 2 * r + a * a;

/** Impact parameters (xi, eta) of the spherical photon orbit at Boyer-Lindquist radius r. */
export function criticalXiEta(r: number, a: number): [number, number] {
  if (Math.abs(a) < A_EPS) {
    // Schwarzschild: the critical curve is the circle eta + xi^2 = 27, degenerate at r = 3.
    return [0, 27];
  }
  const d = delta(r, a);
  const xi = (r * r - a * a - r * d) / (a * (r - 1));
  const eta = (r * r * r * (4 * d - r * (r - 1) * (r - 1))) / (a * a * (r - 1) * (r - 1));
  return [xi, eta];
}

/** [prograde, retrograde] equatorial photon-orbit radii -- the endpoints of the photon shell. */
export function photonShellRange(a: number): [number, number] {
  const lo = 2 * (1 + Math.cos((2 / 3) * Math.acos(-a)));
  const hi = 2 * (1 + Math.cos((2 / 3) * Math.acos(a)));
  return [lo, hi];
}

/** Capture test for a null geodesic with conserved impact parameters (xi, eta).
 *
 *  xi_c(r) is strictly decreasing across the photon shell (asserted in tests), so we invert it by
 *  bisection to find the shell radius r* with xi_c(r*) = xi, then compare eta against eta_c(r*).
 *  Fixed iteration count: WGSL has no while-loops, and the twin must match this exactly. */
export function classify(xi: number, eta: number, a: number): "captured" | "escaped" {
  if (Math.abs(a) < A_EPS) {
    // Schwarzschild: capture iff the impact parameter b^2 = eta + xi^2 is inside b_crit^2 = 27.
    return eta + xi * xi < 27 ? "captured" : "escaped";
  }
  const [lo, hi] = photonShellRange(a);
  const xiHi = criticalXiEta(hi, a)[0]; // smallest xi (xi decreasing in r)
  const xiLo = criticalXiEta(lo, a)[0]; // largest xi
  if (xi <= xiHi || xi >= xiLo) return "escaped"; // no turning point in the shell
  let a0 = lo, b0 = hi;
  for (let k = 0; k < 24; k++) {
    const mid = 0.5 * (a0 + b0);
    if (criticalXiEta(mid, a)[0] > xi) a0 = mid;
    else b0 = mid;
  }
  const etaC = criticalXiEta(0.5 * (a0 + b0), a)[1];
  return eta < etaC ? "captured" : "escaped";
}

/** Shadow outline in celestial (alpha, beta) coordinates for a given spin and inclination.
 *  alpha = -xi/sin(i);  beta^2 = eta + a^2 cos^2(i) - xi^2 cot^2(i)  (both +/- beta branches). */
export function shadowBoundary(a: number, incl: number, nSamples: number): [number, number][] {
  const si = Math.sin(incl), ci = Math.cos(incl);
  const pts: [number, number][] = [];
  if (Math.abs(a) < A_EPS) {
    const b = Math.sqrt(27);
    for (let k = 0; k < nSamples; k++) {
      const t = (2 * Math.PI * k) / nSamples;
      pts.push([b * Math.cos(t), b * Math.sin(t)]);
    }
    return pts;
  }
  const [lo, hi] = photonShellRange(a);
  for (let k = 0; k <= nSamples; k++) {
    const r = lo + ((hi - lo) * k) / nSamples;
    const [xi, eta] = criticalXiEta(r, a);
    const b2 = eta + a * a * ci * ci - (xi * xi * ci * ci) / (si * si);
    if (b2 < 0) continue; // this shell radius is not visible at this inclination
    const al = -xi / si, be = Math.sqrt(b2);
    pts.push([al, be]);
    pts.push([al, -be]);
  }
  return pts;
}
