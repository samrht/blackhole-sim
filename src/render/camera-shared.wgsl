// ---- Camera: screen (alpha,beta) -> conserved quantities + initial 4-momentum. ----
// Twin of src/physics/camera.ts -- keep the two in sync; ?parity now enforces it.
//
// This is the SOLE WGSL copy of the camera mapping. It is prepended as a plain string to
// raytrace.wgsl (by gpu.ts) and to camera-parity.wgsl (by parity.browser.ts), so the ?parity
// route exercises the same bytes the renderer compiles. Do not inline a second copy anywhere:
// a duplicate would let the renderer and the gate disagree, which is exactly how the
// "negate p_t only" bug survived the project's entire history.
//
// Deliberately pure scalar algebra: no State, no gUp, no globals. The inverse-metric components
// are passed in by the caller, so this file can be prepended ahead of everything else without
// any use-before-declaration hazard.

// (xi, eta) from the Bardeen screen coordinates. Both are conserved along the geodesic, so they
// classify a ray regardless of where integration stopped.
//   xi  = -alpha*sin(i)                                  (azimuthal angular momentum L_z/E)
//   eta = beta^2 + xi^2*cot^2(i) - a^2*cos^2(i)          (Carter constant)
fn cameraXiEta(alpha: f32, beta: f32, a: f32, i: f32) -> vec2<f32> {
  let xi = -alpha * sin(i);
  let ci = cos(i); let si = sin(i);
  let eta = beta*beta + xi*xi*(ci*ci)/max(si*si, 1e-8) - a*a*ci*ci;
  return vec2<f32>(xi, eta);
}

// Past-directed initial 4-momentum (pt, pr, pth, pphi) for BACKWARD tracing, given the
// inverse-metric components at the observer (gtt, gtphi, grr, gthth, gphph).
//
// Backward tracing follows the arriving photon's worldline in REVERSE, which negates the whole
// 4-momentum -- not p_t alone. This ray is past-directed (dt/dl < 0) and inward (dr/dl < 0).
// Negating only p_t leaves a future-directed ray falling away from the camera: a different
// geodesic, which renders inclination (pi - incl). See src/physics/camera.ts and tests/camera.test.ts.
// xi = L_z/E is unchanged by the negation, so the g-factor and shadow classifier still use xi.
fn cameraMomenta(xi: f32, beta: f32, gtt: f32, gtphi: f32, grr: f32, gthth: f32, gphph: f32) -> vec4<f32> {
  let pt = 1.0; let pphi = -xi; let pth = -beta;
  let rest = gtt*pt*pt + 2.0*gtphi*pt*pphi + gthth*pth*pth + gphph*pphi*pphi;
  let pr = -sqrt(max(0.0, -rest/grr)); // inward
  return vec4<f32>(pt, pr, pth, pphi);
}
