// ---- Analytic Kerr critical curve. Twin of src/physics/shadow.ts -- keep coefficients in sync. ----
//
// This is the SOLE WGSL copy of the classifier. It is prepended as a plain string to
// raytrace.wgsl (by gpu.ts) and to shadow-parity.wgsl (by parity.browser.ts), so the ?parity
// route exercises the same bytes the renderer compiles. Do not inline a second copy anywhere:
// that is precisely the failure mode this file exists to prevent.
const A_EPS = 1e-4;

fn criticalXi(r: f32, a: f32) -> f32 {
  let d = r*r - 2.0*r + a*a;
  return (r*r - a*a - r*d) / (a*(r - 1.0));
}
fn criticalEta(r: f32, a: f32) -> f32 {
  let d = r*r - 2.0*r + a*a;
  return (r*r*r * (4.0*d - r*(r - 1.0)*(r - 1.0))) / (a*a*(r - 1.0)*(r - 1.0));
}

// Capture test from the conserved impact parameters. Bisection count (24) must match shadow.ts.
fn classifyCaptured(xi: f32, eta: f32, a: f32) -> bool {
  if (abs(a) < A_EPS) { return eta + xi*xi < 27.0; }
  let lo = 2.0 * (1.0 + cos((2.0/3.0) * acos(-a)));
  let hi = 2.0 * (1.0 + cos((2.0/3.0) * acos(a)));
  let xiHi = criticalXi(hi, a);
  let xiLo = criticalXi(lo, a);
  if (xi <= xiHi || xi >= xiLo) { return false; }
  var a0 = lo; var b0 = hi;
  for (var k = 0; k < 24; k++) {
    let mid = 0.5 * (a0 + b0);
    if (criticalXi(mid, a) > xi) { a0 = mid; } else { b0 = mid; }
  }
  return eta < criticalEta(0.5*(a0 + b0), a);
}
