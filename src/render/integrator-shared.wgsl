// ---- Kerr metric + geodesic integrator. Twin of src/physics/kerr.ts, geodesic.ts, trace.ts. ----
//
// This is the SOLE WGSL copy of the metric helpers and the integrator. It is prepended as a plain
// string to raytrace.wgsl (by gpu.ts) and to parity.wgsl / integrator-parity.wgsl (by
// parity.browser.ts), so the ?parity route compiles the same bytes the renderer runs. Do not
// inline a second copy anywhere: a duplicate would let the renderer and the gate disagree.
//
// Self-contained: no uniforms, no bindings. Everything the render loop needs from U is passed in.

const PI = 3.141592653589793;

fn delta_(r: f32, a: f32) -> f32 { return r*r - 2.0*r + a*a; }
fn sigma_(r: f32, th: f32, a: f32) -> f32 { let c = cos(th); return r*r + a*a*c*c; }
fn bigA_(r: f32, th: f32, a: f32) -> f32 { let s = sin(th); return (r*r+a*a)*(r*r+a*a) - a*a*delta_(r,a)*s*s; }

// upper metric components (tt, tphi, rr, thth, phph)
// POLE_S2 floors sin^2(th) in the divergent 1/sin^2 denominator of g^{phi phi}. It is a NaN guard
// (inf * 0 for an exactly-on-axis ray with p_phi = 0), NOT a physics cap: at 1e-12 the affected
// cone is 1e-6 rad, far below a pixel. Matches src/physics/kerr.ts. The old 1e-3 removed the
// centrifugal barrier inside a 1.8 deg cone and let rays tunnel through the axis.
const POLE_S2 = 1e-12;
fn gUp(r: f32, th: f32, a: f32) -> array<f32,5> {
  let sn = sin(th); let cs = cos(th); let s2 = sn*sn; let s2d = max(s2, POLE_S2);
  let a2 = a*a; let ra = r*r + a2;
  let Sig = r*r + a2*cs*cs; let d = r*r - 2.0*r + a2; let A = ra*ra - a2*d*s2;
  return array<f32,5>( -A/(Sig*d), -2.0*a*r/(Sig*d), d/Sig, 1.0/Sig, (d - a2*s2)/(Sig*d*s2d) );
}
fn gLow(r: f32, th: f32, a: f32) -> array<f32,5> {
  let s2 = sin(th)*sin(th); let Sig = sigma_(r,th,a); let d = delta_(r,a);
  return array<f32,5>( -(1.0-2.0*r/Sig), -2.0*a*r*s2/Sig, Sig/d, Sig, (r*r+a*a+2.0*a*a*r*s2/Sig)*s2 );
}
fn omegaKep(r: f32, a: f32) -> f32 { return 1.0/(pow(r,1.5) + a); } // prograde, M=1

// Inverse metric and its exact r- and theta-derivatives. Twin of metricUpperGrad in kerr.ts;
// indices 0=tt, 1=tphi, 2=rr, 3=thth, 4=phph. The POLE_S2 floor's theta-derivative is 0 below it.
struct MetricGrad { g: array<f32,5>, dr: array<f32,5>, dth: array<f32,5> };
fn gUpGrad(r: f32, th: f32, a: f32) -> MetricGrad {
  let sn = sin(th); let cs = cos(th); let s2 = sn*sn; let sc = sn*cs;
  let floored = s2 < POLE_S2; let s2d = select(s2, POLE_S2, floored); let ds2d = select(2.0*sc, 0.0, floored);
  let a2 = a*a; let r2 = r*r; let ra = r2 + a2;
  let Sig = r2 + a2*cs*cs; let dSigR = 2.0*r; let dSigT = -2.0*a2*sc;
  let D = r2 - 2.0*r + a2; let dDR = 2.0*r - 2.0;
  let A = ra*ra - a2*D*s2; let dAR = 4.0*r*ra - a2*dDR*s2; let dAT = -2.0*a2*D*sc;
  let SD = Sig*D; let dSDR = dSigR*D + Sig*dDR; let dSDT = dSigT*D;
  let tt = -A/SD; let tphi = -2.0*a*r/SD; let rr = D/Sig; let thth = 1.0/Sig;
  let P = SD*s2d; let phph = (D - a2*s2)/P;
  let dPR = dSDR*s2d; let dPT = dSDT*s2d + SD*ds2d;
  return MetricGrad(
    array<f32,5>(tt, tphi, rr, thth, phph),
    array<f32,5>((-dAR - tt*dSDR)/SD, (-2.0*a - tphi*dSDR)/SD, (dDR - rr*dSigR)/Sig, -thth*dSigR/Sig, (dDR - phph*dPR)/P),
    array<f32,5>((-dAT - tt*dSDT)/SD, -tphi*dSDT/SD, -rr*dSigT/Sig, -thth*dSigT/Sig, (-2.0*a2*sc - phph*dPT)/P));
}

// state s = (t,r,th,phi, pt,pr,pth,pphi) packed as two vec4
struct State { x: vec4<f32>, p: vec4<f32> };
// Hamilton's equations with exact forces (twin of rhs in geodesic.ts): one metric evaluation per
// call instead of five finite-difference ones.
fn rhs(s: State, a: f32) -> State {
  let m = gUpGrad(s.x.y, s.x.z, a); let g = m.g; let p = s.p;
  let dx = vec4<f32>(g[0]*p.x + g[1]*p.w, g[2]*p.y, g[3]*p.z, g[1]*p.x + g[4]*p.w);
  let dQdr  = m.dr[0]*p.x*p.x + 2.0*m.dr[1]*p.x*p.w + m.dr[2]*p.y*p.y + m.dr[3]*p.z*p.z + m.dr[4]*p.w*p.w;
  let dQdth = m.dth[0]*p.x*p.x + 2.0*m.dth[1]*p.x*p.w + m.dth[2]*p.y*p.y + m.dth[3]*p.z*p.z + m.dth[4]*p.w*p.w;
  return State(dx, vec4<f32>(0.0, -0.5*dQdr, -0.5*dQdth, 0.0));
}
fn addS(s: State, k: State, f: f32) -> State { return State(s.x + k.x*f, s.p + k.p*f); }
fn rk4(s: State, a: f32, dl: f32) -> State {
  let k1 = rhs(s,a); let k2 = rhs(addS(s,k1,dl*0.5),a);
  let k3 = rhs(addS(s,k2,dl*0.5),a); let k4 = rhs(addS(s,k3,dl),a);
  return State(s.x + (k1.x+2.0*k2.x+2.0*k3.x+k4.x)*(dl/6.0),
               s.p + (k1.p+2.0*k2.p+2.0*k3.p+k4.p)*(dl/6.0));
}

// Far-field angular step cap. The far field is unmonitored (H_TOL_FAR below), so nothing but the
// step controller bounds a stride there -- and a ray aimed at screen beta crosses the axis at
// r ~ beta / sin(i), INSIDE the exempt zone whenever beta > 1.5 * rOut * sin(i) (beta > 8.3 M at
// i = 8 deg). A dl = 3 stride at theta ~ 3e-3 carried theta to -9.2 and p_theta to -6e4: the ray
// escaped and painted a dark streak beside the axis column above the shadow. The cap limits the
// far stride to F_AXIS of the affine distance to the axis at the current angular rate
// (theta_d * Sigma / |p_theta|, since dtheta/dl = p_theta / Sigma), floored at DL_FAR_MIN. Chosen
// by tests/sweep-axiscap.test.ts (SWEEP=1). Twin constants in trace.ts.
const F_AXIS = 0.1;
const DL_FAR_MIN = 0.05;

// Far-field stride (see K_FAR / DL_FAR_MAX in trace.ts). Twin constants.
const K_FAR = 0.04;
const DL_FAR_MAX = 6.0;

// Baseline step length: fine in the strong-field/disk region, long strides through the near-flat
// far field (curvature ~M/r^3 is negligible there) so we don't burn thousands of steps just
// travelling in from the distant observer, capped near the axis (F_AXIS). Twin of stepSize() in
// trace.ts.
fn stepSize(s: State, rh: f32, rOut: f32) -> f32 {
  let r = s.x.y;
  if (r > rOut * 1.5) {
    let base = clamp(K_FAR * r, 0.6, DL_FAR_MAX);
    let pth = s.p.z;
    if (pth == 0.0) { return base; } // no angular motion: nothing to cap (and no division by zero)
    // Sigma = r^2 + a^2 cos^2 th; the a^2 cos^2 th <= 1 term is < 3e-4 of r^2 >= 3600 here.
    let thD = min(s.x.z, PI - s.x.z);
    return min(base, max(DL_FAR_MIN, F_AXIS * thD * r * r / abs(pth)));
  }
  return clamp(0.02 * (r - rh), 0.002, 0.5);
}

// ---- Constraint-monitored stepping. Twin of stepGeodesic() in trace.ts. --------------------------
// For a null geodesic H = 1/2 g^{mu nu} p_mu p_nu = 0 exactly. RK4 truncation (the forces are exact,
// see gUpGrad) drifts off that, worst where the theta force is steep (the restored 1/sin^2
// barrier). After each step the CHANGE in H is compared against a tolerance RELATIVE to the size of
// the terms being cancelled (near the capture margin g^tt ~ 1e2, so an absolute tolerance would sit
// at f32 noise); on failure dl is halved and the step redone, up to MAX_RETRY halvings.
const H_TOL = 1e-3;
const MAX_RETRY = 8u;
// Far-field exemption: beyond rOut * 1.5 (the far branch of stepSize) the monitor is OFF. Still a
// NaN guard: abs(NaN) <= x is false. Historical rationale: under finite-difference forces the f32
// force was pure noise at r ~ 1e3 (ulp 6e-5 vs the FD half-step 1e-4; measured |dH|/scale 2.4e-3
// here vs 2e-11 in f64), so halving on dH there cost steps for nothing. Re-checked under exact
// forces (Task 5): |dH|/scale is now 2.7e-8 on the parity "far" step and 9.9e-9 on "far-axis", so
// that premise no longer holds; lifting the exemption is a follow-up. Twin constant in trace.ts.
const H_TOL_FAR = 1e30;

// vec2(g^{mu nu} p_mu p_nu, sum of |terms|). gUp indices: 0=tt, 1=tphi, 2=rr, 3=thth, 4=phph.
// Being RELATIVE, this measure cannot reject a step whose garbage momenta inflate the scale along
// with the drift -- which is why the step controller (stepSize's F_AXIS cap), not the monitor, must
// bound near-axis strides.
fn hquadScaled(r: f32, th: f32, a: f32, p: vec4<f32>) -> vec2<f32> {
  let g = gUp(r, th, a);
  let t0 = g[0]*p.x*p.x; let t1 = 2.0*g[1]*p.x*p.w; let t2 = g[2]*p.y*p.y;
  let t3 = g[3]*p.z*p.z; let t4 = g[4]*p.w*p.w;
  return vec2<f32>(t0 + t1 + t2 + t3 + t4, abs(t0) + abs(t1) + abs(t2) + abs(t3) + abs(t4));
}

// Exact analytic continuation through the polar axis: th -> -th (2pi - th at the south pole),
// phi -> phi + pi, p_th -> -p_th; p_phi unchanged. Identity unless a step carried th past 0 or pi.
fn reflectAxis(s: State) -> State {
  let th = s.x.z;
  if (th >= 0.0 && th <= PI) { return s; }
  let thR = select(2.0*PI - th, -th, th < 0.0);
  return State(vec4<f32>(s.x.x, s.x.y, thR, s.x.w + PI), vec4<f32>(s.p.x, s.p.y, -s.p.z, s.p.w));
}

struct StepOut { s: State, dl: f32, retries: u32, ok: bool };
// The final attempt is returned either way, with ok = false if it still failed; the caller proceeds
// with it regardless (twin of trace.ts: a near-axis ray that exhausts the budget is still an
// axis-crosser by continuity, a diverging one winds to budget exhaustion as before), so ok is
// informational -- ?parity compares it against the CPU twin. A NaN drift compares false against
// the tolerance, so it is never ok.
fn stepGeodesic(s: State, a: f32, dl0: f32, hTol: f32) -> StepOut {
  let h0 = hquadScaled(s.x.y, s.x.z, a, s.p).x;
  var dl = dl0;
  for (var k = 0u; k < MAX_RETRY; k++) {
    let sN = rk4(s, a, dl);
    let hs = hquadScaled(sN.x.y, sN.x.z, a, sN.p);
    if (abs(hs.x - h0) <= hTol * hs.y) { return StepOut(reflectAxis(sN), dl, k, true); }
    dl = dl * 0.5;
  }
  let sN = rk4(s, a, dl);
  let hs = hquadScaled(sN.x.y, sN.x.z, a, sN.p);
  let ok = abs(hs.x - h0) <= hTol * hs.y;
  return StepOut(reflectAxis(sN), dl, MAX_RETRY, ok);
}
