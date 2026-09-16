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
fn bigA_(r: f32, th: f32, a: f32) -> f32 { let s = sin(th); return pow(r*r+a*a,2.0) - a*a*delta_(r,a)*s*s; }

// upper metric components (tt, tphi, rr, thth, phph)
// POLE_S2 floors sin^2(th) in the divergent 1/sin^2 denominator of g^{phi phi}; matches
// src/physics/kerr.ts. Regularizes the Boyer-Lindquist polar axis so axis-grazing rays no
// longer get an unbounded p_th kick (which painted a black meridian seam + central cap).
const POLE_S2 = 1e-3;
fn gUp(r: f32, th: f32, a: f32) -> array<f32,5> {
  let s2 = sin(th)*sin(th); let s2d = max(s2, POLE_S2);
  let Sig = sigma_(r,th,a); let d = delta_(r,a); let A = bigA_(r,th,a);
  return array<f32,5>( -A/(Sig*d), -2.0*a*r/(Sig*d), d/Sig, 1.0/Sig, (d - a*a*s2)/(Sig*d*s2d) );
}
fn gLow(r: f32, th: f32, a: f32) -> array<f32,5> {
  let s2 = sin(th)*sin(th); let Sig = sigma_(r,th,a); let d = delta_(r,a);
  return array<f32,5>( -(1.0-2.0*r/Sig), -2.0*a*r*s2/Sig, Sig/d, Sig, (r*r+a*a+2.0*a*a*r*s2/Sig)*s2 );
}
fn omegaKep(r: f32, a: f32) -> f32 { return 1.0/(pow(r,1.5) + a); } // prograde, M=1

fn hquad(r: f32, th: f32, a: f32, p: vec4<f32>) -> f32 {
  let g = gUp(r,th,a);
  return g[0]*p.x*p.x + 2.0*g[1]*p.x*p.w + g[2]*p.y*p.y + g[3]*p.z*p.z + g[4]*p.w*p.w;
}
// state s = (t,r,th,phi, pt,pr,pth,pphi) packed as two vec4
struct State { x: vec4<f32>, p: vec4<f32> };
fn rhs(s: State, a: f32) -> State {
  let r = s.x.y; let th = s.x.z; let g = gUp(r,th,a);
  let dx = vec4<f32>(g[0]*s.p.x + g[1]*s.p.w, g[2]*s.p.y, g[3]*s.p.z, g[1]*s.p.x + g[4]*s.p.w);
  let h = 1e-4;
  let dQdr = (hquad(r+h,th,a,s.p) - hquad(r-h,th,a,s.p))/(2.0*h);
  let dQdth = (hquad(r,th+h,a,s.p) - hquad(r,th-h,a,s.p))/(2.0*h);
  let dp = vec4<f32>(0.0, -0.5*dQdr, -0.5*dQdth, 0.0);
  return State(dx, dp);
}
fn addS(s: State, k: State, f: f32) -> State { return State(s.x + k.x*f, s.p + k.p*f); }
fn rk4(s: State, a: f32, dl: f32) -> State {
  let k1 = rhs(s,a); let k2 = rhs(addS(s,k1,dl*0.5),a);
  let k3 = rhs(addS(s,k2,dl*0.5),a); let k4 = rhs(addS(s,k3,dl),a);
  return State(s.x + (k1.x+2.0*k2.x+2.0*k3.x+k4.x)*(dl/6.0),
               s.p + (k1.p+2.0*k2.p+2.0*k3.p+k4.p)*(dl/6.0));
}

// Baseline step length: fine in the strong-field/disk region, long strides through the near-flat
// far field (curvature ~M/r^3 is negligible there) so we don't burn thousands of steps just
// travelling in from the distant observer. Twin of stepSize() in trace.ts.
fn stepSize(r: f32, rh: f32, rOut: f32) -> f32 {
  if (r > rOut * 1.5) { return clamp(0.04 * r, 0.6, 6.0); }
  return clamp(0.02 * (r - rh), 0.002, 0.5);
}
