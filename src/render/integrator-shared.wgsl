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

// Far-field angular step cap. The far field was unmonitored when this was added, so nothing but the
// step controller bounded a stride there -- and a ray aimed at screen beta crosses the axis at
// r ~ beta / sin(i), INSIDE the exempt zone whenever beta > 1.5 * rOut * sin(i) (beta > 8.3 M at
// i = 8 deg). A dl = 3 stride at theta ~ 3e-3 carried theta to -9.2 and p_theta to -6e4: the ray
// escaped and painted a dark streak beside the axis column above the shadow. The cap limits the
// far stride to F_AXIS of the affine distance to the axis at the current angular rate
// (theta_d * Sigma / |p_theta|, since dtheta/dl = p_theta / Sigma), floored at DL_FAR_MIN. Chosen
// by tests/sweep-axiscap.test.ts (SWEEP=1). Twin constants in trace.ts.
const F_AXIS = 0.1;
const DL_FAR_MIN = 0.05;

// Far-field azimuthal step cap and stride (see F_PHI, K_FAR / DL_FAR_MAX in trace.ts). Twin constants.
const F_PHI = 0.1;
const K_FAR = 0.08;
const DL_FAR_MAX = 50.0;

// Baseline step length: fine in the strong-field/disk region, long strides through the near-flat
// far field (curvature ~M/r^3 is negligible there) so we don't burn thousands of steps just
// travelling in from the distant observer; both capped by angularCap. Twin of stepSize() in
// trace.ts.
// Angular step caps, applied to both branches of stepSize: at most F_AXIS of the affine distance to
// the axis at the current polar rate (dtheta/dl = p_theta / Sigma ~ p_theta / r^2), and at most
// F_PHI rad of azimuth per step (dphi/dl ~ p_phi / (r^2 sin^2 th)), each floored at `floor`.
// r^2 stands in for Sigma and for the exact g^phiphi: Sigma >= r^2 makes the polar cap slightly
// stricter, and both are step heuristics (the CPU twin uses the same expressions, so parity holds).
fn angularCap(s: State, dl0: f32, floor: f32) -> f32 {
  let r = s.x.y;
  var dl = dl0;
  let pth = s.p.z;
  if (pth != 0.0) { // no polar motion: nothing to cap (and no division by zero)
    let thD = min(s.x.z, PI - s.x.z);
    dl = min(dl, max(floor, F_AXIS * thD * r * r / abs(pth)));
  }
  let pph = s.p.w;
  if (pph != 0.0) {
    let sn = sin(s.x.z);
    dl = min(dl, max(floor, F_PHI * r * r * sn * sn / abs(pph)));
  }
  return dl;
}

fn stepSize(s: State, rh: f32, rOut: f32) -> f32 {
  let r = s.x.y;
  if (r > rOut * 1.5) {
    return angularCap(s, clamp(K_FAR * r, 0.6, DL_FAR_MAX), DL_FAR_MIN);
  }
  return angularCap(s, clamp(0.02 * (r - rh), 0.002, 0.5), 0.002);
}

// ---- Constraint-monitored stepping. Twin of stepGeodesic() in trace.ts. --------------------------
// For a null geodesic H = 1/2 g^{mu nu} p_mu p_nu = 0 exactly. RK4 truncation (the forces are exact,
// see gUpGrad) drifts off that, worst where the theta force is steep (the restored 1/sin^2
// barrier). After each step the CHANGE in H is compared against a tolerance RELATIVE to the size of
// the terms being cancelled (near the capture margin g^tt ~ 1e2, so an absolute tolerance would sit
// at f32 noise); on failure dl is halved and the step redone, up to MAX_RETRY halvings.
const H_TOL = 1e-3;
const MAX_RETRY = 8u;
// Far-field monitor tolerance: beyond rOut * 1.5 (the far branch of stepSize). Was 1e30 (monitor
// OFF) while the forces were finite differences, whose f32 noise at r ~ 1e3 made dH meaningless
// there; with exact forces |dH|/scale on a far step is ~1e-8 on the GPU, so the far field is now
// monitored as a safety net at 1e-5 (it never fires on the tests/sweep-farmonitor.test.ts rays
// once the angular caps bound both fields). Twin constant in trace.ts.
const H_TOL_FAR = 1e-5;

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

// (sin x, cos x) to ~1 ulp: Cephes sinf / cosf on |y| <= pi/4 after Cody-Waite reduction by pi/2. WGSL's built-in sin / cos
// may be off by 2^-11 absolute (the Intel iGPU's cos(1 deg) is 2e-5 off: near a pole a 13 % error in sin^2 theta). CPU
// mirror and accuracy test: src/render/sincos.ts, tests/sincos.test.ts.
fn sinCosP(x: f32) -> vec2<f32> {
  let j = round(x * 0.6366197723675814);
  let y = ((x - j * 1.5703125) - j * 4.837512969970703e-4) - j * 7.549789954891882e-8;
  let z = y * y;
  let s = ((-1.9515295891e-4 * z + 8.3321608736e-3) * z + -1.6666654611e-1) * z * y + y;
  let c = ((2.443315711809948e-5 * z + -1.388731625493765e-3) * z + 4.166664568298827e-2) * z * z - (0.5 * z - 1.0);
  let q = ((i32(j) % 4) + 4) % 4;
  if (q == 0) { return vec2<f32>(s, c); }
  if (q == 1) { return vec2<f32>(c, -s); }
  if (q == 2) { return vec2<f32>(-s, -c); }
  return vec2<f32>(-c, s);
}
// ---- Mino-time integrator (spec 2026-10-07 mino integrator; twin: src/physics/trace-mino.ts) -----------------------
// Carter's separated null-geodesic equations in Mino time (d lambda = dl / Sigma) with w = 1/r (w'' = R~'(w)/2) and the
// polar angle theta itself (theta'' = Theta'(theta)/2); t, phi and l (affine) as quadratures. Per ray: p_t = 1
// (camera-normalised, past-directed), xi = -p_phi, eta (Carter). Dormand-Prince 5(4) with the twin's error norm, the
// first-integral drift check, the polar-oscillation guard and bounded rejects; cubic Hermite dense output. See
// trace-mino.ts for why w = 1/r and theta (not r, not cos theta). Trig through sinCosP (above), never the built-ins.
// Hemisphere (twin: minoHemi): the stored angle v is measured from the NEARER pole, sigma = v.w = +1 north / -1 south;
// a step that starts past the equator first mirrors the state (exact: the equations are even about the equator), so
// f32 keeps relative precision near the south pole too.
const MINO_TOL = 1e-5;
const MINO_UFRAC = 0.25;
const MINO_MAX_REJECT = 10u;
const MINO_CTOL = 1e-3;
const MINO_LAND_ITERS = 2u;      // Newton iterations of the plane landing step (minoLand)
const MINO_LAND_BAND = 0.01;     // relative widening of [rIn, rOut] before landing (minoPlane)
const MINO_EPS = 1.1920929e-7;    // f32 machine epsilon: the drift check's rounding-noise floor
struct MinoRay { a: f32, xi: f32, eta: f32, K: f32, A1: f32, omMax: f32 };
struct Mino { q: vec4<f32>, v: vec4<f32> };   // q = (t, w, v, phi), v = (l, w', v', sigma); theta = v or pi - v
fn minoRay(a: f32, xi: f32, eta: f32) -> MinoRay {
  let om2 = max(0.0, eta + xi * xi - a * a) + 6.0 * a * a;
  return MinoRay(a, xi, eta, eta + (xi - a) * (xi - a), a * a - a * xi, sqrt(om2));
}
fn minoRw(w: f32, c: MinoRay) -> f32 { let Q = 1.0 + c.A1 * w * w; return Q * Q - c.K * w * w * (1.0 - 2.0 * w + c.a * c.a * w * w); }
// Camera at (r0, th0) with screen beta: w' = +sqrt(R~) (r decreasing), theta' = p_theta = -beta.
fn minoInit(r0: f32, th0: f32, beta: f32, c: MinoRay) -> Mino {
  let w0 = 1.0 / r0;
  return minoHemi(Mino(vec4<f32>(0.0, w0, th0, 0.0), vec4<f32>(0.0, sqrt(max(0.0, minoRw(w0, c))), -beta, 1.0)));
}
// Mirror a state past the equator into the other hemisphere's frame (identity otherwise).
fn minoHemi(y: Mino) -> Mino {
  if (y.q.z <= 0.5 * PI) { return y; }
  return Mino(vec4<f32>(y.q.x, y.q.y, PI - y.q.z, y.q.w), vec4<f32>(y.v.x, y.v.y, -y.v.z, -y.v.w));
}
fn minoRhs(y: Mino, c: MinoRay) -> Mino {
  let w = y.q.y; let a = c.a; let xi = c.xi; let a2 = a * a; let w2 = w * w;
  let sc = sinCosP(y.q.z); let s = sc.x; let co = sc.y; let s2 = max(s * s, 1e-30);
  let Q = 1.0 + c.A1 * w2;                       // P w^2
  let Dw = 1.0 - 2.0 * w + a2 * w2;              // Delta w^2
  return Mino(
    vec4<f32>(-((1.0 + a2 * w2) * Q / (w2 * Dw) + a * (xi - a * s2)), y.v.y, y.v.z, -(a * Q / Dw - a + xi / s2)),
    vec4<f32>(1.0 / w2 + a2 * co * co, 2.0 * c.A1 * w * Q - c.K * (w - 3.0 * w2 + 2.0 * a2 * w2 * w),
              co * (xi * xi / (s2 * s) - a2 * s), 0.0));   // sigma' = 0: mStage keeps v.w
}
fn mStage(y: Mino, h: f32, k1: Mino, c1: f32, k2: Mino, c2: f32, k3: Mino, c3: f32, k4: Mino, c4: f32, k5: Mino, c5: f32, k6: Mino, c6: f32) -> Mino {
  return Mino(y.q + h * (k1.q * c1 + k2.q * c2 + k3.q * c3 + k4.q * c4 + k5.q * c5 + k6.q * c6),
              y.v + h * (k1.v * c1 + k2.v * c2 + k3.v * c3 + k4.v * c4 + k5.v * c5 + k6.v * c6));
}
// Error scale per component (twin: errNorm): t, l against the step's change (floor 1); w relative; theta absolute and
// relative to sin(theta) near a pole; phi absolute; w' against max(|w'|, w); theta' against max(1, |theta'|).
fn minoErrNorm(y0: Mino, y1: Mino, e: Mino, tol: f32) -> f32 {
  let sth = min(abs(sinCosP(y0.q.z).x), abs(sinCosP(y1.q.z).x));
  let sq = vec4<f32>(max(1.0, abs(y1.q.x - y0.q.x)), max(y0.q.y, y1.q.y), min(1.0, max(sth, 1e-4)), 1.0);
  let sv = vec3<f32>(max(1.0, abs(y1.v.x - y0.v.x)), max(abs(y1.v.y), y1.q.y), max(1.0, abs(y1.v.z)));
  let m1 = abs(e.q) / (tol * sq); let m2 = abs(e.v.xyz) / (tol * sv);
  return max(max(max(m1.x, m1.y), max(m1.z, m1.w)), max(max(m2.x, m2.y), m2.z));
}
// First-integral drift over a step against max(MINO_CTOL tol x the size of its terms, its rounding noise) (twin: minoDrift),
// from differences so rounding scales with the step. f1 = minoRhs(y1).
fn minoDriftNorm(y0: Mino, y1: Mino, f1: Mino, c: MinoRay, tol: f32) -> f32 {
  let w0 = y0.q.y; let w1 = y1.q.y; let a2 = c.a * c.a;
  let c2 = 2.0 * c.A1 - c.K; let c3 = 2.0 * c.K; let c4 = c.A1 * c.A1 - c.K * a2;
  let sw = w1 + w0; let dR = (w1 - w0) * (c2 * sw + c3 * (w1 * w1 + w1 * w0 + w0 * w0) + c4 * sw * (w1 * w1 + w0 * w0));
  let dW = (y1.v.y - y0.v.y) * (y1.v.y + y0.v.y) - dR;
  let s0 = sinCosP(y0.q.z).x; let sc1 = sinCosP(y1.q.z); let s1 = sc1.x; let co1 = sc1.y;
  let dTh = sinCosP(y1.q.z - y0.q.z).x * sinCosP(y1.q.z + y0.q.z).x * (c.xi * c.xi / max(s0 * s0 * s1 * s1, 1e-37) - a2);
  let dV = (y1.v.z - y0.v.z) * (y1.v.z + y0.v.z) - dTh;
  let w2 = w1 * w1; let Q = 1.0 + c.A1 * w2;
  let scaleW = y1.v.y * y1.v.y + Q * Q + abs(c.K * w2 * (1.0 - 2.0 * w1 + a2 * w2));
  let scaleT = y1.v.z * y1.v.z + abs(c.eta) + a2 * co1 * co1 + c.xi * c.xi * co1 * co1 / max(s1 * s1, 1e-30);
  let noiseW = 4.0 * MINO_EPS * (2.0 * abs(f1.v.y) * w1 + abs(y1.v.y) * (abs(y1.v.y) + abs(y0.v.y)));
  let noiseT = 4.0 * MINO_EPS * (2.0 * abs(f1.v.z) * abs(y1.q.z) + abs(y1.v.z) * (abs(y1.v.z) + abs(y0.v.z))); // theta: relative precision
  return max(abs(dW) / max(MINO_CTOL * tol * scaleW, noiseW), abs(dV) / max(MINO_CTOL * tol * scaleT, noiseT));
}
// y0, f0: the start state and derivative AS USED (mirrored if the step began past the equator); dense output and the
// plane crossing of the step take (y0, y, f0, f1).
struct MinoTryOut { y1: Mino, f1: Mino, en: f32 };
// One DP5(4) attempt of size h from y (f0 = minoRhs(y)): the 5th-order result, its derivative (FSAL) and the error norm.
fn minoTry(y: Mino, f0: Mino, h: f32, c: MinoRay) -> MinoTryOut {
  let z = Mino(vec4<f32>(0.0), vec4<f32>(0.0));
  let k1 = f0;
  let k2 = minoRhs(mStage(y, h, k1, 1.0 / 5.0, z, 0.0, z, 0.0, z, 0.0, z, 0.0, z, 0.0), c);
  let k3 = minoRhs(mStage(y, h, k1, 3.0 / 40.0, k2, 9.0 / 40.0, z, 0.0, z, 0.0, z, 0.0, z, 0.0), c);
  let k4 = minoRhs(mStage(y, h, k1, 44.0 / 45.0, k2, -56.0 / 15.0, k3, 32.0 / 9.0, z, 0.0, z, 0.0, z, 0.0), c);
  let k5 = minoRhs(mStage(y, h, k1, 19372.0 / 6561.0, k2, -25360.0 / 2187.0, k3, 64448.0 / 6561.0, k4, -212.0 / 729.0, z, 0.0, z, 0.0), c);
  let k6 = minoRhs(mStage(y, h, k1, 9017.0 / 3168.0, k2, -355.0 / 33.0, k3, 46732.0 / 5247.0, k4, 49.0 / 176.0, k5, -5103.0 / 18656.0, z, 0.0), c);
  let y1 = mStage(y, h, k1, 35.0 / 384.0, k2, 0.0, k3, 500.0 / 1113.0, k4, 125.0 / 192.0, k5, -2187.0 / 6784.0, k6, 11.0 / 84.0);
  let k7 = minoRhs(y1, c);
  let e = mStage(z, h, k1, 71.0 / 57600.0, k3, -71.0 / 16695.0, k4, 71.0 / 1920.0,
                 k5, -17253.0 / 339200.0, k6, 22.0 / 525.0, k7, -1.0 / 40.0);
  return MinoTryOut(y1, k7, max(minoErrNorm(y, y1, e, MINO_TOL), minoDriftNorm(y, y1, k7, c, MINO_TOL)));
}
// y0, f0: the start state and derivative AS USED (mirrored if the step began past the equator); dense output and the
// plane crossing of the step take (y0, y, f0, f1).
struct MinoStepOut { y0: Mino, y: Mino, f0: Mino, f1: Mino, h: f32, hNext: f32, attempts: u32, en: f32 };
// One accepted DP5(4) step from y (f0 = minoRhs(y), the previous step's FSAL value) with proposed size h0, capped at the
// polar-oscillation guard; rejected attempts shrink h, and after MINO_MAX_REJECT the last attempt is accepted.
fn minoStep(yIn: Mino, f0In: Mino, h0: f32, c: MinoRay) -> MinoStepOut {
  let y = minoHemi(yIn);
  var f0 = f0In;
  if (yIn.q.z > 0.5 * PI) { f0 = minoRhs(y, c); }
  let hMax = select(1e30, MINO_UFRAC * PI / c.omMax, c.omMax > 0.0);
  var h = min(h0, hMax);
  var att = 1u;
  loop {
    let t = minoTry(y, f0, h, c); let en = t.en;
    // finite factor: 5 for a zero error, 0.2 for a non-finite one (NaN compares false everywhere)
    let finite = en < 1e30;
    let fac = select(select(0.2, 0.9 * pow(max(en, 1e-30), -0.2), finite), 5.0, en == 0.0);
    if (en <= 1.0 || att > MINO_MAX_REJECT) {
      return MinoStepOut(y, t.y1, f0, t.f1, h, min(hMax, h * min(5.0, max(0.2, fac))), att, en);
    }
    h = h * max(0.2, min(0.9, fac));
    att = att + 1u;
  }
}
// Cubic Hermite in lambda (values y0, y1, derivatives f0, f1 over step h) at fraction th of the step.
fn minoDense(y0: Mino, y1: Mino, f0: Mino, f1: Mino, h: f32, th: f32) -> Mino {
  if (th == 0.0) { return y0; }
  let t2 = th * th; let t3 = t2 * th;
  let h00 = 2.0 * t3 - 3.0 * t2 + 1.0; let h10 = (t3 - 2.0 * t2 + th) * h; let h01 = -2.0 * t3 + 3.0 * t2; let h11 = (t3 - t2) * h;
  return Mino(y0.q * h00 + f0.q * h10 + y1.q * h01 + f1.q * h11, y0.v * h00 + f0.v * h10 + y1.v * h01 + f1.v * h11);
}
// Fraction of the step where cos(theta) = 0 (the equatorial plane; bisection on the dense theta), or -1 without a sign change.
fn minoCrossing(y0: Mino, y1: Mino, f0: Mino, f1: Mino, h: f32) -> f32 {
  let c0 = sinCosP(y0.q.z).y; let c1 = sinCosP(y1.q.z).y;
  if (!(c0 * c1 < 0.0 || (c1 == 0.0 && c0 != 0.0))) { return -1.0; }
  var lo = 0.0; var hi = 1.0;
  for (var k = 0u; k < 24u; k++) {
    let m = 0.5 * (lo + hi);
    if (sinCosP(minoDense(y0, y1, f0, f1, h, m).q.z).y * c0 > 0.0) { lo = m; } else { hi = m; }
  }
  return 0.5 * (lo + hi);
}
// The state ON the equatorial plane within the step (y0, f0, h) whose dense crossing is at fraction th: a DP5 step of
// size s from y0, s refined by safeguarded Newton on cos(theta) (bisection inside the sign bracket when Newton leaves it).
// The cubic Hermite crossing is only 4th order; the landing step carries the step's own accuracy to the plane (twin:
// minoLand).
fn minoLand(y0: Mino, f0: Mino, h: f32, th: f32, c: MinoRay) -> Mino {
  let c0 = sinCosP(y0.q.z).y;
  var lo = 0.0; var hi = h; var s = th * h;
  for (var k = 0u; k < MINO_LAND_ITERS; k++) {
    let t = minoTry(y0, f0, s, c); let sc = sinCosP(t.y1.q.z);
    if (sc.y * c0 > 0.0) { lo = s; } else { hi = s; }
    let sn = s - sc.y / (-sc.x * t.f1.q.z);
    s = select(0.5 * (lo + hi), sn, sn > lo && sn < hi);
  }
  return minoTry(y0, f0, s, c).y1;
}
struct MinoHit { ok: bool, y: Mino };
// Disk test of one step (twin: minoPlane): the landed plane state when the step crosses the plane with the dense crossing
// radius inside [rIn, rOut] widened by MINO_LAND_BAND; the caller tests the landed radius against [rIn, rOut].
fn minoPlane(st: MinoStepOut, c: MinoRay, rIn: f32, rOut: f32) -> MinoHit {
  let th = minoCrossing(st.y0, st.y, st.f0, st.f1, st.h);
  if (th < 0.0) { return MinoHit(false, st.y); }
  let r = 1.0 / minoDense(st.y0, st.y, st.f0, st.f1, st.h, th).q.y;
  if (!(r >= rIn * (1.0 - MINO_LAND_BAND) && r <= rOut * (1.0 + MINO_LAND_BAND))) { return MinoHit(false, st.y); }
  return MinoHit(true, minoLand(st.y0, st.f0, st.h, th, c));
}
// Today's State (x = (t, r, theta, phi), p = (p_t, p_r, p_theta, p_phi)): p_r = -w' / (1 - 2w + a^2 w^2), p_theta = theta'
// (theta = v, p_theta = v' north; pi - v, -v' south). theta is folded into [0, pi] (through a pole: theta -> 2 pi - theta,
// phi + pi, p_theta -> -p_theta; the same point and direction).
fn minoToState(y: Mino, c: MinoRay) -> State {
  let w = y.q.y; let Dw = 1.0 - 2.0 * w + c.a * c.a * w * w; let south = y.v.w < 0.0;
  let th0 = select(y.q.z, PI - y.q.z, south);
  var th = th0 - 2.0 * PI * floor(th0 / (2.0 * PI)); var ph = y.q.w; var pth = select(y.v.z, -y.v.z, south);
  if (th > PI) { th = 2.0 * PI - th; ph = ph + PI; pth = -pth; }
  return State(vec4<f32>(y.q.x, 1.0 / w, th, ph), vec4<f32>(1.0, -y.v.y / Dw, pth, -c.xi));
}
