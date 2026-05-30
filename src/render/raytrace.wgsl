struct Uniforms {
  res: vec2<f32>, a: f32, incl: f32, rObs: f32, fovScale: f32, rIn: f32, rOut: f32,
  Tpeak: f32, exposure: f32, time: f32, frame: u32, reset: u32, maxSteps: u32,
};
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read_write> accum: array<vec4<f32>>;

const PI = 3.141592653589793;

fn delta_(r: f32, a: f32) -> f32 { return r*r - 2.0*r + a*a; }
fn sigma_(r: f32, th: f32, a: f32) -> f32 { let c = cos(th); return r*r + a*a*c*c; }
fn bigA_(r: f32, th: f32, a: f32) -> f32 { let s = sin(th); return pow(r*r+a*a,2.0) - a*a*delta_(r,a)*s*s; }

// upper metric components (tt, tphi, rr, thth, phph)
fn gUp(r: f32, th: f32, a: f32) -> array<f32,5> {
  let s2 = sin(th)*sin(th); let Sig = sigma_(r,th,a); let d = delta_(r,a); let A = bigA_(r,th,a);
  return array<f32,5>( -A/(Sig*d), -2.0*a*r/(Sig*d), d/Sig, 1.0/Sig, (d - a*a*s2)/(Sig*d*s2) );
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

// temporary analytic temperature (replaced by LUT in Task 10): peaks outside ISCO, ~r^-3/4
fn tempTmp(r: f32, rin: f32) -> f32 {
  if (r <= rin) { return 0.0; }
  let f = pow(rin/r, 0.75) * pow(max(0.0, 1.0 - sqrt(rin/r)), 0.25);
  return f / 0.23; // rough normalization so peak ~1
}
fn cheapColor(T: f32) -> vec3<f32> { // placeholder palette (real color LUT in Task 10)
  let t = clamp(T, 0.0, 1.0);
  return mix(vec3(1.0,0.3,0.05), vec3(0.6,0.8,1.0), t);
}

@compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u32(U.res.x) || gid.y >= u32(U.res.y)) { return; }
  let idx = gid.y * u32(U.res.x) + gid.x;
  let a = U.a; let i = U.incl;

  // pixel -> impact parameters (alpha,beta) in units of M
  let aspect = U.res.x / U.res.y;
  let ndc = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / U.res * 2.0 - 1.0;
  let alpha = ndc.x * U.fovScale * aspect;
  let beta  = -ndc.y * U.fovScale;
  // invert Bardeen: xi = -alpha*sin i ; eta = beta^2 - a^2 cos^2 i + xi^2 cot^2 i
  let xi = -alpha * sin(i);
  let ci = cos(i); let cot2 = (ci*ci)/(sin(i)*sin(i));
  let eta = beta*beta - a*a*ci*ci + xi*xi*cot2;

  // initial state at (rObs, i, 0), E=1
  let r0 = U.rObs; let th0 = i;
  let pt = -1.0; let pphi = xi; let pth = beta; // sign of p_th set by image y
  let gU = gUp(r0, th0, a);
  let rest = gU[0]*pt*pt + 2.0*gU[1]*pt*pphi + gU[3]*pth*pth + gU[4]*pphi*pphi;
  let pr = -sqrt(max(0.0, -rest/gU[2])); // inward
  var s = State(vec4<f32>(0.0, r0, th0, 0.0), vec4<f32>(pt, pr, pth, pphi));

  let rh = 1.0 + sqrt(max(0.0, 1.0 - a*a)); // horizon
  var color = vec3<f32>(0.0);

  for (var step = 0u; step < U.maxSteps; step++) {
    // adaptive step: smaller near the hole
    let r = s.x.y;
    let dl = -clamp(0.02 * (r - rh), 0.002, 0.5);
    let sNew = rk4(s, a, dl);

    // disk crossing: equatorial plane th = PI/2
    let f0 = s.x.z - PI*0.5; let f1 = sNew.x.z - PI*0.5;
    if (f0 * f1 < 0.0) {
      let frac = f0 / (f0 - f1);
      let rHit = mix(s.x.y, sNew.x.y, frac);
      if (rHit >= U.rIn && rHit <= U.rOut) {
        let Tn = tempTmp(rHit, U.rIn);
        let Om = omegaKep(rHit, a);
        let gl = gLow(rHit, PI*0.5, a);
        let rad = -(gl[0] + 2.0*Om*gl[1] + Om*Om*gl[4]);
        let g = sqrt(max(0.0, rad)) / (1.0 - Om*xi);  // redshift factor
        let u = g * Tn;                                // observed color temperature factor
        color = cheapColor(clamp(u, 0.0, 1.0)) * pow(u, 4.0); // brightness ∝ (g*T)^4
        break;
      }
    }
    s = sNew;
    if (s.x.y <= rh * 1.001) { color = vec3(0.0); break; }   // captured -> shadow
    if (s.x.y > r0 * 1.2) { color = vec3(0.0); break; }      // escaped -> background
  }

  let prev = select(accum[idx].rgb, vec3(0.0), U.reset == 1u);
  accum[idx] = vec4<f32>(prev + color, 1.0);
}
