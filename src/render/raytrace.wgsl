struct Uniforms {
  res: vec2<f32>, a: f32, incl: f32, rObs: f32, fovScale: f32, rIn: f32, rOut: f32,
  Tpeak: f32, exposure: f32, time: f32, frame: u32, reset: u32, maxSteps: u32,
};
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> tempLUT: array<f32>;       // normalized T(r) in [0,1]
@group(0) @binding(3) var<storage, read> colorLUT: array<vec4<f32>>; // linear-sRGB blackbody color

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

// linearly-interpolated lookup into a 1-D storage-buffer LUT (portable; no float-filterable feature)
fn sampleTemp(r: f32) -> f32 {
  let n = arrayLength(&tempLUT);
  let u = clamp((r - U.rIn) / (U.rOut - U.rIn), 0.0, 1.0) * f32(n - 1u);
  let i0 = u32(floor(u)); let i1 = min(i0 + 1u, n - 1u);
  return mix(tempLUT[i0], tempLUT[i1], fract(u));
}
fn sampleColor(T_kelvin: f32) -> vec3<f32> {
  let n = arrayLength(&colorLUT);
  // color LUT spans [1000, 40000] K
  let u = clamp((T_kelvin - 1000.0) / (40000.0 - 1000.0), 0.0, 1.0) * f32(n - 1u);
  let i0 = u32(floor(u)); let i1 = min(i0 + 1u, n - 1u);
  return mix(colorLUT[i0].rgb, colorLUT[i1].rgb, fract(u));
}
// per-frame hash jitter for progressive anti-aliasing
fn hash2(p: vec2<u32>, frame: u32) -> vec2<f32> {
  let n = p.x * 1973u + p.y * 9277u + frame * 26699u;
  let h = (n ^ (n >> 15u)) * 2246822519u;
  let h2 = (h ^ (h >> 13u)) * 3266489917u;
  return vec2<f32>(f32(h & 0xffffu)/65535.0, f32(h2 & 0xffffu)/65535.0);
}

@compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u32(U.res.x) || gid.y >= u32(U.res.y)) { return; }
  let idx = gid.y * u32(U.res.x) + gid.x;
  let a = U.a; let i = U.incl;

  // pixel -> impact parameters (alpha,beta) in units of M, with sub-pixel jitter for AA
  let aspect = U.res.x / U.res.y;
  let jit = hash2(gid.xy, U.frame) - 0.5;
  let ndc = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5 + jit) / U.res * 2.0 - 1.0;
  let alpha = ndc.x * U.fovScale * aspect;
  let beta  = -ndc.y * U.fovScale;
  // Bardeen impact parameter -> conserved azimuthal angular momentum
  let xi = -alpha * sin(i);

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
        let Tn = sampleTemp(rHit);
        let Om = omegaKep(rHit, a);
        let gl = gLow(rHit, PI*0.5, a);
        let rad = -(gl[0] + 2.0*Om*gl[1] + Om*Om*gl[4]);
        let g = sqrt(max(0.0, rad)) / (1.0 - Om*xi);
        let Tobs = U.Tpeak * g * Tn;             // observed blackbody temperature
        let bright = pow(g * Tn, 4.0);           // bolometric beaming ∝ (gT)^4
        color = sampleColor(Tobs) * bright;
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
