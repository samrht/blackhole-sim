struct Uniforms {
  res: vec2<f32>, a: f32, incl: f32, rObs: f32, fovScale: f32, rIn: f32, rOut: f32,
  Tpeak: f32, exposure: f32, time: f32, frame: u32, reset: u32, maxSteps: u32,
  blend: f32, timeScale: f32, turbAmp: f32, breatheAmp: f32, nSpots: u32,
  jetStrength: f32, jetGamma: f32, jetLength: f32, jetKnots: f32,
  skyStrength: f32,
};
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> tempLUT: array<f32>;       // normalized T(r) in [0,1]
@group(0) @binding(3) var<storage, read> colorLUT: array<vec4<f32>>; // linear-sRGB blackbody color
@group(0) @binding(4) var<storage, read> hotspots: array<vec4<f32>>; // (r, psi, sigma, amp)
@group(0) @binding(5) var skyTex: texture_2d<f32>;
@group(0) @binding(6) var skySamp: sampler;

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

// Dave Hoskins hash33 -> vec3 in [0,1)
fn hash33(p3: vec3<f32>) -> vec3<f32> {
  var p = fract(p3 * vec3<f32>(0.1031, 0.1030, 0.0973));
  p += dot(p, p.yxz + 33.33);
  return fract((p.xxy + p.yxx) * p.zyx);
}

// Procedural starfield sampled along an escaped ray's asymptotic direction. Because the direction
// has been bent by the geometry, the background appears gravitationally lensed (warped/magnified
// near the shadow) — a physically real effect, and the reason the void reads as deep space.
fn starfield(dir: vec3<f32>) -> vec3<f32> {
  // deep, near-black void with a barely-there cool nebular gradient (keeps space from
  // reading as flat #000 while staying dark enough for the disk to dominate the frame)
  let neb = 0.5 + 0.5 * dir.y;
  var col = mix(vec3<f32>(0.0016, 0.0022, 0.0050), vec3<f32>(0.0030, 0.0024, 0.0042), neb);
  // sparse, crisp stars across three density octaves; rarer stars burn brighter
  for (var k = 0u; k < 3u; k++) {
    let scale = 95.0 * pow(1.7, f32(k));
    let p = dir * scale;
    let cell = floor(p);
    let h = hash33(cell);
    let thresh = 0.989;
    if (h.x > thresh) {
      let center = cell + 0.5 + (h.yzx - 0.5) * 0.6;
      let d = length(p - center);
      let mag = (h.x - thresh) / (1.0 - thresh);            // 0..1 rarity -> brightness
      let bright = smoothstep(0.45, 0.0, d) * (0.35 + 2.2 * mag * mag);
      let tint = mix(vec3<f32>(0.58, 0.72, 1.0), vec3<f32>(1.0, 0.83, 0.60), h.y); // blue..warm
      col += tint * bright;
    }
  }
  return col;
}

// --- Baked sky panorama (equirectangular; twin of src/render/skymap.ts) ------------------------
const SKY_TEXW = 4096.0;
fn tiltDir(d: vec3<f32>) -> vec3<f32> {   // R_SKY = Rz(30°)·Rx(60°), must match skymap.ts
  return vec3<f32>(
    0.866025 * d.x - 0.25 * d.y + 0.433013 * d.z,
    0.5 * d.x + 0.433013 * d.y - 0.75 * d.z,
    0.866025 * d.y + 0.5 * d.z);
}
fn skySample(dir: vec3<f32>) -> vec3<f32> {
  let d = tiltDir(dir);
  let u = atan2(d.z, d.x) * (0.5 / PI) + 0.5;
  let v = acos(clamp(d.y, -1.0, 1.0)) * (1.0 / PI);
  // Compute shaders have no implicit derivatives, so pick LOD analytically from the far-field
  // angular footprint of one pixel. Mip chain + temporal AA absorb residual minification aliasing.
  let anglePerPixel = 2.0 * U.fovScale / U.res.y / U.rObs;
  let lod = max(0.0, log2(SKY_TEXW * anglePerPixel / (2.0 * PI)));
  return textureSampleLevel(skyTex, skySamp, vec2<f32>(u, v), lod).rgb;
}

// --- Tier 2A emission field (WGSL twin of src/physics/emission.ts) -----------------------------
fn ihashE(ix: i32, iy: i32) -> f32 {
  var n = u32(ix) * 1973u + u32(iy) * 9277u;
  n = (n ^ (n >> 15u)) * 2246822519u;
  n = (n ^ (n >> 13u)) * 3266489917u;
  return f32(n & 0xffffffu) / f32(0xffffffu);
}
fn smoothE(t: f32) -> f32 { return t * t * (3.0 - 2.0 * t); }
fn vnoiseE(x: f32, y: f32) -> f32 {
  let ix = i32(floor(x)); let iy = i32(floor(y));
  let fx = smoothE(x - floor(x)); let fy = smoothE(y - floor(y));
  let a00 = ihashE(ix, iy); let a10 = ihashE(ix + 1, iy);
  let a01 = ihashE(ix, iy + 1); let a11 = ihashE(ix + 1, iy + 1);
  return (a00 * (1.0 - fx) + a10 * fx) * (1.0 - fy) + (a01 * (1.0 - fx) + a11 * fx) * fy;
}
fn turbulenceE(logR: f32, psi: f32) -> f32 {
  var sum = 0.0; var amp = 0.5; var freq = 1.0;
  for (var o = 0u; o < 3u; o++) { sum += amp * vnoiseE(logR * freq, psi * freq); amp *= 0.5; freq *= 2.0; }
  return sum;
}
fn hotspotFieldE(rHit: f32, psi: f32) -> f32 {
  var s = 0.0;
  for (var k = 0u; k < U.nSpots; k++) {
    let sp = hotspots[k];
    let dr = rHit - sp.x;
    var dpsi = psi - sp.y;
    dpsi = dpsi - 2.0 * PI * round(dpsi / (2.0 * PI));
    let arc = sp.x * dpsi;
    s += sp.w * exp(-(dr * dr + arc * arc) / (2.0 * sp.z * sp.z));
  }
  return s;
}
fn emissionFieldE(rHit: f32, psi: f32) -> f32 {
  let turb = 1.0 + U.turbAmp * (turbulenceE(log(rHit), psi) - 0.5) * 2.0;
  let breathe = 1.0 + U.breatheAmp * sin(2.0 * PI * U.time / 2000.0);
  return max(0.0, turb * breathe + hotspotFieldE(rHit, psi));
}

// --- Tier 2B jet (WGSL twin of src/physics/jet.ts) --------------------------------------------
const JET_QPEAK = 0.8;   const JET_WWALL = 0.22;
const JET_RHO0  = 0.6;   const JET_SLOPE = 0.7;
const JET_ZBASE = 2.0;   const JET_KZ    = 0.35;  const JET_VKNOT = 6.0;
const JET_PBEAM = 3.5;   const JET_TURB  = 0.35;  const JET_SEED  = 17.0;
const JET_GAIN  = 0.03;  const JET_CEIL  = 4.0;
const JET_TINT  = vec3<f32>(0.55, 0.78, 1.0);

fn smoothstepJ(a: f32, b: f32, x: f32) -> f32 {
  let t = clamp((x - a) / (b - a), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}
fn funnelEdgeJ(z: f32) -> f32 { return JET_RHO0 + JET_SLOPE * sqrt(abs(z)); }
fn wallJ(rho: f32, z: f32) -> f32 {
  let q = rho / funnelEdgeJ(z);
  if (q > 1.2) { return 0.0; }
  let d = q - JET_QPEAK;
  return exp(-(d * d) / (2.0 * JET_WWALL * JET_WWALL));
}
fn lengthFalloffJ(z: f32, zMax: f32) -> f32 {
  let az = abs(z);
  let fadeIn = smoothstepJ(JET_ZBASE, JET_ZBASE + 2.0, az);
  let fadeOut = 1.0 - smoothstepJ(zMax * 0.7, zMax, az);
  let decay = JET_ZBASE / max(az, JET_ZBASE);
  return fadeIn * fadeOut * decay;
}
fn knotsJ(z: f32, t: f32) -> f32 {
  let phase = JET_KZ * abs(z) - JET_VKNOT * t * U.timeScale;
  return 1.0 + U.jetKnots * (vnoiseE(phase, JET_SEED) - 0.5) * 2.0;
}
fn boostJ(mu: f32, gamma: f32) -> f32 {
  let beta = sqrt(max(0.0, 1.0 - 1.0 / (gamma * gamma)));
  let delta = 1.0 / (gamma * (1.0 - beta * mu));
  return pow(delta, JET_PBEAM);
}
// scalar emissivity (no beaming); exactly 0 when jet off / below zBase / beyond zMax / outside wall
fn jetEmissionJ(r: f32, th: f32, t: f32) -> f32 {
  if (U.jetStrength == 0.0) { return 0.0; }
  let z = r * cos(th);
  let az = abs(z);
  if (az < JET_ZBASE || az > U.jetLength) { return 0.0; }
  let rho = r * sin(th);
  let w = wallJ(rho, z);
  if (w <= 0.0) { return 0.0; }
  let turb = 1.0 + JET_TURB * (vnoiseE(log(1.0 + rho), JET_KZ * z) - 0.5) * 2.0;
  return max(0.0, w * lengthFalloffJ(z, U.jetLength) * knotsJ(z, t) * turb);
}
fn cartOf(x: vec4<f32>) -> vec3<f32> {
  let r = x.y; let th = x.z; let ph = x.w; let s = sin(th);
  return vec3<f32>(r * s * cos(ph), r * s * sin(ph), r * cos(th));
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
  var jetAccum = vec3<f32>(0.0); // optically-thin jet emission integrated along the ray

  for (var step = 0u; step < U.maxSteps; step++) {
    // dl > 0 with p_r < 0 integrates INWARD (matches the geodesic capture test; a negative dl
    // would march rays outward -> black screen).
    let r = s.x.y;
    // distance-adaptive step: fine in the strong-field/disk region, large strides through the
    // near-flat far field (curvature ~M/r^3 is negligible there) so we don't burn thousands of
    // steps just travelling in from the distant observer at r0.
    var dl = clamp(0.02 * (r - rh), 0.002, 0.5);
    if (r > U.rOut * 1.5) { dl = clamp(0.04 * r, 0.6, 6.0); }
    let sNew = rk4(s, a, dl);

    // Optically-thin jet: integrate emissivity * relativistic beaming along the ray. The disk
    // hit below still `break`s (opaque), so jet segments behind the disk/horizon are occluded.
    if (U.jetStrength > 0.0) {
      let jz = s.x.y * cos(s.x.z);
      let e = jetEmissionJ(s.x.y, s.x.z, U.time);
      let dvec = cartOf(sNew.x) - cartOf(s.x);                  // inward step (camera -> hole)
      // guard normalize() against a zero-length step: a NaN mu here would poison the EMA accum
      // buffer permanently (mix(accum, NaN, blend) stays NaN). Effectively unreachable, cheap insurance.
      if (e > 0.0 && dot(dvec, dvec) > 1e-12) {
        let marchDir = normalize(dvec);
        let axisSign = select(-1.0, 1.0, jz >= 0.0);
        let mu = -axisSign * marchDir.z;                        // emitter outflow toward observer
        jetAccum += JET_TINT * (e * JET_GAIN) * boostJ(mu, U.jetGamma) * dl;
      }
    }

    // disk crossing: equatorial plane th = PI/2 (take the first hit -> optically-thick top surface)
    let f0 = s.x.z - PI*0.5; let f1 = sNew.x.z - PI*0.5;
    if (f0 * f1 < 0.0) {
      let frac = f0 / (f0 - f1);
      let rHit = mix(s.x.y, sNew.x.y, frac);
      if (rHit >= U.rIn && rHit <= U.rOut) {
        let Tn = sampleTemp(rHit);
        let Om = omegaKep(rHit, a);
        let gl = gLow(rHit, PI*0.5, a);
        let rad = -(gl[0] + 2.0*Om*gl[1] + Om*Om*gl[4]);
        let g = sqrt(max(0.0, rad)) / (1.0 - Om*xi); // Doppler + gravitational redshift factor
        let Tobs = U.Tpeak * g * Tn;                 // observed blackbody temperature
        let phiHit = mix(s.x.w, sNew.x.w, frac);     // azimuth of the emitting matter
        let psi = phiHit - Om * U.time * U.timeScale;// co-rotating pattern phase
        let E = emissionFieldE(rHit, psi);           // time-varying brightness (==1 when features off)
        color = sampleColor(Tobs) * pow(g * Tn, 4.0) * E;
        break;
      }
    }
    s = sNew;
    if (s.x.y <= rh * 1.001) { color = vec3(0.0); break; }   // captured -> shadow
    if (s.x.y > r0 * 1.2) {
      // escaped: sample the background along the ray's (bent) asymptotic direction.
      // The deflected direction makes the starfield appear gravitationally lensed —
      // warped and magnified into a ring around the shadow.
      let th = s.x.z; let ph = s.x.w;
      let dir = normalize(vec3<f32>(sin(th)*cos(ph), sin(th)*sin(ph), cos(th)));
      // Baked panorama crossfaded over the procedural starfield by skyStrength (0 => unchanged).
      let mixT = clamp(U.skyStrength, 0.0, 1.0);
      color = mix(starfield(dir), skySample(dir) * U.skyStrength, mixT);
      break;
    }
  }

  // Temporal EMA: blend = 1/(frame+1) reproduces the Tier-1 running mean when static; a fixed
  // blend (~0.15) tracks an animating scene. blend==1 (first frame after a reset) clears cleanly.
  // Additive optically-thin jet on top of whatever the ray terminated on (disk/starfield/shadow).
  let composited = color + U.jetStrength * min(jetAccum, vec3<f32>(JET_CEIL));
  accum[idx] = vec4<f32>(mix(accum[idx].rgb, composited, U.blend), 1.0);
}
