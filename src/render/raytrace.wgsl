struct Uniforms {
  res: vec2<f32>, a: f32, incl: f32, rObs: f32, fovScale: f32, rIn: f32, rOut: f32,
  Tpeak: f32, exposure: f32, time: f32, frame: u32, reset: u32, maxSteps: u32,
  blend: f32, timeScale: f32, turbAmp: f32, breatheAmp: f32, nSpots: u32,
  jetStrength: f32, jetGamma: f32, jetLength: f32, fluxVar: f32,
  skyStrength: f32, outW: f32, outH: f32,
  jitterMode: u32, setIndex: u32, rowStart: u32, rowEnd: u32,
  lumNorm: f32,
  lightDelay: f32,
  jetB0: f32, jetQ0: f32, rgCm: f32,   // synchrotron jet (CPU: jetUniforms in synchrotron.ts)
  timeEpoch: f32,                      // clock epoch: absolute time = timeEpoch + time (sim-clock.ts)
  band: f32, hotFlow: f32, flowN0: f32, // 1.3 mm view: band 0/1, hot flow 0/1, its density scale (cm^-3)
  panX: f32, panY: f32,                 // zoom toward the cursor: the view's image-plane offset (M; uniforms.ts)
};
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> tempLUT: array<f32>;       // normalized T(r) in [0,1]
@group(0) @binding(3) var<storage, read> colorLUT: array<vec4<f32>>; // visible-band blackbody radiance (log T)
@group(0) @binding(4) var<storage, read> hotspots: array<vec4<f32>>; // (r, psi, sigma, amp)
@group(0) @binding(5) var skyTex: texture_2d<f32>;
@group(0) @binding(6) var skySamp: sampler;
// Geodesic cache (spec 2026-10-01). One Entry per pixel per jitter set; bookmarks are sparse.
struct Entry { word: u32, p0: f32, p1: f32, p2: f32 };        // word = kind | bookmark index << 2; DISK p = (rHit, phiHit, delay)
struct Bookmark { x: vec4<f32>, p: vec4<f32>, nJet: u32 };      // 48 bytes (vec4 alignment)
@group(0) @binding(7) var<storage, read_write> entries: array<Entry>;
@group(0) @binding(8) var<storage, read_write> bookmarks: array<Bookmark>;
@group(0) @binding(9) var<storage, read_write> bmCount: atomic<u32>;
const BM_NONE = 0x3fffffffu; // twin: BM_NONE in cache-plan.ts

// linearly-interpolated lookup into a 1-D storage-buffer LUT (portable; no float-filterable feature)
fn sampleTemp(r: f32) -> f32 {
  let n = arrayLength(&tempLUT);
  let u = clamp((r - U.rIn) / (U.rOut - U.rIn), 0.0, 1.0) * f32(n - 1u);
  let i0 = u32(floor(u)); let i1 = min(i0 + 1u, n - 1u);
  return mix(tempLUT[i0], tempLUT[i1], fract(u));
}
// Visible-band radiance LUT, log-spaced in T over [VIS_TMIN, VIS_TMAX] (lookups.ts), relative to a
// 1e4 K blackbody's luminance. Twin: sampleVisibleLUT in lookups.ts.
const VIS_LN_TMIN = 4.605170186;   // ln(100)
const VIS_LN_TMAX = 20.723265837;  // ln(1e9)
fn sampleColor(T_kelvin: f32) -> vec3<f32> {
  let n = arrayLength(&colorLUT);
  let u = clamp((log(max(T_kelvin, 1.0)) - VIS_LN_TMIN) / (VIS_LN_TMAX - VIS_LN_TMIN), 0.0, 1.0) * f32(n - 1u);
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

// Fixed rotated-grid jitter sets of the geodesic cache (spec 2026-10-01 3.2).
// Twin: JITTER in src/render/cache-plan.ts.
fn fixedJitter(k: u32) -> vec2<f32> {
  switch (k & 3u) {
    case 0u: { return vec2<f32>(-0.125, -0.375); }
    case 1u: { return vec2<f32>(0.375, -0.125); }
    case 2u: { return vec2<f32>(0.125, 0.375); }
    default: { return vec2<f32>(-0.375, 0.125); }
  }
}
fn pixelJitter(p: vec2<u32>) -> vec2<f32> {
  if (U.jitterMode == 1u) { return fixedJitter(U.setIndex); }
  return hash2(p, U.frame) - 0.5;
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
// How the disk is shaded (twin: diskShadeFactors in src/physics/emission.ts). MRI turbulence modulates
// the local flux F = exp(sigma g - sigma^2 / 2) (sigma = U.turbAmp; mean 1, so the bolometric light is
// conserved), which an optically thick disk radiates as a blackbody at T x F^(1/4): x = temperature scale.
// The illustrative breathing and hot spots (co-rotating phase psi) stay a grey factor: y. (1, 1) with all off.
// tEmit is the absolute emission time; tRel the same time less the clock epoch (full precision per pixel).
fn diskShadeFactorsE(rHit: f32, phiHit: f32, psi: f32, tEmit: f32, tRel: f32, a: f32) -> vec2<f32> {
  let s = U.turbAmp;
  var tempScale = 1.0;
  if (s > 0.0) { tempScale = exp(0.25 * (s * turbulenceFieldE(rHit, phiHit, U.timeEpoch, tRel, a) - 0.5 * s * s)); }
  let breathe = 1.0 + U.breatheAmp * sin(2.0 * PI * tEmit / 2000.0);
  return vec2<f32>(tempScale, max(0.0, breathe + hotspotFieldE(rHit, psi)));
}

// --- Tier 2B synchrotron jet ----------------------------------------------------------------
// The jet's physics (density shape, plasma shift, coefficients, slab transfer) lives in
// emission-shared.wgsl (sole copy, also prepended by the ?parity route); the renderer integrates it
// along the ray and adds the on/off switch and its live settings.
fn cartOf(x: vec4<f32>) -> vec3<f32> {
  let r = x.y; let th = x.z; let ph = x.w; let s = sin(th);
  return vec3<f32>(r * s * cos(ph), r * s * sin(ph), r * cos(th));
}

// Asymptotic sky direction of an escaping ray. Built from the propagation direction
// dx^mu/dl = g^{mu nu} p_nu, NOT the position unit vector: at the r>1.2*rObs cutoff those differ
// by ~b/r (up to ~16 mrad, ~11 panorama texels), which displaces every background star radially.
// gUp indices: 0=tt, 1=tphi, 2=rr, 3=thth, 4=phph. State packs momenta in s.p = (pt,pr,pth,pphi).
fn skyDir(s: State, a: f32) -> vec3<f32> {
  let r = s.x.y; let th = s.x.z; let ph = s.x.w;
  let g = gUp(r, th, a);
  let dr  = g[2] * s.p.y;
  let dth = g[3] * s.p.z;
  let dph = g[1] * s.p.x + g[4] * s.p.w;
  let st = sin(th); let ct = cos(th); let sp = sin(ph); let cp = cos(ph);
  return normalize(vec3<f32>(
    dr * st * cp + r * ct * cp * dth - r * st * sp * dph,
    dr * st * sp + r * ct * sp * dth + r * st * cp * dph,
    dr * ct - r * st * dth));
}

// --- Shared pieces of the trace. `main` (live), and the geodesic cache's `build` and `shade` passes
// (spec 2026-10-01) call these, so a cached frame runs the same maths as a live one. -------------

const KIND_SHADOW = 0u; const KIND_DISK = 1u; const KIND_SKY = 2u; const KIND_LIVE = 3u;

// Background along an escaped ray's bent asymptotic direction: the baked panorama crossfaded over
// the procedural starfield by skyStrength (0 => procedural only).
fn skyColor(dir: vec3<f32>) -> vec3<f32> {
  let mixT = clamp(U.skyStrength, 0.0, 1.0);
  // Fully panorama: skip the starfield. mix(a, b, 1) = a + (b - a) is not bitwise b, so a hidden
  // star still leaked one rounding step of its brightness -- and the starfield's hash is not
  // reproducible across separately compiled entry points (geodesic cache, spec 2026-10-01).
  if (mixT >= 1.0) { return skySample(dir) * U.skyStrength; }
  return mix(starfield(dir), skySample(dir) * U.skyStrength, mixT);
}

// Doppler + gravitational redshift factor of the disk matter at rHit seen along a ray with xi.
fn diskG(rHit: f32, xi: f32, a: f32) -> f32 {
  let Om = omegaKep(rHit, a);
  let gl = gLow(rHit, PI*0.5, a);
  let rad = -(gl[0] + 2.0*Om*gl[1] + Om*Om*gl[4]);
  return sqrt(max(0.0, rad)) / (1.0 - Om*xi);
}

// Observed disk colour at a hit: the only time dependence is the co-rotating pattern phase psi.
fn shadeDisk(rHit: f32, phiHit: f32, g: f32, a: f32, tRel: f32) -> vec3<f32> {
  let Tn = sampleTemp(rHit);
  let Om = omegaKep(rHit, a);
  let tEmit = U.timeEpoch + tRel;              // absolute emission time
  let psi = phiHit - Om * tEmit;               // co-rotating pattern phase at emission
  let E = diskShadeFactorsE(rHit, phiHit, psi, tEmit, tRel, a); // (temperature scale, grey); (1, 1) when off
  let Tobs = U.Tpeak * g * Tn * E.x;           // observed blackbody temperature
  // Visible-band radiance of a blackbody at T_obs (I_nu / nu^3 is invariant, so a shifted blackbody
  // is a blackbody at g T): colour AND brightness a camera records, normalised so the disk's
  // rest-frame peak has luminance 1 (spec 2026-10-01 §2.3). Was the bolometric (g Tn)^4 law.
  return sampleColor(Tobs) * U.lumNorm * E.y;
}
// The same disk at 230 GHz (spec 2026-10-04 hot flow): Rayleigh-Jeans (h nu / k = 11 K), so its observed brightness
// temperature is the shifted temperature itself, T_b = g T (every channel; the mm composite reads .x).
fn shadeDiskMm(rHit: f32, phiHit: f32, g: f32, a: f32, tRel: f32) -> vec3<f32> {
  let tEmit = U.timeEpoch + tRel;
  let E = diskShadeFactorsE(rHit, phiHit, phiHit - omegaKep(rHit, a) * tEmit, tEmit, tRel, a);
  return vec3<f32>(U.Tpeak * g * sampleTemp(rHit) * E.x * E.y);
}
fn bandMm() -> bool { return U.band > 0.5; }
fn mmFlow() -> bool { return U.band > 0.5 && U.hotFlow > 0.5; }

// Light-travel delay (spec 2026-10-01): the backward ray starts at t = 0 and t decreases, so an
// emitter at coordinate time t_e is seen delay = -t_e - rObs later than a reference at the camera's
// distance (the constant rObs keeps values in tens of M). Emission time of what this pixel shows:
// U.time is the clock's remainder after the epoch U.timeEpoch (sim-clock.ts), so emitRel keeps full f32
// precision per pixel at any session length; every time-dependent term takes (U.timeEpoch, emitRel(...)).
fn emitRel(delay: f32) -> f32 { return U.time - U.lightDelay * delay; }

// The jet quadrature is a left Riemann sum along the ray with samples at most JET_DL apart,
// independent of the geodesic stride: a step longer than JET_DL is split into n = ceil(dl / JET_DL)
// samples along its Cartesian chord, each weighted dl / n (n = 1 is the original single sample at
// the step's start). The chord is the path to well under a sub-sample at these lengths. Before
// 2026-10-01 the jet was sampled once per step, so its accuracy followed the geodesic stride; the
// longer far strides (K_FAR) lost the emission of steps entering the jet's top from r > 1.5 rOut.
// 0.25 against a fine-step GPU reference (face-on jet scene, pixels > 5 % off: 116 per-step,
// 29 at 1.0, 7 at 0.5, 2 at 0.25). Steps whose chord stays outside the jet's bounding sphere are
// skipped exactly, so rays that never come near the jet pay nothing.
const JET_DL = 0.25;
const JET_NSUB_MAX = 32u;
// The jet sample points of one step, shared by jetStep (what it sums) and jetTouches (what the
// geodesic cache bookmarks), so the two cannot disagree. Sample k of n sits at k/n along the
// step's Cartesian chord; k = 0 is the step's start state itself.
fn jetSubCount(dl: f32) -> u32 { return clamp(u32(ceil(dl / JET_DL)), 1u, JET_NSUB_MAX); }
// Sample k of n: (r, theta, phi). k = 0 is the step's start state itself (its phi may be unwrapped; the
// filaments use it modulo 2 pi); interior samples take phi from the chord's Cartesian point.
fn jetSample(s: State, p0: vec3<f32>, dvec: vec3<f32>, k: u32, n: u32) -> vec3<f32> {
  if (k == 0u) { return vec3<f32>(s.x.y, s.x.z, s.x.w); }
  let p = p0 + dvec * (f32(k) / f32(n));
  let r = length(p);
  return vec3<f32>(r, acos(clamp(p.z / r, -1.0, 1.0)), atan2(p.y, p.x));
}
// Exact skip: the emitter lies inside |z| <= jetLength, rho <= JET_ENV_Q funnelEdge(jetLength), so a
// chord whose closest approach to the hole is beyond that bounding sphere sees no jet (its first
// sample, the only one when n = 1, is then outside too and jetShapeJ would return 0).
fn chordMisses(p0: vec3<f32>, dvec: vec3<f32>, R: f32) -> bool {
  let tc = clamp(-dot(p0, dvec) / dot(dvec, dvec), 0.0, 1.0);
  return length(p0 + dvec * tc) > R;
}
fn jetChordMisses(p0: vec3<f32>, dvec: vec3<f32>) -> bool {
  let fe = JET_ENV_Q * funnelEdgeJ(U.jetLength);
  return chordMisses(p0, dvec, sqrt(U.jetLength * U.jetLength + fe * fe));
}

// Synchrotron emission and absorption along one integrator step (spec 2026-10-02 2.4): each sub-sample
// is a uniform slab of plasma-frame path ds' = r_g D dl / n, with D = nu' / nu_obs from the photon
// momentum (interpolated across the step like the position and time) and the local flow.
fn jetStep(s: State, sNew: State, dl: f32, accIn: JetOut) -> JetOut {
  // a = 0: no Blandford-Znajek power, so the energy budget injects no electrons (q0 = 0): skip before
  // log(q0), which WGSL leaves undefined at 0. Without the coefficient table (1x1 placeholder) nothing either.
  if (U.jetQ0 <= 0.0 || !synchReady()) { return accIn; }
  let p0 = cartOf(s.x);
  let dvec = cartOf(sNew.x) - p0;                           // inward step (camera -> hole)
  if (dot(dvec, dvec) <= 1e-12) { return accIn; }
  if (jetChordMisses(p0, dvec)) { return accIn; }
  let n = jetSubCount(dl);
  var acc = accIn;
  for (var k = 0u; k < n; k++) {
    let q = jetSample(s, p0, dvec, k, n);
    let f = f32(k) / f32(n);
    let tS = select(s.x.x + (sNew.x.x - s.x.x) * f, s.x.x, k == 0u);
    let shape = jetShapeJ(q.x, q.y, q.z, U.timeEpoch, emitRel(-tS - U.rObs), U.jetLength, U.fluxVar, U.jetGamma, U.a);
    if (shape > 0.0) {
      let D = plasmaShiftJ(q.x, q.y, mix(s.p, sNew.p, f), U.a, jetGammaAt(q.x * cos(q.y), U.jetGamma));
      if (D > 1e-6) {                                       // never divide by D -> 0
        let so = synchSampleJ(select(JET_LNNU, vec3<f32>(HF_LNNU), bandMm()), q.x, q.y, D, U.a, U.jetB0, U.jetQ0, shape, U.jetGamma, U.rgCm);
        acc = jetSlabJ(acc, so.j, so.a, U.rgCm * D * dl / f32(n));
      }
    }
  }
  return acc;
}

// Hot flow at 230 GHz along one step (spec 2026-10-04): the jet's quadrature (samples on the step's chord, each a
// uniform slab of plasma-frame path r_g D dl / n), inside r < HF_RMAX and outside 1.01 r_+ (as the CPU calibration in
// hot-flow-image.ts). Samples are <= JET_DL apart inside 8 M and JET_DL r / 8 beyond, where the flow is faint (perf/mm-live:
// one sample per step out there, ~190 fewer per ray; flux changed 2e-6, worst pixel 2e-6 of the peak, sgra-mm/m87-mm). The flow is steady and axisymmetric: no time, no azimuth. One channel, carried as
// acc = (I_nu, tau) so visible frames hold two floats through the ray loop, not two vec3 (the jet's slab, scalar).
// Hotspots (spec 2026-10-04 mm hotspots) add their boosted coefficients to the same slab on live frames (hs).
fn flowStep(s: State, sNew: State, dl: f32, rh: f32, orb: vec3<f32>, accIn: vec2<f32>, hs: bool) -> vec2<f32> {
  let p0 = cartOf(s.x);
  let dvec = cartOf(sNew.x) - p0;
  if (dot(dvec, dvec) <= 1e-12 || chordMisses(p0, dvec, HF_RMAX)) { return accIn; }
  let n = clamp(u32(ceil(dl / (JET_DL * max(1.0, min(s.x.y, sNew.x.y) / 8.0)))), 1u, JET_NSUB_MAX);
  // Hotspot only on live frames (hs; the cache stores the steady flow), with the slider on (main.ts sends fluxVar 0 in mm
  // while no hotspot can be in view), and only on steps whose chord passes within HS_REACH of the hole.
  let hsOn = hs && U.fluxVar > 0.0 && !chordMisses(p0, dvec, HS_REACH);
  var acc = accIn;
  for (var k = 0u; k < n; k++) {
    let q = jetSample(s, p0, dvec, k, n);
    if (q.x >= HF_RMAX || q.x <= rh * 1.01) { continue; }
    let f = f32(k) / f32(n);
    let pk = mix(s.p, sNew.p, f);
    var jE = 0.0; var dTau = 0.0; // this sample's observed emission and optical depth (flow + hotspot, one slab)
    let D = flowShiftOrbJ(q.x, q.y, pk, U.a, orb);
    if (D > 1e-6) {
      let c = flowCoeffsJ(q.x, q.y, HF_LNNU + log(D), U.flowN0);
      if (c.x > 0.0) { let ds = U.rgCm * D * dl / f32(n); jE = c.x / (D * D * D) * ds; dTau = c.y * ds; }
    }
    if (hsOn && q.x < HS_REACH && abs(q.x * cos(q.y)) < HS_CUT * HS_SIGMA) {
      let tS = select(s.x.x + (sNew.x.x - s.x.x) * f, s.x.x, k == 0u);
      let st = hotspotStateJ(U.timeEpoch, emitRel(-tS - U.rObs), U.fluxVar, U.a);
      if (st.w > 0.0) {
        let b = st.z * hotspotBoostJ(q.x, q.y, q.z, st.x, st.y);
        if (b > 0.0) {
          let Dh = hotspotShiftJ(q.x, q.y, pk, U.a, 1.0 / (pow(st.x, 1.5) + U.a));
          if (Dh > 1e-6) {
            let ch = flowCoeffsJ(q.x, q.y, HF_LNNU + log(Dh), U.flowN0);
            let dsh = U.rgCm * Dh * dl / f32(n);
            jE += b * ch.x / (Dh * Dh * Dh) * dsh; dTau += b * ch.y * dsh;
          }
        }
      }
    }
    if (jE > 0.0 || dTau > 0.0) {
      let fac = select((1.0 - exp(-dTau)) / max(dTau, 1e-30), 1.0 - 0.5 * dTau, dTau < 1e-4);
      acc = vec2<f32>(acc.x + jE * fac * exp(-acc.y), min(acc.y + dTau, 1e30));
    }
  }
  return acc;
}

// Geometric (independent of the jet switch and brightness): can this step's jetStep be non-zero at
// SOME time? True iff one of its sample points is inside the jet envelope -- jetShapeJ is zero
// outside it. The geodesic cache bookmarks exactly these steps, so its replay sums the same samples.
fn jetTouches(s: State, sNew: State, dl: f32) -> bool {
  let p0 = cartOf(s.x);
  let dvec = cartOf(sNew.x) - p0;
  if (dot(dvec, dvec) <= 1e-12 || jetChordMisses(p0, dvec)) { return false; }
  let n = jetSubCount(dl);
  for (var k = 0u; k < n; k++) {
    let q = jetSample(s, p0, dvec, k, n);
    if (inJetEnvelope(q.x, q.y)) { return true; }
  }
  return false;
}

// Where jetShapeJ can be non-zero at SOME time (twin: inJetEnvelope in jet.ts).
// Geometric only, so the cache's bookmark never depends on the jet switch or brightness.
fn inJetEnvelope(r: f32, th: f32) -> bool {
  let z = r * cos(th);
  let az = abs(z);
  if (az < JET_ZBASE || az > U.jetLength) { return false; }
  return r * sin(th) / funnelEdgeJ(z) <= JET_ENV_Q;
}

// Pixel -> screen impact parameters (alpha, beta) in M with sub-pixel jitter. Shared by traceRay and
// the cache's shade pass (which recomputes g from them), so both see the same xi bit for bit.
fn pixelImpact(pix: vec2<u32>, jit: vec2<f32>) -> vec2<f32> {
  let aspect = U.res.x / U.res.y;
  let ndc = (vec2<f32>(f32(pix.x), f32(pix.y)) + 0.5 + jit) / U.res * 2.0 - 1.0;
  return vec2<f32>(ndc.x * U.fovScale * aspect + U.panX, -ndc.y * U.fovScale + U.panY);
}

struct TraceOut {
  color: vec3<f32>, jet: JetOut,
  flowI: f32,                      // mm hot-flow mode: the flow's observed I_nu (cgs) along the ray; else 0
  kind: u32, payload: vec3<f32>,   // DISK: (rHit, phiHit, delay); SKY: asymptotic direction; else 0
  hasBm: bool, bm: State, nJet: u32, // record only: first state inside the jet envelope, steps through the last
  resolved: bool,                    // false: the step budget ran out (kind then comes from the classifier)
};

fn traceRay(pix: vec2<u32>, jit: vec2<f32>, record: bool, hs: bool) -> TraceOut {
  var out: TraceOut;
  out.kind = KIND_SHADOW; out.payload = vec3<f32>(0.0); out.hasBm = false; out.nJet = 0u; out.resolved = true;
  let a = U.a; let i = U.incl;

  // pixel -> impact parameters (alpha,beta) in units of M, with sub-pixel jitter for AA
  let ab = pixelImpact(pix, jit);
  let alpha = ab.x; let beta = ab.y;
  // Bardeen impact parameters -> conserved (xi, eta). Sole copy lives in camera-shared.wgsl,
  // which gpu.ts prepends here and parity.browser.ts prepends to camera-parity.wgsl.
  let xe = cameraXiEta(alpha, beta, a, i);
  let xi = xe.x; let eta = xe.y;

  // initial state at (rObs, i, 0), E=1. Past-directed momentum -- see cameraMomenta().
  let r0 = U.rObs; let th0 = i;
  let gU = gUp(r0, th0, a);
  let p0 = cameraMomenta(xi, beta, gU[0], gU[1], gU[2], gU[3], gU[4]);
  var s = State(vec4<f32>(0.0, r0, th0, 0.0), p0);

  let rh = 1.0 + sqrt(max(0.0, 1.0 - a*a)); // horizon
  var color = vec3<f32>(0.0);
  var resolved = false; // set by each real termination; false => the step budget ran out
  var jet: JetOut; jet.I = vec3<f32>(0.0); jet.tau = vec3<f32>(0.0); // synchrotron light and optical depth along the ray
  var firstJ = 0u; var lastJ = 0u;
  var flow = vec2<f32>(0.0); // hot flow (mm): (I_nu, tau)
  let orb = iscoOrbitJ(a);   // the flow's ISCO orbit, once per ray

  for (var step = 0u; step < U.maxSteps; step++) {
    // dl > 0 with p_r < 0 integrates INWARD along the reversed worldline.
    let r = s.x.y;
    let far = r > U.rOut * 1.5; // same threshold as the far branch of stepSize: monitor OFF out there
    let st = stepGeodesic(s, a, stepSize(s, rh, U.rOut), select(H_TOL, H_TOL_FAR, far));
    // On retry exhaustion the smallest-step attempt is accepted and the ray proceeds; st.ok is
    // informational. Breaking to the (xi, eta) classifier here painted starfield over disk hits
    // for near-axis rays (it can only answer captured/escaped) -- a dark seam on the alpha = 0
    // column. A genuinely diverging ray still winds to budget exhaustion and reaches the
    // classifier below as before; a NaN state still ends in the `usable` guard.
    let dl = st.dl; let sNew = st.s;

    // Synchrotron jet: emission and absorption along the ray. The disk hit below still `break`s
    // (opaque), so jet segments behind the disk/horizon are occluded.
    if (U.jetStrength > 0.0) { jet = jetStep(s, sNew, dl, jet); }
    // Hot flow (mm): a volume emitter the ray crosses; it does not stop at the plane. Jet and flow are separate
    // accumulators (neither absorbs the other's light; the jet is faint at 1.3 mm).
    if (mmFlow()) { flow = flowStep(s, sNew, dl, rh, orb, flow, hs); }
    // Cache bookmark: the first step whose jet samples can see the envelope, through the last one.
    if (record && jetTouches(s, sNew, dl)) {
      if (!out.hasBm) { out.hasBm = true; out.bm = s; firstJ = step; }
      lastJ = step;
    }

    // disk crossing: equatorial plane th = PI/2 (take the first hit -> optically-thick top surface).
    // A step that moved theta by more than 0.5 rad is not a plane crossing (a legitimate near-field
    // step moves theta by <= ~0.07 rad): it is a diverged state that reflectAxis's single-crossing
    // reduction cannot have made sense of, and interpolating a disk hit from it would be garbage.
    let f0 = s.x.z - PI*0.5; let f1 = sNew.x.z - PI*0.5;
    if (!mmFlow() && f0 * f1 < 0.0 && abs(sNew.x.z - s.x.z) < 0.5) {
      let frac = f0 / (f0 - f1);
      let rHit = mix(s.x.y, sNew.x.y, frac);
      if (rHit >= U.rIn && rHit <= U.rOut) {
        let g = diskG(rHit, xi, a);
        let phiHit = mix(s.x.w, sNew.x.w, frac);     // azimuth of the emitting matter
        let delay = -mix(s.x.x, sNew.x.x, frac) - U.rObs;
        if (bandMm()) { color = shadeDiskMm(rHit, phiHit, g, a, emitRel(delay)); }
        else { color = shadeDisk(rHit, phiHit, g, a, emitRel(delay)); }
        out.kind = KIND_DISK; out.payload = vec3<f32>(rHit, phiHit, delay);
        resolved = true;
        break;
      }
    }
    s = sNew;
    // captured -> shadow. The margin must exceed one integration step (dl_min = 0.002 moves r by
    // ~4.2e-3 M at a=0.9), otherwise RK4's intermediate stages sample r < r_+, where Delta < 0
    // flips the metric signature and the state explodes to garbage that can pass the escape test
    // and paint starfield inside the shadow.
    if (s.x.y <= rh * 1.005) { color = vec3(0.0); resolved = true; break; }
    // 1.3 mm hot flow: a ray leaving r = HF_RMAX outward has collected all it will (no emitter beyond, no sky at 1.3 mm,
    // and outside Kerr's potential barrier, r <~ 4 M, an outgoing ray cannot turn back). Ending it here instead of at
    // 1.2 rObs gives the same image and skips the long outbound leg.
    if (mmFlow() && s.x.y > HF_RMAX && sNew.x.y > r) { out.kind = KIND_SKY; resolved = true; break; }
    if (s.x.y > r0 * 1.2) {
      // escaped: sample the background along the ray's (bent) asymptotic direction.
      // The deflected direction makes the starfield appear gravitationally lensed —
      // warped and magnified into a ring around the shadow.
      let dir = skyDir(s, a);
      if (!bandMm()) { color = skyColor(dir); } // no sky at 1.3 mm (the CMB's 2.7 K is nothing here)
      out.kind = KIND_SKY; out.payload = dir;
      resolved = true;
      break;
    }
  }

  // Budget exhausted without a real termination. Previously these rays kept color = vec3(0) and so
  // rendered as shadow -- a step-budget artifact that swallowed the n=1 photon subring. Classify
  // them from their conserved (xi, eta) instead: the sign of p_r at an arbitrary cutoff is
  // effectively random for a winding ray and would produce salt-and-pepper noise.
  out.resolved = resolved;
  if (!resolved) {
    let th = s.x.z; let ph = s.x.w;
    // RK4 can diverge for rays near the critical impact parameter, leaving s.x non-finite when the
    // step budget runs out. Reading that state into `dir` would emit a NaN colour into the EMA
    // accumulator below, and bloom.wgsl's separable blur would then smear that single NaN pixel
    // across a whole neighbourhood. NaN comparisons are always false, so this range test rejects
    // non-finite th/ph without needing a bitcast/isnan helper, and we just treat the ray as captured.
    let usable = th > -1e6 && th < 1e6 && ph > -1e6 && ph < 1e6;
    if (classifyCaptured(xi, eta, a) || !usable) {
      color = vec3<f32>(0.0);
    } else {
      let dir = skyDir(s, a);
      if (!bandMm()) { color = skyColor(dir); }
      out.kind = KIND_SKY; out.payload = dir;
    }
  }
  out.color = color; out.jet = jet; out.flowI = flow.x;
  if (out.hasBm) { out.nJet = lastJ - firstJ + 1u; }
  return out;
}

// Temporal EMA: blend = 1/(frame+1) reproduces the Tier-1 running mean when static; a fixed
// blend (~0.15) tracks an animating scene. blend==1 (first frame after a reset) clears cleanly.
// The jet over whatever the ray terminated on (disk/starfield/shadow).
fn storeComposite(idx: u32, color: vec3<f32>, jet: JetOut, flowI: f32) {
  // Light from behind the jet (disk, sky) is absorbed: R, G, B by the 650 / 550 / 450 nm optical depths.
  // The jet's own light enters in the disk's units (band matrix) times the disk's lumNorm (spec 2.5);
  // clamped at 0 like blackbodyVisibleRGB (a pure power law can sit just outside the sRGB gamut).
  // 1.3 mm: brightness temperature (K) in every channel: the disk's T_b behind the jet, plus K_TB (jet + flow) I_nu.
  var raw = color * exp(-vec3<f32>(jet.tau.z, jet.tau.y, jet.tau.x)) + U.lumNorm * max(JET_BAND_M * jet.I, vec3<f32>(0.0));
  if (bandMm()) { raw = vec3<f32>(color.x * exp(-jet.tau.x) + HF_KTB * (max(jet.I.x, 0.0) + flowI)); }
  // Single choke point: nothing non-finite may enter accum. The in-loop escape branch above reads
  // s.x without the `usable` guard, so a diverged RK4 ray (r = +inf compares true, th/ph NaN) can
  // still produce a NaN colour there. A NaN in accum is PERMANENT -- mix(NaN, ..) stays NaN for
  // every later frame -- and bloom.wgsl's separable blur amplifies that one pixel into a whole
  // block. NaN compares false to everything, so this range test rejects NaN and both infinities
  // without a bitcast, and is exactly inert for finite values.
  let finite = all(raw > vec3<f32>(-1e30)) && all(raw < vec3<f32>(1e30));
  let composited = select(vec3<f32>(0.0), raw, finite);
  accum[idx] = vec4<f32>(mix(accum[idx].rgb, composited, U.blend), 1.0);
}

@compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u32(U.res.x) || gid.y >= u32(U.res.y)) { return; }
  let idx = gid.y * u32(U.res.x) + gid.x;
  let t = traceRay(gid.xy, pixelJitter(gid.xy), false, true);
  storeComposite(idx, t.color, t.jet, t.flowI);
}

// Re-integrate a bookmarked jet stretch: the same steps, in the same order, as traceRay took from
// the bookmark through the last step whose jet samples touched the envelope (jetTouches).
fn replayJet(bm: State, nJet: u32) -> JetOut {
  let a = U.a;
  let rh = 1.0 + sqrt(max(0.0, 1.0 - a*a));
  var s = bm; var acc: JetOut; acc.I = vec3<f32>(0.0); acc.tau = vec3<f32>(0.0);
  for (var k = 0u; k < nJet; k++) {
    let far = s.x.y > U.rOut * 1.5;
    let st = stepGeodesic(s, a, stepSize(s, rh, U.rOut), select(H_TOL, H_TOL_FAR, far));
    acc = jetStep(s, st.s, st.dl, acc);
    s = st.s;
  }
  return acc;
}

// Trace rows [rowStart, rowEnd) of jitter set setIndex at full resolution and record them.
@compute @workgroup_size(8,8) fn build(@builtin(global_invocation_id) gid: vec3<u32>) {
  let y = U.rowStart + gid.y;
  if (gid.x >= u32(U.res.x) || y >= U.rowEnd || y >= u32(U.res.y)) { return; }
  let idx = y * u32(U.res.x) + gid.x;
  // No jet bookmarks at 1.3 mm: the jet is not drawn there (spec 2.5), and recording them overflowed the bookmark buffer
  // for face-on M87* (half its pixels fell back to live traces in every cached frame).
  let t = traceRay(vec2<u32>(gid.x, y), fixedJitter(U.setIndex), !bandMm(), false);
  var word = t.kind | (BM_NONE << 2u);
  if (t.hasBm) {
    let b = atomicAdd(&bmCount, 1u);
    if (b < arrayLength(&bookmarks)) {
      bookmarks[b] = Bookmark(t.bm.x, t.bm.p, t.nJet);
      word = t.kind | (b << 2u);
    } else {
      word = KIND_LIVE | (BM_NONE << 2u); // bookmark buffer full: shade traces this pixel in full
    }
  }
  // mm hot-flow mode: no disk kind can occur and the sky is dark, so the entry carries the pixel's steady flow intensity.
  if (mmFlow()) { entries[idx] = Entry(word, t.flowI, 0.0, 0.0); return; }
  entries[idx] = Entry(word, t.payload.x, t.payload.y, t.payload.z);
}

// ?shadow's critical-curve gate (src/test/shadow-gate.ts): every pixel's termination for jitter set setIndex,
// word = kind | 4 when the step budget ran out -- the renderer then takes the kind from the analytic
// classifier, which the gate compares against, so those rays must be visible to it. Validation only.
@compute @workgroup_size(8,8) fn audit(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u32(U.res.x) || gid.y >= u32(U.res.y)) { return; }
  let t = traceRay(gid.xy, fixedJitter(U.setIndex), false, false);
  entries[gid.y * u32(U.res.x) + gid.x] = Entry(t.kind | select(4u, 0u, t.resolved), 0.0, 0.0, 0.0);
}

// Cached frame: re-colour from the record of jitter set setIndex; replay the jet stretch.
@compute @workgroup_size(8,8) fn shade(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u32(U.res.x) || gid.y >= u32(U.res.y)) { return; }
  let idx = gid.y * u32(U.res.x) + gid.x;
  let e = entries[idx];
  let kind = e.word & 3u;
  if (kind == KIND_LIVE) {
    let t = traceRay(gid.xy, fixedJitter(U.setIndex), false, false);
    storeComposite(idx, t.color, t.jet, t.flowI);
    return;
  }
  var color = vec3<f32>(0.0);
  var flowI = 0.0;
  if (mmFlow()) { flowI = e.p0; }
  else if (kind == KIND_DISK) {
    // g is recomputed (the entry's third slot holds the delay): same helper and camera fragment as
    // traceRay, so xi and g match the live trace.
    let ab = pixelImpact(gid.xy, fixedJitter(U.setIndex));
    let g = diskG(e.p0, cameraXiEta(ab.x, ab.y, U.a, U.incl).x, U.a);
    if (bandMm()) { color = shadeDiskMm(e.p0, e.p1, g, U.a, emitRel(e.p2)); }
    else { color = shadeDisk(e.p0, e.p1, g, U.a, emitRel(e.p2)); }
  }
  else if (kind == KIND_SKY && !bandMm()) { color = skyColor(vec3<f32>(e.p0, e.p1, e.p2)); }
  var jet: JetOut; jet.I = vec3<f32>(0.0); jet.tau = vec3<f32>(0.0);
  let bi = e.word >> 2u;
  if (U.jetStrength > 0.0 && bi != BM_NONE) {
    let b = bookmarks[bi];
    jet = replayJet(State(b.x, b.p), b.nJet);
  }
  storeComposite(idx, color, jet, flowI);
}
