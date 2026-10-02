// Sole WGSL copy of the disk turbulence and the synchrotron jet (twins: src/physics/emission.ts,
// src/physics/jet.ts, src/physics/synchrotron.ts). Prepended verbatim by gpu.ts (renderer) and
// parity.browser.ts (?parity: turb and jet cases), so ?parity checks the shipped bytes. No uniforms,
// no bindings; the jet code uses State/gUp from integrator-shared.wgsl, which both prepend too.

// --- Tier 2A turbulence noise -----------------------------------------------------------------
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
const TWO_PI_E = 6.283185307179586;
// Value noise periodic in y with period n cells (twin: vnoiseRing in emission.ts).
fn vnoiseRingE(x: f32, y: f32, n: i32) -> f32 {
  let ix = i32(floor(x)); let iy = i32(floor(y));
  let fx = smoothE(x - floor(x)); let fy = smoothE(y - floor(y));
  let j0 = ((iy % n) + n) % n; let j1 = (j0 + 1) % n;
  let a00 = ihashE(ix, j0); let a10 = ihashE(ix + 1, j0);
  let a01 = ihashE(ix, j1); let a11 = ihashE(ix + 1, j1);
  return (a00 * (1.0 - fx) + a10 * fx) * (1.0 - fy) + (a01 * (1.0 - fx) + a11 * fx) * fy;
}
// 2 pi-periodic in psi (an integer number of cells per octave around the ring): a ray's hit
// azimuth is a continuous geodesic coordinate, and rays passing either side of the hole reach the
// far side of the disk ~2 pi apart; non-periodic noise drew a seam above the shadow there.
fn turbulenceE(logR: f32, psi: f32) -> f32 {
  let turns = fract(psi / TWO_PI_E);
  var sum = 0.0; var amp = 0.5; var freq = 1.0; var norm = 0.0;
  for (var o = 0u; o < 3u; o++) {
    let n = i32(round(TWO_PI_E * freq));
    sum += amp * vnoiseRingE(logR * freq, turns * f32(n), n); norm += amp; amp *= 0.5; freq *= 2.0;
  }
  return sum / norm;
}

// --- Tier 2B jet geometry -------------------------------------------------------------------
const JET_QPEAK = 0.8;   const JET_WWALL = 0.22;
const JET_RHO0  = 0.6;   const JET_SLOPE = 0.7;
const JET_ZBASE = 2.0;   const JET_KZ    = 0.35;
const JET_TURB  = 0.35;  const JET_SEED  = 17.0;

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
fn knotsJ(z: f32, t: f32, gamma: f32, knotAmp: f32) -> f32 {
  // Knots move with the flow at beta(Gamma) < c (twin: knots in jet.ts); t is the emission time.
  let beta = sqrt(max(0.0, 1.0 - 1.0 / (gamma * gamma)));
  let phase = JET_KZ * (abs(z) - beta * t);
  return 1.0 + knotAmp * (vnoiseE(phase, JET_SEED) - 0.5) * 2.0;
}
// --- Tier 2B synchrotron (spec 2026-10-02; twin: src/physics/synchrotron.ts). Constants for p = 2.4,
// gamma_min = 10; tests/synchrotron.test.ts checks every literal against the CPU twin. -------------
const SYN_P = 2.4;
const SYN_LNCJ = -42.13253082;
const SYN_LNCA = 28.35789673;
const SYN_LNNUMIN0 = 19.85549701;
const JET_LNNU = vec3<f32>(34.13261924, 33.93194855, 33.76489446);   // ln nu at 450, 550, 650 nm
// I_nu (cgs) per band -> linear sRGB in the disk's units (before lumNorm); column b = band b.
const JET_BAND_M = mat3x3<f32>(-57.85676090, -199.6091852, 8197.725079, 563.0803652, 5602.051454, -395.0105294, 4364.033919, -151.6783697, -69.56073163);
const GAMMA_REF_Z = 280.0; const GAMMA_SLOPE = 0.58;

// Proper speed Gamma beta = sqrt(G280^2 - 1) (|z|/280)^0.58 (twin: gammaProfile in jet.ts).
fn jetGammaAt(z: f32, g280: f32) -> f32 {
  let ub = sqrt(max(0.0, g280 * g280 - 1.0)) * pow(max(abs(z), 1e-6) / GAMMA_REF_Z, GAMMA_SLOPE);
  return sqrt(1.0 + ub * ub);
}
// Unit flow direction (n_r, n_theta): outward along the funnel family rho = q rho_f(z).
fn streamlineDirJ(r: f32, th: f32) -> vec2<f32> {
  let z = r * cos(th); let rho = r * sin(th); let az = max(abs(z), 1e-6);
  let vr = (rho / funnelEdgeJ(z)) * (JET_SLOPE / (2.0 * sqrt(az)));
  let vz = select(-1.0, 1.0, z >= 0.0);
  let n = sqrt(vr * vr + vz * vz); let ur = vr / n; let uz = vz / n;
  return vec2<f32>(ur * sin(th) + uz * cos(th), ur * cos(th) - uz * sin(th));
}
// nu_plasma / nu_observed = p.u for the camera-normalised past-directed photon momentum p at (r, th);
// the plasma moves at Lorentz factor gamma along the streamline in the ZAMO frame (spec 2.2/2.4).
fn plasmaShiftJ(r: f32, th: f32, p: vec4<f32>, a: f32, gamma: f32) -> f32 {
  let g = gUp(r, th, a);
  let lapse = sqrt(-1.0 / g[0]); let omega = g[1] / g[0];
  let beta = sqrt(max(0.0, 1.0 - 1.0 / (gamma * gamma)));
  let n = streamlineDirJ(r, th);
  return gamma * ((p.x + omega * p.w) / lapse + beta * (n.x * sqrt(g[2]) * p.y + n.y * sqrt(g[3]) * p.z));
}
// Density modulation (wall x length falloff x knots x turbulence); 0 outside the emitting jet.
fn jetShapeJ(r: f32, th: f32, t: f32, jetLength: f32, knotAmp: f32, g280: f32) -> f32 {
  let z = r * cos(th); let az = abs(z);
  if (az < JET_ZBASE || az > jetLength) { return 0.0; }
  let rho = r * sin(th);
  let w = wallJ(rho, z);
  if (w <= 0.0) { return 0.0; }
  let turb = 1.0 + JET_TURB * (vnoiseE(log(1.0 + rho), JET_KZ * z) - 0.5) * 2.0;
  return max(0.0, w * lengthFalloffJ(z, jetLength) * knotsJ(z, t, jetGammaAt(z, g280), knotAmp) * turb);
}
struct SynchOut { j: vec3<f32>, a: vec3<f32> };
// Per band: j = (nu/nu')^3 j'(nu') (observed-frame weighted, cgs per sr) and alpha' (1/cm) at nu' = D nu.
fn synchSampleJ(r: f32, th: f32, D: f32, a: f32, b0: f32, kScale: f32, shape: f32) -> SynchOut {
  let z = r * cos(th); let rho = r * sin(th);
  let rf = funnelEdgeJ(z); let rH = 1.0 + sqrt(max(0.0, 1.0 - a * a)); let w = rho * a / (4.0 * rH);
  let lnB = log(b0 / (rf * rf)) + 0.5 * log(1.0 + w * w);
  let lnK = log(kScale) + 2.0 * lnB + log(shape);
  let lnNuMin = SYN_LNNUMIN0 + lnB;
  let lnD = log(D);
  var o: SynchOut;
  for (var b = 0; b < 3; b++) {
    let lnNu = JET_LNNU[b] + lnD;
    let le = max(lnNu, lnNuMin);
    var lj = SYN_LNCJ + lnK + 0.5 * (SYN_P + 1.0) * lnB - 0.5 * (SYN_P - 1.0) * le;
    var la = SYN_LNCA + lnK + 0.5 * (SYN_P + 2.0) * lnB - 0.5 * (SYN_P + 4.0) * le;
    if (lnNu < lnNuMin) { lj += (lnNu - lnNuMin) / 3.0; la -= (5.0 / 3.0) * (lnNu - lnNuMin); }
    o.j[b] = exp(lj - 3.0 * lnD);
    o.a[b] = exp(la);
  }
  return o;
}
// Light from the jet along one ray: I = observed I_nu per band (cgs), tau = optical depth from the camera.
struct JetOut { I: vec3<f32>, tau: vec3<f32> };
// One sample = a uniform slab of plasma-frame path ds (cm): exact solution, thin -> j ds, thick ->
// source function j / alpha, attenuated by what lies in front (twin: slabStep in synchrotron.ts).
fn jetSlabJ(acc: JetOut, j: vec3<f32>, alpha: vec3<f32>, ds: f32) -> JetOut {
  let dt = alpha * ds;
  let fac = select((vec3<f32>(1.0) - exp(-dt)) / max(dt, vec3<f32>(1e-30)), vec3<f32>(1.0) - 0.5 * dt, dt < vec3<f32>(1e-4));
  var o: JetOut;
  o.I = acc.I + j * ds * fac * exp(-acc.tau);
  o.tau = min(acc.tau + dt, vec3<f32>(1e30));
  return o;
}
