// Sole WGSL copy of the disk turbulence and jet emissivity (twins: src/physics/emission.ts,
// src/physics/jet.ts). Prepended verbatim by gpu.ts (renderer) and parity.browser.ts (?parity: turb
// and jet cases), so ?parity checks the shipped bytes. Self-contained: no uniforms, no bindings.

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

// --- Tier 2B jet emissivity ----------------------------------------------------------------
const JET_QPEAK = 0.8;   const JET_WWALL = 0.22;
const JET_RHO0  = 0.6;   const JET_SLOPE = 0.7;
const JET_ZBASE = 2.0;   const JET_KZ    = 0.35;
const JET_PBEAM = 3.5;   const JET_TURB  = 0.35;  const JET_SEED  = 17.0;

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
fn boostJ(mu: f32, gamma: f32) -> f32 {
  let beta = sqrt(max(0.0, 1.0 - 1.0 / (gamma * gamma)));
  let delta = 1.0 / (gamma * (1.0 - beta * mu));
  return pow(delta, JET_PBEAM);
}
// scalar emissivity (no beaming); exactly 0 when jet off / below zBase / beyond zMax / outside wall
// Scalar jet emissivity (no beaming, no on/off switch); exactly 0 below zBase / beyond jetLength /
// outside the wall. Twin: jetEmission in jet.ts (with jetStrength != 0).
fn jetEmissionCoreJ(r: f32, th: f32, t: f32, jetLength: f32, knotAmp: f32, gamma: f32) -> f32 {
  let z = r * cos(th);
  let az = abs(z);
  if (az < JET_ZBASE || az > jetLength) { return 0.0; }
  let rho = r * sin(th);
  let w = wallJ(rho, z);
  if (w <= 0.0) { return 0.0; }
  let turb = 1.0 + JET_TURB * (vnoiseE(log(1.0 + rho), JET_KZ * z) - 0.5) * 2.0;
  return max(0.0, w * lengthFalloffJ(z, jetLength) * knotsJ(z, t, gamma, knotAmp) * turb);
}
