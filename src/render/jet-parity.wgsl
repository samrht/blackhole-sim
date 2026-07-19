// Standalone twin for CPU<->GPU parity of the Tier 2B jet. Input per case: vec4(r, th, t, mu).
// Output per case: vec4(emission, boost, 0, 0). Constants MUST match src/physics/jet.ts.
@group(0) @binding(0) var<storage, read> inp: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outp: array<vec4<f32>>;

const JET_QPEAK = 0.8; const JET_WWALL = 0.22; const JET_RHO0 = 0.6; const JET_SLOPE = 0.7;
const JET_ZBASE = 2.0; const JET_KZ = 0.35; const JET_VKNOT = 6.0; const JET_PBEAM = 3.5;
const JET_TURB = 0.35; const JET_SEED = 17.0;
const P_JETLEN = 60.0; const P_KNOTS = 0.7; const P_TS = 1.0; const P_GAMMA = 5.0;

fn ihashJ(ix: i32, iy: i32) -> f32 {
  var n = u32(ix) * 1973u + u32(iy) * 9277u;
  n = (n ^ (n >> 15u)) * 2246822519u;
  n = (n ^ (n >> 13u)) * 3266489917u;
  return f32(n & 0xffffffu) / f32(0xffffffu);
}
fn smoothJ(t: f32) -> f32 { return t * t * (3.0 - 2.0 * t); }
fn vnoiseJ(x: f32, y: f32) -> f32 {
  let ix = i32(floor(x)); let iy = i32(floor(y));
  let fx = smoothJ(x - floor(x)); let fy = smoothJ(y - floor(y));
  let a00 = ihashJ(ix, iy); let a10 = ihashJ(ix + 1, iy);
  let a01 = ihashJ(ix, iy + 1); let a11 = ihashJ(ix + 1, iy + 1);
  return (a00 * (1.0 - fx) + a10 * fx) * (1.0 - fy) + (a01 * (1.0 - fx) + a11 * fx) * fy;
}
fn ss(a: f32, b: f32, x: f32) -> f32 { let t = clamp((x - a) / (b - a), 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }
fn edge(z: f32) -> f32 { return JET_RHO0 + JET_SLOPE * sqrt(abs(z)); }
fn wall(rho: f32, z: f32) -> f32 {
  let q = rho / edge(z); if (q > 1.2) { return 0.0; }
  let d = q - JET_QPEAK; return exp(-(d * d) / (2.0 * JET_WWALL * JET_WWALL));
}
fn falloff(z: f32) -> f32 {
  let az = abs(z);
  return ss(JET_ZBASE, JET_ZBASE + 2.0, az) * (1.0 - ss(P_JETLEN * 0.7, P_JETLEN, az)) * (JET_ZBASE / max(az, JET_ZBASE));
}
fn kn(z: f32, t: f32) -> f32 { let ph = JET_KZ * abs(z) - JET_VKNOT * t * P_TS; return 1.0 + P_KNOTS * (vnoiseJ(ph, JET_SEED) - 0.5) * 2.0; }
fn emission(r: f32, th: f32, t: f32) -> f32 {
  let z = r * cos(th); let az = abs(z);
  if (az < JET_ZBASE || az > P_JETLEN) { return 0.0; }
  let rho = r * sin(th); let w = wall(rho, z); if (w <= 0.0) { return 0.0; }
  let turb = 1.0 + JET_TURB * (vnoiseJ(log(1.0 + rho), JET_KZ * z) - 0.5) * 2.0;
  return max(0.0, w * falloff(z) * kn(z, t) * turb);
}
fn boost(mu: f32) -> f32 {
  let beta = sqrt(max(0.0, 1.0 - 1.0 / (P_GAMMA * P_GAMMA)));
  let delta = 1.0 / (P_GAMMA * (1.0 - beta * mu));
  return pow(delta, JET_PBEAM);
}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = inp[gid.x];
  outp[gid.x] = vec4<f32>(emission(c.x, c.y, c.z), boost(c.w), 0.0, 0.0);
}
