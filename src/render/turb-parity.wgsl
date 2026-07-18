// Parity shader: WGSL twin of turbulence() from src/physics/emission.ts. Inputs are (logR, psi).
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
@group(0) @binding(0) var<storage, read> inp: array<vec2<f32>>;   // (logR, psi)
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = arrayLength(&inp);
  if (gid.x >= n) { return; }
  outp[gid.x] = turbulenceE(inp[gid.x].x, inp[gid.x].y);
}
