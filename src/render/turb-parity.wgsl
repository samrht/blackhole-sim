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
@group(0) @binding(0) var<storage, read> inp: array<vec2<f32>>;   // (logR, psi)
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = arrayLength(&inp);
  if (gid.x >= n) { return; }
  outp[gid.x] = turbulenceE(inp[gid.x].x, inp[gid.x].y);
}
