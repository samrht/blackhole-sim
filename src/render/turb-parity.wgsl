// Parity harness for the disk turbulence: turbulenceFieldE comes from emission-shared.wgsl (the sole copy,
// also prepended to raytrace.wgsl), so ?parity checks the renderer's own bytes against turbulenceAt() in
// src/physics/emission.ts. Inputs are (r, phi, t_emit, a).
@group(0) @binding(0) var<storage, read> inp: array<vec4<f32>>;   // (r, phi, t, a)
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = arrayLength(&inp);
  if (gid.x >= n) { return; }
  let c = inp[gid.x];
  outp[gid.x] = turbulenceFieldE(c.x, c.y, c.z, c.w);
}
