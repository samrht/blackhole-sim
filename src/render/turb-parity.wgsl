// Parity harness for the disk turbulence: turbulenceFieldE comes from emission-shared.wgsl (the sole copy,
// also prepended to raytrace.wgsl), so ?parity checks the renderer's own bytes against turbulenceAt() in
// src/physics/emission.ts. Two vec4 per case: (r, phi, t0, tRel), (a, 0, 0, 0); time = epoch t0 + tRel.
@group(0) @binding(0) var<storage, read> inp: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = arrayLength(&inp) / 2u;
  if (gid.x >= n) { return; }
  let c = inp[2u * gid.x]; let d = inp[2u * gid.x + 1u];
  outp[gid.x] = turbulenceFieldE(c.x, c.y, c.z, c.w, d.x);
}
