// Parity harness for the disk turbulence: turbulenceE comes from emission-shared.wgsl (the sole copy,
// also prepended to raytrace.wgsl), so ?parity checks the renderer's own bytes against
// turbulence() in src/physics/emission.ts. Inputs are (logR, psi).
@group(0) @binding(0) var<storage, read> inp: array<vec2<f32>>;   // (logR, psi)
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = arrayLength(&inp);
  if (gid.x >= n) { return; }
  outp[gid.x] = turbulenceE(inp[gid.x].x, inp[gid.x].y);
}
