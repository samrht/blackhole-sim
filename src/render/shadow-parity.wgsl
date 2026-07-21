// Entry point for CPU<->GPU parity of the Kerr critical-curve classifier.
// Input per case: vec4(xi, eta, a, 0). Output per case: vec4(captured ? 1 : 0, 0, 0, 0).
//
// This file deliberately contains NO copy of the classifier math. shadow-shared.wgsl is prepended
// to it by parity.browser.ts, exactly as it is prepended to raytrace.wgsl by gpu.ts -- so this
// route verifies the code the renderer actually runs, not a hand-synced duplicate of it.
@group(0) @binding(0) var<storage, read> inp: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outp: array<vec4<f32>>;

@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = inp[gid.x];
  outp[gid.x] = vec4<f32>(select(0.0, 1.0, classifyCaptured(c.x, c.y, c.z)), 0.0, 0.0, 0.0);
}
