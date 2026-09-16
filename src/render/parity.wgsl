// Parity test shader for the metric/orbit/g-factor helpers. The helpers themselves are NOT
// copied here: integrator-shared.wgsl is prepended by parity.browser.ts, exactly as gpu.ts
// prepends it to raytrace.wgsl, so this compares the TypeScript core against the shipped bytes.

struct In { r: f32, th: f32, a: f32, xi: f32 };
@group(0) @binding(0) var<storage, read> inputs: array<In>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec4<f32>>; // (gUp.tt, gLow.tt, omegaKep, gFactor)

@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = arrayLength(&inputs);
  if (gid.x >= n) { return; }
  let v = inputs[gid.x];
  let gu = gUp(v.r, v.th, v.a); let gl = gLow(v.r, v.th, v.a);
  let Om = omegaKep(v.r, v.a);
  // The disk g-factor is always evaluated in the equatorial plane (matches gFactorKepler),
  // independent of the test case's theta used for the metric-parity checks above.
  let glEq = gLow(v.r, 1.5707963267948966, v.a);
  let rad = -(glEq[0] + 2.0*Om*glEq[1] + Om*Om*glEq[4]);
  let gfac = sqrt(max(0.0, rad)) / (1.0 - Om*v.xi);
  outputs[gid.x] = vec4<f32>(gu[0], gl[0], Om, gfac);
}
