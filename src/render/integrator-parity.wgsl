// Entry point for CPU<->GPU parity of the constraint-monitored integrator step.
// Input per case: state (x, p) + a + dl0 + hTol (the tolerance the renderer would use at that
// state: H_TOL, or H_TOL_FAR in the far field) + rOut. A NEGATIVE dl0 is a sentinel:
// the shader then computes its own stride with the shipped stepSize (the render loop's call,
// including the far-field F_AXIS and F_PHI caps), so the step controller is gated too, not only the step;
// a non-negative dl0 is used as given (the forced-retry case needs an oversized stride).
// Output per case: new state (x, p) + info (retries, ok ? 1 : 0, H_TOL, MAX_RETRY) so the
// constants are checked too + extra (the dl0 actually used, H_TOL_FAR, 0, 0). (The field is named `info`
// because `meta` is a reserved word in WGSL.)
//
// This file deliberately contains NO copy of the integrator. integrator-shared.wgsl is prepended
// by parity.browser.ts, exactly as gpu.ts prepends it to raytrace.wgsl.
struct StepIn { x: vec4<f32>, p: vec4<f32>, a: f32, dl0: f32, hTol: f32, rOut: f32 };
struct StepRes { x: vec4<f32>, p: vec4<f32>, info: vec4<f32>, extra: vec4<f32> };
@group(0) @binding(0) var<storage, read> inp: array<StepIn>;
@group(0) @binding(1) var<storage, read_write> outp: array<StepRes>;

@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= arrayLength(&inp)) { return; }
  let v = inp[gid.x];
  let s = State(v.x, v.p);
  let rh = 1.0 + sqrt(max(0.0, 1.0 - v.a * v.a)); // same horizon expression as raytrace.wgsl
  let dl0 = select(v.dl0, stepSize(s, rh, v.rOut), v.dl0 < 0.0);
  let o = stepGeodesic(s, v.a, dl0, v.hTol);
  outp[gid.x] = StepRes(o.s.x, o.s.p, vec4<f32>(f32(o.retries), select(0.0, 1.0, o.ok), H_TOL, f32(MAX_RETRY)),
                        vec4<f32>(dl0, H_TOL_FAR, 0.0, 0.0));
}
