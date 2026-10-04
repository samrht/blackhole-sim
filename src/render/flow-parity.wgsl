// Parity harness for the hot flow: flowVelocityJ, flowShiftJ and flowCoeffsJ come from emission-shared.wgsl (sole
// copies, also prepended to raytrace.wgsl), so ?parity checks the renderer's own bytes against src/physics/hot-flow.ts.
// Per case: a = (r, th, spin, n0), b = photon momentum (p_t, p_r, p_th, p_phi), covariant, camera-normalised.
// Output: (D, ln j, ln alpha, u^t) with the coefficients at nu' = D x 230 GHz (-1e30 for j = 0).
struct FIn { a: vec4<f32>, b: vec4<f32> };
@group(0) @binding(0) var<storage, read> inp: array<FIn>;
@group(0) @binding(1) var<storage, read_write> outp: array<vec4<f32>>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = inp[gid.x];
  let r = c.a.x; let th = c.a.y; let a = c.a.z;
  let D = flowShiftJ(r, th, c.b, a);
  let k = flowCoeffsJ(r, th, HF_LNNU + log(max(D, 1e-6)), c.a.w);
  // j = 0 (no plasma) comes back as ln j = ln alpha = -1e30: WGSL leaves log(0) undefined.
  outp[gid.x] = vec4<f32>(D, select(log(k.x), -1e30, k.x <= 0.0), select(log(k.y), -1e30, k.x <= 0.0), flowVelocityJ(r, th, a).x);
}
