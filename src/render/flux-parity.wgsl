// Parity harness for the flux history, launch delay, co-moving azimuth and filaments: the functions come from
// emission-shared.wgsl (sole copies, also prepended to raytrace.wgsl), so ?parity checks the renderer's own bytes
// against flux-history.ts and jet.ts. Per case: a = (epoch, rel, z, g280), b = (q, ph, spin, s).
// Output: (fluxRatioJ(epoch, rel, s, spin), launchDelayJ(z, g280), comovingAzimuthJ(ph, z, spin, g280), filamentsJ(q, ph, epoch, rel)).
struct FIn { a: vec4<f32>, b: vec4<f32> };
@group(0) @binding(0) var<storage, read> finp: array<FIn>;
@group(0) @binding(1) var<storage, read_write> foutp: array<vec4<f32>>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = finp[gid.x];
  foutp[gid.x] = vec4<f32>(fluxRatioJ(c.a.x, c.a.y, c.b.w, c.b.z), launchDelayJ(c.a.z, c.a.w),
                           comovingAzimuthJ(c.b.y, c.a.z, c.b.z, c.a.w), filamentsJ(c.b.x, c.b.y, c.a.x, c.a.y));
}
