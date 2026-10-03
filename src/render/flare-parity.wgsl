// Parity for the eruption flares: tubeLightJ (emission-shared.wgsl) vs tubeLight (eruption-spots.ts).
// Per case: a = (epoch, rel, rHit, phiHit), b = (g, spin, s, f), c = (Phi, zeta, rgCm, 0). Out: ln I per band.
struct FlIn { a: vec4<f32>, b: vec4<f32>, c: vec4<f32> };
@group(0) @binding(0) var<storage, read> flin: array<FlIn>;
@group(0) @binding(1) var<storage, read_write> flout: array<vec4<f32>>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = flin[gid.x];
  let I = tubeLightJ(c.a.z, c.a.w, c.b.x, c.a.x, c.a.y, c.b.y, c.b.z, c.b.w, c.c.x, c.c.y, c.c.z);
  flout[gid.x] = vec4<f32>(log(max(I, vec3<f32>(1e-38))), 0.0);
}
