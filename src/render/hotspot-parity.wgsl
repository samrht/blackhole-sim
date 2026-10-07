// Parity harness for the 1.3 mm hotspots: hotspotStateJ / hotspotBoostJ / hotspotShiftJ come from emission-shared.wgsl
// (sole copies, also prepended to raytrace.wgsl), so ?parity checks the renderer's own bytes against src/physics/hotspot.ts.
// Per case: a = (epoch, rel, s, spin), b = (r, th, ph, 0), c = (p_t, p_phi, 0, 0). Output per case, two vec4: the state
// (r_c, phi_c, amp, alive), then (G at b about the GPU's own state, D_h at b, Omega_c, 0); a dead state uses r_c = HS_RMIN.
struct HIn { a: vec4<f32>, b: vec4<f32>, c: vec4<f32> };
@group(0) @binding(0) var<storage, read> inp: array<HIn>;
@group(0) @binding(1) var<storage, read_write> outp: array<vec4<f32>>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = inp[gid.x];
  let st = hotspotStateJ(c.a.x, c.a.y, c.a.z, c.a.w);
  let rc = max(st.x, HS_RMIN); let Om = 1.0 / (pow(rc, 1.5) + c.a.w);
  outp[2u * gid.x] = st;
  outp[2u * gid.x + 1u] = vec4<f32>(hotspotBoostJ(c.b.x, c.b.y, c.b.z, rc, st.y),
    hotspotShiftJ(c.b.x, c.b.y, vec4<f32>(c.c.x, 0.0, 0.0, c.c.y), c.a.w, Om), Om, 0.0);
}
