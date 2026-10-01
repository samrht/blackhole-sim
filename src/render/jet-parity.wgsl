// Parity harness for the Tier 2B jet: jetEmissionCoreJ and boostJ come from emission-shared.wgsl (the
// sole copy, also prepended to raytrace.wgsl), so ?parity checks the renderer's own bytes against
// jetEmission() / dopplerBoost() in src/physics/jet.ts. Input per case: vec4(r, th, t, mu).
// Output per case: vec4(emission, boost, 0, 0). The fixed settings below match parity.browser.ts.
@group(0) @binding(0) var<storage, read> inp: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outp: array<vec4<f32>>;
const P_JETLEN = 60.0; const P_KNOTS = 0.7; const P_GAMMA = 5.0;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = inp[gid.x];
  outp[gid.x] = vec4<f32>(jetEmissionCoreJ(c.x, c.y, c.z, P_JETLEN, P_KNOTS, P_GAMMA), boostJ(c.w, P_GAMMA), 0.0, 0.0);
}
