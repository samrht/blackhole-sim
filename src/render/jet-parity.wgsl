// Parity harness for the synchrotron jet: plasmaShiftJ, jetShapeJ, synchSampleJ (which samples the cooled-jet
// coefficient table, binding 10, declared in emission-shared.wgsl) and jetSlabJ come from emission-shared.wgsl
// (sole copies, also prepended to raytrace.wgsl), so ?parity checks the renderer's own bytes against
// src/physics/synchrotron.ts + cyclosynch.ts + jet.ts. Per case: x = (t, r, th, phi), p = photon momentum
// (covariant, camera-normalised), c = (a, g280, emission time, slab path ds in cm), k = (b0, q0, rgCm, 0).
// Output: o0 = (D, shape, ln j450, ln j550), o1 = (ln j650, ln a450, ln a550, ln a650),
// o2 = (ln I450, tau450, ln I650, tau650) after two jetSlabJ steps of path ds from zero.
struct JIn { x: vec4<f32>, p: vec4<f32>, c: vec4<f32>, k: vec4<f32> };
struct JOutP { o0: vec4<f32>, o1: vec4<f32>, o2: vec4<f32> };
@group(0) @binding(0) var<storage, read> inp: array<JIn>;
@group(0) @binding(1) var<storage, read_write> outp: array<JOutP>;
const P_JETLEN = 60.0; const P_KNOTS = 0.7;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = inp[gid.x];
  let r = c.x.y; let th = c.x.z; let a = c.c.x;
  let D = plasmaShiftJ(r, th, c.p, a, jetGammaAt(r * cos(th), c.c.y));
  let shape = jetShapeJ(r, th, c.c.z, P_JETLEN, P_KNOTS, c.c.y);
  let s = synchSampleJ(r, th, D, a, c.k.x, c.k.y, shape, c.c.y, c.k.z);
  var acc: JetOut; acc.I = vec3<f32>(0.0); acc.tau = vec3<f32>(0.0);
  acc = jetSlabJ(acc, s.j, s.a, c.c.w);
  acc = jetSlabJ(acc, s.j, s.a, c.c.w);
  outp[gid.x] = JOutP(vec4<f32>(D, shape, log(s.j.x), log(s.j.y)),
                      vec4<f32>(log(s.j.z), log(s.a.x), log(s.a.y), log(s.a.z)),
                      vec4<f32>(log(acc.I.x), acc.tau.x, log(acc.I.z), acc.tau.z));
}
