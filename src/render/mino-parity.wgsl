// Parity harness for the Mino-time integrator: minoRay / minoRhs / minoStep / minoDense / minoToState come from
// integrator-shared.wgsl (sole copy, also prepended to raytrace.wgsl), so ?parity checks the renderer's own bytes against
// src/physics/trace-mino.ts. ONE step per case from an identical f32 state (adaptive stepping in two precisions picks
// different step sizes where the error estimate is noise, << 1, so multi-step sequences are not comparable; whole-ray
// accuracy on the GPU is gated by ?accuracy). Per case: q, v = the state y; c = (spin, xi, eta, h0).
// Output: y1 (q, v), (hNext, attempts, err norm of the accepted attempt, MINO_TOL), State(y1) (x, p),
// minoDense(y0 as used, y1, f0, f1, h, 0.37) (q, v), (MINO_MAX_REJECT, h used, MINO_UFRAC, 0), and when the step crosses
// the plane: the landed state minoLand(y0, f0, h, th) (q, v) and (th, 1, 0, 0) (else zeros); when it crosses the escape
// sphere r = 1.2 x 1000: the landed state minoSphere (q, v) and (1, 0, 0, 0) (else zeros).
struct MIn { q: vec4<f32>, v: vec4<f32>, c: vec4<f32> };
struct MOut { q: vec4<f32>, v: vec4<f32>, info: vec4<f32>, sx: vec4<f32>, sp: vec4<f32>, dq: vec4<f32>, dv: vec4<f32>, extra: vec4<f32>,
              lq: vec4<f32>, lv: vec4<f32>, lth: vec4<f32>, eq: vec4<f32>, ev: vec4<f32>, eok: vec4<f32> };
@group(0) @binding(0) var<storage, read> inp: array<MIn>;
@group(0) @binding(1) var<storage, read_write> outp: array<MOut>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= arrayLength(&inp)) { return; }
  let m = inp[gid.x];
  let c = minoRay(m.c.x, m.c.y, m.c.z, 1000.0);
  let y = Mino(m.q, m.v);
  let o = minoStep(y, minoRhs(y, c), m.c.w, c);
  let s = minoToState(o.y, c);
  let d = minoDense(o.y0, o.y, o.f0, o.f1, o.h, 0.37);
  let th = minoCrossing(o.y0, o.y, o.f0, o.f1, o.h);
  var l = Mino(vec4<f32>(0.0), vec4<f32>(0.0)); var lth = vec4<f32>(0.0);
  if (th >= 0.0) { l = minoLand(o.y0, o.f0, o.h, th, c); lth = vec4<f32>(th, 1.0, 0.0, 0.0); }
  let e = minoSphere(o, c, 1.0 / 1200.0);
  var ey = Mino(vec4<f32>(0.0), vec4<f32>(0.0)); var eok = vec4<f32>(0.0);
  if (e.ok) { ey = e.y; eok = vec4<f32>(1.0, 0.0, 0.0, 0.0); }
  outp[gid.x] = MOut(o.y.q, o.y.v, vec4<f32>(o.hNext, f32(o.attempts), o.en, MINO_TOL), s.x, s.p, d.q, d.v,
                     vec4<f32>(f32(MINO_MAX_REJECT), o.h, MINO_UFRAC, 0.0), l.q, l.v, lth, ey.q, ey.v, eok);
}
