// ?drift: how far each ray has left the quantities that should stay fixed, by the end of the ray. A debug route's own
// module (src/test/drift.browser.ts); nothing here is compiled into the renderer. Prepended: shadow-shared, integrator-shared.
// CPU twin of the measure: src/physics/conserved.ts. The loop is traceRay's with the emitters removed, in traceRay's
// order (plane, capture, escape, budget), over the renderer's own minoStep / minoPlane / minoSphere; the route checks its
// fates against the renderer's audit entry, pixel for pixel.
struct DU {
  res: vec2<f32>, a: f32, incl: f32, rObs: f32, fovScale: f32, rIn: f32, rOut: f32,
  jit: vec2<f32>, pan: vec2<f32>, maxSteps: u32, rowStart: u32, rowEnd: u32, pad: u32,
};
struct DriftPx { q0: f32, q1: f32, h0: f32, h1: f32, hScale: f32, qScale: f32, wRes: f32, rEnd: f32, xi: f32, eta: f32, vEnd: f32, word: u32, wEnd: f32, wpEnd: f32, vpEnd: f32, vp0: f32 };
struct MeasureIn { q: vec4<f32>, v: vec4<f32>, a: f32, xi: f32, eta: f32, r0: f32 };
struct MeasureOut { q: f32, qScale: f32, h: f32, hScale: f32, wRes: f32, pad0: f32, pad1: f32, pad2: f32 };
@group(0) @binding(0) var<uniform> U: DU;
@group(0) @binding(1) var<storage, read_write> px: array<DriftPx>;
@group(0) @binding(2) var<storage, read> mIn: array<MeasureIn>;
@group(0) @binding(3) var<storage, read_write> mOut: array<MeasureOut>;

// (Q, its term size). Twin: minoQ.
fn driftQ(y: Mino, c: MinoRay) -> vec2<f32> {
  let sc = sinCosP(y.q.z); let s2 = max(sc.x * sc.x, 1e-30); let c2 = sc.y * sc.y; let vp = y.v.z;
  let t0 = vp * vp; let t1 = c2 * ((c.xi * c.xi) / s2); let t2 = c2 * (c.a * c.a);
  return vec2<f32>(t0 + t1 - t2, t0 + t1 + t2);
}
// (H, its term size, w'^2 - R~(w)). Twin: minoHQ. Through sinCosP, not gUp: the built-in cos can be 2^-11 off (see sinCosP).
fn driftH(y: Mino, c: MinoRay) -> vec3<f32> {
  let a = c.a; let a2 = a * a; let w = y.q.y; let wp = y.v.y; let vp = y.v.z; let xi = c.xi;
  let sc = sinCosP(y.q.z); let s2 = max(sc.x * sc.x, 1e-30);
  let r = 1.0 / w; let r2 = r * r; let ra = r2 + a2;
  let Sig = r2 + a2 * (sc.y * sc.y); let D = r2 - 2.0 * r + a2; let SD = Sig * D;
  let A = ra * ra - a2 * D * s2;
  let pr = -wp / (1.0 - 2.0 * w + a2 * (w * w));
  let t0 = -A / SD; let t1 = 4.0 * a * r * xi / SD; let t2 = (D / Sig) * pr * pr; let t3 = (vp * vp) / Sig;
  let t4 = ((D - a2 * s2) / (SD * s2)) * (xi * xi);
  let sum = t0 + t1 + t2 + t3 + t4;
  let tot = abs(t0) + abs(t1) + abs(t2) + abs(t3) + abs(t4);
  return vec3<f32>(0.5 * sum, 0.5 * tot, wp * wp - minoRw(w, c));
}

// Word: kind (0 captured, 1 disk, 2 sky, as raytrace.wgsl) | 4 when the budget ran out | steps << 3.
@compute @workgroup_size(8, 8) fn trace(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u32(U.res.x) || gid.y < U.rowStart || gid.y >= U.rowEnd) { return; }
  let a = U.a; let i = U.incl; let r0 = U.rObs;
  let ndc = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5 + U.jit) / U.res * 2.0 - 1.0;
  let alpha = ndc.x * U.fovScale * (U.res.x / U.res.y) + U.pan.x; let beta = -ndc.y * U.fovScale + U.pan.y;
  let xe = cameraXiEta(alpha, beta, a, i); let xi = xe.x; let eta = xe.y;
  let c = minoRay(a, xi, eta, r0);
  var y = minoInit(r0, i, beta, c);
  let vp0 = y.v.z;
  var f = minoRhs(y, c);
  var h = 50.0 / (r0 * r0);
  let q0 = driftQ(y, c); let h0 = driftH(y, c);
  let rh = 1.0 + sqrt(max(0.0, 1.0 - a * a));
  let wCap = 1.0 / (rh * 1.005); let wEsc = 1.0 / (r0 * 1.2);
  var end = y; var kind = 0u; var resolved = false; var steps = U.maxSteps;
  for (var step = 0u; step < U.maxSteps; step++) {
    let st = minoStep(y, f, h, c);
    let hit = minoPlane(st, c, U.rIn, U.rOut);
    if (hit.ok) {
      let rHit = 1.0 / hit.y.q.y;
      if (rHit >= U.rIn && rHit <= U.rOut) { end = hit.y; kind = 1u; resolved = true; steps = step + 1u; break; }
    }
    y = st.y; f = st.f1; h = st.hNext;
    if (y.q.y >= wCap) { end = y; kind = 0u; resolved = true; steps = step + 1u; break; }
    let esc = minoSphere(st, c, wEsc);
    if (esc.ok) { end = esc.y; kind = 2u; resolved = true; steps = step + 1u; break; }
  }
  if (!resolved) { end = y; kind = select(2u, 0u, classifyCaptured(xi, eta, a)); }
  let q1 = driftQ(end, c); let h1 = driftH(end, c);
  px[gid.y * u32(U.res.x) + gid.x] = DriftPx(q0.x, q1.x, h0.x, h1.x, h1.y, q1.y, h1.z, 1.0 / end.q.y, xi, eta, end.q.z,
    kind | select(4u, 0u, resolved) | (steps << 3u), end.q.y, end.v.y, end.v.z, vp0);
}

// The measure alone on given states: the route compares it with the CPU twin evaluated in f32.
@compute @workgroup_size(16) fn measure(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= arrayLength(&mIn)) { return; }
  let m = mIn[gid.x]; let c = minoRay(m.a, m.xi, m.eta, m.r0); let y = Mino(m.q, m.v);
  let q = driftQ(y, c); let hh = driftH(y, c);
  mOut[gid.x] = MeasureOut(q.x, q.y, hh.x, hh.y, hh.z, 0.0, 0.0, 0.0);
}
