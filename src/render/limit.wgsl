// ?limit: the same rays through a 5th-order and a 4th-order integrator in f32, reporting what the renderer would use. A debug
// route's own module (src/test/limit.browser.ts); nothing here is compiled into the renderer. Prepended: shadow-shared,
// camera-shared, integrator-shared (MINO_TOL, H_TOL-style knobs are this module's: the route rewrites MINO_TOL in its own copy).
// `mino` is traceRay's loop with the emitters removed (as drift.wgsl). `rk4arm` is the pre-Mino renderer's integrator, twin of
// trace-reference.ts refTrace with RefCfg { scale, hNear, hFar, landEsc: true }, including its LINEAR chord interpolation of the disk hit.
struct LU { a: f32, incl: f32, rObs: f32, rIn: f32, rOut: f32, scale: f32, hNear: f32, hFar: f32, maxSteps: u32, first: u32, count: u32, comp: u32 };
struct RayIn { alpha: f32, beta: f32 };
// word = kind (0 captured, 1 disk, 2 sky) | 4 if the budget ran out
struct RayOut { word: u32, steps: u32, attempts: u32, rHit: f32, phi: f32, delay: f32, dx: f32, dy: f32, dz: f32, pad: f32, w: f32, wp: f32, v: f32, vp: f32, sigma: f32, ph: f32 };
@group(0) @binding(0) var<uniform> U: LU;
@group(0) @binding(1) var<storage, read> rays: array<RayIn>;
@group(0) @binding(2) var<storage, read_write> out: array<RayOut>;

// raytrace.wgsl skyDir: the direction the renderer samples the sky with.
fn limitSkyDir(s: State, a: f32) -> vec3<f32> {
  let r = s.x.y; let th = s.x.z; let ph = s.x.w;
  let g = gUp(r, th, a);
  let dr = g[2] * s.p.y; let dth = g[3] * s.p.z; let dph = g[1] * s.p.x + g[4] * s.p.w;
  let st = sin(th); let ct = cos(th); let sp = sin(ph); let cp = cos(ph);
  return normalize(vec3<f32>(dr * st * cp + r * ct * cp * dth - r * st * sp * dph, dr * st * sp + r * ct * sp * dth + r * st * cp * dph, dr * ct - r * st * dth));
}

@compute @workgroup_size(64) fn mino(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = U.first + gid.x; if (gid.x >= U.count) { return; }
  let a = U.a; let ray = rays[n]; let xe = cameraXiEta(ray.alpha, ray.beta, a, U.incl);
  let c = minoRay(a, xe.x, xe.y, U.rObs);
  var y = minoInit(U.rObs, U.incl, ray.beta, c); var f = minoRhs(y, c); var h = 50.0 / (U.rObs * U.rObs);
  let rh = 1.0 + sqrt(max(0.0, 1.0 - a * a)); let wCap = 1.0 / (rh * 1.005); let wEsc = 1.0 / (U.rObs * 1.2);
  var o = RayOut(4u, U.maxSteps, 0u, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var step = 0u; step < U.maxSteps; step++) {
    let st = minoStep(y, f, h, c); o.attempts += st.attempts;
    let hit = minoPlane(st, c, U.rIn, U.rOut);
    if (hit.ok) {
      let rHit = 1.0 / hit.y.q.y;
      if (rHit >= U.rIn && rHit <= U.rOut) { o.word = 1u; o.steps = step + 1u; o.rHit = rHit; o.phi = minoToState(hit.y, c).x.w; o.delay = minoDelay(hit.y); break; }
    }
    y = st.y; f = st.f1; h = st.hNext;
    if (y.q.y >= wCap) { o.word = 0u; o.steps = step + 1u; break; }
    let esc = minoSphere(st, c, wEsc);
    if (esc.ok) { let d = limitSkyDir(minoToState(esc.y, c), a); o.word = 2u; o.steps = step + 1u; o.dx = d.x; o.dy = d.y; o.dz = d.z; o.w = esc.y.q.y; o.wp = esc.y.v.y; o.v = esc.y.q.z; o.vp = esc.y.v.z; o.sigma = esc.y.v.w; o.ph = esc.y.q.w; break; }
  }
  out[n] = o;
}

@compute @workgroup_size(64) fn rk4arm(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = U.first + gid.x; if (gid.x >= U.count) { return; }
  let a = U.a; let ray = rays[n]; let xe = cameraXiEta(ray.alpha, ray.beta, a, U.incl); let i = U.incl;
  let rh = 1.0 + sqrt(max(0.0, 1.0 - a * a)); let k = U.scale;
  let g0 = gUp(U.rObs, i, a);
  let p0 = cameraMomenta(xe.x, ray.beta, g0[0], g0[1], g0[2], g0[3], g0[4]);
  var s = State(vec4<f32>(0.0, U.rObs, i, 0.0), p0);
  var o = RayOut(4u, U.maxSteps, 0u, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var step = 0u; step < U.maxSteps; step++) {
    let far = s.x.y > U.rOut * 1.5;
    var dl0: f32;
    if (far) { dl0 = angularCap(s, k * clamp(K_FAR * s.x.y, 0.6, 1.0e9), k * DL_FAR_MIN); dl0 = min(dl0, k * DL_FAR_MAX); }
    else { dl0 = angularCap(s, k * clamp(0.02 * (s.x.y - rh), 0.002, 0.5), k * 0.002); }
    let st = stepGeodesic(s, a, dl0, select(U.hNear, U.hFar, far)); o.attempts += st.retries + 1u;
    let sN = st.s;
    let f0 = s.x.z - 0.5 * PI; let f1 = sN.x.z - 0.5 * PI;
    if (f0 * f1 < 0.0 && abs(sN.x.z - s.x.z) < 0.5) {
      let frac = f0 / (f0 - f1); let rHit = s.x.y + frac * (sN.x.y - s.x.y);
      if (rHit >= U.rIn && rHit <= U.rOut) {
        o.word = 1u; o.steps = step + 1u; o.rHit = rHit; o.phi = s.x.w + frac * (sN.x.w - s.x.w);
        o.delay = -(s.x.x + frac * (sN.x.x - s.x.x)) - U.rObs; break;
      }
    }
    let sPrev = s; s = sN;
    if (s.x.y <= rh * 1.005) { o.word = 0u; o.steps = step + 1u; break; }
    if (s.x.y > U.rObs * 1.2) {
      var lo = 0.0; var hi = st.dl;
      for (var b = 0u; b < 30u; b++) { let m = 0.5 * (lo + hi); if (rk4(sPrev, a, m).x.y < U.rObs * 1.2) { lo = m; } else { hi = m; } }
      let d = limitSkyDir(rk4(sPrev, a, 0.5 * (lo + hi)), a); o.word = 2u; o.steps = step + 1u; o.dx = d.x; o.dy = d.y; o.dz = d.z; break;
    }
  }
  out[n] = o;
}

// ---- Compensated summation of the state (Knuth two-sum): y + e carries the bits a step's store would lose, and they are fed back into the
// next increment and into every stage's argument. Twin of nothing: an experiment. comp = 0 keeps e = 0, which must reproduce `mino`.
struct TS4 { s: vec4<f32>, e: vec4<f32> };
fn twoSum4(a: vec4<f32>, b: vec4<f32>) -> TS4 { let s = a + b; let bb = s - a; return TS4(s, (a - (s - bb)) + (b - bb)); }
// A compiler that simplifies (a + b) - a to b zeroes e and would fake a null result: the route checks this entry against exact values.
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = U.first + gid.x; if (gid.x >= U.count) { return; }
  let t = twoSum4(vec4<f32>(rays[n].alpha), vec4<f32>(rays[n].beta));
  out[n] = RayOut(0u, 0u, 0u, 0.0, 0.0, 0.0, t.s.x, t.e.x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}
const PI_RESIDUAL = -8.742278e-8; // pi - f32(pi): the mirror's f32 pi is off by this, every crossing
struct CTry { y1: Mino, e1: Mino, f1: Mino, en: f32 };
fn cArg(y: Mino, e: Mino, d: Mino) -> Mino { return Mino(y.q + (e.q + d.q), y.v + (e.v + d.v)); }
fn cTry(y: Mino, ey: Mino, f0: Mino, h: f32, c: MinoRay) -> CTry {
  let z = Mino(vec4<f32>(0.0), vec4<f32>(0.0));
  let k1 = f0;
  let k2 = minoRhs(cArg(y, ey, mStage(z, h, k1, 1.0 / 5.0, z, 0.0, z, 0.0, z, 0.0, z, 0.0, z, 0.0)), c);
  let k3 = minoRhs(cArg(y, ey, mStage(z, h, k1, 3.0 / 40.0, k2, 9.0 / 40.0, z, 0.0, z, 0.0, z, 0.0, z, 0.0)), c);
  let k4 = minoRhs(cArg(y, ey, mStage(z, h, k1, 44.0 / 45.0, k2, -56.0 / 15.0, k3, 32.0 / 9.0, z, 0.0, z, 0.0, z, 0.0)), c);
  let k5 = minoRhs(cArg(y, ey, mStage(z, h, k1, 19372.0 / 6561.0, k2, -25360.0 / 2187.0, k3, 64448.0 / 6561.0, k4, -212.0 / 729.0, z, 0.0, z, 0.0)), c);
  let k6 = minoRhs(cArg(y, ey, mStage(z, h, k1, 9017.0 / 3168.0, k2, -355.0 / 33.0, k3, 46732.0 / 5247.0, k4, 49.0 / 176.0, k5, -5103.0 / 18656.0, z, 0.0)), c);
  let d = mStage(z, h, k1, 35.0 / 384.0, k2, 0.0, k3, 500.0 / 1113.0, k4, 125.0 / 192.0, k5, -2187.0 / 6784.0, k6, 11.0 / 84.0);
  var y1 = cArg(y, ey, d); var e1 = z;
  if (U.comp == 1u) {
    let tq = twoSum4(y.q, ey.q + d.q); let tv = twoSum4(y.v, ey.v + d.v);
    y1 = Mino(tq.s, tv.s); e1 = Mino(tq.e, tv.e);
  }
  let k7 = minoRhs(y1, c);
  let err = mStage(z, h, k1, 71.0 / 57600.0, k3, -71.0 / 16695.0, k4, 71.0 / 1920.0, k5, -17253.0 / 339200.0, k6, 22.0 / 525.0, k7, -1.0 / 40.0);
  let en = select(3e38, max(minoErrNorm(y, y1, err, MINO_TOL), minoDriftNorm(y, y1, k7, c, MINO_TOL)), y1.q.y > 0.0);
  return CTry(y1, e1, k7, en);
}
struct CStep { y0: Mino, e0: Mino, y: Mino, e: Mino, f0: Mino, f1: Mino, h: f32, hNext: f32, attempts: u32 };
fn cStep(yIn: Mino, eIn: Mino, f0In: Mino, h0: f32, c: MinoRay) -> CStep {
  let y = minoHemi(yIn); var ey = eIn; var f0 = f0In;
  if (yIn.q.z > 0.5 * PI) {
    f0 = minoRhs(y, c);
    if (U.comp == 1u) {
      let pf: f32 = PI; let bb = y.q.z - pf; let mirrorErr = (pf - (y.q.z - bb)) + (-yIn.q.z - bb);
      ey = Mino(vec4<f32>(eIn.q.x, eIn.q.y, -eIn.q.z + mirrorErr + PI_RESIDUAL, eIn.q.w), vec4<f32>(eIn.v.x, eIn.v.y, -eIn.v.z, 0.0));
    }
  }
  let hMax = select(1e30, MINO_UFRAC * PI / c.omMax, c.omMax > 0.0);
  var h = min(h0, hMax); var att = 1u;
  loop {
    let t = cTry(y, ey, f0, h, c); let en = t.en; let finite = en < 1e30;
    let fac = select(select(0.2, 0.9 * pow(max(en, 1e-30), -0.2), finite), 5.0, en == 0.0);
    if (en <= 1.0 || att > MINO_MAX_REJECT) { return CStep(y, ey, t.y1, t.e1, f0, t.f1, h, min(hMax, h * min(5.0, max(0.2, fac))), att); }
    h = h * max(0.2, min(0.9, fac)); att = att + 1u;
  }
}
// Landing: minoLand / minoLandW with the compensated step.
fn cLand(st: CStep, th: f32, c: MinoRay) -> Mino {
  let c0 = sinCosP(st.y0.q.z).y; var lo = 0.0; var hi = st.h; var s = th * st.h;
  for (var k = 0u; k < MINO_LAND_ITERS; k++) {
    let t = cTry(st.y0, st.e0, st.f0, s, c); let sc = sinCosP(t.y1.q.z);
    if (sc.y * c0 > 0.0) { lo = s; } else { hi = s; }
    let sn = s - sc.y / (-sc.x * t.f1.q.z); s = select(0.5 * (lo + hi), sn, sn >= lo && sn <= hi);
  }
  return cTry(st.y0, st.e0, st.f0, s, c).y1;
}
fn cLandW(st: CStep, th: f32, c: MinoRay, wT: f32) -> Mino {
  let g0 = st.y0.q.y - wT; var lo = 0.0; var hi = st.h; var s = th * st.h;
  for (var k = 0u; k < MINO_LAND_ITERS; k++) {
    let t = cTry(st.y0, st.e0, st.f0, s, c); let g = t.y1.q.y - wT;
    if (g * g0 > 0.0) { lo = s; } else { hi = s; }
    let sn = s - g / t.f1.q.y; s = select(0.5 * (lo + hi), sn, sn >= lo && sn <= hi);
  }
  return cTry(st.y0, st.e0, st.f0, s, c).y1;
}
@compute @workgroup_size(64) fn minoComp(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = U.first + gid.x; if (gid.x >= U.count) { return; }
  let a = U.a; let ray = rays[n]; let xe = cameraXiEta(ray.alpha, ray.beta, a, U.incl);
  let c = minoRay(a, xe.x, xe.y, U.rObs);
  var y = minoInit(U.rObs, U.incl, ray.beta, c); var ey = Mino(vec4<f32>(0.0), vec4<f32>(0.0)); var f = minoRhs(y, c); var h = 50.0 / (U.rObs * U.rObs);
  let rh = 1.0 + sqrt(max(0.0, 1.0 - a * a)); let wCap = 1.0 / (rh * 1.005); let wEsc = 1.0 / (U.rObs * 1.2);
  var o = RayOut(4u, U.maxSteps, 0u, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var step = 0u; step < U.maxSteps; step++) {
    let st = cStep(y, ey, f, h, c); o.attempts += st.attempts;
    o.pad = max(o.pad, abs(st.e.v.y));
    let th = minoCrossing(st.y0, st.y, st.f0, st.f1, st.h);
    if (th >= 0.0) {
      let rd = 1.0 / minoDense(st.y0, st.y, st.f0, st.f1, st.h, th).q.y;
      if (rd >= U.rIn * (1.0 - MINO_LAND_BAND) && rd <= U.rOut * (1.0 + MINO_LAND_BAND)) {
        let hit = cLand(st, th, c); let rHit = 1.0 / hit.q.y;
        if (rHit >= U.rIn && rHit <= U.rOut) { o.word = 1u; o.steps = step + 1u; o.rHit = rHit; o.phi = minoToState(hit, c).x.w; o.delay = minoDelay(hit); break; }
      }
    }
    y = st.y; ey = st.e; f = st.f1; h = st.hNext;
    if (y.q.y >= wCap) { o.word = 0u; o.steps = step + 1u; break; }
    if (st.y.q.y < wEsc) {
      var land = st.y0;
      if (st.y0.q.y >= wEsc) {
        var lo = 0.0; var hi = 1.0;
        for (var k = 0u; k < 24u; k++) { let m = 0.5 * (lo + hi); if (minoDense(st.y0, st.y, st.f0, st.f1, st.h, m).q.y >= wEsc) { lo = m; } else { hi = m; } }
        land = cLandW(st, 0.5 * (lo + hi), c, wEsc);
      }
      let d = limitSkyDir(minoToState(land, c), a); o.word = 2u; o.steps = step + 1u; o.dx = d.x; o.dy = d.y; o.dz = d.z; o.w = land.q.y; o.wp = land.v.y; o.v = land.q.z; o.vp = land.v.z; o.sigma = land.v.w; o.ph = land.q.w; break;
    }
  }
  out[n] = o;
}
