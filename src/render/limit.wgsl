// ?limit: the same rays through a 5th-order and a 4th-order integrator in f32, reporting what the renderer would use. A debug
// route's own module (src/test/limit.browser.ts); nothing here is compiled into the renderer. Prepended: shadow-shared,
// camera-shared, integrator-shared (MINO_TOL, H_TOL-style knobs are this module's: the route rewrites MINO_TOL in its own copy).
// `mino` is traceRay's loop with the emitters removed (as drift.wgsl). `rk4arm` is the pre-Mino renderer's integrator, twin of
// trace-reference.ts refTrace with RefCfg { scale, hNear, hFar, landEsc: true }, including its LINEAR chord interpolation of the disk hit.
struct LU { a: f32, incl: f32, rObs: f32, rIn: f32, rOut: f32, scale: f32, hNear: f32, hFar: f32, maxSteps: u32, first: u32, count: u32, pad: u32 };
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
