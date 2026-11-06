// ?limitds: the Mino DP5(4) integrator with every quantity in double-single (a pair of f32, about 48 bits): state, stage sums, force, drift
// check, landing. The "why not just emulate f64?" baseline. A debug route's own module; nothing here is compiled into the renderer.
// Prepended: shadow-shared, camera-shared, integrator-shared (for the error norm, dense output and crossing search, which act on the hi
// parts), and the generated constants (D_*, K_*; physics/double-single.ts dsWgslConstants). CPU twin of the arithmetic: double-single.ts.
struct DU { a: vec2<f32>, si: vec2<f32>, ci: vec2<f32>, v0: vec2<f32>, rObs: f32, rIn: f32, rOut: f32, maxSteps: u32, first: u32, count: u32, pad0: u32, pad1: u32 };
struct DsIn { alpha: f32, beta: f32 };
struct DsOut { word: u32, steps: u32, attempts: u32, sigma: f32, r: vec2<f32>, phi: vec2<f32>, delay: vec2<f32>, w: vec2<f32>, wp: vec2<f32>, v: vec2<f32>, vp: vec2<f32> };
@group(0) @binding(0) var<uniform> U: DU;
@group(0) @binding(1) var<storage, read> rays: array<DsIn>;
@group(0) @binding(2) var<storage, read_write> out: array<DsOut>;

// ---- scalars ----
fn dTwoSum(a: f32, b: f32) -> vec2<f32> { let s = a + b; let bb = s - a; return vec2<f32>(s, (a - (s - bb)) + (b - bb)); }
fn dQuick(a: f32, b: f32) -> vec2<f32> { let s = a + b; return vec2<f32>(s, b - (s - a)); }
fn dSplit(a: f32) -> vec2<f32> { let t = 4097.0 * a; let h = t - (t - a); return vec2<f32>(h, a - h); }
fn dTwoProd(a: f32, b: f32) -> vec2<f32> {
  let p = a * b; let A = dSplit(a); let B = dSplit(b);
  return vec2<f32>(p, ((((A.x * B.x) - p) + (A.x * B.y)) + (A.y * B.x)) + (A.y * B.y));
}
fn dAdd(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let s = dTwoSum(a.x, b.x); let t = dTwoSum(a.y, b.y); let v = dQuick(s.x, s.y + t.x);
  return dQuick(v.x, t.y + v.y);
}
fn dNeg(a: vec2<f32>) -> vec2<f32> { return -a; }
fn dSub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return dAdd(a, -b); }
fn dMul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { let p = dTwoProd(a.x, b.x); return dQuick(p.x, p.y + ((a.x * b.y) + (a.y * b.x))); }
fn dF(x: f32) -> vec2<f32> { return vec2<f32>(x, 0.0); }
fn dMulF(a: vec2<f32>, x: f32) -> vec2<f32> { return dMul(a, dF(x)); }
fn dDiv(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let q1 = a.x / b.x; var r = dSub(a, dMul(dF(q1), b));
  let q2 = r.x / b.x; r = dSub(r, dMul(dF(q2), b));
  let q3 = r.x / b.x;
  return dAdd(dQuick(q1, q2), dF(q3));
}
fn dSqrt(a: vec2<f32>) -> vec2<f32> {
  if (a.x <= 0.0) { return vec2<f32>(0.0); }
  let x = 1.0 / sqrt(a.x); let ax = a.x * x;
  return dAdd(dF(ax), dF(dSub(a, dMul(dF(ax), dF(ax))).x * (x * 0.5)));
}
struct DSC { s: vec2<f32>, c: vec2<f32> };
fn dSinCos(x: vec2<f32>) -> DSC {
  let k = round(x.x / D_PIO2.x);
  var y = x; if (k != 0.0) { y = dSub(x, dMul(D_PIO2, dF(k))); }
  let z = dMul(y, y);
  var s = D_C15; s = dAdd(dMul(-s, z), D_C13); s = dAdd(dMul(-s, z), D_C11); s = dAdd(dMul(-s, z), D_C9); s = dAdd(dMul(-s, z), D_C7);
  s = dAdd(dMul(-s, z), D_C5); s = dAdd(dMul(-s, z), D_C3); s = dAdd(dMul(-s, z), D_C1); s = dMul(s, y);
  var c = D_C16; c = dAdd(dMul(-c, z), D_C14); c = dAdd(dMul(-c, z), D_C12); c = dAdd(dMul(-c, z), D_C10); c = dAdd(dMul(-c, z), D_C8);
  c = dAdd(dMul(-c, z), D_C6); c = dAdd(dMul(-c, z), D_C4); c = dAdd(dMul(-c, z), D_C2); c = dAdd(dMul(-c, z), D_C0);
  let q = ((i32(k) % 4) + 4) % 4;
  if (q == 0) { return DSC(s, c); }
  if (q == 1) { return DSC(c, -s); }
  if (q == 2) { return DSC(-s, -c); }
  return DSC(-c, s);
}
fn dLog(x: vec2<f32>) -> vec2<f32> {
  let e = round(log2(x.x)); let m = vec2<f32>(ldexp(x.x, i32(-e)), ldexp(x.y, i32(-e)));
  let z = dDiv(dSub(m, dF(1.0)), dAdd(m, dF(1.0))); let z2 = dMul(z, z);
  var s = D_L9; s = dAdd(dMul(s, z2), D_L8); s = dAdd(dMul(s, z2), D_L7); s = dAdd(dMul(s, z2), D_L6); s = dAdd(dMul(s, z2), D_L5);
  s = dAdd(dMul(s, z2), D_L4); s = dAdd(dMul(s, z2), D_L3); s = dAdd(dMul(s, z2), D_L2); s = dAdd(dMul(s, z2), D_L1); s = dAdd(dMul(s, z2), D_L0);
  return dAdd(dMulF(dMul(z, s), 2.0), dMulF(D_LN2, e));
}
// a >= b, on the pair's value
fn dGe(a: vec2<f32>, b: vec2<f32>) -> bool { return dSub(a, b).x >= 0.0; }

// ---- eight-component state: q = (tau, w, v, phi), p = (ell, w', v', sigma), each a hi and a lo vec4 ----
struct DM { qh: vec4<f32>, ql: vec4<f32>, ph: vec4<f32>, pl: vec4<f32> };
fn dAt(h: vec4<f32>, l: vec4<f32>, i: u32) -> vec2<f32> { return vec2<f32>(h[i], l[i]); }
fn dmHi(y: DM) -> Mino { return Mino(y.qh, y.ph); }
fn dmFrom(q: array<vec2<f32>, 4>, p: array<vec2<f32>, 4>) -> DM {
  return DM(vec4<f32>(q[0].x, q[1].x, q[2].x, q[3].x), vec4<f32>(q[0].y, q[1].y, q[2].y, q[3].y), vec4<f32>(p[0].x, p[1].x, p[2].x, p[3].x), vec4<f32>(p[0].y, p[1].y, p[2].y, p[3].y));
}
// y + c * k, component by component
fn dmAcc(y: DM, c: vec2<f32>, k: DM) -> DM {
  var q: array<vec2<f32>, 4>; var p: array<vec2<f32>, 4>;
  for (var i = 0u; i < 4u; i++) {
    q[i] = dAdd(dAt(y.qh, y.ql, i), dMul(c, dAt(k.qh, k.ql, i)));
    p[i] = dAdd(dAt(y.ph, y.pl, i), dMul(c, dAt(k.ph, k.pl, i)));
  }
  return dmFrom(q, p);
}
fn dmZero() -> DM { return DM(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)); }

struct DRay { a: vec2<f32>, a2: vec2<f32>, xi: vec2<f32>, xi2: vec2<f32>, eta: vec2<f32>, K: vec2<f32>, A1: vec2<f32>, c2: vec2<f32>, c3: vec2<f32>, c4: vec2<f32>,
  t0: vec2<f32>, t1: vec2<f32>, t2: vec2<f32>, t3: vec2<f32>, t4: vec2<f32>, t5: vec2<f32>, omMax: f32 };
fn dRay(a: vec2<f32>, xi: vec2<f32>, eta: vec2<f32>) -> DRay {
  let a2 = dMul(a, a); let xi2 = dMul(xi, xi); let a4 = dMul(a2, a2); let xma = dSub(xi, a);
  let K = dAdd(eta, dMul(xma, xma)); let A1 = dSub(a2, dMul(a, xi)); let A12 = dMul(A1, A1);
  let om2 = max(0.0, eta.x + xi2.x - a2.x) + 6.0 * a2.x;
  let K2 = dMulF(K, 2.0); let K4 = dMulF(K, 4.0); let K8 = dMulF(K, 8.0); let A1a2 = dMul(A1, a2); let Ka2 = dMul(K, a2); let Ka4 = dMul(K, a4); let A12a2 = dMul(A12, a2);
  return DRay(a, a2, xi, xi2, eta, K, A1, dSub(dMulF(A1, 2.0), K), K2, dSub(A12, Ka2),
    dSub(dSub(A1, K), dF(4.0)), dAdd(K2, dMulF(a2, 2.0)), dAdd(dSub(dSub(dAdd(A12, A1a2), dMulF(A1, 8.0)), dMulF(Ka2, 2.0)), K4),
    dSub(dMulF(A1a2, 4.0), K8), dAdd(dSub(dSub(A12a2, dMulF(A12, 4.0)), Ka4), dMul(K8, a2)), dSub(dMulF(A12a2, 2.0), dMulF(Ka4, 2.0)), sqrt(om2));
}
fn dRw(w: vec2<f32>, c: DRay) -> vec2<f32> {
  let w2 = dMul(w, w); let Q = dAdd(dF(1.0), dMul(c.A1, w2));
  return dSub(dMul(Q, Q), dMul(dMul(c.K, w2), dAdd(dSub(dF(1.0), dMulF(w, 2.0)), dMul(c.a2, w2))));
}
fn dRhs(y: DM, c: DRay) -> DM {
  let w = dAt(y.qh, y.ql, 1u); let v = dAt(y.qh, y.ql, 2u); let sc = dSinCos(v); let s = sc.s; let co = sc.c;
  var s2 = dMul(s, s); if (s2.x < 1e-30) { s2 = dF(1e-30); }
  let w2 = dMul(w, w); let a = c.a; let a2 = c.a2;
  let Q = dAdd(dF(1.0), dMul(c.A1, w2)); let Dw = dAdd(dSub(dF(1.0), dMulF(w, 2.0)), dMul(a2, w2));
  let P = dAdd(c.t0, dMul(w, dAdd(c.t1, dMul(w, dAdd(c.t2, dMul(w, dAdd(c.t3, dMul(w, dAdd(c.t4, dMul(w, c.t5))))))))));
  let rp = dAdd(c.c2, dMul(w, dAdd(dMulF(c.c3, 1.5), dMul(w, dMulF(c.c4, 2.0)))));
  let tauP = dSub(dAdd(dDiv(P, Dw), dMul(rp, dSub(dMul(dMulF(w, 2.0), dLog(w)), dF(1.0)))), dMul(a, dSub(c.xi, dMul(a, s2))));
  let phiP = -dAdd(dSub(dDiv(dMul(a, Q), Dw), a), dDiv(c.xi, s2));
  let ellP = dAdd(dMul(dMul(a2, co), co), dMul(w, dAdd(dMulF(c.c3, 0.5), dMul(w, c.c4))));
  let wpp = dSub(dMul(dMul(dMulF(c.A1, 2.0), w), Q), dMul(c.K, dAdd(dSub(w, dMulF(w2, 3.0)), dMul(dMul(dMulF(a2, 2.0), w2), w))));
  let vpp = dMul(co, dSub(dDiv(c.xi2, dMul(s2, s)), dMul(a2, s)));
  return DM(vec4<f32>(tauP.x, y.ph.y, y.ph.z, phiP.x), vec4<f32>(tauP.y, y.pl.y, y.pl.z, phiP.y), vec4<f32>(ellP.x, wpp.x, vpp.x, 0.0), vec4<f32>(ellP.y, wpp.y, vpp.y, 0.0));
}
// Mirror past the equator, with pi to double-single.
fn dHemi(y: DM) -> DM {
  if (y.qh.z <= 0.5 * PI) { return y; }
  let v = dSub(D_PI, dAt(y.qh, y.ql, 2u));
  return DM(vec4<f32>(y.qh.x, y.qh.y, v.x, y.qh.w), vec4<f32>(y.ql.x, y.ql.y, v.y, y.ql.w),
            vec4<f32>(y.ph.x, y.ph.y, -y.ph.z, -y.ph.w), vec4<f32>(y.pl.x, y.pl.y, -y.pl.z, -y.pl.w));
}
// First-integral drift over a step (minoDrift) with the differences in double-single, against max(tolerance, this arithmetic's noise).
const D_EPS = 1.42e-14; // 2^-46
fn dDriftNorm(y0: DM, y1: DM, f1: DM, c: DRay, tol: f32) -> f32 {
  let w0 = dAt(y0.qh, y0.ql, 1u); let w1 = dAt(y1.qh, y1.ql, 1u); let p0 = dAt(y0.ph, y0.pl, 1u); let p1 = dAt(y1.ph, y1.pl, 1u);
  let sw = dAdd(w1, w0); let w12 = dMul(w1, w1); let w02 = dMul(w0, w0);
  let dR = dMul(dSub(w1, w0), dAdd(dAdd(dMul(c.c2, sw), dMul(c.c3, dAdd(dAdd(w12, dMul(w1, w0)), w02))), dMul(dMul(c.c4, sw), dAdd(w12, w02))));
  let dW = dSub(dMul(dSub(p1, p0), dAdd(p1, p0)), dR);
  let v0 = dAt(y0.qh, y0.ql, 2u); let v1 = dAt(y1.qh, y1.ql, 2u); let q0 = dAt(y0.ph, y0.pl, 2u); let q1 = dAt(y1.ph, y1.pl, 2u);
  let s0 = dSinCos(v0).s; let sc1 = dSinCos(v1); let s1 = sc1.s; let co1 = sc1.c.x;
  var ss = dMul(dMul(s0, s0), dMul(s1, s1)); if (ss.x < 1e-37) { ss = dF(1e-37); }
  let dTh = dMul(dMul(dSinCos(dSub(v1, v0)).s, dSinCos(dAdd(v1, v0)).s), dSub(dDiv(c.xi2, ss), c.a2));
  let dV = dSub(dMul(dSub(q1, q0), dAdd(q1, q0)), dTh);
  let w = w1.x; let Q = 1.0 + c.A1.x * w * w;
  let scaleW = p1.x * p1.x + Q * Q + abs(c.K.x * w * w * (1.0 - 2.0 * w + c.a2.x * w * w));
  let scaleT = q1.x * q1.x + abs(c.eta.x) + c.a2.x * co1 * co1 + c.xi2.x * co1 * co1 / max(s1.x * s1.x, 1e-30);
  let noiseW = 4.0 * D_EPS * (2.0 * abs(f1.ph.y) * w + abs(p1.x) * (abs(p1.x) + abs(p0.x)));
  let noiseT = 4.0 * D_EPS * (2.0 * abs(f1.ph.z) * abs(v1.x) + abs(q1.x) * (abs(q1.x) + abs(q0.x)));
  return max(abs(dW.x) / max(MINO_CTOL * tol * scaleW, noiseW), abs(dV.x) / max(MINO_CTOL * tol * scaleT, noiseT));
}
struct DTry { y1: DM, f1: DM, en: f32 };
// One DP5(4) attempt: y + h * sum(a_i k_i) with the coefficients and the step size in double-single.
fn dStage(y: DM, h: vec2<f32>, d: DM) -> DM { return dmAcc(y, h, d); }
fn dTry(y: DM, f0: DM, h: vec2<f32>, c: DRay) -> DTry {
  let z = dmZero(); let k1 = f0;
  let k2 = dRhs(dStage(y, h, dmAcc(z, K_A21, k1)), c);
  let k3 = dRhs(dStage(y, h, dmAcc(dmAcc(z, K_A31, k1), K_A32, k2)), c);
  let k4 = dRhs(dStage(y, h, dmAcc(dmAcc(dmAcc(z, K_A41, k1), K_A42, k2), K_A43, k3)), c);
  let k5 = dRhs(dStage(y, h, dmAcc(dmAcc(dmAcc(dmAcc(z, K_A51, k1), K_A52, k2), K_A53, k3), K_A54, k4)), c);
  let k6 = dRhs(dStage(y, h, dmAcc(dmAcc(dmAcc(dmAcc(dmAcc(z, K_A61, k1), K_A62, k2), K_A63, k3), K_A64, k4), K_A65, k5)), c);
  let y1 = dStage(y, h, dmAcc(dmAcc(dmAcc(dmAcc(dmAcc(z, K_B1, k1), K_B3, k3), K_B4, k4), K_B5, k5), K_B6, k6));
  let k7 = dRhs(y1, c);
  let e = dStage(z, h, dmAcc(dmAcc(dmAcc(dmAcc(dmAcc(dmAcc(z, K_E1, k1), K_E3, k3), K_E4, k4), K_E5, k5), K_E6, k6), K_E7, k7));
  let en = select(3e38, max(minoErrNorm(dmHi(y), dmHi(y1), dmHi(e), MINO_TOL), dDriftNorm(y, y1, k7, c, MINO_TOL)), y1.qh.y > 0.0);
  return DTry(y1, k7, en);
}
struct DStep { y0: DM, y: DM, f0: DM, f1: DM, h: f32, hNext: f32, attempts: u32 };
fn dStepOnce(yIn: DM, f0In: DM, h0: f32, c: DRay) -> DStep {
  let y = dHemi(yIn); var f0 = f0In; if (yIn.qh.z > 0.5 * PI) { f0 = dRhs(y, c); }
  let hMax = select(1e30, MINO_UFRAC * PI / c.omMax, c.omMax > 0.0);
  var h = min(h0, hMax); var att = 1u;
  loop {
    let t = dTry(y, f0, dF(h), c); let en = t.en; let finite = en < 1e30;
    let fac = select(select(0.2, 0.9 * pow(max(en, 1e-30), -0.2), finite), 5.0, en == 0.0);
    if (en <= 1.0 || att > MINO_MAX_REJECT) { return DStep(y, t.y1, f0, t.f1, h, min(hMax, h * min(5.0, max(0.2, fac))), att); }
    h = h * max(0.2, min(0.9, fac)); att = att + 1u;
  }
}
// The landings: safeguarded Newton with the step size itself in double-single, so the hit is good to the arithmetic.
fn dLand(st: DStep, th: f32, c: DRay) -> DM {
  let c0 = dSinCos(dAt(st.y0.qh, st.y0.ql, 2u)).c.x; var lo = dF(0.0); var hi = dF(st.h); var s = dMulF(dF(st.h), th);
  for (var k = 0u; k < MINO_LAND_ITERS + 1u; k++) {
    let t = dTry(st.y0, st.f0, s, c); let sc = dSinCos(dAt(t.y1.qh, t.y1.ql, 2u));
    if (sc.c.x * c0 > 0.0) { lo = s; } else { hi = s; }
    let sn = dSub(s, dDiv(sc.c, dMul(-sc.s, dAt(t.f1.qh, t.f1.ql, 2u))));
    if (dGe(sn, lo) && dGe(hi, sn)) { s = sn; } else { s = dMulF(dAdd(lo, hi), 0.5); }
  }
  return dTry(st.y0, st.f0, s, c).y1;
}
fn dLandW(st: DStep, th: f32, c: DRay, wT: vec2<f32>) -> DM {
  let g0 = dSub(dAt(st.y0.qh, st.y0.ql, 1u), wT).x; var lo = dF(0.0); var hi = dF(st.h); var s = dMulF(dF(st.h), th);
  for (var k = 0u; k < MINO_LAND_ITERS + 1u; k++) {
    let t = dTry(st.y0, st.f0, s, c); let g = dSub(dAt(t.y1.qh, t.y1.ql, 1u), wT);
    if (g.x * g0 > 0.0) { lo = s; } else { hi = s; }
    let sn = dSub(s, dDiv(g, dAt(t.f1.qh, t.f1.ql, 1u)));
    if (dGe(sn, lo) && dGe(hi, sn)) { s = sn; } else { s = dMulF(dAdd(lo, hi), 0.5); }
  }
  return dTry(st.y0, st.f0, s, c).y1;
}
fn dFill(o: ptr<function, DsOut>, y: DM) {
  (*o).w = dAt(y.qh, y.ql, 1u); (*o).wp = dAt(y.ph, y.pl, 1u); (*o).v = dAt(y.qh, y.ql, 2u); (*o).vp = dAt(y.ph, y.pl, 2u);
  (*o).phi = dAt(y.qh, y.ql, 3u); (*o).sigma = y.ph.w;
}
@compute @workgroup_size(64) fn trace(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = U.first + gid.x; if (gid.x >= U.count) { return; }
  let ray = rays[n]; let a = U.a; let al = dF(ray.alpha); let be = dF(ray.beta);
  // The camera (camera-shared.wgsl cameraXiEta) in double-single
  let xi = dMul(-al, U.si); let c2i = dMul(U.ci, U.ci); var s2i = dMul(U.si, U.si); if (s2i.x < 1e-8) { s2i = dF(1e-8); }
  let eta = dSub(dAdd(dMul(be, be), dDiv(dMul(dMul(xi, xi), c2i), s2i)), dMul(dMul(a, a), c2i));
  let c = dRay(a, xi, eta);
  // minoInit
  let w0 = dDiv(dF(1.0), dF(U.rObs)); let wp0 = dSqrt(dRw(w0, c));
  let poly = dMul(w0, dAdd(c.c2, dMul(w0, dAdd(c.c3, dMul(w0, c.c4))))); let denom = dAdd(dF(1.0), wp0);
  let tau0 = dAdd(dDiv(-poly, denom), dMul(dMulF(wp0, 2.0), dLog(w0))); let ell0 = dDiv(poly, denom);
  var y = dHemi(dmFrom(array<vec2<f32>, 4>(tau0, w0, U.v0, dF(0.0)), array<vec2<f32>, 4>(ell0, wp0, -be, dF(1.0))));
  var f = dRhs(y, c); var h = 50.0 / (U.rObs * U.rObs);
  let rh = 1.0 + sqrt(max(0.0, 1.0 - a.x * a.x)); let wCap = 1.0 / (rh * 1.005); let wEsc = dDiv(dF(1.0), dF(U.rObs * 1.2));
  var o: DsOut; o.word = 4u; o.steps = U.maxSteps; o.attempts = 0u;
  for (var step = 0u; step < U.maxSteps; step++) {
    let st = dStepOnce(y, f, h, c); o.attempts += st.attempts;
    let m0 = dmHi(st.y0); let m1 = dmHi(st.y); let g0 = dmHi(st.f0); let g1 = dmHi(st.f1);
    let th = minoCrossing(m0, m1, g0, g1, st.h);
    if (th >= 0.0) {
      let rd = 1.0 / minoDense(m0, m1, g0, g1, st.h, th).q.y;
      if (rd >= U.rIn * (1.0 - MINO_LAND_BAND) && rd <= U.rOut * (1.0 + MINO_LAND_BAND)) {
        let hit = dLand(st, th, c); let rHit = dDiv(dF(1.0), dAt(hit.qh, hit.ql, 1u));
        if (rHit.x >= U.rIn && rHit.x <= U.rOut) {
          let hw = dAt(hit.qh, hit.ql, 1u); let hp = dAt(hit.ph, hit.pl, 1u);
          let F = dMul(hp, dSub(dMulF(dLog(hw), 2.0), dDiv(dF(1.0), hw)));
          o.word = 1u; o.steps = step + 1u; o.r = rHit; o.delay = dSub(F, dAt(hit.qh, hit.ql, 0u)); dFill(&o, hit); break;
        }
      }
    }
    y = st.y; f = st.f1; h = st.hNext;
    if (y.qh.y >= wCap) { o.word = 0u; o.steps = step + 1u; break; }
    if (st.y.qh.y < wEsc.x) {
      var land = st.y0;
      if (st.y0.qh.y >= wEsc.x) {
        var lo = 0.0; var hi = 1.0;
        for (var k = 0u; k < 24u; k++) { let t = 0.5 * (lo + hi); if (minoDense(m0, m1, g0, g1, st.h, t).q.y >= wEsc.x) { lo = t; } else { hi = t; } }
        land = dLandW(st, 0.5 * (lo + hi), c, wEsc);
      }
      o.word = 2u; o.steps = step + 1u; dFill(&o, land); break;
    }
  }
  out[n] = o;
}
// Every operation on given operands, for the route to check against exact values: a compiler that simplifies the error terms would break this.
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = U.first + gid.x; if (gid.x >= U.count) { return; }
  let x = vec2<f32>(rays[2u * n].alpha, rays[2u * n].beta); let y = vec2<f32>(rays[2u * n + 1u].alpha, rays[2u * n + 1u].beta);
  let sc = dSinCos(x); var o: DsOut; o.word = 0u; o.steps = 0u; o.attempts = 0u; o.sigma = 0.0;
  o.r = dAdd(x, y); o.phi = dMul(x, y); o.delay = dDiv(x, y); o.w = dSqrt(vec2<f32>(abs(x.x), x.y * sign(x.x))); o.wp = sc.s; o.v = sc.c; o.vp = dLog(vec2<f32>(abs(x.x), x.y * sign(x.x)));
  out[n] = o;
}
