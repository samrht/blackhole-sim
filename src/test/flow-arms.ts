// The shader variants of the ?imgflow experiment: the renderer's own 1.3 mm flow loop with only the integrator changed. The validated kernels
// (limit.wgsl, limit-ds.wgsl) are cut at markers rather than refactored, so the results already recorded from them stay reproducible.

export function sliceBetween(src: string, from: string, to: string): string {
  const a = src.indexOf(from); if (a < 0) throw new Error(`shader marker not found: ${from}`);
  const b = src.indexOf(to, a); if (b < 0) throw new Error(`shader marker not found: ${to}`);
  return src.slice(a, b);
}
/** limit.wgsl's compensated integrator, with the route's uniform switch replaced by a constant. */
export function compPart(limit: string, on = true): string {
  const twoSum = sliceBetween(limit, "struct TS4 {", "// A compiler that simplifies");
  const rest = sliceBetween(limit, "const PI_RESIDUAL", "@compute @workgroup_size(64) fn minoComp");
  return (twoSum + rest).split("U.comp == 1u").join(on ? "true" : "false");
}
/** limit-ds.wgsl's arithmetic and step, without its bindings, uniform or entry points. */
export const dsPart = (ds: string): string => sliceBetween(ds, "// ---- scalars ----", "fn dFill(");

export function withTol(integrator: string, tol: number): string {
  const s = integrator.replace(/const MINO_TOL = 1e-5;/, `const MINO_TOL = ${tol.toExponential().replace("e-", "e-")};`);
  if (!/const MINO_TOL = [^;]+;/.test(integrator)) throw new Error("MINO_TOL not found");
  return s;
}
export function withFlowDl(raytrace: string, dl: number): string {
  if (!raytrace.includes("const FLOW_DL = 0.125;")) throw new Error("FLOW_DL not found");
  return raytrace.replace("const FLOW_DL = 0.125;", `const FLOW_DL = ${dl};`);
}

// Both entries follow raytrace.wgsl's build and traceRay in mm mode: the flow on each step, then capture, the exit at r = HF_RMAX, then the sphere.
const HEAD = `  let py = U.rowStart + gid.y;
  if (gid.x >= u32(U.res.x) || py >= U.rowEnd || py >= u32(U.res.y)) { return; }
  let ab = pixelImpact(vec2<u32>(gid.x, py), fixedJitter(U.setIndex));
  let rh = 1.0 + sqrt(max(0.0, 1.0 - U.a * U.a)); let wCap = 1.0 / (rh * 1.005); let wEsc = 1.0 / (U.rObs * 1.2);
  let orb = iscoOrbitJ(U.a); var flow = vec2<f32>(0.0); var kind = KIND_SHADOW;`;
const TAIL = `  entries[py * u32(U.res.x) + gid.x] = Entry(kind | (BM_NONE << 2u), flow.x, 0.0, 0.0);`;
const CHECKS = `    if (w1 >= wCap) { kind = KIND_SHADOW; break; }
    if (w1 < 1.0 / HF_RMAX && w1 < wPrev) { kind = KIND_SKY; break; }
    if (w1 < wEsc) { kind = KIND_SKY; break; }`;
export const FLOW_ENTRIES = {
  comp: `
@compute @workgroup_size(8, 8) fn flowArm(@builtin(global_invocation_id) gid: vec3<u32>) {
${HEAD}
  let xe = cameraXiEta(ab.x, ab.y, U.a, U.incl); let c = minoRay(U.a, xe.x, xe.y, U.rObs);
  var y = minoInit(U.rObs, U.incl, ab.y, c); var ey = Mino(vec4<f32>(0.0), vec4<f32>(0.0)); var f = minoRhs(y, c); var h = 50.0 / (U.rObs * U.rObs);
  for (var step = 0u; step < U.maxSteps; step++) {
    let st = cStep(y, ey, f, h, c);
    flow = flowSeg(Seg(st.y0, st.y, st.f0, st.f1, st.h), c, rh, orb, flow, false);
    let wPrev = y.q.y; y = st.y; ey = st.e; f = st.f1; h = st.hNext; let w1 = y.q.y;
${CHECKS}
  }
${TAIL}
}
`,
  // The path in double-single, handed to the flow as its f32 hi parts: the emission stays exactly the renderer's.
  ds: `
@compute @workgroup_size(8, 8) fn flowArm(@builtin(global_invocation_id) gid: vec3<u32>) {
${HEAD}
  let a = V_A; let al = dF(ab.x); let be = dF(ab.y);
  let xi = dMul(-al, V_SI); let c2i = dMul(V_CI, V_CI); var s2i = dMul(V_SI, V_SI); if (s2i.x < 1e-8) { s2i = dF(1e-8); }
  let eta = dSub(dAdd(dMul(be, be), dDiv(dMul(dMul(xi, xi), c2i), s2i)), dMul(dMul(a, a), c2i));
  let dc = dRay(a, xi, eta); let c = minoRay(U.a, xi.x, eta.x, U.rObs);
  let w0 = dDiv(dF(1.0), dF(U.rObs)); let wp0 = dSqrt(dRw(w0, dc));
  let poly = dMul(w0, dAdd(dc.c2, dMul(w0, dAdd(dc.c3, dMul(w0, dc.c4))))); let denom = dAdd(dF(1.0), wp0);
  let tau0 = dAdd(dDiv(-poly, denom), dMul(dMulF(wp0, 2.0), dLog(w0))); let ell0 = dDiv(poly, denom);
  var y = dHemi(dmFrom(array<vec2<f32>, 4>(tau0, w0, V_V0, dF(0.0)), array<vec2<f32>, 4>(ell0, wp0, -be, dF(1.0))));
  var f = dRhs(y, dc); var h = 50.0 / (U.rObs * U.rObs);
  for (var step = 0u; step < U.maxSteps; step++) {
    let st = dStepOnce(y, f, h, dc);
    flow = flowSeg(Seg(dmHi(st.y0), dmHi(st.y), dmHi(st.f0), dmHi(st.f1), st.h), c, rh, orb, flow, false);
    let wPrev = y.qh.y; y = st.y; f = st.f1; h = st.hNext; let w1 = y.qh.y;
${CHECKS}
  }
${TAIL}
}
`,
};
