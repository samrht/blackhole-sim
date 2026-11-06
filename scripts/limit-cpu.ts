// CPU side of the ?limit experiment: the rays of one configuration, the converged reference and its Jacobians, and every f64 arm.
//   npx vite-node scripts/limit-cpu.ts <config>     (one config per process; run them side by side)
// Output: docs/diagnostics/2026-11-06-accuracy-limit/data/<config>.json. The GPU arms read the rays from it (src/test/limit.browser.ts).
import { mkdirSync, writeFileSync } from "node:fs";
import { minoRay, minoTrace, minoTraceRay, setMinoMethod } from "../src/physics/trace-mino";
import { convergedRef, refTrace, skyDirCPU, SHIP, REF_ROBS, type RefCfg } from "../src/physics/trace-reference";
import { screenToState } from "../src/physics/camera";
import { iscoRadius } from "../src/physics/orbits";
import { classify, shadowBoundary } from "../src/physics/shadow";
import { curveDistance } from "../src/test/drift-analysis";
import { f32Inputs, jacobianOf, obsVector, rayError, roundObs, summarizeArm, type ArmRow, type Obs } from "../src/test/limit-score";
import { CONFIGS, F64_TOLS, REF_TOL, RK4_SCALES, PX_M, H_NEAR, H_FAR, rk4Budget, groupsOf, BS3_TOLS, BS3_BUDGET } from "../src/test/limit-config";

const name = process.argv[2], cfg = CONFIGS.find((c) => c.name === name);
if (!cfg) throw new Error(`config? ${CONFIGS.map((c) => c.name).join(" ")}`);
const { a, inclDeg } = cfg, incl = (inclDeg * Math.PI) / 180, si = Math.sin(incl), ci = Math.cos(incl);
const rIn = cfg.disk ? iscoRadius(a, true) : 1e9, rOut = 40, F = Math.fround, t0 = Date.now();
const log = (s: string) => console.log(`[${name} ${((Date.now() - t0) / 60000).toFixed(1)} min] ${s}`);

// ---- Rays: a grid over the app's view, then near-critical rays at log-spaced distances from the curve. All f32-representable, so an
// arm's error is the arm's own and not the rounding of the pixel coordinate. ----
const curve = shadowBoundary(a, incl, 200000), cen = curve.reduce((s, p) => [s[0] + p[0] / curve.length, s[1] + p[1] / curve.length], [0, 0]);
const rays: [number, number][] = [];
for (let j = 0; j < 36; j++) for (let i = 0; i < 64; i++) rays.push([F((((i + 0.5) / 64) * 2 - 1) * 24.9), F((((j + 0.5) / 36) * 2 - 1) * 14)]);
let seed = 12345; const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
for (let k = 0; k < 1700; k++) {
  const p = curve[Math.floor(rnd() * curve.length)], ux = p[0] - cen[0], uy = p[1] - cen[1], u = Math.hypot(ux, uy);
  const d = 10 ** (-4 + 4 * rnd()) * (k % 2 ? 1 : -1);
  rays.push([F(p[0] + (ux / u) * d), F(p[1] + (uy / u) * d)]);
}
// Rays passing close to the pole (xi ~ 0): the first run's worst rays all had |alpha| < 0.02.
for (let k = 0; k < 300; k++) rays.push([F((k % 2 ? 1 : -1) * 10 ** (-5 + 3.5 * rnd())), F((rnd() * 2 - 1) * 14)]);
const dist = rays.map(([al, be]) => {
  const xi = -al * si, eta = be * be + (xi * xi * ci * ci) / Math.max(si * si, 1e-8) - a * a * ci * ci;
  return curveDistance(al, be, curve) * (classify(xi, eta, a) === "captured" ? -1 : 1);
});
const absDist = dist.map(Math.abs);
log(`${rays.length} rays`);

// ---- Observables ----
const opts = (tol: number) => ({ rIn, rOut, rObs: REF_ROBS, tol });
const fromMino = (m: ReturnType<typeof minoTrace>): Obs => ({ f: m.fate as Obs["f"], r: m.rHit, p: m.phiHit, t: m.tHit, dir: m.fate === "escaped" ? skyDirCPU(m.s, a) : undefined, steps: m.steps, attempts: m.attempts });
const ref = (al: number, be: number, tol = REF_TOL) => fromMino(minoTrace(al, be, a, inclDeg, opts(tol)));

// ---- Reference and Jacobian (per pixel of the app's 720 px over 28 M), the step well inside the distance to the curve ----
const refs: Obs[] = [], jac: (number[][] | undefined)[] = [];
rays.forEach(([al, be], n) => {
  const r = ref(al, be); refs.push(r);
  const dM = Math.min(1e-3 * PX_M, absDist[n] / 4), dPx = dM / PX_M;
  const near = (x: number, y: number) => { const q = ref(x, y); return q.f === r.f ? obsVector(q) : null; };
  jac.push(obsVector(r) ? jacobianOf(near(al + dM, be), near(al - dM, be), near(al, be + dM), near(al, be - dM), dPx) : undefined);
  if (n % 500 === 0) log(`reference ${n}`);
});
const noJ = refs.filter((r, n) => obsVector(r) && !jac[n]).length;
log(`reference done; scorable without a Jacobian: ${noJ}`);

const groups = groupsOf(rays, refs.map((r) => r.f));
const score = (arm: (al: number, be: number) => Obs) => {
  const rows: ArmRow[] = rays.map(([al, be], n) => { const g = arm(al, be); return { ...rayError(refs[n], jac[n], g), steps: g.steps, attempts: g.attempts }; });
  return summarizeArm(absDist, rows, groups);
};
const arms: Record<string, unknown> = {};

// How converged is the reference? Against 1e-12, and on a sample against the independent Hamiltonian formulation (trace-reference.ts).
arms["refSelf 1e-12"] = score((al, be) => ref(al, be, 1e-12));
log("self-convergence done");
{
  const pick = rays.map((_, n) => n).filter((n) => n % 25 === 0), rows: ArmRow[] = [];
  let unconverged = 0;
  for (const n of pick) {
    const [al, be] = rays[n], c = convergedRef(screenToState(al, be, a, incl, REF_ROBS), a, rIn);
    if (!c.converged) unconverged++;
    const g: Obs = { f: c.ref.fate as Obs["f"], r: c.ref.rHit, p: c.ref.phiHit, t: c.ref.tHit, dir: c.ref.dir, steps: c.ref.steps, attempts: c.ref.steps + c.ref.retries };
    rows.push({ ...rayError(refs[n], jac[n], g), steps: g.steps, attempts: g.attempts });
  }
  arms["refIndependent"] = { ...summarizeArm(pick.map((n) => absDist[n]), rows), unconverged };
  log(`independent reference on ${pick.length} rays, unconverged ${unconverged}`);
}

// ---- f64 arms: truncation alone ----
for (const tol of F64_TOLS) { arms[`mino f64 ${tol}`] = score((al, be) => ref(al, be, tol)); log(`mino f64 ${tol}`); }
// f32 inputs, f64 integration at a tight tolerance: rounding that enters with the ray's constants alone.
arms["mino f64, f32 inputs"] = score((al, be) => {
  const q = f32Inputs(al, be, a, incl), m = minoTraceRay(minoRay(q.a, q.xi, q.eta, REF_ROBS), q.incl, q.beta, opts(1e-9));
  return { f: m.fate as Obs["f"], r: m.rHit, p: m.phiHit, t: m.tHit, dir: m.fate === "escaped" ? skyDirCPU(m.s, q.a) : undefined, steps: m.steps, attempts: m.attempts };
});
log("f32 inputs");
// A nearly exact result merely stored in f32: what the output format alone costs.
arms["mino f64 1e-9, f32 output"] = score((al, be) => roundObs(ref(al, be, 1e-9)));
// The clean order test: the same equations, error norm and landing, at 3rd order.
setMinoMethod("bs3");
for (const tol of BS3_TOLS) { arms[`bs3 f64 ${tol}`] = score((al, be) => fromMino(minoTrace(al, be, a, inclDeg, { ...opts(tol), maxSteps: BS3_BUDGET }))); log(`bs3 f64 ${tol}`); }
setMinoMethod("dp5");
for (const k of RK4_SCALES) {
  const c: RefCfg = { ...SHIP, scale: k, hNear: H_NEAR(k), hFar: H_FAR(k), landEsc: true, maxSteps: rk4Budget(k) };
  arms[`rk4 f64 ${k}`] = score((al, be) => {
    const r = refTrace(screenToState(al, be, a, incl, REF_ROBS), a, rIn, c);
    return { f: r.fate as Obs["f"], r: r.rHit, p: r.phiHit, t: r.tHit, dir: r.dir, steps: r.steps, attempts: r.steps + r.retries };
  });
  log(`rk4 f64 ${k}`);
}

mkdirSync("docs/diagnostics/2026-11-06-accuracy-limit/data", { recursive: true });
const clean = (v: number) => (typeof v === "number" && !Number.isFinite(v) ? null : v);
writeFileSync(`docs/diagnostics/2026-11-06-accuracy-limit/data/${name}.json`, JSON.stringify({
  config: cfg, rIn, rOut, rays, dist, noJ, refs: refs.map((r) => ({ f: r.f, r: r.r, p: r.p, t: r.t, dir: r.dir })), jac, arms,
}, (_, v) => clean(v)));
log("written");
