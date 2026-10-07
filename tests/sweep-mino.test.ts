import { describe, it, expect } from "vitest";
import { screenToState } from "../src/physics/camera";
import { photonOrbit } from "../src/physics/orbits";
import { criticalXiEta, photonShellRange } from "../src/physics/shadow";
import { traceRay } from "../src/physics/trace";
import { minoRay, minoInit, minoStep, minoDense, minoPlane, minoSegWMax, minoToState, minoL, minoDl, minoSphere, minoRw, MINO_TOL, MINO_UFRAC, MINO_CTOL } from "../src/physics/trace-mino";
import { flowN0, HOTFLOW } from "../src/physics/hot-flow";
import { PRESETS } from "../src/physics/presets";
import { SHIP, REF1, REF_ROBS, REF_ROUT, refTrace, convergedRef, skyDirCPU, flowSlab, flowAccum, flowMid, refFlowPath, flowReference,
  wrapAngle, pxError, type FlowObj, type PathSeg } from "../src/physics/trace-reference";

/**
 * The accuracy gate of the Mino-time integrator (spec 2026-10-07 §5). Skipped unless SWEEP=1; HOLDOUT=1 swaps in views
 * the choice never saw. NEW (trace-mino.ts at a candidate tolerance / guard) against OLD (today's shipped controller,
 * asserted equal to trace.ts traceRay) against a converged reference (trace-reference.ts: today's Hamiltonian equations
 * with steps scaled 0.1x / 0.05x, converged to ~1e-6, an independent formulation; cross-checked against Mino at 1e-11).
 *
 * Geometry, per ray: fate; disk rHit, phiHit, tHit; sky direction as an ON-SCREEN displacement through the reference's
 * lensing Jacobian. Pass: no new fate flip; error_new <= max(error_old, floor): r and t 1e-5 relative, phi 1e-5 rad,
 * sky 0.01 px; first-integral drift |w'^2 - R~| < 1e-4.
 * 1.3 mm flow, per ray (plan Task 2 ruling: the renderer's 0.25 M left-sample rule has its own ~1e-3 quadrature error):
 *   (a) path: each method's path (OLD the chords of its steps, as the GPU samples them; NEW its dense output) integrated
 *       with a fine midpoint rule, against the exact integral: error_new <= max(error_old, 1e-5 relative);
 *   (b) rule: each method run as the renderer runs it, against the exact integral: error_new <= max(error_old, 2 q,
 *       1e-5 relative), q = the rule's own error along the exact path.
 */
const SWEEP = !!process.env.SWEEP, HOLDOUT = !!process.env.HOLDOUT;
const FOV = 14, MAXSTEPS = 4800, PX = (2 * FOV) / 720;

type Ray = { s0: Float64Array; a: number; rIn: number; al: number; be: number; incl: number; flow?: FlowObj };
type Res = { fate: string; steps: number; retries: number; rHit?: number; phiHit?: number; tHit?: number; dir?: number[]; I?: number; drift: number; path?: PathSeg[] };
const cart = (s: Float64Array) => [s[1] * Math.sin(s[2]) * Math.cos(s[3]), s[1] * Math.sin(s[2]) * Math.sin(s[3]), s[1] * Math.cos(s[2])];

/** NEW: trace-mino.ts in the render loop's order. Flow rays: the renderer's rule on the dense output (affine weights) and
 *  the path as dense sub-segments of <= 0.005 M for the path test. */
function minoLocal(ray: Ray, tol: number, uFrac: number, wantPath = false, ctol = MINO_CTOL): Res {
  const { a, rIn, al, be, incl } = ray, rh = 1 + Math.sqrt(Math.max(0, 1 - a * a)), fl = ray.flow, I = [0, 0], path: PathSeg[] = [];
  const xi = -al * Math.sin(incl), ci = Math.cos(incl), si = Math.sin(incl);
  const c = minoRay(a, xi, be * be + (xi * xi * ci * ci) / Math.max(si * si, 1e-8) - a * a * ci * ci, REF_ROBS);
  let y = minoInit(REF_ROBS, incl, be, c), h = 50 / (REF_ROBS * REF_ROBS), f: Float64Array | undefined, att = 0, drift = 0;
  for (let step = 1; step <= MAXSTEPS; step++) {
    const st = minoStep(y, h, c, tol, uFrac, f, ctol); att += st.attempts;
    drift = Math.max(drift, Math.abs(st.y[5] * st.y[5] - minoRw(st.y[1], c)));
    const rA = 1 / y[1], rB = 1 / st.y[1];
    if (fl && 1 / minoSegWMax(st.y0, st.y, st.f0, st.f1, st.h) <= HOTFLOW.rMax) { // the renderer's exact skip (flowSeg)
      const dl = minoDl(st.y0, st.y), dense = (th: number) => minoDense(st.y0, st.y, st.f0, st.f1, st.h, th);
      // raytrace.wgsl flowSeg: composite midpoint, FLOW_DL 0.125 x max(1, r_end / 8), at most 64 per step
      const n = Math.min(64, Math.max(1, Math.ceil(dl / (0.125 * Math.max(1, Math.min(rA, rB) / 8)))));
      let dk = st.y0;
      for (let k = 0; k < n; k++) {
        const dn = dense((k + 1) / n), s = minoToState(dense((k + 0.5) / n), c);
        const [dI, dT] = flowSlab(s[1], s[2], [s[4], s[5], s[6], s[7]], a, rh, minoDl(dk, dn), fl); flowAccum(I, dI, dT);
        dk = dn;
      }
      if (wantPath) {
        const m = Math.max(1, Math.ceil(dl / 0.005));
        let q0 = st.y0, s0 = minoToState(st.y0, c);
        for (let k = 1; k <= m; k++) {
          const q1 = dense(k / m), s1 = minoToState(q1, c);
          path.push({ l0: minoL(q0, c), l1: minoL(q1, c), x0: cart(s0), x1: cart(s1), p0: [s0[4], s0[5], s0[6], s0[7]], p1: [s1[4], s1[5], s1[6], s1[7]] });
          q0 = q1; s0 = s1;
        }
      }
    }
    if (!fl) {
      const d = minoPlane(st, c, rIn, REF_ROUT);
      if (d) {
        const rHit = 1 / d[1];
        if (rHit >= rIn && rHit <= REF_ROUT) return { fate: "disk", steps: step, retries: att - step, rHit, phiHit: minoToState(d, c)[3], tHit: minoToState(d, c)[0], drift };
      }
    }
    const wPrev = y[1]; y = st.y; h = st.hNext; f = st.f1;
    if (y[1] >= 1 / (rh * 1.005)) return { fate: "captured", steps: step, retries: att - step, I: I[0], drift, path };
    if (fl && 1 / y[1] > HOTFLOW.rMax && y[1] < wPrev) return { fate: "escaped", steps: step, retries: att - step, I: I[0], drift, path };
    const esc = minoSphere(st, c, 1 / (REF_ROBS * 1.2));
    if (esc) return { fate: "escaped", steps: step, retries: att - step, dir: skyDirCPU(minoToState(esc, c), a), I: I[0], drift, path };
  }
  return { fate: "budget", steps: MAXSTEPS, retries: att - MAXSTEPS, I: I[0], drift, path };
}

const lin = (lo: number, hi: number, n: number) => Array.from({ length: n }, (_, k) => lo + ((hi - lo) * k) / (n - 1));
const mk = (al: number, be: number, a: number, inclDeg: number, rIn: number, flow?: FlowObj): Ray => {
  const incl = (inclDeg * Math.PI) / 180;
  return { s0: screenToState(al, be, a, incl, REF_ROBS), a, rIn, al, be, incl, flow };
};
const grid = (a: number, inclDeg: number, n: number, flow?: FlowObj) => {
  const rays: Ray[] = [];
  for (let iy = 0; iy < n; iy++) for (let ix = 0; ix < n; ix++)
    rays.push(mk((((ix + 0.5) / n) * 2 - 1) * FOV, -(((iy + 0.5) / n) * 2 - 1) * FOV, a, inclDeg, photonOrbit(a, true), flow));
  return rays;
};
const band = (inclDeg: number, betas: number[], alphas: number[], rIn: number) => betas.flatMap((b) => alphas.map((al) => mk(al, b, 0, inclDeg, rIn)));
/** Rays within +-0.005 / 0.01 / 0.02 M (screen radius) of the critical curve at spin a, inclination i. */
function critical(a: number, inclDeg: number, nPts: number): Ray[] {
  const i = (inclDeg * Math.PI) / 180, rays: Ray[] = [];
  const [lo, hi] = Math.abs(a) < 1e-6 ? [3, 3] : photonShellRange(a);
  for (let k = 0; k < nPts; k++) {
    const t = (k + 0.5) / nPts, rr = lo + (hi - lo) * t;
    const [xi, eta] = Math.abs(a) < 1e-6 ? [Math.sqrt(27) * Math.cos(Math.PI * t), 27 - 27 * Math.cos(Math.PI * t) ** 2] : criticalXiEta(rr, a);
    const al = -xi / Math.sin(i), b2 = eta - (xi * xi) / Math.tan(i) ** 2 + a * a * Math.cos(i) ** 2;
    if (!(b2 > 0)) continue;
    for (const sg of [1, -1]) {
      const be = sg * Math.sqrt(b2), rho = Math.hypot(al, be);
      for (const d of [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]) rays.push(mk(al * (1 + d / rho), be * (1 + d / rho), a, inclDeg, photonOrbit(a, true)));
    }
  }
  return rays;
}
const flowOf = (id: string): FlowObj => { const p = PRESETS.find((q) => q.id === id)!; return { n0: flowN0(p.massSun, p.lambda), rg: 1.476625e5 * p.massSun }; };
const mmGrid = (id: string, n: number) => { const p = PRESETS.find((q) => q.id === id)!; return grid(p.a, p.inclDeg, n, flowOf(id)).map((r) => ({ ...r, rIn: 0 })); };
function sets(): { name: string; rays: Ray[] }[] {
  if (HOLDOUT) return [
    { name: "H1 a=0.5 i=30", rays: grid(0.5, 30, 24) }, { name: "H2 a=0.99 i=85", rays: grid(0.99, 85, 24) },
    { name: "H3 a=0.9 i=3", rays: grid(0.9, 3, 24) }, { name: "H4 a=0.7 i=55", rays: grid(0.7, 55, 24) },
    { name: "HC critical a=0.5 i=30", rays: critical(0.5, 30, 24) }, { name: "Hmm M87* 17", rays: mmGrid("m87", 20) },
  ];
  const NA = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3];
  return [
    { name: "V1 a=0 i=8", rays: grid(0, 8, 32) },
    { name: "V2 a=0.9 i=72", rays: grid(0.9, 72, 32) },
    { name: "E a=0.99 i=85", rays: grid(0.99, 85, 32) },
    { name: "A axis band i=8", rays: band(8, lin(9, 14, 11), lin(0, 0.8, 11), 6) },
    { name: "B axis column i=1", rays: band(1, lin(2, 14, 13), lin(0, 1, 5), 6) },
    { name: "N near-field axis", rays: [...band(8, [3, 4, 5, 6, 7, 8], NA, 3), ...band(30, [5, 8, 12, 16], NA, 3), ...band(72, [6, 8, 12], NA, 3)] },
    { name: "C critical a=0 i=72", rays: critical(0, 72, 32) },
    { name: "C critical a=0.9 i=72", rays: critical(0.9, 72, 32) },
    { name: "C critical a=0.998 i=72", rays: critical(0.998, 72, 32) },
    // shallow plane crossings (near the image's horizontal line at high inclination): the crossing radius is most
    // sensitive to polar error there, and near-critical rays among them wind first (the GPU found such a pixel)
    { name: "G grazing i=72/85", rays: [72, 85].flatMap((i) => [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05].flatMap((b) => lin(-14, 14, 57).map((al) => mk(al, b, i === 85 ? 0.99 : 0.9, i, photonOrbit(i === 85 ? 0.99 : 0.9, true))))) },
    { name: "mm Sgr A*", rays: mmGrid("sgra", 24) },
    { name: "mm Gargantua", rays: mmGrid("gargantua", 24) },
  ];
}

describe.skipIf(!SWEEP)("Mino integrator accuracy sweep (SWEEP=1)", () => {
  it("prints the table; the shipped tolerance has no new flips and no worse rays", () => {
    // SETS=i,j,... scores a subset (exploration only; the gate is the full run). ROWS="tol,uFrac,ctol;..." replaces the grid.
    const t0 = Date.now(), pick = process.env.SETS?.split(",").map(Number), S = sets().filter((_, i) => !pick || pick.includes(i));
    // 1) the OLD replica IS the shipped loop
    for (const set of S) for (const r of set.rays) if (!r.flow) {
      const loc = refTrace(r.s0, r.a, r.rIn, SHIP), ship = traceRay(r.s0, r.a, { rIn: r.rIn, rOut: REF_ROUT, rObs: REF_ROBS, maxSteps: MAXSTEPS });
      expect(loc.fate).toBe(ship.fate); expect(loc.steps).toBe(ship.steps);
      if (loc.fate === "disk") expect(loc.rHit).toBe(ship.rHit);
    }
    // 2) converged references; flow rays: exact intensity and the rule's resolution q
    type Ref = { fate: string; converged: boolean; rHit?: number; phiHit?: number; tHit?: number; dir?: number[]; I?: number; q?: number; J?: number[][] };
    const ref: Ref[][] = S.map((set) => set.rays.map((r) => {
      if (r.flow) { const f = flowReference(r.s0, r.a, r.flow); return { fate: f.fate, converged: f.converged, I: f.I, q: f.q }; }
      const { ref: q, converged } = convergedRef(r.s0, r.a, r.rIn);
      const out: Ref = { ...q, converged };
      if (converged && q.fate === "escaped") {
        const d1 = refTrace(mk(r.al + PX, r.be, r.a, (r.incl * 180) / Math.PI, r.rIn).s0, r.a, r.rIn, REF1);
        const d2 = refTrace(mk(r.al, r.be + PX, r.a, (r.incl * 180) / Math.PI, r.rIn).s0, r.a, r.rIn, REF1);
        if (d1.fate === "escaped" && d2.fate === "escaped") out.J = [d1.dir!.map((v, k) => v - q.dir![k]), d2.dir!.map((v, k) => v - q.dir![k])];
      }
      return out;
    }));
    S.forEach((set, si) => {
      const un = ref[si].filter((x) => !x.converged).length;
      // independent cross-check: a very tight Mino trace against the (Hamiltonian) reference on the converged rays
      let xr = 0, xp = 0, xI = 0, xflip = 0;
      set.rays.forEach((r, ri) => {
        const q = ref[si][ri]; if (!q.converged) return;
        const x = minoLocal(r, 1e-11, 0.25, !!r.flow);
        if (x.fate !== q.fate) { xflip++; return; }
        if (r.flow) xI = Math.max(xI, Math.abs(flowMid(x.path!, r.a, r.flow) - q.I!) / Math.max(q.I!, 1e-300));
        else if (x.fate === "disk") { xr = Math.max(xr, Math.abs(x.rHit! - q.rHit!) / q.rHit!); xp = Math.max(xp, wrapAngle(x.phiHit! - q.phiHit!)); }
      });
      console.log(`reference ${set.name}: ${set.rays.length} rays, unconverged ${un}; Mino(1e-11) vs reference: flips ${xflip}, max rel r ${xr.toExponential(1)}, phi ${xp.toExponential(1)}, I (path) ${xI.toExponential(1)}`);
      expect(un).toBeLessThan(0.25 * set.rays.length);
      expect(xflip).toBe(0);
    });
    // 3) OLD, once: geometry, and for flow rays the GPU-rule intensity and the chord path
    const old = S.map((set) => set.rays.map((r) => {
      if (!r.flow) return { ...refTrace(r.s0, r.a, r.rIn, SHIP), Ipath: 0 };
      const rule = refTrace(r.s0, r.a, 0, SHIP, r.flow, 0), pth = refFlowPath(r.s0, r.a, SHIP, 0.005);
      return { ...rule, Ipath: flowMid(pth.path, r.a, r.flow) };
    }));
    const score = (tol: number, uFrac: number, ctol: number) => {
      let newFlips = 0, worse = 0, maxDrift = 0; const worst: string[] = [];
      const costNew: number[] = S.map(() => 0), costOld: number[] = S.map(() => 0);
      const err = { r: 0, phi: 0, t: 0, px: 0, Ipath: 0, Irule: 0 };
      S.forEach((set, si) => set.rays.forEach((r, ri) => {
        const x = minoLocal(r, tol, uFrac, !!r.flow, ctol), q = ref[si][ri], o = old[si][ri];
        costNew[si] += x.steps + x.retries; costOld[si] += o.steps + o.retries; maxDrift = Math.max(maxDrift, x.drift);
        if (!q.converged) return;
        const bad = (what: string) => { worse++; if (worst.length < 12) worst.push(`${set.name} #${ri} ${what}`); };
        if (x.fate !== q.fate) { if (o.fate === q.fate) { newFlips++; if (worst.length < 12) worst.push(`${set.name} #${ri} FLIP ${x.fate} vs ${q.fate}`); } return; }
        if (r.flow) {
          const I = Math.max(q.I!, 1e-300);
          const ep = Math.abs(flowMid(x.path!, r.a, r.flow) - q.I!) / I, eop = Math.abs(o.Ipath - q.I!) / I;
          err.Ipath = Math.max(err.Ipath, ep);
          if (ep > Math.max(eop, 1e-5)) bad(`I(path) ${ep.toExponential(2)} > old ${eop.toExponential(2)}`);
          const er = Math.abs(x.I! - q.I!) / I, eor = Math.abs(o.I! - q.I!) / I;
          err.Irule = Math.max(err.Irule, er);
          if (er > Math.max(eor, (2 * q.q!) / I, 1e-5)) bad(`I(rule) ${er.toExponential(2)} > old ${eor.toExponential(2)}, 2q ${(2 * q.q! / I).toExponential(2)}`);
          return;
        }
        if (x.fate === "disk") {
          const er = Math.abs(x.rHit! - q.rHit!), ep = wrapAngle(x.phiHit! - q.phiHit!), et = Math.abs(x.tHit! - q.tHit!);
          err.r = Math.max(err.r, er / q.rHit!); err.phi = Math.max(err.phi, ep); err.t = Math.max(err.t, et / Math.abs(q.tHit!));
          const od = o.fate === "disk";
          const eor = od ? Math.abs(o.rHit! - q.rHit!) : Infinity, eop = od ? wrapAngle(o.phiHit! - q.phiHit!) : Infinity, eot = od ? Math.abs(o.tHit! - q.tHit!) : Infinity;
          if (er > Math.max(eor, 1e-5 * q.rHit!)) bad(`r ${er.toExponential(2)} > old ${eor.toExponential(2)}`);
          if (ep > Math.max(eop, 1e-5)) bad(`phi ${ep.toExponential(2)} > old ${eop.toExponential(2)}`);
          if (et > Math.max(eot, 1e-5 * Math.abs(q.tHit!))) bad(`t ${et.toExponential(2)} > old ${eot.toExponential(2)}`);
        }
        if (x.fate === "escaped") {
          const e = pxError(q.J, x.dir!, q.dir!); if (isNaN(e)) return;
          err.px = Math.max(err.px, e);
          const eo = o.fate === "escaped" ? pxError(q.J, o.dir!, q.dir!) : Infinity;
          if (e > Math.max(eo, 0.01)) bad(`sky ${e.toFixed(4)} px > old ${eo.toFixed(4)}`);
        }
      }));
      return { newFlips, worse, maxDrift, err, worst, ratio: S.map((_, si) => costNew[si] / costOld[si]), costNew, costOld };
    };
    console.log("tol      uFrac  ctol   newFlips  worse  maxDrift   max err: r(rel) phi(rad) t(rel) sky(px) I-path I-rule   cost NEW/OLD per set");
    const rows: [number, number, number][] = [];
    if (process.env.ROWS) for (const r of process.env.ROWS.split(";")) rows.push(r.split(",").map(Number) as [number, number, number]);
    else { for (const tol of [3e-5, 1e-5, 3e-6]) for (const ctol of [1e-3, 3e-4, 1e-4]) rows.push([tol, 0.25, ctol]); rows.push([1e-5, 0.5, 3e-4]); }
    if (!rows.some(([t, u, k]) => t === MINO_TOL && u === MINO_UFRAC && k === MINO_CTOL)) rows.push([MINO_TOL, MINO_UFRAC, MINO_CTOL]);
    let shipped: ReturnType<typeof score> | null = null;
    for (const [tol, uf, ctol] of rows) {
      const s = score(tol, uf, ctol);
      if (tol === MINO_TOL && uf === MINO_UFRAC && ctol === MINO_CTOL) shipped = s;
      console.log(`${tol.toExponential(0).padEnd(9)}${String(uf).padEnd(7)}${ctol.toExponential(0).padEnd(7)}${String(s.newFlips).padEnd(10)}${String(s.worse).padEnd(7)}${s.maxDrift.toExponential(1).padEnd(11)}` +
        `${s.err.r.toExponential(1)} ${s.err.phi.toExponential(1)} ${s.err.t.toExponential(1)} ${s.err.px.toFixed(3)} ${s.err.Ipath.toExponential(1)} ${s.err.Irule.toExponential(1)}   ${s.ratio.map((x) => x.toFixed(2)).join(" ")}`);
      if (s.worst.length) console.log("   worst: " + s.worst.join(" | "));
    }
    console.log("sets: " + S.map((s, i) => `${i}=${s.name}`).join(", "));
    console.log(`shipped tol ${MINO_TOL} uFrac ${MINO_UFRAC} ctol ${MINO_CTOL}: flips ${shipped!.newFlips}, worse ${shipped!.worse}, drift ${shipped!.maxDrift.toExponential(1)}; mean cost per set OLD ${shipped!.costOld.map((c, i) => (c / S[i].rays.length).toFixed(0)).join(" ")} / NEW ${shipped!.costNew.map((c, i) => (c / S[i].rays.length).toFixed(0)).join(" ")}`);
    console.log(`total ${((Date.now() - t0) / 60000).toFixed(1)} min`);
    expect(shipped!.newFlips).toBe(0);
    expect(shipped!.worse).toBe(0);
    expect(shipped!.maxDrift).toBeLessThan(1e-4);
  }, 14_400_000);
});
