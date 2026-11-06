// ?limit&config=<name>: the GPU (f32) arms of the accuracy-limit experiment, on the rays and reference scripts/limit-cpu.ts wrote.
// DP5(4) Mino at each tolerance and RK4 affine at each step scale, through render/limit.wgsl (its own module, nothing here runs on
// the normal page). MINO_TOL is rewritten in this module's copy of integrator-shared.wgsl only.
import limitWGSL from "../render/limit.wgsl?raw";
import shadowSharedWGSL from "../render/shadow-shared.wgsl?raw";
import cameraSharedWGSL from "../render/camera-shared.wgsl?raw";
import integratorSharedWGSL from "../render/integrator-shared.wgsl?raw";
import { f32Inputs, obsVector, pxShift, rayError, summarizeArm, type ArmRow, type ArmSummary, type Obs } from "./limit-score";
import { F32_TOLS, RK4_SCALES, rk4Budget, groupsOf, AXIS_ALPHA, PX_M, BS3_TOLS, BS3_BUDGET, wgslWithBs3, type LimitConfig } from "./limit-config";
import { H_TOL, H_TOL_FAR } from "../physics/trace";
import { minoRay, minoToState, minoTrace } from "../physics/trace-mino";
import { skyDirCPU } from "../physics/trace-reference";

interface Data { config: LimitConfig; rIn: number; rOut: number; rays: [number, number][]; dist: number[]; refs: Obs[]; jac: (number[][] | null)[]; arms: Record<string, ArmSummary> }
const WORDS = 16, CHUNK = 256;
const FATE: Obs["f"][] = ["captured", "disk", "escaped"];

/** One GPU arm over all rays, in short dispatches so a slow arm stays under the driver's watchdog. */
async function runArm(device: GPUDevice, code: string, entry: string, d: Data, u: { scale: number; hNear: number; hFar: number; maxSteps: number }): Promise<Obs[]> {
  const n = d.rays.length, F = (d.config.inclDeg * Math.PI) / 180;
  const mod = device.createShaderModule({ code });
  for (const m of (await mod.getCompilationInfo()).messages) if (m.type === "error") throw new Error(`limit.wgsl:${m.lineNum}: ${m.message}`);
  const pipe = await device.createComputePipelineAsync({ layout: "auto", compute: { module: mod, entryPoint: entry } });
  const inp = new Float32Array(n * 2); d.rays.forEach(([a, b], i) => { inp[i * 2] = a; inp[i * 2 + 1] = b; });
  const rayB = device.createBuffer({ size: inp.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(rayB, 0, inp);
  const size = n * WORDS * 4, outB = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const uni = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: uni } }, { binding: 1, resource: { buffer: rayB } }, { binding: 2, resource: { buffer: outB } }] });
  const ab = new ArrayBuffer(48), f = new Float32Array(ab), w = new Uint32Array(ab);
  f.set([d.config.a, F, 1000, d.rIn, d.rOut, u.scale, u.hNear, u.hFar]); w[8] = u.maxSteps;
  for (let first = 0; first < n; first += CHUNK) {
    w[9] = first; w[10] = Math.min(CHUNK, n - first);
    device.queue.writeBuffer(uni, 0, ab);
    const enc = device.createCommandEncoder(), cp = enc.beginComputePass();
    cp.setPipeline(pipe); cp.setBindGroup(0, bg); cp.dispatchWorkgroups(Math.ceil(w[10] / 64)); cp.end(); device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  const rb = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(outB, 0, rb, 0, size); device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const raw = rb.getMappedRange().slice(0); rb.unmap(); rb.destroy(); rayB.destroy(); outB.destroy(); uni.destroy();
  const U = new Uint32Array(raw), R = new Float32Array(raw), obs: Obs[] = [];
  for (let i = 0; i < n; i++) {
    const o = i * WORDS, word = U[o], f2: Obs["f"] = word & 4 ? "budget" : FATE[word & 3];
    obs.push({ f: f2, r: R[o + 3], p: R[o + 4], t: -R[o + 5] - 1000, dir: [R[o + 6], R[o + 7], R[o + 8]], steps: U[o + 1], attempts: U[o + 2], end: [R[o + 10], R[o + 11], R[o + 12], R[o + 13], R[o + 14], R[o + 15]] } as Obs & { end: number[] });
  }
  return obs;
}

export async function runLimit(): Promise<{ name: string; adapter: string; arms: Record<string, ArmSummary>; worst: Record<string, unknown[]>; axisScan: unknown }> {
  const name = new URLSearchParams(location.search).get("config") ?? "edge-998";
  const res = await fetch(`/docs/diagnostics/2026-11-06-accuracy-limit/data/${name}.json`);
  if (!res.ok) throw new Error(`no CPU data for ${name}: run scripts/limit-cpu.ts first`);
  const d = (await res.json()) as Data;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  const device = await adapter!.requestDevice();
  const groups = groupsOf(d.rays, d.refs.map((r) => r.f)), incl = (d.config.inclDeg * Math.PI) / 180;
  const absDist = d.dist.map(Math.abs), arms: Record<string, ArmSummary> = {}, worst: Record<string, unknown[]> = {};
  const score = (obs: Obs[]) => {
    // t: the GPU's delay as -t - 1000; the reference's t is exact, so the disk delay error compares the same quantity.
    const rows: ArmRow[] = obs.map((g, i) => ({ ...rayError(d.refs[i], d.jac[i] ?? undefined, g), steps: g.steps, attempts: g.attempts }));
    last = rows.map((x, i) => ({ i, x })).filter((q) => !Number.isNaN(q.x.px)).sort((u, v) => v.x.px - u.x.px).slice(0, 5).map(({ i, x }) => {
      // The same ray in f64 at the shipped tolerance, and how much of the error is the reference's Jacobian: |J| per px.
      const ref = d.refs[i], f = minoTrace(d.rays[i][0], d.rays[i][1], d.config.a, d.config.inclDeg, { rIn: d.rIn, rOut: d.rOut, rObs: 1000 });
      const J = d.jac[i]!, ob: Obs = { f: f.fate as Obs["f"], r: f.rHit, p: f.phiHit, t: f.tHit, dir: f.fate === "escaped" ? skyDirCPU(f.s, d.config.a) : undefined, steps: f.steps, attempts: f.attempts };
      return { alpha: d.rays[i][0], beta: d.rays[i][1], dist: d.dist[i], fate: ref.f, px: x.px, refR: ref.r, gotR: obs[i].r, refP: ref.p, gotP: obs[i].p, f64px: rayError(ref, J, ob).px, jNorm: Math.hypot(...J.flat()), steps: x.steps };
    });
    return summarizeArm(absDist, rows, groups);
  };
  let last: unknown[] = [];
  const base = shadowSharedWGSL + cameraSharedWGSL;
  // Per near-axis ray: on-screen error of each tolerance in f32 and f64, and whether the reference itself really moves that far
  // when the ray is displaced by that shift: lin = |obs(shifted) - got| / |ref - got|, well below 1 where the Jacobian holds.
  const axisIdx = d.rays.map((_, i) => i).filter((i) => Math.abs(d.rays[i][0]) < AXIS_ALPHA);
  const axisScan: Record<string, { alpha: number; beta: number; fate: string; px: number; lin: number; flip: boolean }[]> = {};
  const fromMino = (m: ReturnType<typeof minoTrace>): Obs => ({ f: m.fate as Obs["f"], r: m.rHit, p: m.phiHit, t: m.tHit, dir: m.fate === "escaped" ? skyDirCPU(m.s, d.config.a) : undefined, steps: m.steps, attempts: m.attempts });
  const scan = (name: string, got: (i: number) => Obs) => {
    axisScan[name] = axisIdx.map((i) => {
      const ref = d.refs[i], J = d.jac[i] ?? undefined, g = got(i), e = rayError(ref, J, g), [al, be] = d.rays[i];
      let lin = NaN;
      const u = obsVector(ref), v = obsVector(g);
      if (J && u && v && !e.flip && e.px > 1e-3) {
        const [sa, sb] = pxShift(J, v.map((c, k) => c - u[k])), q = obsVector(fromMino(minoTrace(al + sa * PX_M, be + sb * PX_M, d.config.a, d.config.inclDeg, { rIn: d.rIn, rOut: d.rOut, rObs: 1000, tol: 1e-11 })));
        if (q) lin = Math.hypot(...q.map((c, k) => c - v[k])) / Math.hypot(...u.map((c, k) => c - v[k]));
      }
      return { alpha: al, beta: be, fate: ref.f, px: e.px, lin, flip: e.flip };
    });
  };
  for (const tol of F32_TOLS) {
    const shared = integratorSharedWGSL.replace(/const MINO_TOL = 1e-5;/, `const MINO_TOL = ${tol.toExponential()};`);
    if (tol !== 1e-5 && shared === integratorSharedWGSL) throw new Error("MINO_TOL was not rewritten");
    const obs = await runArm(device, base + shared + limitWGSL, "mino", d, { scale: 1, hNear: 0, hFar: 0, maxSteps: 4800 });
    arms[`mino f32 ${tol}`] = score(obs);
    scan(`f32 ${tol}`, (i) => obs[i]);
    scan(`f64 ${tol}`, (i) => fromMino(minoTrace(d.rays[i][0], d.rays[i][1], d.config.a, d.config.inclDeg, { rIn: d.rIn, rOut: d.rOut, rObs: 1000, tol })));
    worst[`mino f32 ${tol}`] = last;
    // The same f32 end states, the sky direction evaluated in f64: separates the integration's rounding from the direction's own.
    arms[`mino f32 ${tol}, dir in f64`] = score(obs.map((g, i) => {
      if (g.f !== "escaped") return g;
      const [w, wp, v, vp, sigma, ph] = (g as Obs & { end: number[] }).end, q = f32Inputs(d.rays[i][0], d.rays[i][1], d.config.a, incl), c = minoRay(q.a, q.xi, q.eta, 1000);
      return { ...g, dir: skyDirCPU(minoToState(Float64Array.from([0, w, v, ph, 0, wp, vp, sigma]), c), q.a) };
    })); console.log(`mino f32 ${tol}`);
  }
  for (const tol of BS3_TOLS) {
    const shared = wgslWithBs3(integratorSharedWGSL).replace(/const MINO_TOL = 1e-5;/, `const MINO_TOL = ${tol.toExponential()};`);
    arms[`bs3 f32 ${tol}`] = score(await runArm(device, base + shared + limitWGSL, "mino", d, { scale: 1, hNear: 0, hFar: 0, maxSteps: BS3_BUDGET }));
    console.log(`bs3 f32 ${tol}`);
  }
  for (const k of RK4_SCALES) {
    arms[`rk4 f32 ${k}`] = score(await runArm(device, base + integratorSharedWGSL + limitWGSL, "rk4arm", d, { scale: k, hNear: H_TOL, hFar: H_TOL_FAR, maxSteps: rk4Budget(k) }));
    worst[`rk4 f32 ${k}`] = last; console.log(`rk4 f32 ${k}`);
  }
  const info = (device as unknown as { adapterInfo?: { vendor?: string; architecture?: string } }).adapterInfo;
  return { name, adapter: `${info?.vendor ?? "?"} ${info?.architecture ?? ""}`.trim(), arms, worst, axisScan };
}
