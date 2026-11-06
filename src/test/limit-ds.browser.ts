// ?limitds&config=<name>: the "why not just emulate f64?" baseline. The Mino DP5(4) integrator in double-single (render/limit-ds.wgsl) on the
// same rays and reference as ?limit, with a real cost measurement: plain f32, compensated f32 and double-single over a large batch of rays.
import limitDsWGSL from "../render/limit-ds.wgsl?raw";
import limitWGSL from "../render/limit.wgsl?raw";
import shadowSharedWGSL from "../render/shadow-shared.wgsl?raw";
import cameraSharedWGSL from "../render/camera-shared.wgsl?raw";
import integratorSharedWGSL from "../render/integrator-shared.wgsl?raw";
import { dsFrom, dsWgslConstants, dsToNumber, type DS } from "../physics/double-single";
import { minoRay, minoToState } from "../physics/trace-mino";
import { skyDirCPU } from "../physics/trace-reference";
import { rayError, roundObs, summarizeArm, type ArmRow, type ArmSummary, type Obs } from "./limit-score";
import { groupsOf, type LimitConfig } from "./limit-config";

interface Data { config: LimitConfig; rIn: number; rOut: number; rays: [number, number][]; dist: number[]; refs: Obs[]; jac: (number[][] | null)[] }
const WORDS = 18, FATE: Obs["f"][] = ["captured", "disk", "escaped"], DS_COST_RAYS = 16000;
export const DS_TOLS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9];
// The DP5(4) tableau, in double-single (trace-mino.ts A and E).
const TABLEAU: Record<string, number> = { K_A21: 1 / 5, K_A31: 3 / 40, K_A32: 9 / 40, K_A41: 44 / 45, K_A42: -56 / 15, K_A43: 32 / 9,
  K_A51: 19372 / 6561, K_A52: -25360 / 2187, K_A53: 64448 / 6561, K_A54: -212 / 729,
  K_A61: 9017 / 3168, K_A62: -355 / 33, K_A63: 46732 / 5247, K_A64: 49 / 176, K_A65: -5103 / 18656,
  K_B1: 35 / 384, K_B3: 500 / 1113, K_B4: 125 / 192, K_B5: -2187 / 6784, K_B6: 11 / 84,
  K_E1: 71 / 57600, K_E3: -71 / 16695, K_E4: 71 / 1920, K_E5: -17253 / 339200, K_E6: 22 / 525, K_E7: -1 / 40 };
export const dsModule = (tol: number) => shadowSharedWGSL + cameraSharedWGSL + integratorSharedWGSL.replace(/const MINO_TOL = 1e-5;/, `const MINO_TOL = ${tol.toExponential()};`) + dsWgslConstants(TABLEAU) + limitDsWGSL;

async function compile(device: GPUDevice, code: string, entry: string): Promise<GPUComputePipeline> {
  const mod = device.createShaderModule({ code });
  for (const m of (await mod.getCompilationInfo()).messages) if (m.type === "error") throw new Error(`wgsl:${m.lineNum}: ${m.message}`);
  return device.createComputePipelineAsync({ layout: "auto", compute: { module: mod, entryPoint: entry } });
}
export interface DsParams { a: number; inclDeg: number; rIn: number; rOut: number; maxSteps?: number }
/** Runs `pipe` over the rays in batches; returns the raw output and the dispatch time in ms. */
export async function dsDispatch(device: GPUDevice, pipe: GPUComputePipeline, rays: Float32Array, nOut: number, p: DsParams, chunk: number): Promise<{ raw: ArrayBuffer; ms: number }> {
  const incl = (p.inclDeg * Math.PI) / 180, v0 = incl <= Math.PI / 2 ? incl : Math.PI - incl;
  const rayB = device.createBuffer({ size: rays.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(rayB, 0, rays as unknown as BufferSource);
  const size = nOut * WORDS * 4, outB = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const uni = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: uni } }, { binding: 1, resource: { buffer: rayB } }, { binding: 2, resource: { buffer: outB } }] });
  const ab = new ArrayBuffer(64), f = new Float32Array(ab), w = new Uint32Array(ab);
  const put = (i: number, d: DS) => { f[i] = d[0]; f[i + 1] = d[1]; };
  put(0, dsFrom(p.a)); put(2, dsFrom(Math.sin(incl))); put(4, dsFrom(Math.cos(incl))); put(6, dsFrom(v0));
  f[8] = 1000; f[9] = p.rIn; f[10] = p.rOut; w[11] = p.maxSteps ?? 4800;
  const t0 = performance.now();
  for (let first = 0; first < nOut; first += chunk) {
    w[12] = first; w[13] = Math.min(chunk, nOut - first);
    device.queue.writeBuffer(uni, 0, ab);
    const enc = device.createCommandEncoder(), cp = enc.beginComputePass();
    cp.setPipeline(pipe); cp.setBindGroup(0, bg); cp.dispatchWorkgroups(Math.ceil(w[13] / 64)); cp.end(); device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  const ms = performance.now() - t0;
  const rb = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(outB, 0, rb, 0, size); device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const raw = rb.getMappedRange().slice(0); rb.unmap(); rb.destroy(); rayB.destroy(); outB.destroy(); uni.destroy();
  return { raw, ms };
}
const toRays = (r: [number, number][]) => { const o = new Float32Array(r.length * 2); r.forEach(([a, b], i) => { o[i * 2] = a; o[i * 2 + 1] = b; }); return o; };

/** Word offsets in DsOut: word, steps, attempts, sigma, then pairs r, phi, delay, w, wp, v, vp. */
export function dsObs(raw: ArrayBuffer, i: number, p: { a: number; xi: number; eta: number }): Obs {
  const U = new Uint32Array(raw), R = new Float32Array(raw), o = i * WORDS, word = U[o], pair = (k: number) => R[o + k] + R[o + k + 1];
  const f: Obs["f"] = word & 4 ? "budget" : FATE[word & 3];
  const base: Obs = { f, steps: U[o + 1], attempts: U[o + 2] };
  if (f === "captured" || f === "budget") return base;
  // Through minoToState in both cases: a ray that steps through a pole has v < 0 and its azimuth folds by pi, as in the renderer.
  const c = minoRay(p.a, p.xi, p.eta, 1000), s = minoToState(Float64Array.from([0, pair(10), pair(14), pair(6), 0, pair(12), pair(16), R[o + 3]]), c);
  if (f === "disk") return { ...base, r: pair(4), p: s[3], t: -pair(8) - 1000 };
  return { ...base, dir: skyDirCPU(s, p.a) };
}

/** Every double-single operation on the GPU against f64: worst error in units of 2^-44, per operation. */
async function probeOps(device: GPUDevice, a: number): Promise<Record<string, number>> {
  const pipe = await compile(device, dsModule(1e-5), "probe"), n = 4096, rays = new Float32Array(n * 4); let seed = 11;
  const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
  const xs: number[] = [], ys: number[] = [];
  for (let k = 0; k < n; k++) {
    const x = (rnd() - 0.5) * 2 * Math.PI, y = (rnd() - 0.5) * 10 ** (3 * rnd() - 1), dx = dsFrom(x), dy = dsFrom(y === 0 ? 1 : y);
    rays.set([dx[0], dx[1], dy[0], dy[1]], k * 4); xs.push(dsToNumber(dx)); ys.push(dsToNumber(dy));
  }
  const { raw } = await dsDispatch(device, pipe, rays, n, { a, inclDeg: 45, rIn: 1, rOut: 2 }, 1024);
  const R = new Float32Array(raw), worst: Record<string, number> = { add: 0, mul: 0, div: 0, sqrt: 0, sin: 0, cos: 0, log: 0 }, UNIT = 2 ** -44;
  for (let k = 0; k < n; k++) {
    const o = k * WORDS, g = (i: number) => R[o + i] + R[o + i + 1], x = xs[k], y = ys[k];
    const check = (name: string, got: number, want: number, scale: number) => { worst[name] = Math.max(worst[name], Math.abs(got - want) / (UNIT * Math.max(scale, 1e-30))); };
    check("add", g(4), x + y, Math.abs(x) + Math.abs(y)); check("mul", g(6), x * y, Math.abs(x * y)); check("div", g(8), x / y, Math.abs(x / y));
    check("sqrt", g(10), Math.sqrt(Math.abs(x)), Math.sqrt(Math.abs(x)));
    check("sin", g(12), Math.sin(x), Math.abs(x) <= Math.PI / 4 ? Math.abs(Math.sin(x)) : 0.5); check("cos", g(14), Math.cos(x), Math.abs(x) <= Math.PI / 4 ? Math.abs(Math.cos(x)) : 0.5);
    check("log", g(16), Math.log(Math.abs(x)), Math.max(Math.abs(Math.log(Math.abs(x))), 0.1));
  }
  return worst;
}

export async function runLimitDs(): Promise<{ name: string; adapter: string; arms: Record<string, ArmSummary>; probe: Record<string, number>; cost: Record<string, number> }> {
  const name = new URLSearchParams(location.search).get("config") ?? "edge-998";
  const res = await fetch(`/docs/diagnostics/2026-11-06-accuracy-limit/data/${name}.json`);
  if (!res.ok) throw new Error(`no CPU data for ${name}`);
  const d = (await res.json()) as Data;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }), device = await adapter!.requestDevice();
  const probe = await probeOps(device, d.config.a);
  console.log(`double-single probe, worst in units of 2^-44: ${JSON.stringify(probe)}`);
  const bad = Object.keys(probe).filter((k) => !(probe[k] <= 4));
  if (bad.length) throw new Error(`double-single probe failed: ${bad.join(", ")} ${JSON.stringify(probe)}`);

  const absDist = d.dist.map(Math.abs), groups = groupsOf(d.rays, d.refs.map((r) => r.f)), incl = (d.config.inclDeg * Math.PI) / 180, arms: Record<string, ArmSummary> = {};
  const rays = toRays(d.rays), p = { a: d.config.a, inclDeg: d.config.inclDeg, rIn: d.rIn, rOut: d.rOut };
  const pixel = d.rays.map(([al, be]) => { const si = Math.sin(incl), ci = Math.cos(incl), xi = -al * si; return { a: d.config.a, xi, eta: be * be + (xi * xi * ci * ci) / Math.max(si * si, 1e-8) - d.config.a * d.config.a * ci * ci }; });
  const score = (obs: Obs[]) => summarizeArm(absDist, obs.map((g, i): ArmRow => ({ ...rayError(d.refs[i], d.jac[i] ?? undefined, g), steps: g.steps, attempts: g.attempts })), groups);
  for (const tol of DS_TOLS) {
    const pipe = await compile(device, dsModule(tol), "trace");
    const { raw } = await dsDispatch(device, pipe, rays, d.rays.length, p, 128);
    const obs = d.rays.map((_, i) => dsObs(raw, i, pixel[i]));
    arms[`ds ${tol}`] = score(obs);
    arms[`ds ${tol}, f32 output`] = score(obs.map(roundObs));
    console.log(`ds ${tol}`);
  }

  // Cost: the same large batch through each arm, best of 3, interleaved. Batches are large so queue overhead does not dominate.
  const cost: Record<string, number> = {}, N = 64000, big = new Float32Array(N * 2);
  for (let k = 0; k < N; k++) { const i = k % d.rays.length; big[k * 2] = d.rays[i][0]; big[k * 2 + 1] = d.rays[i][1]; }
  const plainCode = (tol: number) => shadowSharedWGSL + cameraSharedWGSL + integratorSharedWGSL.replace(/const MINO_TOL = 1e-5;/, `const MINO_TOL = ${tol.toExponential()};`) + limitWGSL;
  for (const tol of [1e-5, 1e-7]) {
    const [plain, comp, ds] = await Promise.all([compile(device, plainCode(tol), "mino"), compile(device, plainCode(tol), "minoComp"), compile(device, dsModule(tol), "trace")]);
    const best = [Infinity, Infinity, Infinity];
    const rayB = device.createBuffer({ size: big.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(rayB, 0, big);
    const outB = device.createBuffer({ size: N * 64, usage: GPUBufferUsage.STORAGE });
    const plainRun = async (pipe: GPUComputePipeline, comp1: number) => {
      const uni = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: uni } }, { binding: 1, resource: { buffer: rayB } }, { binding: 2, resource: { buffer: outB } }] });
      const ab = new ArrayBuffer(48), f = new Float32Array(ab), w = new Uint32Array(ab);
      f.set([d.config.a, incl, 1000, d.rIn, d.rOut, 1, 0, 0]); w[8] = 4800; w[11] = comp1;
      const t0 = performance.now();
      for (let first = 0; first < N; first += 8000) {
        w[9] = first; w[10] = Math.min(8000, N - first); device.queue.writeBuffer(uni, 0, ab);
        const enc = device.createCommandEncoder(), cp = enc.beginComputePass();
        cp.setPipeline(pipe); cp.setBindGroup(0, bg); cp.dispatchWorkgroups(Math.ceil(w[10] / 64)); cp.end(); device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
      }
      uni.destroy();
      return performance.now() - t0;
    };
    for (let rep = 0; rep < 3; rep++) {
      best[0] = Math.min(best[0], await plainRun(plain, 0));
      best[1] = Math.min(best[1], await plainRun(comp, 1));
      best[2] = Math.min(best[2], (await dsDispatch(device, ds, big, DS_COST_RAYS, p, 2048)).ms * (N / DS_COST_RAYS));
    }
    rayB.destroy(); outB.destroy();
    cost[`us per ray plain ${tol}`] = (best[0] * 1000) / N; cost[`us per ray comp ${tol}`] = (best[1] * 1000) / N; cost[`us per ray ds ${tol}`] = (best[2] * 1000) / N;
    console.log(`cost ${tol}: ${JSON.stringify(best)}`);
  }
  const info = (device as unknown as { adapterInfo?: { vendor?: string; architecture?: string } }).adapterInfo;
  return { name, adapter: `${info?.vendor ?? "?"} ${info?.architecture ?? ""}`.trim(), arms, probe, cost };
}
