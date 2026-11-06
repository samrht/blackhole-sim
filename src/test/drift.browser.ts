// ?drift: per-ray conserved-quantity drift, |Q_end - Q_start| / |Q_start| + |H_end|, as a log heat-map. A debug route with
// its own compute module (render/drift.wgsl): nothing here runs on the normal page.
//
// A map of this number alone cannot say WHY a pixel is bright, so the route also answers that:
//   - the fate of every ray is compared with the renderer's own audit entry (same loop, same rays);
//   - the WGSL measure is compared with its CPU twin evaluated in f32;
//   - every cpuStride-th pixel is re-traced in f64: `f64` is integrator truncation alone (nothing else is left), and
//     `floor` is that f64 end state rounded to f32 and measured in f32: what f32 alone gives a ray with almost no drift.
import { Renderer } from "../render/gpu";
import driftWGSL from "../render/drift.wgsl?raw";
import shadowSharedWGSL from "../render/shadow-shared.wgsl?raw";
import cameraSharedWGSL from "../render/camera-shared.wgsl?raw";
import integratorSharedWGSL from "../render/integrator-shared.wgsl?raw";
import { driftMetric, minoHQ, minoQ, preciseDrift } from "../physics/conserved";
import { minoInit, minoRay, minoRw, minoTheta, minoTrace } from "../physics/trace-mino";
import { iscoRadius } from "../physics/orbits";
import { classify, shadowBoundary } from "../physics/shadow";
import { JITTER } from "../render/cache-plan";
import { prepareScene, sceneUniforms, type Scene } from "./scenes";
import { curveDistance, heatColor, logUnit, parseDriftParams, summarize, type DriftParams, type Summary } from "./drift-analysis";

const EPS = 1.1920929e-7;
const STRIDE = 16;          // 32-bit words per pixel (DriftPx)
const CHUNK_ROWS = 45;      // rows per dispatch: long no-disk frames stay under the GPU watchdog
const FATE = ["captured", "disk", "sky"] as const;
const FATE_COLOR: [number, number, number][] = [[42, 120, 214], [235, 104, 52], [27, 175, 122], [227, 73, 72]]; // + budget

export interface DriftFields { w: number; h: number; q0: Float32Array; q1: Float32Array; h0: Float32Array; h1: Float32Array; hScale: Float32Array; qScale: Float32Array; wRes: Float32Array; rEnd: Float32Array; xi: Float32Array; eta: Float32Array; word: Uint32Array; vEnd: Float32Array; wEnd: Float32Array; wpEnd: Float32Array; vpEnd: Float32Array; vp0: Float32Array; pTotal: Float64Array; pRelQ: Float64Array; pH: Float64Array; pW: Float64Array }

async function readBuffer(device: GPUDevice, src: GPUBuffer, size: number): Promise<ArrayBuffer> {
  const rb = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(src, 0, rb, 0, size); device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ); const out = rb.getMappedRange().slice(0); rb.unmap(); rb.destroy();
  return out;
}

/** WGSL measure against its CPU twin in f32, on states on and off the null cone. Worst |difference| in units of the tolerance. */
async function measureParity(device: GPUDevice, mod: GPUShaderModule): Promise<{ worst: number; cases: number }> {
  const fr = Math.fround, cases: { a: number; xi: number; eta: number; y: Float64Array }[] = [];
  for (const a of [0.998, 0.9, 0.3]) for (const [xi, eta] of [[1.5, 3], [-6, 12], [2.1, 0.4], [0.01, 25]])
    for (const r of [1200, 40, 6, 2.2, 1.09]) for (const v of [1.45, 0.9, 0.2]) for (const off of [0, 1e-3]) {
      const c = minoRay(fr(a), fr(xi), fr(eta), 1000), w = fr(1 / r), th = minoTheta(v, c);
      cases.push({ a: fr(a), xi: fr(xi), eta: fr(eta), y: Float64Array.from([0, w, fr(v), 0, 0, fr(Math.sqrt(Math.max(0, minoRw(w, c))) + off), fr(Math.sqrt(Math.max(0, th)) + off), 1]) });
    }
  const inp = new Float32Array(cases.length * 12);
  cases.forEach((k, i) => { inp.set([k.y[0], k.y[1], k.y[2], k.y[3], k.y[4], k.y[5], k.y[6], k.y[7], k.a, k.xi, k.eta, 1000], i * 12); });
  const inB = device.createBuffer({ size: inp.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(inB, 0, inp);
  const outB = device.createBuffer({ size: cases.length * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const pipe = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "measure" } });
  const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 2, resource: { buffer: inB } }, { binding: 3, resource: { buffer: outB } }] });
  const enc = device.createCommandEncoder(), cp = enc.beginComputePass();
  cp.setPipeline(pipe); cp.setBindGroup(0, bg); cp.dispatchWorkgroups(Math.ceil(cases.length / 16)); cp.end(); device.queue.submit([enc.finish()]);
  const o = new Float32Array(await readBuffer(device, outB, cases.length * 32));
  let worst = 0;
  cases.forEach((k, i) => {
    const c = minoRay(k.a, k.xi, k.eta, 1000), Q = minoQ(k.y, c, fr), H = minoHQ(k.y, c, fr), g = o.subarray(i * 8, i * 8 + 8);
    worst = Math.max(worst,
      Math.abs(g[0] - Q.q) / (16 * EPS * Q.qScale), Math.abs(g[2] - H.h) / (32 * EPS * H.hScale),
      Math.abs(g[3] - H.hScale) / (32 * EPS * H.hScale), Math.abs(g[4] - H.wRes) / (32 * EPS * Math.max(1, k.y[5] * k.y[5])));
  });
  inB.destroy(); outB.destroy();
  return { worst, cases: cases.length };
}

async function traceFrame(device: GPUDevice, mod: GPUShaderModule, w: number, h: number, p: DriftParams, rIn: number, rOut: number): Promise<DriftFields> {
  const size = w * h * STRIDE * 4;
  const outB = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const uni = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const pipe = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "trace" } });
  const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: uni } }, { binding: 1, resource: { buffer: outB } }] });
  const ab = new ArrayBuffer(64), f = new Float32Array(ab), u = new Uint32Array(ab);
  f.set([w, h, p.a, (p.inclDeg * Math.PI) / 180, 1000, p.fov, rIn, rOut, JITTER[0][0], JITTER[0][1], p.panX, p.panY]);
  u[12] = p.maxSteps;
  for (let row = 0; row < h; row += CHUNK_ROWS) {
    u[13] = row; u[14] = Math.min(h, row + CHUNK_ROWS);
    device.queue.writeBuffer(uni, 0, ab);
    const enc = device.createCommandEncoder(), cp = enc.beginComputePass();
    cp.setPipeline(pipe); cp.setBindGroup(0, bg); cp.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8)); cp.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  const raw = await readBuffer(device, outB, size), fl = new Float32Array(raw), wd = new Uint32Array(raw), n = w * h;
  const col = (k: number) => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = fl[i * STRIDE + k]; return a; };
  const word = new Uint32Array(n); for (let i = 0; i < n; i++) word[i] = wd[i * STRIDE + 11];
  outB.destroy(); uni.destroy();
  return { w, h, q0: col(0), q1: col(1), h0: col(2), h1: col(3), hScale: col(4), qScale: col(5), wRes: col(6), rEnd: col(7), xi: col(8), eta: col(9), word, vEnd: col(10), wEnd: col(12), wpEnd: col(13), vpEnd: col(14), vp0: col(15), pTotal: new Float64Array(n), pRelQ: new Float64Array(n), pH: new Float64Array(n), pW: new Float64Array(n) };
}

/** One view of the frame as a per-pixel value (Infinity / NaN read as undefined). */
export const VIEWS: Record<string, { title: string; lo: number; hi: number; of: (d: DriftFields, i: number) => number }> = {
  total: { title: "|dQ| / |Q0| + |H_end|", lo: 1e-9, hi: 1e-1, of: (d, i) => driftMetric(d.q0[i], d.q1[i], d.h1[i]) },
  relQ: { title: "|dQ| / |Q0|", lo: 1e-9, hi: 1e-1, of: (d, i) => Math.abs(d.q1[i] - d.q0[i]) / Math.abs(d.q0[i]) },
  absH: { title: "|H_end|", lo: 1e-9, hi: 1e-1, of: (d, i) => Math.abs(d.h1[i]) },
  absdQ: { title: "|dQ| (absolute)", lo: 1e-9, hi: 1e-1, of: (d, i) => Math.abs(d.q1[i] - d.q0[i]) },
  pTotal: { title: "Same drift, the f32 state measured in f64 (no f32 rounding in the measure)", lo: 1e-9, hi: 1e-1, of: (d, i) => d.pTotal[i] },
  pRelQ: { title: "|dQ| / |Q0|, the f32 state measured in f64", lo: 1e-9, hi: 1e-1, of: (d, i) => d.pRelQ[i] },
  pH: { title: "|H_end|, the f32 state measured in f64", lo: 1e-9, hi: 1e-1, of: (d, i) => d.pH[i] },
  pW: { title: "|w'^2 - R~(w)|, the f32 state measured in f64", lo: 1e-9, hi: 1e-1, of: (d, i) => d.pW[i] },
  phase: { title: "Polar phase at the end of the ray, |theta'_end| / sqrt|Q0|: ~1 at an equator crossing, 0 at a turning point", lo: 0.01, hi: 1, of: (d, i) => Math.abs(d.vpEnd[i]) / Math.sqrt(Math.abs(d.q0[i])) },
  wRes: { title: "|w'^2 - R~(w)| at the end: the radial first integral, no 1/Delta factor", lo: 1e-9, hi: 1e-1, of: (d, i) => Math.abs(d.wRes[i]) },
  hUlps: { title: "|H_end| / (f32 eps x its term size): 1 = at the rounding floor", lo: 0.1, hi: 1e4, of: (d, i) => Math.abs(d.h1[i]) / (EPS * d.hScale[i]) },
  qUlps: { title: "|dQ| / (f32 eps x its term size): 1 = at the rounding floor", lo: 0.1, hi: 1e4, of: (d, i) => Math.abs(d.q1[i] - d.q0[i]) / (EPS * d.qScale[i]) },
};

function summaryRow(d: DriftFields, pick: (i: number) => boolean) {
  const idx: number[] = []; const n = d.w * d.h; for (let i = 0; i < n; i++) if (pick(i)) idx.push(i);
  const out: Record<string, Summary> & { count?: never } = {};
  for (const k of ["total", "relQ", "absH", "pTotal", "pRelQ", "pH", "pW", "absdQ", "wRes", "hUlps", "qUlps"]) out[k] = summarize(Float64Array.from(idx, (i) => VIEWS[k].of(d, i)));
  return { count: idx.length, ...out };
}

interface CpuRow { i: number; fate: string; gpuFate: number; total: number; relQ: number; absH: number; floorRelQ: number; floorH: number; reprH: number; reprRelQ: number; steps: number }

/** f64 re-trace of every stride-th pixel: integrator truncation alone, and the f32 floor on the same end state. */
async function cpuSample(d: DriftFields, p: DriftParams, rIn: number, rOut: number): Promise<CpuRow[]> {
  const fr = Math.fround, aspect = d.w / d.h, i0 = (p.inclDeg * Math.PI) / 180, si = Math.sin(i0), ci = Math.cos(i0), rows: CpuRow[] = [];
  for (let y = 0; y < d.h; y += p.cpuStride) {
    for (let x = 0; x < d.w; x += p.cpuStride) {
      const i = y * d.w + x;
      const alpha = fr((((x + 0.5 + JITTER[0][0]) / d.w) * 2 - 1) * p.fov * aspect + p.panX), beta = fr(-(((y + 0.5 + JITTER[0][1]) / d.h) * 2 - 1) * p.fov + p.panY);
      const xi = -alpha * si, eta = beta * beta + (xi * xi * ci * ci) / Math.max(si * si, 1e-8) - p.a * p.a * ci * ci;
      const c = minoRay(p.a, xi, eta, 1000), y0 = minoInit(1000, i0, beta, c);
      const t = minoTrace(alpha, beta, p.a, p.inclDeg, { rIn, rOut, rObs: 1000, maxSteps: p.maxSteps });
      const q0 = minoQ(y0, c).q, q1 = minoQ(t.y, c).q, H = minoHQ(t.y, c);
      const cf = minoRay(fr(p.a), fr(xi), fr(eta), 1000), qf0 = minoQ(y0, cf, fr).q, qf1 = minoQ(t.y, cf, fr).q, Hf = minoHQ(t.y, cf, fr);
      rows.push({ i, fate: t.fate, gpuFate: d.word[i] & 7, total: driftMetric(q0, q1, H.h), relQ: Math.abs(q1 - q0) / Math.abs(q0), absH: Math.abs(H.h),
        floorRelQ: Math.abs(qf1 - qf0) / Math.abs(qf0), floorH: Math.abs(Hf.h), reprH: Math.abs(minoHQ(t.y.map(fr), c).h), reprRelQ: Math.abs(minoQ(t.y.map(fr), c).q - q0) / Math.abs(q0), steps: t.steps });
    }
    await new Promise((r) => setTimeout(r, 0));
  }
  return rows;
}

const CURVE_BINS = [0, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1, 3, Infinity];
const STEP_BINS = [0, 15, 20, 25, 30, 40, 60, 80, 100, 120, 160, 240, Infinity];
const REND_BINS = [0, 1.07, 1.1, 1.2, 1.5, 2, 3, 6, 12, 40, 1e9];
const Q0_BINS = [0, 1e-3, 1e-2, 1e-1, 1, 10, 100, Infinity];
const binLabel = (b: number[], k: number) => `[${b[k]}, ${b[k + 1]})`;

export interface DriftReport { params: DriftParams; adapter: string; w: number; h: number; rIn: number; rOut: number; audit: { kindMismatch: number; budgetMismatch: number };
  measureParity: { worst: number; cases: number }; h0Max: number; byFate: unknown; byCurve: unknown; byREnd: unknown; byQ0: unknown; byStep: unknown; explain: unknown; budget: number; atFloor: unknown; cpu: unknown; worst: unknown; steps: Summary }

export async function runDrift(canvas: HTMLCanvasElement): Promise<{ report: DriftReport; png: (v: string) => string }> {
  const p = parseDriftParams(location.search);
  const r = new Renderer(); await r.init(canvas);
  const incl = (p.inclDeg * Math.PI) / 180, isco = iscoRadius(p.a, true), rOut = 40, rIn = p.disk ? isco : 1e9;
  const device = r.device, w = r.width, h = r.height;
  const scene: Scene = { name: "drift", a: p.a, inclDeg: p.inclDeg, time: 0, frame: 0, jetStrength: 0, skyStrength: 0 };
  prepareScene(r, scene);
  const words = await r.auditKinds(sceneUniforms(r, scene, rIn, { rIn, rOut, fovScale: p.fov, panX: p.panX, panY: p.panY, maxSteps: p.maxSteps, setIndex: 0, turbAmp: 0, breatheAmp: 0, nSpots: 0 }));

  const mod = device.createShaderModule({ code: shadowSharedWGSL + cameraSharedWGSL + integratorSharedWGSL + driftWGSL });
  const info = await mod.getCompilationInfo();
  for (const m of info.messages) if (m.type === "error") throw new Error(`drift.wgsl:${m.lineNum}: ${m.message}`);
  const parity = await measureParity(device, mod);
  const d = await traceFrame(device, mod, w, h, p, rIn, rOut);

  const n = w * h, aspect = w / h, fr = Math.fround;
  const v0 = incl <= Math.PI / 2 ? fr(incl) : fr(fr(Math.PI) - fr(incl));
  for (let i = 0; i < n; i++) {
    const e = preciseDrift({ w: d.wEnd[i], wp: d.wpEnd[i], v: d.vEnd[i], vp: d.vpEnd[i] }, { v: v0, vp: d.vp0[i] }, { a: fr(p.a), xi: d.xi[i], eta: d.eta[i], r0: 1000 });
    d.pTotal[i] = e.total; d.pRelQ[i] = Math.abs(e.q1 - e.q0) / Math.abs(e.q0); d.pH[i] = Math.abs(e.h1); d.pW[i] = Math.abs(e.wRes);
  }
  let kindMismatch = 0, budgetMismatch = 0, budget = 0, h0Max = 0;
  const steps = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    if ((d.word[i] & 3) !== (words[i] & 3)) kindMismatch++;
    if ((d.word[i] & 4) !== (words[i] & 4)) budgetMismatch++;
    if (d.word[i] & 4) budget++;
    steps[i] = d.word[i] >>> 3; h0Max = Math.max(h0Max, Math.abs(d.h0[i]));
  }
  const kind = (i: number) => d.word[i] & 3;

  // Signed distance (M) from the critical curve, negative inside the shadow.
  // A zoomed frame needs curve samples far finer than a pixel; only those near the frame are kept.
  const reach = 1.5 * p.fov * aspect, si = Math.sin(incl), ci = Math.cos(incl);
  const curve = shadowBoundary(p.a, incl, p.fov < 1 ? 400000 : 1500).filter(([al, be]) => Math.abs(al - p.panX) < reach && Math.abs(be - p.panY) < reach);
  const dist = new Float32Array(n);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const al = (((x + 0.5 + JITTER[0][0]) / w) * 2 - 1) * p.fov * aspect + p.panX, be = -(((y + 0.5 + JITTER[0][1]) / h) * 2 - 1) * p.fov + p.panY, i = y * w + x;
    const xi = -al * si, eta = be * be + (xi * xi * ci * ci) / Math.max(si * si, 1e-8) - p.a * p.a * ci * ci;
    dist[i] = curveDistance(al, be, curve) * (classify(xi, eta, p.a) === "captured" ? -1 : 1);
  }
  const byFate: Record<string, unknown> = {};
  for (let k = 0; k < 3; k++) byFate[FATE[k]] = summaryRow(d, (i) => kind(i) === k && !(d.word[i] & 4));
  byFate.budget = summaryRow(d, (i) => (d.word[i] & 4) !== 0);
  const byCurve: Record<string, unknown> = {};
  for (let k = 0; k + 1 < CURVE_BINS.length; k++) {
    byCurve[`outside ${binLabel(CURVE_BINS, k)}`] = summaryRow(d, (i) => dist[i] >= CURVE_BINS[k] && dist[i] < CURVE_BINS[k + 1]);
    byCurve[`inside ${binLabel(CURVE_BINS, k)}`] = summaryRow(d, (i) => -dist[i] > CURVE_BINS[k] && -dist[i] <= CURVE_BINS[k + 1]);
  }
  const byREnd: Record<string, unknown> = {};
  for (let k = 0; k + 1 < REND_BINS.length; k++) byREnd[binLabel(REND_BINS, k)] = summaryRow(d, (i) => d.rEnd[i] >= REND_BINS[k] && d.rEnd[i] < REND_BINS[k + 1]);
  const byQ0: Record<string, unknown> = {};
  for (let k = 0; k + 1 < Q0_BINS.length; k++) byQ0[binLabel(Q0_BINS, k)] = summaryRow(d, (i) => Math.abs(d.q0[i]) >= Q0_BINS[k] && Math.abs(d.q0[i]) < Q0_BINS[k + 1]);

  const byStep: Record<string, unknown> = {};
  for (const [name, fate] of [["sky", 2], ["captured", 0], ["disk", 1]] as const) for (let k = 0; k + 1 < STEP_BINS.length; k++) {
    const row = summaryRow(d, (i) => (d.word[i] & 3) === fate && (d.word[i] >>> 3) >= STEP_BINS[k] && (d.word[i] >>> 3) < STEP_BINS[k + 1]);
    if (row.count) byStep[`${name} ${binLabel(STEP_BINS, k)}`] = row;
  }

  // Two explanations of the captured rays, tested per pixel on the f64-measured state: H is the radial residual times
  // r^4 / (2 Sigma Delta), and the radial residual is 2 |w'| times a fixed absolute error in w'.
  const explain: Record<string, unknown> = {};
  {
    const WP_BINS = [0, 0.1, 0.3, 1, 2, 4, 8, Infinity];
    const amp = (i: number) => { const r = 1 / d.wEnd[i], D = r * r - 2 * r + p.a * p.a, c = Math.cos(d.vEnd[i]), S = r * r + p.a * p.a * c * c; return (r * r * r * r) / (2 * S * D); };
    const cap: number[] = []; for (let i = 0; i < n; i++) if ((d.word[i] & 3) === 0 && !(d.word[i] & 4)) cap.push(i);
    explain.hOverAmpW = summarize(cap.map((i) => d.pH[i] / (amp(i) * d.pW[i])));
    explain.amp = summarize(cap.map(amp));
    const byWp: Record<string, unknown> = {};
    for (let k = 0; k + 1 < WP_BINS.length; k++) {
      const g = cap.filter((i) => Math.abs(d.wpEnd[i]) >= WP_BINS[k] && Math.abs(d.wpEnd[i]) < WP_BINS[k + 1]);
      if (g.length) byWp[binLabel(WP_BINS, k)] = { count: g.length, wp: summarize(g.map((i) => Math.abs(d.wpEnd[i]))), pW: summarize(g.map((i) => d.pW[i])), dWp: summarize(g.map((i) => d.pW[i] / (2 * Math.abs(d.wpEnd[i])))), pH: summarize(g.map((i) => d.pH[i])), meanAlpha: g.reduce((s, i) => s + (-d.xi[i] / si), 0) / g.length };
    }
    explain.byWp = byWp;
    // W against the size of its own terms: is it a fixed number of f32 eps of w'^2?
    explain.wOverWp2 = Object.fromEntries(WP_BINS.slice(0, -1).map((lo, k) => [binLabel(WP_BINS, k), summarize(cap.filter((i) => Math.abs(d.wpEnd[i]) >= lo && Math.abs(d.wpEnd[i]) < WP_BINS[k + 1]).map((i) => d.pW[i] / (d.wpEnd[i] * d.wpEnd[i])))]));
    // Polar phase at the end of the ray: |v'_end| / sqrt|Q0| is ~0 at a turning point and largest at an equator crossing, and dQ = 2 v' dv'.
    const PHASE_BINS = [0, 0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.05, Infinity], phase = (i: number) => Math.abs(d.vpEnd[i]) / Math.sqrt(Math.abs(d.q0[i]));
    const byPhase: Record<string, unknown> = {};
    for (const [name, fate] of [["sky", 2], ["captured", 0], ["disk", 1]] as const) for (let k = 0; k + 1 < PHASE_BINS.length; k++) {
      const g: number[] = []; for (let i = 0; i < n; i++) if ((d.word[i] & 3) === fate && Math.abs(d.q0[i]) >= 1 && phase(i) >= PHASE_BINS[k] && phase(i) < PHASE_BINS[k + 1]) g.push(i);
      if (g.length) byPhase[`${name} ${binLabel(PHASE_BINS, k)}`] = { count: g.length, pRelQ: summarize(g.map((i) => d.pRelQ[i])) };
    }
    explain.byPhase = byPhase;
  }

  let hFloor = 0, qFloor = 0, bothFloor = 0;
  for (let i = 0; i < n; i++) {
    const hf = VIEWS.hUlps.of(d, i) <= 4, qf = VIEWS.qUlps.of(d, i) <= 4;
    if (hf) hFloor++; if (qf) qFloor++; if (hf && qf) bothFloor++;
  }
  const atFloor = { rule: "within 4 f32 eps of the size of the quantity's terms", H: hFloor / n, Q: qFloor / n, both: bothFloor / n };

  // Worst pixels of the requested number.
  const totals = Float64Array.from({ length: n }, (_, i) => VIEWS.total.of(d, i));
  const order = Array.from({ length: n }, (_, i) => i).filter((i) => Number.isFinite(totals[i])).sort((x, y) => totals[y] - totals[x]).slice(0, 15);
  const worst = order.map((i) => ({ x: i % w, y: Math.floor(i / w), total: totals[i], relQ: VIEWS.relQ.of(d, i), absH: VIEWS.absH.of(d, i), q0: d.q0[i], q1: d.q1[i], eta: d.eta[i], xi: d.xi[i], fate: FATE[kind(i)], budget: (d.word[i] & 4) !== 0, steps: d.word[i] >>> 3, rEnd: d.rEnd[i], distCurve: dist[i], hUlps: VIEWS.hUlps.of(d, i), qUlps: VIEWS.qUlps.of(d, i) }));

  // f64 twin: truncation alone, per fate and near the curve.
  const rows = await cpuSample(d, p, rIn, rOut);
  const gpuTotal = (i: number) => totals[i];
  const cpuGroup = (pick: (c: CpuRow) => boolean) => {
    const g = rows.filter(pick);
    return { count: g.length, gpu: summarize(g.map((c) => gpuTotal(c.i))), f64: summarize(g.map((c) => c.total)),
      gpuH: summarize(g.map((c) => VIEWS.absH.of(d, c.i))), f64H: summarize(g.map((c) => c.absH)), floorH: summarize(g.map((c) => c.floorH)),
      preciseH: summarize(g.map((c) => d.pH[c.i])), reprH: summarize(g.map((c) => c.reprH)), preciseRelQ: summarize(g.map((c) => d.pRelQ[c.i])), reprRelQ: summarize(g.map((c) => c.reprRelQ)),
      gpuRelQ: summarize(g.map((c) => VIEWS.relQ.of(d, c.i))), f64RelQ: summarize(g.map((c) => c.relQ)), floorRelQ: summarize(g.map((c) => c.floorRelQ)) };
  };
  const FATE_F64: Record<string, number> = { captured: 0, disk: 1, escaped: 2 };
  const cpu = {
    stride: p.cpuStride,
    fateMismatch: rows.filter((c) => c.fate !== "budget" && FATE_F64[c.fate] !== (c.gpuFate & 3)).length,
    budgetRows: rows.filter((c) => c.fate === "budget").length,
    captured: cpuGroup((c) => c.fate === "captured"), disk: cpuGroup((c) => c.fate === "disk"), escaped: cpuGroup((c) => c.fate === "escaped"), budget: cpuGroup((c) => c.fate === "budget"),
    byStep: Object.fromEntries(STEP_BINS.slice(0, -1).map((lo, k) => [binLabel(STEP_BINS, k), cpuGroup((c) => c.steps >= lo && c.steps < STEP_BINS[k + 1])]).filter(([, g]) => (g as { count: number }).count > 0)),
    nearCurve03: cpuGroup((c) => Math.abs(dist[c.i]) < 0.03), farCurve: cpuGroup((c) => Math.abs(dist[c.i]) >= 3),
  };
  const adapter = ((device as unknown as { adapterInfo?: { vendor?: string; architecture?: string; description?: string } }).adapterInfo);
  const report: DriftReport = { params: p, adapter: `${adapter?.vendor ?? "?"} ${adapter?.architecture ?? ""} ${adapter?.description ?? ""}`.trim(), w, h, rIn, rOut,
    audit: { kindMismatch, budgetMismatch }, measureParity: parity, h0Max, byFate, byCurve, byREnd, byQ0, byStep, explain, budget, atFloor, cpu, worst, steps: summarize(steps) };

  // ---- Picture: the heat-map in impact-parameter axes, the analytic critical curve in the second hue, a colour bar. ----
  const M = 54, W = w + 2 * M, H = h + 112;
  const heat = document.createElement("canvas"); heat.width = w; heat.height = h;
  const png = (view: string): string => {
    const out = document.createElement("canvas"); out.width = W; out.height = H;
    const g = out.getContext("2d")!; g.fillStyle = "#ffffff"; g.fillRect(0, 0, W, H);
    const hg = heat.getContext("2d")!, im = hg.createImageData(w, h);
    const title = view === "fate" ? "How each ray ended" : VIEWS[view].title;
    for (let i = 0; i < n; i++) {
      const c = view === "fate" ? FATE_COLOR[d.word[i] & 4 ? 3 : kind(i)] : heatColor(logUnit(VIEWS[view].of(d, i), VIEWS[view].lo, VIEWS[view].hi));
      im.data[i * 4] = c[0]; im.data[i * 4 + 1] = c[1]; im.data[i * 4 + 2] = c[2]; im.data[i * 4 + 3] = 255;
    }
    hg.putImageData(im, 0, 0); g.drawImage(heat, M, 44);
    const toX = (al: number) => M + ((al - p.panX) / (p.fov * aspect) + 1) * 0.5 * w, toY = (be: number) => 44 + ((-(be - p.panY) / p.fov) + 1) * 0.5 * h;
    g.fillStyle = "#eb6834";
    for (const [al, be] of curve) g.fillRect(toX(al) - 0.5, toY(be) - 0.5, 1.2, 1.2);
    g.fillStyle = "#1a1a19"; g.font = "600 15px system-ui, sans-serif";
    g.fillText(`${title}   a = ${p.a}, i = ${p.inclDeg} deg, ${p.disk ? "disk on" : "no disk"}`, M, 26);
    g.fillStyle = "#55554f"; g.font = "12px system-ui, sans-serif";
    g.fillText(`alpha ${(p.panX - p.fov * aspect).toFixed(1)} to ${(p.panX + p.fov * aspect).toFixed(1)} M, beta ${(p.panY - p.fov).toFixed(1)} to ${(p.panY + p.fov).toFixed(1)} M; orange: critical curve`, M, 44 + h + 18);
    if (view === "fate") {
      let lx = M; const names = ["captured", "disk", "sky", "out of steps"];
      names.forEach((name, k) => { const c = FATE_COLOR[k]; g.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; g.fillRect(lx, 44 + h + 34, 14, 14); g.fillStyle = "#1a1a19"; g.fillText(name, lx + 20, 44 + h + 46); lx += 40 + g.measureText(name).width; });
    } else {
      const v = VIEWS[view], bw = 420, bx = M, by = 44 + h + 32;
      for (let k = 0; k < bw; k++) { const c = heatColor(k / (bw - 1)); g.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; g.fillRect(bx + k, by, 1, 14); }
      g.fillStyle = "#1a1a19"; const decades = Math.round(Math.log10(v.hi / v.lo));
      for (let k = 0; k <= decades; k++) { const e = Math.round(Math.log10(v.lo)) + k, tx = bx + (k / decades) * bw; g.fillRect(tx, by + 14, 1, 4); g.fillText(`1e${e}`, tx - 10, by + 32); }
      const uc = heatColor(NaN); g.fillStyle = `rgb(${uc[0]},${uc[1]},${uc[2]})`; g.fillRect(bx + bw + 40, by, 14, 14); g.fillStyle = "#1a1a19"; g.fillText("undefined (Q0 = 0)", bx + bw + 60, by + 12);
    }
    return out.toDataURL("image/png");
  };

  // ---- Page: view picker, the picture, a per-pixel readout on hover, the report. ----
  document.body.style.cssText = "background:#fff;color:#1a1a19;font:13px system-ui,sans-serif;overflow:auto";
  document.body.innerHTML = `<label style="display:block;padding:8px ${M}px">View <select id="v">${["total", "pTotal", "relQ", "pRelQ", "absH", "pH", "pW", "absdQ", "wRes", "hUlps", "qUlps", "phase", "fate"].map((k) => `<option>${k}</option>`).join("")}</select></label><img id="p" alt="drift heat-map" /><div id="r" style="padding:4px ${M}px;min-height:18px;font-variant-numeric:tabular-nums"></div><pre id="j" style="padding:8px ${M}px"></pre>`;
  const img = document.getElementById("p") as HTMLImageElement, sel = document.getElementById("v") as HTMLSelectElement, readout = document.getElementById("r")!;
  const show = () => { img.src = png(sel.value); }; sel.onchange = show; show();
  img.onmousemove = (e) => {
    const x = Math.floor(e.offsetX - M), y = Math.floor(e.offsetY - 44); if (x < 0 || y < 0 || x >= w || y >= h) { readout.textContent = ""; return; }
    const i = y * w + x;
    readout.textContent = `px ${x},${y}  ${d.word[i] & 4 ? "out of steps" : FATE[kind(i)]}  steps ${d.word[i] >>> 3}  total ${totals[i].toExponential(2)}  f64-measured ${d.pTotal[i].toExponential(2)}  |dQ|/|Q0| ${VIEWS.relQ.of(d, i).toExponential(2)}  |H| ${VIEWS.absH.of(d, i).toExponential(2)}  Q0 ${d.q0[i].toFixed(4)}  r_end ${d.rEnd[i].toFixed(3)}  from curve ${dist[i].toFixed(3)} M`;
  };
  document.getElementById("j")!.textContent = JSON.stringify({ params: p, audit: report.audit, measureParity: parity, atFloor, budget }, null, 1);
  return { report, png };
}
