// ?imgflow&view=<sgra|m87>: does the accuracy limit reach the 1.3 mm hot-flow image, and its science numbers? The flow is integrated along the whole
// ray, so it cannot be coloured from end points: each arm runs the renderer's own flow loop with only the integrator changed. Scored on the
// quantities ?hotflow reports (flux in Jy, ring diameter in uas) and on the displayed pixels. Reference: the double-single path at tol 1e-9 with the
// renderer's f32 emission. What the path cannot fix (f32 emission, quadrature) is measured on sampled pixels against the independent f64 flowReference.
import { Renderer } from "../render/gpu";
import limitWGSL from "../render/limit.wgsl?raw";
import limitDsWGSL from "../render/limit-ds.wgsl?raw";
import shadowSharedWGSL from "../render/shadow-shared.wgsl?raw";
import cameraSharedWGSL from "../render/camera-shared.wgsl?raw";
import integratorSharedWGSL from "../render/integrator-shared.wgsl?raw";
import emissionSharedWGSL from "../render/emission-shared.wgsl?raw";
import raytraceWGSL from "../render/raytrace.wgsl?raw";
import { JITTER } from "../render/cache-plan";
import { packUniforms } from "../render/uniforms";
import { toRGBA } from "../render/screenshot";
import { dsWgslConstants } from "../physics/double-single";
import { HOTFLOW, flowN0 } from "../physics/hot-flow";
import { HOTFLOW_TARGETS, ringDiameterUas } from "../physics/hot-flow-image";
import { PRESETS } from "../physics/presets";
import { jetUniforms, ETA_DEFAULT } from "../physics/synchrotron";
import { screenToState } from "../physics/camera";
import { flowReference } from "../physics/trace-reference";
import { SCENES, prepareScene, sceneUniforms } from "./scenes";
import { TABLEAU } from "./limit-ds.browser";
import { compareDisplay, compareLinear, pixelRays } from "./image-metrics";
import { summarize } from "./drift-analysis";
import { compPart, dsPart, FLOW_ENTRIES, withFlowDl, withTol } from "./flow-arms";

const N = 512, HALF = 14, ROWS = 32;
type Arm = { name: string; kind: "real" | "comp" | "compOff" | "ds"; tol: number; flowDl?: number };
const ARMS: Arm[] = [
  { name: "f32 1e-3 (deliberately loose)", kind: "real", tol: 1e-3 },
  { name: "f32 1e-5 (shipped)", kind: "real", tol: 1e-5 },
  { name: "compensation switched off, 1e-5", kind: "compOff", tol: 1e-5 },
  { name: "f32 1e-6", kind: "real", tol: 1e-6 }, { name: "f32 1e-7", kind: "real", tol: 1e-7 },
  { name: "compensated 1e-5", kind: "comp", tol: 1e-5 }, { name: "compensated 1e-6", kind: "comp", tol: 1e-6 }, { name: "compensated 1e-7", kind: "comp", tol: 1e-7 },
  { name: "double-single path 1e-5", kind: "ds", tol: 1e-5 }, { name: "double-single path 1e-7", kind: "ds", tol: 1e-7 },
  { name: "quadrature step x2, f32 1e-5", kind: "real", tol: 1e-5, flowDl: 0.25 },
  { name: "quadrature step /4, double-single 1e-9", kind: "ds", tol: 1e-9, flowDl: 0.03125 },
];

export async function runFlowError(canvas: HTMLCanvasElement): Promise<{ report: unknown; images: Record<string, string> }> {
  const id = (new URLSearchParams(location.search).get("view") ?? "m87") as "sgra" | "m87";
  const s = SCENES.find((x) => x.preset === id && x.name === `${id === "sgra" ? "sgra" : "m87"}-mm`);
  if (!s) throw new Error(`view? sgra m87`);
  canvas.style.width = `${N}px`; canvas.style.height = `${N}px`;
  const r = new Renderer(); await r.init(canvas); r.renderBloom = false;
  if (r.width !== N || r.height !== N) throw new Error(`canvas is ${r.width}x${r.height}, want ${N}x${N}`);
  const device = r.device, n = N * N, p = PRESETS.find((q) => q.id === id)!, T = HOTFLOW_TARGETS[id], incl = (s.inclDeg * Math.PI) / 180;
  const rIn = prepareScene(r, s), u = sceneUniforms(r, s, rIn, { setIndex: 0 });
  const rg = 1.476625e5 * p.massSun, dist = T.distKpc * 3.0857e21, uasPerM = (rg / dist) * 206264.806e6, pix = ((2 * HALF) / N) * (rg / dist);

  const code = (a: Arm) => {
    const base = shadowSharedWGSL + cameraSharedWGSL + withTol(integratorSharedWGSL, a.tol) + emissionSharedWGSL + (a.flowDl ? withFlowDl(raytraceWGSL, a.flowDl) : raytraceWGSL);
    if (a.kind === "real") return base;
    if (a.kind === "comp" || a.kind === "compOff") return base + compPart(limitWGSL, a.kind === "comp") + FLOW_ENTRIES.comp;
    return base + dsWgslConstants({ ...TABLEAU, V_A: s.a, V_SI: Math.sin(incl), V_CI: Math.cos(incl), V_V0: incl <= Math.PI / 2 ? incl : Math.PI - incl }) + dsPart(limitDsWGSL) + FLOW_ENTRIES.ds;
  };
  const build = async (a: Arm) => {
    const mod = device.createShaderModule({ code: code(a) });
    for (const m of (await mod.getCompilationInfo()).messages) if (m.type === "error") throw new Error(`${a.name} wgsl:${m.lineNum}: ${m.message}`);
    const pipe = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [r.computeLayout] }), compute: { module: mod, entryPoint: a.kind === "real" ? "build" : "flowArm" } });
    const rows = a.kind === "ds" ? ROWS : N, t0 = performance.now();
    for (let row = 0; row < N; row += rows) {
      device.queue.writeBuffer(r.buildUniformBuf, 0, packUniforms({ ...u, setIndex: 0, rowStart: row, rowEnd: Math.min(N, row + rows), jitterMode: 1 }));
      const enc = device.createCommandEncoder(), cp = enc.beginComputePass();
      cp.setPipeline(pipe); cp.setBindGroup(0, r.buildBinds[0]); cp.dispatchWorkgroups(Math.ceil(N / 8), Math.ceil(rows / 8)); cp.end(); device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }
    return performance.now() - t0;
  };
  const render = async () => {
    r.frame(u, { cachedSet: 0 }); await device.queue.onSubmittedWorkDone();
    const linear = new Float32Array(await r.readbackAccum()), disp = await r.readbackDisplay(), I = new Float64Array(n); let sum = 0;
    for (let i = 0; i < n; i++) { I[i] = linear[i * 4] / HOTFLOW.kTb; sum += I[i]; }
    if (!(sum > 0)) throw new Error("a render is black");
    return { linear, display: toRGBA(disp.data, disp.format), I, entries: new Float32Array((await r.readbackEntries(0)).slice(0, n * 4).buffer) };
  };
  const science = (I: Float64Array) => { let F = 0; for (const v of I) F += v * pix * pix; return { jy: F / 1e-23, ringUas: ringDiameterUas(I, N, HALF, uasPerM), ringBlurUas: ringDiameterUas(I, N, HALF, uasPerM, 15) }; };

  device.pushErrorScope("validation");
  // The reference.
  const refMs = await build({ name: "reference", kind: "ds", tol: 1e-9 }), ref = await render(), refSci = science(ref.I);
  console.log(`reference ${refMs.toFixed(0)} ms: ${refSci.jy.toFixed(4)} Jy, ring ${refSci.ringUas.toFixed(3)} uas`);
  // Control: this route's pipeline against the renderer's own build pass, from the same source.
  r.frame(u, { build: { ...u, setIndex: 0, rowStart: 0, rowEnd: N } }); await device.queue.onSubmittedWorkDone();
  const own = await render(); await build({ name: "same source", kind: "real", tol: 1e-5 }); const same = await render();
  let identical = 0; for (let i = 0; i < n * 4; i += 4) if (same.entries[i + 1] === own.entries[i + 1]) identical++;

  const arms: Record<string, unknown> = {}, shipped: Record<string, Awaited<ReturnType<typeof render>>> = {};
  for (const a of ARMS) {
    const ms = await build(a), img = await render(), sci = science(img.I);
    arms[a.name] = { ms, ...sci, dJyRel: Math.abs(sci.jy / refSci.jy - 1), dRingUas: Math.abs(sci.ringUas - refSci.ringUas), dRingBlurUas: Math.abs(sci.ringBlurUas - refSci.ringBlurUas), display: compareDisplay(ref.display, img.display), linear: compareLinear(ref.linear, img.linear) };
    if (a.name === "f32 1e-5 (shipped)" || a.name === "double-single path 1e-7") shipped[a.name] = img;
    console.log(`arm ${a.name}`);
  }
  const armErr = await device.popErrorScope(); if (armErr) throw new Error(`validation error: ${armErr.message}`);

  // What no better path can fix: f32 emission and the quadrature, against the independent f64 reference on bright sampled pixels.
  const rays = pixelRays(N, N, 14, 0, 0, JITTER[0]), peak = ref.entries.reduce((m, v, i) => (i % 4 === 1 && v > m ? v : m), 0);
  const J = jetUniforms(p.massSun, p.a, p.lambda, ETA_DEFAULT, 60, 2, 1), fl = { n0: flowN0(p.massSun, p.lambda), rg: J.rgCm }, M = 150;
  const bright: number[] = []; for (let i = 0; i < n; i++) if (ref.entries[i * 4 + 1] > 0.01 * peak) bright.push(i);
  const refErr: number[] = [], shipErr: number[] = [], dsErr: number[] = []; let unconverged = 0; const t0 = performance.now();
  for (let k = 0; k < M; k++) {
    const i = bright[Math.floor(((k + 0.5) / M) * bright.length)], q = flowReference(screenToState(rays[i * 2], rays[i * 2 + 1], s.a, incl, 1000), s.a, fl);
    if (!q.converged) unconverged++;
    if (!(q.I > 0)) continue;
    refErr.push(Math.abs(ref.entries[i * 4 + 1] / q.I - 1)); shipErr.push(Math.abs(shipped["f32 1e-5 (shipped)"].entries[i * 4 + 1] / q.I - 1)); dsErr.push(Math.abs(shipped["double-single path 1e-7"].entries[i * 4 + 1] / q.I - 1));
    if (k % 25 === 0) { console.log(`arm independent reference ${k}, ${((performance.now() - t0) / 1000).toFixed(0)} s`); await new Promise((f) => setTimeout(f, 0)); }
  }
  const toPng = (data: Uint8Array | Uint8ClampedArray) => { const c = document.createElement("canvas"); c.width = N; c.height = N; const g = c.getContext("2d")!, im = g.createImageData(N, N); im.data.set(data); g.putImageData(im, 0, 0); return c.toDataURL("image/png"); };
  const info = r.adapterInfo;
  return { images: { reference: toPng(ref.display) }, report: { id, a: s.a, inclDeg: s.inclDeg, N, adapter: `${info.vendor} ${info.architecture}`.trim(), refMs, reference: refSci, target: T,
    control: { sameSourceIdenticalPixels: identical, pixels: n, image: compareDisplay(own.display, same.display) }, arms,
    independent: { rays: refErr.length, unconverged, reference: summarize(refErr), shipped: summarize(shipErr), dsPath: summarize(dsErr), seconds: (performance.now() - t0) / 1000 } } };
}
