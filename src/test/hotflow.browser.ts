// ?hotflow (spec 2026-10-04 hot flow): the GPU renderer's own 1.3 mm image of Sgr A* and M87*, measured like the CPU
// calibration (hot-flow-image.ts): total flux at the preset distance and the ring diameter (azimuthal-mean peak,
// unblurred). PASS iff Sgr A*'s flux is within 5 % of 2.4 Jy and its ring within 2 sigma of the EHT's 51.8 uas; M87*'s
// flux and ring are reported. Also reported: what the jet model, if drawn at 230 GHz (the shader can, the app does not),
// would add to M87*'s frame -- the measurement behind spec 2.5's decision not to draw the jet at 1.3 mm.
import { Renderer } from "../render/gpu";
import { SCENES, prepareScene, sceneUniforms, type Scene } from "./scenes";
import { HOTFLOW_TARGETS, ringDiameterUas } from "../physics/hot-flow-image";
import { HOTFLOW } from "../physics/hot-flow";
import { PRESETS } from "../physics/presets";

const N = 512, FRAMES = 16;
export interface HotFlowRow { name: string; jy: number; ringUas: number; peakTb: number }

async function measure(r: Renderer, s: Scene): Promise<HotFlowRow> {
  const rIn = prepareScene(r, s), p = PRESETS.find((q) => q.id === s.preset)!, T = HOTFLOW_TARGETS[s.preset as "sgra" | "m87"];
  for (let k = 0; k < FRAMES; k++) r.frame(sceneUniforms(r, s, rIn, { frame: k, reset: k === 0 ? 1 : 0, blend: 1 / (k + 1) }));
  await r.device.queue.onSubmittedWorkDone();
  const acc = await r.readbackAccum(), I = new Float64Array(N * N);
  let peakTb = 0;
  for (let i = 0; i < N * N; i++) { I[i] = acc[i * 4] / HOTFLOW.kTb; peakTb = Math.max(peakTb, acc[i * 4]); }
  const rg = 1.476625e5 * p.massSun, dist = T.distKpc * 3.0857e21, half = 14; // fovScale 14 on a square canvas
  const pix = ((2 * half) / N) * (rg / dist);
  let F = 0; for (const v of I) F += v * pix * pix;
  return { name: s.name, jy: F / 1e-23, ringUas: ringDiameterUas(I, N, half, (rg / dist) * 206264.806e6), peakTb };
}

export async function runHotFlow(canvas: HTMLCanvasElement): Promise<{ ok: boolean; lines: string[] }> {
  canvas.style.width = `${N}px`; canvas.style.height = `${N}px`;
  const r = new Renderer(); await r.init(canvas);
  if (r.width !== N || r.height !== N) throw new Error(`hotflow: canvas is ${r.width}x${r.height}, want ${N}x${N} (device pixel ratio 1)`);
  const sg = SCENES.find((s) => s.name === "sgra-mm")!, m87 = SCENES.find((s) => s.name === "m87-mm")!;
  const a = await measure(r, sg), b = await measure(r, m87), c = await measure(r, { ...m87, name: "m87-mm with the model jet (not drawn)", jetStrength: 1 });
  const T = HOTFLOW_TARGETS.sgra;
  const fluxOk = Math.abs(a.jy / T.jy - 1) < 0.05, ringOk = Math.abs(a.ringUas - T.ringUas) < 2 * T.ringErr;
  const row = (x: HotFlowRow) => `${x.name}: ${x.jy.toFixed(3)} Jy, ring ${x.ringUas.toFixed(1)} uas, peak T_b ${x.peakTb.toExponential(2)} K`;
  return { ok: fluxOk && ringOk, lines: [
    `${row(a)} (want ${T.jy} Jy +- 5 % ${fluxOk ? "ok" : "FAILED"}; EHT ${T.ringUas} +- ${T.ringErr} ${ringOk ? "ok" : "FAILED"})`,
    `${row(b)} (want ${HOTFLOW_TARGETS.m87.jy} Jy; EHT ${HOTFLOW_TARGETS.m87.ringUas} +- ${HOTFLOW_TARGETS.m87.ringErr}, reported)`,
    `${row(c)}: the jet would add ${(c.jy - b.jy).toFixed(3)} Jy (${((100 * (c.jy - b.jy)) / b.jy).toFixed(0)} % of the flow; EHT: jet base <~10 % of the ring)`,
  ] };
}
