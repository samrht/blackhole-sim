// ?hotflow (spec 2026-10-04 hot flow): the GPU renderer's own 1.3 mm image of Sgr A* and M87*, measured like the CPU
// calibration (hot-flow-image.ts): total flux at the preset distance and the ring diameter (azimuthal-mean peak,
// unblurred). PASS iff Sgr A*'s ring is within 2 sigma of the EHT's 51.8 uas and, for both objects, the flux is within
// 5 % of the measured value its n0 was fitted to and the ring within 0.5 uas of the CPU twin's (twin checks). Also reported: what the jet model, if drawn at 230 GHz (the shader can, the app does not),
// would add to M87*'s frame -- the measurement behind spec 2.5's decision not to draw the jet at 1.3 mm.
import { Renderer } from "../render/gpu";
import { SCENES, prepareScene, sceneUniforms, type Scene } from "./scenes";
import { HOTFLOW_TARGETS, ringDiameterUas, imageCentroid } from "../physics/hot-flow-image";
import { HOTSPOT_TWIN } from "../physics/hotspot";
import type { UniformValues } from "../render/uniforms";
import { HOTFLOW } from "../physics/hot-flow";
import { PRESETS } from "../physics/presets";

const N = 512, FRAMES = 16;
export interface HotFlowRow { name: string; jy: number; ringUas: number; peakTb: number }

async function image(r: Renderer, s: Scene, extra: Partial<UniformValues> = {}): Promise<Float64Array> {
  const rIn = prepareScene(r, s);
  for (let k = 0; k < FRAMES; k++) r.frame(sceneUniforms(r, s, rIn, { frame: k, reset: k === 0 ? 1 : 0, blend: 1 / (k + 1), ...extra }));
  await r.device.queue.onSubmittedWorkDone();
  const acc = await r.readbackAccum(), I = new Float64Array(N * N);
  for (let i = 0; i < N * N; i++) I[i] = acc[i * 4] / HOTFLOW.kTb;
  return I;
}
const HALF = 14; // fovScale 14 on a square canvas
const geom = (id: string) => {
  const p = PRESETS.find((q) => q.id === id)!, T = HOTFLOW_TARGETS[id as "sgra" | "m87"];
  const rg = 1.476625e5 * p.massSun, dist = T.distKpc * 3.0857e21;
  return { rg, dist, uasPerM: (rg / dist) * 206264.806e6 };
};
function fluxJy(I: Float64Array, id: string): number {
  const { rg, dist } = geom(id), pix = ((2 * HALF) / N) * (rg / dist);
  let F = 0; for (const v of I) F += v * pix * pix;
  return F / 1e-23;
}
async function measure(r: Renderer, s: Scene): Promise<HotFlowRow> {
  const I = await image(r, s);
  let peakTb = 0; for (const v of I) peakTb = Math.max(peakTb, v * HOTFLOW.kTb);
  return { name: s.name, jy: fluxJy(I, s.preset!), ringUas: ringDiameterUas(I, N, HALF, geom(s.preset!).uasPerM), peakTb };
}
// Hotspot (spec 2026-10-04 mm hotspots 5): Sgr A* at the peak of eruption 0's hotspot minus the same frame with the slider
// at 0 (in mm, fluxVar drives only the hotspots): the flux it adds and its centroid must match the CPU twin
// (HOTSPOT_TWIN, scripts/calibrate-hotspot.ts) within 5 % and 0.5 M, with light-travel delay on and off.
async function hotspotRow(r: Renderer, ld: 0 | 1) {
  const s = { ...SCENES.find((x) => x.name === "sgra-mm-hotspot")!, lightDelay: ld };
  const on = await image(r, s), off = await image(r, s, { fluxVar: 0 }), d = on.map((v, i) => v - off[i]);
  const want = ld ? HOTSPOT_TWIN.delay : HOTSPOT_TWIN.instant, jy = fluxJy(d, "sgra"), [cx, cy] = imageCentroid(d, N, HALF);
  const ok = Math.abs(jy / want.jy - 1) < 0.05 && Math.hypot(cx - want.cx, cy - want.cy) < 0.5;
  return { ok, line: `hotspot (light delay ${ld}): adds ${jy.toFixed(3)} Jy at (${cx.toFixed(2)}, ${cy.toFixed(2)}) M; CPU twin ${want.jy} Jy at (${want.cx}, ${want.cy}) ${ok ? "ok" : "FAILED"}` };
}

export async function runHotFlow(canvas: HTMLCanvasElement): Promise<{ ok: boolean; lines: string[] }> {
  canvas.style.width = `${N}px`; canvas.style.height = `${N}px`;
  const r = new Renderer(); await r.init(canvas);
  if (r.width !== N || r.height !== N) throw new Error(`hotflow: canvas is ${r.width}x${r.height}, want ${N}x${N} (device pixel ratio 1)`);
  const sg = SCENES.find((s) => s.name === "sgra-mm")!, m87 = SCENES.find((s) => s.name === "m87-mm")!;
  const a = await measure(r, sg), b = await measure(r, m87), c = await measure(r, { ...m87, name: "m87-mm with the model jet (not drawn)", jetStrength: 1 });
  const T = HOTFLOW_TARGETS.sgra, M = HOTFLOW_TARGETS.m87;
  const twin = (x: HotFlowRow, t: typeof T | typeof M) => Math.abs(x.jy / t.jy - 1) < 0.05 && Math.abs(x.ringUas - t.cpuRingUas) < 0.5;
  const sgOk = twin(a, T), m87Ok = twin(b, M), ehtOk = Math.abs(a.ringUas - T.ringUas) < 2 * T.ringErr;
  const h1 = await hotspotRow(r, 1), h0 = await hotspotRow(r, 0);
  const row = (x: HotFlowRow) => `${x.name}: ${x.jy.toFixed(3)} Jy, ring ${x.ringUas.toFixed(1)} uas, peak T_b ${x.peakTb.toExponential(2)} K`;
  return { ok: sgOk && m87Ok && ehtOk && h1.ok && h0.ok, lines: [
    `${row(a)} (twin: ${T.jy} Jy +- 5 %, CPU ring ${T.cpuRingUas} +- 0.5 ${sgOk ? "ok" : "FAILED"}; EHT ${T.ringUas} +- ${T.ringErr} ${ehtOk ? "ok" : "FAILED"})`,
    `${row(b)} (twin: ${M.jy} Jy +- 5 %, CPU ring ${M.cpuRingUas} +- 0.5 ${m87Ok ? "ok" : "FAILED"}; EHT ${M.ringUas} +- ${M.ringErr}, reported)`,
    `${row(c)}: the jet would add ${(c.jy - b.jy).toFixed(3)} Jy (${((100 * (c.jy - b.jy)) / b.jy).toFixed(0)} % of the flow; EHT: jet base <~10 % of the ring)`,
    h1.line, h0.line,
  ] };
}
