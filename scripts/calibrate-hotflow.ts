// Reproduces HOTFLOW_N0 in src/physics/hot-flow.ts (run: npx vite-node scripts/calibrate-hotflow.ts, ~6 min).
import { HOTFLOW_TARGETS, TWIN_GRID, traceFlowSamples, flowImage, imageFluxJy, ringDiameterUas } from "../src/physics/hot-flow-image";
import { PRESETS } from "../src/physics/presets";
for (const id of ["sgra", "m87"] as const) {
  const p = PRESETS.find((q) => q.id === id)!, T = HOTFLOW_TARGETS[id], rg = 1.476625e5 * p.massSun, dist = T.distKpc * 3.0857e21;
  const smp = traceFlowSamples(p.a, p.inclDeg, 96, 13); let lo = 1e3, hi = 1e10;
  for (let k = 0; k < 40; k++) { const m = Math.sqrt(lo * hi); if (imageFluxJy(flowImage(smp, m, rg), 96, 13, rg, dist) < T.jy) lo = m; else hi = m; }
  const I = flowImage(smp, lo, rg), u = (rg / dist) * 206264.806e6;
  console.log(`${id}: n0 ${lo.toPrecision(3)} cm^-3 -> ${imageFluxJy(I, 96, 13, rg, dist).toFixed(3)} Jy; ring ${ringDiameterUas(I, 96, 13, u).toFixed(1)} uas (15 uas blur: ${ringDiameterUas(I, 96, 13, u, 15).toFixed(1)}) vs EHT ${T.ringUas} +- ${T.ringErr}; ${u.toFixed(2)} uas/M`);
  // the CPU twin of ?hotflow's GPU rings, on the GPU's frame (TWIN_GRID), at the n0 written to HOTFLOW_N0 (3 s.f.)
  const n0t = Number(lo.toPrecision(3)), It = flowImage(traceFlowSamples(p.a, p.inclDeg, TWIN_GRID.N, TWIN_GRID.half), n0t, rg);
  console.log(`${id}: twin grid ${TWIN_GRID.N}^2 half ${TWIN_GRID.half}: cpuRingUas ${ringDiameterUas(It, TWIN_GRID.N, TWIN_GRID.half, u).toFixed(1)}, cpuRingBlurUas ${ringDiameterUas(It, TWIN_GRID.N, TWIN_GRID.half, u, 15).toFixed(1)}`);
}
