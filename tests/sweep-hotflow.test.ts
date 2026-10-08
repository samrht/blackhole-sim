import { describe, it, expect } from "vitest";
import { HOTFLOW_TARGETS, RING_ROBUST_UAS, TWIN_GRID, traceFlowSamples, flowImage, imageFluxJy, ringDiameterUas } from "../src/physics/hot-flow-image";
import { flowN0 } from "../src/physics/hot-flow";
import { PRESETS } from "../src/physics/presets";
const RUN = process.env.SWEEP === "1";
// spec 2026-10-08: M87* gated inside the EHT's 2 sigma and robust to a 15 uas blur; Sgr A* reported (user decision)
describe.skipIf(!RUN)("hot flow calibration (96^2 flux + 192^2 twin-grid rings, ~8 min per object)", () => {
  for (const id of ["sgra", "m87"] as const) {
    it(`${id}: tabulated n0 reproduces the measured 230 GHz flux within 5 %; ring = CPU twin value, robust; M87* inside 2 sigma`, () => {
      const p = PRESETS.find((q) => q.id === id)!, T = HOTFLOW_TARGETS[id];
      const rg = 1.476625e5 * p.massSun, dist = T.distKpc * 3.0857e21, uasPerM = (rg / dist) * 206264.806e6;
      const smp = traceFlowSamples(p.a, p.inclDeg, 96, 13);
      const I = flowImage(smp, flowN0(p.massSun, p.lambda), rg);
      expect(Math.abs(imageFluxJy(I, 96, 13, rg, dist) / T.jy - 1)).toBeLessThan(0.05);
      // rings on the GPU's frame (TWIN_GRID): the CPU twin values ?hotflow checks
      const G = TWIN_GRID, It = flowImage(traceFlowSamples(p.a, p.inclDeg, G.N, G.half), flowN0(p.massSun, p.lambda), rg);
      const d = ringDiameterUas(It, G.N, G.half, uasPerM), d15 = ringDiameterUas(It, G.N, G.half, uasPerM, 15);
      console.log(`${id}: ring ${d.toFixed(1)} uas (15 uas blur ${d15.toFixed(1)}) vs EHT ${T.ringUas} +- ${T.ringErr}`);
      expect(Math.abs(d - T.cpuRingUas)).toBeLessThan(0.05);
      expect(Math.abs(d15 - T.cpuRingBlurUas)).toBeLessThan(0.05);
      // EHT-gated rings (M87*) must be robust to the blur and inside 2 sigma; Sgr A*'s flat-topped ring is reported
      if (T.ehtGated) { expect(Math.abs(d - d15)).toBeLessThanOrEqual(RING_ROBUST_UAS); expect(Math.abs(d - T.ringUas)).toBeLessThan(2 * T.ringErr); }
    }, 600000);
  }
});
