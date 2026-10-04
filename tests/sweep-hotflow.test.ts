import { describe, it, expect } from "vitest";
import { HOTFLOW_TARGETS, traceFlowSamples, flowImage, imageFluxJy, ringDiameterUas } from "../src/physics/hot-flow-image";
import { flowN0 } from "../src/physics/hot-flow";
import { PRESETS } from "../src/physics/presets";
const RUN = process.env.SWEEP === "1";
describe.skipIf(!RUN)("hot flow calibration (96^2 CPU trace, ~1.5 min per object; 48^2 and 64^2 read the ring one 2.2 uas bin low, 96^2 = 144^2 = 48.2)", () => {
  for (const id of ["sgra", "m87"] as const) {
    it(`${id}: tabulated n0 reproduces the measured 230 GHz flux within 5 %; ring reported (Sgr A* gated within 2 sigma)`, () => {
      const p = PRESETS.find((q) => q.id === id)!, T = HOTFLOW_TARGETS[id];
      const rg = 1.476625e5 * p.massSun, dist = T.distKpc * 3.0857e21, uasPerM = (rg / dist) * 206264.806e6;
      const smp = traceFlowSamples(p.a, p.inclDeg, 96, 13);
      const I = flowImage(smp, flowN0(p.massSun, p.lambda), rg);
      expect(Math.abs(imageFluxJy(I, 96, 13, rg, dist) / T.jy - 1)).toBeLessThan(0.05);
      const d = ringDiameterUas(I, 96, 13, uasPerM);
      console.log(`${id}: ring ${d.toFixed(1)} uas vs EHT ${T.ringUas} +- ${T.ringErr}`);
      if (id === "sgra") expect(Math.abs(d - T.ringUas)).toBeLessThan(2 * T.ringErr);
    }, 600000);
  }
});
