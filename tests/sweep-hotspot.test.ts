import { describe, it, expect } from "vitest";
import { traceFlowSamples, flowImage, imageFluxJy, HOTFLOW_TARGETS } from "../src/physics/hot-flow-image";
import { flowN0 } from "../src/physics/hot-flow";
import { HOTSPOT, HOTSPOT_REACH, HOTSPOT_LPEAK } from "../src/physics/hotspot";
import { PRESETS } from "../src/physics/presets";
const RUN = process.env.SWEEP === "1";
describe.skipIf(!RUN)("hotspot calibration and the live-window bound (48^2 CPU traces)", () => {
  it("tabulated A0 adds 0.3 Jy to Sgr A* within 5 % (r_c 10, peak of L, frozen, 4 phases)", () => {
    const p = PRESETS.find((q) => q.id === "sgra")!, T = HOTFLOW_TARGETS.sgra;
    const rg = 1.476625e5 * p.massSun, dist = T.distKpc * 3.0857e21, n0 = flowN0(p.massSun, p.lambda);
    const smp = traceFlowSamples(p.a, p.inclDeg, 48, 14), F0 = imageFluxJy(flowImage(smp, n0, rg), 48, 14, rg, dist);
    let add = 0;
    for (const phiC of [0, Math.PI / 2, Math.PI, 1.5 * Math.PI]) {
      const hs = () => ({ rc: 10, phiC, amp: HOTSPOT.A0 * HOTSPOT_LPEAK, Om: 1 / (10 ** 1.5 + p.a) });
      add += (imageFluxJy(flowImage(smp, n0, rg, hs), 48, 14, rg, dist) - F0) / 4;
    }
    console.log(`hotspot adds ${add.toFixed(3)} Jy (target 0.3)`);
    expect(Math.abs(add / 0.3 - 1)).toBeLessThan(0.05);
  }, 600000);
  // The presets with hot flows, and the spin / inclination extremes a Custom view can reach. Rays through the blob region
  // with longer delays are higher-order photon-ring images: demagnified ~e^-pi per half orbit, and at the window's edges
  // the light curve L is already ~0 (it rises over 0.1 P and ends smoothly by 3 P), so they carry no visible hotspot light.
  const VIEWS: [string, number, number][] = [
    ...["sgra", "gargantua", "m87"].map((id): [string, number, number] => { const p = PRESETS.find((q) => q.id === id)!; return [id, p.a, p.inclDeg]; }),
    ["custom a=0.998 i=1", 0.998, 1], ["custom a=0.998 i=89", 0.998, 89],
  ];
  for (const [id, a, incl] of VIEWS) {
    it(`${id}: every ray through the blob region has |light-travel delay| < HOTSPOT.pad`, () => {
      const smp = traceFlowSamples(a, incl, 48, 14);
      let worst = 0;
      for (const q of smp.rays) for (let k = 0; k < q.length; k += 6) {
        const r = q[k], th = q[k + 1], t = q[k + 5];
        if (r < HOTSPOT_REACH && Math.abs(r * Math.cos(th)) < HOTSPOT.cut * HOTSPOT.sigma) worst = Math.max(worst, Math.abs(-t - 1000));
      }
      console.log(`${id}: max |delay| through the blob region ${worst.toFixed(1)} M (pad ${HOTSPOT.pad})`);
      expect(worst).toBeLessThan(HOTSPOT.pad);
    }, 600000);
  }
});
