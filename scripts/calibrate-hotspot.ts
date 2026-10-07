// Reproduces HOTSPOT.A0 and HOTSPOT_TWIN in src/physics/hotspot.ts (run: npx vite-node scripts/calibrate-hotspot.ts, ~10 min).
// A0: Sgr A* (96^2, half 14 = ?hotflow's view), hotspot at r_c = 10 with mean depth at the peak of L (amp = A0 L_peak),
// emission time frozen; the flux it adds, averaged over four orbital phases, is ALMA's 0.3 Jy (Wielgus et al. 2022 S3.1).
// HOTSPOT_TWIN: the CPU twin of ?hotflow's GPU measurement (eruption 0 at its peak, the real schedule, light delay 1 and 0).
import { HOTFLOW_TARGETS, traceFlowSamples, flowImage, imageFluxJy, imageCentroid } from "../src/physics/hot-flow-image";
import { flowN0 } from "../src/physics/hot-flow";
import { HOTSPOT_LPEAK, hotspotAt, hotspotPeakTime } from "../src/physics/hotspot";
import { PRESETS } from "../src/physics/presets";
const p = PRESETS.find((q) => q.id === "sgra")!, T = HOTFLOW_TARGETS.sgra, N = 96, H = 14;
const rg = 1.476625e5 * p.massSun, dist = T.distKpc * 3.0857e21, n0 = flowN0(p.massSun, p.lambda);
const smp = traceFlowSamples(p.a, p.inclDeg, N, H), base = flowImage(smp, n0, rg), F0 = imageFluxJy(base, N, H, rg, dist);
const added = (amp: number) => {
  let s = 0;
  for (const phiC of [0, Math.PI / 2, Math.PI, 1.5 * Math.PI])
    s += imageFluxJy(flowImage(smp, n0, rg, () => ({ rc: 10, phiC, amp, Om: 1 / (10 ** 1.5 + p.a) })), N, H, rg, dist) - F0;
  return s / 4;
};
let lo = 0.1, hi = 200;
for (let k = 0; k < 40; k++) { const m = Math.sqrt(lo * hi); if (added(m) < 0.3) lo = m; else hi = m; }
const A0 = Number((lo / HOTSPOT_LPEAK).toPrecision(3));
console.log(`A0 ${A0} (peak amp ${lo.toFixed(3)}, adds ${added(A0 * HOTSPOT_LPEAK).toFixed(4)} Jy over ${F0.toFixed(3)} Jy)`);
const t = hotspotPeakTime(0, p.a), hs = (te: number) => { const h = hotspotAt(te, 1, p.a, A0); return h.alive ? h : null; };
for (const [name, ld] of [["delay", 1], ["instant", 0]] as const) {
  const I = flowImage(smp, n0, rg, hs, t, ld), d = I.map((v, i) => v - base[i]);
  const [cx, cy] = imageCentroid(d, N, H);
  console.log(`HOTSPOT_TWIN.${name} = { jy: ${(imageFluxJy(d, N, H, rg, dist)).toFixed(4)}, cx: ${cx.toFixed(3)}, cy: ${cy.toFixed(3)} }  (t = ${t.toFixed(2)} M)`);
}
