// Reproduces FLUX.dbar, d1, d2, d3 in src/physics/flux-history.ts (run: npx vite-node scripts/calibrate-flux.ts).
import { FLUX, measureFlux } from "../src/physics/flux-history";
let lo = 0.3, hi = 0.7;
for (let i = 0; i < 40; i++) { const m = (lo + hi) / 2; if (measureFlux(m, 1e6, 2).sigmaOverMu < FLUX.target) lo = m; else hi = m; }
const dbar = Math.round(lo * 1e4) / 1e4, M = measureFlux(dbar, 1e6, 1);
console.log(`dbar: ${dbar}, d1: ${M.d1.toFixed(6)}, d2: ${M.d2.toFixed(6)}, d3: ${M.d3.toFixed(6)}, sigma/mu ${M.sigmaOverMu.toFixed(4)}`);
