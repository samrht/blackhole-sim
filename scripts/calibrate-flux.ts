// Reproduces FLUX_SPIN in src/physics/flux-history.ts (run: npx vite-node scripts/calibrate-flux.ts, ~2 min).
// Per spin: dbar = 2 sigma/mu of the series (self-consistent) and eps so the 1000 M modulation index hits fluxTarget(a).
import { fluxSeries, modulationIndex, seriesSigma, fluxTarget, fluxDeficit } from "../src/physics/flux-history";
for (const a of [0, 0.3, 0.6, 0.9]) {
  const tg = fluxTarget(a); let db = 0.2, eps = 0.06;
  const solveEps = (d: number, it: number) => { let lo = 0, hi = 0.3; for (let k = 0; k < it; k++) { const e = (lo + hi) / 2;
    if (modulationIndex(fluxSeries(d, e), 1000) < tg) lo = e; else hi = e; } return (lo + hi) / 2; };
  for (let it = 0; it < 12; it++) { db = Math.round(db * 1e4) / 1e4; eps = solveEps(db, 22);
    const nd = Math.round(2 * seriesSigma(fluxSeries(db, eps)) * 1e4) / 1e4; if (Math.abs(nd - db) < 2e-4) { db = nd; break; } db = nd; }
  eps = Math.round(solveEps(db, 30) * 1e5) / 1e5;
  let d1 = 0, d2 = 0, d3 = 0, n = 0;
  for (let t = 0.5; t < 1e6; t += 1) { const d = fluxDeficit(t, db); d1 += d; d2 += d * d; d3 += d * d * d; n++; }
  console.log(`{ a: ${a}, dbar: ${db}, eps: ${eps}, d1: ${(d1 / n).toFixed(6)}, d2: ${(d2 / n).toFixed(6)}, d3: ${(d3 / n).toFixed(6)} },`);
}
