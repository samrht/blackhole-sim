// Builds public/synch-table.bin from src/physics/cyclosynch.ts on every core (Node >= 23 runs the .ts directly).
// ~20 min on 8 cores: the exact harmonic sums (u = 0.01..10) dominate. ROWS=<file> caches the s-independent
// kernel rows, so changing only the s grid re-contracts in seconds.
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { writeFileSync, readFileSync, existsSync } from "node:fs";
import { availableParallelism } from "node:os";
const M = await import(new URL("../src/physics/cyclosynch.ts", import.meta.url).href);
if (!isMainThread) {
  const { idx } = workerData, uu = M.uGrid(), g = M.TABLE_GRID;
  for (const k of idx) { const row = new Float64Array(g.nx); M.kernelRow(uu.u[k], g, row); parentPort.postMessage({ k, row }, [row.buffer]); }
  parentPort.postMessage({ done: true });
} else {
  const g = M.TABLE_GRID, uu = M.uGrid(), W = availableParallelism(), rows = new Array(uu.u.length), cache = process.env.ROWS;
  const t0 = Date.now();
  if (cache && existsSync(cache)) {
    const all = new Float64Array(readFileSync(cache).buffer.slice(0));
    for (let k = 0; k < uu.u.length; k++) rows[k] = all.slice(k * g.nx, (k + 1) * g.nx);
  } else {
    let left = W, got = 0;
    await new Promise((resolve, reject) => {
      for (let w = 0; w < W; w++) {
        const idx = []; for (let k = w; k < uu.u.length; k += W) idx.push(k); // interleaved: spreads the costly rows
        const wk = new Worker(new URL(import.meta.url), { workerData: { idx } });
        wk.on("message", (m) => { if (m.done) { if (--left === 0) resolve(); return; } rows[m.k] = m.row;
          if (++got % 50 === 0) console.log(`kernel rows ${got}/${uu.u.length}, ${((Date.now() - t0) / 1000).toFixed(0)} s`); });
        wk.on("error", reject);
      }
    });
    if (cache) { const all = new Float64Array(uu.u.length * g.nx); rows.forEach((r, k) => all.set(r, k * g.nx)); writeFileSync(cache, Buffer.from(all.buffer)); }
  }
  const { J, A } = M.contract(rows, uu, g);
  let zeroA = 0; for (const v of A) if (!(v > 0)) zeroA++;
  writeFileSync(new URL("../public/synch-table.bin", import.meta.url), Buffer.from(M.encodeTable(J, A, g)));
  console.log(`wrote public/synch-table.bin: ${g.nx} x ${g.ns}, non-positive absorption cells ${zeroA}, ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
