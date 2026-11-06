// One more CPU arm without rerunning scripts/limit-cpu.ts: exact f64 arithmetic with only the drift check's noise floor at f32's epsilon,
// at each tolerance. Adds to data/<config>.json; scripts/limit-run.mjs merges it into results.  npx vite-node scripts/limit-eps.ts <config>
import { readFileSync, writeFileSync } from "node:fs";
import { minoTrace, setMinoRounding } from "../src/physics/trace-mino";
import { skyDirCPU } from "../src/physics/trace-reference";
import { rayError, summarizeArm, type ArmRow, type Obs } from "../src/test/limit-score";
import { groupsOf, F32_TOLS } from "../src/test/limit-config";

const name = process.argv[2], file = `docs/diagnostics/2026-11-06-accuracy-limit/data/${name}.json`, d = JSON.parse(readFileSync(file, "utf8"));
const absDist: number[] = d.dist.map(Math.abs), groups = groupsOf(d.rays, d.refs.map((r: Obs) => r.f)), rOut = 40;
setMinoRounding({ eps: true });
for (const tol of F32_TOLS) {
  const rows: ArmRow[] = d.rays.map(([al, be]: [number, number], i: number) => {
    const m = minoTrace(al, be, d.config.a, d.config.inclDeg, { rIn: d.rIn, rOut, rObs: 1000, tol });
    const g: Obs = { f: m.fate as Obs["f"], r: m.rHit, p: m.phiHit, t: m.tHit, dir: m.fate === "escaped" ? skyDirCPU(m.s, d.config.a) : undefined, steps: m.steps, attempts: m.attempts };
    return { ...rayError(d.refs[i], d.jac[i] ?? undefined, g), steps: g.steps, attempts: g.attempts };
  });
  d.arms[`mino f64, f32 noise floor ${tol}`] = summarizeArm(absDist, rows, groups);
  console.log(name, tol, "done");
}
writeFileSync(file, JSON.stringify(d, (_, v) => (typeof v === "number" && !Number.isFinite(v) ? null : v)));
