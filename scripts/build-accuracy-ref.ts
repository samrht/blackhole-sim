// Builds src/test/accuracy-ref.json, the CPU side of ?accuracy (spec 2026-10-07 mino integrator §5, plan Task 3 ruling):
// for each scene, the converged reference (trace-reference.ts) of every pixel ray of a 40 x 40 build at jitter set 0 (the
// rays the renderer's cache build traces, so the shipped integrator is measured, not a copy). Disk rays: rHit, phiHit,
// delay; sky rays: the direction and the local lensing Jacobian at the app's pixel scale (720 px over the 28 M field);
// 1.3 mm flow rays: the 230 GHz intensity. Run: npx vite-node scripts/build-accuracy-ref.ts (~30 min, background).
import { readFileSync, writeFileSync } from "node:fs";
import { screenToState } from "../src/physics/camera";
import { iscoRadius } from "../src/physics/orbits";
import { convergedRef, refTrace, flowReference, REF1, REF_ROBS, type FlowObj } from "../src/physics/trace-reference";
import { flowN0 } from "../src/physics/hot-flow";
import { jetUniforms, ETA_DEFAULT } from "../src/physics/synchrotron";
import { PRESETS } from "../src/physics/presets";
import { JITTER } from "../src/render/cache-plan";
import { pixelSens } from "../src/test/accuracy-sens";

export const ACC_N = 40, ACC_FOV = 14;
const PX = (2 * ACC_FOV) / 720;
const preset = (id: string) => PRESETS.find((p) => p.id === id)!;
const flowOf = (id: string): FlowObj => { const p = preset(id); return { n0: flowN0(p.massSun, p.lambda), rg: jetUniforms(p.massSun, p.a, p.lambda, ETA_DEFAULT, 60, 2, 1).rgCm }; };
export const ACC_SCENES: { name: string; a: number; inclDeg: number; flow?: string }[] = [
  { name: "default", a: 0.9, inclDeg: 72 }, { name: "face-on", a: 0.9, inclDeg: 8 }, { name: "edge-on", a: 0.99, inclDeg: 85 },
  { name: "schwarzschild", a: 0, inclDeg: 30 }, { name: "sgra-mm", a: preset("sgra").a, inclDeg: preset("sgra").inclDeg, flow: "sgra" },
  { name: "garg-mm", a: preset("gargantua").a, inclDeg: preset("gargantua").inclDeg, flow: "gargantua" },
];
/** Pixel (i, j) of an N x N build at jitter set 0: raytrace.wgsl pixelImpact (aspect 1). */
export function accImpact(i: number, j: number, N = ACC_N): [number, number] {
  const [jx, jy] = JITTER[0];
  const nx = ((i + 0.5 + jx) / N) * 2 - 1, ny = ((j + 0.5 + jy) / N) * 2 - 1;
  return [nx * ACC_FOV, -ny * ACC_FOV];
}
const sig = (x: number) => Number(x.toPrecision(10));
const out: Record<string, unknown> = { N: ACC_N, fov: ACC_FOV, jitter: JITTER[0], scenes: [] as unknown[] };
const t0 = Date.now();
// ONLY=sgra-mm,garg-mm rebuilds those scenes and keeps every other scene of the existing fixture as it is; JROWS=a,b
// rebuilds only pixel rows a <= j < b of them (the rest from the existing fixture), so a rebuild can run in short chunks.
const ONLY = process.env.ONLY?.split(",");
const JR = process.env.JROWS?.split(",").map(Number);
const prev = ONLY ? JSON.parse(readFileSync("src/test/accuracy-ref.json", "utf8")) : null;
for (const sc of ACC_SCENES) {
  if (ONLY && !ONLY.includes(sc.name)) { (out.scenes as unknown[]).push(prev.scenes.find((s: { name: string }) => s.name === sc.name)); continue; }
  const prevSc = prev?.scenes.find((s: { name: string }) => s.name === sc.name);
  const incl = (sc.inclDeg * Math.PI) / 180, rIn = iscoRadius(sc.a, true), fl = sc.flow ? flowOf(sc.flow) : undefined;
  const rays: unknown[] = []; let un = 0;
  for (let j = 0; j < ACC_N; j++) for (let i = 0; i < ACC_N; i++) {
    if (JR && (j < JR[0] || j >= JR[1])) { rays.push(prevSc.rays[j * ACC_N + i]); continue; }
    const [al, be] = accImpact(i, j);
    const s0 = screenToState(al, be, sc.a, incl, REF_ROBS);
    if (fl) { // 1.3 mm: the exact intensity and the renderer rule's own resolution q along the exact path
      const f = flowReference(s0, sc.a, fl); if (!f.converged) un++;
      rays.push({ f: f.fate, c: f.converged ? 1 : 0, I: f.I, q: f.q }); continue;
    }
    const { ref, converged } = convergedRef(s0, sc.a, rIn);
    if (!converged) un++;
    const row: Record<string, unknown> = { f: ref.fate, c: converged ? 1 : 0 };
    if (ref.fate === "disk") { row.r = sig(ref.rHit!); row.p = sig(ref.phiHit!); row.d = sig(-ref.tHit! - REF_ROBS); }
    else if (ref.fate === "escaped") {
      row.v = ref.dir!.map(sig);
      if (converged) {
        const d1 = refTrace(screenToState(al + PX, be, sc.a, incl, REF_ROBS), sc.a, rIn, REF1);
        const d2 = refTrace(screenToState(al, be + PX, sc.a, incl, REF_ROBS), sc.a, rIn, REF1);
        if (d1.fate === "escaped" && d2.fate === "escaped") row.J = [d1.dir!.map((v, k) => sig(v - ref.dir![k])), d2.dir!.map((v, k) => sig(v - ref.dir![k]))];
      }
    }
    // f32 input conditioning (src/test/accuracy-sens.ts): the gate's per-pixel floor
    if (converged && (ref.fate === "disk" || ref.fate === "escaped")) row.s = pixelSens(al, be, sc.a, sc.inclDeg, rIn, row.J as number[][] | undefined).map((x) => (Number.isFinite(x) ? sig(x) : 1e9));
    rays.push(row);
  }
  (out.scenes as unknown[]).push({ name: sc.name, a: sc.a, inclDeg: sc.inclDeg, rIn, flow: fl ?? null, rays });
  console.log(`${sc.name}: ${rays.length} rays, unconverged ${un} (${((Date.now() - t0) / 60000).toFixed(1)} min)`);
}
writeFileSync("src/test/accuracy-ref.json", JSON.stringify(out));
console.log("wrote src/test/accuracy-ref.json");
