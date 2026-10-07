// ?accuracy (spec 2026-10-07 mino integrator §5; plan Task 3 ruling): whole-ray accuracy of the SHIPPED renderer on the
// GPU. The cache build pass traces every pixel of a 40 x 40 view at jitter set 0 and stores what the ray found (disk:
// rHit, phiHit, delay; sky: direction; 1.3 mm hot flow: intensity), so reading those entries measures the renderer's own
// integrator, not a copy. Scored against the CPU converged reference (src/test/accuracy-ref.json, built by
// scripts/build-accuracy-ref.ts) and, per pixel, against the pre-Mino renderer's entries recorded on the same adapter
// (src/test/accuracy-old.json, `?accuracy&record` before the swap): no fate flips and no pixel worse than the old
// renderer beyond the floors (r, delay 1e-5 relative; phi 1e-5 rad; sky 0.01 px through the lensing Jacobian; intensity
// max(1e-5, 2 q) relative, q the renderer rule's own quadrature error along the exact path), each raised to the pixel's
// own f32 input conditioning where that is larger (row.s, src/test/accuracy-sens.ts: how far the converged answer moves
// under a 1-ulp nudge of the f32 alpha or beta; near the critical curve 2-4e-4 in rHit, for the old and the new renderer
// alike). A fate flip counts unless such a nudge flips the reference's own fate (s = 1e9).
// A different adapter than the recording's reports SKIP for the per-pixel comparison (like ?golden).
import { Renderer } from "../render/gpu";
import { BuildScheduler } from "../render/cache-plan";
import { prepareScene, sceneUniforms, type Scene } from "./scenes";
import { PRESETS } from "../physics/presets";
import { wrapAngle, pxError } from "../physics/trace-reference";

type RefRow = { f: string; c: number; r?: number; p?: number; d?: number; v?: number[]; J?: number[][]; I?: number; q?: number; s?: number[] };
type RefScene = { name: string; a: number; inclDeg: number; rIn: number; flow: { n0: number; rg: number } | null; rays: RefRow[] };
type Fixture = { N: number; fov: number; scenes: RefScene[] };
export type Entries = Record<string, number[][]>; // per scene: per pixel [kind, p0, p1, p2]
const KIND: Record<string, number> = { captured: 0, disk: 1, escaped: 2 };

function sceneOf(s: RefScene): Scene {
  if (!s.flow) return { name: s.name, a: s.a, inclDeg: s.inclDeg, time: 0, frame: 0, jetStrength: 0, skyStrength: 0 };
  const id = s.name === "sgra-mm" ? "sgra" : "gargantua", p = PRESETS.find((q) => q.id === id)!;
  return { name: s.name, a: s.a, inclDeg: s.inclDeg, time: 0, frame: 0, jetStrength: 0, skyStrength: 0, obj: { massSun: p.massSun, lambda: p.lambda }, band: "mm", preset: id };
}
/** Build jitter set 0 of every fixture scene and read back the entries. */
export async function measureEntries(canvas: HTMLCanvasElement, fx: Fixture): Promise<{ adapter: string; entries: Entries }> {
  canvas.style.width = `${fx.N}px`; canvas.style.height = `${fx.N}px`;
  const entries: Entries = {}; let adapter = "";
  for (const s of fx.scenes) {
    const r = new Renderer(); await r.init(canvas);
    adapter = `${r.adapterInfo.vendor} ${r.adapterInfo.architecture} ${r.adapterInfo.description}`.trim();
    if (r.displayW !== fx.N || r.displayH !== fx.N) throw new Error(`accuracy: canvas ${r.displayW}x${r.displayH}, want ${fx.N}x${fx.N} (device pixel ratio 1)`);
    if (r.cacheSets === 0) throw new Error("accuracy: the cache is unavailable on this adapter");
    const sc = sceneOf(s), rIn = prepareScene(r, sc), base = sceneUniforms(r, sc, rIn);
    r.resetCache();
    const sched = new BuildScheduler(r.cacheSets, r.displayH);
    for (let x = sched.next(); x && x.set === 0; x = sched.next())
      r.frame(base, { build: { ...base, resW: r.displayW, resH: r.displayH, setIndex: 0, rowStart: x.rowStart, rowEnd: x.rowEnd } });
    await r.device.queue.onSubmittedWorkDone();
    const words = await r.readbackEntries(0), f = new Float32Array(words.buffer, words.byteOffset, words.length), rows: number[][] = [];
    for (let k = 0; k < fx.N * fx.N; k++) rows.push([words[4 * k] & 3, f[4 * k + 1], f[4 * k + 2], f[4 * k + 3]]);
    entries[s.name] = rows;
  }
  return { adapter, entries };
}
type Err = { r: number; p: number; d: number; px: number; I: number };
function errors(row: RefRow, e: number[]): Err | null {
  if (e[0] !== KIND[row.f]) return null;
  if (row.I !== undefined) return { r: 0, p: 0, d: 0, px: 0, I: Math.abs(e[1] - row.I) / Math.max(Math.abs(row.I), 1e-300) };
  if (row.f === "disk") return { r: Math.abs(e[1] - row.r!) / row.r!, p: wrapAngle(e[2] - row.p!), d: Math.abs(e[3] - row.d!) / Math.max(1, Math.abs(row.d!)), px: 0, I: 0 };
  if (row.f === "escaped") { const v = [e[1], e[2], e[3]], n = Math.hypot(v[0], v[1], v[2]); return { r: 0, p: 0, d: 0, px: pxError(row.J, v.map((x) => x / n), row.v!), I: 0 }; }
  return { r: 0, p: 0, d: 0, px: 0, I: 0 };
}
const FLOOR: Err = { r: 1e-5, p: 1e-5, d: 1e-5, px: 0.01, I: 1e-5 };
export async function runAccuracy(canvas: HTMLCanvasElement): Promise<{ verdict: "PASS" | "FAIL" | "SKIP"; lines: string[] }> {
  const fx: Fixture = await (await fetch("/src/test/accuracy-ref.json")).json();
  const oldRes = await fetch("/src/test/accuracy-old.json");
  const old: { adapter: string; entries: Entries } | null = oldRes.ok ? await oldRes.json() : null;
  const got = await measureEntries(canvas, fx);
  const lines: string[] = []; let fail = false;
  const same = !!old && old.adapter === got.adapter;
  for (const s of fx.scenes) {
    const E = got.entries[s.name], O = same ? old!.entries[s.name] : null;
    let scored = 0, flips = 0; const worse: Record<string, number> = { r: 0, p: 0, d: 0, px: 0, I: 0 };
    const maxNew: Err = { r: 0, p: 0, d: 0, px: 0, I: 0 }, maxOld: Err = { r: 0, p: 0, d: 0, px: 0, I: 0 }; const examples: string[] = [];
    s.rays.forEach((row, k) => {
      if (!row.c) return;
      scored++;
      const en = errors(row, E[k]), eo = O ? errors(row, O[k]) : null;
      const edge = !!row.s && row.s[0] >= 1e9; // a 1-ulp nudge flips the reference's fate
      if (!en) { if ((!O || eo) && !edge) { flips++; if (examples.length < 4) examples.push(`#${k} fate ${E[k][0]} vs ref ${row.f}`); } return; }
      for (const q of ["r", "p", "d", "px", "I"] as const) {
        if (!Number.isFinite(en[q])) continue;
        maxNew[q] = Math.max(maxNew[q], en[q]);
        if (eo && Number.isFinite(eo[q])) maxOld[q] = Math.max(maxOld[q], eo[q]);
        // 1.3 mm: the renderer's 0.25 M left-sample rule has its own error q along the exact path (plan Task 2 ruling)
        const sens = row.s ? { r: row.s[0], p: row.s[1], d: row.s[2], px: row.s[3], I: 0 }[q] : 0;
        const floor = Math.max(sens, q === "I" && row.q !== undefined ? Math.max(FLOOR.I, (2 * row.q) / Math.max(row.I!, 1e-300)) : FLOOR[q]);
        if (O && en[q] > Math.max(eo ? eo[q] : Infinity, floor)) { worse[q]++; if (examples.length < 4) examples.push(`#${k} ${q} ${en[q].toExponential(2)} > old ${eo![q].toExponential(2)}`); }
      }
    });
    const nWorse = Object.values(worse).reduce((t, x) => t + x, 0);
    if (same && (flips || nWorse)) fail = true; // without a same-adapter baseline, fate edges and f32 noise cannot be judged
    const fmt = (e: Err) => `r ${e.r.toExponential(1)} phi ${e.p.toExponential(1)} delay ${e.d.toExponential(1)} sky ${e.px.toFixed(3)} px I ${e.I.toExponential(1)}`;
    lines.push(`${s.name}: ${scored} scored, flips ${flips}, worse ${nWorse} ${JSON.stringify(worse)}; max err new ${fmt(maxNew)}${O ? `; old ${fmt(maxOld)}` : ""}${examples.length ? " | " + examples.join("; ") : ""}`);
  }
  const verdict = fail ? "FAIL" : same ? "PASS" : "SKIP";
  lines.unshift(`adapter ${got.adapter}; old recorded on ${old ? old.adapter : "(none)"}`);
  return { verdict, lines };
}
