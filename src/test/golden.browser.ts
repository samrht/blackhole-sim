import { Renderer } from "../render/gpu";
import { SCENES, prepareScene, sceneUniforms, fnv1a } from "./scenes";
import goldenRaw from "./golden.json?raw";
import { countNonFinite, type GoldenResult } from "./golden-judge";

export { judgeGolden, type GoldenResult } from "./golden-judge";
/** Recorded by `RECORD_GOLDEN=1 npm run verify:gpu` on the pre-refactor shader. */
export const GOLDEN: GoldenResult = JSON.parse(goldenRaw);

/** Bit-exact identity of the LIVE pass (`main`) on fixed scenes at 320x180, scale 1, blend 1.
 *  Recorded once BEFORE the geodesic-cache refactor (plan 2026-10-01 Task 2); every later task must
 *  reproduce it. Hashes are per adapter: on a different GPU the route reports SKIP, not FAIL. */
export async function runGolden(canvas: HTMLCanvasElement): Promise<GoldenResult> {
  canvas.style.width = "320px"; canvas.style.height = "180px";
  const r = new Renderer(); await r.init(canvas);
  const adapter = `${r.adapterInfo.vendor} ${r.adapterInfo.architecture} ${r.adapterInfo.description}`.trim();
  const hashes: Record<string, string> = {}, nonFinite: Record<string, number> = {};
  for (const s of SCENES) {
    const rIn = prepareScene(r, s);
    r.frame(sceneUniforms(r, s, rIn));
    await r.device.queue.onSubmittedWorkDone();
    const acc = await r.readbackAccum();
    hashes[s.name] = fnv1a(new Uint32Array(acc.buffer));
    nonFinite[s.name] = countNonFinite(acc);
  }
  return { adapter, hashes, nonFinite };
}
