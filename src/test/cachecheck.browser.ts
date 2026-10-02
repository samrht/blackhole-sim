import { Renderer } from "../render/gpu";
import type { UniformValues } from "../render/uniforms";
import { BUILD_SLICES, BM_NONE, BuildScheduler } from "../render/cache-plan";
import { SCENES, prepareScene, sceneUniforms, type Scene } from "./scenes";

/** Exactness gate of the geodesic cache (spec 2026-10-01 3.6): for every jitter set and two times,
 *  a cached frame must equal a live-traced frame with the same fixed jitter. Raw accum floats,
 *  blend 1, scale 1, 480x270. */
type Check = Scene & { buildJet?: number; sky?: boolean; capacity?: number; starfield?: boolean };
// Scenes run with the panorama at skyStrength 1, the app's default. The procedural starfield is not
// reproducible across separately compiled entry points: its hash takes fract() of values in the
// hundreds and scales star brightness by ~90x, so last-bit FMA-contraction differences between
// `main` and `shade` reach ~1e-3 on dim star pixels even though the stored escape direction is
// bit-identical (measured 2026-10-01, plan Task 6 ruling). The `starfield` scene therefore gates every
// pixel class strictly EXCEPT rays that end on the starfield, whose difference is reported.
const withSky = (s: Scene, name = s.name): Check => ({ ...s, name, skyStrength: 1, sky: true });
const CHECKS: Check[] = [
  withSky(SCENES[0], "default-sky"),
  withSky(SCENES[2]),
  withSky(SCENES[3]),
  { ...withSky(SCENES[0], "jet-toggle"), buildJet: 0 },   // built with the jet off, shaded with it on
  { ...withSky(SCENES[0], "overflow"), capacity: 64 },    // bookmark buffer forced tiny -> LIVE fallback
  { ...withSky(SCENES[0], "delay"), lightDelay: 1 },       // light-travel delay on (jet, spots, turbulence)
  { ...SCENES[2], name: "starfield", skyStrength: 0, starfield: true },
  // Cygnus X-1's jet: optically thick over the disk, so the cached frame must attenuate the cached disk exactly
  withSky({ name: "xrb-thick", a: 0.998, inclDeg: 27, time: 7, frame: 3, jetStrength: 1, skyStrength: 0, obj: { massSun: 21.2, lambda: 0.02 } }),
];
const TIMES = [0, 137.5];
const rel = (a: number, b: number) => Math.abs(a - b) / Math.max(Math.abs(a), 1e-3);

async function loadSky(r: Renderer) {
  const bmp = await createImageBitmap(await (await fetch("/sky/milkyway-4k.jpg")).blob());
  r.uploadSky(bmp); r.rebind();
}

export async function runCacheCheck(canvas: HTMLCanvasElement) {
  canvas.style.width = "480px"; canvas.style.height = "270px";
  const lines: string[] = []; let ok = true;
  for (const c of CHECKS) {
    const r = new Renderer();
    r.bookmarkCapacityOverride = c.capacity ?? null;
    await r.init(canvas);
    if (r.cacheSets === 0) { lines.push(`${c.name}: cache unavailable on this adapter`); ok = false; continue; }
    const rIn = prepareScene(r, c);
    if (c.sky) await loadSky(r);
    const base = sceneUniforms(r, c, rIn);
    // Build every set, slice by slice, exactly as the app does.
    r.resetCache();
    const sched = new BuildScheduler(r.cacheSets, r.displayH, BUILD_SLICES);
    for (let x = sched.next(); x; x = sched.next()) {
      r.frame(base, { build: { ...base, jetStrength: c.buildJet ?? c.jetStrength, resW: r.displayW, resH: r.displayH,
        setIndex: x.set, rowStart: x.rowStart, rowEnd: x.rowEnd } });
    }
    await r.device.queue.onSubmittedWorkDone();
    const bm = await r.readbackBookmarks();
    let worstNon = 0, worstLive = 0, nLive = 0, worstStar = 0; const jetDiffs: number[] = [];
    for (let k = 0; k < r.cacheSets; k++) {
      const entries = await r.readbackEntries(k);
      for (const time of TIMES) {
        const u: UniformValues = { ...base, time, setIndex: k, jitterMode: 1, blend: 1, reset: 1 };
        // Each image is rendered twice and the second read: blend = 1 computes mix(a, b, 1) =
        // a + (b - a), which is not bitwise b, so the first render still carries a rounding trace of
        // whatever accum held before (up to ~1 ulp of that old value -- 1e-3 relative on a dark sky
        // pixel that used to be a bright disk sample). With a == b the trace vanishes exactly.
        const render = async (cached: boolean) => {
          for (let rep = 0; rep < 2; rep++) { r.frame(u, cached ? { cachedSet: k } : {}); await r.device.queue.onSubmittedWorkDone(); }
          return r.readbackAccum();
        };
        const live = await render(false);
        const cached = await render(true);
        for (let p = 0; p < r.width * r.height; p++) {
          const d = Math.max(rel(live[4 * p], cached[4 * p]), rel(live[4 * p + 1], cached[4 * p + 1]), rel(live[4 * p + 2], cached[4 * p + 2]));
          const word = entries[4 * p], kind = word & 3, bi = word >>> 2;
          if (c.starfield && kind === 2) { worstStar = Math.max(worstStar, d); continue; }
          if (kind === 3) { worstLive = Math.max(worstLive, d); nLive++; }
          else if (u.jetStrength > 0 && bi !== BM_NONE) jetDiffs.push(d);
          else worstNon = Math.max(worstNon, d);
        }
      }
    }
    jetDiffs.sort((x, y) => x - y);
    const p9999 = jetDiffs.length ? jetDiffs[Math.floor(0.9999 * (jetDiffs.length - 1))] : 0;
    const maxJet = jetDiffs.length ? jetDiffs[jetDiffs.length - 1] : 0;
    const pass = worstNon <= 1e-5 && worstLive <= 1e-5 && p9999 <= 1e-3 && (c.capacity === undefined || nLive > 0);
    ok &&= pass;
    lines.push(`${pass ? "ok  " : "BAD "} ${c.name}: sets ${r.cacheSets}, bookmarks ${bm.count} (mean nJet ${bm.meanNJet.toFixed(1)}), ` +
      `non-jet max ${worstNon.toExponential(2)}, jet p99.99 ${p9999.toExponential(2)} max ${maxJet.toExponential(2)} (n ${jetDiffs.length}), ` +
      `live-fallback ${nLive} max ${worstLive.toExponential(2)}` +
      (c.starfield ? `, starfield sky (reported, not gated) max ${worstStar.toExponential(2)}` : ""));
    r.device.destroy();
  }
  return { ok, lines };
}
