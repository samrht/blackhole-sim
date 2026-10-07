import type { Renderer } from "../render/gpu";
import type { UniformValues } from "../render/uniforms";
import { buildTempLUT, buildVisibleLUT, lumNormFor } from "../physics/lookups";
import { iscoRadius } from "../physics/orbits";
import { jetUniforms, ETA_DEFAULT } from "../physics/synchrotron";
import { CUSTOM_DEFAULT, PRESETS } from "../physics/presets";
import { isHotFlow, flowN0 } from "../physics/hot-flow";
import { sigmaForFlicker, FLICKER_DEFAULT } from "../physics/emission";
import { hotspotPeakTime } from "../physics/hotspot";

/** Fixed scenes shared by the ?golden and ?cachecheck validation routes. */
export interface Scene { name: string; a: number; inclDeg: number; time: number; frame: number; jetStrength: number; skyStrength: number; lightDelay?: number;
  /** The jet's object (mass, accretion) when it is not the default view's. */
  obj?: { massSun: number; lambda: number };
  /** 1.3 mm view (spec 2026-10-04): band, and the preset (?hotflow takes its distance and EHT targets from it). */
  band?: "vis" | "mm"; preset?: string; }
const preset = (id: string) => PRESETS.find((p) => p.id === id)!;
const mmScene = (name: string, id: string, jetStrength: number, time: number): Scene => {
  const p = preset(id);
  return { name, a: p.a, inclDeg: p.inclDeg, time, frame: 0, jetStrength, skyStrength: 0, obj: { massSun: p.massSun, lambda: p.lambda }, band: "mm", preset: id };
};
export const SCENES: Scene[] = [
  { name: "default", a: 0.9, inclDeg: 72, time: 37, frame: 5, jetStrength: 1, skyStrength: 0 },
  { name: "jet-off", a: 0.5, inclDeg: 30, time: 0, frame: 0, jetStrength: 0, skyStrength: 0 },
  { name: "edge-on", a: 0.99, inclDeg: 85, time: 100, frame: 2, jetStrength: 1, skyStrength: 0 },
  { name: "face-on-jet", a: 0.9, inclDeg: 8, time: 12, frame: 1, jetStrength: 1, skyStrength: 0 },
  // The app's default path: light-travel delay on (disk pattern, breathing, jet emission per sample).
  { name: "delay", a: 0.9, inclDeg: 72, time: 37, frame: 5, jetStrength: 1, skyStrength: 0, lightDelay: 1 },
  // 1.3 mm: the hot flows of Sgr A* and M87* (no jet at 1.3 mm, spec 2.5).
  mmScene("sgra-mm", "sgra", 0, 0),
  mmScene("m87-mm", "m87", 0, 37),
  // Sgr A* at the peak TIME of eruption 0's hotspot, light delay on (spec 2026-10-04 mm hotspots; ?hotflow's hotspot gate).
  // The light seen left a few M earlier, so the frame shows the hotspot still rising (0.21 Jy vs 0.60 with the delay off).
  { ...mmScene("sgra-mm-hotspot", "sgra", 0, hotspotPeakTime(0, preset("sgra").a)), lightDelay: 1 },
];
const SPOTS = new Float32Array([8, 0, 1.2, 1.8, 12, 2.1, 1.6, 1.2, 16, 4.3, 2.0, 0.9]);

/** Upload the scene's LUTs and hot spots; returns rIn (the ISCO). */
export function prepareScene(r: Renderer, s: Scene): number {
  const rIn = iscoRadius(s.a, true);
  r.uploadLUTs(buildTempLUT(s.a, true, rIn, 40, 512), buildVisibleLUT());
  r.uploadHotSpots(SPOTS);
  r.rebind();
  return rIn;
}

export function sceneUniforms(r: Renderer, s: Scene, rIn: number, extra: Partial<UniformValues> = {}): UniformValues {
  // The default view's object (1e8 M_sun, 3e4 K peak) at the scene's spin; epsilon at its default.
  const obj = s.obj ?? CUSTOM_DEFAULT, J = jetUniforms(obj.massSun, s.a, obj.lambda, ETA_DEFAULT, 60, 2, 1);
  return {
    resW: r.width, resH: r.height, outW: r.displayW, outH: r.displayH, a: s.a, incl: (s.inclDeg * Math.PI) / 180,
    rObs: 1000, fovScale: 14, rIn, rOut: 40, Tpeak: 3e4, lumNorm: lumNormFor(3e4), lightDelay: s.lightDelay ?? 0, exposure: 1.6, time: s.time, frame: s.frame, reset: 1,
    maxSteps: 4800, blend: 1, timeScale: 1, turbAmp: sigmaForFlicker(FLICKER_DEFAULT), breatheAmp: 0.2, nSpots: 3,
    jetStrength: s.jetStrength, jetGamma: 2, jetLength: 60, fluxVar: 1, skyStrength: s.skyStrength,
    jetB0: J.jetB0, jetQ0: J.jetQ0, rgCm: J.rgCm,
    ...(s.band === "mm" ? { band: 1, hotFlow: isHotFlow(obj.lambda) ? 1 : 0, flowN0: flowN0(obj.massSun, obj.lambda) } : {}),
    ...extra,
  };
}

/** FNV-1a over 32-bit words -> 8 hex chars. Bit-exact image identity. */
export function fnv1a(words: Uint32Array): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < words.length; i++) {
    let w = words[i];
    for (let b = 0; b < 4; b++) { h ^= w & 0xff; h = Math.imul(h, 0x01000193) >>> 0; w >>>= 8; }
  }
  return h.toString(16).padStart(8, "0");
}
