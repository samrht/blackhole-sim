import type { Renderer } from "../render/gpu";
import type { UniformValues } from "../render/uniforms";
import { buildTempLUT, buildVisibleLUT, lumNormFor } from "../physics/lookups";
import { iscoRadius } from "../physics/orbits";
import { jetUniforms, ETA_DEFAULT } from "../physics/synchrotron";
import { CUSTOM_DEFAULT } from "../physics/presets";
import { sigmaForFlicker, FLICKER_DEFAULT } from "../physics/emission";
import { tubeOf, tubeOrbit, flareZeta } from "../physics/eruption-spots";

/** Fixed scenes shared by the ?golden and ?cachecheck validation routes. */
export interface Scene { name: string; a: number; inclDeg: number; time: number; frame: number; jetStrength: number; skyStrength: number; lightDelay?: number;
  /** The jet's object (mass, accretion) when it is not the default view's. */
  obj?: { massSun: number; lambda: number }; }
export const SCENES: Scene[] = [
  { name: "default", a: 0.9, inclDeg: 72, time: 37, frame: 5, jetStrength: 1, skyStrength: 0 },
  { name: "jet-off", a: 0.5, inclDeg: 30, time: 0, frame: 0, jetStrength: 0, skyStrength: 0 },
  { name: "edge-on", a: 0.99, inclDeg: 85, time: 100, frame: 2, jetStrength: 1, skyStrength: 0 },
  { name: "face-on-jet", a: 0.9, inclDeg: 8, time: 12, frame: 1, jetStrength: 1, skyStrength: 0 },
  // The app's default path: light-travel delay on (disk pattern, breathing, jet knots per sample), at a time when
  // eruption 1's flux tube is a quarter orbit into its circular phase (eruption flares, spec 2026-10-04).
  { name: "delay", a: 0.9, inclDeg: 72, time: tubeOf(1, 0.9).t0 + tubeOf(1, 0.9).D + 0.25 * tubeOrbit(tubeOf(1, 0.9), 0.9), frame: 5, jetStrength: 1, skyStrength: 0, lightDelay: 1 },
];

/** Upload the scene's LUTs; returns rIn (the ISCO). */
export function prepareScene(r: Renderer, s: Scene): number {
  const rIn = iscoRadius(s.a, true);
  r.uploadLUTs(buildTempLUT(s.a, true, rIn, 40, 512), buildVisibleLUT());
  r.rebind();
  return rIn;
}

export function sceneUniforms(r: Renderer, s: Scene, rIn: number, extra: Partial<UniformValues> = {}): UniformValues {
  // The default view's object (1e8 M_sun, 3e4 K peak) at the scene's spin; epsilon at its default.
  const obj = s.obj ?? CUSTOM_DEFAULT, J = jetUniforms(obj.massSun, s.a, obj.lambda, ETA_DEFAULT, 60, 2, 1);
  return {
    resW: r.width, resH: r.height, outW: r.displayW, outH: r.displayH, a: s.a, incl: (s.inclDeg * Math.PI) / 180,
    rObs: 1000, fovScale: 14, rIn, rOut: 40, Tpeak: 3e4, lumNorm: lumNormFor(3e4), lightDelay: s.lightDelay ?? 0, exposure: 1.6, time: s.time, frame: s.frame, reset: 1,
    maxSteps: 4800, blend: 1, timeScale: 1, turbAmp: sigmaForFlicker(FLICKER_DEFAULT), breatheAmp: 0.2, flareStrength: 1, flarePhi: J.phi, flareZeta: flareZeta(),
    jetStrength: s.jetStrength, jetGamma: 2, jetLength: 60, fluxVar: 1, skyStrength: s.skyStrength,
    jetB0: J.jetB0, jetQ0: J.jetQ0, rgCm: J.rgCm,
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
