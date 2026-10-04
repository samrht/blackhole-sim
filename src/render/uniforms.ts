// Layout MUST match the `Uniforms` struct in raytrace.wgsl (4-byte scalars, vec2 first).
// floats: resW,resH,a,incl,rObs,fovScale,rIn,rOut,Tpeak,exposure,time (11)
//         + blend,timeScale,turbAmp (lognormal sigma of the MRI turbulence),breatheAmp (4) -> 15
//         + jetStrength,jetGamma,jetLength,fluxVar (4)                -> 19
//         + skyStrength (1)                                             -> 20 floats
//         + outW,outH (2)                                               -> 22 floats
//         + lumNorm (1, visible-band brightness; after the uints)        -> 23 floats
//         + lightDelay (1, 0/1 light-travel delay)                      -> 24 floats
//         + jetB0, jetQ0, rgCm (3, synchrotron jet)                     -> 27 floats
//         + timeEpoch (1; `time` is then the remainder: t = timeEpoch + time) -> 28 floats
//         + band, hotFlow, flowN0 (3, 1.3 mm view and hot flow)       -> 31 floats
// uint:   frame,reset,maxSteps (3) + nSpots (1)                        -> 4 uints
//         + jitterMode,setIndex,rowStart,rowEnd (4, geodesic cache)    -> 8 uints
export interface UniformValues {
  resW: number; resH: number; a: number; incl: number; rObs: number; fovScale: number;
  rIn: number; rOut: number; Tpeak: number; exposure: number; time: number;
  frame: number; reset: number; maxSteps: number;
  blend: number; timeScale: number; turbAmp: number; breatheAmp: number; nSpots: number;
  jetStrength: number; jetGamma: number; jetLength: number; fluxVar: number;
  skyStrength: number;
  outW: number; outH: number;
  /** 1 / visible luminance of the disk's rest-frame peak temperature (lumNormFor). */
  lumNorm: number;
  /** 1 = shade at each pixel's emission time (light-travel delay), 0 = one global instant. */
  lightDelay: number;
  /** Synchrotron jet (specs 2026-10-02): funnel field scale (G), energy-budget injection scale q0 (erg s^-1 cm^-3 G^-2), r_g (cm). */
  jetB0: number; jetQ0: number; rgCm: number;
  /** Clock epoch (M, a multiple of 2048; splitTime in sim-clock.ts): absolute time = timeEpoch + time.
   *  Omitted = 0, so `time` is absolute (tests, golden scenes). */
  timeEpoch?: number;
  /** 1.3 mm view (spec 2026-10-04 hot flow): band 0 visible / 1 mm; hotFlow 1 = draw the hot flow (mm, lambda < 0.01);
   *  flowN0 = its density scale (cm^-3). Omitted = 0. */
  band?: number; hotFlow?: number; flowN0?: number;
  /** Geodesic cache (spec 2026-10-01): 0 = per-frame hash jitter, 1 = fixed jitter set `setIndex`. */
  jitterMode?: number; setIndex?: number;
  /** Build pass row slice [rowStart, rowEnd). */
  rowStart?: number; rowEnd?: number;
}
export const UNIFORM_FLOATS = 31, UNIFORM_UINTS = 8;
export const UNIFORM_SIZE = Math.ceil((UNIFORM_FLOATS + UNIFORM_UINTS) / 4) * 16; // -> 160 bytes

export function packUniforms(u: UniformValues): ArrayBuffer {
  const buf = new ArrayBuffer(UNIFORM_SIZE);
  const f = new Float32Array(buf), i = new Uint32Array(buf);
  f[0] = u.resW; f[1] = u.resH; f[2] = u.a; f[3] = u.incl;
  f[4] = u.rObs; f[5] = u.fovScale; f[6] = u.rIn; f[7] = u.rOut;
  f[8] = u.Tpeak; f[9] = u.exposure; f[10] = u.time;
  i[11] = u.frame; i[12] = u.reset; i[13] = u.maxSteps;
  f[14] = u.blend; f[15] = u.timeScale; f[16] = u.turbAmp; f[17] = u.breatheAmp;
  i[18] = u.nSpots;
  f[19] = u.jetStrength; f[20] = u.jetGamma; f[21] = u.jetLength; f[22] = u.fluxVar;
  f[23] = u.skyStrength;
  f[24] = u.outW; f[25] = u.outH;
  i[26] = u.jitterMode ?? 0; i[27] = u.setIndex ?? 0; i[28] = u.rowStart ?? 0; i[29] = u.rowEnd ?? 0;
  f[30] = u.lumNorm;
  f[31] = u.lightDelay;
  f[32] = u.jetB0; f[33] = u.jetQ0; f[34] = u.rgCm;
  f[35] = u.timeEpoch ?? 0;
  f[36] = u.band ?? 0; f[37] = u.hotFlow ?? 0; f[38] = u.flowN0 ?? 0;
  return buf;
}
