// Layout MUST match the `Uniforms` struct in raytrace.wgsl (4-byte scalars, vec2 first).
// floats: resW,resH,a,incl,rObs,fovScale,rIn,rOut,Tpeak,exposure,time (11)
//         + blend,timeScale,turbAmp,breatheAmp (4)                     -> 15 floats
// uint:   frame,reset,maxSteps (3) + nSpots (1)                        -> 4 uints
export interface UniformValues {
  resW: number; resH: number; a: number; incl: number; rObs: number; fovScale: number;
  rIn: number; rOut: number; Tpeak: number; exposure: number; time: number;
  frame: number; reset: number; maxSteps: number;
  blend: number; timeScale: number; turbAmp: number; breatheAmp: number; nSpots: number;
}
export const UNIFORM_FLOATS = 15, UNIFORM_UINTS = 4;
export const UNIFORM_SIZE = Math.ceil((UNIFORM_FLOATS + UNIFORM_UINTS) / 4) * 16; // -> 80 bytes

export function packUniforms(u: UniformValues): ArrayBuffer {
  const buf = new ArrayBuffer(UNIFORM_SIZE);
  const f = new Float32Array(buf), i = new Uint32Array(buf);
  f[0] = u.resW; f[1] = u.resH; f[2] = u.a; f[3] = u.incl;
  f[4] = u.rObs; f[5] = u.fovScale; f[6] = u.rIn; f[7] = u.rOut;
  f[8] = u.Tpeak; f[9] = u.exposure; f[10] = u.time;
  i[11] = u.frame; i[12] = u.reset; i[13] = u.maxSteps;
  f[14] = u.blend; f[15] = u.timeScale; f[16] = u.turbAmp; f[17] = u.breatheAmp;
  i[18] = u.nSpots;
  return buf;
}
