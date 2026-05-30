import { temperatureShape } from "./disk";
import { blackbodyLinearSRGB } from "./color";

/** Normalized T(r) sampled uniformly over [rIn, rOut]; returns Float32Array(N), peak=1. */
export function buildTempLUT(a: number, prograde: boolean, rIn: number, rOut: number, N: number): Float32Array {
  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const r = rIn + (rOut - rIn) * (i / (N - 1));
    out[i] = temperatureShape(r, a, prograde);
  }
  // Renormalize to exact peak=1 over the discrete grid (temperatureShape peaks between samples)
  const peak = Math.max(...out);
  if (peak > 0) for (let i = 0; i < N; i++) out[i] /= peak;
  return out;
}
/** color(T) over [Tmin,Tmax] as RGBA Float32 (A=1). */
export function buildColorLUT(Tmin: number, Tmax: number, N: number): Float32Array {
  const out = new Float32Array(N * 4);
  for (let i = 0; i < N; i++) {
    const T = Tmin + (Tmax - Tmin) * (i / (N - 1));
    const [r, g, b] = blackbodyLinearSRGB(T);
    out[i * 4] = r; out[i * 4 + 1] = g; out[i * 4 + 2] = b; out[i * 4 + 3] = 1;
  }
  return out;
}
