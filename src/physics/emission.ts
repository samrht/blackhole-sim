// Phenomenological time-varying disk emission (Tier 2A). Pure functions, no DOM/GPU.
// Mirrored in WGSL: raytrace.wgsl (render) and turb-parity.wgsl (parity test).
import { omegaKepler } from "./orbits";

export const T_BREATHE = 2000; // coordinate-time period (in M) of the optional slow "breathing"

export interface HotSpot { r: number; psi: number; sigma: number; amp: number; }

/** Co-rotating pattern phase. Matter at (r, phi) orbits at Omega(r), so a feature fixed in the
 *  co-rotating frame appears at psi = phi - Omega(r) * t * timeScale in the static observer frame. */
export function patternPhase(rHit: number, phiHit: number, t: number, timeScale: number, a: number): number {
  return phiHit - omegaKepler(rHit, a, true) * t * timeScale;
}

/** 32-bit integer cell hash -> [0,1]. Bit-identical to the WGSL twin: Math.imul / >>> 0 reproduce
 *  u32 multiply + logical shift exactly, so only the interpolation arithmetic differs f32 vs f64. */
function ihash(ix: number, iy: number): number {
  let n = (Math.imul(ix >>> 0, 1973) + Math.imul(iy >>> 0, 9277)) >>> 0;
  n = Math.imul(n ^ (n >>> 15), 2246822519) >>> 0;
  n = Math.imul(n ^ (n >>> 13), 3266489917) >>> 0;
  return (n & 0xffffff) / 0xffffff;
}
function smooth(t: number): number { return t * t * (3 - 2 * t); }
export function vnoise(x: number, y: number): number {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = smooth(x - ix), fy = smooth(y - iy);
  const a00 = ihash(ix, iy), a10 = ihash(ix + 1, iy);
  const a01 = ihash(ix, iy + 1), a11 = ihash(ix + 1, iy + 1);
  return (a00 * (1 - fx) + a10 * fx) * (1 - fy) + (a01 * (1 - fx) + a11 * fx) * fy;
}
/** Multi-octave value noise in [0,~1); domain (logR, psi) so features shear with radius and phase. */
export function turbulence(logR: number, psi: number, octaves: number): number {
  let sum = 0, amp = 0.5, freq = 1;
  for (let o = 0; o < octaves; o++) { sum += amp * vnoise(logR * freq, psi * freq); amp *= 0.5; freq *= 2; }
  return sum;
}

const TWO_PI = 2 * Math.PI;
/** Sum of orbiting Gaussian hot-spots, each fixed in the co-rotating (r, psi) frame. */
export function hotspotField(rHit: number, psi: number, spots: HotSpot[]): number {
  let s = 0;
  for (const sp of spots) {
    const dr = rHit - sp.r;
    let dpsi = psi - sp.psi;
    dpsi -= TWO_PI * Math.round(dpsi / TWO_PI); // shortest angular separation
    const arc = sp.r * dpsi;                    // arc length along the ring
    s += sp.amp * Math.exp(-(dr * dr + arc * arc) / (2 * sp.sigma * sp.sigma));
  }
  return s;
}

/** Dimensionless emission multiplier. Exactly 1 when turbAmp=0, breatheAmp=0, no spots -> Tier-1. */
export function emissionField(
  rHit: number, psi: number, t: number,
  turbAmp: number, breatheAmp: number, spots: HotSpot[],
): number {
  const turb = 1 + turbAmp * (turbulence(Math.log(rHit), psi, 3) - 0.5) * 2;
  const breathe = 1 + breatheAmp * Math.sin(TWO_PI * t / T_BREATHE);
  return Math.max(0, turb * breathe + hotspotField(rHit, psi, spots));
}
