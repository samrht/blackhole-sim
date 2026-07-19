// Phenomenological relativistic jet (Tier 2B). Pure functions, no DOM/GPU.
// Mirrored in WGSL: raytrace.wgsl (render) and jet-parity.wgsl (parity test).
// Reuses the Tier 2A value-noise basis (emission.vnoise) so there is one shared noise impl.
import { vnoise } from "./emission";

/** Shared design constants. The WGSL twins hardcode these exact values. */
export const JET = {
  rho0: 0.6, slope: 0.7,      // funnel throat radius (M) and parabolic flare (M^1/2)
  qPeak: 0.8, wWall: 0.22,    // limb-brightening: wall peak position and width (in q units)
  zBase: 2.0,                 // launch height above the pole (M); below this = no jet
  kz: 0.35, vKnot: 6.0,       // knot spatial frequency and outward pattern speed (M / sim-s)
  pBeam: 3.5,                 // beaming exponent (3 + spectral index)
  turbAmpJet: 0.35,           // small cross-funnel churn
  knotSeed: 17.0,             // fixed 2nd-axis coordinate for the 1-D knot noise
  gain: 0.06,                 // per-dl emissivity -> radiance scale
  ceil: 8.0,                  // clamp on accumulated jet radiance (anti-blowout)
} as const;

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

/** Parabolic funnel wall radius at signed axial height z. */
export function funnelEdge(z: number): number {
  return JET.rho0 + JET.slope * Math.sqrt(Math.abs(z));
}

/** Limb-brightened wall profile: gaussian peaked at q = qPeak, zero beyond the wall. */
export function wallProfile(rho: number, z: number): number {
  const q = rho / funnelEdge(z);
  if (q > 1.2) return 0;
  const d = q - JET.qPeak;
  return Math.exp(-(d * d) / (2 * JET.wWall * JET.wWall));
}

/** Axial soft-gate (fade in above zBase, fade out near zMax) times 1/z-style falloff. */
export function lengthFalloff(z: number, zMax: number): number {
  const az = Math.abs(z);
  const fadeIn = smoothstep(JET.zBase, JET.zBase + 2, az);
  const fadeOut = 1 - smoothstep(zMax * 0.7, zMax, az);
  const decay = JET.zBase / Math.max(az, JET.zBase);
  return fadeIn * fadeOut * decay;
}

/** Traveling-wave knots: blobs of brightness marching outward as t advances. */
export function knots(z: number, t: number, timeScale: number, jetKnots: number): number {
  const phase = JET.kz * Math.abs(z) - JET.vKnot * t * timeScale;
  return 1 + jetKnots * (vnoise(phase, JET.knotSeed) - 0.5) * 2;
}

/** Relativistic Doppler boost of emissivity. mu = cos(angle) of emitter outflow toward observer. */
export function dopplerBoost(mu: number, gamma: number): number {
  const beta = Math.sqrt(Math.max(0, 1 - 1 / (gamma * gamma)));
  const delta = 1 / (gamma * (1 - beta * mu));
  return Math.pow(delta, JET.pBeam);
}

/** Scalar jet emissivity (no beaming). Exactly 0 when jetStrength=0, below zBase, beyond zMax,
 *  or outside the funnel wall. Beaming (dopplerBoost) is applied separately at the ray step. */
export function jetEmission(
  r: number, th: number, t: number, timeScale: number,
  jetStrength: number, jetLength: number, jetKnots: number,
): number {
  if (jetStrength === 0) return 0;
  const z = r * Math.cos(th);
  const az = Math.abs(z);
  if (az < JET.zBase || az > jetLength) return 0;
  const rho = r * Math.sin(th);
  const w = wallProfile(rho, z);
  if (w <= 0) return 0;
  const turb = 1 + JET.turbAmpJet * (vnoise(Math.log(1 + rho), JET.kz * z) - 0.5) * 2;
  return Math.max(0, w * lengthFalloff(z, jetLength) * knots(z, t, timeScale, jetKnots) * turb);
}
