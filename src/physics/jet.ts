// Phenomenological relativistic jet (Tier 2B). Pure functions, no DOM/GPU.
// Mirrored in WGSL: raytrace.wgsl (render) and jet-parity.wgsl (parity test).
// Reuses the Tier 2A value-noise basis (emission.vnoise) so there is one shared noise impl.
import { vnoise } from "./emission";

/** Shared design constants.
 *
 * The shape and beaming constants below (rho0 .. knotSeed) are hardcoded identically in both WGSL
 * twins and are covered by the ?parity route, so a desync there fails a gate.
 *
 * `gain` and `ceil` are NOT: they scale accumulated radiance at the integration site in
 * raytrace.wgsl, downstream of the emission function jet-parity.wgsl exercises. Nothing in TS reads
 * them — they are documentation of what the shader does, and they had drifted (0.06/8.0 recorded
 * against the shader's actual 0.03/4.0). Keep them in step with raytrace.wgsl:179 by hand.
 */
export const JET = {
  rho0: 0.6, slope: 0.7,      // funnel throat radius (M) and parabolic flare (M^1/2)
  qPeak: 0.8, wWall: 0.22,    // limb-brightening: wall peak position and width (in q units)
  zBase: 2.0,                 // launch height above the pole (M); below this = no jet
  kz: 0.35,                   // knot spatial frequency (1/M); knots move with the flow at beta(Gamma)
  pBeam: 3.5,                 // beaming exponent (3 + spectral index)
  turbAmpJet: 0.35,           // small cross-funnel churn
  knotSeed: 17.0,             // fixed 2nd-axis coordinate for the 1-D knot noise
  gain: 0.03,                 // per-dl emissivity -> radiance scale   (raytrace.wgsl JET_GAIN)
  ceil: 4.0,                  // clamp on accumulated jet radiance     (raytrace.wgsl JET_CEIL)
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

/** Gamma(z) = max(1, G280 (|z| / 280 r_g)^0.58): M87's measured acceleration (Mertens et al. 2016,
 *  Park et al. 2019). WGSL twin: jetGammaAt in emission-shared.wgsl. */
export const GAMMA_REF_Z = 280, GAMMA_SLOPE = 0.58;
export const gammaProfile = (z: number, g280: number) => Math.max(1, g280 * Math.pow(Math.abs(z) / GAMMA_REF_Z, GAMMA_SLOPE));

/** Traveling-wave knots: blobs of brightness marching outward as t advances. */
/** Knots are blobs carried by the jet plasma, so the pattern moves outward at the flow speed
 *  beta = sqrt(1 - 1/Gamma^2) < c (t is coordinate time, M; c = 1). It used to move at 6/0.35 ~ 17c,
 *  which under light-travel delay averaged the knots out along each line of sight (2026-10-01 review);
 *  at beta < c an approaching jet now shows apparent superluminal motion, as observed in M87. */
export function knots(z: number, t: number, gamma: number, jetKnots: number): number {
  const beta = Math.sqrt(Math.max(0, 1 - 1 / (gamma * gamma)));
  const phase = JET.kz * (Math.abs(z) - beta * t);
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
  r: number, th: number, t: number, gamma: number,
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
  return Math.max(0, w * lengthFalloff(z, jetLength) * knots(z, t, gamma, jetKnots) * turb);
}

/** True where jetEmission can be non-zero for SOME jetStrength and time: zBase <= |z| <= jetLength
 *  and inside the funnel wall (q <= 1.2, wallProfile's cut). Purely geometric, so the geodesic
 *  cache's jet bookmark never depends on jetStrength (spec 2026-10-01 3.3).
 *  WGSL twin: inJetEnvelope in raytrace.wgsl. */
export function inJetEnvelope(r: number, th: number, jetLength: number): boolean {
  const z = r * Math.cos(th), az = Math.abs(z);
  if (az < JET.zBase || az > jetLength) return false;
  return (r * Math.sin(th)) / funnelEdge(z) <= 1.2;
}
