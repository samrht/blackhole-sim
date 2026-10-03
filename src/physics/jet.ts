// Jet geometry and density modulation (Tier 2B); the synchrotron physics is in synchrotron.ts.
// Pure functions, no DOM/GPU. WGSL twins: emission-shared.wgsl (sole copy, prepended by the renderer
// and the ?parity route).
// The filaments use the disk turbulence's integer hash (emission.hash4), so CPU and GPU share one hash.
import { hash4 } from "./emission";
import { fluxRatio } from "./flux-history";
import { horizonOuter } from "./kerr";

/** Shared design constants.
 *
 * The shape constants below are hardcoded identically in emission-shared.wgsl and are covered by the
 * ?parity route, so a desync there fails a gate.
 */
export const JET = {
  rho0: 0.6, slope: 0.7,      // funnel throat radius (M) and parabolic flare (M^1/2)
  qPeak: 0.8, wWall: 0.22,    // limb-brightening: wall peak position and width (in q units)
  zBase: 2.0,                 // launch height above the pole (M); below this = no jet
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

/** Flow Lorentz factor from M87's measured linear acceleration, Gamma ~ R ~ z^0.58 (Mertens et al. 2016):
 *  the power law is applied to the proper speed, Gamma beta = sqrt(G280^2 - 1) (|z| / 280 r_g)^0.58, as in
 *  force-free linear acceleration (Gamma beta ~ R Omega_F / c), so the flow is Gamma = G280 at 280 r_g,
 *  ~ z^0.58 where Gamma >> 1, and mildly relativistic, never static, near the base (2026-10-02 review:
 *  max(1, G280 (|z|/280)^0.58) held the plasma at rest below ~85 r_g). WGSL twin: jetGammaAt. */
export const GAMMA_REF_Z = 280, GAMMA_SLOPE = 0.58;
export const gammaProfile = (z: number, g280: number) => {
  const ub = Math.sqrt(Math.max(0, g280 * g280 - 1)) * Math.pow(Math.abs(z) / GAMMA_REF_Z, GAMMA_SLOPE);
  return Math.sqrt(1 + ub * ub);
};

/** Gauss-Legendre 6-point nodes and weights on [-1, 1] (twin: GL6 constants in emission-shared.wgsl). */
export const GL6: readonly [number, number][] = [
  [-0.9324695142031521, 0.1713244923791704], [-0.6612093864662645, 0.3607615730481386],
  [-0.2386191860831969, 0.4679139345726910], [0.2386191860831969, 0.4679139345726910],
  [0.6612093864662645, 0.3607615730481386], [0.9324695142031521, 0.1713244923791704],
];
/** Coordinate time (M) for plasma to climb from the jet base z_base to |z| at the flow law Gamma beta = A z^p
 *  (spec 2026-10-03 jet flux knots, corrections 1): with v = z^(1-p), tau = Int sqrt(1 + A^2 v^(2p/(1-p))) dv
 *  / (A (1-p)) -- smooth in v, so 6-point Gauss is accurate to 3e-5. Twin: launchDelayJ. */
export function launchDelay(z: number, g280: number): number {
  const az = Math.abs(z);
  if (az <= JET.zBase) return 0;
  const p = GAMMA_SLOPE, A = Math.sqrt(Math.max(1e-12, g280 * g280 - 1)) / Math.pow(GAMMA_REF_Z, p), e = (2 * p) / (1 - p);
  const v0 = Math.pow(JET.zBase, 1 - p), v1 = Math.pow(az, 1 - p), h = 0.5 * (v1 - v0), m = 0.5 * (v1 + v0);
  let s = 0;
  for (const [x, w] of GL6) s += w * Math.sqrt(1 + A * A * Math.pow(m + h * x, e));
  return (s * h) / (A * (1 - p));
}
/** Blandford-Znajek field-line angular velocity Omega_F = Omega_H / 2 = a / (4 r+) (twin: fieldLineOmegaJ). */
export function fieldLineOmega(a: number): number { return a / (4 * horizonOuter(a)); }
/** Azimuth a plasma parcel had at launch: the parcel turns at Omega = Omega_F (1 - beta) (v_phi = Omega_F rho +
 *  v_p B_phi / B_p, B_phi / B_p = -Omega_F rho / c), so Int Omega dt = Omega_F (tau - (|z| - z_base)). */
export function comovingAzimuth(ph: number, z: number, a: number, g280: number): number {
  const az = Math.abs(z);
  if (az <= JET.zBase) return ph;
  return ph - fieldLineOmega(a) * (launchDelay(z, g280) - (az - JET.zBase));
}

/** Filaments frozen into the moving plasma (spec 2.5): value noise on (launch time, co-moving azimuth, field-line
 *  label). Strength is illustrative (no measurement fixes it); the motion is physical. Twin: filamentsJ. */
export const FILAMENT = { amp: 0.35, cellT: 25, cellsPhi: 8, cellsQ: 2.5, salt: 0x46494c } as const;
/** Envelope bound on q = rho / rho_f: 1.2 (the wall's cut) times the widest width factor the slider allows,
 *  sqrt((1 + sMax eps clip) / (1 - sMax <d>)) = sqrt(1.4798) = 1.2165, i.e. 1.460 (flux statistics); 1.51 is kept
 *  from the first calibration so the cache geometry is unchanged. Twin: JET_ENV_Q in emission-shared.wgsl. */
export const JET_ENV_Q = 1.51;

const smooth = (t: number) => t * t * (3 - 2 * t);
const node3 = (ix: number, iy: number, iw: number, salt: number) => (hash4(ix, iy, iw, salt) & 0xffffff) / 16777216;
/** Smooth value noise in [0, 1] on (x, y, w), periodic in y with period n (twin: vnoise3J). */
export function vnoise3(x: number, y: number, n: number, w: number, salt: number): number {
  const ix = Math.floor(x), iy = Math.floor(y), iw = Math.floor(w);
  const fx = smooth(x - ix), fy = smooth(y - iy), fw = smooth(w - iw);
  const y0 = ((iy % n) + n) % n, y1 = (y0 + 1) % n;
  let v = 0;
  for (let c = 0; c < 8; c++) {
    const dx = c & 1, dy = (c >> 1) & 1, dw = (c >> 2) & 1;
    const wt = (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy) * (dw ? fw : 1 - fw);
    v += wt * node3(ix + dx, dy ? y1 : y0, iw + dw, salt);
  }
  return v;
}
export function filaments(q: number, phiC: number, tLaunch: number): number {
  const turns = phiC / (2 * Math.PI), y = (turns - Math.floor(turns)) * FILAMENT.cellsPhi;
  return 1 + FILAMENT.amp * (vnoise3(tLaunch / FILAMENT.cellT, y, FILAMENT.cellsPhi, q * FILAMENT.cellsQ, FILAMENT.salt) - 0.5) * 2;
}

/** Density modulation of the synchrotron jet (spec 2026-10-03 jet flux knots 2.4): the plasma at height z left
 *  the base at t - tau(z), when the horizon flux ratio was f; the funnel there is sqrt(f) wider and f denser
 *  (power per length ~ f^2 ~ phi^2), times the co-moving filaments. 0 outside the emitting region.
 *  t is absolute coordinate time (M); a only sets the filaments' twist. Twin: jetShapeJ. */
export function jetShape(r: number, th: number, ph: number, t: number, jetLength: number, fluxVar: number, g280: number, a: number): number {
  const z = r * Math.cos(th), az = Math.abs(z);
  if (az < JET.zBase || az > jetLength) return 0;
  const rho = r * Math.sin(th);
  if (rho > JET_ENV_Q * funnelEdge(z)) return 0; // exact: beyond the widest wall for any f (twin's early-out)
  const tl = t - launchDelay(z, g280), f = fluxRatio(tl, fluxVar, a), sw = Math.sqrt(f);
  const w = wallProfile(rho / sw, z);
  if (w <= 0) return 0;
  const q = rho / (sw * funnelEdge(z));
  return Math.max(0, f * w * lengthFalloff(z, jetLength) * filaments(q, comovingAzimuth(ph, z, a, g280), tl));
}

/** True where jetShape can be non-zero for SOME time and slider value: zBase <= |z| <= jetLength and
 *  q <= JET_ENV_Q. Purely geometric, so the geodesic cache's jet bookmark never depends on the jet switch,
 *  brightness or flux (spec 2026-10-01 3.3). WGSL twin: inJetEnvelope in raytrace.wgsl. */
export function inJetEnvelope(r: number, th: number, jetLength: number): boolean {
  const z = r * Math.cos(th), az = Math.abs(z);
  if (az < JET.zBase || az > jetLength) return false;
  return (r * Math.sin(th)) / funnelEdge(z) <= JET_ENV_Q;
}
