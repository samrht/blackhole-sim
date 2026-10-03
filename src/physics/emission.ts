// Phenomenological time-varying disk emission (Tier 2A). Pure functions, no DOM/GPU.
// Mirrored in WGSL: raytrace.wgsl (render) and turb-parity.wgsl (parity test).
import { omegaKepler } from "./orbits";

const TWO_PI = 2 * Math.PI;
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
// --- MRI turbulence (spec 2026-10-03) -------------------------------------------------------------
// A unit-Gaussian field on (eta = ln r, phi) whose structures have the size, lifetime, shape and statistics
// measured for magnetorotational turbulence in thin disks (Schnittman, Krolik & Hawley 2006, "SKH06").
// Lattice: hashed standard normals, smoothstep-interpolated and renormalised by sqrt(sum w^2), so every
// point is exactly N(0, 1). Generations: each lattice row i runs a clock tau = t / T_c(r_i) with
// T_c = clock * T_orb; generation k is born at tick k - 1, frozen into the Keplerian flow from birth
// (sampled at phi - Omega(r) * age), and cross-faded out over the next tick (cos/sin weights, unit
// variance). Shear acting on each generation for its age makes trailing spirals; no generation lives more
// than 2 T_c, so the winding never runs away. Twin: turbulenceFieldE in emission-shared.wgsl.
export const TURB = {
  cellEta: 0.086, // ln r cell (octave 1): variance spectrum peaks at lambda_eta = 0.26 (SKH06 dr/r = 0.3)
  cellsPhi: 49,   // cells around the ring (octave 1): peaks at lambda_phi = 25 deg (SKH06)
  clock: 0.32,    // T_c / T_orb: the flow-following correlation falls to 1/e at 0.30 T_orb (SKH06 eq. 36)
  octave2: 0.5,   // weight of the half-size octave (small-scale tail of the spectrum)
} as const;

/** u32 avalanche (twin: mixT). */
export function mixHash(n: number): number {
  n = Math.imul(n ^ (n >>> 15), 2246822519) >>> 0;
  n = Math.imul(n ^ (n >>> 13), 3266489917) >>> 0;
  return (n ^ (n >>> 16)) >>> 0;
}
/** u32 hash of a lattice node (row ix, column iy, generation, octave salt). Negative ints hash through
 *  their u32 bit pattern, as WGSL's u32(i32) does. */
export function hash4(ix: number, iy: number, gen: number, salt: number): number {
  return mixHash((Math.imul(ix >>> 0, 1973) + Math.imul(iy >>> 0, 9277) + Math.imul(gen >>> 0, 26699) + Math.imul(salt >>> 0, 59359)) >>> 0);
}
/** Box-Muller angle, shifted by pi into [-pi, pi]: cos(2 pi u2) = -cos(boxMullerAngle(u2)). WGSL guarantees
 *  cos to 2^-11 absolute only on [-pi, pi], so both twins use this range (final review, 2026-10-03). */
export function boxMullerAngle(u2: number): number { return Math.PI * (2 * u2 - 1); }
/** Standard normal from two u32 hashes (Box-Muller); u1 is offset by half a step so it is never 0. */
export function boxMuller(h1: number, h2: number): number {
  const u1 = ((h1 & 0xffffff) + 0.5) / 16777216, u2 = (h2 & 0xffffff) / 16777216;
  return -Math.sqrt(-2 * Math.log(u1)) * Math.cos(boxMullerAngle(u2));
}
export function gaussNode(ix: number, iy: number, gen: number, salt: number): number {
  const h1 = hash4(ix, iy, gen, salt);
  return boxMuller(h1, mixHash((h1 ^ 0x9e3779b9) >>> 0));
}
/** One octave: rows ix of height cellEta in ln r, cellsPhi columns around the ring. */
function turbOctave(r: number, phi: number, t: number, a: number, cellEta: number, cellsPhi: number, salt: number): number {
  const x = Math.log(r) / cellEta, i0 = Math.floor(x), fx = smooth(x - i0);
  const Om = omegaKepler(r, a, true);
  let num = 0, v = 0;
  for (let d = 0; d < 2; d++) {
    const i = i0 + d, wr = d ? fx : 1 - fx;
    const Tc = (TURB.clock * TWO_PI) / omegaKepler(Math.exp(i * cellEta), a, true);
    const tau = t / Tc + (hash4(i, 0x51ed, 0, salt) & 0xffffff) / 16777216; // per-row clock phase
    const k = Math.floor(tau), f = tau - k;
    for (let e = 0; e < 2; e++) {
      const gen = k + e, wt = e ? Math.sin((Math.PI / 2) * f) : Math.cos((Math.PI / 2) * f);
      const age = (e ? f : 1 + f) * Tc;               // generation k + e was born at tick k + e - 1
      const ph = phi - Om * age, y = (ph / TWO_PI - Math.floor(ph / TWO_PI)) * cellsPhi;
      const j0 = Math.floor(y), fy = smooth(y - j0), ja = j0 % cellsPhi, jb = (j0 + 1) % cellsPhi;
      const w0 = wr * wt * (1 - fy), w1 = wr * wt * fy;
      num += w0 * gaussNode(i, ja, gen, salt) + w1 * gaussNode(i, jb, gen, salt);
      v += w0 * w0 + w1 * w1;
    }
  }
  return num / Math.sqrt(v);
}
/** Unit-Gaussian MRI turbulence g at disk radius r, azimuth phi (any branch: 2 pi-periodic), coordinate
 *  time t (the emission time; negative is fine), spin a. */
export function turbulenceAt(r: number, phi: number, t: number, a: number): number {
  const w = TURB.octave2;
  return (turbOctave(r, phi, t, a, TURB.cellEta, TURB.cellsPhi, 1) + w * turbOctave(r, phi, t, a, TURB.cellEta / 2, 2 * TURB.cellsPhi, 2)) / Math.sqrt(1 + w * w);
}
/** Lognormal brightness factor exp(sigma g - sigma^2 / 2): mean exactly 1 over a unit Gaussian g, so the
 *  turbulence redistributes the Novikov-Thorne light without changing its average (Hogg & Reynolds 2016). */
export function lognormalFactor(g: number, sigma: number): number {
  return Math.exp(sigma * g - 0.5 * sigma * sigma);
}

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

/** Intrinsic flicker of the disk's integrated light (fractional rms, face-on, before beaming) against the
 *  lognormal sigma, measured by scripts/calibrate-turbulence.ts at a = 0.9, r_in = ISCO .. 40 M, 1500
 *  snapshots. The 2 % default sits inside the observed thermal state (whole source < 7.5 % rms at 0.1-10 Hz,
 *  McClintock & Remillard 2006; disk component near-constant, Churazov et al. 2001). SKH06's "<~ 2 %" is a
 *  variance: their simulated thermal disk shows 14-16 % rms. */
export const FLICKER_TABLE: readonly (readonly [number, number])[] = [[0, 0], [0.05, 0.00173], [0.25, 0.00872], [0.5, 0.01787], [0.75, 0.02790], [1, 0.03945], [1.25, 0.05352], [1.5, 0.07198], [1.75, 0.09828], [2, 0.13887]];
export const FLICKER_DEFAULT = 0.02;
/** sigma giving intrinsic flicker `rms` (linear between table points; clamps to the table's top). */
export function sigmaForFlicker(rms: number): number {
  if (rms <= 0) return 0;
  for (let i = 1; i < FLICKER_TABLE.length; i++) {
    const [s0, r0] = FLICKER_TABLE[i - 1], [s1, r1] = FLICKER_TABLE[i];
    if (rms <= r1) return s0 + ((rms - r0) / (r1 - r0)) * (s1 - s0);
  }
  return FLICKER_TABLE[FLICKER_TABLE.length - 1][0];
}

/** How the disk is shaded at (r, phi, t) (final review, 2026-10-03). MRI turbulence changes the local
 *  dissipation, i.e. the emitted flux F = lognormal (mean 1, so the bolometric light is conserved); an
 *  optically thick disk radiates that as a blackbody at T x F^(1/4), which the renderer shades through its
 *  colour table (tempScale). The illustrative breathing and hot spots stay a grey brightness factor. */
export function diskShadeFactors(
  r: number, phi: number, t: number, a: number,
  sigma: number, breatheAmp: number, spots: HotSpot[],
): { tempScale: number; grey: number } {
  const tempScale = sigma > 0 ? Math.pow(lognormalFactor(turbulenceAt(r, phi, t, a), sigma), 0.25) : 1;
  const breathe = 1 + breatheAmp * Math.sin(TWO_PI * t / T_BREATHE);
  return { tempScale, grey: Math.max(0, breathe + hotspotField(r, patternPhase(r, phi, t, 1, a), spots)) };
}
