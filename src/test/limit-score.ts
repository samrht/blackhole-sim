// Scoring of the ?limit experiment (what limits accuracy: f32 rounding or integrator order?). Every arm's rays are compared with
// the converged reference as an ON-SCREEN displacement, through the reference's own Jacobian at each ray.
import { sinCosP } from "../render/sincos";
import { summarize, type Summary } from "./drift-analysis";

export type Fate = "disk" | "captured" | "escaped" | "budget";
export interface Obs { f: Fate; r?: number; p?: number; t?: number; dir?: number[]; steps: number; attempts: number }

/** What a ray reports, as a vector: a disk hit's Cartesian point (no phi wrap), a sky ray's direction; else nothing. */
export function obsVector(o: Obs): number[] | null {
  if (o.f === "disk") return [o.r! * Math.cos(o.p!), o.r! * Math.sin(o.p!)];
  if (o.f === "escaped") return o.dir!;
  return null;
}

/** d(observable) per pixel in alpha and in beta, central differences over +-delta pixels; undefined if a neighbour has none. */
export function jacobianOf(ap: number[] | null, am: number[] | null, bp: number[] | null, bm: number[] | null, delta: number): number[][] | undefined {
  if (!ap || !am || !bp || !bm) return undefined;
  return [ap.map((v, k) => (v - am[k]) / (2 * delta)), bp.map((v, k) => (v - bm[k]) / (2 * delta))];
}

export interface RayErr { flip: boolean; px: number; delay: number }

/** One ray against the reference: a fate flip, or the on-screen error (least squares through J) and the disk delay error in M. */
/** The on-screen shift (px in alpha, px in beta) that the reference's Jacobian J assigns to the observable error e. */
export function pxShift(J: number[][], e: number[]): [number, number] {
  const dot = (x: number[], y: number[]) => x.reduce((s, c, k) => s + c * y[k], 0);
  const a11 = dot(J[0], J[0]), a12 = dot(J[0], J[1]), a22 = dot(J[1], J[1]), b1 = dot(J[0], e), b2 = dot(J[1], e), det = a11 * a22 - a12 * a12;
  return [(a22 * b1 - a12 * b2) / det, (a11 * b2 - a12 * b1) / det];
}

export function rayError(ref: Obs, J: number[][] | undefined, got: Obs): RayErr {
  if (ref.f !== got.f) return { flip: true, px: NaN, delay: NaN };
  const u = obsVector(ref), v = obsVector(got);
  const delay = ref.f === "disk" ? Math.abs(got.t! - ref.t!) : NaN;
  if (!u || !v || !J) return { flip: false, px: NaN, delay };
  return { flip: false, px: Math.hypot(...pxShift(J, v.map((c, k) => c - u[k]))), delay };
}

export const CURVE_BINS = [0, 1e-3, 1e-2, 0.1, 1, Infinity];
export interface ArmRow extends RayErr { steps: number; attempts: number }
export interface ArmSummary { rays: number; flips: number; meanSteps: number; meanAttempts: number; px: Summary; delay: Summary; byCurve: Record<string, { rays: number; flips: number; px: Summary }>; groups?: Record<string, { rays: number; flips: number; px: Summary }> }

/** dist: each ray's |distance| from the critical curve (M). */
export function summarizeArm(dist: ArrayLike<number>, rows: ArmRow[], groups?: Record<string, ArrayLike<boolean>>): ArmSummary {
  const byCurve: ArmSummary["byCurve"] = {};
  const scored = (r: ArmRow[]) => summarize(r.filter((x) => !Number.isNaN(x.px)).map((x) => x.px));
  for (let k = 0; k + 1 < CURVE_BINS.length; k++) {
    const g = rows.filter((_, i) => dist[i] >= CURVE_BINS[k] && dist[i] < CURVE_BINS[k + 1]);
    byCurve[`[${CURVE_BINS[k]}, ${CURVE_BINS[k + 1]})`] = { rays: g.length, flips: g.filter((x) => x.flip).length, px: scored(g) };
  }
  let named: ArmSummary["groups"];
  if (groups) { named = {}; for (const k of Object.keys(groups)) { const g = rows.filter((_, i) => groups[k][i]); named[k] = { rays: g.length, flips: g.filter((x) => x.flip).length, px: scored(g) }; } }
  const mean = (f: (x: ArmRow) => number) => rows.reduce((s, x) => s + f(x), 0) / rows.length;
  return { rays: rows.length, flips: rows.filter((x) => x.flip).length, meanSteps: mean((x) => x.steps), meanAttempts: mean((x) => x.attempts), px: scored(rows), delay: summarize(rows.filter((x) => !Number.isNaN(x.delay)).map((x) => x.delay)), byCurve, groups: named };
}

/** The ray's constants exactly as camera-shared.wgsl's cameraXiEta computes them in f32, op by op. */
export function f32Inputs(alpha: number, beta: number, a: number, incl: number): { a: number; incl: number; xi: number; eta: number; beta: number } {
  const F = Math.fround, A = F(a), I = F(incl), al = F(alpha), be = F(beta), [si, ci] = sinCosP(I, true);
  const xi = F(-al * si), c2 = F(ci * ci), s2 = Math.max(F(si * si), F(1e-8));
  return { a: A, incl: I, xi, beta: be, eta: F(F(F(be * be) + F(F(F(xi * xi) * c2) / s2)) - F(F(A * A) * c2)) };
}

/** A result as the renderer stores it: r, phi, the delay and the sky direction each in f32. */
export function roundObs(o: Obs): Obs {
  const F = Math.fround;
  return { ...o, r: o.r === undefined ? undefined : F(o.r), p: o.p === undefined ? undefined : F(o.p), t: o.t === undefined ? undefined : -F(-o.t - 1000) - 1000, dir: o.dir?.map(F) };
}
