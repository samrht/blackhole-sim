// Geodesic cache planning (spec 2026-10-01): pure, testable decisions the Renderer and main loop act on.

/** Fixed rotated-grid sub-pixel offsets of the cached jitter sets. Twin: fixedJitter in raytrace.wgsl. */
export const JITTER: ReadonlyArray<readonly [number, number]> = [[-0.125, -0.375], [0.375, -0.125], [0.125, 0.375], [-0.375, 0.125]];
export const NSETS_MAX = 4;
export const BUILD_SLICES = 16;
/** Share of (pixels x sets) given a jet bookmark slot. 1.5x the largest share measured by
 *  tests/sweep-jetenvelope.test.ts (plan 2026-10-01 Task 1). Overflow falls back to LIVE pixels. */
export const BOOKMARK_FRAC = 0.21; // measured 2026-10-01: max 13.6 % (face-on) x 1.5
export const ENTRY_BYTES = 16, BOOKMARK_BYTES = 48;
export const CACHE_BUDGET_BYTES = 512 * 2 ** 20;
/** Entry word = kind (2 bits) | bookmark index << 2; this index means "no bookmark". */
export const BM_NONE = 0x3fffffff;

export interface GeometryInputs {
  a: number; incl: number; fovScale: number; rObs: number; rIn: number; rOut: number;
  maxSteps: number; jetLength: number; displayW: number; displayH: number;
}
/** Everything a ray's path depends on. Shading-only inputs are deliberately absent. */
export function geometryKey(g: GeometryInputs): string {
  return [g.a, g.incl, g.fovScale, g.rObs, g.rIn, g.rOut, g.maxSteps, g.jetLength, g.displayW, g.displayH].join("|");
}

export interface CachePlan { nSets: number; entryBytes: number; bookmarkCapacity: number; }
/** Sets 4 -> 2 -> 1 within the binding limit and the total budget; nSets 0 = stay live. */
export function planCache(w: number, h: number, maxBinding: number, budget = CACHE_BUDGET_BYTES): CachePlan {
  const entryBytes = w * h * ENTRY_BYTES;
  if (entryBytes > maxBinding) return { nSets: 0, entryBytes, bookmarkCapacity: 0 };
  for (const nSets of [NSETS_MAX, 2, 1]) {
    const cap = Math.max(1, Math.min(Math.ceil(BOOKMARK_FRAC * w * h * nSets), Math.floor(maxBinding / BOOKMARK_BYTES), BM_NONE - 1));
    if (nSets * entryBytes + cap * BOOKMARK_BYTES <= budget) return { nSets, entryBytes, bookmarkCapacity: cap };
  }
  return { nSets: 0, entryBytes, bookmarkCapacity: 0 };
}

/** Hands out row slices set by set. A set counts as complete on the call that returns its last
 *  slice: that slice's build dispatch precedes the shade dispatch in the same compute pass, whose
 *  storage writes are visible to it, so the set can be shaded that same frame. */
export class BuildScheduler {
  completedSets = 0;
  private set = 0; private slice = 0;
  constructor(private nSets: number, private height: number, private slices = BUILD_SLICES) {}
  reset(nSets: number, height: number) {
    this.nSets = nSets; this.height = height; this.set = 0; this.slice = 0; this.completedSets = 0;
  }
  next(): { set: number; rowStart: number; rowEnd: number } | null {
    if (this.set >= this.nSets || this.height <= 0) return null;
    const rows = Math.ceil(this.height / this.slices);
    const rowStart = this.slice * rows, rowEnd = Math.min(this.height, rowStart + rows);
    const out = { set: this.set, rowStart, rowEnd };
    this.slice++;
    if (rowEnd >= this.height) { this.slice = 0; this.set++; this.completedSets = this.set; }
    return out;
  }
}

export type Mode = "live" | "cached";
export function chooseMode(playing: boolean, completedSets: number, enabled: boolean): Mode {
  return playing && enabled && completedSets > 0 ? "cached" : "live";
}
