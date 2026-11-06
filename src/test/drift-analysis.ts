// Pure helpers of the ?drift route (drift.browser.ts): parameters, heat-map scale, statistics.

export interface DriftParams { a: number; inclDeg: number; disk: boolean; fov: number; panX: number; panY: number; maxSteps: number; cpuStride: number }

/** ?drift[&a=][&i=][&disk=0][&fov=][&px=][&py=][&steps=][&cpu=]. Defaults: a high-spin, near edge-on frame with the disk. */
export function parseDriftParams(search: string): DriftParams {
  const q = new URLSearchParams(search), num = (k: string, d: number) => { const v = q.get(k); return v === null ? d : Number(v); };
  return { a: num("a", 0.998), inclDeg: num("i", 85), disk: q.get("disk") !== "0", fov: num("fov", 14), panX: num("px", 0), panY: num("py", 0), maxSteps: num("steps", 4800), cpuStride: num("cpu", 8) };
}

/** v on a log scale between lo and hi, clamped to [0, 1]; 0 maps to 0; non-finite maps to NaN (undefined, not "large"). */
export function logUnit(v: number, lo: number, hi: number): number {
  if (!Number.isFinite(v)) return NaN;
  if (v <= lo) return 0;
  if (v >= hi) return 1;
  return Math.log(v / lo) / Math.log(hi / lo);
}

/** One hue, light to dark: the reference sequential blue with its ends extended so 8 decades stay legible. */
export const HEAT_RAMP: readonly (readonly [number, number, number])[] = [
  [247, 250, 254], [205, 226, 251], [158, 197, 244], [109, 167, 236], [57, 135, 229], [37, 106, 191], [24, 79, 149], [13, 54, 107], [6, 24, 47],
];
export const HEAT_UNDEFINED: [number, number, number] = [184, 184, 184];

export function heatColor(u: number): [number, number, number] {
  if (Number.isNaN(u)) return HEAT_UNDEFINED;
  const x = Math.min(1, Math.max(0, u)) * (HEAT_RAMP.length - 1), k = Math.min(HEAT_RAMP.length - 2, Math.floor(x)), f = x - k;
  const A = HEAT_RAMP[k], B = HEAT_RAMP[k + 1];
  return [A[0] + (B[0] - A[0]) * f, A[1] + (B[1] - A[1]) * f, A[2] + (B[2] - A[2]) * f];
}

export interface Summary { n: number; undefined: number; p50: number; p90: number; p99: number; max: number }

/** Quantiles (nearest rank) over the finite values; the non-finite ones are counted apart. */
export function summarize(v: ArrayLike<number>): Summary {
  const f: number[] = []; let bad = 0;
  for (let i = 0; i < v.length; i++) { if (Number.isFinite(v[i])) f.push(v[i]); else bad++; }
  if (!f.length) return { n: 0, undefined: bad, p50: NaN, p90: NaN, p99: NaN, max: NaN };
  f.sort((x, y) => x - y);
  const at = (p: number) => f[Math.min(f.length - 1, Math.floor(p * f.length))];
  return { n: f.length, undefined: bad, p50: at(0.5), p90: at(0.9), p99: at(0.99), max: f[f.length - 1] };
}

/** Distance from (x, y) to the nearest point of a densely sampled curve. */
export function curveDistance(x: number, y: number, pts: readonly (readonly [number, number])[]): number {
  let best = Infinity;
  for (let k = 0; k < pts.length; k++) { const dx = x - pts[k][0], dy = y - pts[k][1], d = dx * dx + dy * dy; if (d < best) best = d; }
  return Math.sqrt(best);
}
