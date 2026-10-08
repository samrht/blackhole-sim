// ?accuracy's 1.3 mm intensity judging (pure, unit-tested in tests/accuracy-judge.test.ts).

/** Intensity is judged where the reference OR the renderer reaches DARK_I of the scene's peak. Below it in both, the
 *  pixel is invisible (the afmhot scale spans ~1e-3) and the GPU skips gas under its ln n < -40 density cutoff by
 *  design, which the CPU reference does not (Gargantua's frame corners: 1e-23 of the peak there, exactly 0 here). */
export const DARK_I = 1e-9;
export function intensityScored(refI: number, gotI: number, peak: number): boolean {
  return Math.max(refI, gotI) >= DARK_I * peak;
}
/** A scene's frozen bounds: pixels above their floor, and the worst error / floor. */
export interface Bounds { over: number; worst: number }
/** Why a scene fails its frozen bounds ([] = within them). */
export function frozenFailures(overN: number, maxN: number, b: Bounds): string[] {
  const out: string[] = [];
  if (overN > b.over) out.push(`${overN} px above floor > bound ${b.over}`);
  if (maxN > b.worst) out.push(`worst ${maxN.toFixed(2)}x floor > bound ${b.worst}x`);
  return out;
}
