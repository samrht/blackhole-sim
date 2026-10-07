// Verdict of the ?golden route (pure, unit-tested in tests/golden-judge.test.ts).
export interface GoldenResult { adapter: string; hashes: Record<string, string>; nonFinite?: Record<string, number> }

/** Non-finite values (NaN, +-Infinity) in a readback: a broken image still hashes, so the route counts them too. */
export function countNonFinite(a: Float32Array): number {
  let n = 0; for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) n++;
  return n;
}
/** FAIL on any non-finite pixel (any adapter); SKIP when recorded on a different adapter; else PASS iff every recorded
 *  scene hash matches (scenes added since the recording are not judged until recorded). */
export function judgeGolden(want: GoldenResult, got: GoldenResult): "PASS" | "FAIL" | "SKIP" {
  if (got.nonFinite && Object.values(got.nonFinite).some((n) => n > 0)) return "FAIL";
  if (want.adapter !== got.adapter) return "SKIP";
  return Object.keys(want.hashes).every((k) => want.hashes[k] === got.hashes[k]) ? "PASS" : "FAIL";
}
