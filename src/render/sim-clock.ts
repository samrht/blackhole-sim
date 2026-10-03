// Simulation clock (final review of the MRI turbulence, 2026-10-03).
// The shader reads time in f32. At large t, `time - delay` rounded each pixel's emission time
// differently (ulp(1.8e6) = 0.125 M), which turned the fast-evolving MRI turbulence into per-pixel grain.
// The CPU therefore splits its f64 clock into an epoch (a multiple of TIME_EPOCH, exact in f32) and a
// remainder below TIME_EPOCH; the shader forms each pixel's emission time from the small remainder.

/** Longest real-time step one frame may advance the clock: a tab returning from the background
 *  (requestAnimationFrame paused for hours) must not jump the simulation hours ahead. */
export const MAX_FRAME_DT_MS = 100;
/** Epoch granularity (M): a power of two, so every epoch below 2^24 * 2048 is exact in f32. */
export const TIME_EPOCH = 2048;

/** simTime advanced by one frame of dtMs real milliseconds at `speed` M per second times Motion. */
export function advanceSimTime(simTime: number, dtMs: number, speed: number, motion: number): number {
  return simTime + (Math.min(Math.max(dtMs, 0), MAX_FRAME_DT_MS) / 1000) * speed * motion;
}

/** simTime = epoch + rel with epoch a multiple of TIME_EPOCH and 0 <= rel < TIME_EPOCH. */
export function splitTime(simTime: number): { epoch: number; rel: number } {
  const epoch = TIME_EPOCH * Math.floor(simTime / TIME_EPOCH);
  return { epoch, rel: simTime - epoch };
}
