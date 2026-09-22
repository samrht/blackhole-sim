/** Internal-resolution controller for the animated view ("smooth first", spec 3.3). Feed it each
 *  frame's GPU work time (main.ts passes Renderer.gpuMs; the rAF delta is vsync-quantised and cannot
 *  signal headroom). It keeps a smoothed frame time and nudges the render scale down (x0.9)
 *  when frames are slower than SLOW_MS and up (x1.1) when faster than FAST_MS, within [MIN, MAX],
 *  at most once per HOLD_MS so the image does not pump. Times above MAX_DT_MS (tab switch, stall)
 *  are not a performance signal and are ignored. */
export const SCALE = { MIN: 0.5, MAX: 1, SLOW_MS: 18, FAST_MS: 13, DOWN: 0.9, UP: 1.1, HOLD_MS: 500, ALPHA: 0.2, MAX_DT_MS: 250 };

export class ScaleController {
  scale = 1;
  private ema = -1;
  private lastChange = -Infinity;

  /** Returns the new scale when it changed this frame, else null. */
  update(dtMs: number, nowMs: number): number | null {
    if (!(dtMs > 0) || dtMs > SCALE.MAX_DT_MS) return null;
    this.ema = this.ema < 0 ? dtMs : this.ema + SCALE.ALPHA * (dtMs - this.ema);
    if (nowMs - this.lastChange < SCALE.HOLD_MS) return null;
    let next = this.scale;
    if (this.ema > SCALE.SLOW_MS) next = this.scale * SCALE.DOWN;
    else if (this.ema < SCALE.FAST_MS) next = this.scale * SCALE.UP;
    next = Math.min(SCALE.MAX, Math.max(SCALE.MIN, next));
    if (next === this.scale) return null;
    this.scale = next;
    this.lastChange = nowMs;
    this.ema = -1; // re-measure at the new scale rather than reacting to the old one's history
    return next;
  }

  reset(scale = 1) { this.scale = scale; this.ema = -1; this.lastChange = -Infinity; }
}
