import { describe, it, expect } from "vitest";
import { ScaleController } from "../src/render/scale";

/** Feed `n` frames of `dt` ms starting at time `t0`; returns the time after the last frame. */
function feed(c: ScaleController, dt: number, n: number, t0 = 0) {
  let t = t0;
  for (let i = 0; i < n; i++) { t += dt; c.update(dt, t); }
  return t;
}

describe("ScaleController", () => {
  it("starts at full resolution", () => { expect(new ScaleController().scale).toBe(1); });

  it("steps down under sustained slow frames and stops at 0.5", () => {
    const c = new ScaleController();
    const t = feed(c, 40, 30);                      // 1.2 s of 25 fps
    expect(c.scale).toBeLessThan(1);
    feed(c, 40, 400, t);                            // long enough to hit the floor
    expect(c.scale).toBe(0.5);
  });

  it("holds steady inside the dead band (60 Hz vsync = 16.7 ms)", () => {
    const c = new ScaleController();
    c.reset(0.7);
    feed(c, 16.7, 600);
    expect(c.scale).toBe(0.7);
  });

  it("climbs back to exactly 1.0 when frames are fast", () => {
    const c = new ScaleController();
    c.reset(0.5);
    feed(c, 8, 1000);
    expect(c.scale).toBe(1);
  });

  it("changes at most once per 500 ms", () => {
    const c = new ScaleController();
    const changes: number[] = [];
    let t = 0;
    for (let i = 0; i < 100; i++) { t += 40; if (c.update(40, t) !== null) changes.push(t); }
    for (let k = 1; k < changes.length; k++) expect(changes[k] - changes[k - 1]).toBeGreaterThanOrEqual(500);
    expect(changes.length).toBeGreaterThan(1);
  });

  it("ignores stalls (tab hidden, then shown) instead of collapsing the scale", () => {
    const c = new ScaleController();
    let t = feed(c, 16.7, 60);
    t += 5000; expect(c.update(5000, t)).toBeNull();
    feed(c, 16.7, 60, t);
    expect(c.scale).toBe(1);
  });

  it("ignores non-positive or NaN deltas", () => {
    const c = new ScaleController();
    expect(c.update(0, 100)).toBeNull();
    expect(c.update(-5, 200)).toBeNull();
    expect(c.update(NaN, 300)).toBeNull();
    expect(c.scale).toBe(1);
  });
});
