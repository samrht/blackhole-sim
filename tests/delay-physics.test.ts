import { describe, it, expect } from "vitest";
import { screenToState } from "../src/physics/camera";
import { stepGeodesic } from "../src/physics/trace";

/** Fine-step integration (independent of the renderer's controller) until stop(s) is true. */
function integrate(s0: Float64Array, a: number, stop: (prev: Float64Array, next: Float64Array) => boolean) {
  let s = s0;
  for (let k = 0; k < 400000; k++) {
    const r = s[1], dl = r > 60 ? Math.min(2, 0.01 * r) : 0.005;
    const n = stepGeodesic(s, a, dl, 1e-9, 30).s;
    if (stop(s, n)) return [s, n] as const;
    s = n;
  }
  throw new Error("did not stop");
}

describe("light-travel time (coordinate t along the backward ray)", () => {
  it("a = 0 radial ray from r = 1000 to r = 10 takes r*(1000) - r*(10) = 999.652 M", () => {
    const rStar = (r: number) => r + 2 * Math.log(r / 2 - 1);
    const [p, n] = integrate(screenToState(0, 0, 0, Math.PI / 2, 1000), 0, (_, nx) => nx[1] <= 10);
    const f = (p[1] - 10) / (p[1] - n[1]), t10 = p[0] + f * (n[0] - p[0]);
    expect(t10).toBeLessThan(0); // past-directed: t decreases from the camera's t = 0
    expect(-t10).toBeCloseTo(rStar(1000) - rStar(10), 3);
  });
  it("a near-critical a = 0 ray crosses the equator every pi*sqrt(27) M near r = 3M (subimage delay)", () => {
    const b = Math.sqrt(27) * (1 + 1e-7), inc = (10 * Math.PI) / 180;
    let s = screenToState(0.3, Math.sqrt(b * b - 0.09), 0, inc, 1000);
    const cross: number[] = [];
    for (let k = 0; k < 400000 && cross.length < 5; k++) {
      const r = s[1], dl = r > 60 ? Math.min(2, 0.01 * r) : 0.005;
      const n = stepGeodesic(s, 0, dl, 1e-9, 30).s;
      if ((s[2] - Math.PI / 2) * (n[2] - Math.PI / 2) < 0 && n[1] < 4) {
        const f = (s[2] - Math.PI / 2) / (s[2] - n[2]); cross.push(s[0] + f * (n[0] - s[0]));
      }
      s = n; if (s[1] < 2.01 || s[1] > 1100) break;
    }
    expect(cross.length).toBeGreaterThanOrEqual(4);
    for (let i = 1; i < cross.length; i++) {
      expect(Math.abs(Math.abs(cross[i] - cross[i - 1]) / (Math.PI * Math.sqrt(27)) - 1)).toBeLessThan(0.03);
    }
  }, 120000);
});
