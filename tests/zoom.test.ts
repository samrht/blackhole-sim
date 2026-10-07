import { describe, it, expect } from "vitest";
import { ZOOM, fovForZoom, clampZoom, wheelZoom, pinchZoom, formatZoom } from "../src/render/zoom";

describe("zoom (camera field of view)", () => {
  it("log2 zoom: 1x is today's 14 M half-height; each step of 1 halves the field", () => {
    expect(fovForZoom(0)).toBe(14);
    expect(fovForZoom(1)).toBe(7);
    expect(fovForZoom(ZOOM.min)).toBe(56);   // 0.25x: the whole disk (r_out 40) in view
    expect(fovForZoom(ZOOM.max)).toBe(0.875); // 16x: the photon ring fills the view
  });
  it("clamps to [min, max] and snaps to the slider step", () => {
    expect(clampZoom(-9)).toBe(ZOOM.min); expect(clampZoom(9)).toBe(ZOOM.max);
    expect(clampZoom(0.123456)).toBeCloseTo(0.12, 12);
    expect(clampZoom(NaN)).toBe(0);
  });
  it("wheel: scrolling up (negative deltaY) zooms in; lines and pages count as pixels x 16 / x 400", () => {
    expect(wheelZoom(0, -100, 0)).toBeCloseTo(0.15, 12);
    expect(wheelZoom(0, 100, 0)).toBeCloseTo(-0.15, 12);
    expect(wheelZoom(0, -3, 1)).toBeCloseTo(wheelZoom(0, -48, 0), 12);
    expect(wheelZoom(ZOOM.max, -1000, 0)).toBe(ZOOM.max);
  });
  it("pinch: doubling the finger distance zooms in by 2x (log2 +1) from the zoom at the pinch's start", () => {
    expect(pinchZoom(0.5, 100, 200)).toBeCloseTo(1.5, 12);
    expect(pinchZoom(0.5, 200, 100)).toBeCloseTo(-0.5, 12);
    expect(pinchZoom(0.5, 0, 100)).toBe(0.5); // degenerate start distance: no change
  });
  it("formats the factor", () => {
    expect(formatZoom(0)).toBe("1.00"); expect(formatZoom(-2)).toBe("0.25"); expect(formatZoom(4)).toBe("16.0");
  });
});

import { zoomAbout, clampPan } from "../src/render/zoom";
describe("zoom toward the cursor (image-plane pan)", () => {
  const impact = (ndc: [number, number], z: number, pan: [number, number], aspect: number) =>
    [ndc[0] * fovForZoom(z) * aspect + pan[0], -ndc[1] * fovForZoom(z) + pan[1]];
  it("keeps the point under the cursor fixed while the view stays inside the 1x frame", () => {
    const aspect = 16 / 9, ndc: [number, number] = [0.3, -0.2];
    const pan = zoomAbout(0, 2, [0, 0], ndc, aspect), a0 = impact(ndc, 0, [0, 0], aspect), a1 = impact(ndc, 2, pan, aspect);
    expect(a1[0]).toBeCloseTo(a0[0], 12); expect(a1[1]).toBeCloseTo(a0[1], 12);
    const pan2 = zoomAbout(2, 3, pan, [-0.5, 0.4], aspect), b0 = impact([-0.5, 0.4], 2, pan, aspect), b1 = impact([-0.5, 0.4], 3, pan2, aspect);
    expect(b1[0]).toBeCloseTo(b0[0], 12); expect(b1[1]).toBeCloseTo(b0[1], 12);
  });
  it("clamps the pan to the 1x frame; at 1x and below there is no pan", () => {
    expect(clampPan([100, -100], 2, 1.5)).toEqual([(14 - 3.5) * 1.5, -(14 - 3.5)]);
    expect(clampPan([3, 3], 0, 1.5)).toEqual([0, 0]);
    expect(clampPan([3, 3], -1, 1.5)).toEqual([0, 0]);
    expect(zoomAbout(2, 0, [5, -4], [0.9, 0.9], 1.5)).toEqual([0, 0]); // zooming back out re-centres
  });
});
