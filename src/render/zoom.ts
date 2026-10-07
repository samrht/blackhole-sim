// Camera zoom: the field of view of the ray-traced camera. The control is log2 of the zoom factor; 1x keeps the original
// fovScale of 14 M (the half-height of the view at the observer, in units of M). Pure helpers for main.ts (wheel, pinch,
// keys, slider) and tests/zoom.test.ts. Physics is untouched: the fov only sets which rays each pixel traces.
export const ZOOM = { min: -2, max: 4, step: 0.01, fov0: 14, wheel: 0.0015, key: 0.25 } as const;

/** fovScale (M) for log2 zoom z. */
export function fovForZoom(z: number): number { return ZOOM.fov0 / 2 ** z; }
/** Clamp to [min, max] and snap to the slider step (NaN -> 1x). */
export function clampZoom(z: number): number {
  if (!Number.isFinite(z)) return 0;
  const s = Math.round(Math.min(ZOOM.max, Math.max(ZOOM.min, z)) / ZOOM.step) * ZOOM.step;
  return Number(s.toFixed(2));
}
/** Mouse wheel: scrolling up (deltaY < 0) zooms in. deltaMode 1 (lines) and 2 (pages) count as 16 and 400 pixels. */
export function wheelZoom(z: number, deltaY: number, deltaMode: number): number {
  const px = deltaMode === 1 ? deltaY * 16 : deltaMode === 2 ? deltaY * 400 : deltaY;
  return clampZoom(z - px * ZOOM.wheel);
}
/** Pinch: the zoom at the pinch's start z0 times the change in finger distance d0 -> d1. */
export function pinchZoom(z0: number, d0: number, d1: number): number {
  if (!(d0 > 0) || !(d1 > 0)) return z0;
  return clampZoom(z0 + Math.log2(d1 / d0));
}
/** The factor for the readout: 0.25 ... 9.99 with two decimals, 10 and up with one. */
export function formatZoom(z: number): string { const f = 2 ** z; return f >= 10 ? f.toFixed(1) : f.toFixed(2); }

/** The pan (image-plane offset of the view's centre, in M) is limited so the view stays inside the 1x frame: none at 1x
 *  and below, up to (14 - fov) at zoom z (times the aspect ratio horizontally). */
export function clampPan(pan: [number, number], z: number, aspect: number): [number, number] {
  const m = Math.max(0, ZOOM.fov0 - fovForZoom(z));
  const cx = Math.max(-m * aspect, Math.min(m * aspect, pan[0])), cy = Math.max(-m, Math.min(m, pan[1]));
  return [cx + 0, cy + 0]; // + 0 turns -0 into 0
}
/** The pan after zooming from z0 to z1 about the screen point ndc (x right, y down, in [-1, 1]; aspect = width / height):
 *  the point under the cursor stays put (raytrace.wgsl pixelImpact: alpha = ndc.x fov aspect + panX, beta = -ndc.y fov +
 *  panY), then clampPan. */
export function zoomAbout(z0: number, z1: number, pan: [number, number], ndc: [number, number], aspect: number): [number, number] {
  const f0 = fovForZoom(z0), f1 = fovForZoom(z1);
  const al = ndc[0] * f0 * aspect + pan[0], be = -ndc[1] * f0 + pan[1];
  return clampPan([al - ndc[0] * f1 * aspect, be + ndc[1] * f1], z1, aspect);
}
