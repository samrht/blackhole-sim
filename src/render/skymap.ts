// Equirectangular (lat-long) sampling of the sky panorama, plus a fixed galactic tilt so the
// Milky-Way band sits off-axis from the accretion disk. The WGSL twins (skySample / tiltDir in
// raytrace.wgsl) MUST mirror these formulas and coefficients.

export const SKY_TEXW = 4096;

// R_SKY = Rz(30°)·Rx(60°), applied as R·dir (row-major). Orthonormal (det = 1).
const R_SKY = [
  0.866025, -0.25,      0.433013,
  0.5,       0.433013, -0.75,
  0.0,       0.866025,  0.5,
] as const;

export function tiltDir(d: [number, number, number]): [number, number, number] {
  const m = R_SKY;
  return [
    m[0] * d[0] + m[1] * d[1] + m[2] * d[2],
    m[3] * d[0] + m[4] * d[1] + m[5] * d[2],
    m[6] * d[0] + m[7] * d[1] + m[8] * d[2],
  ];
}

export function dirToEquirectUV(d: [number, number, number]): [number, number] {
  const u = Math.atan2(d[2], d[0]) * (0.5 / Math.PI) + 0.5;
  const v = Math.acos(Math.max(-1, Math.min(1, d[1]))) * (1 / Math.PI);
  return [u, v];
}
