// CPU mirror of camera-shared.wgsl's sinCosP (precise sin / cos for the shader: WGSL's built-ins may be off by 2^-11
// absolute). Cephes sinf / cosf on |y| <= pi / 4 after Cody-Waite reduction by pi / 2. With f32 = true every operation
// is rounded to f32, as on the GPU (tests/sincos.test.ts).
export const SINCOS_COEFFS = [1.5703125, 4.837512969970703e-4, 7.549789954891882e-8,
  -1.9515295891e-4, 8.3321608736e-3, -1.6666654611e-1, 2.443315711809948e-5, -1.388731625493765e-3, 4.166664568298827e-2] as const;
export function sinCosP(x: number, f32 = false): [number, number] {
  const F = f32 ? Math.fround : (v: number) => v, K = SINCOS_COEFFS;
  const j = Math.round(F(x * F(0.6366197723675814)));
  const y = F(F(F(x - F(j * K[0])) - F(j * K[1])) - F(j * K[2])), z = F(y * y);
  const s = F(F(F(F(F(F(F(K[3] * z) + K[4]) * z) + K[5]) * z) * y) + y);
  const c = F(F(F(F(F(F(F(K[6] * z) + K[7]) * z) + K[8]) * z) * z) - F(F(0.5 * z) - 1));
  switch (((j % 4) + 4) % 4) {
    case 0: return [s, c];
    case 1: return [c, -s];
    case 2: return [-s, -c];
    default: return [-c, s];
  }
}
