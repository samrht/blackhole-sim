// Wyman, Sloan & Shirley (2013), "Simple Analytic Approximations to the CIE XYZ
// Color Matching Functions" — single-lobe Gaussian fits. λ in nanometres.
function gaussian(x: number, mu: number, s1: number, s2: number): number {
  const t = (x - mu) * (x < mu ? 1 / s1 : 1 / s2);
  return Math.exp(-0.5 * t * t);
}
function cieX(l: number) { return 1.056 * gaussian(l, 599.8, 37.9, 31.0) + 0.362 * gaussian(l, 442.0, 16.0, 26.7) - 0.065 * gaussian(l, 501.1, 20.4, 26.2); }
function cieY(l: number) { return 0.821 * gaussian(l, 568.8, 46.9, 40.5) + 0.286 * gaussian(l, 530.9, 16.3, 31.1); }
function cieZ(l: number) { return 1.217 * gaussian(l, 437.0, 11.8, 36.0) + 0.681 * gaussian(l, 459.0, 26.0, 13.8); }

const H = 6.62607015e-34, C = 2.99792458e8, KB = 1.380649e-23;
function planck(lambda_m: number, T: number): number {
  return (1 / Math.pow(lambda_m, 5)) / (Math.exp(H * C / (lambda_m * KB * T)) - 1);
}

/** Linear sRGB color of a blackbody at temperature T (K), chromaticity-preserving, max channel = 1. */
export function blackbodyLinearSRGB(T: number): [number, number, number] {
  let X = 0, Y = 0, Z = 0;
  for (let nm = 360; nm <= 830; nm += 5) {
    const p = planck(nm * 1e-9, T);
    X += p * cieX(nm); Y += p * cieY(nm); Z += p * cieZ(nm);
  }
  // XYZ -> linear sRGB (IEC 61966-2-1, D65)
  let r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z;
  let g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z;
  let b = 0.0557 * X - 0.2040 * Y + 1.0570 * Z;
  r = Math.max(0, r); g = Math.max(0, g); b = Math.max(0, b);
  const m = Math.max(r, g, b) || 1;
  return [r / m, g / m, b / m];
}
