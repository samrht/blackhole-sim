import { iscoRadius, omegaKepler, circE, circL } from "./orbits";

const deriv = (f: (x: number) => number, x: number, h = 1e-5) => (f(x + h) - f(x - h)) / (2 * h);

/** Dimensionless Page-Thorne flux shape F(r) (Ṁ=M=1, √-g=r). Returns 0 inside ISCO. */
export function pageThorneFluxShape(r: number, a: number, prograde = true): number {
  const ri = iscoRadius(a, prograde);
  if (r <= ri) return 0;
  const E = (x: number) => circE(x, a, prograde);
  const L = (x: number) => circL(x, a, prograde);
  const Om = (x: number) => omegaKepler(x, a, prograde);
  const integrand = (x: number) => (E(x) - Om(x) * L(x)) * deriv(L, x);
  // Composite Simpson over [ri, r]
  const n = 400, h = (r - ri) / n;
  let I = integrand(ri) + integrand(r);
  for (let k = 1; k < n; k++) I += (k % 2 ? 4 : 2) * integrand(ri + k * h);
  I *= h / 3;
  const Er = E(r), Lr = L(r), Omr = Om(r), denom = Er - Omr * Lr;
  return (-deriv(Om, r) / (r * denom * denom)) * I;
}

/** Normalized temperature shape T(r)/T_peak in [0,1] (∝ F^{1/4}). */
let _cache: { a: number; prograde: boolean; peak: number } | null = null;
function peakFlux(a: number, prograde: boolean): number {
  if (_cache && _cache.a === a && _cache.prograde === prograde) return _cache.peak;
  const ri = iscoRadius(a, prograde);
  let peak = 0;
  for (let r = ri * 1.01; r < ri * 20; r *= 1.01) peak = Math.max(peak, pageThorneFluxShape(r, a, prograde));
  _cache = { a, prograde, peak };
  return peak;
}
export function temperatureShape(r: number, a: number, prograde = true): number {
  const f = pageThorneFluxShape(r, a, prograde);
  if (f <= 0) return 0;
  return Math.pow(f / peakFlux(a, prograde), 0.25);
}
