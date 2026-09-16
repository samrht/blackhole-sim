export const M = 1;

// Boyer–Lindquist coordinates are singular on the polar axis: g^{φφ} carries a 1/sin²θ that
// diverges as θ→0,π. POLE_S2 floors sin²θ in that denominator ONLY to stop an exactly-on-axis
// ray with p_φ = 0 from evaluating inf·0 = NaN. It is a NaN guard, not a physics cap: at 1e-12
// the affected cone has half-angle 1e-6 rad, thousands of times below a pixel, so the centrifugal
// barrier is exact everywhere a ray can feel it. (It used to be 1e-3 — a 1.8° cone inside which
// the barrier vanished and rays tunnelled through the axis; see trace.ts for the integrator side.)
export const POLE_S2 = 1e-12;

export interface Metric { tt: number; tphi: number; rr: number; thth: number; phph: number; }

export function sigma(r: number, theta: number, a: number): number {
  const c = Math.cos(theta);
  return r * r + a * a * c * c;
}
export function delta(r: number, a: number): number {
  return r * r - 2 * M * r + a * a;
}
export function bigA(r: number, theta: number, a: number): number {
  const s = Math.sin(theta);
  return (r * r + a * a) ** 2 - a * a * delta(r, a) * s * s;
}
export function horizonOuter(a: number): number {
  return M + Math.sqrt(Math.max(0, M * M - a * a));
}
export function ergosphere(theta: number, a: number): number {
  const c = Math.cos(theta);
  return M + Math.sqrt(Math.max(0, M * M - a * a * c * c));
}
export function omegaHorizon(a: number): number {
  const rp = horizonOuter(a);
  return a / (rp * rp + a * a);
}
export function metricLower(r: number, theta: number, a: number): Metric {
  const s2 = Math.sin(theta) ** 2;
  const Sig = sigma(r, theta, a), d = delta(r, a);
  return {
    tt: -(1 - 2 * M * r / Sig),
    tphi: -2 * M * a * r * s2 / Sig,
    rr: Sig / d,
    thth: Sig,
    phph: (r * r + a * a + 2 * M * a * a * r * s2 / Sig) * s2,
  };
}
export function metricUpper(r: number, theta: number, a: number): Metric {
  const s2 = Math.sin(theta) ** 2;
  const s2d = Math.max(s2, POLE_S2); // pole-regularized denominator (see POLE_S2)
  const Sig = sigma(r, theta, a), d = delta(r, a), A = bigA(r, theta, a);
  return {
    tt: -A / (Sig * d),
    tphi: -2 * M * a * r / (Sig * d),
    rr: d / Sig,
    thth: 1 / Sig,
    phph: (d - a * a * s2) / (Sig * d * s2d),
  };
}
export function omegaZAMO(r: number, theta: number, a: number): number {
  return 2 * M * a * r / bigA(r, theta, a);
}
