import { metricUpper } from "./kerr";

function hquad(r: number, th: number, a: number, pt: number, pr: number, pth: number, pphi: number): number {
  const g = metricUpper(r, th, a);
  return g.tt * pt * pt + 2 * g.tphi * pt * pphi + g.rr * pr * pr + g.thth * pth * pth + g.phph * pphi * pphi;
}

export function nullRadialMomentum(r: number, th: number, a: number, pt: number, pphi: number, pth: number): number {
  // solve g^{rr} p_r^2 = -(g^tt pt^2 + 2 g^tφ pt pφ + g^θθ pθ^2 + g^φφ pφ^2)
  const g = metricUpper(r, th, a);
  const rest = g.tt * pt * pt + 2 * g.tphi * pt * pphi + g.thth * pth * pth + g.phph * pphi * pphi;
  return Math.sqrt(Math.max(0, -rest / g.rr));
}

export function rhs(s: Float64Array, a: number): Float64Array {
  const [, r, th, , pt, pr, pth, pphi] = s;
  const g = metricUpper(r, th, a);
  const dt = g.tt * pt + g.tphi * pphi;
  const dr = g.rr * pr;
  const dth = g.thth * pth;
  const dphi = g.tphi * pt + g.phph * pphi;
  const h = 1e-5;
  const dQdr = (hquad(r + h, th, a, pt, pr, pth, pphi) - hquad(r - h, th, a, pt, pr, pth, pphi)) / (2 * h);
  const dQdth = (hquad(r, th + h, a, pt, pr, pth, pphi) - hquad(r, th - h, a, pt, pr, pth, pphi)) / (2 * h);
  return new Float64Array([dt, dr, dth, dphi, 0, -0.5 * dQdr, -0.5 * dQdth, 0]);
}

export function rk4(s: Float64Array, a: number, dl: number): Float64Array {
  const add = (x: Float64Array, k: Float64Array, f: number) => {
    const o = new Float64Array(8);
    for (let i = 0; i < 8; i++) o[i] = x[i] + k[i] * f;
    return o;
  };
  const k1 = rhs(s, a);
  const k2 = rhs(add(s, k1, dl / 2), a);
  const k3 = rhs(add(s, k2, dl / 2), a);
  const k4 = rhs(add(s, k3, dl), a);
  const o = new Float64Array(8);
  for (let i = 0; i < 8; i++) o[i] = s[i] + (dl / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
  return o;
}

export function conserved(s: Float64Array, a: number, mu = 0) {
  const [, r, th, , pt, pr, pth, pphi] = s;
  const E = -pt, Lz = pphi;
  const c = Math.cos(th), sn = Math.sin(th);
  const Q = pth * pth + c * c * (a * a * (mu * mu - E * E) + (Lz * Lz) / (sn * sn));
  const H = 0.5 * hquad(r, th, a, pt, pr, pth, pphi);
  return { E, Lz, Q, H };
}
