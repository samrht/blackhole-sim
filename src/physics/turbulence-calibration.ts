// Calibration of the turbulence amplitude (spec 2026-10-03 §2.2): the intrinsic (face-on, before
// relativistic beaming) fractional rms of the disk's integrated light, for each lognormal sigma.
// Not imported by the app; scripts/calibrate-turbulence.ts prints FLICKER_TABLE from it.
import { turbulenceAt, lognormalFactor } from "./emission";
import { iscoRadius } from "./orbits";
import { pageThorneFluxShape } from "./disk";

export interface FlickerGrid { ne: number; np: number; ns: number; dt: number; a: number; rOut: number; }
/** ne log-spaced radii from the ISCO to rOut, np azimuths, ns snapshots dt apart (M). */
export const FULL_GRID: FlickerGrid = { ne: 160, np: 256, ns: 1500, dt: 397, a: 0.9, rOut: 40 };

export function diskFlickerRms(sigmas: number[], g: FlickerGrid): number[] {
  const ri = iscoRadius(g.a), l0 = Math.log(ri), l1 = Math.log(g.rOut);
  const rs: number[] = [], w: number[] = [];
  let tot = 0;
  for (let i = 0; i < g.ne; i++) {
    const r = Math.exp(l0 + ((l1 - l0) * (i + 0.5)) / g.ne);
    const wi = pageThorneFluxShape(r, g.a) * r * r; // F dA = F r dr dphi = F r^2 d(ln r) dphi
    rs.push(r); w.push(wi); tot += wi * g.np;
  }
  const L = sigmas.map(() => [] as number[]);
  for (let s = 0; s < g.ns; s++) {
    const t = 1000 + s * g.dt, sums = sigmas.map(() => 0);
    for (let i = 0; i < g.ne; i++) for (let j = 0; j < g.np; j++) {
      const gv = turbulenceAt(rs[i], (2 * Math.PI * j) / g.np, t, g.a);
      for (let q = 0; q < sigmas.length; q++) sums[q] += w[i] * lognormalFactor(gv, sigmas[q]);
    }
    sums.forEach((v, q) => L[q].push(v / tot));
  }
  return L.map((xs) => {
    const m = xs.reduce((p, c) => p + c, 0) / xs.length;
    return Math.sqrt(xs.reduce((p, c) => p + (c - m) ** 2, 0) / xs.length) / m;
  });
}
