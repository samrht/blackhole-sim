// CPU image of the hot flow at 230 GHz (calibration and validation; the renderer's twin is the GPU path).
import { screenToState } from "./camera";
import { stepGeodesic, stepSize, H_TOL, H_TOL_FAR } from "./trace";
import { horizonOuter } from "./kerr";
import { HOTFLOW, flowShift, flowCoeffs } from "./hot-flow";

export const HOTFLOW_TARGETS = {
  sgra: { distKpc: 8.2, jy: 2.4, ringUas: 51.8, ringErr: 2.3 },   // EHT 2022 (Sgr A* Papers I, IV)
  m87: { distKpc: 16800, jy: 0.5, ringUas: 42, ringErr: 3 },      // EHT 2019 (Papers I, IV, VI); not gated (spec 2.4)
} as const;
export interface FlowSamples { N: number; half: number; rays: Float64Array[] } // per ray: (r, th, D, dl) quads
export function traceFlowSamples(a: number, inclDeg: number, N: number, half: number): FlowSamples {
  const incl = (inclDeg * Math.PI) / 180, rh = horizonOuter(a), rays: Float64Array[] = [];
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const al = -half + (2 * half * (i + 0.5)) / N, be = half - (2 * half * (j + 0.5)) / N;
    let s = screenToState(al, be, a, incl, 1000); const q: number[] = [];
    for (let k = 0; k < 30000; k++) {
      const out = stepGeodesic(s, a, Math.min(stepSize(s, rh, 40), 0.5), s[1] > 60 ? H_TOL_FAR : H_TOL);
      if (!out.ok) break;
      const r = out.s[1];
      if (r < HOTFLOW.rMax && r > rh * 1.01) { const D = flowShift(out.s, a); if (D !== null && D > 0) q.push(r, out.s[2], D, out.dl); }
      s = out.s;
      if (r <= rh * 1.01 || r > 1100) break;
    }
    rays.push(Float64Array.from(q));
  }
  return { N, half, rays };
}
export function flowImage(smp: FlowSamples, n0: number, rgCm: number): Float64Array {
  const I = new Float64Array(smp.N * smp.N);
  smp.rays.forEach((q, idx) => {
    let Iv = 0, tau = 0;
    for (let k = 0; k < q.length; k += 4) {
      const [r, th, D, dl] = [q[k], q[k + 1], q[k + 2], q[k + 3]];
      const [j, al] = flowCoeffs(r, th, D * HOTFLOW.nu, n0); if (j === 0) continue;
      const ds = rgCm * D * dl, dt = al * ds, fac = dt < 1e-4 ? 1 - 0.5 * dt : (1 - Math.exp(-dt)) / dt;
      Iv += (j / D ** 3) * ds * fac * Math.exp(-tau); tau += dt;
    }
    I[idx] = Iv;
  });
  return I;
}
export function imageFluxJy(I: Float64Array, N: number, half: number, rgCm: number, distCm: number): number {
  const pix = ((2 * half) / N) * (rgCm / distCm); let F = 0; for (const v of I) F += v * pix * pix; return F / 1e-23;
}
export function ringDiameterUas(I: Float64Array, N: number, half: number, uasPerM: number, blurUas = 0): number {
  let img = I; const px = (2 * half) / N;
  if (blurUas > 0) {
    const sig = blurUas / 2.3548 / uasPerM, R = Math.ceil((3 * sig) / px); img = new Float64Array(N * N);
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { let v = 0, w = 0;
      for (let dj = -R; dj <= R; dj++) for (let di = -R; di <= R; di++) { const ii = i + di, jj = j + dj; if (ii < 0 || jj < 0 || ii >= N || jj >= N) continue;
        const ww = Math.exp(-((di * px) ** 2 + (dj * px) ** 2) / (2 * sig * sig)); v += ww * I[jj * N + ii]; w += ww; } img[j * N + i] = v / w; }
  }
  const nb = 60, prof = new Float64Array(nb), cnt = new Float64Array(nb);
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const b = Math.hypot(-half + px * (i + 0.5), -half + px * (j + 0.5)), k = Math.floor((b / half) * nb);
    if (k < nb) { prof[k] += img[j * N + i]; cnt[k]++; } }
  const m = (k: number) => (cnt[k] ? prof[k] / cnt[k] : 0);
  let kmax = 0; for (let k = 1; k < nb; k++) if (cnt[k] && m(k) > m(kmax)) kmax = k;
  // Sub-bin peak: a parabola through the peak bin and its neighbours (the bin centre alone quantised the diameter to
  // 2 half / nb, ~2.3 uas for Sgr A*, enough to move it by one bin between frames and resolutions).
  let dk = 0;
  if (kmax > 0 && kmax < nb - 1 && cnt[kmax - 1] && cnt[kmax + 1]) {
    const y0 = m(kmax - 1), y1 = m(kmax), y2 = m(kmax + 1), den = y0 - 2 * y1 + y2;
    if (den < 0) dk = Math.max(-0.5, Math.min(0.5, (0.5 * (y0 - y2)) / den));
  }
  return 2 * ((kmax + 0.5 + dk) / nb) * half * uasPerM;
}
