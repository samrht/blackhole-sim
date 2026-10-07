// CPU image of the hot flow at 230 GHz (calibration and validation; the renderer's twin is the GPU path).
import { screenToState } from "./camera";
import { stepGeodesic, stepSize, H_TOL, H_TOL_FAR } from "./trace";
import { horizonOuter } from "./kerr";
import { HOTFLOW, flowShift, flowCoeffs } from "./hot-flow";
import { HOTSPOT, HOTSPOT_REACH, hotspotBoost, hotspotShift } from "./hotspot";

export const HOTFLOW_TARGETS = {
  // cpuRingUas: this CPU twin's ring at the calibrated n0 (scripts/calibrate-hotflow.ts, 96^2), which ?hotflow's GPU
  // image must reproduce (a twin check; the EHT comparison alone has a window of +-4.6 uas)
  sgra: { distKpc: 8.2, jy: 2.4, ringUas: 51.8, ringErr: 2.3, cpuRingUas: 48.2 },   // EHT 2022 (Sgr A* Papers I, IV)
  m87: { distKpc: 16800, jy: 0.5, ringUas: 42, ringErr: 3, cpuRingUas: 35.6 },      // EHT 2019 (Papers I, IV, VI); not gated (spec 2.4)
} as const;
// per ray: (r, th, D, dl, phi, t) per sample, D = -1 where the flow has no velocity (the hotspot can still emit there);
// mom: the ray's conserved (p_t, p_phi). t is the coordinate time (0 at the camera, decreasing): delay = -t - 1000.
export interface FlowSamples { N: number; half: number; a: number; rays: Float64Array[]; mom: Float64Array }
export function traceFlowSamples(a: number, inclDeg: number, N: number, half: number): FlowSamples {
  const incl = (inclDeg * Math.PI) / 180, rh = horizonOuter(a), rays: Float64Array[] = [], mom = new Float64Array(2 * N * N);
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const al = -half + (2 * half * (i + 0.5)) / N, be = half - (2 * half * (j + 0.5)) / N;
    let s = screenToState(al, be, a, incl, 1000); const q: number[] = [];
    mom[2 * (j * N + i)] = s[4]; mom[2 * (j * N + i) + 1] = s[7];
    for (let k = 0; k < 30000; k++) {
      const out = stepGeodesic(s, a, Math.min(stepSize(s, rh, 40), 0.5), s[1] > 60 ? H_TOL_FAR : H_TOL);
      if (!out.ok) break;
      const r = out.s[1];
      if (r < HOTFLOW.rMax && r > rh * 1.01) { const D = flowShift(out.s, a); q.push(r, out.s[2], D !== null && D > 0 ? D : -1, out.dl, out.s[3], out.s[0]); }
      s = out.s;
      if (r <= rh * 1.01 || r > 1100) break;
    }
    rays.push(Float64Array.from(q));
  }
  return { N, half, a, rays, mom };
}
/** The hotspot seen at emission time tEmit, or null (frozen calibrations return a constant). */
export type HotspotFn = (tEmit: number) => { rc: number; phiC: number; amp: number; Om: number } | null;
/** I_nu per pixel. With hs, each sample also carries the hotspot's boosted coefficients at its own Doppler factor, both
 *  emitters in one slab (spec 2026-10-04 mm hotspots 2.3); the sample's emission time is tObs - lightDelay * delay. */
export function flowImage(smp: FlowSamples, n0: number, rgCm: number, hs?: HotspotFn, tObs = 0, lightDelay = 1): Float64Array {
  const I = new Float64Array(smp.N * smp.N), zCut = HOTSPOT.cut * HOTSPOT.sigma;
  smp.rays.forEach((q, idx) => {
    const pt = smp.mom[2 * idx], pphi = smp.mom[2 * idx + 1];
    let Iv = 0, tau = 0;
    for (let k = 0; k < q.length; k += 6) {
      const r = q[k], th = q[k + 1], D = q[k + 2], dl = q[k + 3];
      let jE = 0, dTau = 0;
      if (D > 0) {
        const [j, al] = flowCoeffs(r, th, D * HOTFLOW.nu, n0);
        if (j > 0) { const ds = rgCm * D * dl; jE = (j / D ** 3) * ds; dTau = al * ds; }
      }
      if (hs && r < HOTSPOT_REACH && Math.abs(r * Math.cos(th)) < zCut) {
        const h = hs(tObs - lightDelay * (-q[k + 5] - 1000));
        const b = h ? h.amp * hotspotBoost(r, th, q[k + 4], h.rc, h.phiC) : 0;
        const Dh = b > 0 ? hotspotShift(r, th, pt, pphi, smp.a, h!.Om) : null;
        if (Dh !== null && Dh > 1e-6) {
          const [jh, ah] = flowCoeffs(r, th, Dh * HOTFLOW.nu, n0), dsh = rgCm * Dh * dl;
          jE += ((b * jh) / Dh ** 3) * dsh; dTau += b * ah * dsh;
        }
      }
      if (jE === 0 && dTau === 0) continue;
      const fac = dTau < 1e-4 ? 1 - 0.5 * dTau : (1 - Math.exp(-dTau)) / dTau;
      Iv += jE * fac * Math.exp(-tau); tau += dTau;
    }
    I[idx] = Iv;
  });
  return I;
}
/** Intensity-weighted centre (alpha, beta) in M (pixel convention of traceFlowSamples). */
export function imageCentroid(I: Float64Array, N: number, half: number): [number, number] {
  let w = 0, x = 0, y = 0;
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const v = I[j * N + i]; w += v; x += v * (-half + (2 * half * (i + 0.5)) / N); y += v * (half - (2 * half * (j + 0.5)) / N);
  }
  return [x / w, y / w];
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
