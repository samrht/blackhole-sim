// The ?limit experiment's design, shared by the CPU arms (scripts/limit-cpu.ts) and the GPU arms (limit.browser.ts).
import { H_TOL, H_TOL_FAR } from "../physics/trace";

export interface LimitConfig { name: string; a: number; inclDeg: number; disk: boolean }
export const CONFIGS: LimitConfig[] = [
  { name: "edge-998", a: 0.998, inclDeg: 85, disk: true },
  { name: "edge-998-nodisk", a: 0.998, inclDeg: 85, disk: false },
  { name: "default-9", a: 0.9, inclDeg: 72, disk: true },
  { name: "schw", a: 0, inclDeg: 30, disk: true },
];
export const PX_M = (2 * 14) / 720;           // the app's pixel: 720 px over the 28 M field
export const REF_TOL = 1e-11;                 // the reference: DP5(4) in f64, checked against 1e-12 and against an independent formulation
export const F64_TOLS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9];
export const F32_TOLS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7];
export const RK4_SCALES = [1, 0.5, 0.25, 0.125]; // step-size scale of the pre-Mino integrator (1 = as it shipped)
// The monitor tightens with the method's order, so it never holds back the smaller step.
export const H_NEAR = (k: number) => H_TOL * k ** 4;
export const H_FAR = (k: number) => H_TOL_FAR * k ** 4;
export const rk4Budget = (k: number) => Math.ceil(4800 / k);
/** Named groups of rays reported per arm: by fate, and whether the ray passes close to the pole (xi ~ 0). */
export const AXIS_ALPHA = 0.03;
export function groupsOf(rays: [number, number][], fates: string[]): Record<string, boolean[]> {
  const axis = rays.map(([al]) => Math.abs(al) < AXIS_ALPHA);
  return { disk: fates.map((f, i) => f === "disk" && !axis[i]), sky: fates.map((f, i) => f === "escaped" && !axis[i]), axis };
}
export const BS3_TOLS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7];
export const BS3_BUDGET = 20000;
/** This module's copy of integrator-shared.wgsl with DP5(4) swapped for Bogacki-Shampine 3(2): same equations, error norm, drift check and landing. */
export function wgslWithBs3(raw: string): string {
  const src = raw.split(String.fromCharCode(13)).join("");
  const start = src.indexOf("fn minoTry(y: Mino, f0: Mino, h: f32, c: MinoRay) -> MinoTryOut {");
  const end = src.indexOf("\n}\n", start);
  if (start < 0 || end < 0 || !src.includes("pow(max(en, 1e-30), -0.2)")) throw new Error("integrator-shared.wgsl: minoTry not found");
  const body = `fn minoTry(y: Mino, f0: Mino, h: f32, c: MinoRay) -> MinoTryOut {
  let z = Mino(vec4<f32>(0.0), vec4<f32>(0.0));
  let k1 = f0;
  let k2 = minoRhs(mStage(y, h, k1, 1.0 / 2.0, z, 0.0, z, 0.0, z, 0.0, z, 0.0, z, 0.0), c);
  let k3 = minoRhs(mStage(y, h, k2, 3.0 / 4.0, z, 0.0, z, 0.0, z, 0.0, z, 0.0, z, 0.0), c);
  let y1 = mStage(y, h, k1, 2.0 / 9.0, k2, 1.0 / 3.0, k3, 4.0 / 9.0, z, 0.0, z, 0.0, z, 0.0);
  let k7 = minoRhs(y1, c);
  let e = mStage(z, h, k1, -5.0 / 72.0, k2, 1.0 / 12.0, k3, 1.0 / 9.0, k7, -1.0 / 8.0, z, 0.0, z, 0.0);
  let en = select(3e38, max(minoErrNorm(y, y1, e, MINO_TOL), minoDriftNorm(y, y1, k7, c, MINO_TOL)), y1.q.y > 0.0);
  return MinoTryOut(y1, k7, en);`;
  return (src.slice(0, start) + body + src.slice(end)).replace("pow(max(en, 1e-30), -0.2)", "pow(max(en, 1e-30), -0.3333333)");
}
