// Per-pixel float32 input conditioning for ?accuracy (spec 2026-10-07 mino integrator §5; ledger ruling "f32 conditioning
// floor"). A pixel's ray enters the renderer as f32 (alpha, beta); near the critical curve the result is chaotic in their
// last bit (Schwarzschild i = 30: a 1-ulp nudge moved rHit by 2.6e-4 to 3.9e-4 relative in the pre-Mino AND the Mino GPU
// renderer alike). No f32 renderer can be held to better than that, so the gate's floor for a pixel is the larger of the
// stated floor and how far the converged answer moves under the four 1-ulp nudges (alpha +-, beta +-), measured here in
// f64 with a tight Mino trace (tol 1e-10, verified against the reference to ~1e-7).
import { minoTrace } from "../physics/trace-mino";
import { skyDirCPU, wrapAngle, pxError } from "../physics/trace-reference";

export type Sens = [number, number, number, number]; // r (rel), phi (rad), delay (rel to max(1, |d|)), sky (px)
const ULP = 2 ** -23;
export function pixelSens(al: number, be: number, a: number, inclDeg: number, rIn: number, J: number[][] | undefined): Sens {
  const o = { rIn, rOut: 40, rObs: 1000, tol: 1e-10 };
  const base = minoTrace(al, be, a, inclDeg, o);
  const s: Sens = [0, 0, 0, 0];
  const dir = (m: ReturnType<typeof minoTrace>) => skyDirCPU(m.s, a);
  for (const [da, db] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
    const m = minoTrace(al * (1 + da * ULP), be * (1 + db * ULP), a, inclDeg, o);
    if (m.fate !== base.fate) return [Infinity, Infinity, Infinity, Infinity]; // a fate edge: any f32 renderer may flip it
    if (m.fate === "disk") {
      s[0] = Math.max(s[0], Math.abs(m.rHit! / base.rHit! - 1)); s[1] = Math.max(s[1], wrapAngle(m.phiHit! - base.phiHit!));
      const d0 = -base.tHit! - 1000, d1 = -m.tHit! - 1000; s[2] = Math.max(s[2], Math.abs(d1 - d0) / Math.max(1, Math.abs(d0)));
    } else if (m.fate === "escaped") {
      const e = pxError(J, dir(m), dir(base)); if (Number.isFinite(e)) s[3] = Math.max(s[3], e);
    }
  }
  return s;
}
