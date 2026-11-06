import { describe, it, expect, afterEach } from "vitest";
import { minoInit, minoRay, minoRhs, minoStep, minoTrace, minoTry, setMinoMethod } from "../src/physics/trace-mino";

const a = 0.9, i = (72 * Math.PI) / 180, al = 6, be = 3, xi = -al * Math.sin(i);
const eta = be * be + (xi * xi * Math.cos(i) ** 2) / Math.sin(i) ** 2 - a * a * Math.cos(i) ** 2;
const c = minoRay(a, xi, eta, 1000);
// A state at r ~ 10 on a real ray: strong field, and h = 0.02 is well into the asymptotic range there (farther out, or at
// larger h, the measured DP5 rate reads above 7 because the leading coefficient is small).
function midState(): Float64Array {
  let y = minoInit(1000, i, be, c), h = 50 / 1e6;
  while (1 / y[1] > 10) { const s = minoStep(y, h, c, 1e-7); y = s.y; h = s.hNext; }
  return y;
}
/** Local error of one step of size h against many tiny DP5 steps, on theta' (a state component of order 1). */
function localError(y: Float64Array, h: number, method: "dp5" | "bs3"): number {
  setMinoMethod("dp5"); let x = y; const n = 400;
  for (let k = 0; k < n; k++) x = minoTry(x, h / n, c, minoRhs(x, c), 1, 1).y1;
  setMinoMethod(method);
  return Math.abs(minoTry(y, h, c, minoRhs(y, c), 1, 1).y1[6] - x[6]);
}
afterEach(() => setMinoMethod("dp5"));

describe("the integrator's order", () => {
  it("DP5 has local error h^6 and BS3 h^4, in the same equations", () => {
    setMinoMethod("dp5"); const y = midState(), h = 0.02;
    const p = (m: "dp5" | "bs3") => Math.log2(localError(y, h, m) / localError(y, h / 2, m));
    expect(p("dp5")).toBeGreaterThan(5.5); expect(p("dp5")).toBeLessThan(7.2);
    expect(p("bs3")).toBeGreaterThan(3.7); expect(p("bs3")).toBeLessThan(4.3);
  });
  it("the default method is unchanged, and BS3 takes more steps for the same tolerance", () => {
    const o = { rIn: 2.32, rOut: 40, rObs: 1000, tol: 1e-6, maxSteps: 20000 };
    const d = minoTrace(al, be, a, 72, o); setMinoMethod("bs3");
    const b = minoTrace(al, be, a, 72, o); setMinoMethod("dp5");
    const again = minoTrace(al, be, a, 72, o);
    expect(again.rHit).toBe(d.rHit); expect(again.steps).toBe(d.steps);
    expect(b.fate).toBe(d.fate); expect(b.steps).toBeGreaterThan(2 * d.steps);
    expect(Math.abs(b.rHit! - d.rHit!)).toBeLessThan(1e-3);
  });
});
