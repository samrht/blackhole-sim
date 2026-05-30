import { describe, it, expect } from "vitest";
import { rhs, rk4, conserved, nullRadialMomentum } from "../src/physics/geodesic";

function makeEquatorialPhoton(r0: number, b: number, a: number): Float64Array {
  // E=1, L_z=b, equatorial (θ=π/2, p_θ=0), inward p_r<0
  const th = Math.PI / 2, pt = -1, pphi = b, pth = 0;
  const pr = -Math.abs(nullRadialMomentum(r0, th, a, pt, pphi, pth));
  return new Float64Array([0, r0, th, 0, pt, pr, pth, pphi]);
}

describe("geodesic integrator", () => {
  it("conserves E, L_z, Q, H along a photon trajectory", () => {
    const a = 0.8, s = makeEquatorialPhoton(20, 4.5, a);
    const c0 = conserved(s, a);
    let st = s;
    for (let i = 0; i < 2000; i++) st = rk4(st, a, -0.02);
    const c1 = conserved(st, a);
    expect(Math.abs(c1.E - c0.E)).toBeLessThan(1e-6);
    expect(Math.abs(c1.Lz - c0.Lz)).toBeLessThan(1e-6);
    expect(Math.abs(c1.Q - c0.Q)).toBeLessThan(1e-4);
    expect(Math.abs(c1.H)).toBeLessThan(1e-4); // null: H≈0
  });
  it("Schwarzschild capture threshold near b_c=sqrt(27)≈5.196", () => {
    const a = 0, rp = 2;
    const fate = (b: number) => {
      let st = makeEquatorialPhoton(50, b, a);
      for (let i = 0; i < 20000; i++) {
        st = rk4(st, a, +0.02); // pr<0 = inward requires positive dl for forward integration
        if (st[1] <= rp * 1.0001) return "captured";
        if (st[1] > 60) return "escaped";
      }
      return "stuck";
    };
    expect(fate(5.0)).toBe("captured");
    expect(fate(5.4)).toBe("escaped");
  });
  it("rhs keeps p_t and p_φ constant (Killing)", () => {
    const a = 0.5, s = makeEquatorialPhoton(15, 3, a);
    const d = rhs(s, a);
    expect(Math.abs(d[4])).toBeLessThan(1e-12); // dp_t/dλ
    expect(Math.abs(d[7])).toBeLessThan(1e-12); // dp_φ/dλ
  });
});
