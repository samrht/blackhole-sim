import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { sinCosP, SINCOS_COEFFS } from "../src/render/sincos";

// WGSL's built-in sin / cos may be off by 2^-11 absolute (the Intel iGPU's cos(1 deg) is 2e-5 off), which near a pole is
// a 13 % error in sin^2 theta. sinCosP (camera-shared.wgsl, mirrored here) is Cephes sinf / cosf with Cody-Waite
// reduction by pi / 2: ~1 ulp.
describe("precise sin / cos (integrator-shared.wgsl sinCosP)", () => {
  it("matches Math.sin / Math.cos to 2e-7 absolute over [-10, 10] (f32-rounded arithmetic)", () => {
    let worst = 0;
    for (let k = 0; k <= 20000; k++) {
      const x = Math.fround(-10 + k * 0.001), [s, c] = sinCosP(x, true);
      worst = Math.max(worst, Math.abs(s - Math.sin(x)), Math.abs(c - Math.cos(x)));
    }
    expect(worst).toBeLessThan(2e-7);
  });
  it("is accurate near 0 and pi relative to the value (sin 1 deg, 1 - cos 1 deg)", () => {
    const x = Math.fround(Math.PI / 180), [s, c] = sinCosP(x, true);
    expect(Math.abs(s / Math.sin(x) - 1)).toBeLessThan(3e-7);
    expect(Math.abs((1 - c) / (1 - Math.cos(x)) - 1)).toBeLessThan(1e-3); // 1 - cos ~ 1.5e-4: limited by f32 cos itself
  });
  it("the WGSL copy carries the same coefficients", () => {
    const W = readFileSync(join(__dirname, "../src/render/integrator-shared.wgsl"), "utf8");
    const body = W.slice(W.indexOf("fn sinCosP"), W.indexOf("// ---- Mino-time integrator"));
    const nums = (body.match(/-?\d+\.\d+(e-?\d+)?/g) ?? []).map(Number);
    for (const c of SINCOS_COEFFS) expect(nums.some((v) => Math.abs(v - c) <= 1e-12 * Math.abs(c))).toBe(true);
  });
});
