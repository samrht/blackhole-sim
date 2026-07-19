import { describe, it, expect } from "vitest";
import { packUniforms, UNIFORM_SIZE, type UniformValues } from "../src/render/uniforms";

describe("uniforms packing", () => {
  it("is 96 bytes and packs all fields (incl. Tier 2B jet) at the expected offsets", () => {
    expect(UNIFORM_SIZE).toBe(96);
    const u: UniformValues = {
      resW: 100, resH: 50, a: 0.9, incl: 1.2, rObs: 1000, fovScale: 14, rIn: 5, rOut: 40,
      Tpeak: 3e4, exposure: 1.6, time: 7, frame: 3, reset: 0, maxSteps: 1200,
      blend: 0.15, timeScale: 2, turbAmp: 0.6, breatheAmp: 0.1, nSpots: 4,
      jetStrength: 1.0, jetGamma: 5.0, jetLength: 60.0, jetKnots: 0.7,
    };
    const dv = new DataView(packUniforms(u));
    expect(dv.getFloat32(0, true)).toBeCloseTo(100);   // resW
    expect(dv.getFloat32(40, true)).toBeCloseTo(7);     // time (index 10)
    expect(dv.getUint32(44, true)).toBe(3);             // frame (index 11)
    expect(dv.getFloat32(56, true)).toBeCloseTo(0.15);  // blend (index 14)
    expect(dv.getUint32(72, true)).toBe(4);             // nSpots (index 18)
    expect(dv.getFloat32(76, true)).toBeCloseTo(1.0);   // jetStrength (index 19)
    expect(dv.getFloat32(80, true)).toBeCloseTo(5.0);   // jetGamma (index 20)
    expect(dv.getFloat32(84, true)).toBeCloseTo(60.0);  // jetLength (index 21)
    expect(dv.getFloat32(88, true)).toBeCloseTo(0.7);   // jetKnots (index 22)
  });
});
