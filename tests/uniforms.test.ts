import { describe, it, expect } from "vitest";
import { packUniforms, UNIFORM_SIZE, type UniformValues } from "../src/render/uniforms";

describe("uniforms packing", () => {
  it("is 80 bytes and packs new fields at the expected offsets", () => {
    expect(UNIFORM_SIZE).toBe(80);
    const u: UniformValues = {
      resW: 100, resH: 50, a: 0.9, incl: 1.2, rObs: 1000, fovScale: 14, rIn: 5, rOut: 40,
      Tpeak: 3e4, exposure: 1.6, time: 7, frame: 3, reset: 0, maxSteps: 1200,
      blend: 0.15, timeScale: 2, turbAmp: 0.6, breatheAmp: 0.1, nSpots: 4,
    };
    const dv = new DataView(packUniforms(u));
    expect(dv.getFloat32(0, true)).toBeCloseTo(100);   // resW
    expect(dv.getFloat32(40, true)).toBeCloseTo(7);     // time (index 10)
    expect(dv.getUint32(44, true)).toBe(3);             // frame (index 11)
    expect(dv.getFloat32(56, true)).toBeCloseTo(0.15);  // blend (index 14)
    expect(dv.getFloat32(60, true)).toBeCloseTo(2);     // timeScale (index 15)
    expect(dv.getFloat32(64, true)).toBeCloseTo(0.6);   // turbAmp (index 16)
    expect(dv.getFloat32(68, true)).toBeCloseTo(0.1);   // breatheAmp (index 17)
    expect(dv.getUint32(72, true)).toBe(4);             // nSpots (index 18)
  });
});
