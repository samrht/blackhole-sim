import { describe, it, expect } from "vitest";
import { packUniforms, UNIFORM_SIZE, type UniformValues } from "../src/render/uniforms";

describe("uniforms packing", () => {
  it("is 128 bytes and packs all fields (incl. display size) at the expected offsets", () => {
    expect(UNIFORM_SIZE).toBe(128);
    const u: UniformValues = {
      resW: 100, resH: 50, a: 0.9, incl: 1.2, rObs: 1000, fovScale: 14, rIn: 5, rOut: 40,
      Tpeak: 3e4, exposure: 1.6, time: 7, frame: 3, reset: 0, maxSteps: 1200,
      blend: 0.15, timeScale: 2, turbAmp: 0.6, breatheAmp: 0.1, nSpots: 4,
      jetStrength: 1.0, jetGamma: 5.0, jetLength: 60.0, jetKnots: 0.7,
      skyStrength: 0.6, outW: 200, outH: 100,
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
    expect(dv.getFloat32(92, true)).toBeCloseTo(0.6);   // skyStrength (index 23)
    expect(dv.getFloat32(96, true)).toBeCloseTo(200);   // outW (index 24)
    expect(dv.getFloat32(100, true)).toBeCloseTo(100);  // outH (index 25)
  });
  it("packs the geodesic-cache fields after outW/outH and defaults them to 0", () => {
    const base: UniformValues = {
      resW: 1, resH: 1, a: 0, incl: 0, rObs: 1000, fovScale: 14, rIn: 6, rOut: 40, Tpeak: 3e4, exposure: 1,
      time: 0, frame: 0, reset: 0, maxSteps: 1, blend: 1, timeScale: 1, turbAmp: 0, breatheAmp: 0, nSpots: 0,
      jetStrength: 0, jetGamma: 5, jetLength: 60, jetKnots: 0, skyStrength: 0, outW: 1, outH: 1,
    };
    const d0 = new DataView(packUniforms(base));
    for (const off of [104, 108, 112, 116]) expect(d0.getUint32(off, true)).toBe(0);
    const d1 = new DataView(packUniforms({ ...base, jitterMode: 1, setIndex: 3, rowStart: 64, rowEnd: 128 }));
    expect(d1.getUint32(104, true)).toBe(1);   // jitterMode (index 26)
    expect(d1.getUint32(108, true)).toBe(3);   // setIndex (index 27)
    expect(d1.getUint32(112, true)).toBe(64);  // rowStart (index 28)
    expect(d1.getUint32(116, true)).toBe(128); // rowEnd (index 29)
  });
});
