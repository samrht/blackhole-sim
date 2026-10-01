import { describe, it, expect } from "vitest";
import { PRESETS, CUSTOM_DEFAULT } from "../src/physics/presets";
import { peakTemperature, lambdaFromMdot } from "../src/physics/units";

const T_TABLE: Record<string, number> = { m87: 4.1e3, sgra: 1.1e4, cygx1: 6.1e6, grs1915: 1.1e7, gargantua: 5.8e3 };

describe("presets", () => {
  it("covers the five agreed objects with captions", () => {
    expect(PRESETS.map((p) => p.id)).toEqual(["m87", "sgra", "cygx1", "grs1915", "gargantua"]);
    for (const p of PRESETS) expect(p.caption.length).toBeGreaterThan(80);
  });
  it("every value lies inside the slider ranges", () => {
    for (const p of PRESETS) {
      expect(p.a).toBeGreaterThanOrEqual(0); expect(p.a).toBeLessThanOrEqual(0.998);
      expect(p.inclDeg).toBeGreaterThanOrEqual(1); expect(p.inclDeg).toBeLessThanOrEqual(89);
      expect(Math.log10(p.massSun)).toBeGreaterThanOrEqual(0); expect(Math.log10(p.massSun)).toBeLessThanOrEqual(10);
      expect(Math.log10(p.lambda)).toBeGreaterThanOrEqual(-10); expect(Math.log10(p.lambda)).toBeLessThanOrEqual(0);
    }
  });
  it("peak temperatures match the spec table within 10 % (Gargantua within 5 % of 5,800 K)", () => {
    for (const p of PRESETS) {
      const t = peakTemperature(p.massSun, p.a, p.lambda), want = T_TABLE[p.id];
      expect(Math.abs(t / want - 1)).toBeLessThan(p.id === "gargantua" ? 0.05 : 0.1);
    }
  });
  it("M87* and Sgr A* lambda are derived from the cited accretion rates", () => {
    const m87 = PRESETS.find((p) => p.id === "m87")!, sgra = PRESETS.find((p) => p.id === "sgra")!;
    expect(m87.lambda).toBeCloseTo(lambdaFromMdot(6.5e9, 0.9, 7.7e-4), 12);
    expect(sgra.lambda).toBeCloseTo(lambdaFromMdot(4.3e6, 0.94, 1e-8), 15);
  });
  it("the Custom default reproduces today's 30,000 K at a = 0.9", () => {
    expect(peakTemperature(CUSTOM_DEFAULT.massSun, 0.9, CUSTOM_DEFAULT.lambda)).toBeCloseTo(3e4, 3);
  });
});
