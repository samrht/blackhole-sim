import { describe, it, expect } from "vitest";
import { PRESETS, CUSTOM_DEFAULT } from "../src/physics/presets";
import { peakTemperature, lambdaFromMdot, eddingtonLuminosity } from "../src/physics/units";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const T_TABLE: Record<string, number> = { m87: 4.1e3, sgra: 1.1e4, cygx1: 6.1e6, grs1915: 1.1e7, gargantua: 5.8e3 };

describe("presets", () => {
  it("covers the seven agreed objects with captions", () => {
    expect(PRESETS.map((p) => p.id)).toEqual(["m87", "sgra", "cygx1", "grs1915", "gargantua", "ton618", "phoenixa"]);
    for (const p of PRESETS) expect(p.caption.length).toBeGreaterThan(80);
  });
  it("every value lies inside the slider ranges", () => {
    for (const p of PRESETS) {
      expect(p.a).toBeGreaterThanOrEqual(0); expect(p.a).toBeLessThanOrEqual(0.998);
      expect(p.inclDeg).toBeGreaterThanOrEqual(1); expect(p.inclDeg).toBeLessThanOrEqual(89);
      expect(Math.log10(p.massSun)).toBeGreaterThanOrEqual(0); expect(Math.log10(p.massSun)).toBeLessThanOrEqual(11.5);
      expect(Math.log10(p.lambda)).toBeGreaterThanOrEqual(-10); expect(Math.log10(p.lambda)).toBeLessThanOrEqual(0);
    }
  });
  it("peak temperatures match the spec table within 10 % (Gargantua within 5 % of 5,800 K)", () => {
    for (const p of PRESETS.filter((q) => T_TABLE[q.id] !== undefined)) {
      const t = peakTemperature(p.massSun, p.a, p.lambda), want = T_TABLE[p.id];
      expect(Math.abs(t / want - 1)).toBeLessThan(p.id === "gargantua" ? 0.05 : 0.1);
    }
  });
  it("M87* and Sgr A* lambda are derived from the cited accretion rates", () => {
    const m87 = PRESETS.find((p) => p.id === "m87")!, sgra = PRESETS.find((p) => p.id === "sgra")!;
    expect(m87.lambda).toBeCloseTo(lambdaFromMdot(6.5e9, 0.9, 7.7e-4), 12);
    expect(sgra.lambda).toBeCloseTo(lambdaFromMdot(4.3e6, 0.94, 1e-8), 15);
  });
  it("jets: on for M87*, Cygnus X-1, GRS 1915+105; off for Sgr A* and Gargantua", () => {
    const on = Object.fromEntries(PRESETS.map((p) => [p.id, p.jet]));
    expect(on).toEqual({ m87: true, sgra: false, cygx1: true, grs1915: true, gargantua: false, ton618: true, phoenixa: true });
  });
  it("X-ray-binary captions describe the energy-conserving jet (no overestimate caveat)", () => {
    for (const id of ["cygx1", "grs1915"]) {
      const c = PRESETS.find((p) => p.id === id)!.caption;
      expect(c).not.toMatch(/overestimate/); expect(c).toMatch(/fraction η of its jet power/);
    }
  });
  it("TON 618 and Phoenix A: cited masses, lambda = L_bol / L_Edd, thin disks (lambda > 0.01)", () => {
    const ton = PRESETS.find((p) => p.id === "ton618")!, phx = PRESETS.find((p) => p.id === "phoenixa")!;
    expect(ton.massSun).toBe(6.6e10);                                          // Shemmer et al. 2004 (H-beta)
    expect(ton.lambda).toBeCloseTo(4e40 / eddingtonLuminosity(6.6e10), 12);   // L_bol 4e40 W
    expect(phx.massSun).toBe(1e11);                                            // Brockamp et al. 2016 (model)
    expect(phx.lambda).toBeCloseTo(6e40 / eddingtonLuminosity(1e11), 12);     // Ueda et al. 2013: 6e47 erg/s
    for (const p of [ton, phx]) { expect(p.lambda).toBeGreaterThan(0.01); expect(p.caption).toMatch(/not measured/); }
  });
  it("the mass slider reaches 10^11.5 M_sun (index.html)", () => {
    const html = readFileSync(join(__dirname, "../index.html"), "utf8");
    expect(html).toMatch(/<input id="mass" type="range" min="0" max="11.5"/);
  });
  it("the Custom default reproduces today's 30,000 K at a = 0.9", () => {
    expect(peakTemperature(CUSTOM_DEFAULT.massSun, 0.9, CUSTOM_DEFAULT.lambda)).toBeCloseTo(3e4, 3);
  });
});
