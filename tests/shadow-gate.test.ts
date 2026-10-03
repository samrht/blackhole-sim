import { describe, it, expect } from "vitest";
import { analyticClassImage, compareShadow, AUDIT_UNRESOLVED } from "../src/test/shadow-gate";

const SHADOW = 0, DISK = 1, SKY = 2;
/** GPU-style audit words that agree with an analytic image everywhere. */
const wordsFrom = (an: Uint8Array) => Uint32Array.from(an, (c) => (c ? SHADOW : SKY));

describe("critical-curve gate (?shadow)", () => {
  const w = 160, h = 90, fov = 14;
  const an = analyticClassImage(w, h, fov, 0, Math.PI / 18, [-0.125, -0.375]);

  it("analytic image at a = 0 is the sqrt(27) disk: centre captured, corners escaped, area pi 27", () => {
    const at = (x: number, y: number) => an[y * w + x];
    expect(at(w >> 1, h >> 1)).toBe(1);
    expect(at(0, 0)).toBe(0); expect(at(w - 1, h - 1)).toBe(0);
    const pxArea = ((2 * fov * (w / h)) / w) * ((2 * fov) / h);
    const area = an.reduce((s, c) => s + c, 0) * pxArea;
    expect(Math.abs(area / (Math.PI * 27) - 1)).toBeLessThan(0.02);
  });
  it("a GPU image identical to the analytic one passes with area ratio 1", () => {
    const g = compareShadow(wordsFrom(an), an, w, h);
    expect(g).toMatchObject({ offBand: 0, inBand: 0, unresolved: 0, other: 0, pass: true });
    expect(g.areaRatio).toBe(1);
  });
  it("one wrong pixel far from the critical curve fails", () => {
    const words = wordsFrom(an); words[(h >> 1) * w + (w >> 1)] = SKY; // centre of the shadow
    const g = compareShadow(words, an, w, h);
    expect(g.offBand).toBe(1); expect(g.pass).toBe(false);
  });
  it("a wrong pixel on the critical curve is reported, not gated", () => {
    let k = -1;
    for (let i = 0; i < w * h && k < 0; i++) { const x = i % w, y = (i / w) | 0; if (x > 0 && an[i] === 1 && an[y * w + x - 1] === 0) k = i; }
    const words = wordsFrom(an); words[k] = SKY;
    const g = compareShadow(words, an, w, h);
    expect(g.inBand).toBe(1); expect(g.offBand).toBe(0); expect(g.pass).toBe(true);
  });
  it("a ray that ran out of steps away from the curve fails (the shader's classifier fallback would hide it)", () => {
    const words = wordsFrom(an); words[0] = SKY | AUDIT_UNRESOLVED;
    const g = compareShadow(words, an, w, h);
    expect(g.unresolved).toBe(1); expect(g.unresolvedOffBand).toBe(1); expect(g.pass).toBe(false);
  });
  it("any disk or other kind fails (the audit scene has no emitter)", () => {
    const words = wordsFrom(an); words[5] = DISK;
    expect(compareShadow(words, an, w, h)).toMatchObject({ other: 1, pass: false });
  });
  it("a uniformly larger traced shadow fails the 0.5 % area gate even when every flip sits on the curve", () => {
    const words = wordsFrom(an);
    for (let i = 0; i < w * h; i++) { const x = i % w, y = (i / w) | 0;
      if (an[i] === 0 && [[1,0],[-1,0],[0,1],[0,-1]].some(([dx,dy]) => an[(y+dy)*w + x+dx] === 1)) words[i] = SHADOW; }
    const g = compareShadow(words, an, w, h);
    expect(g.offBand).toBe(0); expect(g.areaRatio).toBeGreaterThan(1.005); expect(g.pass).toBe(false);
  });
});
