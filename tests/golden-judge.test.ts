import { describe, it, expect } from "vitest";
import { judgeGolden, countNonFinite } from "../src/test/golden-judge";

describe("golden verdict", () => {
  const want = { adapter: "intel", hashes: { a: "1", b: "2" } };
  it("PASS when every recorded hash matches, SKIP on another adapter, FAIL on a mismatch", () => {
    expect(judgeGolden(want, { adapter: "intel", hashes: { a: "1", b: "2", c: "9" } })).toBe("PASS");
    expect(judgeGolden(want, { adapter: "nvidia", hashes: { a: "7", b: "8" } })).toBe("SKIP");
    expect(judgeGolden(want, { adapter: "intel", hashes: { a: "1", b: "3" } })).toBe("FAIL");
  });
  it("FAILs on any non-finite pixel, on any adapter (a NaN image still hashes)", () => {
    expect(judgeGolden(want, { adapter: "intel", hashes: { a: "1", b: "2" }, nonFinite: { b: 1 } })).toBe("FAIL");
    expect(judgeGolden(want, { adapter: "nvidia", hashes: { a: "7" }, nonFinite: { a: 3 } })).toBe("FAIL");
    expect(judgeGolden(want, { adapter: "intel", hashes: { a: "1", b: "2" }, nonFinite: { a: 0, b: 0 } })).toBe("PASS");
  });
  it("countNonFinite counts NaN and both infinities", () => {
    expect(countNonFinite(new Float32Array([0, 1, NaN, Infinity, -Infinity, 5]))).toBe(3);
    expect(countNonFinite(new Float32Array([0, 1e30, -2]))).toBe(0);
  });
});
