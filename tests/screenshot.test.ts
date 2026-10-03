import { describe, it, expect } from "vitest";
import { toRGBA, screenshotName } from "../src/render/screenshot";

describe("screenshot export helpers", () => {
  it("bgra8unorm readback is swizzled to RGBA with opaque alpha", () => {
    const bgra = new Uint8Array([10, 20, 30, 0, 1, 2, 3, 77]);
    expect([...toRGBA(bgra, "bgra8unorm")]).toEqual([30, 20, 10, 255, 3, 2, 1, 255]);
  });
  it("rgba8unorm readback keeps channel order, alpha forced opaque", () => {
    const rgba = new Uint8Array([10, 20, 30, 0]);
    expect([...toRGBA(rgba, "rgba8unorm")]).toEqual([10, 20, 30, 255]);
  });
  it("does not modify the readback buffer", () => {
    const src = new Uint8Array([10, 20, 30, 0]);
    toRGBA(src, "bgra8unorm");
    expect([...src]).toEqual([10, 20, 30, 0]);
  });
  it("names the file after the scene and a filesystem-safe local timestamp", () => {
    const d = new Date(2026, 9, 3, 21, 4, 5); // local time
    expect(screenshotName("m87", 0.9, 17, d)).toBe("blackhole-m87-a0.90-i17-2026-10-03T21-04-05.png");
    expect(screenshotName("custom", -0.5, 72.4, d)).toBe("blackhole-custom-a-0.50-i72-2026-10-03T21-04-05.png");
    expect(screenshotName("default", 0.998, 85, d)).toBe("blackhole-default-a1.00-i85-2026-10-03T21-04-05.png");
  });
});
