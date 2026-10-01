import { describe, it, expect } from "vitest";
import { Renderer } from "../src/render/gpu";

// Node has no WebGPU: a minimal fake device that records buffers and their destruction.
(globalThis as unknown as { GPUBufferUsage: Record<string, number> }).GPUBufferUsage = { STORAGE: 128, COPY_DST: 8 };
type FakeBuf = { size: number; destroyed: boolean; destroy(): void };
function fakeDevice() {
  const made: FakeBuf[] = [];
  return {
    made,
    createBuffer: (d: { size: number }) => {
      const b: FakeBuf = { size: d.size, destroyed: false, destroy() { this.destroyed = true; } };
      made.push(b); return b;
    },
    queue: { writeBuffer: () => {} },
  };
}

describe("Renderer.uploadLUTs", () => {
  it("keeps an unchanged colour table and frees every buffer it replaces (spin ticks re-upload)", () => {
    const r = new Renderer(), dev = fakeDevice();
    (r as unknown as { device: unknown }).device = dev;
    const color = new Float32Array(4096 * 4); // the 64 KB visible LUT, uploaded on every spin tick
    r.uploadLUTs(new Float32Array(512), color);
    const temp1 = r.tempBuf as unknown as FakeBuf, color1 = r.colorBuf as unknown as FakeBuf;
    r.uploadLUTs(new Float32Array(512), color);
    expect(r.colorBuf as unknown).toBe(color1);  // same table: no new 64 KB buffer
    expect(temp1.destroyed).toBe(true);           // the replaced temperature LUT is freed
    expect(dev.made.length).toBe(3);
    r.uploadLUTs(new Float32Array(512), new Float32Array(16));
    expect(color1.destroyed).toBe(true);          // a different colour table replaces and frees it
  });
});
