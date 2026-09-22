import { describe, it, expect } from "vitest";
import { describeGpu, isIntegratedGpu, type GpuInfo } from "../src/render/gpuinfo";

const g = (vendor: string, architecture = "", description = ""): GpuInfo => ({ vendor, architecture, description });

describe("gpuinfo", () => {
  it("flags Intel integrated graphics (what Chrome returned on the dev laptop)", () => {
    expect(isIntegratedGpu(g("intel", "gen-12lp"))).toBe(true);
    expect(isIntegratedGpu(g("intel", "xe-lpg"))).toBe(true);
  });
  it("does not flag discrete GPUs", () => {
    expect(isIntegratedGpu(g("nvidia", "ampere"))).toBe(false);
    expect(isIntegratedGpu(g("nvidia", "blackwell"))).toBe(false);
    expect(isIntegratedGpu(g("intel", "xe-hpg"))).toBe(false);      // Arc A-series
    expect(isIntegratedGpu(g("intel", "xe2-hpg"))).toBe(false);     // Arc B-series
    expect(isIntegratedGpu(g("amd", "rdna-3", "AMD Radeon RX 7600"))).toBe(false);
  });
  it("flags AMD integrated only when the description says so", () => {
    expect(isIntegratedGpu(g("amd", "rdna-3", "AMD Radeon(TM) Graphics"))).toBe(true);
    expect(isIntegratedGpu(g("amd", "gcn-5", "AMD Radeon Vega 8 Graphics"))).toBe(true);
    expect(isIntegratedGpu(g("amd", "rdna-3", ""))).toBe(false);    // unknown: no false alarm
  });
  it("never flags unknown adapters", () => {
    expect(isIntegratedGpu(g(""))).toBe(false);
  });
  it("describes the adapter compactly", () => {
    expect(describeGpu(g("nvidia", "ampere"))).toBe("nvidia ampere");
    expect(describeGpu(g("", ""))).toBe("unknown");
  });
});
