/** What WebGPU reports about the adapter it handed us (GPUAdapter.info; strings may be empty). */
export interface GpuInfo { vendor: string; architecture: string; description: string; }

export function describeGpu(i: GpuInfo): string {
  return `${i.vendor} ${i.architecture}`.trim() || "unknown";
}

/** True only when the adapter is recognisably an integrated GPU. On hybrid laptops Chrome can hand
 *  WebGPU the iGPU even when a discrete card is present (measured on the dev machine: Iris Xe for
 *  every powerPreference, 5-8x slower than the RTX). Unknown adapters are never flagged, so a
 *  missing description can only cost a missing warning, never a false one. */
export function isIntegratedGpu(i: GpuInfo): boolean {
  const v = i.vendor.toLowerCase(), arch = i.architecture.toLowerCase(), d = i.description.toLowerCase();
  if (v === "intel") return !arch.includes("hpg"); // Arc discrete parts report xe-hpg / xe2-hpg
  if (v === "amd" || v === "ati") return /radeon(\(tm\))? graphics|vega \d+ graphics/.test(d);
  return false;
}
