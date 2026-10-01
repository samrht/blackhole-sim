import { describe, it, expect } from "vitest";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

// Shared-fragment discipline (camera, shadow, integrator, emission): every WGSL function lives in
// exactly one file, and the renderer and the ?parity route prepend the same fragment, so ?parity
// verifies the shipped bytes. A second definition anywhere is a copy that can silently diverge.
describe("WGSL shared fragments", () => {
  it("no function is defined in more than one shader file", () => {
    const dir = join(__dirname, "../src/render");
    const where = new Map<string, string[]>();
    for (const f of readdirSync(dir).filter((n) => n.endsWith(".wgsl"))) {
      for (const m of readFileSync(join(dir, f), "utf8").matchAll(/^\s*fn\s+(\w+)\s*\(/gm)) {
        if (m[1] === "main") continue; // every entry-point file has its own main
        where.set(m[1], [...(where.get(m[1]) ?? []), f]);
      }
    }
    const dupes = [...where].filter(([, fs]) => fs.length > 1).map(([fn, fs]) => `${fn}: ${fs.join(", ")}`);
    expect(dupes).toEqual([]);
  });
});
