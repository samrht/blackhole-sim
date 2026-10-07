// Adds the per-pixel f32 conditioning (src/test/accuracy-sens.ts) to an existing src/test/accuracy-ref.json without
// rebuilding the references (build-accuracy-ref.ts computes the same field). Run: npx vite-node scripts/augment-accuracy-sens.ts
import { readFileSync, writeFileSync } from "node:fs";
import { JITTER } from "../src/render/cache-plan";
import { pixelSens } from "../src/test/accuracy-sens";
const fx = JSON.parse(readFileSync("src/test/accuracy-ref.json", "utf8"));
const sig = (x: number) => Number(x.toPrecision(10));
for (const sc of fx.scenes) {
  if (sc.flow) continue;
  let n = 0;
  sc.rays.forEach((row: any, k: number) => {
    if (!row.c || (row.f !== "disk" && row.f !== "escaped")) return;
    const [jx, jy] = JITTER[0], i = k % fx.N, j = Math.floor(k / fx.N);
    const al = (((i + 0.5 + jx) / fx.N) * 2 - 1) * fx.fov, be = -((((j + 0.5 + jy) / fx.N) * 2 - 1) * fx.fov);
    row.s = pixelSens(al, be, sc.a, sc.inclDeg, sc.rIn, row.J).map((x) => (Number.isFinite(x) ? sig(x) : 1e9)); n++;
  });
  console.log(`${sc.name}: ${n} pixels`);
}
writeFileSync("src/test/accuracy-ref.json", JSON.stringify(fx));
console.log("augmented src/test/accuracy-ref.json");
