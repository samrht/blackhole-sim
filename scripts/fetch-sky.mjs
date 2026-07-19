// scripts/fetch-sky.mjs
// Downloads the ESO/S. Brunier Milky-Way panorama (CC BY 4.0) and downscales it to a
// power-of-two 4096×2048 equirectangular JPG committed under public/sky/.
import { chromium } from "playwright-core";
import { writeFileSync, mkdirSync } from "fs";

const SRC = "https://cdn.eso.org/images/publicationjpg/eso0932a.jpg";
const OUT = "public/sky/milkyway-4k.jpg";
const W = 4096, H = 2048;

const resp = await fetch(SRC);
if (!resp.ok) throw new Error(`fetch failed: ${resp.status}`);
const srcB64 = Buffer.from(await resp.arrayBuffer()).toString("base64");

const browser = await chromium.launch({ channel: "chrome", headless: true,
  args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"] });
const page = await browser.newPage();
const { outB64, w, h } = await page.evaluate(async ({ b64, W, H }) => {
  const img = new Image();
  img.src = "data:image/jpeg;base64," + b64;
  await img.decode();
  const c = document.createElement("canvas"); c.width = W; c.height = H;
  const g = c.getContext("2d"); g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
  g.drawImage(img, 0, 0, W, H);
  return { outB64: c.toDataURL("image/jpeg", 0.85).split(",")[1], w: img.naturalWidth, h: img.naturalHeight };
}, { b64: srcB64, W, H });
await browser.close();

mkdirSync("public/sky", { recursive: true });
writeFileSync(OUT, Buffer.from(outB64, "base64"));
console.log(`source ${w}×${h} → ${OUT} (${W}×${H}), ${(Buffer.from(outB64, "base64").length / 1e6).toFixed(2)} MB`);
