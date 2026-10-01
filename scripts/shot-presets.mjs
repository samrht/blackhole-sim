// Writes preset-<id>.png for every Object preset (dev server on :5173), for visual review.
import { chromium } from "playwright-core";
const BASE = process.env.BASE || "http://localhost:5173";
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 760 } });
await page.goto(BASE + "/", { waitUntil: "load" });
await page.waitForFunction(() => /^\d+$/.test(document.getElementById("spp").textContent), null, { timeout: 30000 });
const ids = await page.evaluate(() => [...document.getElementById("preset").options].map((o) => o.value));
for (const id of ids) {
  await page.selectOption("#preset", id);
  await page.waitForTimeout(7000);
  await page.screenshot({ path: `preset-${id}.png` });
  console.log(`• preset-${id}.png`);
}
await browser.close();
