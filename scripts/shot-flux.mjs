// Writes flux-<preset>-NN.png: frames across one horizon-flux eruption (dev server on :5173), for the default
// view and M87*, with Motion at its maximum (100 M/s, one eruption cycle ~15 s), one frame per second for 20 s.
import { chromium } from "playwright-core";
const BASE = process.env.BASE || "http://localhost:5173";
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 760 } });
await page.goto(BASE + "/", { waitUntil: "load" });
await page.waitForFunction(() => /^\d+$/.test(document.getElementById("spp").textContent), null, { timeout: 30000 });
for (const id of ["default", "m87"]) {
  await page.selectOption("#preset", id);
  await page.evaluate(() => { const ts = document.getElementById("ts"); ts.value = "5"; ts.dispatchEvent(new Event("input")); });
  await page.waitForTimeout(3000);
  for (let k = 0; k < 20; k++) { await page.screenshot({ path: `flux-${id}-${String(k).padStart(2, "0")}.png` }); await page.waitForTimeout(1000); }
  console.log(`• flux-${id}-00..19.png`);
}
await browser.close();
