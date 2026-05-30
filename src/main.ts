import { Renderer } from "./render/gpu";
import type { UniformValues } from "./render/uniforms";

const canvas = document.getElementById("c") as HTMLCanvasElement;
const r = new Renderer();
const u: UniformValues = { resW: 0, resH: 0, a: 0.9, incl: 72 * Math.PI / 180, rObs: 1000,
  fovScale: 14, rIn: 0, rOut: 40, Tpeak: 1.5e7, exposure: 0, time: 0, frame: 0, reset: 1, maxSteps: 4000 };

await r.init(canvas);
function loop() {
  u.resW = r.width; u.resH = r.height;
  r.frame(u);
  requestAnimationFrame(loop);
}
loop();
