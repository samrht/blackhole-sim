// ?imgerr&view=<name>: does any of the accuracy limit reach the image? For each integrator arm, every pixel's ray is traced, its result is written
// into the renderer's cache entries, and the renderer's own shade pass and present pass make the picture, so arms differ in geometry alone.
// Reference: double-single at tol 1e-9 (checked against f64 on sampled pixels), stored in f32 as the renderer stores everything.
// Controls: the renderer's own build pass against this route's plain f32 kernel.
import { Renderer } from "../render/gpu";
import limitWGSL from "../render/limit.wgsl?raw";
import shadowSharedWGSL from "../render/shadow-shared.wgsl?raw";
import cameraSharedWGSL from "../render/camera-shared.wgsl?raw";
import integratorSharedWGSL from "../render/integrator-shared.wgsl?raw";
import { BM_NONE, JITTER } from "../render/cache-plan";
import { toRGBA } from "../render/screenshot";
import { minoTrace } from "../physics/trace-mino";
import { skyDirCPU } from "../physics/trace-reference";
import { prepareScene, sceneUniforms, type Scene } from "./scenes";
import { dsDispatch, dsModule, dsObs } from "./limit-ds.browser";
import { compareDisplay, compareLinear, IMAGE_VIEWS, pixelRays } from "./image-metrics";
import { H_TOL, H_TOL_FAR } from "../physics/trace";

const F = Math.fround;
const plainCode = (tol: number) => shadowSharedWGSL + cameraSharedWGSL + integratorSharedWGSL.replace(/const MINO_TOL = 1e-5;/, `const MINO_TOL = ${tol.toExponential()};`) + limitWGSL;
type Arm = { name: string; kind: "plain" | "comp" | "rk4" | "ds"; tol: number };
const ARMS: Arm[] = [
  { name: "f32 1e-5 (shipped)", kind: "plain", tol: 1e-5 }, { name: "f32 1e-6", kind: "plain", tol: 1e-6 }, { name: "f32 1e-7", kind: "plain", tol: 1e-7 },
  { name: "compensated 1e-5", kind: "comp", tol: 1e-5 }, { name: "compensated 1e-6", kind: "comp", tol: 1e-6 }, { name: "compensated 1e-7", kind: "comp", tol: 1e-7 },
  { name: "double-single 1e-5", kind: "ds", tol: 1e-5 }, { name: "double-single 1e-7", kind: "ds", tol: 1e-7 },
  { name: "RK4 (pre-Mino)", kind: "rk4", tol: 0 },
];

async function compile(device: GPUDevice, code: string, entry: string): Promise<GPUComputePipeline> {
  const mod = device.createShaderModule({ code });
  for (const m of (await mod.getCompilationInfo()).messages) if (m.type === "error") throw new Error(`wgsl:${m.lineNum}: ${m.message}`);
  return device.createComputePipelineAsync({ layout: "auto", compute: { module: mod, entryPoint: entry } });
}
/** Plain, compensated or RK4 f32: entries from limit.wgsl's RayOut (word, steps, attempts, r, phi, delay, dir). */
async function traceF32(device: GPUDevice, arm: Arm, rays: Float32Array, n: number, p: { a: number; incl: number; rIn: number; rOut: number }): Promise<{ raw: ArrayBuffer; ms: number }> {
  const entry = arm.kind === "plain" ? "mino" : arm.kind === "comp" ? "minoComp" : "rk4arm", pipe = await compile(device, plainCode(arm.tol || 1e-5), entry);
  const rayB = device.createBuffer({ size: rays.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(rayB, 0, rays as unknown as BufferSource);
  const size = n * 64, outB = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }), uni = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: uni } }, { binding: 1, resource: { buffer: rayB } }, { binding: 2, resource: { buffer: outB } }] });
  const ab = new ArrayBuffer(48), f = new Float32Array(ab), w = new Uint32Array(ab), chunk = arm.kind === "rk4" ? 8192 : 32768;
  f.set([p.a, p.incl, 1000, p.rIn, p.rOut, 1, H_TOL, H_TOL_FAR]); w[8] = 4800; w[11] = arm.kind === "comp" ? 1 : 0;
  const t0 = performance.now();
  for (let first = 0; first < n; first += chunk) {
    w[9] = first; w[10] = Math.min(chunk, n - first); device.queue.writeBuffer(uni, 0, ab);
    const enc = device.createCommandEncoder(), cp = enc.beginComputePass();
    cp.setPipeline(pipe); cp.setBindGroup(0, bg); cp.dispatchWorkgroups(Math.ceil(w[10] / 64)); cp.end(); device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  const ms = performance.now() - t0, rb = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(outB, 0, rb, 0, size); device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ); const raw = rb.getMappedRange().slice(0); rb.unmap(); rb.destroy(); rayB.destroy(); outB.destroy(); uni.destroy();
  return { raw, ms };
}
const entriesFromF32 = (raw: ArrayBuffer, n: number): ArrayBuffer => {
  const U = new Uint32Array(raw), R = new Float32Array(raw), out = new ArrayBuffer(n * 16), eu = new Uint32Array(out), ef = new Float32Array(out);
  for (let i = 0; i < n; i++) {
    const o = i * 16, word = U[o], kind = word & 4 ? 0 : word & 3;
    eu[i * 4] = kind | (BM_NONE << 2);
    if (kind === 1) { ef[i * 4 + 1] = R[o + 3]; ef[i * 4 + 2] = R[o + 4]; ef[i * 4 + 3] = R[o + 5]; } else if (kind === 2) { ef[i * 4 + 1] = R[o + 6]; ef[i * 4 + 2] = R[o + 7]; ef[i * 4 + 3] = R[o + 8]; }
  }
  return out;
};
function entriesFromDs(raw: ArrayBuffer, rays: Float32Array, n: number, a: number, incl: number): ArrayBuffer {
  const out = new ArrayBuffer(n * 16), eu = new Uint32Array(out), ef = new Float32Array(out), si = Math.sin(incl), ci = Math.cos(incl);
  for (let i = 0; i < n; i++) {
    const al = rays[i * 2], be = rays[i * 2 + 1], xi = -al * si, eta = be * be + (xi * xi * ci * ci) / Math.max(si * si, 1e-8) - a * a * ci * ci, g = dsObs(raw, i, { a, xi, eta });
    const kind = g.f === "disk" ? 1 : g.f === "escaped" ? 2 : 0;
    eu[i * 4] = kind | (BM_NONE << 2);
    if (kind === 1) { ef[i * 4 + 1] = F(g.r!); ef[i * 4 + 2] = F(g.p!); ef[i * 4 + 3] = F(-g.t! - 1000); } else if (kind === 2) { ef[i * 4 + 1] = F(g.dir![0]); ef[i * 4 + 2] = F(g.dir![1]); ef[i * 4 + 3] = F(g.dir![2]); }
  }
  return out;
}

export async function runImageError(canvas: HTMLCanvasElement): Promise<{ report: unknown; images: Record<string, string> }> {
  const viewName = new URLSearchParams(location.search).get("view") ?? "default", view = IMAGE_VIEWS.find((v) => v.name === viewName);
  if (!view) throw new Error(`view? ${IMAGE_VIEWS.map((v) => v.name).join(" ")}`);
  const r = new Renderer(); await r.init(canvas); r.renderBloom = false;
  const device = r.device, w = r.width, h = r.height, n = w * h, incl = (view.inclDeg * Math.PI) / 180;
  const scene: Scene = { name: view.name, a: view.a, inclDeg: view.inclDeg, time: view.time, frame: 0, jetStrength: 0, skyStrength: 0, lightDelay: 1 };
  const rIn = prepareScene(r, scene), rOut = 40;
  const u = sceneUniforms(r, scene, rIn, { fovScale: view.fov, panX: view.panX, panY: view.panY, setIndex: 0 });
  const rays = pixelRays(w, h, view.fov, view.panX, view.panY, JITTER[0]);
  const render = async (entries: ArrayBuffer | null) => {
    if (entries) device.queue.writeBuffer(r.entryBufs[0], 0, entries);
    r.frame(u, { cachedSet: 0 }); await device.queue.onSubmittedWorkDone();
    const linear = new Float32Array(await r.readbackAccum()), disp = await r.readbackDisplay();
    return { linear, display: toRGBA(disp.data, disp.format), dw: disp.w, dh: disp.h };
  };
  const kinds = (e: ArrayBuffer) => new Uint32Array(e).filter((_, i) => i % 4 === 0).map((x) => x & 3);

  // ---- The reference ----
  const dsRun = async (tol: number, chunk: number) => { const pipe = await compile(device, dsModule(tol), "trace"); return dsDispatch(device, pipe, rays, n, { a: view.a, inclDeg: view.inclDeg, rIn, rOut }, chunk); };
  const refRun = await dsRun(1e-9, 1024), refEntries = entriesFromDs(refRun.raw, rays, n, view.a, incl), ref = await render(refEntries), refKinds = kinds(refEntries);
  console.log(`reference traced in ${(refRun.ms / 1000).toFixed(1)} s`);
  // Sampled against f64 at tol 1e-11: the largest difference in the stored quantities.
  let check = { rays: 0, fateMismatch: 0, rRelMax: 0, phiMax: 0, dirMax: 0 };
  for (let k = 0; k < 3000; k++) {
    const i = Math.floor((k / 3000) * n), al = rays[i * 2], be = rays[i * 2 + 1], m = minoTrace(al, be, view.a, view.inclDeg, { rIn, rOut, rObs: 1000, tol: 1e-11 }), si = Math.sin(incl), ci = Math.cos(incl), xi = -al * si;
    const g = dsObs(refRun.raw, i, { a: view.a, xi, eta: be * be + (xi * xi * ci * ci) / Math.max(si * si, 1e-8) - view.a * view.a * ci * ci });
    check.rays++;
    if (g.f !== m.fate) { check.fateMismatch++; continue; }
    if (g.f === "disk") { check.rRelMax = Math.max(check.rRelMax, Math.abs(g.r! / m.rHit! - 1)); check.phiMax = Math.max(check.phiMax, Math.abs(Math.atan2(Math.sin(g.p! - m.phiHit!), Math.cos(g.p! - m.phiHit!)))); }
    else if (g.f === "escaped") { const q = skyDirCPU(m.s, view.a); check.dirMax = Math.max(check.dirMax, Math.hypot(g.dir![0] - q[0], g.dir![1] - q[1], g.dir![2] - q[2])); }
  }
  console.log(`reference against f64: ${JSON.stringify(check)}`);

  // ---- The renderer's own build pass, as a control on the route's plain f32 kernel ----
  r.frame(u, { build: { ...u, setIndex: 0, rowStart: 0, rowEnd: h } }); await device.queue.onSubmittedWorkDone();
  const built = (await r.readbackEntries(0)).slice(0, n * 4), builtBuf = built.buffer.slice(built.byteOffset, built.byteOffset + n * 16);
  const realImg = await render(null);

  const arms: Record<string, unknown> = {}, images: Record<string, string> = {};
  const toPng = (data: Uint8Array | Uint8ClampedArray) => { const c = document.createElement("canvas"); c.width = w; c.height = h; const g = c.getContext("2d")!; const im = g.createImageData(w, h); im.data.set(data); g.putImageData(im, 0, 0); return c.toDataURL("image/png"); };
  const diffPng = (a: Uint8Array | Uint8ClampedArray, b: Uint8Array | Uint8ClampedArray, gain: number) => { const o = new Uint8Array(a.length); for (let i = 0; i < a.length / 4; i++) { let m = 0; for (let c = 0; c < 3; c++) m = Math.max(m, Math.abs(a[i * 4 + c] - b[i * 4 + c])); const v = Math.min(255, m * gain); o[i * 4] = 255 - v; o[i * 4 + 1] = 255 - v; o[i * 4 + 2] = 255 - v; o[i * 4 + 3] = 255; } return toPng(o); };
  const score = (img: Awaited<ReturnType<typeof render>>, e: ArrayBuffer) => {
    const k = kinds(e); let flips = 0; for (let i = 0; i < n; i++) if (k[i] !== refKinds[i]) flips++;
    const disp = compareDisplay(ref.display, img.display);
    return { flips, display: disp, all: compareLinear(ref.linear, img.linear), disk: compareLinear(ref.linear, img.linear, (i) => refKinds[i] === 1 && k[i] === 1), sky: compareLinear(ref.linear, img.linear, (i) => refKinds[i] === 2 && k[i] === 2) };
  };
  images.reference = toPng(ref.display);
  arms["renderer's own build"] = score(realImg, builtBuf);

  for (const arm of ARMS) {
    let entries: ArrayBuffer, ms: number;
    if (arm.kind === "ds") { const q = await dsRun(arm.tol, 1024); ms = q.ms; entries = entriesFromDs(q.raw, rays, n, view.a, incl); }
    else { const q = await traceF32(device, arm, rays, n, { a: view.a, incl, rIn, rOut }); ms = q.ms; entries = entriesFromF32(q.raw, n); }
    const img = await render(entries), s = score(img, entries) as Record<string, unknown>;
    if (arm.name === "f32 1e-5 (shipped)") {
      // Control: this kernel's entries against the renderer's own, bit by bit where the kind agrees.
      const a = new Float32Array(entries), b = new Float32Array(builtBuf), au = new Uint32Array(entries), bu = new Uint32Array(builtBuf); let same = 0, kindDiff = 0, rMax = 0;
      for (let i = 0; i < n; i++) {
        if ((au[i * 4] & 3) !== (bu[i * 4] & 3)) { kindDiff++; continue; }
        if (a[i * 4 + 1] === b[i * 4 + 1] && a[i * 4 + 2] === b[i * 4 + 2] && a[i * 4 + 3] === b[i * 4 + 3]) same++;
        else if ((au[i * 4] & 3) === 1) rMax = Math.max(rMax, Math.abs(a[i * 4 + 1] / b[i * 4 + 1] - 1));
      }
      s.controlAgainstBuild = { identicalPixels: same, kindDiff, pixels: n, diskRRelMax: rMax };
      images["shipped"] = toPng(img.display); images["shipped-diff"] = diffPng(ref.display, img.display, 32);
    }
    if (arm.name === "compensated 1e-6") images["compensated-diff"] = diffPng(ref.display, img.display, 32);
    arms[arm.name] = { ...s, ms };
    console.log(`arm ${arm.name}`);
  }
  const info = r.adapterInfo;
  return { report: { view, w, h, adapter: `${info.vendor} ${info.architecture}`.trim(), rIn, referenceMs: refRun.ms, referenceCheck: check, arms }, images };
}
