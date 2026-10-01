import { packUniforms, UniformValues, UNIFORM_SIZE } from "./uniforms";
import type { GpuInfo } from "./gpuinfo";
import presentWGSL from "./present.wgsl?raw";
import raytraceWGSL from "./raytrace.wgsl?raw";
import shadowSharedWGSL from "./shadow-shared.wgsl?raw";
import cameraSharedWGSL from "./camera-shared.wgsl?raw";
import integratorSharedWGSL from "./integrator-shared.wgsl?raw";
import emissionSharedWGSL from "./emission-shared.wgsl?raw";
import bloomWGSL from "./bloom.wgsl?raw";
import { planCache, BOOKMARK_BYTES, type CachePlan } from "./cache-plan";

/** Per-frame geodesic-cache work. `build` traces one row slice of a jitter set (full resolution,
 *  its own uniform buffer); `cachedSet` shades from that set instead of tracing (`main`). */
export interface FrameOpts { cachedSet?: number; build?: UniformValues & { setIndex: number; rowStart: number; rowEnd: number } }

export class Renderer {
  device!: GPUDevice; ctx!: GPUCanvasContext; format!: GPUTextureFormat;
  uniformBuf!: GPUBuffer; accumBuf!: GPUBuffer;
  tempBuf!: GPUBuffer; colorBuf!: GPUBuffer;
  spotBuf!: GPUBuffer;   // hot-spot params: array of vec4 (r, psi, sigma, amp)
  skyTex!: GPUTexture; skySampler!: GPUSampler;
  bloomA!: GPUBuffer; bloomB!: GPUBuffer;       // half-res ping/pong glow buffers
  computePipe!: GPUComputePipeline; presentPipe!: GPURenderPipeline;
  brightHPipe!: GPUComputePipeline; blurVPipe!: GPUComputePipeline;
  presentBind!: GPUBindGroup;
  computeLayout!: GPUBindGroupLayout;
  buildPipe!: GPUComputePipeline; shadePipe!: GPUComputePipeline;
  buildUniformBuf!: GPUBuffer;
  entryBufs: GPUBuffer[] = []; bookmarkBuf!: GPUBuffer; bmCountBuf!: GPUBuffer;
  computeBinds: GPUBindGroup[] = []; buildBinds: GPUBindGroup[] = [];
  plan: CachePlan = { nSets: 0, entryBytes: 0, bookmarkCapacity: 0 };
  /** Validation only: force a tiny bookmark buffer to exercise the LIVE fallback. */
  bookmarkCapacityOverride: number | null = null;
  get cacheSets() { return this.plan.nSets; }
  /** Bumped by every cache reallocation; part of the geometry key (see GeometryInputs.epoch). */
  cacheEpoch = 0;
  brightHBind!: GPUBindGroup; blurVBind!: GPUBindGroup;
  displayW = 0; displayH = 0;          // framebuffer size (canvas pixels)
  scale = 1;                           // internal render scale in [0.5, 1]; see setScale
  width = 0; height = 0; bw = 0; bh = 0; // INTERNAL trace size and its quarter-res bloom size
  renderBloom = true; // off for the structural shadow test (measures the raw geometric shadow)
  /** GPU work time (ms) of the most recently completed frame: its busy interval, from the later of
   *  its submit and the previous frame's completion to its own completion (onSubmittedWorkDone,
   *  never awaited). NaN until the first frame completes. Feeds the ScaleController. */
  gpuMs = NaN;
  private lastDone = 0; // performance.now() at the previous frame's completion
  adapterInfo: GpuInfo = { vendor: "", architecture: "", description: "" };

  async init(canvas: HTMLCanvasElement) {
    if (!navigator.gpu) throw new Error("WebGPU not available — use Chrome/Edge.");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No GPU adapter.");
    const info = (adapter as GPUAdapter & { info?: Partial<GpuInfo> }).info ?? {};
    this.adapterInfo = { vendor: info.vendor ?? "", architecture: info.architecture ?? "", description: info.description ?? "" };
    // The cache's per-set entry buffer is W*H*16 bytes; ask for the adapter's maximum binding so
    // large canvases still cache (planCache falls back to fewer sets / live within it).
    this.device = await adapter.requestDevice({ requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    } });
    this.ctx = canvas.getContext("webgpu")!;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.buildUniformBuf = this.device.createBuffer({ size: UNIFORM_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bmCountBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.resize(canvas);
    this.uniformBuf = this.device.createBuffer({ size: UNIFORM_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // Placeholder LUT buffers so the first bind group is valid; replaced by uploadLUTs().
    this.tempBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.colorBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.spotBuf = this.device.createBuffer({ size: 8 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    // 1x1 placeholder sky texture so the first bind group is valid; replaced by uploadSky().
    this.skyTex = this.device.createTexture({ size: [1, 1], format: "rgba8unorm-srgb",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    this.skySampler = this.device.createSampler({ magFilter: "linear", minFilter: "linear",
      mipmapFilter: "linear", addressModeU: "repeat", addressModeV: "clamp-to-edge" });
    this.buildPipelines();
  }

  resize(canvas: HTMLCanvasElement) {
    const dpr = Math.min(devicePixelRatio || 1, 1.5);
    this.displayW = Math.floor(canvas.clientWidth * dpr);
    this.displayH = Math.floor(canvas.clientHeight * dpr);
    canvas.width = this.displayW; canvas.height = this.displayH;
    this.ctx.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
    // Sized for scale 1 (the largest the internal size can reach), so setScale never reallocates.
    this.accumBuf = this.device.createBuffer({ size: this.displayW * this.displayH * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const bloomBytes = Math.ceil(this.displayW / 4) * Math.ceil(this.displayH / 4) * 16; // quarter-res bloom
    this.bloomA = this.device.createBuffer({ size: bloomBytes, usage: GPUBufferUsage.STORAGE });
    this.bloomB = this.device.createBuffer({ size: bloomBytes, usage: GPUBufferUsage.STORAGE });
    this.applyScale();
    this.allocCache();
  }

  /** (Re)allocate the geodesic cache for the display size. Caller must rebind() (resize callers do). */
  private allocCache() {
    for (const b of this.entryBufs) b.destroy();
    this.bookmarkBuf?.destroy();
    this.plan = planCache(this.displayW, this.displayH, this.device.limits.maxStorageBufferBindingSize);
    this.cacheEpoch++;
    const cap = this.bookmarkCapacityOverride ?? this.plan.bookmarkCapacity;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    // With nSets 0 one 16-byte placeholder keeps every bind group valid.
    this.entryBufs = Array.from({ length: Math.max(1, this.plan.nSets) }, () =>
      this.device.createBuffer({ size: this.plan.nSets ? this.plan.entryBytes : 16, usage }));
    this.bookmarkBuf = this.device.createBuffer({ size: Math.max(1, cap) * BOOKMARK_BYTES, usage });
    this.resetCache();
  }
  /** Start a rebuild: bookmark slots are handed out from 0 again. */
  resetCache() { this.device.queue.writeBuffer(this.bmCountBuf, 0, new Uint32Array([0])); }

  /** Internal size from the display size and scale. Buffers are sized for scale 1 and indexed at the
   *  internal width, so a scale change never reallocates (and needs no rebind). */
  // Each side rounds independently, so the internal aspect can differ from the display's by under
  // one internal pixel (e.g. 301x157 -> 151x79 at 0.5, 0.3 %); the present pass stretches it back.
  // Accepted: invisible at these sizes and it keeps the internal grid integral.
  private applyScale() {
    this.width = Math.max(1, Math.round(this.displayW * this.scale));
    this.height = Math.max(1, Math.round(this.displayH * this.scale));
    this.bw = Math.ceil(this.width / 4); this.bh = Math.ceil(this.height / 4);
  }
  /** Returns true when the internal size changed (the caller must restart accumulation). */
  setScale(s: number): boolean {
    const v = Math.min(1, Math.max(0.5, s));
    if (v === this.scale) return false;
    const w = this.width, h = this.height;
    this.scale = v; this.applyScale();
    return this.width !== w || this.height !== h;
  }

  /** Upload the CPU-computed T(r) and color(T) lookup tables as read-only storage buffers. */
  uploadLUTs(tempLUT: Float32Array, colorLUT: Float32Array) {
    // The LUTs come from `new Float32Array(n)`, so they are ArrayBuffer-backed; the cast
    // narrows the TS 5.7+ default `Float32Array<ArrayBufferLike>` to satisfy writeBuffer.
    // Replaced buffers are destroyed (callers rebind() straight after, and work already submitted
    // keeps them alive until it completes). The colour LUT (64 KB) is the same array on every spin
    // tick, so it is only re-created when a different table arrives.
    this.tempBuf?.destroy();
    this.tempBuf = this.device.createBuffer({ size: tempLUT.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.tempBuf, 0, tempLUT as Float32Array<ArrayBuffer>);
    if (colorLUT !== this.colorSrc) {
      this.colorBuf?.destroy();
      this.colorBuf = this.device.createBuffer({ size: colorLUT.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(this.colorBuf, 0, colorLUT as Float32Array<ArrayBuffer>);
      this.colorSrc = colorLUT;
    }
  }
  private colorSrc: Float32Array | null = null;

  /** Upload packed hot-spot params (Float32Array of (r,psi,sigma,amp) per spot). Capped at the
   *  8-vec4 buffer capacity created in init(); extra spots would overflow the storage buffer. */
  uploadHotSpots(spots: Float32Array) {
    const clamped = spots.length > 8 * 4 ? spots.subarray(0, 8 * 4) : spots;
    this.device.queue.writeBuffer(this.spotBuf, 0, clamped as Float32Array<ArrayBuffer>);
  }

  /** Upload the equirectangular sky panorama as an sRGB texture with a full mip chain. Mips are
   *  generated on the CPU via canvas downscales (WebGPU has no built-in generateMipmaps), which
   *  keeps the lensed/minified sky from aliasing. Caller must call rebind() afterward. */
  uploadSky(bitmap: ImageBitmap) {
    const W = 4096, H = 2048;
    const mips = Math.floor(Math.log2(Math.max(W, H))) + 1;
    this.skyTex = this.device.createTexture({ size: [W, H], mipLevelCount: mips, format: "rgba8unorm-srgb",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    const canvas = document.createElement("canvas");
    const g = canvas.getContext("2d")!;
    for (let lvl = 0; lvl < mips; lvl++) {
      const lw = Math.max(1, W >> lvl), lh = Math.max(1, H >> lvl);
      canvas.width = lw; canvas.height = lh;
      g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
      g.drawImage(bitmap, 0, 0, lw, lh);
      this.device.queue.copyExternalImageToTexture({ source: canvas }, { texture: this.skyTex, mipLevel: lvl }, [lw, lh]);
    }
  }

  buildPipelines() {
    // Prepended, not appended: declarations must precede use. shadow-shared.wgsl (critical-curve
    // classifier) and camera-shared.wgsl (screen -> xi/eta + initial momentum) are the sole copies
    // of their math and are shared verbatim with the ?parity route. Both are self-contained --
    // camera-shared.wgsl takes the inverse-metric components as arguments rather than calling
    // gUp() -- so the concatenation order between them does not matter.
    // integrator-shared.wgsl (metric + RK4 + step controller) is likewise the sole copy; it
    // defines PI, so raytrace.wgsl no longer does.
    // emission-shared.wgsl (disk turbulence + jet emissivity) is the sole copy of that math too;
    // ?parity's turb and jet cases prepend the same bytes.
    const cMod = this.device.createShaderModule({ code: shadowSharedWGSL + cameraSharedWGSL + integratorSharedWGSL + emissionSharedWGSL + raytraceWGSL });
    const pMod = this.device.createShaderModule({ code: presentWGSL });
    const bMod = this.device.createShaderModule({ code: bloomWGSL });
    const st = (type: GPUBufferBindingType): GPUBindGroupLayoutEntry["buffer"] => ({ type });
    this.computeLayout = this.device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: st("uniform") },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: st("storage") },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: st("read-only-storage") },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: st("read-only-storage") },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: st("read-only-storage") },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: st("storage") },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: st("storage") },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: st("storage") }] });
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] });
    this.computePipe = this.device.createComputePipeline({ layout, compute: { module: cMod, entryPoint: "main" } });
    this.buildPipe = this.device.createComputePipeline({ layout, compute: { module: cMod, entryPoint: "build" } });
    this.shadePipe = this.device.createComputePipeline({ layout, compute: { module: cMod, entryPoint: "shade" } });
    this.brightHPipe = this.device.createComputePipeline({ layout: "auto", compute: { module: bMod, entryPoint: "bright_h" } });
    this.blurVPipe = this.device.createComputePipeline({ layout: "auto", compute: { module: bMod, entryPoint: "blur_v" } });
    this.presentPipe = this.device.createRenderPipeline({
      layout: "auto", vertex: { module: pMod, entryPoint: "vs" },
      fragment: { module: pMod, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });
    this.rebind();
  }

  rebind() {
    const common = (ub: GPUBuffer, entryBuf: GPUBuffer): GPUBindGroupEntry[] => [
      { binding: 0, resource: { buffer: ub } },
      { binding: 1, resource: { buffer: this.accumBuf } },
      { binding: 2, resource: { buffer: this.tempBuf } },
      { binding: 3, resource: { buffer: this.colorBuf } },
      { binding: 4, resource: { buffer: this.spotBuf } },
      { binding: 5, resource: this.skyTex.createView() },
      { binding: 6, resource: this.skySampler },
      { binding: 7, resource: { buffer: entryBuf } },
      { binding: 8, resource: { buffer: this.bookmarkBuf } },
      { binding: 9, resource: { buffer: this.bmCountBuf } }];
    this.computeBinds = this.entryBufs.map((b) => this.device.createBindGroup({ layout: this.computeLayout, entries: common(this.uniformBuf, b) }));
    this.buildBinds = this.entryBufs.map((b) => this.device.createBindGroup({ layout: this.computeLayout, entries: common(this.buildUniformBuf, b) }));
    // bloom pass 1: accum -> bloomA ; pass 2: bloomA -> bloomB
    this.brightHBind = this.device.createBindGroup({ layout: this.brightHPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.uniformBuf } },
      { binding: 1, resource: { buffer: this.accumBuf } },
      { binding: 2, resource: { buffer: this.bloomA } }] });
    this.blurVBind = this.device.createBindGroup({ layout: this.blurVPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.uniformBuf } },
      { binding: 1, resource: { buffer: this.bloomA } },
      { binding: 2, resource: { buffer: this.bloomB } }] });
    this.presentBind = this.device.createBindGroup({ layout: this.presentPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.uniformBuf } },
      { binding: 1, resource: { buffer: this.accumBuf } },
      { binding: 2, resource: { buffer: this.bloomB } }] });
  }

  /** Record raytrace + the two bloom dispatches into one compute pass. Dispatches in a single
   *  pass execute in order with their storage writes visible to the next, so bright_h sees the
   *  freshly-traced accum and blur_v sees bloomA. */
  private recordCompute(enc: GPUCommandEncoder, opts: FrameOpts = {}) {
    const cp = enc.beginComputePass();
    if (opts.build) {
      // Before shade in the same pass: its entries writes are visible to a shade of the same set.
      const b = opts.build;
      cp.setPipeline(this.buildPipe); cp.setBindGroup(0, this.buildBinds[b.setIndex]);
      cp.dispatchWorkgroups(Math.ceil(this.displayW / 8), Math.ceil((b.rowEnd - b.rowStart) / 8));
    }
    if (opts.cachedSet !== undefined) {
      cp.setPipeline(this.shadePipe); cp.setBindGroup(0, this.computeBinds[opts.cachedSet]);
    } else {
      cp.setPipeline(this.computePipe); cp.setBindGroup(0, this.computeBinds[0]);
    }
    cp.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
    if (this.renderBloom) {
      cp.setPipeline(this.brightHPipe); cp.setBindGroup(0, this.brightHBind);
      cp.dispatchWorkgroups(Math.ceil(this.bw / 8), Math.ceil(this.bh / 8));
      cp.setPipeline(this.blurVPipe); cp.setBindGroup(0, this.blurVBind);
      cp.dispatchWorkgroups(Math.ceil(this.bw / 8), Math.ceil(this.bh / 8));
    }
    cp.end();
  }

  frame(u: UniformValues, opts: FrameOpts = {}) {
    this.device.queue.writeBuffer(this.uniformBuf, 0, packUniforms(u));
    if (opts.build) this.device.queue.writeBuffer(this.buildUniformBuf, 0, packUniforms({ ...opts.build, jitterMode: 1 }));
    const enc = this.device.createCommandEncoder();
    this.recordCompute(enc, opts);
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }] });
    rp.setPipeline(this.presentPipe); rp.setBindGroup(0, this.presentBind); rp.draw(3); rp.end();
    const t0 = performance.now();
    this.device.queue.submit([enc.finish()]);
    // Not awaited: the render loop stays non-blocking; the value lands a frame or so later. This
    // frame's work starts at the later of its submit and the previous frame's completion, so the
    // busy interval excludes time spent queued behind earlier frames. (Plain submit-to-done latency
    // counted the whole uncapped queue, read ~4x the real work at 500x340 and never let the
    // controller climb; measured 2026-09-23, see final-fix-report.md.)
    this.device.queue.onSubmittedWorkDone().then(() => {
      const t1 = performance.now();
      this.gpuMs = t1 - Math.max(t0, this.lastDone); this.lastDone = t1;
    }, () => {});
  }

  /** Render one frame to an offscreen texture and read the presented pixels back to the CPU
   *  (tightly-packed RGBA8/BGRA8, row-stride removed). Used by validation harnesses. */
  async readbackPresented(u: UniformValues): Promise<{ data: Uint8Array; w: number; h: number }> {
    if (this.scale !== 1) throw new Error("readbackPresented requires scale 1 (validation routes run at full resolution)");
    const tex = this.device.createTexture({ size: [this.displayW, this.displayH], format: this.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    this.device.queue.writeBuffer(this.uniformBuf, 0, packUniforms(u));
    const enc = this.device.createCommandEncoder();
    this.recordCompute(enc);
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: tex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }] });
    rp.setPipeline(this.presentPipe); rp.setBindGroup(0, this.presentBind); rp.draw(3); rp.end();
    const bpr = Math.ceil(this.displayW * 4 / 256) * 256; // bytesPerRow must be a multiple of 256
    const buf = this.device.createBuffer({ size: bpr * this.displayH, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [this.displayW, this.displayH]);
    this.device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(buf.getMappedRange().slice(0));
    const data = new Uint8Array(this.displayW * this.displayH * 4);
    for (let y = 0; y < this.displayH; y++) data.set(padded.subarray(y * bpr, y * bpr + this.displayW * 4), y * this.displayW * 4);
    buf.unmap();
    return { data, w: this.displayW, h: this.displayH };
  }
  /** Copy `bytes` from the start of a COPY_SRC buffer to the CPU. Validation harnesses only. */
  private async readback(src: GPUBuffer, bytes: number): Promise<ArrayBuffer> {
    const buf = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, buf, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const out = buf.getMappedRange().slice(0);
    buf.unmap(); buf.destroy();
    return out;
  }
  /** Raw accum (internal width x height vec4<f32>) after the last submitted frame. */
  async readbackAccum(): Promise<Float32Array> {
    return new Float32Array(await this.readback(this.accumBuf, this.width * this.height * 16));
  }
  async readbackEntries(set: number): Promise<Uint32Array> {
    return new Uint32Array(await this.readback(this.entryBufs[set], this.displayW * this.displayH * 16));
  }
  /** Bookmarks handed out since resetCache() and their mean replay length. */
  async readbackBookmarks(): Promise<{ count: number; meanNJet: number }> {
    const handed = new Uint32Array(await this.readback(this.bmCountBuf, 4))[0];
    const count = Math.min(handed, this.bookmarkBuf.size / BOOKMARK_BYTES);
    if (!count) return { count: handed, meanNJet: 0 };
    const words = new Uint32Array(await this.readback(this.bookmarkBuf, count * BOOKMARK_BYTES));
    let sum = 0; for (let k = 0; k < count; k++) sum += words[k * 12 + 8];
    return { count: handed, meanNJet: sum / count };
  }
}
