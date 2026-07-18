import { packUniforms, UniformValues, UNIFORM_SIZE } from "./uniforms";
import presentWGSL from "./present.wgsl?raw";
import raytraceWGSL from "./raytrace.wgsl?raw";
import bloomWGSL from "./bloom.wgsl?raw";

export class Renderer {
  device!: GPUDevice; ctx!: GPUCanvasContext; format!: GPUTextureFormat;
  uniformBuf!: GPUBuffer; accumBuf!: GPUBuffer;
  tempBuf!: GPUBuffer; colorBuf!: GPUBuffer;
  spotBuf!: GPUBuffer;   // hot-spot params: array of vec4 (r, psi, sigma, amp)
  bloomA!: GPUBuffer; bloomB!: GPUBuffer;       // half-res ping/pong glow buffers
  computePipe!: GPUComputePipeline; presentPipe!: GPURenderPipeline;
  brightHPipe!: GPUComputePipeline; blurVPipe!: GPUComputePipeline;
  computeBind!: GPUBindGroup; presentBind!: GPUBindGroup;
  brightHBind!: GPUBindGroup; blurVBind!: GPUBindGroup;
  width = 0; height = 0; bw = 0; bh = 0; // bw/bh = quarter-res bloom dimensions
  renderBloom = true; // off for the structural shadow test (measures the raw geometric shadow)

  async init(canvas: HTMLCanvasElement) {
    if (!navigator.gpu) throw new Error("WebGPU not available — use Chrome/Edge.");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No GPU adapter.");
    this.device = await adapter.requestDevice();
    this.ctx = canvas.getContext("webgpu")!;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.resize(canvas);
    this.uniformBuf = this.device.createBuffer({ size: UNIFORM_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // Placeholder LUT buffers so the first bind group is valid; replaced by uploadLUTs().
    this.tempBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.colorBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.spotBuf = this.device.createBuffer({ size: 8 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.buildPipelines();
  }

  resize(canvas: HTMLCanvasElement) {
    const dpr = Math.min(devicePixelRatio || 1, 1.5);
    this.width = Math.floor(canvas.clientWidth * dpr);
    this.height = Math.floor(canvas.clientHeight * dpr);
    canvas.width = this.width; canvas.height = this.height;
    this.ctx.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
    this.accumBuf = this.device.createBuffer({ size: this.width * this.height * 16, usage: GPUBufferUsage.STORAGE });
    this.bw = Math.ceil(this.width / 4); this.bh = Math.ceil(this.height / 4); // quarter-res bloom

    this.bloomA = this.device.createBuffer({ size: this.bw * this.bh * 16, usage: GPUBufferUsage.STORAGE });
    this.bloomB = this.device.createBuffer({ size: this.bw * this.bh * 16, usage: GPUBufferUsage.STORAGE });
  }

  /** Upload the CPU-computed T(r) and color(T) lookup tables as read-only storage buffers. */
  uploadLUTs(tempLUT: Float32Array, colorLUT: Float32Array) {
    // The LUTs come from `new Float32Array(n)`, so they are ArrayBuffer-backed; the cast
    // narrows the TS 5.7+ default `Float32Array<ArrayBufferLike>` to satisfy writeBuffer.
    this.tempBuf = this.device.createBuffer({ size: tempLUT.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.tempBuf, 0, tempLUT as Float32Array<ArrayBuffer>);
    this.colorBuf = this.device.createBuffer({ size: colorLUT.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.colorBuf, 0, colorLUT as Float32Array<ArrayBuffer>);
  }

  /** Upload packed hot-spot params (Float32Array of (r,psi,sigma,amp) per spot). */
  uploadHotSpots(spots: Float32Array) {
    this.device.queue.writeBuffer(this.spotBuf, 0, spots as Float32Array<ArrayBuffer>);
  }

  buildPipelines() {
    const cMod = this.device.createShaderModule({ code: raytraceWGSL });
    const pMod = this.device.createShaderModule({ code: presentWGSL });
    const bMod = this.device.createShaderModule({ code: bloomWGSL });
    this.computePipe = this.device.createComputePipeline({ layout: "auto", compute: { module: cMod, entryPoint: "main" } });
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
    this.computeBind = this.device.createBindGroup({ layout: this.computePipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.uniformBuf } },
      { binding: 1, resource: { buffer: this.accumBuf } },
      { binding: 2, resource: { buffer: this.tempBuf } },
      { binding: 3, resource: { buffer: this.colorBuf } },
      { binding: 4, resource: { buffer: this.spotBuf } }] });
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
  private recordCompute(enc: GPUCommandEncoder) {
    const cp = enc.beginComputePass();
    cp.setPipeline(this.computePipe); cp.setBindGroup(0, this.computeBind);
    cp.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
    if (this.renderBloom) {
      cp.setPipeline(this.brightHPipe); cp.setBindGroup(0, this.brightHBind);
      cp.dispatchWorkgroups(Math.ceil(this.bw / 8), Math.ceil(this.bh / 8));
      cp.setPipeline(this.blurVPipe); cp.setBindGroup(0, this.blurVBind);
      cp.dispatchWorkgroups(Math.ceil(this.bw / 8), Math.ceil(this.bh / 8));
    }
    cp.end();
  }

  frame(u: UniformValues) {
    this.device.queue.writeBuffer(this.uniformBuf, 0, packUniforms(u));
    const enc = this.device.createCommandEncoder();
    this.recordCompute(enc);
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }] });
    rp.setPipeline(this.presentPipe); rp.setBindGroup(0, this.presentBind); rp.draw(3); rp.end();
    this.device.queue.submit([enc.finish()]);
  }

  /** Render one frame to an offscreen texture and read the presented pixels back to the CPU
   *  (tightly-packed RGBA8/BGRA8, row-stride removed). Used by validation harnesses. */
  async readbackPresented(u: UniformValues): Promise<{ data: Uint8Array; w: number; h: number }> {
    const tex = this.device.createTexture({ size: [this.width, this.height], format: this.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    this.device.queue.writeBuffer(this.uniformBuf, 0, packUniforms(u));
    const enc = this.device.createCommandEncoder();
    this.recordCompute(enc);
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: tex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }] });
    rp.setPipeline(this.presentPipe); rp.setBindGroup(0, this.presentBind); rp.draw(3); rp.end();
    const bpr = Math.ceil(this.width * 4 / 256) * 256; // bytesPerRow must be a multiple of 256
    const buf = this.device.createBuffer({ size: bpr * this.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [this.width, this.height]);
    this.device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(buf.getMappedRange().slice(0));
    const data = new Uint8Array(this.width * this.height * 4);
    for (let y = 0; y < this.height; y++) data.set(padded.subarray(y * bpr, y * bpr + this.width * 4), y * this.width * 4);
    buf.unmap();
    return { data, w: this.width, h: this.height };
  }
}
