import { packUniforms, UniformValues, UNIFORM_SIZE } from "./uniforms";
import presentWGSL from "./present.wgsl?raw";
import raytraceWGSL from "./raytrace.wgsl?raw";

export class Renderer {
  device!: GPUDevice; ctx!: GPUCanvasContext; format!: GPUTextureFormat;
  uniformBuf!: GPUBuffer; accumBuf!: GPUBuffer;
  tempBuf!: GPUBuffer; colorBuf!: GPUBuffer;
  computePipe!: GPUComputePipeline; presentPipe!: GPURenderPipeline;
  computeBind!: GPUBindGroup; presentBind!: GPUBindGroup;
  width = 0; height = 0;

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
    this.buildPipelines(raytraceWGSL, presentWGSL);
  }

  resize(canvas: HTMLCanvasElement) {
    const dpr = Math.min(devicePixelRatio || 1, 1.5);
    this.width = Math.floor(canvas.clientWidth * dpr);
    this.height = Math.floor(canvas.clientHeight * dpr);
    canvas.width = this.width; canvas.height = this.height;
    this.ctx.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
    this.accumBuf = this.device.createBuffer({ size: this.width * this.height * 16, usage: GPUBufferUsage.STORAGE });
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

  buildPipelines(computeSrc: string, presentSrc: string) {
    const cMod = this.device.createShaderModule({ code: computeSrc });
    const pMod = this.device.createShaderModule({ code: presentSrc });
    this.computePipe = this.device.createComputePipeline({ layout: "auto", compute: { module: cMod, entryPoint: "main" } });
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
      { binding: 3, resource: { buffer: this.colorBuf } }] });
    this.presentBind = this.device.createBindGroup({ layout: this.presentPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.uniformBuf } },
      { binding: 1, resource: { buffer: this.accumBuf } }] });
  }

  frame(u: UniformValues) {
    this.device.queue.writeBuffer(this.uniformBuf, 0, packUniforms(u));
    const enc = this.device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(this.computePipe); cp.setBindGroup(0, this.computeBind);
    cp.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8)); cp.end();
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }] });
    rp.setPipeline(this.presentPipe); rp.setBindGroup(0, this.presentBind); rp.draw(3); rp.end();
    this.device.queue.submit([enc.finish()]);
  }
}
