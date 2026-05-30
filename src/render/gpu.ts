import { packUniforms, UniformValues, UNIFORM_SIZE } from "./uniforms";
import presentWGSL from "./present.wgsl?raw";

const TEST_COMPUTE = /* wgsl */`
struct Uniforms { res: vec2<f32>, a: f32, incl: f32, rObs: f32, fovScale: f32, rIn: f32, rOut: f32, Tpeak: f32, exposure: f32, time: f32, frame: u32, reset: u32, maxSteps: u32, };
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read_write> accum: array<vec4<f32>>;
@compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u32(U.res.x) || gid.y >= u32(U.res.y)) { return; }
  let idx = gid.y * u32(U.res.x) + gid.x;
  let uv = vec2<f32>(f32(gid.x)/U.res.x, f32(gid.y)/U.res.y);
  accum[idx] = vec4<f32>(uv.x, uv.y, 0.2, 1.0); // gradient
}`;

export class Renderer {
  device!: GPUDevice; ctx!: GPUCanvasContext; format!: GPUTextureFormat;
  uniformBuf!: GPUBuffer; accumBuf!: GPUBuffer;
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
    this.buildPipelines(TEST_COMPUTE, presentWGSL);
  }

  resize(canvas: HTMLCanvasElement) {
    const dpr = Math.min(devicePixelRatio || 1, 1.5);
    this.width = Math.floor(canvas.clientWidth * dpr);
    this.height = Math.floor(canvas.clientHeight * dpr);
    canvas.width = this.width; canvas.height = this.height;
    this.ctx.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
    this.accumBuf = this.device.createBuffer({ size: this.width * this.height * 16, usage: GPUBufferUsage.STORAGE });
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
      { binding: 0, resource: { buffer: this.uniformBuf } }, { binding: 1, resource: { buffer: this.accumBuf } }] });
    this.presentBind = this.device.createBindGroup({ layout: this.presentPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.uniformBuf } }, { binding: 1, resource: { buffer: this.accumBuf } }] });
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
