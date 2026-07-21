import { metricUpper, metricLower } from "../physics/kerr";
import { omegaKepler } from "../physics/orbits";
import { gFactorKepler } from "../physics/redshift";
import parityWGSL from "../render/parity.wgsl?raw";
import { turbulence } from "../physics/emission";
import turbParityWGSL from "../render/turb-parity.wgsl?raw";
import { jetEmission, dopplerBoost } from "../physics/jet";
import jetParityWGSL from "../render/jet-parity.wgsl?raw";
import { classify, criticalXiEta, photonShellRange } from "../physics/shadow";
import shadowSharedWGSL from "../render/shadow-shared.wgsl?raw";
import shadowParityWGSL from "../render/shadow-parity.wgsl?raw";
import { screenToState, screenToXiEta } from "../physics/camera";
import cameraSharedWGSL from "../render/camera-shared.wgsl?raw";
import cameraParityWGSL from "../render/camera-parity.wgsl?raw";

/** Runs the WGSL metric/orbit/g-factor helpers on fixed inputs and returns the max relative
 *  error vs the TypeScript core. f32 GPU vs f64 CPU keeps this in the ~1e-6..1e-4 range. */
export async function runParity(): Promise<{ maxErr: number; rows: number }> {
  const cases = [
    { r: 8, th: Math.PI / 2, a: 0.0, xi: 3 }, { r: 6, th: 1.2, a: 0.5, xi: 2 },
    { r: 12, th: Math.PI / 2, a: 0.9, xi: -4 }, { r: 20, th: 0.9, a: 0.99, xi: 5 },
  ];
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter!.requestDevice();
  const inBuf = device.createBuffer({ size: cases.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const inArr = new Float32Array(cases.length * 4);
  cases.forEach((c, i) => { inArr.set([c.r, c.th, c.a, c.xi], i * 4); });
  device.queue.writeBuffer(inBuf, 0, inArr);
  const outBuf = device.createBuffer({ size: cases.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: cases.length * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const mod = device.createShaderModule({ code: parityWGSL });
  const pipe = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
  const bind = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: inBuf } }, { binding: 1, resource: { buffer: outBuf } }] });
  const enc = device.createCommandEncoder();
  const cp = enc.beginComputePass(); cp.setPipeline(pipe); cp.setBindGroup(0, bind); cp.dispatchWorkgroups(1); cp.end();
  enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, cases.length * 16);
  device.queue.submit([enc.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ);
  const gpu = new Float32Array(readBuf.getMappedRange().slice(0));
  let maxErr = 0;
  cases.forEach((c, i) => {
    const cpu = [metricUpper(c.r, c.th, c.a).tt, metricLower(c.r, c.th, c.a).tt,
                 omegaKepler(c.r, c.a, true), gFactorKepler(c.r, c.a, c.xi, true)];
    for (let k = 0; k < 4; k++) maxErr = Math.max(maxErr, Math.abs(gpu[i * 4 + k] - cpu[k]) / (1 + Math.abs(cpu[k])));
  });
  // --- turbulence parity (CPU emission.ts vs GPU turb-parity.wgsl) ---
  const tcases = [
    { logR: Math.log(6), psi: 0.4 }, { logR: Math.log(9), psi: 1.7 },
    { logR: Math.log(14), psi: 3.9 }, { logR: Math.log(22), psi: 5.2 },
  ];
  const tin = device.createBuffer({ size: tcases.length * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const tarr = new Float32Array(tcases.length * 2);
  tcases.forEach((c, i) => { tarr.set([c.logR, c.psi], i * 2); });
  device.queue.writeBuffer(tin, 0, tarr);
  const tout = device.createBuffer({ size: tcases.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const tread = device.createBuffer({ size: tcases.length * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const tmod = device.createShaderModule({ code: turbParityWGSL });
  const tpipe = device.createComputePipeline({ layout: "auto", compute: { module: tmod, entryPoint: "main" } });
  const tbind = device.createBindGroup({ layout: tpipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: tin } }, { binding: 1, resource: { buffer: tout } }] });
  const tenc = device.createCommandEncoder();
  const tcp = tenc.beginComputePass(); tcp.setPipeline(tpipe); tcp.setBindGroup(0, tbind); tcp.dispatchWorkgroups(1); tcp.end();
  tenc.copyBufferToBuffer(tout, 0, tread, 0, tcases.length * 4);
  device.queue.submit([tenc.finish()]);
  await tread.mapAsync(GPUMapMode.READ);
  const tgpu = new Float32Array(tread.getMappedRange().slice(0));
  tcases.forEach((c, i) => {
    const cpu = turbulence(c.logR, c.psi, 3);
    maxErr = Math.max(maxErr, Math.abs(tgpu[i] - cpu) / (1 + Math.abs(cpu)));
  });
  // --- jet parity (CPU jet.ts vs GPU jet-parity.wgsl) ---
  const jcases = [
    { r: 8,  th: 0.12, t: 0.0, mu: 0.9 },
    { r: 14, th: 0.20, t: 1.3, mu: 0.3 },
    { r: 20, th: 0.10, t: 2.7, mu: -0.6 },
    { r: 6,  th: 0.30, t: 0.5, mu: -0.9 },
  ];
  const jin = device.createBuffer({ size: jcases.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const jarr = new Float32Array(jcases.length * 4);
  jcases.forEach((c, i) => { jarr.set([c.r, c.th, c.t, c.mu], i * 4); });
  device.queue.writeBuffer(jin, 0, jarr);
  const jout = device.createBuffer({ size: jcases.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const jread = device.createBuffer({ size: jcases.length * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const jmod = device.createShaderModule({ code: jetParityWGSL });
  const jpipe = device.createComputePipeline({ layout: "auto", compute: { module: jmod, entryPoint: "main" } });
  const jbind = device.createBindGroup({ layout: jpipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: jin } }, { binding: 1, resource: { buffer: jout } }] });
  const jenc = device.createCommandEncoder();
  const jcp = jenc.beginComputePass(); jcp.setPipeline(jpipe); jcp.setBindGroup(0, jbind); jcp.dispatchWorkgroups(jcases.length); jcp.end();
  jenc.copyBufferToBuffer(jout, 0, jread, 0, jcases.length * 16);
  device.queue.submit([jenc.finish()]);
  await jread.mapAsync(GPUMapMode.READ);
  const jgpu = new Float32Array(jread.getMappedRange().slice(0));
  jcases.forEach((c, i) => {
    const cpuE = jetEmission(c.r, c.th, c.t, 1, 1, 60, 0.7);
    const cpuB = dopplerBoost(c.mu, 5);
    maxErr = Math.max(maxErr, Math.abs(jgpu[i * 4 + 0] - cpuE) / (1 + Math.abs(cpuE)));
    maxErr = Math.max(maxErr, Math.abs(jgpu[i * 4 + 1] - cpuB) / (1 + Math.abs(cpuB)));
  });
  // --- shadow-classifier parity (CPU shadow.ts vs the SHIPPED classifier in shadow-shared.wgsl) ---
  // Unlike the blocks above, this does NOT compare against a separate copy of the math: the same
  // shadow-shared.wgsl that gpu.ts prepends to raytrace.wgsl is prepended here, so a desync
  // between the renderer and shadow.ts cannot hide. Cases straddle the critical curve at +/-1% in
  // eta, which is where a formula drift would first flip a pixel.
  const scases: { xi: number; eta: number; a: number }[] = [];
  for (const a of [0.9, 0.998]) {
    const [slo, shi] = photonShellRange(a);
    for (let i = 1; i <= 5; i++) {
      const r = slo + ((shi - slo) * i) / 6; // strictly inside the photon shell
      const [xiC, etaC] = criticalXiEta(r, a);
      scases.push({ xi: xiC, eta: 0.99 * etaC, a }); // just inside -> captured
      scases.push({ xi: xiC, eta: 1.01 * etaC, a }); // just outside -> escaped
    }
  }
  // The shell sampling never reaches these two branches, so cover them explicitly:
  scases.push({ xi: 0, eta: 26.5, a: 0 }, { xi: 0, eta: 27.5, a: 0 });      // |a| < A_EPS short-circuit
  scases.push({ xi: 1000, eta: 1, a: 0.9 }, { xi: -1000, eta: 1, a: 0.9 }); // out-of-bracket early return
  const sin_ = device.createBuffer({ size: scases.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const sarr = new Float32Array(scases.length * 4);
  scases.forEach((c, i) => { sarr.set([c.xi, c.eta, c.a, 0], i * 4); });
  device.queue.writeBuffer(sin_, 0, sarr);
  const sout = device.createBuffer({ size: scases.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const sread = device.createBuffer({ size: scases.length * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const smod = device.createShaderModule({ code: shadowSharedWGSL + shadowParityWGSL });
  const spipe = device.createComputePipeline({ layout: "auto", compute: { module: smod, entryPoint: "main" } });
  const sbind = device.createBindGroup({ layout: spipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: sin_ } }, { binding: 1, resource: { buffer: sout } }] });
  const senc = device.createCommandEncoder();
  const scp = senc.beginComputePass(); scp.setPipeline(spipe); scp.setBindGroup(0, sbind); scp.dispatchWorkgroups(scases.length); scp.end();
  senc.copyBufferToBuffer(sout, 0, sread, 0, scases.length * 16);
  device.queue.submit([senc.finish()]);
  await sread.mapAsync(GPUMapMode.READ);
  const sgpu = new Float32Array(sread.getMappedRange().slice(0));
  scases.forEach((c, i) => {
    // Absolute difference, not relative: the value is already a 0/1 flag, so a mismatch scores 1.0
    // -- three orders of magnitude above the 1e-3 pass threshold -- and agreement scores exactly 0,
    // leaving the documented 9.690e-7 gate untouched.
    const cpu = classify(c.xi, c.eta, c.a) === "captured" ? 1 : 0;
    maxErr = Math.max(maxErr, Math.abs(sgpu[i * 4 + 0] - cpu));
  });
  // --- camera parity (CPU camera.ts vs the SHIPPED mapping in camera-shared.wgsl) ---
  // Like the shadow block, this compares against the exact bytes gpu.ts prepends to raytrace.wgsl,
  // not a hand-synced duplicate. The inverse-metric components are computed on the CPU (kerr.ts is
  // parity-covered separately, above) and passed in, so this case isolates the camera algebra.
  //
  // Case selection targets the historical bug -- negating p_t alone instead of the whole momentum.
  // That bug leaves pth = +beta and pr > 0 (outward), so it is only visible when beta != 0 and the
  // sign of beta is checked BOTH ways, and its inclination error (pi - i) vanishes at i = pi/2.
  // Hence: both signs of beta, a = 0 and a ~ 0.9, and inclinations well away from 90 degrees.
  const ccases = [
    { alpha:  4.0, beta:  3.0, a: 0.0,  incl: 1.2 },
    { alpha:  4.0, beta: -3.0, a: 0.0,  incl: 1.2 },  // beta sign flip: catches pth = +beta
    { alpha: -5.0, beta:  2.0, a: 0.9,  incl: 0.45 }, // far from pi/2: catches the pi - i mapping
    { alpha: -5.0, beta: -2.0, a: 0.9,  incl: 0.45 },
    { alpha:  6.0, beta:  1.5, a: 0.5,  incl: 2.6 },  // i > pi/2, mirror of the above
    { alpha:  2.0, beta: -4.5, a: 0.998,incl: 1.0 },
    { alpha:  0.0, beta:  3.5, a: 0.9,  incl: 0.8 },  // alpha = 0 edge: xi = 0
    { alpha:  5.0, beta:  0.0, a: 0.9,  incl: 0.8 },  // beta = 0 edge: pth = 0
    { alpha:  0.0, beta:  0.0, a: 0.0,  incl: 1.5 },  // both zero: radial ray
  ];
  const rObs = 100;
  const cin = device.createBuffer({ size: ccases.length * 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const carr = new Float32Array(ccases.length * 12);
  ccases.forEach((c, i) => {
    const g = metricUpper(rObs, c.incl, c.a); // observer sits at theta = incl
    carr.set([c.alpha, c.beta, c.a, c.incl], i * 12);
    carr.set([g.tt, g.tphi, g.rr, g.thth], i * 12 + 4);
    carr.set([g.phph, 0, 0, 0], i * 12 + 8);
  });
  device.queue.writeBuffer(cin, 0, carr);
  const cout = device.createBuffer({ size: ccases.length * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const cread = device.createBuffer({ size: ccases.length * 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const cmod = device.createShaderModule({ code: cameraSharedWGSL + cameraParityWGSL });
  const cpipe = device.createComputePipeline({ layout: "auto", compute: { module: cmod, entryPoint: "main" } });
  const cbind = device.createBindGroup({ layout: cpipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: cin } }, { binding: 1, resource: { buffer: cout } }] });
  const cenc = device.createCommandEncoder();
  const ccp = cenc.beginComputePass(); ccp.setPipeline(cpipe); ccp.setBindGroup(0, cbind); ccp.dispatchWorkgroups(ccases.length); ccp.end();
  cenc.copyBufferToBuffer(cout, 0, cread, 0, ccases.length * 32);
  device.queue.submit([cenc.finish()]);
  await cread.mapAsync(GPUMapMode.READ);
  const cgpu = new Float32Array(cread.getMappedRange().slice(0));
  ccases.forEach((c, i) => {
    const st = screenToState(c.alpha, c.beta, c.a, c.incl, rObs); // [t,r,th,phi, pt,pr,pth,pphi]
    const [xi, eta] = screenToXiEta(c.alpha, c.beta, c.a, c.incl);
    const cpu = [st[4], st[5], st[6], st[7], xi, eta];
    const got = [cgpu[i*8+0], cgpu[i*8+1], cgpu[i*8+2], cgpu[i*8+3], cgpu[i*8+4], cgpu[i*8+5]];
    for (let k = 0; k < 6; k++) maxErr = Math.max(maxErr, Math.abs(got[k] - cpu[k]) / (1 + Math.abs(cpu[k])));
  });
  return { maxErr, rows: cases.length + tcases.length + jcases.length + scases.length + ccases.length };
}
