import { metricUpper, metricLower } from "../physics/kerr";
import { omegaKepler } from "../physics/orbits";
import { gFactorKepler } from "../physics/redshift";
import parityWGSL from "../render/parity.wgsl?raw";
import integratorSharedWGSL from "../render/integrator-shared.wgsl?raw";
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
import { stepGeodesic, stepSize, H_TOL, H_TOL_FAR, MAX_RETRY } from "../physics/trace";
import integratorParityWGSL from "../render/integrator-parity.wgsl?raw";

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
  const mod = device.createShaderModule({ code: integratorSharedWGSL + parityWGSL });
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
  // --- integrator parity (CPU trace.ts vs the SHIPPED stepGeodesic in integrator-shared.wgsl) ---
  // Same shared-fragment discipline as the shadow and camera blocks. The cases are found by
  // walking real trajectories on the CPU until a state with the wanted property appears, so each
  // one is guaranteed to exercise the branch it is named for; a case that cannot be found throws,
  // which fails the route rather than silently testing nothing.
  const I8 = (8 * Math.PI) / 180;
  // hTol is what the renderer would use at that state (H_TOL, or H_TOL_FAR in the unmonitored far
  // field). nCmp = how many leading state components are compared: 8 = the whole state, 4 = the
  // positions only (see "far"), 0 = only retries/ok/constants (see "barrier").
  type ICase = { label: string; s: Float64Array; a: number; dl0: number; hTol: number; nCmp: number };
  const icases: ICase[] = [];
  const rhOf = (a: number) => 1 + Math.sqrt(Math.max(0, 1 - a * a));
  /** Walk from s0 with the shipped step controller (renderer tolerances: H_TOL_FAR beyond
   *  rOut * 1.5, H_TOL inside) until pred(s, out) holds; return that pre-step state. */
  function findState(label: string, s0: Float64Array, a: number, pred: (s: Float64Array, out: ReturnType<typeof stepGeodesic>) => boolean): ICase {
    let s = s0;
    for (let k = 0; k < 20000; k++) {
      const dl0 = stepSize(s[1], rhOf(a), 40);
      const hTol = s[1] > 40 * 1.5 ? H_TOL_FAR : H_TOL;
      const out = stepGeodesic(s, a, dl0, hTol);
      if (pred(s, out)) return { label, s, a, dl0, hTol, nCmp: 8 };
      if (!out.ok || out.s[1] <= rhOf(a) * 1.005 || out.s[1] > 1200) break;
      s = out.s;
    }
    throw new Error(`integrator parity: no state found for case "${label}"`);
  }
  // far field, long stride, a != 0. Unmonitored there (H_TOL_FAR): the f32 finite-difference force
  // at r = 1000 is noise (measured |dH|/scale 2.4e-3 on the GPU vs 2e-11 in f64), so with H_TOL the
  // GPU would halve where the CPU does not. Retries must be 0/0 (with H_TOL the GPU retries once,
  // so 0/0 proves stepGeodesic honours its hTol argument -- the select() that picks H_TOL_FAR in
  // raytrace.wgsl's loop is not exercised here, only by the ?shadow re-baseline and the fps) and
  // the POSITIONS must agree (7.1e-6 measured, dominated by r: 994.000073 CPU vs 993.992981 GPU
  // after the dl = 6 stride; they verify the stride and the x-update path). The momenta are not compared: the same FD noise puts
  // ~4e-4 per unit dl into p_r on the GPU (measured p_r -1.004375 vs CPU -1.002003 after dl = 6,
  // where the true change is 1.2e-5), a pre-existing far-field precision limit of rhs(), not a
  // statement about the shipped stepGeodesic bytes.
  icases.push({ label: "far", s: screenToState(4, 3, 0.9, 1.2, 1000), a: 0.9, dl0: stepSize(1000, rhOf(0.9), 40), hTol: H_TOL_FAR, nCmp: 4 });
  // strong field, equatorial (beta = 0 at i = pi/2 stays in the plane); alpha = 2 => L_z = 2, well
  // inside the prograde critical curve at a = 0.9, so the ray reaches r < 6 before capture
  const strongEq = findState("strong-eq", screenToState(2, 0, 0.9, Math.PI / 2, 1000), 0.9, (s) => s[1] < 6);
  icases.push(strongEq);
  // near the capture margin (a = 0, b = 4 is captured; rh = 2)
  icases.push(findState("near-horizon", screenToState(4, 0, 0, Math.PI / 2, 1000), 0, (s) => s[1] < 2.3));
  // approaching the axis, before the turning point, no retry expected
  icases.push(findState("near-axis", screenToState(0.05, 6, 0, I8, 1000), 0, (s) => s[2] < 0.01));
  // a step that the monitor actually halves (retries >= 1) -- the whole point of the feature. A
  // well-conditioned forced retry: the strong-eq state with an oversized stride (the renderer
  // would use ~0.09 here). Both sides must halve the same number of times AND agree on the state.
  // A vacuous case (CPU retries = 0) throws, so the route cannot pass while testing nothing.
  const retryDl0 = 2.0;
  if (stepGeodesic(strongEq.s, strongEq.a, retryDl0, H_TOL).retries < 1) throw new Error("integrator parity: \"retry\" case is vacuous (CPU retries = 0)");
  icases.push({ label: "retry", s: strongEq.s, a: strongEq.a, dl0: retryDl0, hTol: H_TOL, nCmp: 8 });
  // a barely-resolved barrier step near the axis (alpha = 0.05 ray, theta ~ 1.6e-3, dl0 = 0.5):
  // retries and ok are compared, the STATE is not. One RK4 step here takes p_theta from -4.20 to
  // -1.35 through the 1/sin^2 barrier; the f32 FD force is only ~1e-2 accurate there and the two
  // sides land 2% apart (CPU p_theta -1.3487, GPU -1.3212, resolved 32-substep reference -1.2954)
  // while 1-ulp input perturbations move it by 1e-6. State agreement here would be an
  // integrator-precision statement, not a shipped-bytes statement; the disk-hit radius downstream
  // is still within 0.01 M of the converged reference (Task 2 sweep). The retry count IS the
  // shipped-bytes statement: the GPU must halve exactly where trace.ts does.
  const barrier = findState("barrier", screenToState(0.05, 6, 0, I8, 1000), 0, (_, out) => out.retries >= 1);
  icases.push({ ...barrier, nCmp: 0 });
  // a step that crosses the axis (xi = 0): only reflectAxis can flip the sign of p_theta here
  icases.push(findState("reflect", screenToState(0, 6, 0, I8, 1000), 0, (s, out) => out.s[6] * s[6] < 0));
  const iin = device.createBuffer({ size: icases.length * 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const iarr = new Float32Array(icases.length * 12);
  icases.forEach((c, i) => { iarr.set([c.s[0], c.s[1], c.s[2], c.s[3], c.s[4], c.s[5], c.s[6], c.s[7], c.a, c.dl0, c.hTol, 0], i * 12); });
  device.queue.writeBuffer(iin, 0, iarr);
  const iout = device.createBuffer({ size: icases.length * 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const iread = device.createBuffer({ size: icases.length * 48, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const imod = device.createShaderModule({ code: integratorSharedWGSL + integratorParityWGSL });
  const ipipe = device.createComputePipeline({ layout: "auto", compute: { module: imod, entryPoint: "main" } });
  const ibind = device.createBindGroup({ layout: ipipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: iin } }, { binding: 1, resource: { buffer: iout } }] });
  const ienc = device.createCommandEncoder();
  const icp = ienc.beginComputePass(); icp.setPipeline(ipipe); icp.setBindGroup(0, ibind); icp.dispatchWorkgroups(icases.length); icp.end();
  ienc.copyBufferToBuffer(iout, 0, iread, 0, icases.length * 48);
  device.queue.submit([ienc.finish()]);
  await iread.mapAsync(GPUMapMode.READ);
  const igpu = new Float32Array(iread.getMappedRange().slice(0));
  icases.forEach((c, i) => {
    // The GPU starts from the f32-rounded state, so compare against the CPU stepping that same
    // rounded state; otherwise the input rounding (not the shader) would dominate the error.
    const s32 = Float64Array.from(Array.from(c.s, Math.fround));
    const cpu = stepGeodesic(s32, c.a, Math.fround(c.dl0), Math.fround(c.hTol));
    for (let k = 0; k < c.nCmp; k++) {
      const got = igpu[i * 12 + k], want = cpu.s[k];
      maxErr = Math.max(maxErr, Math.abs(got - want) / (1 + Math.abs(want)));
    }
    // Retry count and ok flag: any mismatch scores >= 1.0 and fails the route outright. A retry
    // mismatch on the "retry" case means H_TOL is at the f32 noise floor -- see the plan, Task 4.
    maxErr = Math.max(maxErr, Math.abs(igpu[i * 12 + 8] - cpu.retries));
    maxErr = Math.max(maxErr, Math.abs(igpu[i * 12 + 9] - (cpu.ok ? 1 : 0)));
    // The constants themselves, so a desync between trace.ts and the fragment cannot hide.
    maxErr = Math.max(maxErr, Math.abs(igpu[i * 12 + 10] - H_TOL) / H_TOL);
    maxErr = Math.max(maxErr, Math.abs(igpu[i * 12 + 11] - MAX_RETRY));
  });
  console.log("integrator parity cases", icases.map((c) => c.label),
    "cpu retries", icases.map((c) => stepGeodesic(Float64Array.from(Array.from(c.s, Math.fround)), c.a, Math.fround(c.dl0), Math.fround(c.hTol)).retries),
    "gpu retries", icases.map((_, i) => igpu[i * 12 + 8]));
  return { maxErr, rows: cases.length + tcases.length + jcases.length + scases.length + ccases.length + icases.length };
}
