import { metricUpper, metricLower } from "../physics/kerr";
import { omegaKepler } from "../physics/orbits";
import { gFactorKepler } from "../physics/redshift";
import parityWGSL from "../render/parity.wgsl?raw";
import integratorSharedWGSL from "../render/integrator-shared.wgsl?raw";
import { turbulence } from "../physics/emission";
import turbParityWGSL from "../render/turb-parity.wgsl?raw";
import emissionSharedWGSL from "../render/emission-shared.wgsl?raw";
import { jetEmission, dopplerBoost } from "../physics/jet";
import jetParityWGSL from "../render/jet-parity.wgsl?raw";
import { classify, criticalXiEta, photonShellRange } from "../physics/shadow";
import shadowSharedWGSL from "../render/shadow-shared.wgsl?raw";
import shadowParityWGSL from "../render/shadow-parity.wgsl?raw";
import { screenToState, screenToXiEta } from "../physics/camera";
import cameraSharedWGSL from "../render/camera-shared.wgsl?raw";
import cameraParityWGSL from "../render/camera-parity.wgsl?raw";
import { stepGeodesic, stepSize, H_TOL, H_TOL_FAR, MAX_RETRY, F_PHI, DL_FAR_MIN } from "../physics/trace";
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
  const tmod = device.createShaderModule({ code: emissionSharedWGSL + turbParityWGSL });
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
  const jmod = device.createShaderModule({ code: emissionSharedWGSL + jetParityWGSL });
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
    const cpuE = jetEmission(c.r, c.th, c.t, 5, 1, 60, 0.7); // Gamma = P_GAMMA in jet-parity.wgsl
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
  // --- integrator parity (CPU trace.ts vs the SHIPPED stepGeodesic + stepSize in integrator-shared.wgsl) ---
  // Same shared-fragment discipline as the shadow and camera blocks. The cases are found by
  // walking real trajectories on the CPU until a state with the wanted property appears, so each
  // one is guaranteed to exercise the branch it is named for; a case that cannot be found throws,
  // which fails the route rather than silently testing nothing.
  //
  // What this gate covers: stepGeodesic (monitor, halving, retry count, ok flag), stepSize (the
  // shader computes its own stride on the dl0 < 0 sentinel -- near branch, far branch and the
  // F_AXIS and F_PHI caps of angularCap in both branches -- and reports it back), reflectAxis, and the constants
  // H_TOL/MAX_RETRY/H_TOL_FAR.
  // What it does NOT cover: the render loop in raytrace.wgsl around those calls -- its
  // select(H_TOL, H_TOL_FAR, far) predicate, the disk test and its |delta theta| guard, the
  // capture/escape tests -- which are exercised only by ?shadow and scripts/probe-axis.mjs.
  // Catch threshold, concretely: a one-step state desync of >~ 2e-3 absolute in the momenta or
  // ~6e-3 in r on the near-field cases (relative error 1e-3 on values of O(1..6)); a retry, ok,
  // constant or stride mismatch scores >= 1 (retry/ok/constants) or >~ 1e-3 relative (dl0).
  const I8 = (8 * Math.PI) / 180;
  const ROUT = 40; // the renderer's disk edge; sets the far-field threshold rOut * 1.5 in stepSize
  // hTol is what the renderer would use at that state (H_TOL, or H_TOL_FAR in the far field). dl0 < 0 = sentinel: both sides compute the stride with their own stepSize from the same
  // (f32-rounded) state and the strides are compared; "retry" and "barrier" supply their own dl0.
  // nCmp = how many leading state components are compared: 8 = the whole state, 4 = the positions
  // only (no case uses it any more; kept for a future far-field precision story).
  type ICase = { label: string; s: Float64Array; a: number; dl0: number; hTol: number; nCmp: number };
  const icases: ICase[] = [];
  const rhOf = (a: number) => 1 + Math.sqrt(Math.max(0, 1 - a * a));
  const SENTINEL = -1;
  /** Walk from s0 with the shipped step controller (renderer tolerances: H_TOL_FAR beyond
   *  rOut * 1.5, H_TOL inside) until pred(s, out) holds; return that pre-step state with the
   *  tolerance the renderer would use there. */
  function findState(label: string, s0: Float64Array, a: number, pred: (s: Float64Array, out: ReturnType<typeof stepGeodesic>) => boolean): ICase {
    let s = s0;
    for (let k = 0; k < 20000; k++) {
      const hTol = s[1] > ROUT * 1.5 ? H_TOL_FAR : H_TOL;
      const out = stepGeodesic(s, a, stepSize(s, rhOf(a), ROUT), hTol);
      if (pred(s, out)) return { label, s, a, dl0: SENTINEL, hTol, nCmp: 8 };
      if (!out.ok || out.s[1] <= rhOf(a) * 1.005 || out.s[1] > 1200) break;
      s = out.s;
    }
    throw new Error(`integrator parity: no state found for case "${label}"`);
  }
  // far field, long stride, a != 0, monitored at H_TOL_FAR. Retries must agree (0/0) and the WHOLE
  // state is compared; the stride is compared through the sentinel (uncapped far branch: K_FAR r
  // clamped to DL_FAR_MAX = 50). History: under finite-difference forces the f32 force at r = 1000
  // was noise (GPU |dH|/scale 2.4e-3 vs 2e-11 in f64), so this case compared positions only and the
  // far field was unmonitored; with exact forces the GPU's |dH|/scale for this step is 2.7e-8 and
  // the momenta agree to <= 3.0e-8 relative, which is what made monitoring the far field possible.
  icases.push({ label: "far", s: screenToState(4, 3, 0.9, 1.2, 1000), a: 0.9, dl0: SENTINEL, hTol: H_TOL_FAR, nCmp: 8 });
  // far field, NEAR THE AXIS: the F_AXIS cap branch of stepSize. The reviewer's streak ray
  // (alpha = 0.1, beta = 12 at i = 8 deg) crosses the axis at r ~ 80, in the far field; the
  // pre-cap stride of 3.2 carried theta to -9.2 there. Found by walking until theta < 0.02 while
  // still in the far field, so the capped stride (well below the K_FAR r base) is what both sides
  // must compute. Monitored at H_TOL_FAR; whole state compared.
  icases.push(findState("far-axis", screenToState(0.1, 12, 0, I8, 1000), 0, (s) => s[2] < 0.02 && s[1] > ROUT * 1.5));
  // far field, the F_PHI azimuthal cap BINDING: a p_phi != 0 ray looking down the axis (i = 1 deg)
  // swings through ~pi of azimuth where it turns in theta, p_theta ~ 0 so F_AXIS is inactive.
  // Found by walking until the azimuthal cap is the smallest stride and above its DL_FAR_MIN floor.
  icases.push(findState("far-phi", screenToState(0.75, 8, 0, Math.PI / 180, 1000), 0, (s) => {
    if (s[1] <= ROUT * 1.5 || s[7] === 0) return false;
    const sn = Math.sin(s[2]), capPhi = F_PHI * s[1] * s[1] * sn * sn / Math.abs(s[7]);
    return capPhi > DL_FAR_MIN && Math.abs(stepSize(s, rhOf(0), ROUT) - capPhi) < 1e-12 * capPhi;
  }));
  // NEAR field, an angular cap BINDING (angularCap also bounds the near branch since 2026-10-01): the
  // alpha = 0.05 ray turns beside the axis at r ~ 43, inside rOut * 1.5. Found by walking until the
  // capped stride is below the near branch's own clamp(0.02 (r - rh), 0.002, 0.5).
  icases.push(findState("near-cap", screenToState(0.05, 6, 0, I8, 1000), 0, (s) =>
    s[1] <= ROUT * 1.5 && stepSize(s, rhOf(0), ROUT) < 0.5 * Math.min(0.5, Math.max(0.002, 0.02 * (s[1] - rhOf(0))))));
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
  // A vacuous case (CPU retries = 0) throws, so the route cannot pass while testing nothing. This
  // and "barrier" are the cases that supply their own dl0 (no sentinel).
  const retryDl0 = 2.0;
  if (stepGeodesic(strongEq.s, strongEq.a, retryDl0, H_TOL).retries < 1) throw new Error("integrator parity: \"retry\" case is vacuous (CPU retries = 0)");
  icases.push({ label: "retry", s: strongEq.s, a: strongEq.a, dl0: retryDl0, hTol: H_TOL, nCmp: 8 });
  // a barely-resolved barrier step near the axis (alpha = 0.05 ray, theta ~ 1.6e-3, dl0 = 0.5 ->
  // accepted at 0.25 after one halving). One RK4 step takes p_theta from -4.20 to -1.32 through
  // the 1/sin^2 barrier. The state IS compared. Both twins now use exact metric derivatives
  // (metricUpperGrad / gUpGrad), so the comparator steps the same equations as the GPU with no
  // finite-difference half-step (earlier revisions had to pass the GPU's h = 1e-4 to the CPU: the
  // CPU default 1e-5 differed by ~0.8 % in p_theta at this theta, and even at h = 1e-4 the state
  // agreed only to 2.3e-4). Measured under exact forces: 6.79e-5 relative, the largest of all 53
  // parity rows (f32 through the barrier; every other integrator case is <= 2.9e-7). The retry
  // count is still the shipped-bytes statement: the GPU must halve exactly where trace.ts does.
  // Since the angular caps bound the near field (2026-10-01) the renderer's own stride resolves
  // this passage and never halves, so the case supplies the pre-cap stride (0.5, the near branch's
  // ceiling) at the first state within 1.6e-3 rad of the axis; a vacuous case (CPU retries = 0) throws.
  const barrierAt = findState("barrier", screenToState(0.05, 6, 0, I8, 1000), 0, (s) => s[2] < 1.6e-3);
  const barrierDl0 = 0.5;
  if (stepGeodesic(barrierAt.s, 0, barrierDl0, H_TOL).retries < 1) throw new Error("integrator parity: \"barrier\" case is vacuous (CPU retries = 0)");
  icases.push({ ...barrierAt, dl0: barrierDl0 });
  // a step that crosses the axis (xi = 0): only reflectAxis can flip the sign of p_theta here
  icases.push(findState("reflect", screenToState(0, 6, 0, I8, 1000), 0, (s, out) => out.s[6] * s[6] < 0));
  const IN_F = 12, OUT_F = 16; // floats per StepIn (48 bytes) / StepRes (64 bytes: x, p, info, extra)
  const iin = device.createBuffer({ size: icases.length * IN_F * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const iarr = new Float32Array(icases.length * IN_F);
  icases.forEach((c, i) => { iarr.set([c.s[0], c.s[1], c.s[2], c.s[3], c.s[4], c.s[5], c.s[6], c.s[7], c.a, c.dl0, c.hTol, ROUT], i * IN_F); });
  device.queue.writeBuffer(iin, 0, iarr);
  const iout = device.createBuffer({ size: icases.length * OUT_F * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const iread = device.createBuffer({ size: icases.length * OUT_F * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const imod = device.createShaderModule({ code: integratorSharedWGSL + integratorParityWGSL });
  const ipipe = device.createComputePipeline({ layout: "auto", compute: { module: imod, entryPoint: "main" } });
  const ibind = device.createBindGroup({ layout: ipipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: iin } }, { binding: 1, resource: { buffer: iout } }] });
  const ienc = device.createCommandEncoder();
  const icp = ienc.beginComputePass(); icp.setPipeline(ipipe); icp.setBindGroup(0, ibind); icp.dispatchWorkgroups(icases.length); icp.end();
  ienc.copyBufferToBuffer(iout, 0, iread, 0, icases.length * OUT_F * 4);
  device.queue.submit([ienc.finish()]);
  await iread.mapAsync(GPUMapMode.READ);
  const igpu = new Float32Array(iread.getMappedRange().slice(0));
  // The GPU starts from the f32-rounded state, so compare against the CPU stepping that same
  // rounded state; otherwise the input rounding (not the shader) would dominate the error. Both
  // twins use exact metric derivatives (metricUpperGrad / gUpGrad), so the CPU steps the same
  // equations as the GPU with no finite-difference half-step to match.
  const cpuStep = (c: ICase) => {
    const s32 = Float64Array.from(Array.from(c.s, Math.fround));
    const dl0 = c.dl0 < 0 ? Math.fround(stepSize(s32, rhOf(c.a), ROUT)) : Math.fround(c.dl0);
    return { dl0, out: stepGeodesic(s32, c.a, dl0, Math.fround(c.hTol), MAX_RETRY) };
  };
  icases.forEach((c, i) => {
    const { dl0, out: cpu } = cpuStep(c);
    for (let k = 0; k < c.nCmp; k++) {
      const got = igpu[i * OUT_F + k], want = cpu.s[k];
      maxErr = Math.max(maxErr, Math.abs(got - want) / (1 + Math.abs(want)));
    }
    // Retry count and ok flag: any mismatch scores >= 1.0 and fails the route outright. A retry
    // mismatch on the "retry" case means H_TOL is at the f32 noise floor (see the "far" and
    // "barrier" comments above for the two precision stories this harness had to separate).
    maxErr = Math.max(maxErr, Math.abs(igpu[i * OUT_F + 8] - cpu.retries));
    maxErr = Math.max(maxErr, Math.abs(igpu[i * OUT_F + 9] - (cpu.ok ? 1 : 0)));
    // The constants themselves, so a desync between trace.ts and the fragment cannot hide.
    maxErr = Math.max(maxErr, Math.abs(igpu[i * OUT_F + 10] - H_TOL) / H_TOL);
    maxErr = Math.max(maxErr, Math.abs(igpu[i * OUT_F + 11] - MAX_RETRY));
    maxErr = Math.max(maxErr, Math.abs(igpu[i * OUT_F + 13] - H_TOL_FAR) / H_TOL_FAR);
    // The stride: the shader's own stepSize on sentinel cases (its far-field caps included), the
    // supplied dl0 otherwise. Relative, like the state.
    maxErr = Math.max(maxErr, Math.abs(igpu[i * OUT_F + 12] - dl0) / (1 + Math.abs(dl0)));
  });
  console.log("integrator parity cases", icases.map((c) => c.label),
    "cpu retries", icases.map((c) => cpuStep(c).out.retries),
    "gpu retries", icases.map((_, i) => igpu[i * OUT_F + 8]),
    "cpu dl0", icases.map((c) => cpuStep(c).dl0.toPrecision(6)),
    "gpu dl0", icases.map((_, i) => igpu[i * OUT_F + 12].toPrecision(6)),
    "state relErr", icases.map((c, i) => { const { out } = cpuStep(c); let e = 0; for (let k = 0; k < c.nCmp; k++) e = Math.max(e, Math.abs(igpu[i * OUT_F + k] - out.s[k]) / (1 + Math.abs(out.s[k]))); return e.toExponential(2); }));
  return { maxErr, rows: cases.length + tcases.length + jcases.length + scases.length + ccases.length + icases.length };
}
