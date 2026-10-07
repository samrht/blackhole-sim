import { metricUpper, metricLower } from "../physics/kerr";
import { omegaKepler } from "../physics/orbits";
import { gFactorKepler } from "../physics/redshift";
import parityWGSL from "../render/parity.wgsl?raw";
import integratorSharedWGSL from "../render/integrator-shared.wgsl?raw";
import { turbulenceAt } from "../physics/emission";
import turbParityWGSL from "../render/turb-parity.wgsl?raw";
import emissionSharedWGSL from "../render/emission-shared.wgsl?raw";
import jetParityWGSL from "../render/jet-parity.wgsl?raw";
import fluxParityWGSL from "../render/flux-parity.wgsl?raw";
import flowParityWGSL from "../render/flow-parity.wgsl?raw";
import hotspotParityWGSL from "../render/hotspot-parity.wgsl?raw";
import { HOTSPOT, hotspotAt, hotspotBoost, hotspotShift, hotspotPeriod, hotspotRadius } from "../physics/hotspot";
import { HOTFLOW, flowVelocity, flowShift, flowCoeffs, flowDensity } from "../physics/hot-flow";
import { fluxRatio, eruptionTime } from "../physics/flux-history";
import { jetShape, launchDelay, comovingAzimuth, filaments } from "../physics/jet";
import { plasmaShift, streamlineDir, gammaProfile, jetField, jetCoeffs, flowTime, slabStep, JET_BANDS_NM, C_CGS, LN_K0 } from "../physics/synchrotron";
import { parseTable } from "../physics/cyclosynch";
import { classify, criticalXiEta, photonShellRange } from "../physics/shadow";
import shadowSharedWGSL from "../render/shadow-shared.wgsl?raw";
import shadowParityWGSL from "../render/shadow-parity.wgsl?raw";
import { screenToState, screenToXiEta } from "../physics/camera";
import cameraSharedWGSL from "../render/camera-shared.wgsl?raw";
import cameraParityWGSL from "../render/camera-parity.wgsl?raw";
import { stepGeodesic, stepSize, H_TOL, H_TOL_FAR, MAX_RETRY, F_PHI, DL_FAR_MIN } from "../physics/trace";
import integratorParityWGSL from "../render/integrator-parity.wgsl?raw";
import minoParityWGSL from "../render/mino-parity.wgsl?raw";
import { minoRay, minoInit, minoRhs, minoStep, minoTry, minoDense, minoToState, minoHemi, minoCrossing, minoLand, MINO_TOL as MINO_TOL_PARITY, MINO_MAX_REJECT as MINO_MAX_REJECT_PARITY, MINO_UFRAC as MINO_UFRAC_PARITY } from "../physics/trace-mino";

/** Runs the WGSL metric/orbit/g-factor helpers on fixed inputs and returns the max relative
 *  error vs the TypeScript core. f32 GPU vs f64 CPU keeps this in the ~1e-6..1e-4 range. */
export async function runParity(): Promise<{ maxErr: number; rows: number; jetLogErr: number; turbErr: number; turbWorst: string; turbRough: { gpu: number; old: number; cpu: number }; fluxErr: number; fluxWorst: string; flowErr: number; flowWorst: string; hsErr: number; hsWorst: string; minoErr: number; minoWorst: string }> {
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
  // --- turbulence parity (CPU emission.ts turbulenceAt vs the shipped turbulenceFieldE) ---
  // The shader takes time as an epoch t0 (a multiple of 2048, exact in f32) plus a small remainder tRel,
  // as the renderer does (sim-clock.ts). Late (t0 ~ 2e5), negative (-50) and both spin extremes are
  // Review Focus 1-3. Then a line of 256 pixels at t ~ 1.8e6 M whose emission times step by 0.05 M (a
  // light-delay gradient) checks smoothness: the old composition, f32(t - delay), turned this into grain.
  const tcases = [
    { r: 6, phi: 0.4, t0: 0, tr: 1.7, a: 0.9 }, { r: 9, phi: 1.7, t0: 0, tr: 500, a: 0.9 },
    { r: 14, phi: -3.9, t0: 4096, tr: 904, a: 0.9 }, { r: 22, phi: 5.2, t0: 198656, tr: 1344, a: 0.9 },
    { r: 4, phi: 2.2, t0: 0, tr: -50, a: 0.9 }, { r: 7, phi: 0.9, t0: 0, tr: 333, a: 0 },
    { r: 1.3, phi: 4.4, t0: 0, tr: 81, a: 0.998 }, { r: 35, phi: -0.6, t0: 12288, tr: 57, a: 0.5 },
  ];
  const LINE_N = 256, LINE_T0 = 1835008, LINE_R = 3, LINE_PHI = 0.7, LINE_A = 0.9;
  const lineRel = (k: number) => 500 - 0.05 * k;
  const nT = tcases.length + 2 * LINE_N;
  const tarr = new Float32Array(nT * 8);
  tcases.forEach((c, i) => { tarr.set([c.r, c.phi, c.t0, c.tr, c.a, 0, 0, 0], i * 8); });
  for (let k = 0; k < LINE_N; k++) {
    tarr.set([LINE_R, LINE_PHI, LINE_T0, lineRel(k), LINE_A, 0, 0, 0], (tcases.length + k) * 8);                    // epoch + remainder
    tarr.set([LINE_R, LINE_PHI, Math.fround(LINE_T0 + lineRel(k)), 0, LINE_A, 0, 0, 0], (tcases.length + LINE_N + k) * 8); // old: f32(t - delay)
  }
  const tin = device.createBuffer({ size: tarr.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(tin, 0, tarr);
  const tout = device.createBuffer({ size: nT * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const tread = device.createBuffer({ size: nT * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const tmod = device.createShaderModule({ code: integratorSharedWGSL + emissionSharedWGSL + turbParityWGSL }) // emission-shared uses gUp;
  const tpipe = device.createComputePipeline({ layout: "auto", compute: { module: tmod, entryPoint: "main" } });
  const tbind = device.createBindGroup({ layout: tpipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: tin } }, { binding: 1, resource: { buffer: tout } }] });
  const tenc = device.createCommandEncoder();
  const tcp = tenc.beginComputePass(); tcp.setPipeline(tpipe); tcp.setBindGroup(0, tbind); tcp.dispatchWorkgroups(Math.ceil(nT / 64)); tcp.end();
  tenc.copyBufferToBuffer(tout, 0, tread, 0, nT * 4);
  device.queue.submit([tenc.finish()]);
  await tread.mapAsync(GPUMapMode.READ);
  const tgpu = new Float32Array(tread.getMappedRange().slice(0));
  // Tolerance per case: 2e-3, plus 8x the field's change over one f32 step of the epoch t0. The GPU's
  // t0 / T_c rounds by a few such steps: one offset per lattice row and epoch, shared by every pixel (a
  // coherent time shift); the CPU, in f64, sees the exact time. turbErr is the worst |d g| / tolerance.
  let turbErr = 0, turbWorst = "";
  tcases.forEach((c, i) => {
    const r = Math.fround(c.r), phi = Math.fround(c.phi), t = c.t0 + Math.fround(c.tr), a = Math.fround(c.a);
    const cpu = turbulenceAt(r, phi, t, a);
    const ulpT = Math.max(Math.abs(c.t0), 1) * 2 ** -23;
    const tol = 2e-3 + 8 * Math.abs(turbulenceAt(r, phi, t + ulpT, a) - cpu);
    const e = Math.abs(tgpu[i] - cpu) / tol;
    if (e > turbErr) { turbErr = e; turbWorst = `t=${t}: |d g| ${Math.abs(tgpu[i] - cpu).toExponential(2)} tol ${tol.toExponential(2)}`; }
  });
  // Smoothness: rms second difference along the line, GPU (epoch + remainder) against the exact field.
  const rough = (v: (k: number) => number) => {
    let s = 0; for (let k = 1; k < LINE_N - 1; k++) s += (v(k + 1) - 2 * v(k) + v(k - 1)) ** 2;
    return Math.sqrt(s / (LINE_N - 2));
  };
  const turbRough = {
    gpu: rough((k) => tgpu[tcases.length + k]),
    old: rough((k) => tgpu[tcases.length + LINE_N + k]),
    cpu: rough((k) => turbulenceAt(LINE_R, Math.fround(LINE_PHI), LINE_T0 + Math.fround(lineRel(k)), LINE_A)),
  };
  // --- jet parity (CPU synchrotron.ts / cyclosynch.ts / jet.ts vs the SHIPPED emission-shared.wgsl) ---
  // Cases are real photon states inside the jet: rays walked with the CPU integrator from the camera until
  // the sample lies in the emitting region, so D sees realistic momenta (approaching and receding lobes).
  // Two jets: an M87*-like one (b0 632 G; visible far above the cyclotron frequency) and Cygnus X-1's
  // (b0 3.8e8 G; visible near the cyclotron frequency, where the table holds the exact harmonic sums).
  const T = parseTable(await (await fetch("/synch-table.bin")).arrayBuffer());
  const synchTex = device.createTexture({ size: [T.nx, T.ns], format: "rg32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: synchTex }, T.data, { bytesPerRow: T.nx * 8 }, [T.nx, T.ns]);
  const JETS = [{ b0: 632.39, rg: 9.5984e14, q0: 1.8049e-9 }, { b0: 3.7838e8, rg: 3.1305e6, q0: 0.86887 }]; // M87*, Cyg X-1 at ETA_DEFAULT
  // The GPU gets the emission time as clock epoch + f32 remainder; the CPU uses the same f32-rounded remainder.
  const J_LEN = 60, J_FLUX = 1, T_EM = Math.fround(1.7), J_EPOCH = 2048 * 3;
  type JCase = { s: Float64Array; a: number; g280: number; jet: number; ds: number };
  const jcases: JCase[] = [];
  const rays: [number, number, number, number, number, number][] = [ // alpha, beta, a, incl (rad), g280, jet
    [0.5, 4, 0.9, (17 * Math.PI) / 180, 2, 0], [1.5, 8, 0.9, (17 * Math.PI) / 180, 4, 0],
    [0.5, -4, 0.5, (60 * Math.PI) / 180, 2, 0], [1.5, 8, 0.5, (60 * Math.PI) / 180, 6, 0],
    [0.3, 6, 0.998, (27 * Math.PI) / 180, 2, 1], [0.8, -6, 0.998, (27 * Math.PI) / 180, 3, 1],
  ];
  const planeCoeffs = (s: Float64Array, a: number, g280: number, jet: number, nm: number, D: number) => {
    const r = s[1], th = s[2], z = r * Math.cos(th), J = JETS[jet], B = jetField(r * Math.sin(th), z, a, J.b0);
    const sc = Math.exp(LN_K0) * B * B * flowTime(z, g280, J.rg), shape = jetShape(r, th, s[3], J_EPOCH + T_EM, J_LEN, J_FLUX, g280, a);
    return jetCoeffs(T, (C_CGS / (nm * 1e-7)) * D, B, sc, J.q0, shape);
  };
  for (const [al, be, a, inc, g280, jet] of rays) {
    let s = screenToState(al, be, a, inc, 1000);
    const rh = 1 + Math.sqrt(1 - a * a);
    for (let k = 0; k < 20000 && jetShape(s[1], s[2], s[3], J_EPOCH + T_EM, J_LEN, J_FLUX, g280, a) <= 0; k++) {
      const out = stepGeodesic(s, a, stepSize(s, rh, 40), s[1] > 60 ? H_TOL_FAR : H_TOL);
      if (!out.ok || out.s[1] <= rh * 1.005 || out.s[1] > 1200) break;
      s = out.s;
    }
    if (jetShape(s[1], s[2], s[3], J_EPOCH + T_EM, J_LEN, J_FLUX, g280, a) <= 0) throw new Error(`jet parity: ray (${al}, ${be}) never entered the jet`);
    // ds from this sample's own 550 nm absorption: the two slab steps are genuinely thin (dtau = 0.01) or thick (20)
    const D0 = plasmaShift(s, a, gammaProfile(s[1] * Math.cos(s[2]), g280), ...streamlineDir(s[1], s[2]));
    const al0 = planeCoeffs(s, a, g280, jet, 550, D0)[1];
    jcases.push({ s, a, g280, jet, ds: (jcases.length % 2 ? 20 : 0.01) / al0 });
  }
  const J_IN = 16, J_OUT = 12;
  const jin = device.createBuffer({ size: jcases.length * J_IN * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const jarr = new Float32Array(jcases.length * J_IN);
  jcases.forEach((c, i) => { const J = JETS[c.jet]; jarr.set([...c.s.slice(0, 8), c.a, c.g280, T_EM, c.ds, J.b0, J.q0, J.rg, J_EPOCH], i * J_IN); });
  device.queue.writeBuffer(jin, 0, jarr);
  const jout = device.createBuffer({ size: jcases.length * J_OUT * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const jread = device.createBuffer({ size: jcases.length * J_OUT * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const jmod = device.createShaderModule({ code: integratorSharedWGSL + emissionSharedWGSL + jetParityWGSL });
  const jpipe = device.createComputePipeline({ layout: "auto", compute: { module: jmod, entryPoint: "main" } });
  const jbind = device.createBindGroup({ layout: jpipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: jin } }, { binding: 1, resource: { buffer: jout } }, { binding: 10, resource: synchTex.createView() }] });
  const jenc = device.createCommandEncoder();
  const jcp = jenc.beginComputePass(); jcp.setPipeline(jpipe); jcp.setBindGroup(0, jbind); jcp.dispatchWorkgroups(jcases.length); jcp.end();
  jenc.copyBufferToBuffer(jout, 0, jread, 0, jcases.length * J_OUT * 4);
  device.queue.submit([jenc.finish()]);
  await jread.mapAsync(GPUMapMode.READ);
  const jgpu = new Float32Array(jread.getMappedRange().slice(0));
  // D and shape: relative metric (like every other case); ln outputs and tau: ABSOLUTE error, gated at 2e-3
  // (f32 logs of j ~ e^-50 carry ~4e-6 absolute; a relative metric on logs would tolerate ~5 % in j).
  let jetLogErr = 0; const jetErrs: string[] = [];
  jcases.forEach((c, i) => {
    const s = Float64Array.from(jarr.subarray(i * J_IN, i * J_IN + 8)); // the same f32-rounded inputs the GPU got
    const r = s[1], th = s[2], z = r * Math.cos(th);
    const D = plasmaShift(s, c.a, gammaProfile(z, c.g280), ...streamlineDir(r, th));
    const shape = jetShape(r, th, s[3], J_EPOCH + T_EM, J_LEN, J_FLUX, c.g280, c.a);
    const ja = JET_BANDS_NM.map((nm) => planeCoeffs(s, c.a, c.g280, c.jet, nm, D));
    const ds = jarr[i * J_IN + 11];
    const slab = (b: number) => { let I = 0, tau = 0; const j = ja[b][0] / D ** 3;
      for (let k = 0; k < 2; k++) [I, tau] = slabStep(I, tau, j, ja[b][1], ds); return [Math.log(I), tau]; };
    const want = [D, shape, Math.log(ja[0][0] / D ** 3), Math.log(ja[1][0] / D ** 3),
      Math.log(ja[2][0] / D ** 3), Math.log(ja[0][1]), Math.log(ja[1][1]), Math.log(ja[2][1]), ...slab(0), ...slab(2)];
    let eRel = 0, eLog = 0;
    for (let k = 0; k < J_OUT; k++) { const d = Math.abs(jgpu[i * J_OUT + k] - want[k]);
      if (k < 2) eRel = Math.max(eRel, d / (1 + Math.abs(want[k]))); else eLog = Math.max(eLog, k === 9 || k === 11 ? d / (1 + Math.abs(want[k])) : d); }
    maxErr = Math.max(maxErr, eRel); jetLogErr = Math.max(jetLogErr, eLog);
    jetErrs.push(`${eRel.toExponential(1)}/${eLog.toExponential(1)}`);
  });
  console.log("jet parity per case (rel D,shape / abs ln)", jetErrs);
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
  const cmod = device.createShaderModule({ code: integratorSharedWGSL + cameraSharedWGSL + cameraParityWGSL }); // sinCosP
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
  // --- Mino-time integrator (CPU trace-mino.ts vs the SHIPPED minoRhs / minoStep / minoDense / minoToState in
  // integrator-shared.wgsl): ONE step from an identical f32 state per case (see mino-parity.wgsl for why). States are
  // found by walking the CPU twin: the camera's first step at r = 1000; a ray at r ~ 60; near the horizon at a = 0.9 and
  // 0.998; near the axis (xi = -0.001); at a radial turning point (photon-ring ray); a forced reject (the r ~ 60
  // state with its proposal x10); a state just past the equator (the step mirrors it into the south frame); and near the
  // SOUTH pole in that frame (the i = 1 deg ray that grazes it).
  type MCase = { label: string; y: Float64Array; a: number; xi: number; eta: number; h: number };
  const mcases: MCase[] = [];
  const mkRay = (al: number, be: number, a: number, iDeg: number) => {
    const i = (iDeg * Math.PI) / 180, xi = -al * Math.sin(i), ci = Math.cos(i), si = Math.sin(i);
    return { th0: i, be, a: Math.fround(a), xi: Math.fround(xi), eta: Math.fround(be * be + (xi * xi * ci * ci) / Math.max(si * si, 1e-8) - a * a * ci * ci) };
  };
  /** Walk the CPU twin from the camera until pred holds BEFORE a step; that state and its proposal are the case. */
  const caseAt = (label: string, r: ReturnType<typeof mkRay>, pred: (y: Float64Array, prev: Float64Array) => boolean, mul = 1) => {
    const c = minoRay(r.a, r.xi, r.eta); let y = minoInit(1000, r.th0, r.be, c), prev = y, f = minoRhs(y, c), h = 50 / 1e6;
    for (let k = 0; k < 4000 && !pred(y, prev); k++) { const o = minoStep(y, h, c, undefined, undefined, f); prev = y; y = o.y; f = o.f1; h = o.hNext; }
    mcases.push({ label, y: Float64Array.from(y, Math.fround), a: r.a, xi: r.xi, eta: r.eta, h: Math.fround(h * mul) });
  };
  const def = mkRay(4, 3, 0.9, 72);
  caseAt("first", def, () => true);
  caseAt("r~60", def, (y) => 1 / y[1] < 60);
  caseAt("near-horizon a=0.9", mkRay(2, 0.5, 0.9, 90), (y) => 1 / y[1] < 1.6);
  caseAt("near-horizon a=0.998", mkRay(1.5, 0.5, 0.998, 90), (y) => 1 / y[1] < 1.15);
  caseAt("near-axis", mkRay(0.0072, 6, 0, 8), (y) => Math.sin(y[2]) ** 2 < 1e-3);
  caseAt("turning point", mkRay(5.3, 0, 0, 90), (y, prev) => y[5] * prev[5] < 0);
  caseAt("forced reject", def, (y) => 1 / y[1] < 60, 10);
  caseAt("past equator", mkRay(6, 0.5, 0.9, 80), (y) => y[2] > Math.PI / 2);
  caseAt("near south pole", mkRay(-0.52, 13.48, 0.9, 1), (y) => y[7] < 0 && Math.sin(y[2]) ** 2 < 1e-3);
  if (!mcases.some((m) => m.label === "past equator" && m.y[2] > Math.PI / 2)) throw new Error("mino parity: no past-equator case");
  { // the state whose step crosses the plane (the landing step's case)
    const r = mkRay(8, 3, 0.9, 72), c = minoRay(r.a, r.xi, r.eta); let y = minoInit(1000, r.th0, r.be, c), f = minoRhs(y, c), h = 50 / 1e6;
    for (let k = 0; k < 4000; k++) {
      const o = minoStep(y, h, c, undefined, undefined, f);
      if (minoCrossing(o.y0, o.y, o.f0, o.f1, o.h) >= 0) { mcases.push({ label: "plane crossing", y: Float64Array.from(y, Math.fround), a: r.a, xi: r.xi, eta: r.eta, h: Math.fround(h) }); break; }
      y = o.y; f = o.f1; h = o.hNext;
    }
    if (!mcases.some((m) => m.label === "plane crossing")) throw new Error("mino parity: no plane-crossing case");
  }
  if (!mcases.some((m) => m.label === "near south pole" && m.y[7] < 0 && Math.sin(m.y[2]) ** 2 < 1e-3)) throw new Error("mino parity: no south-pole case");
  const mIn = new Float32Array(mcases.length * 12);
  mcases.forEach((m, i) => mIn.set([m.y[0], m.y[1], m.y[2], m.y[3], m.y[4], m.y[5], m.y[6], m.y[7], m.a, m.xi, m.eta, m.h], i * 12));
  const mInBuf = device.createBuffer({ size: mIn.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(mInBuf, 0, mIn);
  const MOUT = 44; // floats per MOut (11 vec4)
  const mOutBuf = device.createBuffer({ size: mcases.length * MOUT * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const mRead = device.createBuffer({ size: mcases.length * MOUT * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const mMod = device.createShaderModule({ code: integratorSharedWGSL + minoParityWGSL });
  const mPipe = device.createComputePipeline({ layout: "auto", compute: { module: mMod, entryPoint: "main" } });
  const mBind = device.createBindGroup({ layout: mPipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: mInBuf } }, { binding: 1, resource: { buffer: mOutBuf } }] });
  const mEnc = device.createCommandEncoder();
  const mCp = mEnc.beginComputePass(); mCp.setPipeline(mPipe); mCp.setBindGroup(0, mBind); mCp.dispatchWorkgroups(mcases.length); mCp.end();
  mEnc.copyBufferToBuffer(mOutBuf, 0, mRead, 0, mcases.length * MOUT * 4);
  device.queue.submit([mEnc.finish()]);
  await mRead.mapAsync(GPUMapMode.READ);
  const mGpu = new Float32Array(mRead.getMappedRange().slice(0));
  // Errors / tolerance: attempts exact; the error norm within 0.02 + 10 % (it decides acceptance near 1; far below 1 it is
  // f32 noise); w, l relative 1e-5; u, u' absolute 1e-5 (u' against max(1, |u'|)); t, phi against max(1, |value|) at
  // 1e-5; w' against max(|w'|, w) at 1e-5; the State: r relative 1e-5, theta absolute 1e-5, p_r and p_theta 1e-4 against
  // max(1, |value|); the constants exact.
  let minoErr = 0, minoWorst = "";
  const mset = (i: number, what: string, g: number, w: number, tol: number) => { const e = Math.abs(g - w) / tol;
    if (!(e <= minoErr)) { minoErr = Number.isFinite(e) ? e : Infinity; minoWorst = `${mcases[i].label} ${what}: gpu ${g} cpu ${w}`; } };
  const diag: string[] = [];
  mcases.forEach((m, i) => {
    const g = Array.from(mGpu.subarray(i * MOUT, i * MOUT + MOUT));
    // After a rejected attempt the accepted h derives from the attempt's f32 error norm, so it differs from the CPU's in
    // the 4th digit; the state is compared against ONE CPU attempt at the GPU's own h, and the two h are compared.
    // eps = 2^-23: the expectation models the f32 GPU's drift-noise floor (MINO_EPS)
    // the step mirrors a past-equator start into the south frame first (minoHemi); the expectation starts from that state
    const c = minoRay(m.a, m.xi, m.eta), o = minoStep(m.y, m.h, c, undefined, undefined, undefined, undefined, 2 ** -23), y0 = minoHemi(m.y), f0 = minoRhs(y0, c), tr = minoTry(y0, g[29], c, f0, undefined, undefined, 2 ** -23);
    const st = minoToState(tr.y1, c), d = minoDense(y0, tr.y1, f0, tr.f1, g[29], 0.37);
    mset(i, "h used", g[29], o.h, 1e-2 * o.h); // after a reject h derives from the rejected attempt's f32 error norm
    const cmpY = (base: number, yy: Float64Array, tag: string) => {
      mset(i, tag + "t", g[base], yy[0], 1e-5 * Math.max(1, Math.abs(yy[0]))); mset(i, tag + "w", g[base + 1], yy[1], 1e-5 * yy[1]);
      mset(i, tag + "theta", g[base + 2], yy[2], 1e-5 * Math.max(1, Math.abs(yy[2]))); mset(i, tag + "phi", g[base + 3], yy[3], 1e-5 * Math.max(1, Math.abs(yy[3])));
      mset(i, tag + "l", g[base + 4], yy[4], 1e-5 * Math.max(1, Math.abs(yy[4]))); mset(i, tag + "w'", g[base + 5], yy[5], 1e-5 * Math.max(Math.abs(yy[5]), yy[1]));
      mset(i, tag + "theta'", g[base + 6], yy[6], 1e-5 * Math.max(1, Math.abs(yy[6])));
      mset(i, tag + "sigma", g[base + 7], yy[7], 1e-6); // hemisphere sign, exactly +-1
    };
    mset(i, "attempts", g[9], o.attempts, 1e-6);
    // 10 %: near the horizon of a = 0.998 the f32 estimate reads 8 % low (1/Delta cancellation); that moves the accepted h
    // by < 2 % (err ~ h^5). Whole-ray accuracy on the GPU is gated by ?accuracy.
    // + 0.1 absolute: far below 1 the f32 estimate is noise (first step: 0.054 vs 0.0007); acceptance is decided near 1 and
    // the attempts, compared exactly above, prove those decisions agree.
    mset(i, "err norm", g[10], o.en, 0.1 + 0.1 * o.en);
    cmpY(0, tr.y1, ""); cmpY(20, d, "dense ");
    // the landing step at the GPU's crossing fraction (bisection on the dense f32 cos(theta): th agrees to ~1e-6)
    const th = minoCrossing(y0, tr.y1, f0, tr.f1, g[29]);
    mset(i, "crossing found", g[41], th >= 0 ? 1 : 0, 1e-6);
    if (th >= 0 && g[41] === 1) { mset(i, "crossing th", g[40], th, 1e-4); cmpY(32, minoLand(y0, f0, g[29], g[40], c), "landed "); }
    mset(i, "r", g[13], st[1], 1e-5 * st[1]); mset(i, "theta", g[14], st[2], 1e-5);
    mset(i, "p_r", g[17], st[5], 1e-4 * Math.max(1, Math.abs(st[5]))); mset(i, "p_theta", g[18], st[6], 1e-4 * Math.max(1, Math.abs(st[6])));
    mset(i, "p_phi", g[19], st[7], 1e-6 * Math.max(1, Math.abs(st[7])));
    mset(i, "MINO_TOL", g[11], Math.fround(MINO_TOL_PARITY), 1e-6 * MINO_TOL_PARITY); mset(i, "MINO_MAX_REJECT", g[28], MINO_MAX_REJECT_PARITY, 1e-6);
    mset(i, "MINO_UFRAC", g[30], Math.fround(MINO_UFRAC_PARITY), 1e-6);
    diag.push(`${m.label}: att ${g[9]}/${o.attempts} en ${g[10].toPrecision(3)}/${o.en.toPrecision(3)} h ${g[29].toPrecision(4)}/${o.h.toPrecision(4)}`);
  });
  console.log("mino parity worst (|err| / tol)", minoErr.toExponential(2), minoWorst, "|", diag.join(" | "));
  // --- flux history / launch delay / co-moving azimuth / filaments (CPU flux-history.ts + jet.ts vs the SHIPPED
  // emission-shared.wgsl). Epochs up to 2048 x 8000 (~16 M M: hours of play, Review Focus 1); rel spans eruption
  // drops and refills and negative launch remainders; both lobes; spins 0, 0.9, 0.998.
  const fcases: number[][] = [];
  for (const epoch of [0, 2048 * 7, 2048 * 5000, 2048 * 8000])
    for (const rel of [-900.5, 13.25, 377.75, 1024.5, 1999.875])
      for (const [z, g, q, ph, a, s] of [[5, 2, 0.8, 0.3, 0.9, 1], [-40, 5, 1.1, -7.2, 0.998, 1.4], [59, 1.5, 0.2, 12.9, 0, 0.5], [20, 3, 0.6, 2.2, 0.6, 1], [33, 2, 0.9, 4.4, 0.45, 1.2]])
        fcases.push([epoch, rel, z, g, q, ph, a, s]);
  const farr = new Float32Array(fcases.length * 8);
  fcases.forEach((c, i) => farr.set(c, i * 8));
  const fin = device.createBuffer({ size: farr.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(fin, 0, farr);
  const fout = device.createBuffer({ size: fcases.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const fread = device.createBuffer({ size: fcases.length * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const fmod = device.createShaderModule({ code: integratorSharedWGSL + emissionSharedWGSL + fluxParityWGSL });
  const fpipe = device.createComputePipeline({ layout: "auto", compute: { module: fmod, entryPoint: "main" } });
  const fbind = device.createBindGroup({ layout: fpipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: fin } }, { binding: 1, resource: { buffer: fout } }] });
  const fenc = device.createCommandEncoder();
  const fcp = fenc.beginComputePass(); fcp.setPipeline(fpipe); fcp.setBindGroup(0, fbind); fcp.dispatchWorkgroups(fcases.length); fcp.end();
  fenc.copyBufferToBuffer(fout, 0, fread, 0, fcases.length * 16);
  device.queue.submit([fenc.finish()]);
  await fread.mapAsync(GPUMapMode.READ);
  const fgpu = new Float32Array(fread.getMappedRange().slice(0));
  let fluxErr = 0, fluxWorst = "";
  fcases.forEach((_, i) => {
    const [epoch, rel, z, g, q, ph, a, s] = Array.from(farr.subarray(i * 8, i * 8 + 8)); // f32-rounded, as the GPU got
    const t = epoch + rel, tau = launchDelay(z, g);
    // Errors: absolute on f and filaments (bounded ~0.2-1.6 and 0.65-1.35), relative on tau, absolute on the azimuth.
    const want = [fluxRatio(t, s, a), tau, comovingAzimuth(ph, z, a, g), filaments(q, ph, t)];
    const tol = [2e-3, 1e-4 * Math.max(1, tau), 2e-3, 2e-3];
    for (let k = 0; k < 4; k++) { const e = Math.abs(fgpu[i * 4 + k] - want[k]) / tol[k];
      if (e > fluxErr) { fluxErr = e; fluxWorst = `case ${i} out ${k}: gpu ${fgpu[i * 4 + k]} cpu ${want[k]}`; } }
  });
  console.log("flux parity worst (|err| / tol)", fluxErr.toExponential(2), fluxWorst);
  // --- hot flow (CPU hot-flow.ts vs the SHIPPED flowVelocityJ / flowShiftJ / flowCoeffsJ in emission-shared.wgsl) ---
  // Spins 0.1 / 0.94; r from inside the ISCO (plunge, near the horizon) to the edge of the flow; latitudes at the plane,
  // mid, near the axis (th 0.15: density down e^-22) and on it (th 0.05: ln n < -40, the GPU must return j = 0);
  // n0 at M87* and Sgr A*. Each point gets two null momenta (p_t = 1, prograde and retrograde, in- and outgoing),
  // with p_r from the null condition and L shrunk until one exists, so D sees both blue- and redshifts.
  const hcases: { s: Float64Array; a: number; n0: number }[] = [];
  for (const a of [0.1, 0.94]) for (const r of [1.5, 2.5, 5, 12, 30, 49]) for (const th of [1.55, 1.0, 0.15, 0.05]) {
    if (r <= (1 + Math.sqrt(1 - a * a)) * 1.01) continue;
    const gu = metricUpper(r, th, a);
    for (const [L0, pth, sgn] of [[2, 0.5, 1], [-3, -1, -1]]) {
      let L = L0, R = -(gu.tt + 2 * gu.tphi * L + gu.thth * pth * pth + gu.phph * L * L);
      for (let k = 0; k < 30 && R <= 0; k++) { L *= 0.5; R = -(gu.tt + 2 * gu.tphi * L + gu.thth * pth * pth + gu.phph * L * L); }
      for (const n0 of [5.03e5, 1.5e7]) hcases.push({ s: Float64Array.from([0, r, th, 0, 1, sgn * Math.sqrt(R / gu.rr), pth, L]), a, n0 });
    }
  }
  const harr = new Float32Array(hcases.length * 8);
  hcases.forEach((c, i) => harr.set([c.s[1], c.s[2], c.a, c.n0, c.s[4], c.s[5], c.s[6], c.s[7]], i * 8));
  const hin = device.createBuffer({ size: harr.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(hin, 0, harr);
  const hout = device.createBuffer({ size: hcases.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const hread = device.createBuffer({ size: hcases.length * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const hmod = device.createShaderModule({ code: integratorSharedWGSL + emissionSharedWGSL + flowParityWGSL });
  const hpipe = device.createComputePipeline({ layout: "auto", compute: { module: hmod, entryPoint: "main" } });
  const hbind = device.createBindGroup({ layout: hpipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: hin } }, { binding: 1, resource: { buffer: hout } }] });
  const henc = device.createCommandEncoder();
  const hcp = henc.beginComputePass(); hcp.setPipeline(hpipe); hcp.setBindGroup(0, hbind); hcp.dispatchWorkgroups(hcases.length); hcp.end();
  henc.copyBufferToBuffer(hout, 0, hread, 0, hcases.length * 16);
  device.queue.submit([henc.finish()]);
  await hread.mapAsync(GPUMapMode.READ);
  const hgpu = new Float32Array(hread.getMappedRange().slice(0));
  // Errors / tolerance: 1e-3 on D and u^t (the metric cases' tolerance above), D's measured against the size of its terms
  // |u^t p_t| + |u^r p_r| + |Omega u^t p_phi| (it cancels near the horizon); absolute 2e-3 on ln j and ln alpha (as for
  // the jet). Why not tighter: WGSL allows sin/cos 2^-11 absolute error and the Intel iGPU's sin(1.55) is off by ~3e-5,
  // so gLow's sin^2 carries ~7e-5; the K0 cancellation lifts that to 2.4e-4 in u^t at r = 1.5, a = 0.94 (the
  // renderer's metric shares it). No velocity (K0 <= 0) must come back as D = -1, u^t = 0; ln n < -40 as j = 0
  // (the shader writes ln j = -1e30).
  let flowErr = 0, flowWorst = "", nNoVel = 0, nCut = 0, nTiny = 0, nEmit = 0;
  const hbad = (i: number, k: number, g: number, w: number) => { flowErr = Infinity; flowWorst = `case ${i} out ${k}: gpu ${g} cpu ${w}`; };
  hcases.forEach((c, i) => {
    const v = Array.from(harr.subarray(i * 8, i * 8 + 8)), s = Float64Array.from([0, v[0], v[1], 0, v[4], v[5], v[6], v[7]]);
    const [r, th, a, n0] = v, g = Array.from(hgpu.subarray(i * 4, i * 4 + 4)), u = flowVelocity(r, th, a), D = flowShift(s, a);
    if (!u || D === null) { nNoVel++; if (g[0] !== -1 || g[3] !== 0) hbad(i, 0, g[0], NaN); return; }
    const rel = (k: number, w: number, scale: number) => { const e = Math.abs(g[k] - w) / scale / 1e-3; if (e > flowErr) { flowErr = e; flowWorst = `case ${i} out ${k}: gpu ${g[k]} cpu ${w}`; } };
    rel(0, D, Math.abs(u.ut * s[4]) + Math.abs(u.ur * s[5]) + Math.abs(u.Om * u.ut * s[7])); rel(3, u.ut, u.ut);
    if (D <= 0) return;
    if (Math.log(flowDensity(r, th, n0)) < -40) { nCut++; if (g[1] !== Math.fround(-1e30)) hbad(i, 1, g[1], -1e30); return; }
    // at the GPU's own D, so this gates flowCoeffsJ alone (D is gated above; alpha ~ nu^-3 would re-count its error)
    const [j, al] = flowCoeffs(r, th, g[0] * HOTFLOW.nu, n0);
    // Far out on the 230 GHz Wien tail j drops below f32's smallest normal (e^-87.3; the CPU's own f64 M(X) underflows
    // to 0 further out): there the shader must return none, and within e^7 of that edge either answer is right.
    if (!(Math.log(j) > -80)) {
      if (g[1] === Math.fround(-1e30)) { nTiny++; return; }
      if (!(Math.log(j) > -87.3)) { nTiny++; hbad(i, 1, g[1], Math.log(j)); return; }
    }
    nEmit++;
    for (const [k, w] of [[1, Math.log(j)], [2, Math.log(al)]]) { const e = Math.abs(g[k] - w) / 2e-3;
      if (e > flowErr) { flowErr = e; flowWorst = `case ${i} out ${k}: gpu ${g[k]} cpu ${w}`; } }
  });
  console.log("flow parity worst (|err| / tol)", flowErr.toExponential(2), flowWorst, "cases", hcases.length,
    `(no velocity ${nNoVel}, density cut ${nCut}, j below f32 ${nTiny}, compared ${nEmit})`);
  // --- hotspots (CPU hotspot.ts vs the SHIPPED hotspotStateJ / hotspotBoostJ / hotspotShiftJ in emission-shared.wgsl) ---
  // Times across eruption k's life (before birth, the rise, the peak, mid-life, the cut, after) for k from 0 to 5000
  // (clock up to 7.5e6 M: the f32 epoch split must stay exact), spins 0 / 0.94 / 0.998, slider 0 / 0.6 / 1.4; points at the
  // centre, 2-3 M off it in r, above the plane, ahead in phi, and near the horizon (r 1.6: not timelike, D_h = -1).
  const hsCases: { a: number[]; b: number[]; c: number[] }[] = [];
  for (const k of [0, 1, 3, 700, 5000]) for (const a of [0, 0.94, 0.998]) for (const fr of [-0.02, 0.003, 0.05, 0.098, 0.5, 1.7, 2.6, 2.95, 3.05]) {
    const t = eruptionTime(k) + fr * hotspotPeriod(hotspotRadius(k), a), epoch = 2048 * Math.floor(t / 2048);
    for (const s of fr === 0.5 ? [0, 0.6, 1.4] : [1]) {
      const h = hotspotAt(t, 1, a), rc = h.rc;
      for (const [r, th, dph] of [[rc, Math.PI / 2, 0], [rc + 2, Math.PI / 2, 0.25], [rc - 3, 1.3, -0.2], [1.6, Math.PI / 2, 0], [3, 1.4, 0.1]])
        hsCases.push({ a: [epoch, t - epoch, s, a], b: [r, th, h.phiC + dph, 0], c: [1, (k & 1) ? 3 : -2.5, 0, 0] });
    }
  }
  const hsArr = new Float32Array(hsCases.length * 12);
  hsCases.forEach((c, i) => hsArr.set([...c.a, ...c.b, ...c.c], i * 12));
  const hsIn = device.createBuffer({ size: hsArr.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(hsIn, 0, hsArr);
  const hsOut = device.createBuffer({ size: hsCases.length * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const hsRead = device.createBuffer({ size: hsCases.length * 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const hsMod = device.createShaderModule({ code: integratorSharedWGSL + emissionSharedWGSL + hotspotParityWGSL });
  const hsPipe = device.createComputePipeline({ layout: "auto", compute: { module: hsMod, entryPoint: "main" } });
  const hsBind = device.createBindGroup({ layout: hsPipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: hsIn } }, { binding: 1, resource: { buffer: hsOut } }] });
  const hsEnc = device.createCommandEncoder();
  const hsCp = hsEnc.beginComputePass(); hsCp.setPipeline(hsPipe); hsCp.setBindGroup(0, hsBind); hsCp.dispatchWorkgroups(hsCases.length); hsCp.end();
  hsEnc.copyBufferToBuffer(hsOut, 0, hsRead, 0, hsCases.length * 32);
  device.queue.submit([hsEnc.finish()]);
  await hsRead.mapAsync(GPUMapMode.READ);
  const hsGpu = new Float32Array(hsRead.getMappedRange().slice(0));
  // Errors / tolerance: alive must agree exactly (no case sits within 0.02 P of a boundary); absolute 1e-4 on r_c, 2e-3 on
  // phi_c (f32 tau ~ 1e3 M carries ~1e-4 M), 1e-4 on G; relative 1e-3 on the amplitude, Omega_c and D_h (D_h's against the
  // size of its terms, as for the flow); D_h = -1 exactly where the CPU has no timelike motion.
  let hsErr = 0, hsWorst = "";
  const hset = (i: number, k: number, g: number, w: number, tol: number) => { const e = Math.abs(g - w) / tol;
    if (!(e <= hsErr)) { hsErr = Number.isFinite(e) ? e : Infinity; hsWorst = `case ${i} out ${k}: gpu ${g} cpu ${w}`; } };
  hsCases.forEach((_, i) => {
    const v = Array.from(hsArr.subarray(i * 12, i * 12 + 12)), [epoch, rel, s, a] = v, [r, th, ph] = v.slice(4, 7), [pt, pphi] = v.slice(8, 10);
    const g = Array.from(hsGpu.subarray(i * 8, i * 8 + 8)), h = hotspotAt(epoch + rel, s, a);
    hset(i, 3, g[3], h.alive ? 1 : 0, 1e-6);
    if (h.alive) { hset(i, 0, g[0], h.rc, 1e-4); hset(i, 1, g[1], h.phiC, 2e-3); hset(i, 2, g[2], h.amp, 1e-3 * Math.max(1, h.amp)); }
    const rc = h.alive ? g[0] : HOTSPOT.rMin, Om = 1 / (rc ** 1.5 + a);
    hset(i, 4, g[4], hotspotBoost(r, th, ph, rc, g[1]), 1e-4);
    hset(i, 6, g[6], Om, 1e-3 * Om);
    const Dh = hotspotShift(r, th, pt, pphi, a, g[6]);
    if (Dh === null) hset(i, 5, g[5], -1, 1e-6);
    else hset(i, 5, g[5], Dh, 1e-3 * (Math.abs(pt) + Math.abs(Om * pphi)) * Math.max(1, Math.abs(Dh)));
  });
  console.log("hotspot parity worst (|err| / tol)", hsErr.toExponential(2), hsWorst, "cases", hsCases.length);
  return { maxErr, rows: cases.length + tcases.length + jcases.length + scases.length + ccases.length + icases.length + fcases.length + hcases.length + hsCases.length + mcases.length, jetLogErr, turbErr, turbWorst, turbRough, fluxErr, fluxWorst, flowErr, flowWorst, hsErr, hsWorst, minoErr, minoWorst };
}
