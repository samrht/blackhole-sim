// Sole WGSL copy of the disk turbulence and the synchrotron jet (twins: turbulenceAt in src/physics/emission.ts,
// src/physics/jet.ts, src/physics/synchrotron.ts). Prepended verbatim by gpu.ts (renderer) and
// parity.browser.ts (?parity: turb and jet cases), so ?parity checks the shipped bytes. No uniforms,
// no bindings; the jet code uses State/gUp and the turbulence omegaKep from integrator-shared.wgsl, which both
// prepend too.

// --- Value noise (the jet's knots and churn) -----------------------------------------------------------------
fn ihashE(ix: i32, iy: i32) -> f32 {
  var n = u32(ix) * 1973u + u32(iy) * 9277u;
  n = (n ^ (n >> 15u)) * 2246822519u;
  n = (n ^ (n >> 13u)) * 3266489917u;
  return f32(n & 0xffffffu) / f32(0xffffffu);
}
fn smoothE(t: f32) -> f32 { return t * t * (3.0 - 2.0 * t); }
fn vnoiseE(x: f32, y: f32) -> f32 {
  let ix = i32(floor(x)); let iy = i32(floor(y));
  let fx = smoothE(x - floor(x)); let fy = smoothE(y - floor(y));
  let a00 = ihashE(ix, iy); let a10 = ihashE(ix + 1, iy);
  let a01 = ihashE(ix, iy + 1); let a11 = ihashE(ix + 1, iy + 1);
  return (a00 * (1.0 - fx) + a10 * fx) * (1.0 - fy) + (a01 * (1.0 - fx) + a11 * fx) * fy;
}
const TWO_PI_E = 6.283185307179586;
// --- MRI turbulence (spec 2026-10-03; twin: turbulenceAt in src/physics/emission.ts) ------------------
// Unit-Gaussian lattice field on (ln r, phi) in overlapping generations frozen into the Keplerian flow;
// see the CPU twin for the construction. u32 arithmetic wraps exactly as Math.imul/>>>0 on the CPU.
const TURB_CELL_ETA = 0.086; const TURB_CELLS_PHI = 49u; const TURB_CLOCK = 0.32; const TURB_OCT2 = 0.5;
fn mixT(n0: u32) -> u32 {
  var n = (n0 ^ (n0 >> 15u)) * 2246822519u;
  n = (n ^ (n >> 13u)) * 3266489917u;
  return n ^ (n >> 16u);
}
fn hash4T(ix: i32, iy: u32, gen: i32, salt: u32) -> u32 {
  return mixT(u32(ix) * 1973u + iy * 9277u + u32(gen) * 26699u + salt * 59359u);
}
fn gaussT(ix: i32, iy: u32, gen: i32, salt: u32) -> f32 {
  let h1 = hash4T(ix, iy, gen, salt); let h2 = mixT(h1 ^ 0x9e3779b9u);
  let u1 = (f32(h1 & 0xffffffu) + 0.5) / 16777216.0;  // f32 may round the top value to 1: log -> 0, finite
  let u2 = f32(h2 & 0xffffffu) / 16777216.0;
  // cos(2 pi u2) = -cos(pi (2 u2 - 1)): the argument stays in [-pi, pi], where WGSL bounds cos to 2^-11.
  return -sqrt(max(0.0, -2.0 * log(u1))) * cos(0.5 * TWO_PI_E * (2.0 * u2 - 1.0));
}
fn turbOctaveT(r: f32, phi: f32, t0: f32, tRel: f32, a: f32, cellEta: f32, cellsPhi: u32, salt: u32) -> f32 {
  let x = log(r) / cellEta; let i0 = i32(floor(x)); let fx = smoothE(x - floor(x));
  let Om = omegaKep(r, a);
  var num = 0.0; var v = 0.0;
  for (var d = 0; d < 2; d++) {
    let i = i0 + d; let wr = select(1.0 - fx, fx, d == 1);
    let Tc = TURB_CLOCK * TWO_PI_E / omegaKep(exp(f32(i) * cellEta), a);
    // tau = (t0 + tRel) / T_c + row phase, in two parts: the epoch's (rounded once per row: a coherent
    // offset shared by every pixel) and the small per-pixel remainder's, which keeps full precision.
    let tau0 = t0 / Tc + f32(hash4T(i, 0x51edu, 0, salt) & 0xffffffu) / 16777216.0;
    let k0 = floor(tau0);
    let tau = (tau0 - k0) + tRel / Tc;
    let k = i32(k0) + i32(floor(tau)); let f = tau - floor(tau);
    for (var e = 0; e < 2; e++) {
      let wt = select(cos(0.25 * TWO_PI_E * f), sin(0.25 * TWO_PI_E * f), e == 1);
      let age = select(1.0 + f, f, e == 1) * Tc;
      let y = fract((phi - Om * age) / TWO_PI_E) * f32(cellsPhi);
      let jf = floor(y); let fy = smoothE(y - jf);
      let ja = u32(jf) % cellsPhi; let jb = (ja + 1u) % cellsPhi;
      let w0 = wr * wt * (1.0 - fy); let w1 = wr * wt * fy;
      num += w0 * gaussT(i, ja, k + e, salt) + w1 * gaussT(i, jb, k + e, salt);
      v += w0 * w0 + w1 * w1;
    }
  }
  return num / sqrt(v);
}
// Unit-Gaussian turbulence g at (r, phi, a) and emission time t0 + tRel: t0 the clock epoch (a multiple of
// 2048 M, sim-clock.ts), tRel the small remainder (it carries the per-pixel light delay). 2 pi-periodic in phi.
fn turbulenceFieldE(r: f32, phi: f32, t0: f32, tRel: f32, a: f32) -> f32 {
  return (turbOctaveT(r, phi, t0, tRel, a, TURB_CELL_ETA, TURB_CELLS_PHI, 1u)
        + TURB_OCT2 * turbOctaveT(r, phi, t0, tRel, a, 0.5 * TURB_CELL_ETA, 2u * TURB_CELLS_PHI, 2u)) / sqrt(1.0 + TURB_OCT2 * TURB_OCT2);
}

// --- Tier 2B jet geometry -------------------------------------------------------------------
const JET_QPEAK = 0.8;   const JET_WWALL = 0.22;
const JET_RHO0  = 0.6;   const JET_SLOPE = 0.7;
const JET_ZBASE = 2.0;   const JET_KZ    = 0.35;
const JET_TURB  = 0.35;  const JET_SEED  = 17.0;

fn smoothstepJ(a: f32, b: f32, x: f32) -> f32 {
  let t = clamp((x - a) / (b - a), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}
fn funnelEdgeJ(z: f32) -> f32 { return JET_RHO0 + JET_SLOPE * sqrt(abs(z)); }
fn wallJ(rho: f32, z: f32) -> f32 {
  let q = rho / funnelEdgeJ(z);
  if (q > 1.2) { return 0.0; }
  let d = q - JET_QPEAK;
  return exp(-(d * d) / (2.0 * JET_WWALL * JET_WWALL));
}
fn lengthFalloffJ(z: f32, zMax: f32) -> f32 {
  let az = abs(z);
  let fadeIn = smoothstepJ(JET_ZBASE, JET_ZBASE + 2.0, az);
  let fadeOut = 1.0 - smoothstepJ(zMax * 0.7, zMax, az);
  let decay = JET_ZBASE / max(az, JET_ZBASE);
  return fadeIn * fadeOut * decay;
}
fn knotsJ(z: f32, t: f32, gamma: f32, knotAmp: f32) -> f32 {
  // Knots move with the flow at beta(Gamma) < c (twin: knots in jet.ts); t is the emission time.
  let beta = sqrt(max(0.0, 1.0 - 1.0 / (gamma * gamma)));
  let phase = JET_KZ * (abs(z) - beta * t);
  return 1.0 + knotAmp * (vnoiseE(phase, JET_SEED) - 0.5) * 2.0;
}
// --- Tier 2B synchrotron jet (specs 2026-10-02; twin: src/physics/synchrotron.ts): bands, colour, flow,
// shift, density shape; tests/synchrotron.test.ts checks every literal against the CPU twin. ------------
const JET_LNNU = vec3<f32>(34.13261924, 33.93194855, 33.76489446);   // ln nu at 450, 550, 650 nm
// I_nu (cgs) per band -> linear sRGB in the disk's units (before lumNorm); column b = band b.
const JET_BAND_M = mat3x3<f32>(-57.85676090, -199.6091852, 8197.725079, 563.0803652, 5602.051454, -395.0105294, 4364.033919, -151.6783697, -69.56073163);
const GAMMA_REF_Z = 280.0; const GAMMA_SLOPE = 0.58;

// Proper speed Gamma beta = sqrt(G280^2 - 1) (|z|/280)^0.58 (twin: gammaProfile in jet.ts).
fn jetGammaAt(z: f32, g280: f32) -> f32 {
  let ub = sqrt(max(0.0, g280 * g280 - 1.0)) * pow(max(abs(z), 1e-6) / GAMMA_REF_Z, GAMMA_SLOPE);
  return sqrt(1.0 + ub * ub);
}
// Unit flow direction (n_r, n_theta): outward along the funnel family rho = q rho_f(z).
fn streamlineDirJ(r: f32, th: f32) -> vec2<f32> {
  let z = r * cos(th); let rho = r * sin(th); let az = max(abs(z), 1e-6);
  let vr = (rho / funnelEdgeJ(z)) * (JET_SLOPE / (2.0 * sqrt(az)));
  let vz = select(-1.0, 1.0, z >= 0.0);
  let n = sqrt(vr * vr + vz * vz); let ur = vr / n; let uz = vz / n;
  return vec2<f32>(ur * sin(th) + uz * cos(th), ur * cos(th) - uz * sin(th));
}
// nu_plasma / nu_observed = p.u for the camera-normalised past-directed photon momentum p at (r, th);
// the plasma moves at Lorentz factor gamma along the streamline in the ZAMO frame (spec 2.2/2.4).
fn plasmaShiftJ(r: f32, th: f32, p: vec4<f32>, a: f32, gamma: f32) -> f32 {
  let g = gUp(r, th, a);
  let lapse = sqrt(-1.0 / g[0]); let omega = g[1] / g[0];
  let beta = sqrt(max(0.0, 1.0 - 1.0 / (gamma * gamma)));
  let n = streamlineDirJ(r, th);
  return gamma * ((p.x + omega * p.w) / lapse + beta * (n.x * sqrt(g[2]) * p.y + n.y * sqrt(g[3]) * p.z));
}
// Density modulation (wall x length falloff x knots x turbulence); 0 outside the emitting jet.
fn jetShapeJ(r: f32, th: f32, t: f32, jetLength: f32, knotAmp: f32, g280: f32) -> f32 {
  let z = r * cos(th); let az = abs(z);
  if (az < JET_ZBASE || az > jetLength) { return 0.0; }
  let rho = r * sin(th);
  let w = wallJ(rho, z);
  if (w <= 0.0) { return 0.0; }
  let turb = 1.0 + JET_TURB * (vnoiseE(log(1.0 + rho), JET_KZ * z) - 0.5) * 2.0;
  return max(0.0, w * lengthFalloffJ(z, jetLength) * knotsJ(z, t, jetGammaAt(z, g280), knotAmp) * turb);
}
struct SynchOut { j: vec3<f32>, a: vec3<f32> };
// --- Cooled synchrotron jet (spec 2026-10-02 cooled jet; twins: src/physics/synchrotron.ts, cyclosynch.ts) --
// Exact cyclo-synchrotron coefficients of the cooled population, tabulated as (ln J^, ln A^) over
// (ln x = ln nu'/nu_B, ln s = ln cooling depth); every literal is checked by tests/synchrotron.test.ts.
@group(0) @binding(10) var synchTab: texture_2d<f32>;
const SYN_LNCJ = -18.74798845;      // ln[3 e^3 / (2 I_g sigma_T m_e c^3)]
const SYN_LNCA = 13.13221847;       // ln[3 pi^2 e / (c I_g sigma_T)]
const SYN_LNNUB0 = 14.84486172;     // ln(nu_B / B)
const SYN_LNK0 = -20.46682379;      // ln(k / B^2), k = sigma_T B^2 / (6 pi m_e c)
const SYN_LNT0 = -19.98809263;      // ln[280^0.58 / (0.42 c)]
const SYN_LNX0 = -9.210340372; const SYN_LNX1 = 23.02585093;
const SYN_LNS0 = -16.11809565; const SYN_LNS1 = 3.688879454;
fn synchReady() -> bool { return textureDimensions(synchTab).x >= 2u; } // 1x1 placeholder: table not loaded
// Twin of lookup() in cyclosynch.ts: bilinear between cell centres, with the same edge extensions (above the s grid:
// linear in s from the last two rows, the cold electrons' cyclotron-fundamental absorption growing with s; above the
// x grid: the last two columns' slope).
fn synchLookupJ(lnx: f32, lns: f32) -> vec2<f32> {
  let dims = textureDimensions(synchTab); let nx = f32(dims.x); let ns = f32(dims.y);
  let h = (SYN_LNX1 - SYN_LNX0) / nx;
  var fx = (lnx - SYN_LNX0) / h - 0.5; var extX = 0.0;
  if (fx > nx - 1.0) { extX = (fx - (nx - 1.0)) * h; fx = nx - 1.0; }
  fx = max(fx, 0.0);
  var fs = (lns - SYN_LNS0) / (SYN_LNS1 - SYN_LNS0) * (ns - 1.0); var extS = 0.0; var above = 0.0;
  if (fs < 0.0) { extS = lns - SYN_LNS0; fs = 0.0; }
  if (fs > ns - 1.0) { above = exp(lns) - exp(SYN_LNS1); fs = ns - 1.0; }
  let ix = min(i32(dims.x) - 2, i32(floor(fx))); let iy = min(i32(dims.y) - 2, i32(floor(fs)));
  let ax = fx - f32(ix); let ay = fs - f32(iy);
  let v00 = textureLoad(synchTab, vec2<i32>(ix, iy), 0).xy;
  let v10 = textureLoad(synchTab, vec2<i32>(ix + 1, iy), 0).xy;
  let v01 = textureLoad(synchTab, vec2<i32>(ix, iy + 1), 0).xy;
  let v11 = textureLoad(synchTab, vec2<i32>(ix + 1, iy + 1), 0).xy;
  var v = mix(mix(v00, v10, ax), mix(v01, v11, ax), ay);
  if (above > 0.0) {
    let sPrev = exp(SYN_LNS0 + (SYN_LNS1 - SYN_LNS0) * (ns - 2.0) / (ns - 1.0));
    let prev = mix(v00, v10, ax); // row ns - 2 (iy = ns - 2, ay = 1 here)
    let slope = max(vec2<f32>(0.0), (exp(v) - exp(prev)) / (exp(SYN_LNS1) - sPrev));
    v = log(exp(v) + slope * above);
  }
  // above the x grid: the slope of the last two columns (the spectrum's own power law, cooled or not)
  var slope = vec2<f32>(0.0);
  if (extX > 0.0) {
    let c1 = mix(textureLoad(synchTab, vec2<i32>(i32(dims.x) - 1, iy), 0).xy, textureLoad(synchTab, vec2<i32>(i32(dims.x) - 1, iy + 1), 0).xy, ay);
    let c0 = mix(textureLoad(synchTab, vec2<i32>(i32(dims.x) - 2, iy), 0).xy, textureLoad(synchTab, vec2<i32>(i32(dims.x) - 2, iy + 1), 0).xy, ay);
    slope = (c1 - c0) / h;
  }
  return v + slope * extX + vec2<f32>(extS, extS);
}
// Per band: j = (nu/nu')^3 j'(nu') (cgs per sr) and alpha'(nu') (1/cm) of the cooled population at nu' = D nu.
fn synchSampleJ(r: f32, th: f32, D: f32, a: f32, b0: f32, q0: f32, shape: f32, g280: f32, rgCm: f32) -> SynchOut {
  let z = r * cos(th); let rho = r * sin(th);
  let rf = funnelEdgeJ(z); let rH = 1.0 + sqrt(max(0.0, 1.0 - a * a)); let w = rho * a / (4.0 * rH);
  let lnB = log(b0 / (rf * rf)) + 0.5 * log(1.0 + w * w);
  // cooling depth s = k t', t' = (r_g / c) 280^0.58 / U0 (|z|^0.42 - z_b^0.42) / 0.42
  let U0 = sqrt(max(g280 * g280 - 1.0, 1e-12));
  let dz = max(pow(abs(z), 0.42) - pow(JET_ZBASE, 0.42), 1e-6);
  let lns = max(SYN_LNK0 + 2.0 * lnB + log(rgCm) + SYN_LNT0 - log(U0) + log(dz), -30.0);
  let base = log(q0) + log(shape);
  let lnD = log(D);
  var o: SynchOut;
  for (var b = 0; b < 3; b++) {
    let lnx = JET_LNNU[b] + lnD - SYN_LNNUB0 - lnB;
    let t = synchLookupJ(lnx, lns);
    o.j[b] = exp(SYN_LNCJ + base + lnB + t.x - 3.0 * lnD);
    o.a[b] = exp(SYN_LNCA + base - lnB - 2.0 * lnx + t.y);
  }
  return o;
}
// Light from the jet along one ray: I = observed I_nu per band (cgs), tau = optical depth from the camera.
struct JetOut { I: vec3<f32>, tau: vec3<f32> };
// One sample = a uniform slab of plasma-frame path ds (cm): exact solution, thin -> j ds, thick ->
// source function j / alpha, attenuated by what lies in front (twin: slabStep in synchrotron.ts).
fn jetSlabJ(acc: JetOut, j: vec3<f32>, alpha: vec3<f32>, ds: f32) -> JetOut {
  let dt = alpha * ds;
  let fac = select((vec3<f32>(1.0) - exp(-dt)) / max(dt, vec3<f32>(1e-30)), vec3<f32>(1.0) - 0.5 * dt, dt < vec3<f32>(1e-4));
  var o: JetOut;
  o.I = acc.I + j * ds * fac * exp(-acc.tau);
  o.tau = min(acc.tau + dt, vec3<f32>(1e30));
  return o;
}
