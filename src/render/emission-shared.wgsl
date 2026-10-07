// Sole WGSL copy of the disk turbulence and the synchrotron jet (twins: turbulenceAt in src/physics/emission.ts,
// src/physics/jet.ts, src/physics/synchrotron.ts). Prepended verbatim by gpu.ts (renderer) and
// parity.browser.ts (?parity: turb and jet cases), so ?parity checks the shipped bytes. No uniforms,
// no bindings; the jet code uses State/gUp and the turbulence omegaKep from integrator-shared.wgsl, which both
// prepend too.

fn smoothE(t: f32) -> f32 { return t * t * (3.0 - 2.0 * t); }
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
const JET_ZBASE = 2.0;

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
// --- Horizon-flux history and launch time (spec 2026-10-03 jet flux knots; twins: flux-history.ts, jet.ts) ----
const FLUX_T = 1500.0; const FLUX_JIT = 0.33333334; const FLUX_TAUD = 500.0; const FLUX_SPREAD = 0.5;
const FLUX_FLOOR = 0.05;
// Flicker and the calibration (spec 2026-10-03 flux statistics; spin-independent per Narayan et al. 2022 S3.3)
// (twin: FLUX in flux-history.ts).
const FLUX_CELL_N = 50.0; const FLUX_CLIP = 3.0; const FLUX_SALT_N = 0x464eu;
const FLUX_DBAR = 0.1842; const FLUX_EPS = 0.06844; const FLUX_D1 = 0.092856;
const FLUX_SALT_T = 0x464cu; const FLUX_SALT_D = 0x4458u;
const JET_ENV_Q = 1.51;   // envelope bound on rho / rho_f: 1.2 x the widest flux-driven width (jet.ts JET_ENV_Q)
// Absolute time epoch + rel as (whole periods k, remainder in [0, P)): the epoch is a multiple of 2048, so for
// an integer P both epoch and kE P are exact f32 integers below 2^24 and the remainder keeps rel's precision.
fn splitPeriodJ(epoch: f32, rel: f32, P: f32) -> vec2<f32> {
  let kE = floor(epoch / P);
  let x = (epoch - kE * P) + rel;
  let kl = floor(x / P);
  return vec2<f32>(kE + kl, x - kl * P);
}
fn fluxHashJ(k: i32, salt: u32) -> f32 { return f32(hash4T(k, 0x7a11u, 0, salt) & 0xffffffu) / 16777216.0 - 0.5; }
fn fluxDeficitJ(kf: f32, loc: f32, dbar: f32) -> f32 {
  var k = i32(kf);
  var dt = loc - FLUX_T * (0.5 + FLUX_JIT * fluxHashJ(k, FLUX_SALT_T));
  if (dt < 0.0) { k = k - 1; dt = loc + FLUX_T - FLUX_T * (0.5 + FLUX_JIT * fluxHashJ(k, FLUX_SALT_T)); }
  let gap = FLUX_T * (1.0 + FLUX_JIT * (fluxHashJ(k + 1, FLUX_SALT_T) - fluxHashJ(k, FLUX_SALT_T)));
  let depth = dbar * (1.0 + FLUX_SPREAD * fluxHashJ(k, FLUX_SALT_D));
  let D = -FLUX_TAUD * log(1.0 - depth);
  if (dt < D) { return 1.0 - exp(-dt / FLUX_TAUD); }
  return depth * (gap - dt) / (gap - D);
}
// Flicker: unit-Gaussian lattice in time, 50 M cells (gaussT nodes), smoothstep weights renormalised, clipped
// at +-3 (twin: fluxFlicker). The time is split into whole cells + remainder (exact at any epoch).
fn fluxFlickerJ(epoch: f32, rel: f32) -> f32 {
  let kc = splitPeriodJ(epoch, rel, FLUX_CELL_N);
  let i = i32(kc.x); let f = smoothE(kc.y / FLUX_CELL_N); let w0 = 1.0 - f; let w1 = f;
  let n = (w0 * gaussT(i, 0x0f1cu, 0, FLUX_SALT_N) + w1 * gaussT(i + 1, 0x0f1cu, 0, FLUX_SALT_N)) / sqrt(w0 * w0 + w1 * w1);
  return clamp(n, -FLUX_CLIP, FLUX_CLIP);
}
// (dbar, eps, <d>): spin-independent (twin: fluxParams).
fn fluxParamsJ(a: f32) -> vec3<f32> { return vec3<f32>(FLUX_DBAR, FLUX_EPS, FLUX_D1); }
// f = Phi / <Phi> at absolute time epoch + rel, slider s, spin a (s = 0: exactly 1; twin: fluxRatio).
fn fluxRatioJ(epoch: f32, rel: f32, s: f32, a: f32) -> f32 {
  if (s == 0.0) { return 1.0; }
  let p = fluxParamsJ(a);
  let kl = splitPeriodJ(epoch, rel, FLUX_T);
  return max(FLUX_FLOOR, (1.0 - s * fluxDeficitJ(kl.x, kl.y, p.x)) * (1.0 + s * p.y * fluxFlickerJ(epoch, rel))) / (1.0 - s * p.z);
}
const GL6_X = array<f32, 6>(-0.9324695142031521, -0.6612093864662645, -0.2386191860831969, 0.2386191860831969, 0.6612093864662645, 0.9324695142031521);
const GL6_W = array<f32, 6>(0.1713244923791704, 0.3607615730481386, 0.4679139345726910, 0.4679139345726910, 0.3607615730481386, 0.1713244923791704);
// Coordinate time for plasma to climb from z_base to |z| (6-point Gauss in v = z^(1-p); twin: launchDelay).
fn launchDelayJ(z: f32, g280: f32) -> f32 {
  let az = abs(z);
  if (az <= JET_ZBASE) { return 0.0; }
  let A = sqrt(max(1e-12, g280 * g280 - 1.0)) / pow(GAMMA_REF_Z, GAMMA_SLOPE);
  let e = 2.0 * GAMMA_SLOPE / (1.0 - GAMMA_SLOPE);
  let v0 = pow(JET_ZBASE, 1.0 - GAMMA_SLOPE); let v1 = pow(az, 1.0 - GAMMA_SLOPE);
  let h = 0.5 * (v1 - v0); let m = 0.5 * (v1 + v0);
  // Function-scope copies: WGSL guarantees dynamic indexing for var arrays.
  var xs = GL6_X; var ws = GL6_W;
  var s = 0.0;
  for (var i = 0; i < 6; i++) { s += ws[i] * sqrt(1.0 + A * A * pow(m + h * xs[i], e)); }
  return s * h / (A * (1.0 - GAMMA_SLOPE));
}
fn fieldLineOmegaJ(a: f32) -> f32 { return a / (4.0 * (1.0 + sqrt(max(0.0, 1.0 - a * a)))); }
fn comovingAzimuthJ(ph: f32, z: f32, a: f32, g280: f32) -> f32 {
  let az = abs(z);
  if (az <= JET_ZBASE) { return ph; }
  return ph - fieldLineOmegaJ(a) * (launchDelayJ(z, g280) - (az - JET_ZBASE));
}
// --- Filaments frozen into the moving plasma (spec 2.5; twin: filaments in jet.ts) ---------------------------
const FIL_AMP = 0.35; const FIL_CELL_T = 25.0; const FIL_CELLS_PHI = 8u; const FIL_CELLS_Q = 2.5; const FIL_SALT = 0x46494cu;
fn node3J(ix: i32, iy: u32, iw: i32) -> f32 { return f32(hash4T(ix, iy, iw, FIL_SALT) & 0xffffffu) / 16777216.0; }
// Value noise on (ix + fx, y periodic in FIL_CELLS_PHI, w); x is passed split (time cell index + fraction).
fn vnoise3J(ix: i32, fx0: f32, y: f32, w: f32) -> f32 {
  let iyf = floor(y); let iw = i32(floor(w));
  let fx = smoothE(fx0); let fy = smoothE(y - iyf); let fw = smoothE(w - floor(w));
  let y0 = u32(iyf) % FIL_CELLS_PHI; let y1 = (y0 + 1u) % FIL_CELLS_PHI;
  var v = 0.0;
  for (var c = 0u; c < 8u; c++) {
    let dx = c & 1u; let dy = (c >> 1u) & 1u; let dw = (c >> 2u) & 1u;
    let wt = select(1.0 - fx, fx, dx == 1u) * select(1.0 - fy, fy, dy == 1u) * select(1.0 - fw, fw, dw == 1u);
    v += wt * node3J(ix + i32(dx), select(y0, y1, dy == 1u), iw + i32(dw));
  }
  return v;
}
fn filamentsJ(q: f32, phiC: f32, epoch: f32, relL: f32) -> f32 {
  let kt = splitPeriodJ(epoch, relL, FIL_CELL_T);
  let turns = phiC / TWO_PI_E;
  let y = (turns - floor(turns)) * f32(FIL_CELLS_PHI);
  return 1.0 + FIL_AMP * (vnoise3J(i32(kt.x), kt.y / FIL_CELL_T, y, q * FIL_CELLS_Q) - 0.5) * 2.0;
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
// Density modulation (spec 2026-10-03 jet flux knots 2.4; twin: jetShape in jet.ts): the plasma at height z left
// the base at (epoch + rel) - tau(z) with flux ratio f; the funnel is sqrt(f) wider and f denser, times the
// co-moving filaments. rel carries the per-pixel emission time (light delay); 0 outside the emitting jet.
fn jetShapeJ(r: f32, th: f32, ph: f32, epoch: f32, rel: f32, jetLength: f32, fluxVar: f32, g280: f32, a: f32) -> f32 {
  let z = r * cos(th); let az = abs(z);
  if (az < JET_ZBASE || az > jetLength) { return 0.0; }
  // Exact early-out before the launch-time / flux work: beyond the widest flux-driven wall the shape is 0
  // for any f (final review: running launchDelayJ first slowed the live trace ~30 %).
  if (r * sin(th) > JET_ENV_Q * funnelEdgeJ(z)) { return 0.0; }
  let relL = rel - launchDelayJ(z, g280);
  let f = fluxRatioJ(epoch, relL, fluxVar, a); let sw = sqrt(f);
  let rho = r * sin(th);
  let w = wallJ(rho / sw, z);
  if (w <= 0.0) { return 0.0; }
  let q = rho / (sw * funnelEdgeJ(z));
  return max(0.0, f * w * lengthFalloffJ(z, jetLength) * filamentsJ(q, comovingAzimuthJ(ph, z, a, g280), epoch, relL));
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
// Per band (frequencies e^lnNu: JET_LNNU in the visible, HF_LNNU x 3 at 1.3 mm): j = (nu/nu')^3 j'(nu') (cgs per sr)
// and alpha'(nu') (1/cm) of the cooled population at nu' = D nu.
fn synchSampleJ(lnNu: vec3<f32>, r: f32, th: f32, D: f32, a: f32, b0: f32, q0: f32, shape: f32, g280: f32, rgCm: f32) -> SynchOut {
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
    let lnx = lnNu[b] + lnD - SYN_LNNUB0 - lnB;
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
// --- Hot flow at 230 GHz (spec 2026-10-04 hot flow; twin: src/physics/hot-flow.ts) ---------------------------------
// Broderick et al. 2011 RIAF profiles, Pu et al. 2016 velocity (Keplerian / free fall mixed 50/50), thermal synchrotron
// (Mahadevan et al. 1996 fit) with Kirchhoff absorption. Coefficients are built in logs so n0 (5e5 for M87* up to ~1e18
// for dense custom objects) and j (~1e-20 cgs) stay inside f32: ln X spans about [-28, 36] and every exp stays below 88.
const HF_LNNU = 26.16134515;        // ln(230e9)
const HF_T0 = 1e11; const HF_BETA = 10.0; const HF_RMAX = 50.0; const HF_KTB = 6.1528e13;
// cgs logs: ln e^2, ln(k / m_e c^2), ln(8 pi m_p c^2 * 2 / 12) [B^2 = e^(that + ln n - ln r) / beta], ln(2 k / c^2),
// ln(2 sqrt(3) c); ln(e / 2 pi m_e c) is the jet's SYN_LNNUB0.
const HF_LN_QE2 = -42.91313517; const HF_LN_THE = -22.50327261; const HF_LN_B2 = -5.06769552;
const HF_LN_RJ = -84.07320297; const HF_LN_2S3C = 25.3662245;
fn iscoJ(a: f32) -> f32 { // prograde ISCO (Bardeen, Press & Teukolsky 1972; twin: iscoRadius in orbits.ts)
  let z1 = 1.0 + pow(max(1.0 - a * a, 0.0), 1.0 / 3.0) * (pow(1.0 + a, 1.0 / 3.0) + pow(max(1.0 - a, 0.0), 1.0 / 3.0));
  let z2 = sqrt(3.0 * a * a + z1 * z1);
  return 3.0 + z2 - sqrt(max((3.0 - z1) * (3.0 + z1 + 2.0 * z2), 0.0));
}
// ln of the Mahadevan isotropic fit M(X), written out so it does not underflow at large X.
fn lnMahadevanJ(lnX: f32) -> f32 {
  return 1.39884033 - lnX / 6.0 + log(1.0 + 0.4 * exp(-0.25 * lnX) + 0.5316 * exp(-0.5 * lnX)) - 1.8899 * exp(lnX / 3.0);
}
// The prograde ISCO orbit (r_isco, E, L) per unit mass: depends on the spin alone, so a ray computes it once
// (flowStep's callers) instead of per sample.
fn iscoOrbitJ(a: f32) -> vec3<f32> {
  let risco = iscoJ(a);
  let sq = sqrt(risco); let den = pow(risco, 0.75) * sqrt(risco * sq - 3.0 * sq + 2.0 * a);
  return vec3<f32>(risco, (risco * sq - 2.0 * sq + a) / den, (risco * risco - 2.0 * a * sq + a * a) / den);
}
// (u^t, u^r, Omega, ok) of the flow; ok = 0 where no timelike velocity exists (K0 <= 0).
fn flowVelocityJ(r: f32, th: f32, a: f32) -> vec4<f32> { return flowVelocityOrbJ(r, th, a, iscoOrbitJ(a)); }
fn flowVelocityOrbJ(r: f32, th: f32, a: f32, orb: vec3<f32>) -> vec4<f32> {
  let gu = gUp(r, 0.5 * PI, a);                         // 0 tt, 1 tphi, 2 rr, 3 thth, 4 phph
  let risco = orb.x;
  var urK = 0.0; var OmK = omegaKep(r, a);
  if (r < risco) {
    let E = orb.y; let L = orb.z;
    let uT = -gu[0] * E + gu[1] * L; let uP = -gu[1] * E + gu[4] * L;
    let rest = -1.0 - (gu[0] * E * E - 2.0 * gu[1] * E * L + gu[4] * L * L);
    urK = -sqrt(max(0.0, rest * gu[2])); OmK = uP / uT;
  }
  let urFF = -sqrt(max(0.0, gu[2] * (-1.0 - gu[0]))); let OmFF = gu[1] / gu[0];
  let ur = urK + 0.5 * (urFF - urK); let Om = OmK + 0.5 * (OmFF - OmK);
  let g = gLow(r, th, a);                                // 0 tt, 1 tphi, 2 rr, 3 thth, 4 phph
  let K0 = -(g[0] + 2.0 * Om * g[1] + Om * Om * g[4]);
  if (K0 <= 0.0) { return vec4<f32>(0.0); }
  return vec4<f32>(sqrt((1.0 + g[2] * ur * ur) / K0), ur, Om, 1.0);
}
// nu_plasma / nu_observed for the camera-normalised covariant momentum p = (p_t, p_r, p_th, p_phi); -1 if no velocity.
fn flowShiftJ(r: f32, th: f32, p: vec4<f32>, a: f32) -> f32 { return flowShiftOrbJ(r, th, p, a, iscoOrbitJ(a)); }
fn flowShiftOrbJ(r: f32, th: f32, p: vec4<f32>, a: f32, orb: vec3<f32>) -> f32 {
  let u = flowVelocityOrbJ(r, th, a, orb);
  if (u.w == 0.0) { return -1.0; }
  return u.x * p.x + u.y * p.y + u.z * u.x * p.w;
}
// Plasma-frame (j_nu, alpha_nu) at nu = e^lnNu; (0, 0) where there is no plasma.
fn flowCoeffsJ(r: f32, th: f32, lnNu: f32, n0: f32) -> vec2<f32> {
  let z = r * cos(th); let rho = r * sin(th);
  if (rho < 1e-6) { return vec2<f32>(0.0); }
  let lnN = log(n0) - 1.1 * log(0.5 * r) - z * z / (2.0 * rho * rho);
  if (lnN < -40.0) { return vec2<f32>(0.0); }
  let lnT = log(HF_T0) - 0.84 * log(0.5 * r);
  let lnThe = HF_LN_THE + lnT;
  let lnB = 0.5 * (HF_LN_B2 - log(HF_BETA) + lnN - log(r));
  let lnX = log(2.0 / 3.0) + lnNu - (SYN_LNNUB0 + lnB) - 2.0 * lnThe;
  let lnJ = lnN + HF_LN_QE2 + lnNu - HF_LN_2S3C - 2.0 * lnThe + lnMahadevanJ(lnX);
  let lnA = lnJ - (HF_LN_RJ + 2.0 * lnNu + lnT);
  return vec2<f32>(exp(lnJ), exp(lnA));
}
// --- Hotspot flares at 1.3 mm (spec 2026-10-04 mm hotspots; twin: src/physics/hotspot.ts) -----------------------------
const HS_RMIN = 8.0; const HS_RSPAN = 4.0; const HS_SIGMA = 2.548; const HS_CUT = 4.0;
const HS_RISE = 0.1; const HS_CUT_FROM = 2.5; const HS_CUT_TO = 3.0;
const HS_SALT_R = 0x4853u; const HS_SALT_PHI = 0x4850u;
const HS_A0 = 9.22;       // scripts/calibrate-hotspot.ts (twin: HOTSPOT.A0)
const HS_REACH = 22.192;  // HS_RMIN + HS_RSPAN + HS_CUT * HS_SIGMA: no blob reaches beyond this radius
fn hotspotLightJ(tau: f32, P: f32) -> f32 {
  if (tau < 0.0 || tau >= HS_CUT_TO * P) { return 0.0; }
  return smoothstepJ(0.0, HS_RISE * P, tau) * exp(-tau / P) * (1.0 - smoothstepJ(HS_CUT_FROM * P, HS_CUT_TO * P, tau));
}
// (r_c, phi_c, A_k L, 1) of the hotspot at absolute time epoch + rel, or 0 when none is alive (twin: hotspotAt). The
// latest eruption k comes from fluxDeficitJ's split (exact at any epoch); at most one hotspot is alive at a time.
fn hotspotStateJ(epoch: f32, rel: f32, s: f32, a: f32) -> vec4<f32> {
  if (s <= 0.0) { return vec4<f32>(0.0); }
  let kl = splitPeriodJ(epoch, rel, FLUX_T);
  var k = i32(kl.x);
  var tau = kl.y - FLUX_T * (0.5 + FLUX_JIT * fluxHashJ(k, FLUX_SALT_T));
  if (tau < 0.0) { k = k - 1; tau = kl.y + FLUX_T - FLUX_T * (0.5 + FLUX_JIT * fluxHashJ(k, FLUX_SALT_T)); }
  let rc = HS_RMIN + HS_RSPAN * (fluxHashJ(k, HS_SALT_R) + 0.5);
  let q = pow(rc, 1.5) + a; let P = TWO_PI_E * q;
  if (tau >= HS_CUT_TO * P) { return vec4<f32>(0.0); }
  let phiC = TWO_PI_E * (fluxHashJ(k, HS_SALT_PHI) + 0.5) + tau / q;
  let amp = HS_A0 * s * (1.0 + FLUX_SPREAD * fluxHashJ(k, FLUX_SALT_D)) * hotspotLightJ(tau, P);
  return vec4<f32>(rc, phiC, amp, 1.0);
}
// Gaussian G(d) about (r_c, pi/2, phi_c), 0 beyond 4 sigma (twin: hotspotBoost).
fn hotspotBoostJ(r: f32, th: f32, ph: f32, rc: f32, phiC: f32) -> f32 {
  let d2 = r * r + rc * rc - 2.0 * r * rc * sin(th) * cos(ph - phiC);
  if (d2 >= HS_CUT * HS_CUT * HS_SIGMA * HS_SIGMA) { return 0.0; }
  return exp(-d2 / (2.0 * HS_SIGMA * HS_SIGMA));
}
// nu_plasma / nu_obs of matter at (r, th) moving at Om, p = (p_t, p_r, p_th, p_phi); -1 where not timelike (twin: hotspotShift).
fn hotspotShiftJ(r: f32, th: f32, p: vec4<f32>, a: f32, Om: f32) -> f32 {
  let g = gLow(r, th, a);                                // 0 tt, 1 tphi, 2 rr, 3 thth, 4 phph
  let K = -(g[0] + 2.0 * Om * g[1] + Om * Om * g[4]);
  if (K <= 0.0) { return -1.0; }
  return (p.x + Om * p.w) / sqrt(K);
}
