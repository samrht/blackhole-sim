// Quarter-resolution, separable-Gaussian bloom over the HDR accumulation buffer.
// Pass 1 (bright_h): downsample accum -> 1/4-res, keep only the luminous core, blur horizontally.
// Pass 2 (blur_v):   blur that low-res image vertically.
// The result is composited additively in present.wgsl, giving the hot disk a soft glow. Working
// at 1/4 res (rather than 1/2) dilutes thin 1-2px features ~13x, so razor lensing caustics barely
// bloom while the broad disk blooms fully — and the wider blur reads as natural cinematic glare.
struct Uniforms {
  res: vec2<f32>, a: f32, incl: f32, rObs: f32, fovScale: f32, rIn: f32, rOut: f32,
  Tpeak: f32, exposure: f32, time: f32, frame: u32, reset: u32, maxSteps: u32,
};
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;

const DOWN: u32 = 4u;            // bloom downsample factor
const RADIUS: i32 = 6;           // taps each side
const SIGMA2: f32 = 8.0;         // gaussian variance (in tap units)

fn dims_full() -> vec2<u32> { return vec2<u32>(u32(U.res.x), u32(U.res.y)); }
fn dims_lo() -> vec2<u32> { let f = dims_full(); return (f + vec2<u32>(DOWN - 1u)) / vec2<u32>(DOWN); }

// Pass 1: downsample full-res accum -> 1/4-res, bright-pass, horizontal blur.
@compute @workgroup_size(8, 8) fn bright_h(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lo = dims_lo();
  if (gid.x >= lo.x || gid.y >= lo.y) { return; }
  let full = dims_full();
  let cx = i32(gid.x * DOWN);
  let cy = i32(gid.y * DOWN);
  var sum = vec3<f32>(0.0);
  var wsum = 0.0;
  for (var i = -RADIUS; i <= RADIUS; i++) {
    let sx = clamp(cx + i, 0, i32(full.x) - 1);
    let idx = u32(cy) * full.x + u32(sx);
    var c = src[idx].rgb; // accum already holds normalized radiance (EMA / running mean)
    // bright-pass: isolate the luminous core (hot disk + bright stars), reject the dark void
    let lum = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    c = c * smoothstep(0.75, 1.7, lum);
    let fi = f32(i);
    let w = exp(-fi * fi / SIGMA2);
    sum += c * w;
    wsum += w;
  }
  dst[gid.y * lo.x + gid.x] = vec4<f32>(sum / wsum, 1.0);
}

// Pass 2: vertical blur at 1/4-res.
@compute @workgroup_size(8, 8) fn blur_v(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lo = dims_lo();
  if (gid.x >= lo.x || gid.y >= lo.y) { return; }
  var sum = vec3<f32>(0.0);
  var wsum = 0.0;
  for (var i = -RADIUS; i <= RADIUS; i++) {
    let sy = clamp(i32(gid.y) + i, 0, i32(lo.y) - 1);
    let idx = u32(sy) * lo.x + gid.x;
    let fi = f32(i);
    let w = exp(-fi * fi / SIGMA2);
    sum += src[idx].rgb * w;
    wsum += w;
  }
  dst[gid.y * lo.x + gid.x] = vec4<f32>(sum / wsum, 1.0);
}
