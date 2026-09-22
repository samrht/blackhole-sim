struct Uniforms {
  res: vec2<f32>, a: f32, incl: f32, rObs: f32, fovScale: f32, rIn: f32, rOut: f32,
  Tpeak: f32, exposure: f32, time: f32, frame: u32, reset: u32, maxSteps: u32,
  blend: f32, timeScale: f32, turbAmp: f32, breatheAmp: f32, nSpots: u32,
  jetStrength: f32, jetGamma: f32, jetLength: f32, jetKnots: f32,
  skyStrength: f32, outW: f32, outH: f32,
};
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> accum: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bloom: array<vec4<f32>>; // half-res glow

@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  // fullscreen triangle
  var p = array<vec2<f32>, 3>(vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
  return vec4<f32>(p[vi], 0.0, 1.0);
}

fn aces(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3(0.0), vec3(1.0));
}

const BLOOM = 0.85; // glow intensity

@fragment fn fs(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
  // U.res is the INTERNAL trace size; the framebuffer is (outW, outH). q is this fragment's centre
  // in internal-pixel units. At scale 1 the ratio is exactly 1, ip is an integer and f == 0, so
  // every tap below reduces to the old nearest read -- the full-resolution image is unchanged.
  let res = vec2<u32>(u32(U.res.x), u32(U.res.y));
  let q = fragCoord.xy * (U.res / vec2<f32>(U.outW, U.outH));
  let ip = clamp(q - 0.5, vec2<f32>(0.0), U.res - 1.0);
  let i0 = floor(ip);
  let f = ip - i0;
  let ax0 = u32(i0.x); let ay0 = u32(i0.y);
  let ax1 = min(ax0 + 1u, res.x - 1u); let ay1 = min(ay0 + 1u, res.y - 1u);
  var hdr = mix(mix(accum[ay0 * res.x + ax0].rgb, accum[ay0 * res.x + ax1].rgb, f.x),
                mix(accum[ay1 * res.x + ax0].rgb, accum[ay1 * res.x + ax1].rgb, f.x), f.y);

  // additive quarter-res bloom, bilinearly upsampled (nearest-tap would show 4x4 blocks)
  let bw = (res.x + 3u) / 4u;
  let bh = (res.y + 3u) / 4u;
  let bp = q / 4.0 - 0.5;
  let bi = floor(bp);
  let bf = bp - bi;
  let x0 = u32(clamp(bi.x, 0.0, f32(bw - 1u)));
  let y0 = u32(clamp(bi.y, 0.0, f32(bh - 1u)));
  let x1 = min(x0 + 1u, bw - 1u);
  let y1 = min(y0 + 1u, bh - 1u);
  let cb = mix(mix(bloom[y0 * bw + x0].rgb, bloom[y0 * bw + x1].rgb, bf.x),
               mix(bloom[y1 * bw + x0].rgb, bloom[y1 * bw + x1].rgb, bf.x), bf.y);
  hdr += cb * BLOOM;

  hdr = hdr * exp2(U.exposure);

  // subtle radial vignette to settle the eye on the hole
  let uv = q / U.res - 0.5;
  let vig = 1.0 - 0.32 * dot(uv, uv) * 2.0;
  hdr = hdr * vig;

  let mapped = aces(hdr);
  let srgb = pow(mapped, vec3(1.0 / 2.2)); // gamma encode
  return vec4<f32>(srgb, 1.0);
}
