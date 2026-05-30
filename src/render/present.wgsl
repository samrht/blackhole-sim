struct Uniforms {
  res: vec2<f32>, a: f32, incl: f32,
  rObs: f32, fovScale: f32, rIn: f32, rOut: f32,
  Tpeak: f32, exposure: f32, time: f32,
  frame: u32, reset: u32, maxSteps: u32,
};
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> accum: array<vec4<f32>>;

@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  // fullscreen triangle
  var p = array<vec2<f32>, 3>(vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
  return vec4<f32>(p[vi], 0.0, 1.0);
}

fn aces(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3(0.0), vec3(1.0));
}

@fragment fn fs(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
  let px = vec2<u32>(u32(fragCoord.x), u32(fragCoord.y));
  let idx = px.y * u32(U.res.x) + px.x;
  let samples = f32(U.frame + 1u);
  var hdr = accum[idx].rgb / samples;
  hdr = hdr * exp2(U.exposure);
  let mapped = aces(hdr);
  let srgb = pow(mapped, vec3(1.0 / 2.2)); // gamma encode
  return vec4<f32>(srgb, 1.0);
}
