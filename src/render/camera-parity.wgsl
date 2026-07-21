// Entry point for CPU<->GPU parity of the camera mapping (screen -> xi/eta + initial momentum).
// Input: three vec4 per case --
//   inp[3k+0] = (alpha, beta, a, incl)
//   inp[3k+1] = (gtt, gtphi, grr, gthth)
//   inp[3k+2] = (gphph, 0, 0, 0)
// Output per case: two vec4 -- outp[2k] = (pt, pr, pth, pphi), outp[2k+1] = (xi, eta, 0, 0).
//
// This file deliberately contains NO copy of the camera math. camera-shared.wgsl is prepended to
// it by parity.browser.ts, exactly as it is prepended to raytrace.wgsl by gpu.ts -- so this route
// verifies the code the renderer actually runs, not a hand-synced duplicate of it.
@group(0) @binding(0) var<storage, read> inp: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outp: array<vec4<f32>>;

@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = inp[gid.x*3u + 0u];
  let g = inp[gid.x*3u + 1u];
  let gphph = inp[gid.x*3u + 2u].x;
  let xe = cameraXiEta(s.x, s.y, s.z, s.w);
  let p = cameraMomenta(xe.x, s.y, g.x, g.y, g.z, g.w, gphph);
  outp[gid.x*2u + 0u] = p;
  outp[gid.x*2u + 1u] = vec4<f32>(xe.x, xe.y, 0.0, 0.0);
}
