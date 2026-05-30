// Parity test shader: mirrors the metric/orbit/g-factor helpers from raytrace.wgsl so we
// can compare the WGSL math against the verified TypeScript core (src/physics/*) numerically.
fn delta_(r: f32, a: f32) -> f32 { return r*r - 2.0*r + a*a; }
fn sigma_(r: f32, th: f32, a: f32) -> f32 { let c = cos(th); return r*r + a*a*c*c; }
fn bigA_(r: f32, th: f32, a: f32) -> f32 { let s = sin(th); return pow(r*r+a*a,2.0) - a*a*delta_(r,a)*s*s; }
fn gUp(r: f32, th: f32, a: f32) -> array<f32,5> {
  let s2 = sin(th)*sin(th); let Sig = sigma_(r,th,a); let d = delta_(r,a); let A = bigA_(r,th,a);
  return array<f32,5>( -A/(Sig*d), -2.0*a*r/(Sig*d), d/Sig, 1.0/Sig, (d - a*a*s2)/(Sig*d*s2) );
}
fn gLow(r: f32, th: f32, a: f32) -> array<f32,5> {
  let s2 = sin(th)*sin(th); let Sig = sigma_(r,th,a); let d = delta_(r,a);
  return array<f32,5>( -(1.0-2.0*r/Sig), -2.0*a*r*s2/Sig, Sig/d, Sig, (r*r+a*a+2.0*a*a*r*s2/Sig)*s2 );
}
fn omegaKep(r: f32, a: f32) -> f32 { return 1.0/(pow(r,1.5) + a); }

struct In { r: f32, th: f32, a: f32, xi: f32 };
@group(0) @binding(0) var<storage, read> inputs: array<In>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec4<f32>>; // (gUp.tt, gLow.tt, omegaKep, gFactor)

@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = arrayLength(&inputs);
  if (gid.x >= n) { return; }
  let v = inputs[gid.x];
  let gu = gUp(v.r, v.th, v.a); let gl = gLow(v.r, v.th, v.a);
  let Om = omegaKep(v.r, v.a);
  let rad = -(gl[0] + 2.0*Om*gl[1] + Om*Om*gl[4]);
  let gfac = sqrt(max(0.0, rad)) / (1.0 - Om*v.xi);
  outputs[gid.x] = vec4<f32>(gu[0], gl[0], Om, gfac);
}
