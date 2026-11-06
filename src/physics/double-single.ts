// Double-single arithmetic: a value as a pair of f32 (hi, lo), about 48 bits, every operation built from f32 additions and multiplications
// only (Dekker, Knuth). For the ?limitds diagnostic's "why not just emulate f64?" baseline. WGSL twin: src/render/limit-ds.wgsl; the route
// checks the GPU's results against exact values, because a shader compiler that simplifies (a + b) - a would silently break all of this.
export type DS = [number, number];
const F = Math.fround;

export function twoSum(a: number, b: number): DS { const s = F(a + b), bb = F(s - a); return [s, F(F(a - F(s - bb)) + F(b - bb))]; }
const quick = (a: number, b: number): DS => { const s = F(a + b); return [s, F(b - F(s - a))]; };
const split = (a: number): DS => { const t = F(4097 * a), h = F(t - F(t - a)); return [h, F(a - h)]; };
/** p + e is exactly a * b (Dekker, no fused multiply-add: WGSL's fma need not be fused). */
export function twoProd(a: number, b: number): DS {
  const p = F(a * b), [ah, al] = split(a), [bh, bl] = split(b);
  return [p, F(F(F(F(F(ah * bh) - p) + F(ah * bl)) + F(al * bh)) + F(al * bl))];
}
export function dsFrom(x: number): DS { const h = F(x); return quick(h, F(x - h)); }
export const dsToNumber = (a: DS) => a[0] + a[1];
export function dsAdd(a: DS, b: DS): DS {
  const s = twoSum(a[0], b[0]), t = twoSum(a[1], b[1]), v = quick(s[0], F(s[1] + t[0]));
  return quick(v[0], F(t[1] + v[1]));
}
export const dsNeg = (a: DS): DS => [-a[0], -a[1]];
export const dsSub = (a: DS, b: DS) => dsAdd(a, dsNeg(b));
export function dsMul(a: DS, b: DS): DS {
  const p = twoProd(a[0], b[0]);
  return quick(p[0], F(p[1] + F(F(a[0] * b[1]) + F(a[1] * b[0]))));
}
export function dsDiv(a: DS, b: DS): DS {
  const q1 = F(a[0] / b[0]); let r = dsSub(a, dsMul([q1, 0], b));
  const q2 = F(r[0] / b[0]); r = dsSub(r, dsMul([q2, 0], b));
  const q3 = F(r[0] / b[0]);
  return dsAdd(quick(q1, q2), [q3, 0]);
}
export function dsSqrt(a: DS): DS {
  if (a[0] <= 0) return [0, 0];
  const x = F(1 / F(Math.sqrt(a[0]))), ax = F(a[0] * x);
  return dsAdd([ax, 0], [F(dsSub(a, dsMul([ax, 0], [ax, 0]))[0] * F(x * 0.5)), 0]);
}
export const DS_PI = dsFrom(Math.PI), DS_PIO2 = dsFrom(Math.PI / 2), DS_LN2 = dsFrom(Math.LN2);
/** 1 / n!, n = 0..16: the sin and cos Taylor coefficients. */
export const DS_COEFFS: DS[] = (() => { const o: DS[] = []; let f = 1; for (let n = 0; n <= 16; n++) { if (n > 0) f *= n; o.push(dsFrom(1 / f)); } return o; })();
/** 1 / (2k + 1), k = 0..9: the log's atanh series. */
export const DS_LOGC: DS[] = Array.from({ length: 10 }, (_, k) => dsFrom(1 / (2 * k + 1)));

/** (sin x, cos x): reduced by pi/2, Taylor on |y| <= pi/4. At k = 0 y is x itself, so sin keeps relative precision near a pole. */
export function dsSinCos(x: DS): [DS, DS] {
  const k = Math.round(F(x[0] / DS_PIO2[0])), y = k === 0 ? x : dsSub(x, dsMul(DS_PIO2, [k, 0])), z = dsMul(y, y);
  // Horner: sin y = y (1 - z/3! + z^2/5! - ...), alternating through the negated accumulator
  let s = DS_COEFFS[15];
  for (const n of [13, 11, 9, 7, 5, 3, 1]) s = dsAdd(dsMul(dsNeg(s), z), DS_COEFFS[n]);
  s = dsMul(s, y);
  let c = DS_COEFFS[16];
  for (const n of [14, 12, 10, 8, 6, 4, 2, 0]) c = dsAdd(dsMul(dsNeg(c), z), DS_COEFFS[n]);
  const q = ((k % 4) + 4) % 4;
  return q === 0 ? [s, c] : q === 1 ? [c, dsNeg(s)] : q === 2 ? [dsNeg(s), dsNeg(c)] : [dsNeg(c), s];
}
export const dsSin = (x: DS) => dsSinCos(x)[0];
export const dsCos = (x: DS) => dsSinCos(x)[1];
/** ln x: x = m 2^e with m near 1, ln m = 2 atanh((m - 1) / (m + 1)) by series (|z| <= 0.18). */
export function dsLog(x: DS): DS {
  const e = Math.round(Math.log2(x[0])), p = 2 ** -e, m: DS = [x[0] * p, x[1] * p], one: DS = [1, 0];
  const z = dsDiv(dsSub(m, one), dsAdd(m, one)), z2 = dsMul(z, z);
  let s = DS_LOGC[9]; for (let k = 8; k >= 0; k--) s = dsAdd(dsMul(s, z2), DS_LOGC[k]);
  return dsAdd(dsMul([2, 0], dsMul(z, s)), dsMul(DS_LN2, [e, 0]));
}

const lit = (v: number) => { const s = String(v); return /[.e]/.test(s) ? s : `${s}.0`; };
const pair = (d: DS) => `vec2<f32>(${lit(d[0])}, ${lit(d[1])})`;
/** WGSL consts for the shared tables plus any named values, as exact f32 pairs. */
export function dsWgslConstants(named: Record<string, number>): string {
  const lines = [`const D_PI = ${pair(DS_PI)};`, `const D_PIO2 = ${pair(DS_PIO2)};`, `const D_LN2 = ${pair(DS_LN2)};`];
  DS_COEFFS.forEach((c, n) => lines.push(`const D_C${n} = ${pair(c)};`));
  DS_LOGC.forEach((c, k) => lines.push(`const D_L${k} = ${pair(c)};`));
  for (const k of Object.keys(named)) lines.push(`const ${k} = ${pair(dsFrom(named[k]))};`);
  return lines.join("\n") + "\n";
}
