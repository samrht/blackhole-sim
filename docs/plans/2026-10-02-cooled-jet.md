# Cooled Synchrotron Jet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the visible-band energy budget with a full-spectrum one: electrons injected with a fraction
η of the jet's Blandford–Znajek power, cooled exactly by synchrotron losses, radiating the exact
cyclo-synchrotron spectrum (precomputed into a table), so every object's jet brightness conserves energy.

**Architecture:** `src/physics/cyclosynch.ts` holds the exact single-electron spectrum, the cooled population,
the table build and its lookup; `scripts/build-synch-table.mjs` runs it on all cores into
`public/synch-table.bin` (committed). `synchrotron.ts` keeps the field/flow/shift/transfer code and replaces
the budget (η × P_BZ with the high-spin correction, curved-space volume, α² Γ weighting) and the
coefficients (table-based). The shader samples the table as a texture in `emission-shared.wgsl`; the
transfer code downstream is unchanged.

**Tech Stack:** TypeScript, WebGPU/WGSL, Vite, Vitest, Node 24 (runs the .ts physics module directly for the
table build), Playwright (headless Chrome gates).

**Spec:** `docs/specs/2026-10-02-cooled-jet-design.md`

## Global Constraints

- Gaussian cgs; σ_T = 6.6524587e-25; ν_B = qB/(2π m_e c); u = γβ.
- Injection: γ_min = 40, γ_br = 1200, p = 2.2 below the break and 3.0 above (continuous); q′ = q₀ B² shape.
- Cooling: dγ/dt = −k u², k = σ_T B²/(6π m_e c); s = k t′; t′ = (r_g/c) 280^0.58/U₀ (|z|^0.42 − z_b^0.42)/0.42,
  U₀ = √(Γ₂₈₀² − 1), z_b = 2.
- Exact cyclo-synchrotron sum for γ ≤ γ_s = 10, pitch-averaged synchrotron kernel above.
- Table: ln x ∈ [ln 1e-4, ln 1e10] step 0.02 (1612 columns), ln s ∈ [ln 1e-3, ln 40] (160 rows).
- Budget: L_inj = Σ ∫ α_lapse² Γ q′ √(g_rr g_θθ g_φφ) d³x = η P_BZ; P_BZ × f(Ω_H), f = 1 + 1.38ω² − 9.2ω⁴,
  ω = a/(2 r_H). η slider log 1e-4–1, default ETA_DEFAULT (Task 4, from the M87 anchor νL_ν(550 nm) = 1e41).
- Gates green after every task: `npm test`, `npm run build`; GPU gates (`npm run verify:gpu`, dev server
  :5173, any console warning fails; `node scripts/probe-axis.mjs`, `node scripts/probe-scale.mjs`) from Task 5 on.
- `?golden` re-recorded once (Task 6) with the jet-off A/B rule (2026-10-01: if `jet-off` moves, show that a
  jet-free kernel is bit-identical to the previous build's jet-free kernel, ledger the ruling).
- No Co-Authored-By trailer on commits in this repo.

## Review Focus

1. **Spin 0 (no BZ power) and a missing table:** the jet must draw nothing, never NaN. Pinned by Task 4's
   extremes test (q₀ = 0 at a = 0) and Task 5's placeholder-texture guard + `parseTable` rejection test.
2. **Slider extremes reaching the table edges** (mass 1–1e10, accretion 1e-10–1, η 1e-4–1, Γ₂₈₀ 1.5–8):
   x and s must stay finite and the edge extensions continuous. Pinned by Task 3's edge-continuity test and
   Task 4's slider-corner test (ln x ≥ ln 1e-4 − margin, finite ln s, finite coefficients).
3. **Optically thick X-ray-binary jets in cached mode:** the cached frame must attenuate the disk exactly as
   live. Pinned by Task 6's new `xrb-thick` `?cachecheck` scene.
4. **Stale table vs code:** a table built for other injection constants must fail a test, not render.
   Pinned by Task 3's header test.
5. **Cyclotron lines from cold electrons (x ≈ 1, 2):** no spikes beyond the bin-averaged value, no negative
   absorption reaching the shader. Pinned by Task 3 (non-positive cells stored as ln = −80; count reported) and
   Task 5 parity on an X-ray-binary photon state.

---

## File Structure

- Create `src/physics/cyclosynch.ts` (+ `tests/cyclosynch.test.ts`): Bessel, single-electron kernel, synchrotron
  kernel, cooled population, table grid/build/lookup. No imports (Node runs it directly in the build).
- Create `scripts/build-synch-table.mjs` → `public/synch-table.bin` (committed, 2.0 MB).
- Modify `src/physics/synchrotron.ts` (+ `tests/synchrotron.test.ts`): budget, P_BZ correction, coefficients.
- Modify `src/render/emission-shared.wgsl`, `src/render/raytrace.wgsl`, `src/render/jet-parity.wgsl`,
  `src/render/gpu.ts`, `src/render/uniforms.ts` (+ test), `src/test/parity.browser.ts`, `src/test/scenes.ts`,
  `src/test/cachecheck.browser.ts`,
  `src/test/shadow.browser.ts`, `scripts/bench.mjs`, `scripts/shot-delay.mjs`, `src/main.ts`, `index.html`,
  `src/physics/presets.ts` (+ test), `README.md`, `docs/ROADMAP.md`, the spec's status line.

---

### Task 1: Exact single-electron spectrum

**Files:** Create `src/physics/cyclosynch.ts` (first part), `tests/cyclosynch.test.ts`

**Interfaces — Produces:** `bessel(n, z): [J, J']`, `besselRecurrence(n, z): [J, J']` (exact, any n; tests),
`gaussLegendre(n): [nodes, weights]`, `harmonicPower(x, gamma, n)`, `kernelExact(x, gamma)`,
`harmonicRange(gamma, n): [xLo, xHi]`, `synchF(y)`, `kernelSync(x, gamma)`, `GAMMA_SEAM = 10`,
`kernelBin(xa, xb, gamma)`. Units: e = m_e = c = ω_B = 1; x = ν/ν_B; kernel = power per unit angular
frequency (units e²ω_B/c), direction-integrated, pitch-averaged.

- [ ] **Step 1: Failing tests** — `tests/cyclosynch.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { bessel, besselRecurrence, kernelExact, kernelSync, synchF } from "../src/physics/cyclosynch";

describe("Bessel functions", () => {
  it("recurrence matches reference values", () => {
    expect(bessel(5, 3)[0]).toBeCloseTo(0.04302843487704758, 12);
    expect(bessel(1, 1)[0]).toBeCloseTo(0.44005058574493355, 12);
    expect(bessel(1, 1)[1]).toBeCloseTo(0.7651976865579666 - 0.44005058574493355, 12); // J1' = J0 - J1/z
  });
  it("large orders (uniform expansion, J' by recurrence) agree with the exact recurrence within 1e-3", () => {
    for (const n of [41, 100, 400]) for (const w of [0.5, 0.9, 0.99]) {
      const a = besselRecurrence(n, n * w), b = bessel(n, n * w);
      expect(Math.abs(b[0] / a[0] - 1)).toBeLessThan(1e-3);
      expect(Math.abs(b[1] / a[1] - 1)).toBeLessThan(1e-3);
    }
  });
});
describe("single-electron cyclo-synchrotron spectrum", () => {
  it("integrated over frequency it equals the Larmor power (4/9) u^2 within 0.5 %", () => {
    for (const u of [0.3, 1]) {
      const g = Math.sqrt(1 + u * u), beta = u / g, N = 1500;
      const l0 = Math.log(0.5 / g / (1 + beta)), l1 = Math.log(Math.min(4e4, 40 * (1 + 1.5 * g ** 3) / (1 - beta) / g)); let s = 0;
      for (let i = 0; i < N; i++) { const x = Math.exp(l0 + (l1 - l0) * (i + 0.5) / N); s += kernelExact(x, g) * x * (l1 - l0) / N; }
      expect(Math.abs(s / ((4 / 9) * u * u) - 1)).toBeLessThan(5e-3); // measured 0.07 % / 0.13 %
    }
  }, 120000);
  it("synchrotron function F matches reference values", () => {
    expect(Math.abs(synchF(1) / 0.651423 - 1)).toBeLessThan(1e-4);
    expect(Math.abs(synchF(0.29) / 0.918 - 1)).toBeLessThan(5e-4);
    expect(synchF(1e-6 * 1.001) / synchF(1e-6 * 0.999)).toBeCloseTo(Math.cbrt(1.001 / 0.999), 3);
  });
  it("at the seam (gamma = 10) the synchrotron kernel is within 1.5 % of the exact sum, power-weighted", () => {
    const g = 10; let num = 0, den = 0;
    for (let i = 0; i < 40; i++) { const x = Math.exp(Math.log(0.5) + (Math.log(40 * 1.5 * g * g) - Math.log(0.5)) * i / 39);
      const e = kernelExact(x, g), s = kernelSync(x, g); num += Math.abs(s - e) * x; den += e * x; }
    expect(num / den).toBeLessThan(0.015); // measured 1.3e-2 (14 % at gamma 3, 5.1 % at 5)
  }, 120000);
});
```

- [ ] **Step 2: Run** — `npx vitest run tests/cyclosynch.test.ts` → FAIL (module missing).

- [ ] **Step 3: Implement** — `src/physics/cyclosynch.ts`:

```ts
// Exact cyclo-synchrotron emission and the cooled electron population (spec 2026-10-02 cooled jet).
// Self-contained (no imports): scripts/build-synch-table.mjs runs it directly in Node to build the table.
// Kernel units: e = m_e = c = omega_B = 1, x = nu / nu_B; a kernel is the power per unit angular frequency
// (units e^2 omega_B / c), integrated over emission direction and averaged over an isotropic pitch angle.

// ---- Bessel J_n(z), J_n'(z) ------------------------------------------------------------------------------
const NM = 40;
/** Exact J_n, J_n' by Miller's backward recurrence (normalised with J_0 + 2 sum J_2k = 1). */
export function besselRecurrence(n: number, z: number): [number, number] {
  if (z === 0) return [n === 0 ? 1 : 0, n === 1 ? 0.5 : 0];
  const N = Math.max(n, Math.ceil(z)) + 40 + Math.ceil(Math.sqrt(40 * Math.max(n, z)));
  let jp1 = 0, j = 1e-300, sum = 0, Jn = 0, Jn1 = 0;
  for (let k = N; k >= 1; k--) {
    const jm1 = (2 * k / z) * j - jp1; jp1 = j; j = jm1;
    if (Math.abs(j) > 1e250) { j *= 1e-250; jp1 *= 1e-250; sum *= 1e-250; Jn *= 1e-250; Jn1 *= 1e-250; }
    if (k - 1 === n) { Jn = j; Jn1 = jp1; }
    if (k - 1 > 0 && (k - 1) % 2 === 0) sum += 2 * j;
  }
  sum += j;
  const jn = Jn / sum, jn1 = Jn1 / sum;
  return [jn, (n / z) * jn - jn1];
}
function airy(t: number): [number, number] {
  if (t > 6) {
    const zeta = (2 / 3) * Math.pow(t, 1.5), e = Math.exp(-zeta) / (2 * Math.sqrt(Math.PI));
    return [e * Math.pow(t, -0.25) * (1 - 5 / (72 * zeta) + 385 / (10368 * zeta * zeta)),
      -e * Math.pow(t, 0.25) * (1 + 7 / (72 * zeta) - 455 / (10368 * zeta * zeta))];
  }
  if (t < -6) {
    const at = -t, zeta = (2 / 3) * Math.pow(at, 1.5), ph = zeta + Math.PI / 4;
    return [Math.pow(at, -0.25) / Math.sqrt(Math.PI) * (Math.sin(ph) * (1 - 385 / (10368 * zeta * zeta)) - Math.cos(ph) * 5 / (72 * zeta)),
      -Math.pow(at, 0.25) / Math.sqrt(Math.PI) * (Math.cos(ph) * (1 + 455 / (10368 * zeta * zeta)) + Math.sin(ph) * 7 / (72 * zeta))];
  }
  const c1 = 0.355028053887817, c2 = 0.258819403792807;
  let f = 1, g = t, fp = 0, gp = 1, tf = 1, tg = t; const t3 = t * t * t;
  for (let k = 1; k < 60; k++) {
    tf *= t3 / ((3 * k - 1) * (3 * k)); tg *= t3 / ((3 * k) * (3 * k + 1));
    f += tf; g += tg; if (t !== 0) { fp += tf * 3 * k / t; gp += tg * (3 * k + 1) / t; }
    if (Math.abs(tf) + Math.abs(tg) < 1e-17 * (Math.abs(f) + Math.abs(g))) break;
  }
  return [c1 * f - c2 * g, c1 * fp - c2 * gp];
}
/** J_nu(z) for real order nu > 0 from the uniform (Airy) expansion. */
function jAiry(nu: number, z: number): number {
  const w = z / nu;
  if (Math.abs(w - 1) < 1e-9) return Math.pow(2, 1 / 3) * airy(0)[0] / Math.pow(nu, 1 / 3);
  let zeta: number;
  if (w > 1) { const s = Math.sqrt(w * w - 1); zeta = -Math.pow(1.5 * (s - Math.acos(1 / w)), 2 / 3); }
  else { const s = Math.sqrt(1 - w * w); zeta = Math.pow(1.5 * (Math.log((1 + s) / w) - s), 2 / 3); }
  return Math.pow(4 * zeta / (1 - w * w), 0.25) * airy(Math.pow(nu, 2 / 3) * zeta)[0] / Math.pow(nu, 1 / 3);
}
/** [J_n(z), J_n'(z)]: exact recurrence for n <= 40; above, the uniform expansion for J (6e-4 at n = 41, falling
 *  with n) and J' = (J_{n-1} - J_{n+1}) / 2 (the expansion's own J' term is 1.8 % off at n = 41). */
export function bessel(n: number, z: number): [number, number] {
  if (n <= NM) return besselRecurrence(n, z);
  return [jAiry(n, z), 0.5 * (jAiry(n - 1, z) - jAiry(n + 1, z))];
}

// ---- quadrature ------------------------------------------------------------------------------------------
export function gaussLegendre(n: number): [number[], number[]] {
  const x: number[] = [], w: number[] = [];
  for (let i = 1; i <= n; i++) {
    let r = Math.cos(Math.PI * (i - 0.25) / (n + 0.5)), dp = 0;
    for (let it = 0; it < 100; it++) {
      let p0 = 1, p1 = r;
      for (let k = 2; k <= n; k++) { const p2 = ((2 * k - 1) * r * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
      dp = n * (r * p1 - p0) / (r * r - 1);
      const dr = p1 / dp; r -= dr; if (Math.abs(dr) < 1e-15) break;
    }
    x.push(r); w.push(2 / ((1 - r * r) * dp * dp));
  }
  return [x, w];
}
const [GX, GW] = gaussLegendre(48);
const GL8 = gaussLegendre(8);

// ---- single electron -------------------------------------------------------------------------------------
/** Harmonic n's contribution at x for an electron of Lorentz factor gamma (Bekefi 1966 emissivity; the resonance
 *  delta removes the direction integral; mu = cos pitch, mu' = cos(view angle to B), mu mu' = C). */
export function harmonicPower(x: number, gamma: number, n: number): number {
  const beta = Math.sqrt(1 - 1 / (gamma * gamma)), Om = 1 / gamma;
  const C = (1 - n * Om / x) / beta, aC = Math.abs(C); if (aC > 1) return 0;
  const oneMinus = 1 - beta * C;
  const wMax = Math.min(1, beta * (1 - aC) / oneMinus); // J_n(n w) ~ exp(-n (atanh s - s)): prune the negligible
  if (wMax < 1) { const sq = Math.sqrt(1 - wMax * wMax); if (n * (Math.atanh(sq) - sq) > 60) return 0; }
  let part = 0;
  for (const sgn of [1, -1]) for (let k = 0; k < GX.length; k++) {
    const v = 0.5 * (GX[k] + 1), dv = 0.5 * GW[k], mu = sgn * (aC + (1 - aC) * v * v), dmu = (1 - aC) * 2 * v * dv;
    const mup = C / mu; if (Math.abs(mup) > 1) continue;
    const sinxi = Math.sqrt(Math.max(0, 1 - mu * mu)), sinth = Math.sqrt(Math.max(0, 1 - mup * mup));
    const bpar = beta * mu, bperp = beta * sinxi, z = n * bperp * sinth / oneMinus, [J, Jp] = bessel(n, z);
    const ang = sinth > 1e-14 ? Math.pow((mup - bpar) / sinth, 2) * J * J : 0;
    part += (x * x / (2 * Math.PI)) * (ang + bperp * bperp * Jp * Jp) / (x * beta * Math.abs(mu)) * dmu;
  }
  return Math.PI * part;
}
/** Exact single-electron kernel: the sum over every harmonic. */
export function kernelExact(x: number, gamma: number): number {
  const beta = Math.sqrt(1 - 1 / (gamma * gamma)), Om = 1 / gamma;
  const nLo = Math.max(1, Math.ceil(x * (1 - beta) / Om - 1e-12)), nHi = Math.floor(x * (1 + beta) / Om + 1e-12);
  let tot = 0;
  for (let n = nLo; n <= nHi; n++) tot += harmonicPower(x, gamma, n);
  return tot;
}
/** Harmonic n's frequency interval [x_lo, x_hi] (pitch-averaged). */
export function harmonicRange(gamma: number, n: number): [number, number] {
  const beta = Math.sqrt(1 - 1 / (gamma * gamma));
  return [n / (gamma * (1 + beta)), n / (gamma * (1 - beta))];
}

// ---- synchrotron kernel (gamma >> 1) ---------------------------------------------------------------------
// F(y) = y Int_y^inf K_5/3 = y Int_0^inf exp(-y cosh tau) cosh(5 tau / 3) / cosh(tau) dtau, tabulated once.
function fDirect(y: number): number {
  const N = 4000, tmax = Math.acosh(1 + 60 / y), h = tmax / N; let s = 0;
  for (let i = 0; i <= N; i++) { const tau = i * h, w = i === 0 || i === N ? 1 : i % 2 ? 4 : 2, c = Math.cosh(tau);
    s += w * Math.exp(-y * c) * Math.cosh((5 * tau) / 3) / c; }
  return (y * s * h) / 3;
}
// Built on first use: the browser imports this module (table lookup) but never evaluates F.
let F_TAB: { a: number; b: number; n: number; lnF: number[] } | null = null;
function fTab() {
  if (!F_TAB) {
    const n = 1200, a = Math.log(1e-6), b = Math.log(60), lnF: number[] = [];
    for (let i = 0; i <= n; i++) lnF.push(Math.log(fDirect(Math.exp(a + (b - a) * i / n))));
    F_TAB = { a, b, n, lnF };
  }
  return F_TAB;
}
/** Synchrotron function F(y) (Rybicki & Lightman 6.31c). */
export function synchF(y: number): number {
  if (y <= 1e-6) return 2.1495282 * Math.cbrt(y);
  if (y >= 60) return Math.sqrt(Math.PI / 2) * Math.sqrt(y) * Math.exp(-y) * (1 + 55 / (72 * y));
  const T = fTab(), f = (Math.log(y) - T.a) / (T.b - T.a) * T.n, i = Math.min(T.n - 1, Math.floor(f)), t = f - i;
  return Math.exp(T.lnF[i] * (1 - t) + T.lnF[i + 1] * t);
}
/** Pitch-averaged synchrotron kernel, same units as kernelExact. */
export function kernelSync(x: number, gamma: number): number {
  let s = 0; const N = 64;
  for (let i = 0; i < N; i++) { const mu = (i + 0.5) / N, sx = Math.sqrt(1 - mu * mu); s += (Math.sqrt(3) * sx / (2 * Math.PI)) * synchF(x / (1.5 * gamma * gamma * sx)); }
  return s / N;
}
/** Exact sum up to the seam, synchrotron kernel above (spec 2.3: 1.3 % power-weighted at gamma = 10). */
export const GAMMA_SEAM = 10;
/** Kernel averaged over the bin [xa, xb]: harmonic lines (few per bin) integrated over their overlap with the bin,
 *  the bin centre where more than 50 overlap (the sum is smooth there). */
export function kernelBin(xa: number, xb: number, gamma: number): number {
  if (gamma > GAMMA_SEAM) return kernelSync(Math.sqrt(xa * xb), gamma);
  const beta = Math.sqrt(1 - 1 / (gamma * gamma));
  const nLo = Math.max(1, Math.ceil(xa * gamma * (1 - beta) - 1e-12)), nHi = Math.floor(xb * gamma * (1 + beta) + 1e-12);
  if (nHi < nLo) return 0;
  if (nHi - nLo > 50) return kernelExact(Math.sqrt(xa * xb), gamma);
  let tot = 0; const [gx, gw] = GL8;
  for (let n = nLo; n <= nHi; n++) {
    const [ra, rb] = harmonicRange(gamma, n), a = Math.max(xa, ra), b = Math.min(xb, rb);
    if (b <= a) continue;
    for (let k = 0; k < gx.length; k++) { // x = a + (b-a)(1 - cos(pi t))/2 clusters nodes at the line edges
      const t = 0.5 * (gx[k] + 1), x = a + (b - a) * (1 - Math.cos(Math.PI * t)) / 2, dx = (b - a) * Math.PI * Math.sin(Math.PI * t) / 2 * 0.5 * gw[k];
      tot += harmonicPower(x, gamma, n) * dx;
    }
  }
  return tot / (xb - xa);
}
```

- [ ] **Step 4: Run** — `npx vitest run tests/cyclosynch.test.ts` → PASS (5; ~40 s). `npm test`, `npm run build` green.

- [ ] **Step 5: Commit** — `git add src/physics/cyclosynch.ts tests/cyclosynch.test.ts && git commit -m "Exact cyclo-synchrotron single-electron spectrum, validated against Larmor and the synchrotron limit"`

---

### Task 2: Cooled electron population

**Files:** Modify `src/physics/cyclosynch.ts`, `tests/cyclosynch.test.ts`

**Interfaces — Produces:** `G_MIN = 40, G_BR = 1200, P1 = 2.2, P2 = 3.0`; `gInj(gamma)`; `gTail(gamma)` (∫_γ^∞ g);
`I_G` (∫ (γ − 1) g dγ = 1.4122096931705266); `coolFromU(u, s)`; `nHatU(u, s)` (N k/Q₀ per unit γ);
`dNOverGuU(u, s)` (∂/∂γ [N̂/(γu)]); `nHat(gamma, s)`, `dNOverGu(gamma, s)` (γ wrappers).

- [ ] **Step 1: Failing tests** — append (add `nHatU, nHat, dNOverGu, gInj, I_G` to the import):

```ts
describe("cooled population (exact, constant injection for a cooling depth s)", () => {
  it("solves the continuity equation d/dgamma [u^2 N^] = -g in the fully cooled range", () => {
    for (const g of [50, 500, 5000]) { const h = 1e-4 * g;
      const d = ((g + h) ** 2 - 1) * nHat(g + h, 10) - ((g - h) ** 2 - 1) * nHat(g - h, 10);
      expect(d / (2 * h) / -gInj(g)).toBeCloseTo(1, 6); }
  });
  it("radiates what it is given when fast cooling, less when slow (calorimetry)", () => {
    const frac = (s: number) => { let r = 0; const N = 200000, l0 = Math.log(1e-12), l1 = Math.log(1e7);
      for (let i = 0; i < N; i++) { const u = Math.exp(l0 + (l1 - l0) * (i + 0.5) / N); r += nHatU(u, s) * u * u * (u * u / Math.sqrt(1 + u * u)) * (l1 - l0) / N; } return r / I_G; };
    expect(frac(10)).toBeCloseTo(1, 4);
    expect(frac(1e-3)).toBeLessThan(0.3); // measured 0.285: most injected energy not yet radiated
  });
  it("I_G matches a direct integral", () => {
    let s = 0; const N = 400000, l0 = Math.log(40), l1 = Math.log(1e9);
    for (let i = 0; i < N; i++) { const g = Math.exp(l0 + (l1 - l0) * (i + 0.5) / N); s += (g - 1) * gInj(g) * g * (l1 - l0) / N; }
    expect(s / I_G).toBeCloseTo(1, 5);
  });
  it("the absorption derivative is the analytic derivative of N^ / (gamma u)", () => {
    for (const [g, s] of [[3, 10], [50, 10], [2000, 10], [60, 0.01]]) { const h = 1e-6 * g, f = (x: number) => nHat(x, s) / (x * Math.sqrt(x * x - 1));
      expect(dNOverGu(g, s) / ((f(g + h) - f(g - h)) / (2 * h))).toBeCloseTo(1, 5); }
  });
  it("slow cooling (s -> 0) gives N^ = s g above gamma_min", () => {
    expect(nHat(100, 1e-6) / (1e-6 * gInj(100))).toBeCloseTo(1, 3);
  });
});
```
Run → FAIL.

- [ ] **Step 2: Implement** — append to `cyclosynch.ts`:

```ts
// ---- cooled electron population (spec 2.1-2.2) -----------------------------------------------------------
/** Injection fitted to M87's core SED: N ~ gamma^-2.2 (alpha 1.1) for 40..1200, gamma^-3.0 (alpha 1.5) above. */
export const G_MIN = 40, G_BR = 1200, P1 = 2.2, P2 = 3.0;
export const gInj = (g: number) => (g < G_MIN ? 0 : g <= G_BR ? Math.pow(g, -P1) : Math.pow(G_BR, P2 - P1) * Math.pow(g, -P2));
/** Int_gamma^inf g. */
export function gTail(g: number): number {
  if (!Number.isFinite(g)) return 0;
  const a = Math.max(g, G_MIN), t2 = Math.pow(G_BR, P2 - P1) * Math.pow(Math.max(a, G_BR), 1 - P2) / (P2 - 1);
  return a >= G_BR ? t2 : (Math.pow(a, 1 - P1) - Math.pow(G_BR, 1 - P1)) / (P1 - 1) + t2;
}
/** I_g = Int (gamma - 1) g dgamma: injected kinetic energy per unit Q0, in m_e c^2. */
export const I_G = (Math.pow(G_BR, 2 - P1) - Math.pow(G_MIN, 2 - P1)) / (2 - P1) - (Math.pow(G_MIN, 1 - P1) - Math.pow(G_BR, 1 - P1)) / (P1 - 1)
  + Math.pow(G_BR, P2 - P1) * (Math.pow(G_BR, 2 - P2) / (P2 - 2) - Math.pow(G_BR, 1 - P2) / (P2 - 1));
/** g(G) (G^2 - 1) without overflow for huge G. */
const gInjG2 = (G: number) => (G > 1e100 ? Math.pow(G_BR, P2 - P1) * Math.pow(G, 2 - P2) : gInj(G) * (G * G - 1));
/** acoth(gamma) from u = gamma beta without cancellation: ln((gamma + 1) / u). */
const acothU = (u: number) => Math.log((1 + Math.sqrt(1 + u * u)) / u);
/** Injection energy that cools to momentum u within cooling depth s: coth(acoth(gamma) - s), or Infinity. */
export function coolFromU(u: number, s: number): number { const d = acothU(u) - s; return d > 0 ? 1 / Math.tanh(d) : Infinity; }
/** N^ = N k / Q0 per unit gamma at momentum u: (1/u^2) Int_gamma^G g (spec 2.2). Written in u: gamma rounds to 1
 *  for u < 1e-8 in double precision, and the cold electrons matter for absorption at the fundamental. */
export function nHatU(u: number, s: number): number { const g = Math.sqrt(1 + u * u); return (gTail(g) - gTail(coolFromU(u, s))) / (u * u); }
/** d/dgamma [N^ / (gamma u)] (analytic; dG/dgamma = (G^2 - 1) / u^2). */
export function dNOverGuU(u: number, s: number): number {
  const g = Math.sqrt(1 + u * u), u2 = u * u, G = coolFromU(u, s), T = gTail(g) - gTail(G);
  const dT = -gInj(g) + (Number.isFinite(G) ? gInjG2(G) / u2 : 0);
  const den = g * u2 * u;
  return (dT * den - T * (u2 * u + 3 * g * g * u)) / (den * den);
}
export const nHat = (g: number, s: number) => nHatU(Math.sqrt(g * g - 1), s);
export const dNOverGu = (g: number, s: number) => dNOverGuU(Math.sqrt(g * g - 1), s);
```

- [ ] **Step 3: Run / Commit** — tests PASS, suite + build green;
`git add src/physics/cyclosynch.ts tests/cyclosynch.test.ts && git commit -m "Exact synchrotron-cooled electron population with closed-form cooling time"`

---

### Task 3: Coefficient table: build, file, lookup

**Files:** Modify `src/physics/cyclosynch.ts`, `tests/cyclosynch.test.ts`; Create `scripts/build-synch-table.mjs`,
`public/synch-table.bin`

**Interfaces — Produces:** `TABLE_GRID = { lnx0, lnx1, nx: 1612, lns0, lns1, ns: 160 }`, `U_COLD = 0.01`,
`uGrid(): { u: number[]; dlnu: number[] }`, `kernelRow(u, grid, out: Float64Array)`,
`contract(rows, uu, grid, pop?): { J, A }` (`pop = { n(u, s), d(u, s) }`, default the cooled population),
`SynchTable = { nx, ns, lnx0, lnx1, lns0, lns1, data: Float32Array }` (pairs ln Ĵ, ln Â; s rows, x columns),
`parseTable(buf: ArrayBuffer): SynchTable` (throws on bad magic/version or constants that differ from the code),
`encodeTable(J, A, grid): ArrayBuffer`, `lookup(t, lnx, lns): [lnJ, lnA]`.
Table meaning: Ĵ(x, s) = ∫ N̂ p̂ dγ, Â(x, s) = −∫ p̂ γu ∂_γ(N̂/(γu)) dγ, with p̂ = 2π × kernel (per Hz, units e³B/m_e c²).

- [ ] **Step 1: Failing tests** — append (import the new names; `readFileSync` from `node:fs`, `join` from `node:path`):

```ts
const TAB = () => parseTable(readFileSync(join(__dirname, "../public/synch-table.bin")).buffer.slice(0) as ArrayBuffer);
describe("coefficient table (public/synch-table.bin)", () => {
  it("header matches the code's grid and injection constants (a stale table fails here)", () => {
    const t = TAB();
    expect([t.nx, t.ns]).toEqual([TABLE_GRID.nx, TABLE_GRID.ns]);
    expect(t.lnx0).toBeCloseTo(TABLE_GRID.lnx0, 5); expect(t.lns1).toBeCloseTo(TABLE_GRID.lns1, 5);
    const bad = new Uint8Array(readFileSync(join(__dirname, "../public/synch-table.bin"))); bad[0] ^= 1;
    expect(() => parseTable(bad.buffer)).toThrow();
  });
  it("calorimetry: Int J^ d omega over the table equals the population's radiated power (within 0.5 %)", () => {
    const t = TAB(), h = (t.lnx1 - t.lnx0) / t.nx;
    for (const js of [0, 80, 159]) {
      const s = Math.exp(t.lns0 + (t.lns1 - t.lns0) * js / (t.ns - 1)); let tot = 0, pop = 0;
      for (let i = 0; i < t.nx; i++) { const x = Math.exp(t.lnx0 + h * (i + 0.5)); tot += Math.exp(t.data[2 * (js * t.nx + i)]) / (2 * Math.PI) * x * h; }
      const uu = uGrid(); uu.u.forEach((u, k) => { pop += nHatU(u, s) * u * u * (u * u / Math.sqrt(1 + u * u)) * uu.dlnu[k]; });
      expect(Math.abs(tot / ((4 / 9) * pop) - 1)).toBeLessThan(5e-3); // measured 4e-3 (s = 1e-3), 1e-4 elsewhere
    }
  });
  it("lookup agrees with a direct computation at off-grid points (1 % emission, 2 % absorption)", () => {
    const t = TAB(), uu = uGrid(), h = (t.lnx1 - t.lnx0) / t.nx;
    for (const [x, s] of [[3.7e2, 5], [2.1e4, 0.3], [2.5e6, 12], [1.6, 30], [55, 0.01]] as [number, number][]) {
      const xa = x * Math.exp(-h / 2), xb = x * Math.exp(h / 2); let J = 0, A = 0;
      uu.u.forEach((u, k) => { const g = Math.sqrt(1 + u * u), dg = u * u / g * uu.dlnu[k];
        const kb = u < U_COLD || xa > 90 * g * g + 10 ? 0 : kernelBin(xa, xb, g);
        J += nHatU(u, s) * 2 * Math.PI * kb * dg; A += -2 * Math.PI * g * u * dNOverGuU(u, s) * kb * dg; });
      const [lJ, lA] = lookup(t, Math.log(x), Math.log(s));
      expect(Math.abs(Math.exp(lJ) / J - 1)).toBeLessThan(1e-2);
      expect(Math.abs(Math.exp(lA) / A - 1)).toBeLessThan(2e-2); // measured <= 2.3e-3 except 1.04e-2 at (55, 0.01): slow-cooling absorption varies steeply in s
    }
  }, 300000);
  it("Kirchhoff: for a thermal population the contraction gives A^ = J^ / Theta", () => {
    const g = { ...TABLE_GRID, nx: 40, lnx0: Math.log(0.5), lnx1: Math.log(400), ns: 2 }, uu = uGrid();
    const keep = uu.u.map((u, k) => k).filter((k) => uu.u[k] > 0.05 && uu.u[k] < 30);
    const sub = { u: keep.map((k) => uu.u[k]), dlnu: keep.map((k) => uu.dlnu[k]) };
    const rows = sub.u.map((u) => { const r = new Float64Array(g.nx); kernelRow(u, g, r); return r; });
    for (const Th of [0.3, 1, 3]) {
      // thermal (Maxwell-Juttner) N = gamma u exp(-gamma/Theta) per unit gamma: N/(gamma u) = exp(-gamma/Theta)
      const th = { n: (u: number) => { const gm = Math.sqrt(1 + u * u); return gm * u * Math.exp(-gm / Th); },
                   d: (u: number) => -Math.exp(-Math.sqrt(1 + u * u) / Th) / Th };
      const { J, A } = contract(rows, sub, g, th);
      for (let i = 5; i < g.nx; i += 7) if (J[i] > 0) expect(J[i] / (A[i] * Th)).toBeCloseTo(1, 6);
    }
  }, 120000);
  it("edge extensions are continuous (top of x, bottom of s) and the slider range stays inside x >= 1e-4", () => {
    const t = TAB(), e = 1e-6;
    for (const lns of [Math.log(0.1), Math.log(10)]) {
      const a = lookup(t, t.lnx1 - (t.lnx1 - t.lnx0) / t.nx * 0.5 - e, lns), b = lookup(t, t.lnx1 - (t.lnx1 - t.lnx0) / t.nx * 0.5 + e, lns);
      expect(Math.abs(a[0] - b[0])).toBeLessThan(1e-4); expect(Math.abs(a[1] - b[1])).toBeLessThan(1e-4);
    }
    const c = lookup(t, Math.log(1e5), t.lns0 + e), d = lookup(t, Math.log(1e5), t.lns0 - e);
    expect(Math.abs(c[0] - d[0])).toBeLessThan(1e-4);
  });
});
```
Run → FAIL (exports and file missing).

- [ ] **Step 2: Implement** — append to `cyclosynch.ts`:

```ts
// ---- the table (spec 2.3) --------------------------------------------------------------------------------
export interface Grid { lnx0: number; lnx1: number; nx: number; lns0: number; lns1: number; ns: number }
export const TABLE_GRID: Grid = { lnx0: Math.log(1e-4), lnx1: Math.log(1e10), nx: 1612, lns0: Math.log(1e-3), lns1: Math.log(40), ns: 160 };
/** Below this momentum the dipole limit: the fundamental line alone, carrying the Larmor power. */
export const U_COLD = 0.01;
/** Momentum grid for the gamma integral: cold (1e-18..0.01), exact (to the seam), synchrotron (to 1e7). */
export function uGrid(): { u: number[]; dlnu: number[] } {
  const u: number[] = [], d: number[] = [];
  const seg = (a: number, b: number, step: number) => { const n = Math.ceil((Math.log(b) - Math.log(a)) / step), h = (Math.log(b) - Math.log(a)) / n;
    for (let i = 0; i < n; i++) { u.push(Math.exp(Math.log(a) + h * (i + 0.5))); d.push(h); } };
  seg(1e-18, U_COLD, 0.1);
  seg(U_COLD, Math.sqrt(GAMMA_SEAM * GAMMA_SEAM - 1), 0.03);
  seg(Math.sqrt(GAMMA_SEAM * GAMMA_SEAM - 1), 1e7, 0.05);
  return { u, dlnu: d };
}
/** Bin-averaged kernel for one momentum over all x bins (independent of s). */
export function kernelRow(u: number, g: Grid, out: Float64Array) {
  const gam = Math.sqrt(1 + u * u), h = (g.lnx1 - g.lnx0) / g.nx;
  out.fill(0);
  if (u < U_COLD) { // fundamental line [1/(gamma(1+beta)), gamma(1+beta)] with power (4/9) u^2, spread over its bins
    const beta = u / gam, a = 1 / (gam * (1 + beta)), b = gam * (1 + beta), P = (4 / 9) * u * u;
    if (!(b - a > 1e-9 * a)) { // narrower than double precision resolves: all in the bin holding x = 1/gamma
      const i = Math.floor((Math.log(1 / gam) - g.lnx0) / h);
      if (i >= 0 && i < g.nx) out[i] = P / (Math.exp(g.lnx0 + h * (i + 1)) - Math.exp(g.lnx0 + h * i));
      return;
    }
    const ia = Math.floor((Math.log(a) - g.lnx0) / h), ib = Math.floor((Math.log(b) - g.lnx0) / h);
    for (let i = Math.max(0, ia); i <= Math.min(g.nx - 1, ib); i++) {
      const xa = Math.exp(g.lnx0 + h * i), xb = Math.exp(g.lnx0 + h * (i + 1)), ov = Math.max(0, Math.min(b, xb) - Math.max(a, xa));
      out[i] = P * (ov / (b - a)) / (xb - xa);
    }
    return;
  }
  const xmax = 90 * gam * gam + 10; // beyond 60 x the critical frequency the kernel is below exp(-60)
  for (let i = 0; i < g.nx; i++) {
    const xa = Math.exp(g.lnx0 + h * i), xb = Math.exp(g.lnx0 + h * (i + 1));
    if (xa > xmax) break;
    out[i] = kernelBin(xa, xb, gam);
  }
}
export interface Population { n(u: number, s: number): number; d(u: number, s: number): number }
const COOLED: Population = { n: nHatU, d: dNOverGuU };
/** J^ = Int N^ p^ dgamma, A^ = -Int p^ gamma u d/dgamma(N^/(gamma u)) dgamma on the (x, s) grid (p^ = 2 pi kernel). */
export function contract(rows: Float64Array[], uu: { u: number[]; dlnu: number[] }, g: Grid, pop: Population = COOLED) {
  const J = new Float64Array(g.nx * g.ns), A = new Float64Array(g.nx * g.ns);
  for (let js = 0; js < g.ns; js++) {
    const s = Math.exp(g.lns0 + (g.lns1 - g.lns0) * js / (g.ns - 1));
    for (let k = 0; k < uu.u.length; k++) {
      const u = uu.u[k], gam = Math.sqrt(1 + u * u), dg = (u * u / gam) * uu.dlnu[k];
      const n = pop.n(u, s), dn = pop.d(u, s); if (n === 0 && dn === 0) continue;
      const wj = n * 2 * Math.PI * dg, wa = -2 * Math.PI * gam * u * dn * dg, row = rows[k];
      for (let i = 0; i < g.nx; i++) { const r = row[i]; if (r === 0) continue; J[js * g.nx + i] += wj * r; A[js * g.nx + i] += wa * r; }
    }
  }
  return { J, A };
}
// File: Uint32 [magic 'SYNT', version 1, nx, ns], Float32 [lnx0, lnx1, lns0, lns1, G_MIN, G_BR, P1, P2, GAMMA_SEAM,
// U_COLD, 0, 0], then Float32 pairs (ln J^, ln A^), s rows of x columns. Non-positive values (frequencies no
// electron reaches; a few slow-cooling maser cells) are stored as ln = -80, i.e. zero.
const MAGIC = 0x544e5953, HEAD = 16;
export function encodeTable(J: Float64Array, A: Float64Array, g: Grid): ArrayBuffer {
  const buf = new ArrayBuffer(HEAD * 4 + g.nx * g.ns * 8), u32 = new Uint32Array(buf, 0, 4), f32 = new Float32Array(buf);
  u32.set([MAGIC, 1, g.nx, g.ns]);
  f32.set([g.lnx0, g.lnx1, g.lns0, g.lns1, G_MIN, G_BR, P1, P2, GAMMA_SEAM, U_COLD, 0, 0], 4);
  for (let i = 0; i < g.nx * g.ns; i++) { f32[HEAD + 2 * i] = J[i] > 0 ? Math.log(J[i]) : -80; f32[HEAD + 2 * i + 1] = A[i] > 0 ? Math.log(A[i]) : -80; }
  return buf;
}
export interface SynchTable { nx: number; ns: number; lnx0: number; lnx1: number; lns0: number; lns1: number; data: Float32Array }
export function parseTable(buf: ArrayBuffer): SynchTable {
  const u32 = new Uint32Array(buf, 0, 4), f32 = new Float32Array(buf);
  if (u32[0] !== MAGIC || u32[1] !== 1) throw new Error("synch-table.bin: bad magic or version");
  const c = [G_MIN, G_BR, P1, P2, GAMMA_SEAM, U_COLD];
  c.forEach((v, i) => { if (Math.abs(f32[8 + i] - v) > 1e-6 * Math.abs(v)) throw new Error("synch-table.bin: built for other constants; rebuild it"); });
  return { nx: u32[2], ns: u32[3], lnx0: f32[4], lnx1: f32[5], lns0: f32[6], lns1: f32[7], data: f32.subarray(HEAD) };
}
/** Bilinear (ln J^, ln A^) at (ln x, ln s) between cell centres. Above the x grid: the fast-cooled gamma^-4
 *  asymptote (J^ ~ x^-1.5, A^ ~ x^-2); below: the first column; below the s grid: slope 1 in ln s (N^ ~ s);
 *  above: the last row. Twin of synchLookupJ in emission-shared.wgsl. */
export function lookup(t: SynchTable, lnx: number, lns: number): [number, number] {
  const h = (t.lnx1 - t.lnx0) / t.nx;
  let fx = (lnx - t.lnx0) / h - 0.5, extX = 0;
  if (fx > t.nx - 1) { extX = (fx - (t.nx - 1)) * h; fx = t.nx - 1; }
  if (fx < 0) fx = 0;
  let fs = (lns - t.lns0) / (t.lns1 - t.lns0) * (t.ns - 1), extS = 0;
  if (fs < 0) { extS = lns - t.lns0; fs = 0; }
  if (fs > t.ns - 1) fs = t.ns - 1;
  const ix = Math.min(t.nx - 2, Math.floor(fx)), is = Math.min(t.ns - 2, Math.floor(fs)), ax = fx - ix, as = fs - is;
  const v = (i: number, j: number, c: number) => t.data[2 * (j * t.nx + i) + c];
  const bil = (c: number) => (1 - as) * ((1 - ax) * v(ix, is, c) + ax * v(ix + 1, is, c)) + as * ((1 - ax) * v(ix, is + 1, c) + ax * v(ix + 1, is + 1, c));
  return [bil(0) - 1.5 * extX + extS, bil(1) - 2 * extX + extS];
}
```

- [ ] **Step 3: Build script** — `scripts/build-synch-table.mjs`:

```js
// Builds public/synch-table.bin from src/physics/cyclosynch.ts on every core (Node >= 23 runs the .ts directly).
// ~20 min on 8 cores: the exact harmonic sums (u = 0.01..10) dominate. ROWS=<file> caches the s-independent
// kernel rows, so changing only the s grid re-contracts in seconds.
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { writeFileSync, readFileSync, existsSync } from "node:fs";
import { availableParallelism } from "node:os";
const M = await import(new URL("../src/physics/cyclosynch.ts", import.meta.url).href);
if (!isMainThread) {
  const { idx } = workerData, uu = M.uGrid(), g = M.TABLE_GRID;
  for (const k of idx) { const row = new Float64Array(g.nx); M.kernelRow(uu.u[k], g, row); parentPort.postMessage({ k, row }, [row.buffer]); }
  parentPort.postMessage({ done: true });
} else {
  const g = M.TABLE_GRID, uu = M.uGrid(), W = availableParallelism(), rows = new Array(uu.u.length), cache = process.env.ROWS;
  const t0 = Date.now();
  if (cache && existsSync(cache)) {
    const all = new Float64Array(readFileSync(cache).buffer.slice(0));
    for (let k = 0; k < uu.u.length; k++) rows[k] = all.slice(k * g.nx, (k + 1) * g.nx);
  } else {
    let left = W, got = 0;
    await new Promise((resolve, reject) => {
      for (let w = 0; w < W; w++) {
        const idx = []; for (let k = w; k < uu.u.length; k += W) idx.push(k); // interleaved: spreads the costly rows
        const wk = new Worker(new URL(import.meta.url), { workerData: { idx } });
        wk.on("message", (m) => { if (m.done) { if (--left === 0) resolve(); return; } rows[m.k] = m.row;
          if (++got % 50 === 0) console.log(`kernel rows ${got}/${uu.u.length}, ${((Date.now() - t0) / 1000).toFixed(0)} s`); });
        wk.on("error", reject);
      }
    });
    if (cache) { const all = new Float64Array(uu.u.length * g.nx); rows.forEach((r, k) => all.set(r, k * g.nx)); writeFileSync(cache, Buffer.from(all.buffer)); }
  }
  const { J, A } = M.contract(rows, uu, g);
  let zeroA = 0; for (const v of A) if (!(v > 0)) zeroA++;
  writeFileSync(new URL("../public/synch-table.bin", import.meta.url), Buffer.from(M.encodeTable(J, A, g)));
  console.log(`wrote public/synch-table.bin: ${g.nx} x ${g.ns}, non-positive absorption cells ${zeroA}, ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
```

- [ ] **Step 4: Build the table** — `node scripts/build-synch-table.mjs` (~20 min; record the time and the
non-positive count in the ledger; measured while planning: 1205 s and 861 cells of 257,920: the frequencies below every
electron's lowest harmonic plus slow-cooling maser cells).

- [ ] **Step 5: Run / Commit** — `npx vitest run tests/cyclosynch.test.ts` → PASS; suite + build green;
`git add src/physics/cyclosynch.ts tests/cyclosynch.test.ts scripts/build-synch-table.mjs public/synch-table.bin && git commit -m "Cyclo-synchrotron coefficient table: parallel build, file format, lookup"`

---

### Task 4: Full-spectrum energy budget (CPU)

**Files:** Modify `src/physics/synchrotron.ts`, `tests/synchrotron.test.ts`, `src/render/uniforms.ts`,
`tests/uniforms.test.ts`, `src/render/raytrace.wgsl` (struct field rename only), `src/main.ts`, `src/test/scenes.ts`,
`src/test/shadow.browser.ts`, `scripts/bench.mjs`, `scripts/shot-delay.mjs`

**Interfaces:**
- Consumes: Task 3 `SynchTable`, `parseTable`, `lookup`, `I_G`.
- Produces (synchrotron.ts): `SIGMA_T`; `jetEnergetics(mSun, a, lambda)` now with `fH` (P_BZ includes f(Ω_H));
  `flowTime(z, g280, rgCm)` (s); `volumeWeight(r, th, a, g280)`; `injUnit(a, b0, rgCm, jetLength, g280)`;
  `LN_CJ, LN_CA, LN_NUB0, LN_K0, LN_T0`; `jetCoeffs(t, nuP, B, s, q0, shape): [j, alpha]`;
  `nuLnuAt(t, nu, a, b0, rgCm, jetLength, g280, q0)`; `ETA_DEFAULT`;
  `jetUniforms(mSun, a, lambda, eta, jetLength, g280): { jetB0; jetQ0; rgCm; pBZ }`.
  Removed: `SYN_P, GAMMA_MIN, lnGamma, avgSinPow, synchConsts, synchCoeffs, visLuminanceUnit, VIS_NU_LO/HI`.
- Uniform field `jetKScale` → `jetQ0` everywhere (same slot f[33]).

- [ ] **Step 1: Failing tests** — in `tests/synchrotron.test.ts` delete the `synchrotron coefficients` describe, the
`numeric`/`F`/`K53` helpers, the `energy budget` and `extremes` cases and the WGSL-constant describe (Task 5 writes
the new one); keep field/flow/shift/streamline/slab/band-matrix tests; fix the imports; add:

```ts
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseTable } from "../src/physics/cyclosynch";
import { jetEnergetics, jetUniforms, injUnit, nuLnuAt, flowTime, jetField, ETA_DEFAULT, C_CGS, LN_NUB0 } from "../src/physics/synchrotron";
import { TABLE_GRID } from "../src/physics/cyclosynch";
import { gammaProfile, JET } from "../src/physics/jet";
import { PRESETS } from "../src/physics/presets";
const T = parseTable(readFileSync(join(__dirname, "../public/synch-table.bin")).buffer.slice(0) as ArrayBuffer);

describe("full-spectrum energy budget (spec 2.4)", () => {
  it("P_BZ carries the high-spin correction f = 1 + 1.38 w^2 - 9.2 w^4; M87* stays in 1e43-1e44", () => {
    expect(jetEnergetics(6.5e9, 0.9, 1e-5).fH).toBeCloseTo(1.0468, 3);
    expect(jetEnergetics(21.2, 0.998, 0.02).fH).toBeCloseTo(0.8576, 3);
    const m = PRESETS.find((p) => p.id === "m87")!, E = jetEnergetics(m.massSun, m.a, m.lambda);
    expect(E.pBZ).toBeGreaterThan(1e43); expect(E.pBZ).toBeLessThan(1e44); // measured 4.46e43
  });
  it("injected power at infinity equals eta P_BZ", () => {
    for (const [m, a, l, eta] of [[6.5e9, 0.9, 8e-6, 0.05], [21.2, 0.998, 0.02, 1e-3], [1e8, 0.5, 3e-4, 1]]) {
      const U = jetUniforms(m, a, l, eta, 60, 2);
      expect((U.jetQ0 * injUnit(a, U.jetB0, U.rgCm, 60, 2)) / (eta * U.pBZ)).toBeCloseTo(1, 9);
    }
  });
  it("the default eta reproduces M87's optical nucleus: nu L_nu(550 nm) = 1e41 erg/s within 1 %", () => {
    const m = PRESETS.find((p) => p.id === "m87")!, U = jetUniforms(m.massSun, m.a, m.lambda, ETA_DEFAULT, 60, 2);
    const L = nuLnuAt(T, C_CGS / 550e-7, m.a, U.jetB0, U.rgCm, 60, 2, U.jetQ0);
    expect(Math.abs(L / 1e41 - 1)).toBeLessThan(0.01);
  });
  it("flow time is the integral of dz / (Gamma beta c)", () => {
    const rg = 1e15, z = 37; let s = 0; const N = 20000;
    for (let i = 0; i < N; i++) { const zz = 2 + (z - 2) * (i + 0.5) / N, G = gammaProfile(zz, 2.5); s += (z - 2) / N / Math.sqrt(G * G - 1); }
    expect(flowTime(z, 2.5, rg) / (s * rg / C_CGS)).toBeCloseTo(1, 5);
  });
  it("slider corners: q0 finite and >= 0 (0 at spin 0); the strongest field keeps x inside the table (Review Focus 1, 2)", () => {
    for (const [m, a, l, eta, g] of [[1, 0, 1e-10, 1e-4, 1.5], [1e10, 0.998, 1, 1, 8], [1, 0.998, 1, 1, 1.5], [1e10, 0.001, 1e-10, 1e-4, 8]]) {
      const U = jetUniforms(m, a, l, eta, 60, g);
      expect(Number.isFinite(U.jetQ0)).toBe(true); expect(U.jetQ0).toBeGreaterThanOrEqual(0);
      if (a === 0) expect(U.jetQ0).toBe(0);
      // the reddest plasma-frame frequency (650 nm, D = 0.1) at the strongest field (the funnel base)
      const B = jetField(0.5, JET.zBase + 0.05, a, U.jetB0), lnx = Math.log(0.1 * C_CGS / 650e-7) - LN_NUB0 - Math.log(B);
      expect(lnx).toBeGreaterThan(TABLE_GRID.lnx0);
      expect(Number.isFinite(Math.log(flowTime(30, g, U.rgCm)))).toBe(true);
    }
  });
});
```
Run → FAIL.

- [ ] **Step 2: Implement** — `synchrotron.ts`: delete the closed-form coefficient block (`SYN_P` … `synchCoeffs`),
`VIS_NU_*`, `visLuminanceUnit`, the old `jetUniforms`; change the header comment to cite this spec; add
`import { metricLower } from "./kerr";` and `import { I_G, lookup, type SynchTable } from "./cyclosynch";`, then:

```ts
export const SIGMA_T = 6.6524587e-25;
/** r_g, Mdot, horizon flux, Blandford-Znajek power with the high-spin correction f(Omega_H) (Tchekhovskoy,
 *  Narayan & McKinney 2010), b0 = Phi / (pi r_g^2). */
export function jetEnergetics(mSun: number, a: number, lambda: number) {
  const rgCm = (G_CGS * mSun * MSUN_G) / C_CGS ** 2, mdot = mdotFromLambda(mSun, a, lambda) * 1e3;
  const phi = PHI_MAD * Math.sqrt(mdot * C_CGS) * rgCm, w = a / (2 * horizonOuter(a)); // Omega_H r_g / c
  const fH = 1 + 1.38 * w * w - 9.2 * w ** 4;
  const pBZ = (KAPPA_BZ / (4 * Math.PI * C_CGS)) * phi * phi * (w * C_CGS / rgCm) ** 2 * fH;
  return { rgCm, mdot, phi, pBZ, b0: phi / (Math.PI * rgCm * rgCm), fH };
}
/** Plasma proper time (s) since leaving the base z_b: Int dz r_g / (c Gamma beta), Gamma beta = U0 (|z|/280)^0.58. */
export function flowTime(z: number, g280: number, rgCm: number): number {
  const U0 = Math.sqrt(Math.max(1e-12, g280 * g280 - 1)), az = Math.max(Math.abs(z), JET.zBase);
  return (rgCm / C_CGS) * Math.pow(280, 0.58) / U0 * (Math.pow(az, 0.42) - Math.pow(JET.zBase, 0.42)) / 0.42;
}
/** Energy at infinity per flat volume element: lapse^2 Gamma sqrt(g_rr g_thth g_phph) / (r^2 sin th) (spec 2.4). */
export function volumeWeight(r: number, th: number, a: number, g280: number): number {
  const gl = metricLower(r, th, a), lapse2 = -1 / metricUpper(r, th, a).tt;
  return lapse2 * gammaProfile(r * Math.cos(th), g280) * Math.sqrt(gl.rr * gl.thth * gl.phph) / (r * r * Math.sin(th));
}
/** Sum of f(rho, z) weighted by volumeWeight over the rendered region (both lobes), in r_g^3. */
function regionSum(a: number, jetLength: number, g280: number, f: (rho: number, z: number) => number): number {
  const NZ = 160, NR = 48, zb = JET.zBase; let L = 0;
  for (let iz = 0; iz < NZ; iz++) {
    const dz = (jetLength - zb) / NZ, z = zb + dz * (iz + 0.5), dr = (1.2 * funnelEdge(z)) / NR;
    for (let ir = 0; ir < NR; ir++) {
      const rho = dr * (ir + 0.5), v = f(rho, z); if (v === 0) continue;
      L += 2 * 2 * Math.PI * rho * dr * dz * volumeWeight(Math.hypot(rho, z), Math.atan2(rho, z), a, g280) * v;
    }
  }
  return L;
}
/** Injected power at infinity for q0 = 1 (knots and turbulence are mean-one, so they drop out). */
export function injUnit(a: number, b0: number, rgCm: number, jetLength: number, g280: number): number {
  return regionSum(a, jetLength, g280, (rho, z) => { const sh = wallProfile(rho, z) * lengthFalloff(z, jetLength);
    if (sh <= 0) return 0; const B = jetField(rho, z, a, b0); return B * B * sh; }) * rgCm ** 3;
}
// j' = C_J q0 shape B J^(x, s), alpha' = C_A q0 shape J^... (spec 2.3): logs of the prefactors, nu_B / B, k / B^2,
// and of the flow-time constant 280^0.58 / (0.42 c).
export const LN_CJ = Math.log((3 * Q_E ** 3) / (2 * I_G * SIGMA_T * M_E * C_CGS ** 3));
export const LN_CA = Math.log((3 * Math.PI ** 2 * Q_E) / (C_CGS * I_G * SIGMA_T));
export const LN_NUB0 = Math.log(Q_E / (2 * Math.PI * M_E * C_CGS));
export const LN_K0 = Math.log(SIGMA_T / (6 * Math.PI * M_E * C_CGS));
export const LN_T0 = 0.58 * Math.log(280) - Math.log(C_CGS) - Math.log(0.42);
/** Plasma-frame j' (erg s^-1 cm^-3 Hz^-1 sr^-1) and alpha' (cm^-1) at nu' (twin of synchSampleJ). */
export function jetCoeffs(t: SynchTable, nuP: number, B: number, s: number, q0: number, shape: number): [number, number] {
  const lnB = Math.log(B), lnx = Math.log(nuP) - LN_NUB0 - lnB, [lJ, lA] = lookup(t, lnx, Math.max(Math.log(s), -30));
  const base = Math.log(q0) + Math.log(shape);
  return [Math.exp(LN_CJ + base + lnB + lJ), Math.exp(LN_CA + base - lnB - 2 * lnx + lA)];
}
/** Comoving, optically thin nu L_nu (energy at infinity) of the rendered jet at q0. */
export function nuLnuAt(t: SynchTable, nu: number, a: number, b0: number, rgCm: number, jetLength: number, g280: number, q0: number): number {
  return 4 * Math.PI * nu * rgCm ** 3 * regionSum(a, jetLength, g280, (rho, z) => {
    const sh = wallProfile(rho, z) * lengthFalloff(z, jetLength); if (sh <= 0) return 0;
    const B = jetField(rho, z, a, b0), s = Math.exp(LN_K0) * B * B * flowTime(z, g280, rgCm);
    return jetCoeffs(t, nu, B, s, q0, sh)[0];
  });
}
/** Default eta: M87*'s nu L_nu(550 nm) = 1e41 erg/s (spec 2.4; the anchor test re-derives it). */
export const ETA_DEFAULT = 0.05527;
/** The jet's uniforms: q0 = eta P_BZ / (injected power at q0 = 1); 0 at spin 0 (no BZ power). */
export function jetUniforms(mSun: number, a: number, lambda: number, eta: number, jetLength: number, g280: number) {
  const E = jetEnergetics(mSun, a, lambda);
  return { jetB0: E.b0, jetQ0: E.pBZ > 0 ? (eta * E.pBZ) / injUnit(a, E.b0, E.rgCm, jetLength, g280) : 0, rgCm: E.rgCm, pBZ: E.pBZ };
}
```
(also add `gammaProfile` to the `./jet` import.)

- [ ] **Step 3: ETA_DEFAULT** — 0.05527 was solved while planning (M87\* preset, 1612 × 160 table: νL_ν(550 nm) at
η = 1 is 1.809e42 erg/s, so η = 1e41 / 1.809e42 = 0.05527). The anchor test re-derives it; if it fails after a table
rebuild, print `1e41 / nuLnuAt(..., q0 at η = 1)` and update the constant (ledger the new value).
Measured with it (thin νL_ν(550 nm) / P_BZ): M87\* 2.24e-3, Default 5.47e-3, Cygnus X-1 4.0e-4, GRS 1915+105 1.0e-4;
q₀: M87\* 1.805e-9, Default 1.173e-7, Cygnus X-1 0.869, GRS 1915+105 1.364.

- [ ] **Step 4: Callers** — `jetUniforms(...)` gains `g280` and returns `jetQ0`:
  - `uniforms.ts`: field `jetKScale` → `jetQ0` (comment: "energy-budget injection scale q0, erg s^-1 cm^-3 G^-2");
    `tests/uniforms.test.ts` likewise; `raytrace.wgsl` struct field `jetKScale` → `jetQ0` and its two uses
    (`if (U.jetQ0 <= 0.0)`, the `synchSampleJ` argument — Task 5 replaces that call).
  - `scenes.ts`, `bench.mjs`: `jetUniforms(CUSTOM_DEFAULT.massSun, a, CUSTOM_DEFAULT.lambda, ETA_DEFAULT, 60, 2)`.
  - `shadow.browser.ts`, `shot-delay.mjs`: `jetQ0: 0`.
  - `main.ts`: `state.jetEff` → `state.jetEta = ETA_DEFAULT`; `refreshJet` key and call include `state.jetGamma`;
    the Jet speed handler calls `physicsChanged()` (the budget depends on Γ through the volume weight).
    The ε slider keeps working until Task 7 (its handler sets `state.jetEta`).

- [ ] **Step 5: Run / Commit** — `npm test`, `npm run build` green;
`git add -A src tests scripts && git commit -m "Full-spectrum energy budget: eta x P_BZ (high-spin corrected), curved-space volume, M87 anchor"`

---

### Task 5: Table texture and shared WGSL

**Files:** Modify `src/render/emission-shared.wgsl`, `src/render/raytrace.wgsl`, `src/render/jet-parity.wgsl`,
`src/render/gpu.ts`, `src/test/parity.browser.ts`, `tests/synchrotron.test.ts`

**Interfaces:**
- Consumes: Task 3 table file and `lookup`; Task 4 `LN_*`, `jetCoeffs`, `flowTime`.
- Produces (WGSL): `@group(0) @binding(10) var synchTab: texture_2d<f32>` (declared in emission-shared;
  layout-"auto" consumers that never call the jet functions ignore it); consts `SYN_LNCJ, SYN_LNCA, SYN_LNNUB0,
  SYN_LNK0, SYN_LNT0, SYN_LNX0, SYN_LNX1, SYN_LNS0, SYN_LNS1`; `synchReady() -> bool`;
  `synchLookupJ(lnx, lns) -> vec2<f32>`; `synchSampleJ(r, th, D, a, b0, q0, shape, g280, rgCm) -> SynchOut`.
- Produces (gpu.ts): `Renderer.synchTex`; `init` fetches `/synch-table.bin` (a 1×1 placeholder until then or on failure).

- [ ] **Step 1: WGSL constant twin test (failing)** — append to `tests/synchrotron.test.ts`:

```ts
import { LN_CJ, LN_CA, LN_NUB0, LN_K0, LN_T0 } from "../src/physics/synchrotron";
import { TABLE_GRID } from "../src/physics/cyclosynch";
const WGSL = readFileSync(join(__dirname, "../src/render/emission-shared.wgsl"), "utf8");
const constOf = (name: string) => { const m = WGSL.match(new RegExp(`const ${name}\\s*=\\s*([^;]+);`)); if (!m) throw new Error(`no ${name}`); return +m[1]; };
describe("emission-shared.wgsl constants match the CPU twin", () => {
  it("coefficient prefactors, cyclotron and cooling scales, table grid", () => {
    for (const [n, v] of [["SYN_LNCJ", LN_CJ], ["SYN_LNCA", LN_CA], ["SYN_LNNUB0", LN_NUB0], ["SYN_LNK0", LN_K0], ["SYN_LNT0", LN_T0],
      ["SYN_LNX0", TABLE_GRID.lnx0], ["SYN_LNX1", TABLE_GRID.lnx1], ["SYN_LNS0", TABLE_GRID.lns0], ["SYN_LNS1", TABLE_GRID.lns1]] as [string, number][])
      expect(constOf(n)).toBeCloseTo(v, 6);
  });
});
```
Run → FAIL (`no SYN_LNNUB0`).

- [ ] **Step 2: WGSL** — in `emission-shared.wgsl` replace the block from `const SYN_P` through the end of
`synchSampleJ` (keep `JET_LNNU`, `JET_BAND_M`, `GAMMA_REF_Z`, `GAMMA_SLOPE`, `jetGammaAt`, `streamlineDirJ`,
`plasmaShiftJ`, `jetShapeJ`, `SynchOut`, `JetOut`, `jetSlabJ`) with:

```wgsl
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
const SYN_LNS0 = -6.907755279; const SYN_LNS1 = 3.688879454;
fn synchReady() -> bool { return textureDimensions(synchTab).x >= 2u; } // 1x1 placeholder: table not loaded
// Twin of lookup() in cyclosynch.ts: bilinear between cell centres, with the same edge extensions.
fn synchLookupJ(lnx: f32, lns: f32) -> vec2<f32> {
  let dims = textureDimensions(synchTab); let nx = f32(dims.x); let ns = f32(dims.y);
  let h = (SYN_LNX1 - SYN_LNX0) / nx;
  var fx = (lnx - SYN_LNX0) / h - 0.5; var extX = 0.0;
  if (fx > nx - 1.0) { extX = (fx - (nx - 1.0)) * h; fx = nx - 1.0; }
  fx = max(fx, 0.0);
  var fs = (lns - SYN_LNS0) / (SYN_LNS1 - SYN_LNS0) * (ns - 1.0); var extS = 0.0;
  if (fs < 0.0) { extS = lns - SYN_LNS0; fs = 0.0; }
  fs = min(fs, ns - 1.0);
  let ix = min(i32(dims.x) - 2, i32(floor(fx))); let iy = min(i32(dims.y) - 2, i32(floor(fs)));
  let ax = fx - f32(ix); let ay = fs - f32(iy);
  let v00 = textureLoad(synchTab, vec2<i32>(ix, iy), 0).xy;
  let v10 = textureLoad(synchTab, vec2<i32>(ix + 1, iy), 0).xy;
  let v01 = textureLoad(synchTab, vec2<i32>(ix, iy + 1), 0).xy;
  let v11 = textureLoad(synchTab, vec2<i32>(ix + 1, iy + 1), 0).xy;
  return mix(mix(v00, v10, ax), mix(v01, v11, ax), ay) + vec2<f32>(-1.5 * extX + extS, -2.0 * extX + extS);
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
```
(Executor: print `LN_*` with `npx vitest` / node to ≥ 10 digits and confirm these literals; the Step 1 test is the gate.)

- [ ] **Step 3: Renderer call sites** — `raytrace.wgsl` `jetStep`: guard `if (U.jetQ0 <= 0.0 || !synchReady()) { return accIn; }`
and call `synchSampleJ(q.x, q.y, D, U.a, U.jetB0, U.jetQ0, shape, U.jetGamma, U.rgCm)`.
`gpu.ts`: in the compute layout add `{ binding: 10, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } }`;
a field `synchTex: GPUTexture` created in `init` as a 1×1 `rg32float` placeholder; after the device is ready,
```ts
    try {
      const res = await fetch(new URL("/synch-table.bin", location.href));
      if (res.ok) this.uploadSynchTable(parseTable(await res.arrayBuffer()));
    } catch { /* keep the placeholder: synchReady() is false and the jet draws nothing */ }
```
with
```ts
  uploadSynchTable(t: SynchTable) {
    const tex = this.device.createTexture({ size: [t.nx, t.ns], format: "rg32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.device.queue.writeTexture({ texture: tex }, t.data, { bytesPerRow: t.nx * 8 }, [t.nx, t.ns]);
    this.synchTex?.destroy(); this.synchTex = tex;
  }
```
called before the first `rebind()`; `rebind()`'s common entries add `{ binding: 10, resource: this.synchTex.createView() }`.

- [ ] **Step 4: Parity** — `jet-parity.wgsl` (the table binding 10 comes from emission-shared): `JIn` gains
`k: vec4<f32>` = (b0, q0, rgCm, 0) per case, and the sample becomes
`let s = synchSampleJ(r, th, D, a, c.k.x, c.k.y, shape, c.c.y, c.k.z);` (delete `P_B0`, `P_KSCALE`). In
`parity.browser.ts` replace the jet block's physics with:

```ts
  const synchBuf = await (await fetch("/synch-table.bin")).arrayBuffer(), T = parseTable(synchBuf);
  const synchTex = device.createTexture({ size: [T.nx, T.ns], format: "rg32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: synchTex }, T.data, { bytesPerRow: T.nx * 8 }, [T.nx, T.ns]);
  // Two jets: an M87*-like one (b0 632 G) and Cygnus X-1's (b0 3.8e8 G: visible near the cyclotron frequency).
  const JETS = [{ b0: 632.39, rg: 9.5984e14, q0: 1.8049e-9 }, { b0: 3.7838e8, rg: 3.1305e6, q0: 0.86887 }]; // M87*, Cyg X-1 at ETA_DEFAULT
  // rays: alpha, beta, a, incl, g280, jet index
  const rays: [number, number, number, number, number, number][] = [
    [0.5, 4, 0.9, (17 * Math.PI) / 180, 2, 0], [1.5, 8, 0.9, (17 * Math.PI) / 180, 4, 0],
    [0.5, -4, 0.5, (60 * Math.PI) / 180, 2, 0], [1.5, 8, 0.5, (60 * Math.PI) / 180, 6, 0],
    [0.3, 6, 0.998, (27 * Math.PI) / 180, 2, 1], [0.8, -6, 0.998, (27 * Math.PI) / 180, 3, 1],
  ];
```
(walk each ray into the jet exactly as now; if a ray misses, swap its beta's sign, keeping both lobes covered).
Per case the CPU expectation is, with `J = JETS[jet]`: `B = jetField(rho, z, a, J.b0)`,
`s = Math.exp(LN_K0) * B * B * flowTime(z, g280, J.rg)`, and per band
`[jP, aP] = jetCoeffs(T, (C_CGS / (nm * 1e-7)) * D, B, s, J.q0, shape)`; outputs `ln(jP / D^3)`, `ln aP` and the two
slab steps with `ds` = 0.01 or 20 optical depths of the 550 nm `aP` (as now). Input layout: 16 floats per case
(x, p, c, k). The jet bind group adds `{ binding: 10, resource: synchTex.createView() }`. Error metric: D and shape
keep `|d| / (1 + |want|)`; every ln output and tau uses the absolute `|d|`, and the route FAILS if any exceeds 2e-3
(f32 logs of j ~ e^-50 carry ~4e-6 absolute; the old relative metric tolerated ~5 % in j). Report the largest log
error in the route's text next to maxRelErr.

- [ ] **Step 5: Run** — `npx vitest run` → PASS; `npm run build`; dev server; `npm run verify:gpu` → `?parity` PASS
(record the jet errors), `?golden` FAIL expected (jet scenes change; Task 6 records), everything else PASS.

- [ ] **Step 6: Commit** — `git add -A src tests && git commit -m "Shader samples the cyclo-synchrotron table; parity on the shipped bytes"`

---

### Task 6: Renderer gates and the optically thick cache scene

**Files:** Modify `src/test/scenes.ts`, `src/test/cachecheck.browser.ts`, `src/test/golden.json` (re-recorded), `README.md` (perf row only)

- [ ] **Step 1: Thick scene** — `scenes.ts`: `Scene` gains optional `obj?: { massSun: number; lambda: number }`;
`sceneUniforms` takes the jet's object from `s.obj ?? CUSTOM_DEFAULT` (`jetUniforms(obj.massSun, s.a, obj.lambda,
ETA_DEFAULT, 60, 2)`). `cachecheck.browser.ts` (its own scene list) appends
`{ name: "xrb-thick", a: 0.998, inclDeg: 27, time: 7, frame: 3, jetStrength: 1, skyStrength: 0, obj: { massSun: 21.2, lambda: 0.02 } }`
(Cygnus X-1's jet, optically thick over the disk). Golden keeps its five scenes.

- [ ] **Step 2: A/B and golden** — `npm run verify:gpu` without recording: if `jet-off` ≠ the current hash, run the
jet-free experiment (remove the `jetStep` call and use `raw = color` in this build and in `main`'s; compare accum
captures; bit-identical ⇒ compiler rescheduling ⇒ ledger the ruling). Then `RECORD_GOLDEN=1 npm run verify:gpu` →
all PASS incl. `?cachecheck` with `xrb-thick` (cached = live, max ≤ 4e-7).

- [ ] **Step 3: Probes and bench** — `node scripts/probe-axis.mjs`, `node scripts/probe-scale.mjs` PASS;
`npm run bench` before (on `main`) and after; record the jet-on live and cached frame times in the README table.

- [ ] **Step 4: Commit** — `git add src/test README.md && git commit -m "Gates for the cooled jet: golden re-recorded, optically thick cache scene, bench"`

---

### Task 7: Panel, presets, measurements

**Files:** Modify `index.html`, `src/main.ts`, `src/physics/presets.ts`, `tests/presets.test.ts`

- [ ] **Step 1: Captions test** — in `tests/presets.test.ts` replace the X-ray-binary caption test with:
```ts
  it("X-ray-binary captions describe the energy-conserving jet (no overestimate caveat)", () => {
    for (const id of ["cygx1", "grs1915"]) {
      const c = PRESETS.find((p) => p.id === id)!.caption;
      expect(c).not.toMatch(/overestimate/); expect(c).toMatch(/fraction η of its jet power/);
    }
  });
```
Run → FAIL. Captions (Cyg X-1 and GRS): replace the "Its jet uses the same energy budget, but here …
Its real brightness this close to the hole has not been measured." text with "Its jet's electrons radiate a fraction η of its jet power (the
value fixed by M87's optical nucleus); here the field puts visible light near the electrons' cyclotron frequency,
computed exactly. Its real brightness this close to the hole has not been measured." M87\*: replace "it radiates ε
(≈ 0.2 % by default, from M87's optical nucleus vs its jet power) of its Blandford–Znajek power as visible light"
with "its electrons radiate a fraction η of its Blandford–Znajek power, the η that reproduces M87's optical nucleus".
Tests PASS.

- [ ] **Step 2: Panel** — `index.html`: the ε control becomes
```html
    <div class="ctrl" title="Fraction of the jet's Blandford–Znajek power its electrons radiate (all frequencies)">
      <div class="row"><label>Jet efficiency&nbsp;η</label><span class="val"><b id="jeteffv">—</b></span></div>
      <input id="jeteff" type="range" min="-4" max="0" step="0.05">
    </div>
```
`main.ts`: `jeteff.value = String(Math.log10(ETA_DEFAULT))` at start-up; handler sets `state.jetEta = 10 ** +jeteff.value`;
`showEff` prints `state.jetEta.toExponential(1)`.

- [ ] **Step 3: Gates, screenshots, shares** — `npm run verify:gpu`, probes PASS; `node scripts/shot-presets.mjs` and look at
every image; measure jet ÷ disk light (jet-on minus jet-off accumulation, 640×360, sky off, 5 times) for Default,
M87\*, Cygnus X-1, GRS 1915+105 and record them (reported, not tuned).

- [ ] **Step 4: Commit** — `git add index.html src tests && git commit -m "Jet efficiency eta on the panel; presets describe the energy-conserving jet"`

---

### Task 8: Documentation

- [ ] README: Features jet bullet ("… cooled by its own radiation, exact cyclo-synchrotron coefficients, brightness
from η × the jet's power"); a Status paragraph **Cooled jet (feat/cooled-jet, 2026-10-02)**: the problem (visible-band
budget impossible for X-ray binaries), fast cooling everywhere (the γ_break numbers), the injection fit to M87's core SED
with sources, the exact cooling solution, the single-electron validation (Larmor 0.03 %, seam 1.3 %), the table
(grid, build time, calorimetry 4e-3, lookup accuracy), the energy weighting (α² Γ √γ), the P_BZ correction, the anchor
η, the measured shares, the gates; remove the "outside the energy-consistent range" sentence and the "follow-up" note.
- [ ] ROADMAP: under Physics features add "- [x] **Cooled jet** (2026-10-02): full-spectrum energy budget …".
- [ ] Spec status → "implemented on `feat/cooled-jet`".
- [ ] Commit `"Document the cooled jet"`.
