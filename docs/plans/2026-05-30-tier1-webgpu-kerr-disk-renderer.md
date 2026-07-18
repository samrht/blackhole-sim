# Tier-1 WebGPU Kerr Accretion-Disk Renderer — Implementation Plan

**Goal:** A browser app that renders a physically-correct, off-axis image of a Kerr black hole's accretion disk — shadow, photon ring, lensed far-side arc, Doppler beaming, gravitational redshift, and blackbody color — HDR-tonemapped with ACES and progressively anti-aliased.

**Architecture:** A pure-TypeScript physics core (Kerr metric, geodesic integrator, ISCO/orbits, redshift g-factor, Novikov–Thorne/Page–Thorne temperature, blackbody→sRGB color) is unit-tested with Vitest against the spec's verification gates. The core's results are baked into two 1-D lookup tables (T(r) and color(T)) uploaded as GPU textures. A WebGPU compute shader (`raytrace.wgsl`) mirrors the core's math, backward-integrates one null geodesic per pixel, samples the LUTs at the disk hit, and accumulates HDR radiance into a storage buffer; a present pass divides by the sample count, applies exposure + ACES, and writes to the canvas. A CPU↔GPU parity test guards the port.

**Tech Stack:** TypeScript, Vite (dev server/build), Vitest (unit tests), WebGPU (WGSL compute + render), Playwright (optional headless render check). Units: geometrized G=c=M=1 throughout the core and shader; spin `a` equals `a_*∈[0,1)`; coordinates `(t,r,θ,φ)` Boyer–Lindquist.

**Spec:** `docs/specs/2026-05-30-relativistic-blackhole-accretion-design.md`

**Out of scope (follow-on plans):** jets/synchrotron (BZ), MRI flares/hotspots & time evolution, lensed starfield background, bloom/glare, native C++/CUDA port, Tier-2/Tier-3.

---

## File Structure

```
Projects/blackhole-sim/
├─ package.json, tsconfig.json, vite.config.ts, vitest.config.ts
├─ index.html                      # canvas + control panel
├─ src/
│  ├─ main.ts                      # bootstrap: init renderer, wire controls
│  ├─ physics/
│  │  ├─ kerr.ts                   # metric (lower/upper), horizons, ergosphere, ZAMO, Ω_H
│  │  ├─ orbits.ts                 # Ω(r), Ẽ, L̃, ISCO, r_ph, r_mb, efficiency
│  │  ├─ geodesic.ts               # Hamiltonian RHS, RK4, conserved quantities
│  │  ├─ redshift.ts               # g-factor
│  │  ├─ disk.ts                   # Page–Thorne flux shape, normalized T(r)
│  │  ├─ color.ts                  # Planck→CIE(Wyman fit)→linear sRGB
│  │  └─ lookups.ts                # build T(r) and color(T) Float32 LUTs
│  ├─ render/
│  │  ├─ gpu.ts                    # Renderer class: device, buffers, pipelines, frame loop
│  │  ├─ uniforms.ts               # Uniforms struct <-> ArrayBuffer packing (shared layout)
│  │  ├─ raytrace.wgsl             # compute: per-pixel geodesic + emission + accumulate
│  │  ├─ present.wgsl              # fullscreen: resolve accum, exposure, ACES, gamma
│  │  └─ parity.wgsl              # compute: run core fns on test inputs -> readback buffer
│  └─ test/
│     └─ parity.browser.ts         # WebGPU parity harness (run in page/Playwright)
└─ tests/
   ├─ kerr.test.ts  orbits.test.ts  geodesic.test.ts
   ├─ redshift.test.ts  disk.test.ts  color.test.ts  lookups.test.ts
```

Each `physics/*.ts` file has one responsibility and no WebGPU/DOM dependency, so it runs under Vitest in Node. `render/*` is the only DOM/WebGPU code.

---

## Task 0: Scaffold project (Vite + TS + Vitest)

**Files:**
- Create: `package.json`, `tsconfig.json`, `vite.config.ts`, `vitest.config.ts`, `index.html`, `src/main.ts`, `tests/smoke.test.ts`, `.gitignore`

- [ ] **Step 1: Verify Node/npm are available**

Run: `node --version; npm --version`
Expected: both print versions (Node ≥ 18). If missing, install Node LTS before continuing.

- [ ] **Step 2: Create `package.json`**

```json
{
  "name": "blackhole-sim",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc --noEmit && vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "devDependencies": {
    "typescript": "^5.5.0",
    "vite": "^5.4.0",
    "vitest": "^2.0.0",
    "@webgpu/types": "^0.1.40"
  }
}
```

- [ ] **Step 3: Create config files**

`tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "noUnusedLocals": true,
    "types": ["@webgpu/types", "vitest/globals"],
    "lib": ["ES2022", "DOM", "DOM.Iterable"]
  },
  "include": ["src", "tests"]
}
```

`vite.config.ts`:
```ts
import { defineConfig } from "vite";
export default defineConfig({ server: { port: 5173 } });
```

`vitest.config.ts`:
```ts
import { defineConfig } from "vitest/config";
export default defineConfig({ test: { globals: true, environment: "node" } });
```

`.gitignore`:
```
node_modules/
dist/
*.log
```

- [ ] **Step 4: Create `index.html` and a placeholder `src/main.ts`**

`index.html`:
```html
<!doctype html>
<html lang="en">
<head><meta charset="utf-8" /><title>Kerr Black Hole</title>
<style>
  html,body{margin:0;height:100%;background:#000;color:#ccc;font:13px system-ui}
  #c{display:block;width:100vw;height:100vh}
  #panel{position:fixed;top:8px;left:8px;background:#0008;padding:10px;border-radius:8px}
  label{display:block;margin:4px 0}
</style></head>
<body>
  <canvas id="c"></canvas>
  <div id="panel">
    <label>spin a* <input id="spin" type="range" min="0" max="0.998" step="0.001" value="0.9"><span id="spinv">0.9</span></label>
    <label>inclination° <input id="incl" type="range" min="1" max="89" step="1" value="72"><span id="inclv">72</span></label>
    <label>exposure <input id="exp" type="range" min="-4" max="4" step="0.1" value="0"><span id="expv">0</span></label>
  </div>
  <script type="module" src="/src/main.ts"></script>
</body></html>
```

`src/main.ts`:
```ts
console.log("blackhole-sim boot");
```

- [ ] **Step 5: Write a smoke test**

`tests/smoke.test.ts`:
```ts
import { describe, it, expect } from "vitest";
describe("smoke", () => { it("runs", () => { expect(1 + 1).toBe(2); }); });
```

- [ ] **Step 6: Install and verify**

Run: `npm install`
Run: `npm test`
Expected: 1 passed.
Run: `npm run dev` then open `http://localhost:5173` — page loads, console prints `blackhole-sim boot`. Stop the server (Ctrl+C).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: scaffold Vite+TS+Vitest project for blackhole-sim"
```

---

## Task 1: Kerr metric core (`src/physics/kerr.ts`)

**Files:**
- Create: `src/physics/kerr.ts`
- Test: `tests/kerr.test.ts`

- [ ] **Step 1: Write the failing tests**

`tests/kerr.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import {
  sigma, delta, bigA, horizonOuter, ergosphere, omegaHorizon,
  metricLower, metricUpper, omegaZAMO,
} from "../src/physics/kerr";

const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol * (1 + Math.abs(b));

describe("kerr metric", () => {
  it("Schwarzschild horizon is 2M", () => { expect(close(horizonOuter(0), 2)).toBe(true); });
  it("extremal horizon is M", () => { expect(close(horizonOuter(1), 1)).toBe(true); });
  it("ergosphere at equator is 2M for any spin", () => {
    expect(close(ergosphere(Math.PI / 2, 0.9), 2)).toBe(true);
  });
  it("lower and upper metric are inverse in the (t,phi) block and diagonal", () => {
    const r = 8, th = 1.1, a = 0.7;
    const g = metricLower(r, th, a), gi = metricUpper(r, th, a);
    // (M·M^-1)_tt = g_tt g^tt + g_tφ g^φt = 1
    expect(close(g.tt * gi.tt + g.tphi * gi.tphi, 1)).toBe(true);
    // (M·M^-1)_φφ = g_φt g^tφ + g_φφ g^φφ = 1
    expect(close(g.tphi * gi.tphi + g.phph * gi.phph, 1)).toBe(true);
    // off-diagonal (M·M^-1)_tφ = g_tt g^tφ + g_tφ g^φφ = 0
    expect(close(g.tt * gi.tphi + g.tphi * gi.phph, 0, 1e-9)).toBe(true);
    expect(close(g.rr * gi.rr, 1)).toBe(true);
    expect(close(g.thth * gi.thth, 1)).toBe(true);
  });
  it("ZAMO omega -> Omega_H at the horizon", () => {
    const a = 0.6, rp = horizonOuter(a);
    expect(close(omegaZAMO(rp, Math.PI / 2, a), omegaHorizon(a), 1e-6)).toBe(true);
  });
  it("Sigma, Delta, A basic identities", () => {
    expect(close(sigma(5, Math.PI / 2, 0.5), 25)).toBe(true);       // r^2 at equator
    expect(close(delta(2, 0), 0)).toBe(true);                        // r_+=2M for a=0
    expect(bigA(6, 1.0, 0.5)).toBeGreaterThan(0);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/kerr.test.ts`
Expected: FAIL — cannot find module `../src/physics/kerr`.

- [ ] **Step 3: Implement `src/physics/kerr.ts`**

```ts
export const M = 1;

export interface Metric { tt: number; tphi: number; rr: number; thth: number; phph: number; }

export function sigma(r: number, theta: number, a: number): number {
  const c = Math.cos(theta);
  return r * r + a * a * c * c;
}
export function delta(r: number, a: number): number {
  return r * r - 2 * M * r + a * a;
}
export function bigA(r: number, theta: number, a: number): number {
  const s = Math.sin(theta);
  return (r * r + a * a) ** 2 - a * a * delta(r, a) * s * s;
}
export function horizonOuter(a: number): number {
  return M + Math.sqrt(Math.max(0, M * M - a * a));
}
export function ergosphere(theta: number, a: number): number {
  const c = Math.cos(theta);
  return M + Math.sqrt(Math.max(0, M * M - a * a * c * c));
}
export function omegaHorizon(a: number): number {
  const rp = horizonOuter(a);
  return a / (rp * rp + a * a);
}
export function metricLower(r: number, theta: number, a: number): Metric {
  const s2 = Math.sin(theta) ** 2;
  const Sig = sigma(r, theta, a), d = delta(r, a);
  return {
    tt: -(1 - 2 * M * r / Sig),
    tphi: -2 * M * a * r * s2 / Sig,
    rr: Sig / d,
    thth: Sig,
    phph: (r * r + a * a + 2 * M * a * a * r * s2 / Sig) * s2,
  };
}
export function metricUpper(r: number, theta: number, a: number): Metric {
  const s2 = Math.sin(theta) ** 2;
  const Sig = sigma(r, theta, a), d = delta(r, a), A = bigA(r, theta, a);
  return {
    tt: -A / (Sig * d),
    tphi: -2 * M * a * r / (Sig * d),
    rr: d / Sig,
    thth: 1 / Sig,
    phph: (d - a * a * s2) / (Sig * d * s2),
  };
}
export function omegaZAMO(r: number, theta: number, a: number): number {
  return 2 * M * a * r / bigA(r, theta, a);
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npx vitest run tests/kerr.test.ts`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/physics/kerr.ts tests/kerr.test.ts
git commit -m "feat(physics): Kerr metric, horizons, ergosphere, ZAMO"
```

---

## Task 2: Orbits & ISCO (`src/physics/orbits.ts`)

**Files:**
- Create: `src/physics/orbits.ts`
- Test: `tests/orbits.test.ts`

- [ ] **Step 1: Write the failing tests**

`tests/orbits.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { omegaKepler, circE, circL, iscoRadius, photonOrbit, marginallyBound, efficiency } from "../src/physics/orbits";
const close = (a: number, b: number, tol = 1e-4) => Math.abs(a - b) <= tol * (1 + Math.abs(b));

describe("orbits", () => {
  it("ISCO: 6M (a=0), ->M (a=1 pro), 9M (a=1 retro)", () => {
    expect(close(iscoRadius(0), 6)).toBe(true);
    expect(close(iscoRadius(1, true), 1, 2e-3)).toBe(true);
    expect(close(iscoRadius(1, false), 9)).toBe(true);
  });
  it("photon orbit: 3M (a=0), M (a=1 pro), 4M (a=1 retro)", () => {
    expect(close(photonOrbit(0), 3)).toBe(true);
    expect(close(photonOrbit(1, true), 1, 2e-3)).toBe(true);
    expect(close(photonOrbit(1, false), 4)).toBe(true);
  });
  it("marginally bound: 4M (a=0)", () => { expect(close(marginallyBound(0), 4)).toBe(true); });
  it("efficiency: ~0.057 (a=0), ~0.42 (a=1 pro)", () => {
    expect(close(efficiency(0), 0.0572, 2e-3)).toBe(true);
    expect(efficiency(0.998, true)).toBeGreaterThan(0.3);
  });
  it("Keplerian Omega -> sqrt(1/r^3) far out", () => {
    expect(close(omegaKepler(1000, 0), Math.sqrt(1 / 1e9), 1e-2)).toBe(true);
  });
  it("circular E,L finite at r=6M (a=0): E=sqrt(8/9), L=2sqrt(3)", () => {
    expect(close(circE(6, 0), Math.sqrt(8 / 9))).toBe(true);
    expect(close(circL(6, 0), 2 * Math.sqrt(3))).toBe(true);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/orbits.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `src/physics/orbits.ts`**

```ts
import { M } from "./kerr";
// prograde uses upper signs; s = +1 prograde, -1 retrograde.

export function omegaKepler(r: number, a: number, prograde = true): number {
  const s = prograde ? 1 : -1;
  return s * Math.sqrt(M) / (Math.pow(r, 1.5) + s * a * Math.sqrt(M));
}
export function circE(r: number, a: number, prograde = true): number {
  const s = prograde ? 1 : -1, rt = Math.sqrt(r), sM = Math.sqrt(M);
  const num = r * rt - 2 * M * rt + s * a * sM;
  const den = Math.pow(r, 0.75) * Math.sqrt(r * rt - 3 * M * rt + 2 * s * a * sM);
  return num / den;
}
export function circL(r: number, a: number, prograde = true): number {
  const s = prograde ? 1 : -1, rt = Math.sqrt(r), sM = Math.sqrt(M);
  const den = Math.pow(r, 0.75) * Math.sqrt(r * rt - 3 * M * rt + 2 * s * a * sM);
  return s * sM * (r * r - s * 2 * a * sM * rt + a * a) / den;
}
export function iscoRadius(a: number, prograde = true): number {
  const as = a / M;
  const Z1 = 1 + Math.cbrt(1 - as * as) * (Math.cbrt(1 + as) + Math.cbrt(1 - as));
  const Z2 = Math.sqrt(3 * as * as + Z1 * Z1);
  const s = prograde ? -1 : 1;
  return M * (3 + Z2 + s * Math.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2)));
}
export function photonOrbit(a: number, prograde = true): number {
  const as = a / M, sgn = prograde ? -1 : 1;
  return 2 * M * (1 + Math.cos((2 / 3) * Math.acos(sgn * as)));
}
export function marginallyBound(a: number, prograde = true): number {
  const s = prograde ? -1 : 1;
  return 2 * M + s * a + 2 * Math.sqrt(M) * Math.sqrt(M + s * a);
}
export function efficiency(a: number, prograde = true): number {
  return 1 - circE(iscoRadius(a, prograde), a, prograde);
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npx vitest run tests/orbits.test.ts`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/physics/orbits.ts tests/orbits.test.ts
git commit -m "feat(physics): circular orbits, ISCO, photon/mb radii, efficiency"
```

---

## Task 3: Geodesic integrator (`src/physics/geodesic.ts`)

**Files:**
- Create: `src/physics/geodesic.ts`
- Test: `tests/geodesic.test.ts`

State vector is `Float64Array` of length 8: `[t, r, θ, φ, p_t, p_r, p_θ, p_φ]`.

- [ ] **Step 1: Write the failing tests**

`tests/geodesic.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { rhs, rk4, conserved, nullRadialMomentum } from "../src/physics/geodesic";

function makeEquatorialPhoton(r0: number, b: number, a: number): Float64Array {
  // E=1, L_z=b, equatorial (θ=π/2, p_θ=0), inward p_r<0
  const th = Math.PI / 2, pt = -1, pphi = b, pth = 0;
  const pr = -Math.abs(nullRadialMomentum(r0, th, a, pt, pphi, pth));
  return new Float64Array([0, r0, th, 0, pt, pr, pth, pphi]);
}

describe("geodesic integrator", () => {
  it("conserves E, L_z, Q, H along a photon trajectory", () => {
    const a = 0.8, s = makeEquatorialPhoton(20, 4.5, a);
    const c0 = conserved(s, a);
    let st = s;
    for (let i = 0; i < 2000; i++) st = rk4(st, a, -0.02);
    const c1 = conserved(st, a);
    expect(Math.abs(c1.E - c0.E)).toBeLessThan(1e-6);
    expect(Math.abs(c1.Lz - c0.Lz)).toBeLessThan(1e-6);
    expect(Math.abs(c1.Q - c0.Q)).toBeLessThan(1e-4);
    expect(Math.abs(c1.H)).toBeLessThan(1e-4); // null: H≈0
  });
  it("Schwarzschild capture threshold near b_c=sqrt(27)≈5.196", () => {
    const a = 0, rp = 2;
    const fate = (b: number) => {
      let st = makeEquatorialPhoton(50, b, a);
      for (let i = 0; i < 20000; i++) {
        st = rk4(st, a, -0.02);
        if (st[1] <= rp * 1.0001) return "captured";
        if (st[1] > 60) return "escaped";
      }
      return "stuck";
    };
    expect(fate(5.0)).toBe("captured");
    expect(fate(5.4)).toBe("escaped");
  });
  it("rhs keeps p_t and p_φ constant (Killing)", () => {
    const a = 0.5, s = makeEquatorialPhoton(15, 3, a);
    const d = rhs(s, a);
    expect(Math.abs(d[4])).toBeLessThan(1e-12); // dp_t/dλ
    expect(Math.abs(d[7])).toBeLessThan(1e-12); // dp_φ/dλ
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/geodesic.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `src/physics/geodesic.ts`**

```ts
import { metricUpper } from "./kerr";

function hquad(r: number, th: number, a: number, pt: number, pr: number, pth: number, pphi: number): number {
  const g = metricUpper(r, th, a);
  return g.tt * pt * pt + 2 * g.tphi * pt * pphi + g.rr * pr * pr + g.thth * pth * pth + g.phph * pphi * pphi;
}

export function nullRadialMomentum(r: number, th: number, a: number, pt: number, pphi: number, pth: number): number {
  // solve g^{rr} p_r^2 = -(g^tt pt^2 + 2 g^tφ pt pφ + g^θθ pθ^2 + g^φφ pφ^2)
  const g = metricUpper(r, th, a);
  const rest = g.tt * pt * pt + 2 * g.tphi * pt * pphi + g.thth * pth * pth + g.phph * pphi * pphi;
  return Math.sqrt(Math.max(0, -rest / g.rr));
}

export function rhs(s: Float64Array, a: number): Float64Array {
  const [, r, th, , pt, pr, pth, pphi] = s;
  const g = metricUpper(r, th, a);
  const dt = g.tt * pt + g.tphi * pphi;
  const dr = g.rr * pr;
  const dth = g.thth * pth;
  const dphi = g.tphi * pt + g.phph * pphi;
  const h = 1e-5;
  const dQdr = (hquad(r + h, th, a, pt, pr, pth, pphi) - hquad(r - h, th, a, pt, pr, pth, pphi)) / (2 * h);
  const dQdth = (hquad(r, th + h, a, pt, pr, pth, pphi) - hquad(r, th - h, a, pt, pr, pth, pphi)) / (2 * h);
  return new Float64Array([dt, dr, dth, dphi, 0, -0.5 * dQdr, -0.5 * dQdth, 0]);
}

export function rk4(s: Float64Array, a: number, dl: number): Float64Array {
  const add = (x: Float64Array, k: Float64Array, f: number) => {
    const o = new Float64Array(8);
    for (let i = 0; i < 8; i++) o[i] = x[i] + k[i] * f;
    return o;
  };
  const k1 = rhs(s, a);
  const k2 = rhs(add(s, k1, dl / 2), a);
  const k3 = rhs(add(s, k2, dl / 2), a);
  const k4 = rhs(add(s, k3, dl), a);
  const o = new Float64Array(8);
  for (let i = 0; i < 8; i++) o[i] = s[i] + (dl / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
  return o;
}

export function conserved(s: Float64Array, a: number, mu = 0) {
  const [, r, th, , pt, pr, pth, pphi] = s;
  const E = -pt, Lz = pphi;
  const c = Math.cos(th), sn = Math.sin(th);
  const Q = pth * pth + c * c * (a * a * (mu * mu - E * E) + (Lz * Lz) / (sn * sn));
  const H = 0.5 * hquad(r, th, a, pt, pr, pth, pphi);
  return { E, Lz, Q, H };
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npx vitest run tests/geodesic.test.ts`
Expected: PASS (3 tests). (The capture test runs up to 20k steps × 3 calls; it completes in a few seconds.)

- [ ] **Step 5: Commit**

```bash
git add src/physics/geodesic.ts tests/geodesic.test.ts
git commit -m "feat(physics): Hamiltonian geodesic RHS + RK4 + conserved quantities"
```

---

## Task 4: Redshift g-factor (`src/physics/redshift.ts`)

**Files:**
- Create: `src/physics/redshift.ts`
- Test: `tests/redshift.test.ts`

- [ ] **Step 1: Write the failing tests**

`tests/redshift.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { gFactor, gFactorKepler } from "../src/physics/redshift";
const close = (a: number, b: number, tol = 1e-6) => Math.abs(a - b) <= tol * (1 + Math.abs(b));

describe("redshift g-factor", () => {
  it("static emitter in Schwarzschild gives g = sqrt(1-2M/r)", () => {
    // Ω=0, ξ=0 -> g = sqrt(-g_tt) = sqrt(1-2/r)
    expect(close(gFactor(10, 0, 0, 0), Math.sqrt(1 - 2 / 10))).toBe(true);
    expect(close(gFactor(4, 0, 0, 0), Math.sqrt(1 - 2 / 4))).toBe(true);
  });
  it("orbiting disk: approaching side (ξ>0) brighter than receding (ξ<0)", () => {
    const r = 10, a = 0.5, b = 3;
    expect(gFactorKepler(r, a, +b)).toBeGreaterThan(gFactorKepler(r, a, -b));
  });
  it("g-factor is positive and finite in the emitting region", () => {
    expect(gFactorKepler(8, 0.9, 2)).toBeGreaterThan(0);
    expect(Number.isFinite(gFactorKepler(8, 0.9, 2))).toBe(true);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/redshift.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `src/physics/redshift.ts`**

```ts
import { metricLower } from "./kerr";
import { omegaKepler } from "./orbits";

/** General g = sqrt(-(g_tt + 2Ω g_tφ + Ω² g_φφ)) / (1 - Ω ξ), emitter at equator, observer at infinity. */
export function gFactor(r: number, a: number, xi: number, Omega: number): number {
  const g = metricLower(r, Math.PI / 2, a);
  const rad = -(g.tt + 2 * Omega * g.tphi + Omega * Omega * g.phph);
  return Math.sqrt(Math.max(0, rad)) / (1 - Omega * xi);
}
/** Convenience for a Keplerian disk element. */
export function gFactorKepler(r: number, a: number, xi: number, prograde = true): number {
  return gFactor(r, a, xi, omegaKepler(r, a, prograde));
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npx vitest run tests/redshift.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/physics/redshift.ts tests/redshift.test.ts
git commit -m "feat(physics): combined Doppler+gravitational redshift g-factor"
```

---

## Task 5: Disk temperature profile (`src/physics/disk.ts`)

**Files:**
- Create: `src/physics/disk.ts`
- Test: `tests/disk.test.ts`

- [ ] **Step 1: Write the failing tests**

`tests/disk.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { pageThorneFluxShape, temperatureShape } from "../src/physics/disk";
import { iscoRadius } from "../src/physics/orbits";

describe("Novikov-Thorne / Page-Thorne disk", () => {
  it("flux vanishes at the ISCO (zero-torque inner boundary)", () => {
    const a = 0, ri = iscoRadius(a);
    expect(temperatureShape(ri * 1.0001, a)).toBeLessThan(0.05);
  });
  it("temperature peaks just OUTSIDE the ISCO, not at it", () => {
    const a = 0, ri = iscoRadius(a);
    const tAtEdge = temperatureShape(ri * 1.001, a);
    const tPeakish = temperatureShape(ri * 1.5, a);
    expect(tPeakish).toBeGreaterThan(tAtEdge);
  });
  it("far-field flux ~ r^-3 => T ~ r^-3/4", () => {
    const a = 0;
    const ratio = temperatureShape(400, a) / temperatureShape(100, a);
    expect(ratio).toBeCloseTo(Math.pow(4, -0.75), 1); // ≈0.354
  });
  it("flux is positive in the emitting region", () => {
    expect(pageThorneFluxShape(20, 0.9)).toBeGreaterThan(0);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/disk.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `src/physics/disk.ts`**

```ts
import { iscoRadius, omegaKepler, circE, circL } from "./orbits";

const deriv = (f: (x: number) => number, x: number, h = 1e-5) => (f(x + h) - f(x - h)) / (2 * h);

/** Dimensionless Page-Thorne flux shape F(r) (Ṁ=M=1, √-g=r). Returns 0 inside ISCO. */
export function pageThorneFluxShape(r: number, a: number, prograde = true): number {
  const ri = iscoRadius(a, prograde);
  if (r <= ri) return 0;
  const E = (x: number) => circE(x, a, prograde);
  const L = (x: number) => circL(x, a, prograde);
  const Om = (x: number) => omegaKepler(x, a, prograde);
  const integrand = (x: number) => (E(x) - Om(x) * L(x)) * deriv(L, x);
  // Composite Simpson over [ri, r]
  const n = 400, h = (r - ri) / n;
  let I = integrand(ri) + integrand(r);
  for (let k = 1; k < n; k++) I += (k % 2 ? 4 : 2) * integrand(ri + k * h);
  I *= h / 3;
  const Er = E(r), Lr = L(r), Omr = Om(r), denom = Er - Omr * Lr;
  return (-deriv(Om, r) / (r * denom * denom)) * I;
}

/** Normalized temperature shape T(r)/T_peak in [0,1] (∝ F^{1/4}). */
let _cache: { a: number; prograde: boolean; peak: number } | null = null;
function peakFlux(a: number, prograde: boolean): number {
  if (_cache && _cache.a === a && _cache.prograde === prograde) return _cache.peak;
  const ri = iscoRadius(a, prograde);
  let peak = 0;
  for (let r = ri * 1.01; r < ri * 20; r *= 1.01) peak = Math.max(peak, pageThorneFluxShape(r, a, prograde));
  _cache = { a, prograde, peak };
  return peak;
}
export function temperatureShape(r: number, a: number, prograde = true): number {
  const f = pageThorneFluxShape(r, a, prograde);
  if (f <= 0) return 0;
  return Math.pow(f / peakFlux(a, prograde), 0.25);
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npx vitest run tests/disk.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/physics/disk.ts tests/disk.test.ts
git commit -m "feat(physics): Page-Thorne flux + normalized temperature profile"
```

---

## Task 6: Blackbody color (`src/physics/color.ts`)

**Files:**
- Create: `src/physics/color.ts`
- Test: `tests/color.test.ts`

Uses Wyman et al. (2013) analytic CIE color-matching-function fits, then XYZ→linear sRGB.

- [ ] **Step 1: Write the failing tests**

`tests/color.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { blackbodyLinearSRGB } from "../src/physics/color";

describe("blackbody color", () => {
  it("returns 3 finite, non-negative channels normalized to max=1", () => {
    const c = blackbodyLinearSRGB(6500);
    expect(c.length).toBe(3);
    for (const x of c) { expect(Number.isFinite(x)).toBe(true); expect(x).toBeGreaterThanOrEqual(0); }
    expect(Math.max(...c)).toBeCloseTo(1, 5);
  });
  it("cool stars are red-dominant (r>b)", () => {
    const c = blackbodyLinearSRGB(3000);
    expect(c[0]).toBeGreaterThan(c[2]);
  });
  it("hot stars are blue-dominant (b>r)", () => {
    const c = blackbodyLinearSRGB(20000);
    expect(c[2]).toBeGreaterThan(c[0]);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/color.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `src/physics/color.ts`**

```ts
// Wyman, Sloan & Shirley (2013), "Simple Analytic Approximations to the CIE XYZ
// Color Matching Functions" — single-lobe Gaussian fits. λ in nanometres.
function gaussian(x: number, mu: number, s1: number, s2: number): number {
  const t = (x - mu) * (x < mu ? 1 / s1 : 1 / s2);
  return Math.exp(-0.5 * t * t);
}
function cieX(l: number) { return 1.056 * gaussian(l, 599.8, 37.9, 31.0) + 0.362 * gaussian(l, 442.0, 16.0, 26.7) - 0.065 * gaussian(l, 501.1, 20.4, 26.2); }
function cieY(l: number) { return 0.821 * gaussian(l, 568.8, 46.9, 40.5) + 0.286 * gaussian(l, 530.9, 16.3, 31.1); }
function cieZ(l: number) { return 1.217 * gaussian(l, 437.0, 11.8, 36.0) + 0.681 * gaussian(l, 459.0, 26.0, 13.8); }

const H = 6.62607015e-34, C = 2.99792458e8, KB = 1.380649e-23;
function planck(lambda_m: number, T: number): number {
  return (1 / Math.pow(lambda_m, 5)) / (Math.exp(H * C / (lambda_m * KB * T)) - 1);
}

/** Linear sRGB color of a blackbody at temperature T (K), chromaticity-preserving, max channel = 1. */
export function blackbodyLinearSRGB(T: number): [number, number, number] {
  let X = 0, Y = 0, Z = 0;
  for (let nm = 360; nm <= 830; nm += 5) {
    const p = planck(nm * 1e-9, T);
    X += p * cieX(nm); Y += p * cieY(nm); Z += p * cieZ(nm);
  }
  // XYZ -> linear sRGB (IEC 61966-2-1, D65)
  let r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z;
  let g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z;
  let b = 0.0557 * X - 0.2040 * Y + 1.0570 * Z;
  r = Math.max(0, r); g = Math.max(0, g); b = Math.max(0, b);
  const m = Math.max(r, g, b) || 1;
  return [r / m, g / m, b / m];
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npx vitest run tests/color.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/physics/color.ts tests/color.test.ts
git commit -m "feat(physics): physical blackbody -> linear sRGB color (Wyman CMF fit)"
```

---

## Task 7: Lookup-table builders (`src/physics/lookups.ts`)

**Files:**
- Create: `src/physics/lookups.ts`
- Test: `tests/lookups.test.ts`

- [ ] **Step 1: Write the failing tests**

`tests/lookups.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { buildTempLUT, buildColorLUT } from "../src/physics/lookups";

describe("lookups", () => {
  it("temp LUT: length N, ~0 at inner edge, has an interior peak of 1", () => {
    const lut = buildTempLUT(0, true, 6, 40, 256);
    expect(lut.length).toBe(256);
    expect(lut[0]).toBeLessThan(0.1);
    expect(Math.max(...lut)).toBeCloseTo(1, 5);
    const argmax = lut.indexOf(Math.max(...lut));
    expect(argmax).toBeGreaterThan(0);          // peak is not at the inner edge
    expect(argmax).toBeLessThan(255);
  });
  it("color LUT: length 4N (RGBA), all finite in [0,1]", () => {
    const lut = buildColorLUT(1000, 40000, 64);
    expect(lut.length).toBe(64 * 4);
    for (const x of lut) { expect(x).toBeGreaterThanOrEqual(0); expect(x).toBeLessThanOrEqual(1); }
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/lookups.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `src/physics/lookups.ts`**

```ts
import { temperatureShape } from "./disk";
import { blackbodyLinearSRGB } from "./color";

/** Normalized T(r) sampled uniformly over [rIn, rOut]; returns Float32Array(N), peak=1. */
export function buildTempLUT(a: number, prograde: boolean, rIn: number, rOut: number, N: number): Float32Array {
  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const r = rIn + (rOut - rIn) * (i / (N - 1));
    out[i] = temperatureShape(r, a, prograde);
  }
  return out;
}
/** color(T) over [Tmin,Tmax] as RGBA Float32 (A=1). */
export function buildColorLUT(Tmin: number, Tmax: number, N: number): Float32Array {
  const out = new Float32Array(N * 4);
  for (let i = 0; i < N; i++) {
    const T = Tmin + (Tmax - Tmin) * (i / (N - 1));
    const [r, g, b] = blackbodyLinearSRGB(T);
    out[i * 4] = r; out[i * 4 + 1] = g; out[i * 4 + 2] = b; out[i * 4 + 3] = 1;
  }
  return out;
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npx vitest run tests/lookups.test.ts`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the whole core suite and commit**

Run: `npm test`
Expected: all suites PASS (kerr, orbits, geodesic, redshift, disk, color, lookups, smoke).
```bash
git add src/physics/lookups.ts tests/lookups.test.ts
git commit -m "feat(physics): T(r) and color(T) lookup-table builders"
```

---

## Task 8: WebGPU init + present/tonemap pipeline (`src/render/gpu.ts`, `present.wgsl`, `uniforms.ts`)

This task brings up WebGPU and proves the present (ACES) path by filling the accumulation buffer with a test gradient — no ray tracing yet.

**Files:**
- Create: `src/render/uniforms.ts`, `src/render/present.wgsl`, `src/render/gpu.ts`; modify `src/main.ts`

- [ ] **Step 1: Define the shared uniform layout (`src/render/uniforms.ts`)**

```ts
// Layout MUST match the `Uniforms` struct in WGSL (std140-ish; padded to 16 bytes).
// floats: resW, resH, a, incl, rObs, fovScale, rIn, rOut, Tpeak, exposure, time  (11)
// uint:   frame, reset, maxSteps                                                (3)
export interface UniformValues {
  resW: number; resH: number; a: number; incl: number; rObs: number; fovScale: number;
  rIn: number; rOut: number; Tpeak: number; exposure: number; time: number;
  frame: number; reset: number; maxSteps: number;
}
export const UNIFORM_FLOATS = 11, UNIFORM_UINTS = 3;
export const UNIFORM_SIZE = Math.ceil((UNIFORM_FLOATS + UNIFORM_UINTS) / 4) * 16; // -> 64 bytes

export function packUniforms(u: UniformValues): ArrayBuffer {
  const buf = new ArrayBuffer(UNIFORM_SIZE);
  const f = new Float32Array(buf), i = new Uint32Array(buf);
  f[0] = u.resW; f[1] = u.resH; f[2] = u.a; f[3] = u.incl;
  f[4] = u.rObs; f[5] = u.fovScale; f[6] = u.rIn; f[7] = u.rOut;
  f[8] = u.Tpeak; f[9] = u.exposure; f[10] = u.time;
  i[11] = u.frame; i[12] = u.reset; i[13] = u.maxSteps;
  return buf;
}
```

- [ ] **Step 2: Write the present shader (`src/render/present.wgsl`)**

```wgsl
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
```

- [ ] **Step 3: Write the Renderer (`src/render/gpu.ts`) with a test-gradient compute placeholder**

```ts
import { packUniforms, UniformValues, UNIFORM_SIZE } from "./uniforms";
import presentWGSL from "./present.wgsl?raw";

const TEST_COMPUTE = /* wgsl */`
struct Uniforms { res: vec2<f32>, a: f32, incl: f32, rObs: f32, fovScale: f32, rIn: f32, rOut: f32, Tpeak: f32, exposure: f32, time: f32, frame: u32, reset: u32, maxSteps: u32, };
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read_write> accum: array<vec4<f32>>;
@compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u32(U.res.x) || gid.y >= u32(U.res.y)) { return; }
  let idx = gid.y * u32(U.res.x) + gid.x;
  let uv = vec2<f32>(f32(gid.x)/U.res.x, f32(gid.y)/U.res.y);
  accum[idx] = vec4<f32>(uv.x, uv.y, 0.2, 1.0); // gradient
}`;

export class Renderer {
  device!: GPUDevice; ctx!: GPUCanvasContext; format!: GPUTextureFormat;
  uniformBuf!: GPUBuffer; accumBuf!: GPUBuffer;
  computePipe!: GPUComputePipeline; presentPipe!: GPURenderPipeline;
  computeBind!: GPUBindGroup; presentBind!: GPUBindGroup;
  width = 0; height = 0;

  async init(canvas: HTMLCanvasElement) {
    if (!navigator.gpu) throw new Error("WebGPU not available — use Chrome/Edge.");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No GPU adapter.");
    this.device = await adapter.requestDevice();
    this.ctx = canvas.getContext("webgpu")!;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.resize(canvas);
    this.uniformBuf = this.device.createBuffer({ size: UNIFORM_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.buildPipelines(TEST_COMPUTE, presentWGSL);
  }

  resize(canvas: HTMLCanvasElement) {
    const dpr = Math.min(devicePixelRatio || 1, 1.5);
    this.width = Math.floor(canvas.clientWidth * dpr);
    this.height = Math.floor(canvas.clientHeight * dpr);
    canvas.width = this.width; canvas.height = this.height;
    this.ctx.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
    this.accumBuf = this.device.createBuffer({ size: this.width * this.height * 16, usage: GPUBufferUsage.STORAGE });
  }

  buildPipelines(computeSrc: string, presentSrc: string) {
    const cMod = this.device.createShaderModule({ code: computeSrc });
    const pMod = this.device.createShaderModule({ code: presentSrc });
    this.computePipe = this.device.createComputePipeline({ layout: "auto", compute: { module: cMod, entryPoint: "main" } });
    this.presentPipe = this.device.createRenderPipeline({
      layout: "auto", vertex: { module: pMod, entryPoint: "vs" },
      fragment: { module: pMod, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });
    this.rebind();
  }

  rebind() {
    this.computeBind = this.device.createBindGroup({ layout: this.computePipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.uniformBuf } }, { binding: 1, resource: { buffer: this.accumBuf } }] });
    this.presentBind = this.device.createBindGroup({ layout: this.presentPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.uniformBuf } }, { binding: 1, resource: { buffer: this.accumBuf } }] });
  }

  frame(u: UniformValues) {
    this.device.queue.writeBuffer(this.uniformBuf, 0, packUniforms(u));
    const enc = this.device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(this.computePipe); cp.setBindGroup(0, this.computeBind);
    cp.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8)); cp.end();
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }] });
    rp.setPipeline(this.presentPipe); rp.setBindGroup(0, this.presentBind); rp.draw(3); rp.end();
    this.device.queue.submit([enc.finish()]);
  }
}
```

- [ ] **Step 4: Wire `src/main.ts` to render the gradient**

```ts
import { Renderer } from "./render/gpu";
import type { UniformValues } from "./render/uniforms";

const canvas = document.getElementById("c") as HTMLCanvasElement;
const r = new Renderer();
const u: UniformValues = { resW: 0, resH: 0, a: 0.9, incl: 72 * Math.PI / 180, rObs: 1000,
  fovScale: 14, rIn: 0, rOut: 40, Tpeak: 1.5e7, exposure: 0, time: 0, frame: 0, reset: 1, maxSteps: 4000 };

await r.init(canvas);
function loop() {
  u.resW = r.width; u.resH = r.height;
  r.frame(u);
  requestAnimationFrame(loop);
}
loop();
```

- [ ] **Step 5: Verify in the browser**

Run: `npm run dev`, open `http://localhost:5173` in Chrome/Edge.
Expected: a smooth red→green gradient fills the canvas (proves WebGPU device, compute write, storage buffer, present pass, ACES, gamma). If you see "WebGPU not available", enable it in the browser. Stop the server.

- [ ] **Step 6: Commit**

```bash
git add src/render/ src/main.ts
git commit -m "feat(render): WebGPU bring-up — compute->storage->ACES present (gradient)"
```

---

## Task 9: Kerr ray-tracing compute shader (`src/render/raytrace.wgsl`)

Port the verified core math to WGSL and replace the gradient. The shader integrates one backward null geodesic per pixel and writes the emitted disk radiance (no LUTs yet — uses a temporary analytic T(r) so we can see structure; LUTs arrive in Task 10).

**Files:**
- Create: `src/render/raytrace.wgsl`; modify `src/render/gpu.ts` (swap compute source)

- [ ] **Step 1: Write `src/render/raytrace.wgsl`**

```wgsl
struct Uniforms {
  res: vec2<f32>, a: f32, incl: f32, rObs: f32, fovScale: f32, rIn: f32, rOut: f32,
  Tpeak: f32, exposure: f32, time: f32, frame: u32, reset: u32, maxSteps: u32,
};
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read_write> accum: array<vec4<f32>>;

const PI = 3.141592653589793;

fn delta_(r: f32, a: f32) -> f32 { return r*r - 2.0*r + a*a; }
fn sigma_(r: f32, th: f32, a: f32) -> f32 { let c = cos(th); return r*r + a*a*c*c; }
fn bigA_(r: f32, th: f32, a: f32) -> f32 { let s = sin(th); return pow(r*r+a*a,2.0) - a*a*delta_(r,a)*s*s; }

// upper metric components (tt, tphi, rr, thth, phph)
fn gUp(r: f32, th: f32, a: f32) -> array<f32,5> {
  let s2 = sin(th)*sin(th); let Sig = sigma_(r,th,a); let d = delta_(r,a); let A = bigA_(r,th,a);
  return array<f32,5>( -A/(Sig*d), -2.0*a*r/(Sig*d), d/Sig, 1.0/Sig, (d - a*a*s2)/(Sig*d*s2) );
}
fn gLow(r: f32, th: f32, a: f32) -> array<f32,5> {
  let s2 = sin(th)*sin(th); let Sig = sigma_(r,th,a); let d = delta_(r,a);
  return array<f32,5>( -(1.0-2.0*r/Sig), -2.0*a*r*s2/Sig, Sig/d, Sig, (r*r+a*a+2.0*a*a*r*s2/Sig)*s2 );
}
fn omegaKep(r: f32, a: f32) -> f32 { return 1.0/(pow(r,1.5) + a); } // prograde, M=1

fn hquad(r: f32, th: f32, a: f32, p: vec4<f32>) -> f32 {
  let g = gUp(r,th,a);
  return g[0]*p.x*p.x + 2.0*g[1]*p.x*p.w + g[2]*p.y*p.y + g[3]*p.z*p.z + g[4]*p.w*p.w;
}
// state s = (t,r,th,phi, pt,pr,pth,pphi) packed as two vec4
struct State { x: vec4<f32>, p: vec4<f32> };
fn rhs(s: State, a: f32) -> State {
  let r = s.x.y; let th = s.x.z; let g = gUp(r,th,a);
  let dx = vec4<f32>(g[0]*s.p.x + g[1]*s.p.w, g[2]*s.p.y, g[3]*s.p.z, g[1]*s.p.x + g[4]*s.p.w);
  let h = 1e-4;
  let dQdr = (hquad(r+h,th,a,s.p) - hquad(r-h,th,a,s.p))/(2.0*h);
  let dQdth = (hquad(r,th+h,a,s.p) - hquad(r,th-h,a,s.p))/(2.0*h);
  let dp = vec4<f32>(0.0, -0.5*dQdr, -0.5*dQdth, 0.0);
  return State(dx, dp);
}
fn addS(s: State, k: State, f: f32) -> State { return State(s.x + k.x*f, s.p + k.p*f); }
fn rk4(s: State, a: f32, dl: f32) -> State {
  let k1 = rhs(s,a); let k2 = rhs(addS(s,k1,dl*0.5),a);
  let k3 = rhs(addS(s,k2,dl*0.5),a); let k4 = rhs(addS(s,k3,dl),a);
  return State(s.x + (k1.x+2.0*k2.x+2.0*k3.x+k4.x)*(dl/6.0),
               s.p + (k1.p+2.0*k2.p+2.0*k3.p+k4.p)*(dl/6.0));
}

// temporary analytic temperature (replaced by LUT in Task 10): peaks outside ISCO, ~r^-3/4
fn tempTmp(r: f32, rin: f32) -> f32 {
  if (r <= rin) { return 0.0; }
  let f = pow(rin/r, 0.75) * pow(max(0.0, 1.0 - sqrt(rin/r)), 0.25);
  return f / 0.23; // rough normalization so peak ~1
}
fn cheapColor(T: f32) -> vec3<f32> { // placeholder palette (real color LUT in Task 10)
  let t = clamp(T, 0.0, 1.0);
  return mix(vec3(1.0,0.3,0.05), vec3(0.6,0.8,1.0), t);
}

@compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u32(U.res.x) || gid.y >= u32(U.res.y)) { return; }
  let idx = gid.y * u32(U.res.x) + gid.x;
  let a = U.a; let i = U.incl;

  // pixel -> impact parameters (alpha,beta) in units of M
  let aspect = U.res.x / U.res.y;
  let ndc = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / U.res * 2.0 - 1.0;
  let alpha = ndc.x * U.fovScale * aspect;
  let beta  = -ndc.y * U.fovScale;
  // invert Bardeen: xi = -alpha*sin i ; eta = beta^2 - a^2 cos^2 i + xi^2 cot^2 i
  let xi = -alpha * sin(i);
  let ci = cos(i); let cot2 = (ci*ci)/(sin(i)*sin(i));
  let eta = beta*beta - a*a*ci*ci + xi*xi*cot2;

  // initial state at (rObs, i, 0), E=1
  let r0 = U.rObs; let th0 = i;
  let pt = -1.0; let pphi = xi; let pth = beta; // sign of p_th set by image y
  let gU = gUp(r0, th0, a);
  let rest = gU[0]*pt*pt + 2.0*gU[1]*pt*pphi + gU[3]*pth*pth + gU[4]*pphi*pphi;
  let pr = -sqrt(max(0.0, -rest/gU[2])); // inward
  var s = State(vec4<f32>(0.0, r0, th0, 0.0), vec4<f32>(pt, pr, pth, pphi));

  let rh = 1.0 + sqrt(max(0.0, 1.0 - a*a)); // horizon
  var color = vec3<f32>(0.0);
  var prevTh = s.x.z;

  for (var step = 0u; step < U.maxSteps; step++) {
    // adaptive step: smaller near the hole
    let r = s.x.y;
    let dl = -clamp(0.02 * (r - rh), 0.002, 0.5);
    let sNew = rk4(s, a, dl);

    // disk crossing: equatorial plane th = PI/2
    let f0 = s.x.z - PI*0.5; let f1 = sNew.x.z - PI*0.5;
    if (f0 * f1 < 0.0) {
      let frac = f0 / (f0 - f1);
      let rHit = mix(s.x.y, sNew.x.y, frac);
      if (rHit >= U.rIn && rHit <= U.rOut) {
        let Tn = tempTmp(rHit, U.rIn);
        let Om = omegaKep(rHit, a);
        let gl = gLow(rHit, PI*0.5, a);
        let rad = -(gl[0] + 2.0*Om*gl[1] + Om*Om*gl[4]);
        let g = sqrt(max(0.0, rad)) / (1.0 - Om*xi);  // redshift factor
        let u = g * Tn;                                // observed color temperature factor
        color = cheapColor(clamp(u, 0.0, 1.0)) * pow(u, 4.0); // brightness ∝ (g*T)^4
        break;
      }
    }
    s = sNew;
    if (s.x.y <= rh * 1.001) { color = vec3(0.0); break; }   // captured -> shadow
    if (s.x.y > r0 * 1.2) { color = vec3(0.0); break; }      // escaped -> background
  }

  let prev = select(accum[idx].rgb, vec3(0.0), U.reset == 1u);
  accum[idx] = vec4<f32>(prev + color, 1.0);
}
```

- [ ] **Step 2: Swap the compute source in `gpu.ts`**

Modify `src/render/gpu.ts`: replace the inline `TEST_COMPUTE` import/use with the real shader.
```ts
// at top, add:
import raytraceWGSL from "./raytrace.wgsl?raw";
// remove the TEST_COMPUTE constant, and in init() change:
this.buildPipelines(raytraceWGSL, presentWGSL);
```

- [ ] **Step 3: Verify in the browser**

Run: `npm run dev`, open the page.
Expected: a recognizable warped accretion disk with a dark central shadow and a bright/dim asymmetry (Doppler) — colors are placeholder. Geometry (shadow, lensed top arc) should look right at a*=0.9, i=72°. If the image is upside-down or mirrored, note it for Task 12 (the `pth = beta` / `xi` sign). Stop the server.

- [ ] **Step 4: Commit**

```bash
git add src/render/raytrace.wgsl src/render/gpu.ts
git commit -m "feat(render): Kerr backward-geodesic ray tracer (placeholder color)"
```

---

## Task 10: Wire LUTs, progressive accumulation, real camera/uniforms (first physical render)

Replace placeholder T(r)/color with the verified CPU LUTs (uploaded as textures) and add progressive anti-aliasing (jittered samples accumulated across frames; reset on parameter change).

**Files:**
- Modify: `src/render/gpu.ts` (create LUT textures + sampler, add bindings, jitter), `src/render/raytrace.wgsl` (sample LUTs, jitter), `src/main.ts` (build LUTs, frame counter/reset)

- [ ] **Step 1: Build and upload LUT textures in `gpu.ts`**

Add to `Renderer`:
```ts
tempTex!: GPUTexture; colorTex!: GPUTexture; lutSampler!: GPUSampler;

uploadLUTs(tempLUT: Float32Array, colorLUT: Float32Array) {
  const mk = (w: number) => this.device.createTexture({ size: [w, 1], format: "rgba32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST, dimension: "2d" });
  // temp LUT is scalar -> expand to RGBA (store in .r)
  const tempRGBA = new Float32Array(tempLUT.length * 4);
  for (let i = 0; i < tempLUT.length; i++) tempRGBA[i * 4] = tempLUT[i];
  this.tempTex = mk(tempLUT.length);
  this.device.queue.writeTexture({ texture: this.tempTex }, tempRGBA, { bytesPerRow: tempLUT.length * 16 }, [tempLUT.length, 1]);
  const colorW = colorLUT.length / 4;
  this.colorTex = mk(colorW);
  this.device.queue.writeTexture({ texture: this.colorTex }, colorLUT, { bytesPerRow: colorW * 16 }, [colorW, 1]);
  this.lutSampler = this.device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge" });
}
```
Note: `rgba32float` is not filterable by default; request the feature in `init()`:
```ts
const adapter = await navigator.gpu.requestAdapter();
const canFilter = adapter!.features.has("float32-filterable");
this.device = await adapter!.requestDevice({ requiredFeatures: canFilter ? ["float32-filterable"] : [] });
```
If `float32-filterable` is unavailable, set the LUT sampler `magFilter`/`minFilter` to `"nearest"` (still correct, slightly stair-stepped).

Extend `rebind()` compute bind group entries to include `{ binding: 2, sampler: this.lutSampler }`, `{ binding: 3, resource: this.tempTex.createView() }`, `{ binding: 4, resource: this.colorTex.createView() }`.

- [ ] **Step 2: Update `raytrace.wgsl` to sample LUTs + jitter**

Add bindings and replace the placeholder color/temperature:
```wgsl
@group(0) @binding(2) var lutSamp: sampler;
@group(0) @binding(3) var tempLUT: texture_2d<f32>;   // .r = normalized T(r)
@group(0) @binding(4) var colorLUT: texture_2d<f32>;  // rgb = blackbody color

fn sampleTemp(r: f32) -> f32 {
  let u = clamp((r - U.rIn) / (U.rOut - U.rIn), 0.0, 1.0);
  return textureSampleLevel(tempLUT, lutSamp, vec2<f32>(u, 0.5), 0.0).r;
}
fn sampleColor(T_kelvin: f32) -> vec3<f32> {
  // color LUT spans [1000, 40000] K
  let u = clamp((T_kelvin - 1000.0) / (40000.0 - 1000.0), 0.0, 1.0);
  return textureSampleLevel(colorLUT, lutSamp, vec2<f32>(u, 0.5), 0.0).rgb;
}
// simple per-frame hash jitter for progressive AA
fn hash2(p: vec2<u32>, frame: u32) -> vec2<f32> {
  let n = p.x * 1973u + p.y * 9277u + frame * 26699u;
  let h = (n ^ (n >> 15u)) * 2246822519u;
  let h2 = (h ^ (h >> 13u)) * 3266489917u;
  return vec2<f32>(f32(h & 0xffffu)/65535.0, f32(h2 & 0xffffu)/65535.0);
}
```
Replace the NDC line with a jittered version:
```wgsl
  let jit = hash2(gid.xy, U.frame) - 0.5;
  let ndc = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5 + jit) / U.res * 2.0 - 1.0;
```
Replace the hit-color block:
```wgsl
        let Tn = sampleTemp(rHit);
        let Om = omegaKep(rHit, a);
        let gl = gLow(rHit, PI*0.5, a);
        let rad = -(gl[0] + 2.0*Om*gl[1] + Om*Om*gl[4]);
        let g = sqrt(max(0.0, rad)) / (1.0 - Om*xi);
        let Tobs = U.Tpeak * g * Tn;             // observed blackbody temperature
        let bright = pow(g * Tn, 4.0);           // bolometric beaming ∝ (gT)^4
        color = sampleColor(Tobs) * bright;
        break;
```

- [ ] **Step 3: Build LUTs and drive the frame loop in `main.ts`**

```ts
import { Renderer } from "./render/gpu";
import type { UniformValues } from "./render/uniforms";
import { buildTempLUT, buildColorLUT } from "./physics/lookups";
import { iscoRadius } from "./physics/orbits";

const canvas = document.getElementById("c") as HTMLCanvasElement;
const r = new Renderer();
await r.init(canvas);

const state = { a: 0.9, incl: 72, exposure: 0 };
const rOut = 40;
function rebuildLUTs() {
  const rin = iscoRadius(state.a, true);
  r.uploadLUTs(buildTempLUT(state.a, true, rin, rOut, 512), buildColorLUT(1000, 40000, 256));
  r.rebind();
  return rin;
}
let rIn = rebuildLUTs();
let frame = 0, reset = 1;

function loop() {
  const u: UniformValues = {
    resW: r.width, resH: r.height, a: state.a, incl: state.incl * Math.PI / 180,
    rObs: 1000, fovScale: 14, rIn, rOut, Tpeak: 1.5e7, exposure: state.exposure,
    time: performance.now() / 1000, frame, reset, maxSteps: 6000,
  };
  r.frame(u);
  frame = reset ? 0 : frame + 1; // first accumulated frame index is 0
  reset = 0;
  requestAnimationFrame(loop);
}
loop();
```
(Controls wiring comes in Task 13; for now this renders and progressively refines.)

- [ ] **Step 4: Verify in the browser**

Run: `npm run dev`.
Expected: the disk now shows physically-derived color — a hot blue-white inner region grading through yellow/orange to deep red at the outer edge, with the approaching side visibly **bluer and brighter** and the receding side dimmer/redder; the image sharpens over ~1–2 s as samples accumulate. Stop the server.

- [ ] **Step 5: Commit**

```bash
git add src/render/gpu.ts src/render/raytrace.wgsl src/main.ts
git commit -m "feat(render): physical T(r)/color LUTs + g-factor color + progressive AA"
```

---

## Task 11: CPU↔GPU parity test (`src/render/parity.wgsl`, `src/test/parity.browser.ts`)

Guards the WGSL port: runs the shader's metric/orbit/g-factor functions on a fixed set of inputs and compares to the TypeScript core.

**Files:**
- Create: `src/render/parity.wgsl`, `src/test/parity.browser.ts`; modify `index.html` (add a `?parity` entry) — runnable manually or via Playwright.

- [ ] **Step 1: Write `src/render/parity.wgsl`**

Reuse the metric/orbit functions (copy the `delta_/sigma_/bigA_/gUp/gLow/omegaKep` block from `raytrace.wgsl`), then:
```wgsl
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
```

- [ ] **Step 2: Write the harness `src/test/parity.browser.ts`**

```ts
import { metricUpper, metricLower } from "../physics/kerr";
import { omegaKepler } from "../physics/orbits";
import { gFactorKepler } from "../physics/redshift";
import parityWGSL from "../render/parity.wgsl?raw";

export async function runParity(): Promise<{ maxErr: number; rows: number }> {
  const cases = [
    { r: 8, th: Math.PI / 2, a: 0.0, xi: 3 }, { r: 6, th: 1.2, a: 0.5, xi: 2 },
    { r: 12, th: Math.PI / 2, a: 0.9, xi: -4 }, { r: 20, th: 0.9, a: 0.99, xi: 5 },
  ];
  const adapter = await navigator.gpu.requestAdapter(); const device = await adapter!.requestDevice();
  const inBuf = device.createBuffer({ size: cases.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const inArr = new Float32Array(cases.length * 4);
  cases.forEach((c, i) => { inArr.set([c.r, c.th, c.a, c.xi], i * 4); });
  device.queue.writeBuffer(inBuf, 0, inArr);
  const outBuf = device.createBuffer({ size: cases.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: cases.length * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const mod = device.createShaderModule({ code: parityWGSL });
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
  return { maxErr, rows: cases.length };
}
```

- [ ] **Step 3: Add a parity entry to `main.ts`**

```ts
if (location.search.includes("parity")) {
  const { runParity } = await import("./test/parity.browser");
  const res = await runParity();
  const ok = res.maxErr < 1e-3;
  document.body.innerHTML = `<pre style="color:${ok ? "#6f6" : "#f66"};font-size:18px;padding:20px">
PARITY ${ok ? "PASS" : "FAIL"} — maxRelErr=${res.maxErr.toExponential(3)} over ${res.rows} cases</pre>`;
  console.log("parity", res);
}
```

- [ ] **Step 4: Verify**

Run: `npm run dev`, open `http://localhost:5173/?parity`.
Expected: green "PARITY PASS — maxRelErr ≈ 1e-6..1e-4". (f32 GPU vs f64 CPU, finite-diff differences keep it < 1e-3.) If it FAILS, a WGSL function diverged from the core — fix the shader to match. Stop the server.

- [ ] **Step 5: Commit**

```bash
git add src/render/parity.wgsl src/test/parity.browser.ts src/main.ts
git commit -m "test(render): CPU<->GPU parity harness for metric/orbit/g-factor"
```

---

## Task 12: Shadow-radius validation (a=0 ⇒ √27 M)

Renders Schwarzschild, reads back the framebuffer, measures the shadow's apparent radius along the horizontal midline, and checks it against $\sqrt{27}\,M$ scaled by the camera FOV. Also confirms image orientation (fixing the `pth/xi` sign if Task 9 looked mirrored).

**Files:**
- Create: `src/test/shadow.browser.ts`; modify `src/main.ts` (add `?shadow` entry); modify `gpu.ts` (add `readbackPresented()` helper)

- [ ] **Step 1: Add a framebuffer readback helper to `gpu.ts`**

Render the present pass into an offscreen `rgba8unorm` texture (with `COPY_SRC`) instead of the swapchain when requested, then copy to a mappable buffer. Add:
```ts
async readbackPresented(u: UniformValues): Promise<{ data: Uint8Array; w: number; h: number }> {
  const tex = this.device.createTexture({ size: [this.width, this.height], format: this.format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  this.device.queue.writeBuffer(this.uniformBuf, 0, packUniforms(u));
  const enc = this.device.createCommandEncoder();
  const cp = enc.beginComputePass(); cp.setPipeline(this.computePipe); cp.setBindGroup(0, this.computeBind);
  cp.dispatchWorkgroups(Math.ceil(this.width/8), Math.ceil(this.height/8)); cp.end();
  const rp = enc.beginRenderPass({ colorAttachments: [{ view: tex.createView(), clearValue: {r:0,g:0,b:0,a:1}, loadOp: "clear", storeOp: "store" }] });
  rp.setPipeline(this.presentPipe); rp.setBindGroup(0, this.presentBind); rp.draw(3); rp.end();
  const bpr = Math.ceil(this.width * 4 / 256) * 256;
  const buf = this.device.createBuffer({ size: bpr * this.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [this.width, this.height]);
  this.device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const padded = new Uint8Array(buf.getMappedRange().slice(0));
  const data = new Uint8Array(this.width * this.height * 4);
  for (let y = 0; y < this.height; y++) data.set(padded.subarray(y*bpr, y*bpr + this.width*4), y*this.width*4);
  return { data, w: this.width, h: this.height };
}
```

- [ ] **Step 2: Write `src/test/shadow.browser.ts`**

```ts
import { Renderer } from "../render/gpu";
import { buildTempLUT, buildColorLUT } from "../physics/lookups";
import { iscoRadius } from "../physics/orbits";
import type { UniformValues } from "../render/uniforms";

export async function measureShadow(canvas: HTMLCanvasElement) {
  const r = new Renderer(); await r.init(canvas);
  const a = 0, rin = iscoRadius(a, true), rOut = 40, fovScale = 14;
  r.uploadLUTs(buildTempLUT(a, true, rin, rOut, 512), buildColorLUT(1000, 40000, 256)); r.rebind();
  // FACE-ON (i≈0.01) so the shadow is a centered disk; integrate one full frame
  const u: UniformValues = { resW: r.width, resH: r.height, a, incl: 0.01, rObs: 1000,
    fovScale, rIn: rin, rOut, Tpeak: 1.5e7, exposure: 0, time: 0, frame: 0, reset: 1, maxSteps: 8000 };
  const { data, w, h } = await r.readbackPresented(u);
  // scan the central row for the dark (shadow) span
  const y = (h >> 1) * w * 4; let lo = -1, hi = -1;
  for (let x = 0; x < w; x++) {
    const lum = data[y + x*4] + data[y + x*4 + 1] + data[y + x*4 + 2];
    if (lum < 20) { if (lo < 0) lo = x; hi = x; }
  }
  const shadowPx = hi - lo;
  // expected diameter in pixels: 2*sqrt(27)*M mapped through fovScale (half-width = fovScale*M across w/2 px)
  const expectedPx = (Math.sqrt(27) / fovScale) * (w / 2) * 2;
  const relErr = Math.abs(shadowPx - expectedPx) / expectedPx;
  return { shadowPx, expectedPx, relErr };
}
```

- [ ] **Step 3: Add a `?shadow` entry to `main.ts`**

```ts
if (location.search.includes("shadow")) {
  const { measureShadow } = await import("./test/shadow.browser");
  const res = await measureShadow(canvas);
  const ok = res.relErr < 0.1;
  document.body.innerHTML = `<pre style="color:${ok ? "#6f6" : "#f66"};font-size:18px;padding:20px">
SHADOW ${ok ? "PASS" : "FAIL"} — measured=${res.shadowPx.toFixed(1)}px expected=${res.expectedPx.toFixed(1)}px relErr=${(res.relErr*100).toFixed(1)}%</pre>`;
}
```

- [ ] **Step 4: Verify**

Run: `npm run dev`, open `http://localhost:5173/?shadow`.
Expected: green "SHADOW PASS" with measured ≈ expected within 10% (the apparent Schwarzschild shadow diameter is $2\sqrt{27}\,M$). If FAIL, check `fovScale` mapping and the camera setup. While here, open the normal view and confirm the disk is **not** mirrored (approaching side on the expected side); if it is, flip the sign of `pth = beta` → `pth = -beta` in `raytrace.wgsl` and re-run parity + shadow. Stop the server.

- [ ] **Step 5: Commit**

```bash
git add src/render/gpu.ts src/test/shadow.browser.ts src/main.ts
git commit -m "test(render): Schwarzschild shadow-radius validation (=sqrt(27)M)"
```

---

## Task 13: Interactive controls (spin, inclination, exposure)

**Files:**
- Modify: `src/main.ts`

- [ ] **Step 1: Wire the existing sliders and reset accumulation on change**

Replace the control-related part of `main.ts` (`index.html` already has `#spin`, `#incl`, `#exp`, and value spans):
```ts
const $ = (id: string) => document.getElementById(id) as HTMLInputElement;
function bind(id: string, valId: string, onChange: (v: number) => void) {
  const el = $(id), out = document.getElementById(valId)!;
  const update = () => { out.textContent = el.value; onChange(parseFloat(el.value)); reset = 1; };
  el.addEventListener("input", update); update();
}
bind("spin", "spinv", v => { state.a = v; rIn = rebuildLUTs(); });
bind("incl", "inclv", v => { state.incl = v; });
bind("exp",  "expv",  v => { state.exposure = v; });
```
(`reset = 1` restarts progressive accumulation whenever a parameter changes, so the image re-converges cleanly.)

- [ ] **Step 2: Verify**

Run: `npm run dev`.
Expected: dragging **spin** changes the ISCO/shadow asymmetry and re-converges; **inclination** tilts the disk from near face-on (small angle) to the dramatic edge-on warp (high angle); **exposure** brightens/darkens via the ACES curve. Each change resets accumulation and re-sharpens. Stop the server.

- [ ] **Step 3: Run full unit suite + commit**

Run: `npm test`
Expected: all PASS.
```bash
git add src/main.ts
git commit -m "feat(ui): interactive spin/inclination/exposure controls"
```

---

## Self-Review

**Spec coverage (Tier-1 scope):**
- Kerr metric, horizons, ergosphere, frame-dragging → Task 1. ✓
- ISCO inner boundary, relativistic Keplerian Ω, orbits → Task 2. ✓
- Null geodesics (Hamiltonian), shadow, photon ring, higher-order arc → Tasks 3, 9 (emerge from integration), 12 (shadow validated). ✓
- Novikov–Thorne/Page–Thorne T(r), peak outside ISCO → Task 5. ✓
- Blackbody emission + physical color (Planck→CIE→sRGB) → Tasks 6, 10. ✓
- Doppler beaming + gravitational redshift via the single g-factor, $I\propto g^3$/$g^4$ → Tasks 4, 10. ✓
- Off-axis observer (i≈72°) → Tasks 9, 10. ✓
- HDR + ACES tonemapping + anti-aliasing → Tasks 8, 10. ✓
- *Deferred (separate plans, stated in scope):* jets/synchrotron, MRI flares & visible time-evolution, lensed starfield, bloom, CUDA port, Tier-2/3. These are intentionally not Tier-1-core requirements.

**Placeholder scan:** No "TBD"/"handle later". The two temporary shader stubs in Task 9 (`tempTmp`, `cheapColor`) are explicitly replaced with verified LUTs in Task 10. ✓

**Type consistency:** `Metric` shape `{tt,tphi,rr,thth,phph}` used identically in `kerr.ts`, `geodesic.ts`, `redshift.ts`, and mirrored as `array<f32,5>` index order `(tt,tphi,rr,thth,phph)` in WGSL. `UniformValues` field order matches `packUniforms` indices and the WGSL `Uniforms` struct (resolution vec2 first, then floats, then the three u32). State vector order `[t,r,θ,φ,p_t,p_r,p_θ,p_φ]` consistent across `geodesic.ts` and the WGSL `State` (`x`=first four, `p`=last four). LUT spans: temp over `[rIn,rOut]`, color over `[1000,40000]` K — same constants in `main.ts`, `lookups.ts`, and `raytrace.wgsl`. ✓

**Known precision note (not a blocker):** the shader uses f32; near-horizon geodesics can show mild noise. The adaptive step (`dl ∝ r − r_h`) and progressive accumulation mitigate it. If artifacts appear at high spin, reduce min step in Task 9 or raise `maxSteps`.
