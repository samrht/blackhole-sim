# Eruption Flares Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Each horizon-flux eruption launches a reconnection-heated flux tube that spirals out to 5–30 r_g, orbits
the disk at the Keplerian speed for up to two orbits and shines as optically thin synchrotron light added to the
disk where rays hit it; it replaces the illustrative Flares.

**Architecture:** `src/physics/eruption-spots.ts` (CPU twin) defines tube k from the flux history (birth t_k, drop
duration D_k, depth δ_k), its orbit (r_c, R, φ₀), its path and light curve, its field (flux conservation), its
injected power (ζ × its field energy, ζ from the Sgr A* anchor) and its column intensity through the jet's
cooled-synchrotron coefficients. The coefficient step of `synchSampleJ` is factored into `synchCoeffsJ` shared by
jet and tubes. `emission-shared.wgsl` gets the WGSL twins; `shadeDisk` adds the tube light (shading-only, so the
geodesic cache needs nothing new). The illustrative hot spots (`HotSpot`, storage binding 4, `nSpots`) are removed.

**Tech Stack:** TypeScript, WebGPU/WGSL, Vitest, Playwright (GPU gates).

**Spec:** `docs/specs/2026-10-04-eruption-flares-design.md` — read "Corrections from planning" (they override §2.3).

## Global Constraints

- `FLARE = { rMin: 5, rMax: 30, sizeFrac: 0.2, lifeOrbits: 2, saltR: 0x5243, saltPhi: 0x5048, anchorErg: 1e38, cA: 0.48452046 }`
  — cA = ∫₀^∞ e^{−2u}(1 − smoothstep(1.5, 2, u)) du (so ∫A dτ = D/2 + P·cA).
- r_c,k = max(5 + 25 u_k, r_ISCO/(1 − 2·0.2)) (≥ r_ISCO + 2R); u_k = fluxHash(k, saltR) + 0.5, φ₀ = 2π(fluxHash(k, saltPhi) + 0.5).
- Spiral r(τ) = r_in + (r_c − r_in) τ/D with r_in = r_ISCO; φ(τ) = φ₀ + (D/(r_c − r_in)) ∫_{r_in}^{r(τ)} Ω_K dr (GL6); Ω_K = 1/(r^{1.5} + a).
- Field top-hat: B = δ s Φ / (π R² r_g²) (R in r_g); field energy E = (δ s)² Φ² / (4π² R r_g); injected power
  P(τ) = ζ f E A(τ)/(D/2 + P_orb cA); emission Gaussian exp(−d²/2R²) with q = P/(B² · 4π R³ r_g³); cooling depth
  s_c = e^{LN_K0} B² τ r_g/c; column path ds = 2R r_g; plasma shift D = 1/g (the disk's g).
- ζ = 10³⁸ / ⟨E⟩ at the Sgr A* preset, ⟨E⟩ = δ̄² (1 + 1/48) Φ² (ln 6/25)/(0.8 π² r_g) — expected 1.061 × 10⁻².
- Uniforms: i[18] `nSpots` → f[18] `flareStrength`; f[36] `flarePhi` (G cm²), f[37] `flareZeta`; UNIFORM_FLOATS 30,
  size 160 B. WGSL struct: `breatheAmp: f32, flareStrength: f32,` and after `timeEpoch: f32,` add `flarePhi: f32, flareZeta: f32,`.
- No Co-Authored-By trailer. Merge, push and deploy when done (user request).

## Review Focus

1. **Tube birth at large epochs** — tube times come from the split clock (whole flux periods + remainder) exactly as
   `fluxRatioJ`; parity cases at epochs to 2048 × 8000 (Task 2).
2. **Low spin / ISCO** — at a = 0 (ISCO 6) the clamp keeps r_c ≥ 10 and the spiral starts at the ISCO; no NaN (Task 1).
3. **Slider and switches** — Eruption flares 0, Flux variability 0 → exactly no tube light; jet on/off independent (Tasks 1, 3).
4. **Cache** — tube light is pure shading: `?cachecheck` must stay 0.00e+0 (Task 3).
5. **Jet unchanged** — the `synchCoeffsJ` refactor must keep the jet's parity numbers (Task 2) and its pixels (golden scenes
   without tubes alive) — verified by the golden re-record diff and the jet-free-kernel check if needed (Task 3).

---

### Task 1: Tube model on the CPU

**Files:** Create `src/physics/eruption-spots.ts`; Test `tests/eruption-spots.test.ts`.

**Produces:** `FLARE`; `interface Tube { k; t0; D; depth; rc; R; phi0 }`; `tubeOf(k: number, a: number): Tube`;
`tubeAt(tube: Tube, t: number, a: number): { r: number; phi: number; A: number }` (A = 0 outside its life);
`tubeLife(tube, a): number` (D + 2 P_orb); `tubeField(tube, s, phi, rgCm): number` (G); `tubeEnergy(tube, s, phi, rgCm): number`
(erg); `tubePower(tube, t, a, s, f, phi, zeta, rgCm): number` (erg/s); `flareZeta(): number`; `meanTubeEnergy(s, phi, rgCm): number`.

- [ ] **Step 1: Failing tests** (`tests/eruption-spots.test.ts`):

```ts
import { describe, it, expect } from "vitest";
import { FLARE, tubeOf, tubeAt, tubeLife, tubeField, tubeEnergy, tubePower, flareZeta, meanTubeEnergy } from "../src/physics/eruption-spots";
import { eruptionTime, eruptionDepth, FLUX } from "../src/physics/flux-history";
import { iscoRadius } from "../src/physics/orbits";
import { jetEnergetics } from "../src/physics/synchrotron";
import { PRESETS } from "../src/physics/presets";

const sgra = PRESETS.find((p) => p.id === "sgra")!, E = jetEnergetics(sgra.massSun, sgra.a, sgra.lambda);
describe("eruption flux tubes (spec 2026-10-04 eruption flares)", () => {
  it("one tube per eruption, born at the eruption time with its depth and drop duration", () => {
    for (const k of [0, 7, 123]) {
      const t = tubeOf(k, 0.9);
      expect(t.t0).toBe(eruptionTime(k)); expect(t.depth).toBe(eruptionDepth(k, FLUX.dbar));
      expect(t.D).toBeCloseTo(-FLUX.tauD * Math.log(1 - t.depth), 12);
    }
  });
  it("orbit radius in [5, 30] r_g, size 0.2 r_c, outside ISCO + 2R at every spin (Review Focus 2)", () => {
    for (const a of [0, 0.5, 0.94]) for (let k = 0; k < 500; k++) {
      const t = tubeOf(k, a);
      expect(t.rc).toBeLessThanOrEqual(30); expect(t.R).toBeCloseTo(0.2 * t.rc, 12);
      expect(t.rc).toBeGreaterThanOrEqual(Math.max(5, iscoRadius(a, true) + 2 * t.R) - 1e-9);
      expect(t.phi0).toBeGreaterThanOrEqual(0); expect(t.phi0).toBeLessThan(2 * Math.PI);
    }
  });
  it("spirals from the ISCO to r_c over D with continuous azimuth, then orbits at the Keplerian rate", () => {
    const a = 0.9, t = tubeOf(5, a), rin = iscoRadius(a, true);
    expect(tubeAt(t, t.t0 + 1e-6, a).r).toBeCloseTo(rin, 4);
    const end = tubeAt(t, t.t0 + t.D - 1e-6, a), after = tubeAt(t, t.t0 + t.D + 1e-6, a);
    expect(end.r).toBeCloseTo(t.rc, 4); expect(after.phi).toBeCloseTo(end.phi, 4);
    const Om = 1 / (t.rc ** 1.5 + a), p1 = tubeAt(t, t.t0 + t.D + 50, a).phi, p2 = tubeAt(t, t.t0 + t.D + 150, a).phi;
    expect((p2 - p1) / 100).toBeCloseTo(Om, 10);
    // spiral azimuth = brute-force integral of Omega_K along r(tau)
    const tau = 0.6 * t.D; let ph = t.phi0; const n = 20000;
    for (let i = 0; i < n; i++) { const s = (i + 0.5) / n * tau, r = rin + (t.rc - rin) * s / t.D; ph += tau / n / (r ** 1.5 + a); }
    expect(tubeAt(t, t.t0 + tau, a).phi).toBeCloseTo(ph, 6);
  });
  it("light curve: linear rise over D, continuous, half-orbit e-fold, zero after 2 orbits and before birth", () => {
    const a = 0.9, t = tubeOf(9, a), P = 2 * Math.PI * (t.rc ** 1.5 + a);
    expect(tubeAt(t, t.t0 - 1, a).A).toBe(0);
    expect(tubeAt(t, t.t0 + 0.5 * t.D, a).A).toBeCloseTo(0.5, 12);
    expect(tubeAt(t, t.t0 + t.D + 1e-9, a).A).toBeCloseTo(1, 6);
    expect(tubeAt(t, t.t0 + t.D + 0.5 * P, a).A).toBeCloseTo(Math.exp(-1), 10);
    expect(tubeAt(t, t.t0 + t.D + 2 * P + 1e-6, a).A).toBe(0);
    expect(tubeLife(t, a)).toBeCloseTo(t.D + 2 * P, 9);
    let I = 0; const n = 200000, L = tubeLife(t, a);
    for (let i = 0; i < n; i++) I += tubeAt(t, t.t0 + (i + 0.5) / n * L, a).A * L / n;
    expect(I / (t.D / 2 + P * FLARE.cA)).toBeCloseTo(1, 4);
  });
  it("at most three tubes alive at once", () => {
    for (let tt = 0; tt < 3e5; tt += 37) {
      const k0 = Math.floor(tt / FLUX.T); let alive = 0;
      for (let k = k0 - 4; k <= k0 + 1; k++) if (tubeAt(tubeOf(k, 0.9), tt, 0.9).A > 0) alive++;
      expect(alive).toBeLessThanOrEqual(3);
      for (let k = k0 - 6; k < k0 - 2; k++) expect(tubeAt(tubeOf(k, 0.9), tt, 0.9).A).toBe(0); // k-2..k suffice
    }
  });
  it("field by flux conservation and energy (delta s)^2 Phi^2 / (4 pi^2 R r_g)", () => {
    const t = tubeOf(3, sgra.a);
    expect(tubeField(t, 1, E.phi, E.rgCm)).toBeCloseTo(t.depth * E.phi / (Math.PI * (t.R * E.rgCm) ** 2), 6);
    const B = tubeField(t, 1, E.phi, E.rgCm), V = Math.PI * (t.R * E.rgCm) ** 2 * 2 * t.R * E.rgCm;
    expect(tubeEnergy(t, 1, E.phi, E.rgCm) / (B * B * V / (8 * Math.PI))).toBeCloseTo(1, 10);
  });
  it("anchor: zeta = 1e38 / <E> at the Sgr A* preset (~1 % of the tube field energy)", () => {
    expect(flareZeta() * meanTubeEnergy(1, E.phi, E.rgCm)).toBeCloseTo(1e38, -33);
    expect(flareZeta()).toBeGreaterThan(1.0e-2); expect(flareZeta()).toBeLessThan(1.12e-2);
    let m = 0; const n = 20000; for (let k = 0; k < n; k++) m += tubeEnergy(tubeOf(k, sgra.a), 1, E.phi, E.rgCm);
    expect(m / n / meanTubeEnergy(1, E.phi, E.rgCm)).toBeCloseTo(1, 1); // closed form vs the hashed draws (sampling)
  });
  it("injected energy over a tube's life = zeta f E; f = 0 or s = 0 gives nothing (Review Focus 3)", () => {
    const a = sgra.a, t = tubeOf(11, a), L = tubeLife(t, a), z = flareZeta();
    let W = 0; const n = 100000; for (let i = 0; i < n; i++) W += tubePower(t, t.t0 + (i + 0.5) / n * L, a, 1, 1, E.phi, z, E.rgCm) * L / n * E.rgCm / 2.99792458e10;
    expect(W / (z * tubeEnergy(t, 1, E.phi, E.rgCm))).toBeCloseTo(1, 3);
    expect(tubePower(t, t.t0 + 0.3 * L, a, 1, 0, E.phi, z, E.rgCm)).toBe(0);
    expect(tubePower(t, t.t0 + 0.3 * L, a, 0, 1, E.phi, z, E.rgCm)).toBe(0);
  });
});
```

- [ ] **Step 2:** `npx vitest run tests/eruption-spots.test.ts` — FAIL (module missing).
- [ ] **Step 3: Implement** `src/physics/eruption-spots.ts`:

```ts
// Eruption flares (spec 2026-10-04): each horizon-flux eruption ejects a reconnection-heated flux tube that spirals
// out to r_c (5-30 r_g), orbits at the Keplerian rate for up to two orbits and shines as optically thin synchrotron
// (Porth et al. 2021; Ripperda et al. 2022; GRAVITY). Field by flux conservation; radiates zeta x its field energy,
// zeta fixed so a default Sgr A* tube injects a typical NIR flare's 1e38 erg (Yusef-Zadeh et al. 2006). WGSL twins
// in emission-shared.wgsl.
import { FLUX, fluxHash, eruptionTime, eruptionDepth } from "./flux-history";
import { GL6 } from "./jet";
import { iscoRadius } from "./orbits";
import { jetEnergetics, C_CGS } from "./synchrotron";
import { PRESETS } from "./presets";

export const FLARE = { rMin: 5, rMax: 30, sizeFrac: 0.2, lifeOrbits: 2, saltR: 0x5243, saltPhi: 0x5048, anchorErg: 1e38, cA: 0.48452046 } as const;
export interface Tube { k: number; t0: number; D: number; depth: number; rc: number; R: number; phi0: number }
const omegaK = (r: number, a: number) => 1 / (r ** 1.5 + a);
const smooth = (x: number) => { const c = Math.min(1, Math.max(0, x)); return c * c * (3 - 2 * c); };

export function tubeOf(k: number, a: number): Tube {
  const depth = eruptionDepth(k, FLUX.dbar), D = -FLUX.tauD * Math.log(1 - depth);
  const u = fluxHash(k, FLARE.saltR) + 0.5, v = fluxHash(k, FLARE.saltPhi) + 0.5;
  const rc = Math.max(FLARE.rMin + (FLARE.rMax - FLARE.rMin) * u, iscoRadius(a, true) / (1 - 2 * FLARE.sizeFrac));
  return { k, t0: eruptionTime(k), D, depth, rc, R: FLARE.sizeFrac * rc, phi0: 2 * Math.PI * v };
}
export const tubeOrbit = (t: Tube, a: number) => 2 * Math.PI / omegaK(t.rc, a);
export const tubeLife = (t: Tube, a: number) => t.D + FLARE.lifeOrbits * tubeOrbit(t, a);
/** Azimuth gained spiralling from r_in to r (Omega_K integrated along r(tau) linear in tau; GL6). */
function spiralPhase(r: number, rin: number, t: Tube, a: number): number {
  if (r <= rin) return 0;
  const h = 0.5 * (r - rin), m = 0.5 * (r + rin); let s = 0;
  for (const [x, w] of GL6) s += w * omegaK(m + h * x, a);
  return (t.D / (t.rc - rin)) * s * h;
}
export function tubeAt(t: Tube, time: number, a: number): { r: number; phi: number; A: number } {
  const tau = time - t.t0, rin = iscoRadius(a, true);
  if (tau < 0 || tau >= tubeLife(t, a)) return { r: t.rc, phi: t.phi0, A: 0 };
  if (tau < t.D) { const r = rin + (t.rc - rin) * tau / t.D; return { r, phi: t.phi0 + spiralPhase(r, rin, t, a), A: tau / t.D }; }
  const tc = tau - t.D, P = tubeOrbit(t, a);
  return { r: t.rc, phi: t.phi0 + spiralPhase(t.rc, rin, t, a) + omegaK(t.rc, a) * tc, A: Math.exp(-2 * tc / P) * (1 - smooth((tc / P - 1.5) / 0.5)) };
}
/** Top-hat field through the tube cross-section pi R^2 (flux conservation): B = delta s Phi / (pi R^2 r_g^2). */
export function tubeField(t: Tube, s: number, phi: number, rgCm: number): number {
  return (t.depth * s * phi) / (Math.PI * (t.R * rgCm) ** 2);
}
export function tubeEnergy(t: Tube, s: number, phi: number, rgCm: number): number {
  return ((t.depth * s) ** 2 * phi * phi) / (4 * Math.PI ** 2 * t.R * rgCm);
}
export function tubePower(t: Tube, time: number, a: number, s: number, f: number, phi: number, zeta: number, rgCm: number): number {
  if (f === 0 || s === 0) return 0;
  const A = tubeAt(t, time, a).A; if (A === 0) return 0;
  const norm = (t.D / 2 + tubeOrbit(t, a) * FLARE.cA) * rgCm / C_CGS; // integral of A in seconds
  return (zeta * f * tubeEnergy(t, s, phi, rgCm) * A) / norm;
}
export function meanTubeEnergy(s: number, phi: number, rgCm: number): number {
  return ((FLUX.dbar * s) ** 2 * (1 + 1 / 48) * phi * phi * (Math.log(6) / 25)) / (0.8 * Math.PI ** 2 * rgCm);
}
let zetaCache = 0;
/** zeta = anchor / <E> at the Sgr A* preset (s = 1). */
export function flareZeta(): number {
  if (!zetaCache) { const p = PRESETS.find((q) => q.id === "sgra")!, E = jetEnergetics(p.massSun, p.a, p.lambda);
    zetaCache = FLARE.anchorErg / meanTubeEnergy(1, E.phi, E.rgCm); }
  return zetaCache;
}
```

- [ ] **Step 4:** `npx vitest run tests/eruption-spots.test.ts` — PASS (record the printed ζ in the ledger).
- [ ] **Step 5:** `git commit -m "Eruption flares: CPU tube model (birth, orbit, light curve, flux-conserving field, zeta anchor)"`.

---

### Task 2: Emission, shared WGSL and `?parity`

**Files:** Modify `src/render/emission-shared.wgsl`, `src/physics/eruption-spots.ts`, `src/test/parity.browser.ts`;
Create `src/render/flare-parity.wgsl`; Test `tests/eruption-spots.test.ts`, `tests/jet.test.ts` (WGSL constants).

**Produces:** CPU `tubeLight(T: SynchTable, rHit, phiHit, g, time, a, s, f, phi, zeta, rgCm): [number, number, number]`
(observed I_ν per band, cgs); WGSL `fn synchCoeffsJ(lnB: f32, lns: f32, lnq: f32, lnD: f32) -> SynchOut`,
`fn tubeLightJ(rHit, phiHit, g, epoch, rel, a, s, f, Phi, zeta, rgCm: f32) -> vec3<f32>`; constants `FLARE_*`.

- [ ] **Step 1: Failing tests** — append to `tests/eruption-spots.test.ts` (load the table like `tests/synchrotron.test.ts`):

```ts
import { readFileSync } from "node:fs"; import { join } from "node:path";
import { parseTable } from "../src/physics/cyclosynch";
import { tubeLight } from "../src/physics/eruption-spots";
import { jetCoeffs, JET_BANDS_NM, C_CGS as C } from "../src/physics/synchrotron";
const TAB = parseTable(readFileSync(join(__dirname, "../public/synch-table.bin")).buffer.slice(0) as ArrayBuffer);
describe("tube light", () => {
  const a = sgra.a, z = flareZeta(), t = tubeOf(4, a), when = t.t0 + t.D + 0.2 * 2 * Math.PI * (t.rc ** 1.5 + a);
  const at = tubeAt(t, when, a);
  it("bright at the tube centre, Gaussian fall-off, zero far away and with flares off", () => {
    const c = tubeLight(TAB, at.r, at.phi, 1, when, a, 1, 1, E.phi, z, E.rgCm);
    expect(c[1]).toBeGreaterThan(0);
    const off = tubeLight(TAB, at.r + t.R, at.phi, 1, when, a, 1, 1, E.phi, z, E.rgCm);
    expect(off[1] / c[1]).toBeCloseTo(Math.exp(-0.5), 2);
    expect(tubeLight(TAB, at.r + 10 * t.R, at.phi, 1, when, a, 1, 1, E.phi, z, E.rgCm)[1]).toBe(0);
    expect(tubeLight(TAB, at.r, at.phi, 1, when, a, 1, 0, E.phi, z, E.rgCm)).toEqual([0, 0, 0]);
  });
  it("radiated / injected >= 0.4 over the r_c range (outer tubes are not fully fast-cooling; spec corrections 3)", () => {
    for (const k of [1, 2, 3, 4, 5, 6]) {
      const tk = tubeOf(k, a), B = tubeField(tk, 1, E.phi, E.rgCm), ageS = 600 * E.rgCm / C;
      const sc = Math.exp(-20.46682379) * B * B * ageS; let rad = 0; const n = 3000, l0 = Math.log(1e3), l1 = Math.log(1e24);
      for (let i = 0; i < n; i++) { const nu = Math.exp(l0 + (l1 - l0) * (i + 0.5) / n); rad += jetCoeffs(TAB, nu, B, sc, 1, 1)[0] * nu * (l1 - l0) / n; }
      expect(4 * Math.PI * rad / (B * B)).toBeGreaterThan(0.4);
    }
  });
});
```

  and in `tests/jet.test.ts`'s WGSL-constants pairs add
  `["FLARE_RMIN", 5], ["FLARE_RMAX", 30], ["FLARE_SIZE", 0.2], ["FLARE_CA", FLARE.cA], ["FLARE_SALT_R", FLARE.saltR], ["FLARE_SALT_PHI", FLARE.saltPhi]`
  (import `FLARE`).
- [ ] **Step 2:** run both files — FAIL (`tubeLight` missing, `no FLARE_RMIN`).
- [ ] **Step 3: CPU `tubeLight`** (in `eruption-spots.ts`; import `SynchTable` type, `jetCoeffs`, `slabStep`, `JET_BANDS_NM`, `LN_K0`):

```ts
/** Observed I_nu per band (cgs) where a ray hits the disk at (rHit, phiHit) with disk shift g = nu_obs/nu_emit:
 *  optically thin synchrotron of tubes k-2..k alive at `time` (twin: tubeLightJ). */
export function tubeLight(T: SynchTable, rHit: number, phiHit: number, g: number, time: number, a: number, s: number, f: number,
  phi: number, zeta: number, rgCm: number): [number, number, number] {
  const out: [number, number, number] = [0, 0, 0];
  if (f === 0 || s === 0) return out;
  const k0 = Math.floor(time / FLUX.T);
  for (let k = k0 - 2; k <= k0; k++) {
    const t = tubeOf(k, a), p = tubeAt(t, time, a); if (p.A === 0) continue;
    const d2 = rHit * rHit + p.r * p.r - 2 * rHit * p.r * Math.cos(phiHit - p.phi), G = Math.exp(-d2 / (2 * t.R * t.R));
    if (G < 1e-6) continue;
    const B = tubeField(t, s, phi, rgCm), P = tubePower(t, time, a, s, f, phi, zeta, rgCm);
    const q = P / (B * B * 4 * Math.PI * (t.R * rgCm) ** 3), sc = Math.exp(LN_K0) * B * B * (time - t.t0) * rgCm / C_CGS;
    const D = 1 / g, ds = 2 * t.R * rgCm;
    JET_BANDS_NM.forEach((nm, b) => {
      const [jp, al] = jetCoeffs(T, (C_CGS / (nm * 1e-7)) * D, B, Math.max(sc, 1e-13), q, G);
      out[b] += slabStep(0, 0, jp / D ** 3, al, ds)[0];
    });
  }
  return out;
}
```

  (`Math.max(sc, 1e-13)`: the table's s floor is e^{−30} in the jet path; keep the same −30 clamp in ln — WGSL uses
  `max(lns, -30.0)`.)
- [ ] **Step 4: WGSL.** In `emission-shared.wgsl` factor the per-band loop of `synchSampleJ` into

```wgsl
// Per band: j = (nu/nu')^3 j'(nu') and alpha'(nu') of the cooled population at nu' = D nu, field e^lnB, cooling
// depth e^lns, injection e^lnq (shared by the jet and the eruption flares; twin: jetCoeffs).
fn synchCoeffsJ(lnB: f32, lns: f32, lnq: f32, lnD: f32) -> SynchOut {
  var o: SynchOut;
  for (var b = 0; b < 3; b++) {
    let lnx = JET_LNNU[b] + lnD - SYN_LNNUB0 - lnB;
    let t = synchLookupJ(lnx, lns);
    o.j[b] = exp(SYN_LNCJ + lnq + lnB + t.x - 3.0 * lnD);
    o.a[b] = exp(SYN_LNCA + lnq - lnB - 2.0 * lnx + t.y);
  }
  return o;
}
```

  and make `synchSampleJ` end with `return synchCoeffsJ(lnB, lns, log(q0) + log(shape), log(D));` (identical arithmetic).
  Then add (after the flux-history block; constants `FLARE_RMIN = 5.0, FLARE_RMAX = 30.0, FLARE_SIZE = 0.2,
  FLARE_CA = 0.48452046, FLARE_SALT_R = 0x5243u, FLARE_SALT_PHI = 0x5048u`):

```wgsl
fn omegaKJ(r: f32, a: f32) -> f32 { return 1.0 / (pow(r, 1.5) + a); }
// Tube k (k relative to the clock's whole flux periods kE: k = kE + dk). Returns (r, phi, A, tau since birth).
fn tubeAtJ(kE: f32, dk: i32, loc: f32, a: f32) -> vec4<f32> {
  let k = i32(kE) + dk;
  let tau = loc - FLUX_T * (f32(dk) + 0.5 + FLUX_JIT * fluxHashJ(k, FLUX_SALT_T));   // time since birth
  let depth = FLUX_DBAR * (1.0 + FLUX_SPREAD * fluxHashJ(k, FLUX_SALT_D));
  let D = -FLUX_TAUD * log(1.0 - depth);
  let rin = iscoJ(a);
  let rc = max(FLARE_RMIN + (FLARE_RMAX - FLARE_RMIN) * (fluxHashJ(k, FLARE_SALT_R) + 0.5), rin / (1.0 - 2.0 * FLARE_SIZE));
  let phi0 = TWO_PI_E * (fluxHashJ(k, FLARE_SALT_PHI) + 0.5);
  let P = TWO_PI_E / omegaKJ(rc, a);
  if (tau < 0.0 || tau >= D + 2.0 * P) { return vec4<f32>(rc, phi0, 0.0, tau); }
  let r = select(rc, rin + (rc - rin) * tau / D, tau < D);
  // spiral phase to r (GL6 on [rin, r]), plus the circular phase after D
  let h = 0.5 * (r - rin); let m = 0.5 * (r + rin); var xs = GL6_X; var ws = GL6_W; var sp = 0.0;
  for (var i = 0; i < 6; i++) { sp += ws[i] * omegaKJ(m + h * xs[i], a); }
  sp = sp * h * D / (rc - rin);
  let tc = max(tau - D, 0.0);
  let A = select(exp(-2.0 * tc / P) * (1.0 - smoothstepJ(1.5, 2.0, tc / P)), tau / D, tau < D);
  return vec4<f32>(r, phi0 + sp + omegaKJ(rc, a) * tc, A, tau);
}
fn tubeLightJ(rHit: f32, phiHit: f32, g: f32, epoch: f32, rel: f32, a: f32, s: f32, f: f32, Phi: f32, zeta: f32, rgCm: f32) -> vec3<f32> {
  var I = vec3<f32>(0.0);
  if (f == 0.0 || s == 0.0) { return I; }
  let kl = splitPeriodJ(epoch, rel, FLUX_T);
  for (var dk = -2; dk <= 0; dk++) {
    let p = tubeAtJ(kl.x, dk, kl.y, a);
    if (p.z <= 0.0) { continue; }
    let k = i32(kl.x) + dk;
    let depth = FLUX_DBAR * (1.0 + FLUX_SPREAD * fluxHashJ(k, FLUX_SALT_D));
    let rc = max(FLARE_RMIN + (FLARE_RMAX - FLARE_RMIN) * (fluxHashJ(k, FLARE_SALT_R) + 0.5), iscoJ(a) / (1.0 - 2.0 * FLARE_SIZE));
    let R = FLARE_SIZE * rc; let D = -FLUX_TAUD * log(1.0 - depth); let P = TWO_PI_E / omegaKJ(rc, a);
    let d2 = rHit * rHit + p.x * p.x - 2.0 * rHit * p.x * cos(phiHit - p.y);
    let G = exp(-d2 / (2.0 * R * R));
    if (G < 1e-6) { continue; }
    // ln B = ln(delta s Phi) - ln(pi R^2 r_g^2); ln E = 2 ln(delta s Phi) - ln(4 pi^2 R r_g)
    let lnF = log(depth * s * Phi);
    let lnB = lnF - log(PI_E * R * R) - 2.0 * log(rgCm);
    let lnE = 2.0 * lnF - log(4.0 * PI_E * PI_E * R) - log(rgCm);
    let lnNorm = log(0.5 * D + P * FLARE_CA) + log(rgCm) - log(2.99792458e10);
    let lnP = log(zeta * f) + lnE + log(p.z) - lnNorm;                     // injected power (erg/s)
    let lnq = lnP - 2.0 * lnB - log(4.0 * PI_E) - 3.0 * (log(R) + log(rgCm)) + log(G);
    let lns = max(SYN_LNK0 + 2.0 * lnB + log(max(p.w, 1e-6)) + log(rgCm) - log(2.99792458e10), -30.0);
    let o = synchCoeffsJ(lnB, lns, lnq, -log(g));
    var acc: JetOut; acc.I = vec3<f32>(0.0); acc.tau = vec3<f32>(0.0);
    acc = jetSlabJ(acc, o.j, o.a, 2.0 * R * rgCm);
    I += acc.I;
  }
  return I;
}
```

  Helpers needed in `emission-shared.wgsl` if absent: `PI_E` (= 3.14159265…; or use `0.5 * TWO_PI_E`) and `iscoJ(a)`
  (prograde ISCO, Bardeen–Press–Teukolsky; copy the formula from `src/physics/orbits.ts` `iscoRadius`). Check first:
  `grep -n "fn isco\|PI_E" src/render/*.wgsl` — reuse an existing ISCO function if `integrator-shared.wgsl` or
  `shadow-shared.wgsl` has one (do not duplicate; `tests/shader-twins.test.ts` guards duplicates). Note the CPU twin
  passes `G` as `shape` into `jetCoeffs` (q0·shape), the WGSL folds it into `lnq` — the same product.
- [ ] **Step 5: parity.** `src/render/flare-parity.wgsl` (binding 0 in, 1 out, 10 the synch table, as `jet-parity.wgsl`):

```wgsl
// Parity for the eruption flares: tubeLightJ (emission-shared.wgsl) vs tubeLight (eruption-spots.ts).
// Per case: a = (epoch, rel, rHit, phiHit), b = (g, spin, s, f), c = (Phi, zeta, rgCm, 0). Out: ln I per band.
struct FlIn { a: vec4<f32>, b: vec4<f32>, c: vec4<f32> };
@group(0) @binding(0) var<storage, read> flin: array<FlIn>;
@group(0) @binding(1) var<storage, read_write> flout: array<vec4<f32>>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = flin[gid.x];
  let I = tubeLightJ(c.a.z, c.a.w, c.b.x, c.a.x, c.a.y, c.b.y, c.b.z, c.b.w, c.c.x, c.c.y, c.c.z);
  flout[gid.x] = vec4<f32>(log(max(I, vec3<f32>(1e-38))), 0.0);
}
```

  In `parity.browser.ts` add a block (modelled on the flux block): Sgr A* energetics (`jetEnergetics` of the preset),
  ζ = `flareZeta()`; for epochs `[0, 2048 * 7, 2048 * 8000]` and tube indices k relative to the epoch's period
  (k = floor(epoch/1500) + {1, 2, 3}), times inside each tube's life (spiral at 0.5 D, orbit at D + 0.3 P, fade at
  D + 1.6 P), hit points at the tube centre and at 1 R offset, spins 0.3 and 0.94, g ∈ {0.7, 1.3}; rel = time − epoch.
  CPU expectation: `Math.log(Math.max(tubeLight(...)[b], 1e-38))`. Error metric: absolute on ln I, tolerance 2e-3
  (like the jet's logs). Report `flareErr` in the PARITY line and fail when > 2e-3.
- [ ] **Step 6: Run.** CPU tests PASS; dev server; `?parity` PASS with the jet numbers unchanged (maxRelErr 1.654e-4,
  jet |d ln| ≤ 2e-3) and the flare line ≤ 2e-3. Mutation: `FLARE_SIZE = 0.21` → FAIL; revert by exact text.
- [ ] **Step 7:** `git commit -m "Eruption flares: tube light (CPU + WGSL, shared synchCoeffsJ), ?parity flare cases"`.

---

### Task 3: Renderer, uniforms, UI; remove the illustrative hot spots

**Files:** `src/render/uniforms.ts`, `raytrace.wgsl`, `present.wgsl`, `gpu.ts`, `src/physics/emission.ts`,
`src/physics/synchrotron.ts` (`jetUniforms` returns `phi`), `src/main.ts`, `index.html`, `src/test/scenes.ts`,
`scripts/bench.mjs`, `scripts/shot-delay.mjs`, `scripts/verify-gpu.mjs`, tests `uniforms.test.ts`, `emission.test.ts`
(hot-spot tests removed), `cache-plan.test.ts` if it references `nSpots`.

- [ ] **Step 1: Failing tests.** `tests/uniforms.test.ts`: size 160; `f[18]` = flareStrength (float), floats 36–37
  = flarePhi, flareZeta (assert their byte offsets 144 and 148). `tests/emission.test.ts`: delete hot-spot tests; add a
  test that `diskShadeFactors` no longer takes spots (its signature without the `spots` argument; grey = breathe).
  Run — FAIL.
- [ ] **Step 2: Uniforms.** In `uniforms.ts` replace `nSpots` by `flareStrength` (f[18]), add `flarePhi`, `flareZeta`
  (f[36], f[37]), `UNIFORM_FLOATS = 30`; update the layout comment. In `raytrace.wgsl` and `present.wgsl` structs:
  `breatheAmp: f32, flareStrength: f32,` and `timeEpoch: f32, flarePhi: f32, flareZeta: f32,`.
- [ ] **Step 3: Remove hot spots.** Delete `hotspots` (binding 4) and `hotspotFieldE` from `raytrace.wgsl`;
  `diskShadeFactorsE` returns `vec2(tempScale, max(0.0, breathe))`. `gpu.ts`: remove the hot-spot buffer,
  `uploadHotSpots` and its bind-group entries (grep `hotspot|binding: 4`). `emission.ts`: remove `HotSpot`,
  `hotspotField` and the `spots` parameter of `diskShadeFactors`; fix callers (grep).
- [ ] **Step 4: Tube light in the disk.** In `shadeDisk`, after computing the disk colour:

```wgsl
  let tube = tubeLightJ(rHit, phiHit, g, U.timeEpoch, tRel, a, U.fluxVar, U.flareStrength, U.flarePhi, U.flareZeta, U.rgCm);
  return sampleColor(Tobs) * U.lumNorm * E.y + U.lumNorm * max(JET_BAND_M * tube, vec3<f32>(0.0));
```

- [ ] **Step 5: CPU wiring.** `jetUniforms` returns `phi: E.phi` too. `main.ts`: state `flareScale` default 1.0;
  uniforms `flareStrength: state.flareScale, flarePhi: jetU.phi, flareZeta: flareZeta()`; remove `baseSpots`/
  `packSpots`/`uploadHotSpots`; the flare slider handler just sets the state and `reset()`. `index.html`: label
  "Eruption flares", `max="3"`, `value="1"`, value text `1.0×`, tooltip: "Flux tubes thrown out by each horizon-flux
  eruption: they spiral out to 5–30 r_g, orbit for up to two orbits and shine as synchrotron. 1× radiates ~1 % of each
  tube's magnetic energy, so a Sgr A* tube emits a typical observed flare (~10³⁸ erg). Seen only where rays hit the
  disk." The share-link key `hs` keeps working (same control id `flare`; its default changes to 1).
  `scenes.ts`/`bench.mjs`/`shot-delay.mjs`: remove `uploadHotSpots`/`nSpots`; pass `flareStrength`, `flarePhi`,
  `flareZeta` (Sgr A*-like values from `jetUniforms` of their object). In `scenes.ts` set the "delay" scene's time to
  `tubeOf(1, a).t0 + tubeOf(1, a).D + 0.25 × orbit` so a golden scene shows a live tube; `shot-delay.mjs` picks the
  first k with r_c < 10 and its time `t0 + D + 0.2 P`.
- [ ] **Step 6: App check** in `verify-gpu.mjs` (after the share-link check): open `/#p=sgra&t=5&hs=0` and
  `/#p=sgra&t=5` in turn; for each, sample the canvas mean brightness right of the panel every 0.5 s for 24 s
  (≥ one eruption at 100 M/s); PASS if max(flares on) > max(flares off) × 1.02 and no console warnings.
- [ ] **Step 7: Gates.** `npm test`, `npm run build`, `RECORD_GOLDEN=1 npm run verify:gpu` — all PASS; `?cachecheck`
  0.00e+0. Golden: every scene changes by design only if a tube is alive there (the delay scene) or by compiler
  rescheduling (verify with the jet-free + tube-free kernel on main vs branch if jet-off changed: stub both `jetStep`
  and `tubeLightJ` to return zero).
- [ ] **Step 8:** `git commit -m "Renderer: eruption flares in the disk shading (replace the illustrative hot spots), uniforms 160 B, Eruption flares slider, app check"`.

---

### Task 4: Visual check, docs, ship

- [ ] **Step 1:** Frames: Sgr A* preset (`#p=sgra&t=5`), one per second for 30 s from a scratch directory (adapt
  `scripts/shot-flux.mjs` to the preset hash); contact sheet; confirm a patch appears near the inner disk, moves out,
  orbits (Doppler-bright on the approaching side) and fades; and `node scripts/shot-delay.mjs` echo pair.
- [ ] **Step 2:** README section "Eruption flares" (physics, sources, ζ = 1.06 %, the radiated/injected range, the
  Porth energy comparison, the surface approximation and the Keplerian choice, checks, golden hashes); the README's
  old "Flares (illustrative)" text marked replaced; ROADMAP follow-up 2 done; spec status.
- [ ] **Step 3:** Final whole-branch review (fresh reviewer, most capable model); fix pass; then fast-forward main,
  push, `vercel deploy --prod` (PowerShell), verify the live bundle contains `tubeLightJ`.
