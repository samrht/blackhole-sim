# Relativistic Black Hole Accretion Disk & Imaging — Physical & Mathematical Specification

**Date:** 2026-05-30
**Status:** Verified (independent multi-agent equation check, 2026-05-30) — brainstorming deliverable
**Scope:** Architecture-agnostic physics/math spec. Implementation tier (1/2/3) chosen separately.

## Decisions captured during brainstorming

- **Deliverable:** a rigorous, corrected, equation-complete specification — *not* a committed architecture. It fixes the physics conflations in the original prompt and states every governing equation exactly.
- **Depth:** "complete & rigorous" — equations stated exactly, all symbols/units defined, key results summarized + cited; not every derivation reproduced.
- **Tier-3 (GRMHD):** governing equations stated precisely, numerical scheme cited only.
- **Worked anchor:** **M87\*** ($M\approx6.5\times10^{9}\,M_\odot$, $a_*\approx0.9$), with a stellar-mass row in the parameter table.
- **Target machine (for later implementation):** Windows, NVIDIA RTX 3050 Laptop (4 GB). Implies **Tier 1** is the feasible build target.

## Corrections to the original prompt (the heart of "make it accurate")

1. **α-disk ⇎ MRI.** The Shakura–Sunyaev α prescription is a *parametrization that stands in for* MRI-driven turbulent transport. You do **not** add MRI on top of an α-disk; α *is* the MRI proxy. Resolving MRI requires solving the MHD equations (Tier 3).
2. **Shakura–Sunyaev → Novikov–Thorne.** The relativistic disk flux (Page–Thorne) **vanishes at the ISCO** (zero-torque inner boundary) and **peaks just outside it**. $T\propto r^{-3/4}$ is only the Newtonian asymptotic tail; $T$ does not diverge at the ISCO.
3. **Beaming + redshift are one quantity.** Doppler beaming and gravitational redshift are unified in the single redshift factor $g=\nu_{\rm obs}/\nu_{\rm emit}$ applied through the Lorentz invariant $I_\nu/\nu^3$.
4. **$3M$ / $\sqrt{27}\,M$ are Schwarzschild-only.** Photon sphere $r_{\rm ph}=3M$ and shadow radius $\sqrt{27}\,M$ hold for $a=0$; for Kerr they become spin- and inclination-dependent curves.
5. **Blandford–Znajek is emergent (GRMHD) or prescribed.** In Tier 1/2 the jet is prescribed (BZ-scaled power + parametric field + synchrotron electrons); only Tier 3 launches it self-consistently.
6. **Temperature ∝ $M^{-1/4}$.** The prompt's "$10^7$ K soft-X-ray core" is a *stellar-mass* BH number. At fixed Eddington ratio $T_{\rm peak}\propto(\dot M/M^2)^{1/4}\propto M^{-1/4}$, so a supermassive disk (M87\*) peaks in the **UV** ($\sim10^5$ K).
7. **Color from physics, not a hue ramp.** Map temperature → RGB via the Planck spectrum → CIE color-matching functions → sRGB, so "blue-white core → red edge" is a physical consequence of the blackbody locus.
8. **Inclination convention.** "15–20° above the disk plane" ⇒ polar angle $\theta_{\rm obs}=90^\circ-(15\text{–}20)^\circ$, i.e. inclination $i\approx70\text{–}75^\circ$ from the spin axis (near edge-on — the dramatic warped look).

---

## 0. Conventions & units

- Geometrized units $G=c=1$; mass $M$ is the length/time scale. Lengths quoted in $M$ (i.e. $GM/c^2$), times in $GM/c^3$.
- Signature $(-,+,+,+)$. Indices $\mu,\nu\in\{0,1,2,3\}=(t,r,\theta,\phi)$. Einstein summation.
- Spin: angular momentum $J$, $a\equiv J/M$ (length), dimensionless $a_*\equiv a/M=Jc/(GM^2)\in[0,1]$ (extremal at 1).
- Gravitational radius $r_g\equiv GM/c^2$. Useful: $r_g\approx1.477\,\mathrm{km}\,(M/M_\odot)$; light-crossing $r_g/c\approx4.93\,\mu\mathrm{s}\,(M/M_\odot)$.
- M87\* check: $M=6.5\times10^9 M_\odot\Rightarrow r_g\approx9.6\times10^9\,\mathrm{km}\approx64\,\mathrm{AU}$.
- Eddington luminosity $L_{\rm Edd}=4\pi GMm_pc/\sigma_T\approx1.26\times10^{38}\,(M/M_\odot)\,\mathrm{erg\,s^{-1}}$; Eddington rate $\dot M_{\rm Edd}=L_{\rm Edd}/(\eta c^2)$; Eddington ratio $\dot m=\dot M/\dot M_{\rm Edd}$. Radiative efficiency $\eta=1-\tilde E_{\rm ISCO}$ (§3).

---

## 1. Spacetime — the Kerr metric

**Boyer–Lindquist line element:**
```
ds² = −(1 − 2Mr/Σ) dt²
      − (4 M a r sin²θ / Σ) dt dφ
      + (Σ/Δ) dr²
      + Σ dθ²
      + ( r² + a² + 2 M a² r sin²θ / Σ ) sin²θ dφ²
```
with
```
Σ ≡ r² + a² cos²θ ,   Δ ≡ r² − 2Mr + a² ,   A ≡ (r²+a²)² − a² Δ sin²θ
```

**Inverse metric** (for the geodesic Hamiltonian $H=\tfrac12 g^{\mu\nu}p_\mu p_\nu$):
```
g^{tt}  = −A / (Σ Δ)
g^{tφ}  = −2 M a r / (Σ Δ)
g^{rr}  =  Δ / Σ
g^{θθ}  =  1 / Σ
g^{φφ}  =  (Δ − a² sin²θ) / (Σ Δ sin²θ)
```

**Determinant:** $\sqrt{-g}=\Sigma\sin\theta$ (equals $r^2$ at the equator). Distinct from Page–Thorne's reduced "$\sqrt{-g}=r$" in §4, which is their $(t,r,\phi)$ equatorial convention.

**Horizons** ($\Delta=0$): $r_\pm = M \pm \sqrt{M^2 - a^2}$. Event horizon $r_+$.

**Ergosphere / static limit** ($g_{tt}=0$): $r_{\rm ergo}(\theta)=M+\sqrt{M^2-a^2\cos^2\theta}$. Between $r_+$ and $r_{\rm ergo}$ no static observer can exist (frame dragging forces corotation).

**Frame dragging (Lense–Thirring):** the ZAMO/LNRF angular velocity
```
ω(r,θ) = −g_{tφ}/g_{φφ} = 2 M a r / A
```
At the horizon $\omega\to\Omega_H = a/(2 M r_+) = a/(r_+^2+a^2)$. Differential $\omega(r,\theta)$ drives differential precession of non-equatorial orbits.

**Horizon-penetrating coordinates:** BL is singular at $r_+$. For integration across the horizon (Tier 3, jet base) use **Kerr–Schild**, $g_{\mu\nu}=\eta_{\mu\nu}+f\,l_\mu l_\nu$ with $f=2Mr^3/(r^4+a^2z^2)$ and $l_\mu$ the principal null congruence (regular at $r_+$).

---

## 2. Geodesics & conserved quantities

**Hamiltonian** (affine parameter $\lambda$):
```
H = ½ g^{μν} p_μ p_ν = −½ μ²        (μ = rest mass: μ=0 photons, μ=1 timelike)
dx^μ/dλ =  ∂H/∂p_μ = g^{μν} p_ν
dp_μ/dλ = −∂H/∂x^μ = −½ (∂_μ g^{αβ}) p_α p_β
```

**Conserved quantities** (Killing vectors $\partial_t,\partial_\phi$ + Carter separability):
```
E   = −p_t                      (energy)
L_z =  p_φ                      (axial angular momentum)
Q   =  p_θ² + cos²θ [ a²(μ² − E²) + L_z²/sin²θ ]      (Carter constant)
```
(Alternative Carter constant $K=Q+(L_z-aE)^2$.)

**Separated first-order form** (Carter 1968):
```
Σ dr/dλ = ± √R(r),   R(r) = [E(r²+a²) − a L_z]² − Δ[ μ²r² + (L_z − aE)² + Q ]
Σ dθ/dλ = ± √Θ(θ),   Θ(θ) = Q − cos²θ[ a²(μ² − E²) + L_z²/sin²θ ]
Σ dφ/dλ = −(aE − L_z/sin²θ) + (a/Δ)[E(r²+a²) − a L_z]
Σ dt/dλ = −a(aE sin²θ − L_z) + ((r²+a²)/Δ)[E(r²+a²) − a L_z]
```
Photons ($\mu=0$) depend only on two impact parameters $\xi=L_z/E$, $\eta=Q/E^2$. **Numerically** integrate the Hamiltonian form (no $\pm$ sign bookkeeping at turning points).

---

## 3. Circular orbits, ISCO & disk kinematics

**Relativistic Keplerian angular velocity** (equatorial circular geodesics, observed at infinity):
```
Ω(r) = ± M^{1/2} / ( r^{3/2} ± a M^{1/2} )      (+ prograde, − retrograde)
```

**Specific energy & angular momentum** of equatorial circular orbits (Bardeen–Press–Teukolsky 1972):
```
Ẽ(r) = ( r^{3/2} − 2M r^{1/2} ± a M^{1/2} ) / ( r^{3/4} ( r^{3/2} − 3M r^{1/2} ± 2a M^{1/2} )^{1/2} )
L̃(r) = ± M^{1/2}( r² ∓ 2a M^{1/2} r^{1/2} + a² ) / ( r^{3/4}( r^{3/2} − 3M r^{1/2} ± 2a M^{1/2} )^{1/2} )
```

**ISCO** (Bardeen–Press–Teukolsky 1972):
```
Z1 = 1 + (1 − a_*²)^{1/3} [ (1 + a_*)^{1/3} + (1 − a_*)^{1/3} ]
Z2 = ( 3 a_*² + Z1² )^{1/2}
r_isco/M = 3 + Z2 ∓ √[ (3 − Z1)(3 + Z1 + 2 Z2) ]      (− prograde, + retrograde)
```
Checks: $a_*=0\to6M$; $a_*=1$ prograde $\to M$; retrograde $\to9M$.

**Other radii:** photon orbit $r_{\rm ph}=2M\{1+\cos[\tfrac23\arccos(\mp a_*)]\}$ ($a=0\to3M$); marginally bound $r_{\rm mb}=2M\mp a+2M^{1/2}(M\mp a)^{1/2}$ ($a=0\to4M$).

**Inner boundary:** matter at $r\le r_{\rm isco}$ is unstable and **plunges** (free-fall geodesic, conserving $\tilde E,\tilde L$ from the ISCO). Disk inner edge = $r_{\rm isco}$ with near-zero torque.

**Radiative efficiency:** $\eta = 1 - \tilde E_{\rm ISCO}$. $a_*=0\Rightarrow\tilde E=\sqrt{8/9}\Rightarrow\eta\approx0.057$; $a_*=1\Rightarrow\tilde E=1/\sqrt3\Rightarrow\eta\approx0.42$. (Thin-disk values; near-extremal spin slightly lowers the *observed* efficiency via photon capture — Thorne 1974.)

---

## 4. Disk structure — α-disk corrected to Novikov–Thorne

**Newtonian Shakura–Sunyaev baseline** (1973), thin disk $H\ll r$:
```
viscous stress      t_{rφ} = α P            (α ≈ 0.01–0.3)
kinematic viscosity ν = α c_s H
surface density Σ = ∫ ρ dz
Σ evolution         ∂Σ/∂t = (3/r) ∂/∂r [ √r ∂/∂r ( ν Σ √r ) ]    (viscous diffusion)
dissipation/area    D(r) = (9/8) ν Σ Ω²
```

**Relativistic correction — Novikov–Thorne (1973) / Page–Thorne (1974):** time-averaged flux from one face,
```
F(r) = (Ṁ / 4π√(−g)) · ( −Ω_{,r} / (Ẽ − ΩL̃)² ) · ∫_{r_isco}^{r} (Ẽ − ΩL̃) L̃_{,r} dr′
```
with $\Omega,\tilde E,\tilde L$ from §3 and $\sqrt{-g}=r$ in the equatorial reduction. **Zero-torque boundary** ⇒ integral starts at $r_{\rm isco}$ and $F\to0$ there. Newtonian limit:
```
F(r) → (3 G M Ṁ / 8π r³) [ 1 − (r_in/r)^{1/2} ]   ⇒  T ∝ r^{−3/4} far out
```

**Effective temperature:** $\sigma_{\rm SB}T_{\rm eff}^4=F(r)\Rightarrow T_{\rm eff}(r)=[F(r)/\sigma_{\rm SB}]^{1/4}$. Peaks **just outside** ISCO (not at it).

> **MRI clarification (boxed).** $\alpha$ parametrizes turbulent transport; the **MRI** (Balbus–Hawley 1991) is its physical origin. You do not add MRI to an α-disk. Fastest-growing mode $\lambda_{\rm MRI}=2\pi v_A/\Omega$ ($v_A=B/\sqrt{4\pi\rho}$), growth rate $\sim\tfrac34\Omega$; resolving it numerically requires quality factor $Q_{\rm MRI}=\lambda_{\rm MRI}/\Delta x\gtrsim6\text{–}10$ → **Tier-3** upgrade.

---

## 5. Radiation & thermodynamics

- **Local emission** (optically thick ⇒ quasi-blackbody at $T_{\rm eff}$): Planck intensity
  ```
  B_ν(T) = (2hν³/c²) / ( exp(hν/k_B T) − 1 )
  ```
- **Spectral hardening:** $T_{\rm col}=f_{\rm col}T_{\rm eff}$, $f_{\rm col}\approx1.7\text{–}2.0$ (optional).
- **Mass scaling:** $T_{\rm eff}\propto(\dot M/M^2)^{1/4}$; at fixed $\dot m$, $\dot M\propto M\Rightarrow T_{\rm peak}\propto M^{-1/4}$. Stellar-mass ($\sim10\,M_\odot$): $T_{\rm in}\sim10^7$ K (soft X-ray). SMBH (M87\*): $T_{\rm in}\sim10^5$ K (UV).
- **Wien peak:** $\nu_{\rm peak}\approx5.88\times10^{10}\,(T/\mathrm K)\ \mathrm{Hz}$.
- **Physically-based color:** emitted spectrum → CIE 1931 tristimulus $X=\int B_\lambda\bar x\,d\lambda$ (and $Y,Z$) → linear sRGB via the standard $3\times3$ matrix → exposure + tonemap (§8) → gamma. Yields the blackbody color locus (blue-white → yellow → orange → red) physically.

---

## 6. Relativistic optics — backward ray tracing

**Camera:** distant observer at $(r_{\rm obs}\gg M,\ \theta_{\rm obs}=i)$, $i\approx72^\circ$ (15–20° above the plane). Screen pixel ↔ incoming photon defined in the observer's orthonormal tetrad (ZAMO tetrad at $r_{\rm obs}$). Apparent celestial coordinates (Bardeen 1973), large $r$:
```
α = −ξ / sin i
β = ± √( η + a² cos²i − ξ² cot²i )
```
*Implementation: the minus sign in $\alpha=-\xi/\sin i$ is load-bearing — dropping it mirrors the image left–right and inverts the Doppler bright/dim sides. For the Kerr critical curve only the arc with $\eta\ge0$ is the physical shadow boundary.*

**Backward integration:** shoot each pixel's photon into the scene, integrate the §2 Hamiltonian geodesics (adaptive RK). Terminate on:
- (a) disk-plane crossing $\theta=\pi/2$ within $[r_{\rm isco},r_{\rm out}]$ → record hit, evaluate emission;
- (b) $r\to r_+$ (horizon) → **shadow** (black);
- (c) escape $r>r_{\rm obs}$ → background (lensed starfield).
Multiple crossings → primary + higher-order images.

**Redshift factor $g$** (gravitational + Doppler in one):
```
g ≡ ν_obs/ν_emit = (p_μ u^μ)_obs / (p_μ u^μ)_emit
```
For observer at infinity and a disk element on a circular orbit (4-velocity $u^\mu=u^t(1,0,0,\Omega)$, $u^t=1/\sqrt{-(g_{tt}+2\Omega g_{tφ}+\Omega^2 g_{φφ})}$):
```
g = √( −(g_{tt} + 2Ω g_{tφ} + Ω² g_{φφ}) ) / ( 1 − Ω ξ ),   ξ = L_z/E
```
Approaching side $g>1$ (blue, bright); receding $g<1$ (red, dim).

**Radiative transfer invariant** (Liouville): $I_\nu/\nu^3$ is conserved along a ray in vacuum, so
```
I_ν^obs(ν_obs) = g³ I_ν^emit(ν_obs/g) = g³ B_{ν_obs/g}(T_emit)
I_bol^obs = g⁴ I_bol^emit            (⇒ relativistic beaming)
```
With absorption/emission (Tier 2/3), the covariant transfer equation:
```
d/dλ ( I_ν / ν³ ) = ( j_ν / ν² ) − ( ν α_ν )( I_ν / ν³ )
```
Lorentz invariants (Rybicki & Lightman §4.9): $I_\nu/\nu^3$, $j_\nu/\nu^2$, and $\nu\alpha_\nu$. The absorption coefficient is $\nu\alpha_\nu$ because optical depth $d\tau=\alpha_\nu\,ds=\nu\alpha_\nu\,d\lambda$ with $ds=\nu\,d\lambda$ along the ray.

**Shadow & photon ring:**
- Schwarzschild: photon sphere $r_{\rm ph}=3M$, critical impact parameter $b_c=3\sqrt3\,M=\sqrt{27}\,M\approx5.196\,M$ = apparent shadow radius.
- Kerr critical curve (unstable spherical photon orbits, parameter $r$; Bardeen 1973):
  ```
  ξ(r) = [ r²(3M − r) − a²(r + M) ] / [ a (r − M) ]
  η(r) = r³ [ 4 a² M − r (r − 3M)² ] / [ a² (r − M)² ]
  ```
  mapped to the screen via the $(\alpha,\beta)$ relations above → non-circular, spin/inclination-dependent shadow.
- **Higher-order images / photon ring:** photons winding $n=1,2,\dots$ times produce self-similar subrings converging on the critical curve, each demagnified by $e^{-\gamma}$ with $\gamma$ the Lyapunov exponent of the nearly-bound orbit (Gralla–Holz–Wald 2019; Johnson et al. 2020). Disk light from behind the hole appears as these secondary/tertiary arcs over/under the shadow (the Gargantua look).

---

## 7. Jets — Blandford–Znajek (prescribed vs emergent)

**BZ power** (Blandford–Znajek 1977):
```
P_BZ = (κ / 4π) Φ² Ω_H²            Ω_H = a/(2 M r_+)
```
Φ = magnetic flux threading one horizon hemisphere. The robust result is the scaling $P_{BZ}\propto\Phi^2\Omega_H^2$; $\kappa$ is a **geometry-dependent** prefactor of order $0.01$–$0.1$: $\kappa\approx0.053\approx1/(6\pi)$ for a split-monopole field (Tchekhovskoy et al. 2010 simulations), while BZ77's original slow-spin perturbative value and parabolic-field geometries differ by an $\mathcal{O}(1)$ factor. Higher-spin corrections: $P_{BZ}\propto\Phi^2\Omega_H^2(1+\alpha_2(\Omega_H M)^2+\dots)$. **MAD** flux ceiling $\phi_{\rm BH}\equiv\Phi/\sqrt{\dot M r_g^2 c}\approx50$ (Tchekhovskoy 2011) sets max jet power.

**Prescribed jet (Tier 1/2):** parabolic/conical field geometry; bulk Lorentz factor $\Gamma(z)$ → $\Gamma_\infty\sim$ few–tens; opening angle $\theta_j$; nonthermal electrons $N(\gamma)d\gamma=N_0\gamma^{-p}d\gamma$, $\gamma\in[\gamma_{\min},\gamma_{\max}]$, $p\approx2\text{–}3$.

**Synchrotron emission:**
```
single-electron critical freq:  ν_c = (3/2) γ² (eB sinα)/(2π m_e c)
power-law emissivity:           j_ν ∝ N_0 B^{(p+1)/2} ν^{−(p−1)/2}     (spectral index s=(p−1)/2)
self-absorption:                α_ν ∝ ν^{−(p+4)/2};  optically thick I_ν ∝ ν^{5/2}
jet Doppler factor:             δ = 1/[ Γ(1 − β cosθ_obs) ];  observed flux ∝ δ^{3+s}
```
Emissivities feed the §6 covariant transfer (Tier-3 imaging = GRRT, e.g. ipole/RAPTOR).

---

## 8. Visualization pipeline

- **Geometry:** $i\approx72^\circ$; $r_{\rm obs}\sim10^2\text{–}10^3 M$; FOV framing $\sim\pm20\text{–}30\,M$.
- **Background:** textured celestial sphere, **lensed by the same geodesics** → background-star lensing / Einstein ring.
- **HDR accumulation:** accumulate physical (spectral) radiance per pixel; dynamic range spans many decades (photon ring/jet base ≫ outer disk).
- **Tonemapping (exact operators):**
  ```
  Reinhard (extended):  L_d = L (1 + L/L_white²) / (1 + L)
  ACES (Narkowicz fit): f(x) = clamp01( x(2.51x + 0.03) / ( x(2.43x + 0.59) + 0.14 ) )   per channel, linear → sRGB OETF
  ```
- **Bloom/glare** on supra-threshold pixels (photon ring, jet) — aesthetic glow.
- **Anti-aliasing:** multiple jittered rays/pixel + accumulation (essential at the near-discontinuous shadow edge and photon ring).
- **Time evolution:** differential rotation from $\Omega(r)$; coordinate-time orbital period $T(r)=2\pi(r^{3/2}\pm aM^{1/2})/M^{1/2}$ (inner annuli sweep faster). **Flares:** phenomenological orbiting hot-spots / stochastic α-fluctuations advected at local $\Omega$ (Tier 1/2); emergent from MRI turbulence (Tier 3).

---

## 9. Tier map & numerical methods

| Subsystem | Tier 1 (image) | Tier 2 (hybrid) | Tier 3 (GRMHD) |
|---|---|---|---|
| Spacetime | exact Kerr (analytic) | exact Kerr | exact Kerr (Kerr–Schild) |
| Disk structure | Novikov–Thorne (static) | evolving α-disk (Σ diffusion) | emergent from MHD |
| MRI/turbulence | none (α implicit) | phenomenological flares | emergent, resolved |
| Jet | prescribed BZ-scaled | prescribed | emergent BZ |
| Emission | thermal BB + $g$-factor | BB + synchrotron | full GRRT |
| Optics | exact null geodesics | exact | exact (+polarized GRRT) |
| Cost | single GPU, ~real-time | single GPU | multi-GPU/cluster, offline |

**GRMHD governing equations (Tier 3, stated; scheme cited):**
```
∇_μ T^{μν} = 0,  T^{μν} = (ρ + u_g + p + b²) u^μ u^ν + (p + b²/2) g^{μν} − b^μ b^ν
∇_μ (ρ u^μ) = 0
∇_μ (*F^{μν}) = 0      (induction; ideal MHD)
p = (Γ − 1) u_g       (gamma-law EOS)
```
Scheme (cited): conservative finite-volume, HLL/HLLC Riemann solver, constrained transport for $\nabla\!\cdot\!B=0$, primitive recovery — HARM (Gammie et al. 2003), BHAC, KORAL, Athena++, H-AMR.

**Tier-1 numerics (the build core):**
- Geodesics: Hamiltonian ODEs, adaptive Dormand–Prince RK45 with error control; smaller steps near periapsis/horizon.
- Avoid BL horizon singularity: terminate at $r=r_+(1+\varepsilon)$; use Kerr–Schild for near-horizon rays.
- **Verification tests:**
  1. $a=0$ shadow radius $=\sqrt{27}\,M\approx5.196\,M$.
  2. ISCO: $a=0\to6M$; $a_*=1$ pro $\to M$, retro $\to9M$.
  3. $E,L_z,Q$ conserved to tolerance; $H\approx0$ (null) drift bounded.
  4. Flat-space limit ($M\to0$): straight rays.
  5. Gravitational redshift of a static emitter: $g=\sqrt{1-2M/r}$ (Schwarzschild) matches analytic.
  6. Novikov–Thorne flux → $T\propto r^{-3/4}$ far out.

**Master parameters:** $M$ (and SI mass), $a_*$, $\dot m$ (or $\dot M$), $\alpha$, $r_{\rm out}$, $i$, $r_{\rm obs}$, resolution, samples/pixel, tonemap operator, exposure.

| Example | $M$ | $a_*$ | $\dot m$ | $T_{\rm in}$ | Peak band |
|---|---|---|---|---|---|
| M87\* (anchor) | $6.5\times10^9 M_\odot$ | 0.9 | $\sim10^{-5}$ | $\sim10^5$ K | UV |
| Stellar BH (e.g. Cyg X-1) | $\sim15 M_\odot$ | 0.9 | $\sim0.1$ | $\sim10^7$ K | soft X-ray |

---

## 10. References (selected; each anchors a result above)

- Kerr (1963) — metric. Boyer & Lindquist (1967) — coordinates. Carter (1968) — separability/Carter constant.
- Bardeen, Press & Teukolsky (1972) — circular orbits, ISCO. Bardeen (1973) — photon orbits, shadow/critical curve. Cunningham (1975) — disk transfer functions / $g$-factor imaging.
- Shakura & Sunyaev (1973) — α-disk. Novikov & Thorne (1973); Page & Thorne (1974) — relativistic thin-disk flux. Balbus & Hawley (1991) — MRI.
- Blandford & Znajek (1977) — jet mechanism. Tchekhovskoy, Narayan & McKinney (2010, 2011) — MAD, spin scaling.
- Luminet (1979) — first BH disk image. Gammie, McKinney & Tóth (2003) — HARM GRMHD.
- James, von Tunzelmann, Franklin & Thorne (2015) — Interstellar/Gargantua (DNGR).
- Gralla, Holz & Wald (2019); Johnson et al. (2020) — photon-ring substructure.
- EHT Collaboration (2019, M87\*; 2022, Sgr A\*); Mościbrodzka & Gammie (ipole); Bronzwaer et al. (RAPTOR) — GRRT imaging.

---

## 11. Verification log (2026-05-30)

All governing equations independently checked by five parallel verification agents against primary sources (BPT 1972; Bardeen 1973; Carter 1968; Page–Thorne 1974; Shakura–Sunyaev 1973; Balbus–Hawley 1991; Blandford–Znajek 1977; Tchekhovskoy+ 2010/2011; Rybicki–Lightman; MTW; Chandrasekhar; Narkowicz 2016; Reinhard+ 2002; CIE 1931 / IEC 61966-2-1).

- **Spacetime (§1), orbits/ISCO (§3), geodesics & shadow (§2, §6):** all checks PASS. Schwarzschild limit $\xi^2+\eta\to27M^2$ confirmed analytically and numerically; ISCO $6M/M/9M$ and $r_{\rm ph}\,3M/M/4M$ confirmed.
- **Disk & radiation (§4, §5):** all PASS. Page–Thorne flux structure and zero-torque ISCO confirmed; Wien constant $5.879\times10^{10}$ Hz/K confirmed; $T_{\rm peak}\propto M^{-1/4}$ scaling confirmed.
- **Transfer equation (§6):** absorption coefficient confirmed as $\nu\alpha_\nu$ (the invariant), via $d\tau=\alpha_\nu\,ds=\nu\alpha_\nu\,d\lambda$ — an agent-proposed change to $\alpha_\nu/\nu$ was checked and rejected.
- **Jets & rendering (§7, §8):** ACES constants, Reinhard operator, synchrotron indices $j_\nu\propto B^{(p+1)/2}\nu^{-(p-1)/2}$, $\alpha_\nu\propto\nu^{-(p+4)/2}$, Doppler $\delta^{3+s}$/$\delta^{2+s}$, and CIE→sRGB all PASS. BZ $\kappa$ flagged as geometry-dependent (now caveated).
- **Refinements applied:** $\sigma_T$ typo; $\sqrt{-g}=\Sigma\sin\theta$ note; photon-capture efficiency caveat; image-mirror sign warning; BZ $\kappa$ geometry caveat; invariant definitions for the transfer equation.

**Implementation pitfalls carried forward to the plan:** (1) switch from Boyer–Lindquist to Kerr–Schild/ingoing-Kerr coordinates to integrate across $r_+$ (BL is singular at $\Delta=0$); (2) guard $a^2\le M^2$; (3) the $\alpha=-\xi/\sin i$ sign; (4) supersample the near-discontinuous shadow edge and photon ring.
