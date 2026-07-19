# Lensed Sky Background — Design

**Status:** approved (brainstorm) — pending spec review
**Date:** 2026-07-19
**Depends on:** Tier 1 (geodesic tracer), Tier 2A (Living Disk), Tier 2B (Relativistic Jet) — all shipped on `main`.

## 1. Goal

Replace the sparse procedural starfield behind the black hole with a **real, photographic
equirectangular sky panorama**, sampled along each escaped ray's already-bent asymptotic
direction. The gravitational lensing is *already* physically applied (the ray direction is the
output of the geodesic integration); this feature only enriches the **content** of the sky so the
warping reads clearly — most notably the **Einstein ring**, which emerges for free when the bright
Milky-Way band wraps around the shadow.

Non-goal: changing any geodesic / disk / jet physics. This is a background-shading change only.

## 2. Asset

- **Source:** ESO `eso0932a`, "The Milky Way panorama" — credit **ESO/S. Brunier**, license
  **CC BY 4.0**. Native is a full-sky equirectangular (2:1) projection.
  Fetch URL: `https://cdn.eso.org/images/large/eso0932a.jpg` (~8 MB) or
  `.../publicationjpg/eso0932a.jpg` (~5 MB).
- **Processing:** downscale to **4096×2048** (power-of-two, exact 2:1) and commit as
  `public/sky/milkyway-4k.jpg` (~2–3 MB). Downscaling is done at build-of-asset time; if no
  offline image tool is available, downscale at load time by drawing the decoded image to a
  4096×2048 offscreen canvas before upload (see §4).
- **Attribution:** a line in `README.md` and a `public/sky/CREDIT.txt` naming ESO/S. Brunier and
  CC BY 4.0 with the source URL.

## 3. Coordinate mapping

Equirectangular lookup from a unit direction `dir` (the escaped ray's asymptotic direction, same
`dir` the procedural starfield uses today):

```
u = atan2(dir.z, dir.x) * (0.5 / PI) + 0.5      // longitude -> [0,1), wraps
v = acos(clamp(dir.y, -1, 1)) * (1.0 / PI)       // latitude  -> [0,1], poles at v=0/1
```

A fixed 3×3 rotation `R_sky` is applied to `dir` before the lookup so the galactic plane sits at a
pleasing tilt relative to the accretion disk (avoids the band lying exactly along an image axis).
`R_sky` is a hardcoded constant in the shader (no runtime control in v1).

## 4. GPU resources (`src/render/gpu.ts`)

- **Texture:** `skyTex`, format `rgba8unorm-srgb` so sampling returns **linear** radiance
  automatically (no manual de-gamma). Full **mipmap chain**.
- **Sampler:** `skySampler`, `minFilter`/`magFilter` = linear, `mipmapFilter` = linear,
  `addressModeU` = repeat (longitude wraps), `addressModeV` = clamp-to-edge (poles).
- **`uploadSky(bitmap)` method:** decode the committed jpg → draw to a 4096×2048 offscreen canvas
  → `device.queue.copyExternalImageToTexture` into mip 0 → generate mips via a small blit pass
  (WebGPU has no built-in `generateMipmaps`; a minimal downsample render pass over each level).
- **Bind group:** add `skyTex` and `skySampler` at the next two binding slots (after the existing
  hotspots buffer). `rebind()` includes them. The compute pipeline layout gains a
  `texture_2d<f32>` (filterable float) and a filtering `sampler`.
- **Load path (`main.ts`):** `fetch('/sky/milkyway-4k.jpg')` → `createImageBitmap` → `uploadSky`.
  On failure (offline / decode error) the sky texture is never bound-in and the shader falls back
  to the procedural starfield (see §6). Load is fire-and-forget; the scene renders immediately and
  the sky pops in when ready (an EMA reset is triggered on arrival).

## 5. Shader (`src/render/raytrace.wgsl`)

Compute shaders have **no implicit derivatives**, so the sky is sampled with **explicit LOD** via
`textureSampleLevel`:

```wgsl
fn sky(dir: vec3<f32>) -> vec3<f32> {
  let d = R_SKY * dir;
  let u = atan2(d.z, d.x) * (0.5 / PI) + 0.5;
  let v = acos(clamp(d.y, -1.0, 1.0)) * (1.0 / PI);
  let lod = max(0.0, log2(SKY_TEXW * angularFootprint / (2.0 * PI)));
  return textureSampleLevel(skyTex, skySamp, vec2<f32>(u, v), lod).rgb;
}
```

`angularFootprint` is an analytic estimate of how much solid angle one pixel subtends after
lensing — small far from the shadow (crisp stars), large in the magnified ring (auto-blur, which is
physically correct: magnification stretches the sky, so per-texel detail drops). v1 uses a simple
estimate from `fovScale / res` scaled by the local ray spread; exact screen-space differencing is
not available in compute, and mipmaps + the existing temporal AA absorb the residual.

The escaped-ray branch becomes:
```wgsl
color = mix(starfield(dir), sky(dir), skyMix) * skyBrightnessOrExisting;
```
where the blend/selection is governed by `U.skyStrength` and whether the texture is loaded (§6).

## 6. Uniforms, controls, and the off-invariant

- **New uniform `skyStrength: f32`** appended at float slot **`f[23]`** — the existing padding slot,
  so `UNIFORM_SIZE` stays **96 bytes** (no buffer growth; the galactic tilt is a shader constant,
  not a uniform). `UNIFORM_FLOATS` 19 → 20.
- **UI:** a **"Sky"** slider (id `sky`, range 0–2, step 0.05, default **0.6**) with a readout,
  wired like the other sliders (updates `state.skyStrength`, calls `reset()`).
- **Off-invariant:** at `skyStrength = 0`, or when the texture failed to load, the escaped-ray
  path is **bit-identical to the current procedural void** — preserving the project's standing
  "all features off ⇒ unchanged" property. The sky texture, when present and `skyStrength > 0`,
  scales the sampled panorama and crossfades out the procedural stars.
- **Validation routes:** `?shadow` renders with `skyStrength` **forced to 0** so its structural
  "dark shadow / ringed by disk" assertions remain valid against a dark background. `?parity` is
  math-only (no scene render) and is unaffected.

## 7. Testing & verification

- **Unit (`tests/`):** extract the pure equirect UV mapping (`dirToEquirectUV`) into a tiny TS
  helper and unit-test it against known directions (+X → u=0.5, ±Y → v=0/1, wrap continuity at the
  ±Z seam). No new physics.
- **GPU (`scripts/verify-gpu.mjs`):** (a) `?parity` and `?shadow` still PASS (shadow with sky off);
  (b) a new sky-on capture asserts zero page errors and a background mean **above** the pure-void
  floor (the sky is actually present); (c) reference render saved.
- **Manual:** headless captures at ~17° (M87-like, band wraps into a clear Einstein ring), ~45°,
  and ~85° (edge-on) confirming the ring and no seam/pole artifacts at the panorama's V-clamped
  poles.

## 8. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| Star aliasing / twinkle on minification in the lensed ring | Full mipmap chain + linear mip filtering + explicit LOD from angular footprint; temporal EMA further averages residual. |
| Compute-shader LOD is approximate (no derivatives) | Conservative `max(lod,0)`; mipmaps make over/under-estimates graceful rather than sparkly. |
| Repo gains a multi-MB binary | Downscale to 4096×2048 (~2–3 MB); single asset under `public/sky/`. |
| Bright sky breaks the `?shadow` darkness check | Force `skyStrength = 0` in the shadow validation route. |
| Offline / fetch failure | Procedural starfield fallback; render never blocks on the asset. |
| Panorama pole pinching (equirect singularity at V=0/1) | Clamp-to-edge V addressing; poles are away from the disk/shadow in normal views. |

## 9. Out of scope (v1)

- User-controllable sky rotation / orientation (fixed tilt only).
- Cubemap projection (equirect single texture is sufficient and simpler).
- HDR / bloom contribution from the sky (sky feeds the accumulation buffer like the disk, so bloom
  already picks up bright band pixels; no special HDR path added).
- Multiple sky presets or swappable panoramas.
