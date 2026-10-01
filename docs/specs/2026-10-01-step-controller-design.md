# Step controller: fewer steps, no ray less accurate — design and outcome

**Date:** 2026-10-01
**Status:** implemented on `perf/far-monitor` (from `main` at `da1f216`)

## Goal

Speed up the live trace (dragging, ray-path sliders, paused stills; the geodesic cache already covers
still-camera playback) without changing the physics. Hard constraint from the user: do not lose
realism. Operationalised as: against a converged reference, **no fate flip and no ray worse than the
old controller** (disk +0.02 M; sky +0.5 px of on-screen displacement), every existing gate green.

## What the measurement found

The 2026-09-23 far-stride sweep blamed longer far strides' failures on the unmonitored near-axis
far-field passage. Re-measured with a per-ray error split (near field vs far field converged
separately), the binding error was **near-field near-axis error**: rays passing the polar axis inside
r = 1.5 rOut, bounded only by H_TOL, were up to 0.3 M off, and the near-critical sky rays of the
default view were off because their near-field passage was under-resolved (a converged far field
made them *worse*: the old controller's two errors cancelled by luck). A second defect: at longer far
strides the ~pi azimuth swing of a p_phi != 0 ray turning beside the axis (p_theta ~ 0 there, so the
polar cap is inactive) was under-resolved.

## Design

1. `angularCap` bounds both branches of `stepSize`: F_AXIS (polar, existing) and a new F_PHI = 0.1
   rad of azimuth per step; floors DL_FAR_MIN (far) and 0.002 (near).
2. Far stride K_FAR = 0.08, DL_FAR_MAX = 50 (was 0.04 / 6).
3. Far field monitored at H_TOL_FAR = 1e-5 (was 1e30) as a safety net; it never fires on a sweep ray
   once (1) is in.
4. Jet quadrature decoupled from the stride: samples at most JET_DL = 0.25 apart along the step
   chord, exact skip outside the jet's bounding sphere; the geodesic cache bookmarks by the same
   sample points (`jetTouches`) so cached playback still equals live.

## Evidence

- `tests/sweep-farmonitor.test.ts` (SWEEP=1, HOLDOUT=1): tables in its header. Sweep: disk bad
  77 -> 0, sky bad 3 -> 0, worst sky 1.0 -> 0.15 px, cost 348 -> 220 (-37 %); holdout 353 -> 217.
- GPU, the four `?golden` scenes vs a fine-step render: pixels > 5 % off 12->7, 17->5, 14->8,
  116->3; wrong-fate pixels 43 -> 18 (edge-on +2 traced to f32: both controllers match the f64
  reference on all of them).
- `?parity` 55 cases (new: far-phi, near-cap; barrier now supplies its own pre-cap stride).
- `tests/trace.test.ts` pole passage: Carter constant drift 3.5e-3 -> 2.8e-6, bounds tightened.

## Rejected

- 0.08/20 (sky 0.07 px, -31 %): safe but slower. 0.1/50, 0.12/50 (-40 %): pass, worse worst sky ray
  (0.21 px). 0.08/150: 0.32 px. Far monitor 1e-6: identical results to 1e-5, closer to the f32 floor.
- Jet sampling per step (old) or every 1.0 / 0.5: face-on jet 116 / 29 / 7 pixels > 5 % off vs 2.
