import { Renderer } from "./render/gpu";
import type { UniformValues } from "./render/uniforms";
import { ScaleController } from "./render/scale";
import { geometryKey, BuildScheduler, chooseMode } from "./render/cache-plan";
import { describeGpu, isIntegratedGpu } from "./render/gpuinfo";
import { buildTempLUT, buildVisibleLUT } from "./physics/lookups";
import { iscoRadius, photonOrbit } from "./physics/orbits";
import { PRESETS, CUSTOM_DEFAULT, type Preset } from "./physics/presets";
import { computeReadouts, type Readouts } from "./physics/readouts";
import { formatLength, formatDuration } from "./physics/units";
import type { HotSpot } from "./physics/emission";

const canvas = document.getElementById("c") as HTMLCanvasElement;

if (location.search.includes("parity")) {
  // Validation entry: CPU<->GPU parity for the metric/orbit/g-factor math.
  const { runParity } = await import("./test/parity.browser");
  const res = await runParity();
  const ok = res.maxErr < 1e-3;
  document.body.innerHTML = `<pre style="color:${ok ? "#6f6" : "#f66"};font-size:18px;padding:20px">
PARITY ${ok ? "PASS" : "FAIL"} — maxRelErr=${res.maxErr.toExponential(3)} over ${res.rows} cases</pre>`;
  console.log("parity", res);
} else if (location.search.includes("shadow")) {
  // Validation entry: Schwarzschild shadow-radius check (a=0, pole-on).
  const { measureShadow } = await import("./test/shadow.browser");
  const stepsParam = new URLSearchParams(location.search).get("steps");
  const res = await measureShadow(canvas, stepsParam ? +stepsParam : 8000);
  document.body.innerHTML = `<pre style="color:${res.ok ? "#6f6" : "#f66"};font-size:18px;padding:20px">
SHADOW ${res.ok ? "PASS" : "FAIL"} (structural) — centred dark shadow=${res.hasShadow}, ringed by disk=${res.hasDisk}
apparent radius ≈ ${res.shadowRadiusM} M; analytic critical curve = ${res.analyticRadiusM} M
ratio to analytic critical curve = ${res.calibration} (NOT a calibration — the route measures the emitter's lensed inner edge, not the critical curve; see README)</pre>`;
  console.log("shadow", res);
} else if (location.search.includes("cachecheck")) {
  // Validation entry: cached frames equal live frames (spec 2026-10-01 3.6).
  const { runCacheCheck } = await import("./test/cachecheck.browser");
  const res = await runCacheCheck(canvas);
  document.body.innerHTML = `<pre style="color:${res.ok ? "#6f6" : "#f66"};font-size:15px;padding:20px">CACHECHECK ${res.ok ? "PASS" : "FAIL"}\n${res.lines.join("\n")}</pre>`;
  console.log("cachecheck", res);
} else if (location.search.includes("golden")) {
  // Validation entry: bit-exact live-pass hashes (plan 2026-10-01 Task 2). `&record` prints JSON to commit.
  const { runGolden, judgeGolden, GOLDEN: want } = await import("./test/golden.browser");
  const got = await runGolden(canvas);
  if (location.search.includes("record")) {
    document.body.innerHTML = `<pre>GOLDEN RECORD</pre><pre id="json">${JSON.stringify(got)}</pre>`;
  } else {
    const lines = Object.keys(want.hashes).map((k) => `${k}: want ${want.hashes[k]} got ${got.hashes[k]}`);
    const verdict = judgeGolden(want, got);
    document.body.innerHTML = `<pre style="color:${verdict === "FAIL" ? "#f66" : "#6f6"};font-size:16px;padding:20px">GOLDEN ${verdict} (${got.adapter}; recorded on ${want.adapter})\n${lines.join("\n")}</pre>`;
  }
} else if (location.search.includes("bench")) {
  // Idle route for scripts/bench.mjs: no render loop, no validation pass. The benchmark imports the
  // Renderer through the dev server and drives it itself, so nothing else may compete for the GPU.
  document.body.innerHTML = `<pre style="color:#888;padding:20px">bench route (idle)</pre>`;
} else {
  // Normal interactive render.
  const r = new Renderer();
  try {
    await r.init(canvas);
  } catch (e) {
    const err = document.getElementById("err")!;
    err.style.display = "grid";
    err.innerHTML = `${(e as Error).message}<br><br>This renderer needs WebGPU (Chrome/Edge 113+, or Safari 18+).`;
    throw e;
  }

  const $ = (id: string) => document.getElementById(id)!;
  $("gpu").textContent = describeGpu(r.adapterInfo);
  if (isIntegratedGpu(r.adapterInfo)) $("gpuwarn").hidden = false;
  // ?scale=0.5 .. 1 pins the internal render scale and disables the controller (debugging, probes).
  const scaleParam = new URLSearchParams(location.search).get("scale");
  const pinnedScale = scaleParam !== null && isFinite(+scaleParam) ? Math.min(1, Math.max(0.5, +scaleParam)) : null;
  if (pinnedScale !== null) r.setScale(pinnedScale);
  const ctl = new ScaleController();
  const rscaleEl = $("rscale");
  const showScale = () => { rscaleEl.textContent = `${Math.round(r.scale * 100)}%`; };
  showScale();
  const fpsEl = $("fps");
  // Geodesic cache (spec 2026-10-01): playing with an unchanged camera re-shades cached geodesics
  // instead of re-tracing them. ?nocache (or a pinned ?scale) keeps the live path only.
  const cacheOn = !new URLSearchParams(location.search).has("nocache") && pinnedScale === null;
  const cmodeEl = $("cmode");
  const sched = new BuildScheduler(r.cacheSets, r.displayH);
  let geoKey = "", cachedFrame = 0, wasCached = false, liveScale = r.scale;
  let dtEma = 0, lastFpsShow = 0; // display rate (what the user sees): EMA of rAF deltas, shown <= 2x/s

  const state = { a: 0.9, incl: 72, exposure: -1.0, timeScale: 1.0, turbAmp: 0.6, breatheAmp: 0.0, playing: true, flareScale: 1.0, jetStrength: 1.0, jetGamma: 5.0, jetLength: 60.0, jetKnots: 0.7, skyStrength: 1.0, maxSteps: 4800, massSun: CUSTOM_DEFAULT.massSun, lambda: CUSTOM_DEFAULT.lambda };
  const SPEED = 20;        // coordinate-time M advanced per real second at timeScale = 1
  const EMA_BLEND = 0.15;  // trailing-window weight while animating
  let simTime = 0, lastNow = 0;
  const baseSpots: HotSpot[] = [
    { r: 8,  psi: 0.0, sigma: 1.2, amp: 1.8 },
    { r: 12, psi: 2.1, sigma: 1.6, amp: 1.2 },
    { r: 16, psi: 4.3, sigma: 2.0, amp: 0.9 },
  ];
  const packSpots = (scale: number) => {
    const f = new Float32Array(baseSpots.length * 4);
    baseSpots.forEach((s, i) => f.set([s.r, s.psi, s.sigma, s.amp * scale], i * 4));
    return f;
  };
  const rOut = 40;
  const COLOR_LUT = buildVisibleLUT();
  // Physical state -> peak disk temperature, lumNorm and the panel's physical readouts (spec
  // 2026-10-01). Mass and accretion are shading-only: they never enter the geometry key.
  let phys: Readouts = computeReadouts(state, SPEED);
  const refreshPhysics = () => { phys = computeReadouts(state, SPEED); };

  // --- DOM controls + live physics readouts -------------------------------------------------
  const spin = $("spin") as HTMLInputElement, incl = $("incl") as HTMLInputElement, exp = $("exp") as HTMLInputElement;
  const spinv = $("spinv"), inclv = $("inclv"), expv = $("expv");
  const rhEl = $("rh"), riscoEl = $("risco"), rphEl = $("rph"), sppEl = $("spp");
  const tpkEl = $("tpk"), piscoEl = $("pisco"), tscaleEl = $("tscale");
  const inM = (x: number) => `${x.toFixed(2)}<i>M</i>`;

  let rIn = iscoRadius(state.a, true);
  let sample = 0; // progressive-accumulation sample index; setting to 0 re-converges the image
  const reset = () => { sample = 0; };

  function rebuildLUTs() {
    rIn = iscoRadius(state.a, true);
    r.uploadLUTs(buildTempLUT(state.a, true, rIn, rOut, 512), COLOR_LUT);
    r.rebind();
  }
  function refreshReadouts() {
    const rh = 1 + Math.sqrt(Math.max(0, 1 - state.a * state.a));
    rhEl.innerHTML = `${inM(rh)} <i>${formatLength(phys.horizonM)}</i>`;
    riscoEl.innerHTML = `${inM(iscoRadius(state.a, true))} <i>${formatLength(phys.iscoM)}</i>`;
    rphEl.innerHTML = `${inM(photonOrbit(state.a, true))} <i>${formatLength(phys.photonM)}</i>`;
    tpkEl.textContent = `${Math.round(phys.tPeakK).toLocaleString("en-US", { maximumSignificantDigits: 3 })} K`;
    piscoEl.textContent = formatDuration(phys.iscoPeriodS);
    tscaleEl.textContent = state.timeScale > 0 ? `≈ ${formatDuration(phys.realSecondsPerScreenSecond)}` : "—";
  }

  spin.addEventListener("input", () => {
    state.a = +spin.value; spinv.textContent = state.a.toFixed(3);
    rebuildLUTs(); markCustom(); physicsChanged(); // spin changes T_peak too (eta(a), flux profile)
  });
  incl.addEventListener("input", () => {
    state.incl = +incl.value; inclv.textContent = String(state.incl); markCustom(); reset();
  });
  exp.addEventListener("input", () => {
    // exposure is applied at present time, so it updates live without re-accumulating
    state.exposure = +exp.value; expv.textContent = (state.exposure >= 0 ? "+" : "") + state.exposure.toFixed(1);
  });

  const ts = $("ts") as HTMLInputElement, turb = $("turb") as HTMLInputElement, flare = $("flare") as HTMLInputElement;
  const tsv = $("tsv"), turbv = $("turbv"), flarev = $("flarev"), playBtn = $("playpause") as HTMLButtonElement;

  ts.addEventListener("input", () => { state.timeScale = +ts.value; tsv.textContent = state.timeScale.toFixed(1); refreshPhysics(); refreshReadouts(); });
  turb.addEventListener("input", () => { state.turbAmp = +turb.value; turbv.textContent = state.turbAmp.toFixed(2); reset(); });
  flare.addEventListener("input", () => {
    state.flareScale = +flare.value; flarev.textContent = state.flareScale.toFixed(1);
    r.uploadHotSpots(packSpots(state.flareScale)); reset();
  });
  playBtn.addEventListener("click", () => {
    state.playing = !state.playing;
    playBtn.textContent = state.playing ? "Pause" : "Play";
    reset(); // clean restart (play) or fresh convergence to a still (pause)
  });

  const jet = $("jet") as HTMLInputElement, jg = $("jg") as HTMLInputElement, jk = $("jk") as HTMLInputElement;
  const jetv = $("jetv"), jgv = $("jgv"), jkv = $("jkv");

  // Two significant digits below 0.1 (M87*'s calibrated jet is 0.0034).
  const showJet = () => { jetv.textContent = state.jetStrength >= 0.1 || state.jetStrength === 0 ? state.jetStrength.toFixed(1) : state.jetStrength.toPrecision(2); };
  jet.addEventListener("input", () => { state.jetStrength = +jet.value; showJet(); reset(); });
  jg.addEventListener("input", () => { state.jetGamma = +jg.value; jgv.textContent = state.jetGamma.toFixed(1); reset(); });
  jk.addEventListener("input", () => { state.jetKnots = +jk.value; jkv.textContent = state.jetKnots.toFixed(2); reset(); });

  // --- Object presets, Mass and Accretion (spec 2026-10-01) ---------------------------------
  const presetSel = $("preset") as HTMLSelectElement, pcap = $("pcap");
  for (const p of PRESETS) presetSel.add(new Option(p.name, p.id));
  const mass = $("mass") as HTMLInputElement, acc = $("acc") as HTMLInputElement, massv = $("massv"), accv = $("accv");
  const showMass = () => { massv.textContent = state.massSun.toExponential(2); };
  const showAcc = () => { accv.textContent = state.lambda.toExponential(1); };
  /** Any change to spin, inclination, mass or accretion (drag-tilt included) leaves the preset (or
   *  the Default view) for "Custom", a disabled option that only ever shows the state. */
  function markCustom() { if (presetSel.value !== "custom") { presetSel.value = "custom"; pcap.hidden = true; } }
  function physicsChanged() { refreshPhysics(); refreshReadouts(); reset(); }

  mass.addEventListener("input", () => { state.massSun = 10 ** +mass.value; showMass(); markCustom(); physicsChanged(); });
  acc.addEventListener("input", () => { state.lambda = 10 ** +acc.value; showAcc(); markCustom(); physicsChanged(); });

  function applyPreset(p: Preset) {
    state.a = p.a; state.incl = p.inclDeg; state.massSun = p.massSun; state.lambda = p.lambda;
    state.jetStrength = p.jetStrength;
    spin.value = String(p.a); spinv.textContent = p.a.toFixed(3);
    incl.value = String(p.inclDeg); inclv.textContent = String(p.inclDeg);
    mass.value = String(Math.log10(p.massSun)); showMass();
    acc.value = String(Math.log10(p.lambda)); showAcc();
    jet.value = String(state.jetStrength); showJet();
    pcap.textContent = p.caption; pcap.hidden = !p.caption;
    rebuildLUTs(); physicsChanged();
  }
  /** The opening view: what "Default view" restores after a preset or custom changes. */
  const DEFAULT_VIEW: Preset = { id: "default", name: "Default view", massSun: CUSTOM_DEFAULT.massSun, a: 0.9, inclDeg: 72,
    lambda: CUSTOM_DEFAULT.lambda, jetStrength: 1, caption: "" };
  presetSel.addEventListener("change", () => {
    const p = presetSel.value === "default" ? DEFAULT_VIEW : PRESETS.find((q) => q.id === presetSel.value);
    if (p) applyPreset(p);
  });

  const sky = $("sky") as HTMLInputElement, skyv = $("skyv");
  sky.addEventListener("input", () => { state.skyStrength = +sky.value; skyv.textContent = state.skyStrength.toFixed(2); reset(); });

  const detail = $("detail") as HTMLInputElement, detailv = $("detailv");
  detail.addEventListener("input", () => { state.maxSteps = +detail.value; detailv.textContent = String(state.maxSteps); reset(); });

  // Drag vertically to tilt the camera (inclination); keeps the slider + readouts in sync.
  let dragging = false, lastY = 0;
  canvas.addEventListener("pointerdown", (e) => { dragging = true; lastY = e.clientY; canvas.classList.add("drag"); canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointerup", (e) => { dragging = false; canvas.classList.remove("drag"); canvas.releasePointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const next = Math.min(89, Math.max(1, state.incl + (e.clientY - lastY) * 0.25));
    lastY = e.clientY;
    if (Math.round(next) !== state.incl) {
      state.incl = Math.round(next); incl.value = String(state.incl); inclv.textContent = String(state.incl); markCustom(); reset();
    }
  });

  // Keep the render crisp across window resizes.
  let resizeTimer = 0;
  addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => { r.resize(canvas); r.rebind(); reset(); }, 120);
  });

  rebuildLUTs();
  showMass(); showAcc();
  refreshReadouts();
  r.uploadHotSpots(packSpots(state.flareScale));

  // Load the baked sky panorama asynchronously; on failure keep the procedural starfield fallback.
  let skyReady = false; // the sky panorama only contributes once the async upload has landed;
                        // until then (and on load failure) skyStrength is forced to 0 so the
                        // escaped-ray path stays bit-identical to the procedural void.
  fetch("/sky/milkyway-4k.jpg")
    .then((res) => { if (!res.ok) throw new Error(`sky ${res.status}`); return res.blob(); })
    .then(createImageBitmap)
    .then((bmp) => { r.uploadSky(bmp); r.rebind(); skyReady = true; reset(); })
    .catch(() => { /* offline / decode error — skyReady stays false, procedural starfield stays */ });

  // Uncapped: render every animation frame. While animating, the ScaleController trades internal
  // resolution for frame rate (it targets ~60 fps, never below half resolution; what a given GPU
  // actually reaches is shown in the FPS and Render-scale readouts); paused, the scale snaps back to
  // 1 and the progressive running mean converges to a sharp still.
  function loop(now: number) {
    const dt = lastNow ? now - lastNow : 0; lastNow = now;
    if (state.playing) simTime += (dt / 1000) * SPEED;
    if (dt > 0 && dt < 250) dtEma = dtEma ? dtEma + 0.1 * (dt - dtEma) : dt;
    if (now - lastFpsShow >= 500 && dtEma > 0) { fpsEl.textContent = (1000 / dtEma).toFixed(0); lastFpsShow = now; }
    const geo = geometryKey({ a: state.a, incl: state.incl, fovScale: 14, rObs: 1000, rIn, rOut,
      maxSteps: state.maxSteps, jetLength: state.jetLength, displayW: r.displayW, displayH: r.displayH, epoch: r.cacheEpoch });
    if (geo !== geoKey) { geoKey = geo; sched.reset(r.cacheSets, r.displayH); r.resetCache(); }
    // One background build slice per playing frame (the cache is only used while playing).
    const slice = state.playing && cacheOn ? sched.next() : null;
    const mode = chooseMode(state.playing, sched.completedSets, cacheOn);

    if (mode === "cached") {
      if (!wasCached) { liveScale = r.scale; r.setScale(1); showScale(); cachedFrame = 0; }
    } else {
      if (wasCached) { r.setScale(liveScale); ctl.reset(liveScale); reset(); showScale(); }
      if (pinnedScale === null) {
        if (state.playing) {
          // Controlled on GPU work time per frame, not the rAF delta: rAF is vsync-quantised (never
          // below 16.7 ms at 60 Hz, inside the 13-18 ms dead band), so after any slow spell a
          // rAF-driven controller could only ratchet down. NaN until the first frame completes.
          const ns = ctl.update(r.gpuMs, now);
          // Show the controller's new scale even when the internal size did not change (then no reset).
          if (ns !== null) { if (r.setScale(ns)) reset(); showScale(); }
        } else if (r.scale !== 1) {
          r.setScale(1); ctl.reset(1); reset(); showScale();
        }
      }
    }
    // Playing: fixed EMA (blend==1 on the reset frame to clear). Paused: progressive running mean.
    // A cached run starts with blend 1 too, which rewrites every pixel (no stale live content).
    const blend = mode === "cached" ? (cachedFrame === 0 ? 1 : EMA_BLEND)
      : state.playing ? (sample === 0 ? 1 : EMA_BLEND) : 1 / (sample + 1);
    const setIndex = mode === "cached" ? cachedFrame % sched.completedSets : 0;
    const u: UniformValues = {
      resW: r.width, resH: r.height, outW: r.displayW, outH: r.displayH, a: state.a, incl: state.incl * Math.PI / 180,
      rObs: 1000, fovScale: 14, rIn, rOut, Tpeak: phys.tPeakK, lumNorm: phys.lumNorm, exposure: state.exposure,
      time: simTime, frame: sample, reset: sample === 0 ? 1 : 0, maxSteps: state.maxSteps,
      blend, timeScale: state.timeScale, turbAmp: state.turbAmp,
      breatheAmp: state.breatheAmp, nSpots: baseSpots.length,
      jetStrength: state.jetStrength, jetGamma: state.jetGamma,
      jetLength: state.jetLength, jetKnots: state.jetKnots,
      skyStrength: skyReady ? state.skyStrength : 0, setIndex,
    };
    r.frame(u, {
      cachedSet: mode === "cached" ? setIndex : undefined,
      build: slice ? { ...u, resW: r.displayW, resH: r.displayH, setIndex: slice.set, rowStart: slice.rowStart, rowEnd: slice.rowEnd } : undefined,
    });
    if (mode === "cached") cachedFrame++;
    wasCached = mode === "cached";
    cmodeEl.textContent = mode === "cached" ? `cached ${sched.completedSets}/${r.cacheSets}` : (r.cacheSets && cacheOn ? "live" : "off");
    sample++;
    if ((sample & 7) === 0 || sample < 4) sppEl.textContent = String(sample);
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}
