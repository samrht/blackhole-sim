import { Renderer } from "./render/gpu";
import type { UniformValues } from "./render/uniforms";
import { buildTempLUT, buildColorLUT } from "./physics/lookups";
import { iscoRadius, photonOrbit } from "./physics/orbits";
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
  const res = await measureShadow(canvas);
  document.body.innerHTML = `<pre style="color:${res.ok ? "#6f6" : "#f66"};font-size:18px;padding:20px">
SHADOW ${res.ok ? "PASS" : "FAIL"} (structural) — centred dark shadow=${res.hasShadow}, ringed by disk=${res.hasDisk}
apparent radius ≈ ${res.shadowRadiusM} M  (ideal sqrt(27) = ${res.bCritM} M; camera scale factor ≈ ${res.scaleVsBcrit})</pre>`;
  console.log("shadow", res);
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

  const state = { a: 0.9, incl: 72, exposure: 1.6, timeScale: 1.0, turbAmp: 0.6, breatheAmp: 0.0, playing: true, flareScale: 1.0 };
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
  // Effective color-temperature scale (K) for visualization. The true Novikov–Thorne peak for a
  // stellar-mass disk (~1e7 K) radiates in X-ray, so we map the normalized profile into the visible
  // blackbody range spanned by the color LUT ([1000, 40000] K).
  const T_PEAK = 3.0e4;

  // --- DOM controls + live physics readouts -------------------------------------------------
  const $ = (id: string) => document.getElementById(id)!;
  const spin = $("spin") as HTMLInputElement, incl = $("incl") as HTMLInputElement, exp = $("exp") as HTMLInputElement;
  const spinv = $("spinv"), inclv = $("inclv"), expv = $("expv");
  const rhEl = $("rh"), riscoEl = $("risco"), rphEl = $("rph"), sppEl = $("spp");
  const inM = (x: number) => `${x.toFixed(2)}<i>M</i>`;

  let rIn = iscoRadius(state.a, true);
  let sample = 0; // progressive-accumulation sample index; setting to 0 re-converges the image
  const reset = () => { sample = 0; };

  function rebuildLUTs() {
    rIn = iscoRadius(state.a, true);
    r.uploadLUTs(buildTempLUT(state.a, true, rIn, rOut, 512), buildColorLUT(1000, 40000, 256));
    r.rebind();
  }
  function refreshReadouts() {
    rhEl.innerHTML = inM(1 + Math.sqrt(Math.max(0, 1 - state.a * state.a)));
    riscoEl.innerHTML = inM(iscoRadius(state.a, true));
    rphEl.innerHTML = inM(photonOrbit(state.a, true));
  }

  spin.addEventListener("input", () => {
    state.a = +spin.value; spinv.textContent = state.a.toFixed(3);
    rebuildLUTs(); refreshReadouts(); reset();
  });
  incl.addEventListener("input", () => {
    state.incl = +incl.value; inclv.textContent = String(state.incl); reset();
  });
  exp.addEventListener("input", () => {
    // exposure is applied at present time, so it updates live without re-accumulating
    state.exposure = +exp.value; expv.textContent = (state.exposure >= 0 ? "+" : "") + state.exposure.toFixed(1);
  });

  const ts = $("ts") as HTMLInputElement, turb = $("turb") as HTMLInputElement, flare = $("flare") as HTMLInputElement;
  const tsv = $("tsv"), turbv = $("turbv"), flarev = $("flarev"), playBtn = $("playpause") as HTMLButtonElement;

  ts.addEventListener("input", () => { state.timeScale = +ts.value; tsv.textContent = state.timeScale.toFixed(1); });
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

  // Drag vertically to tilt the camera (inclination); keeps the slider + readouts in sync.
  let dragging = false, lastY = 0;
  canvas.addEventListener("pointerdown", (e) => { dragging = true; lastY = e.clientY; canvas.classList.add("drag"); canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointerup", (e) => { dragging = false; canvas.classList.remove("drag"); canvas.releasePointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const next = Math.min(89, Math.max(1, state.incl + (e.clientY - lastY) * 0.25));
    lastY = e.clientY;
    if (Math.round(next) !== state.incl) {
      state.incl = Math.round(next); incl.value = String(state.incl); inclv.textContent = String(state.incl); reset();
    }
  });

  // Keep the render crisp across window resizes.
  let resizeTimer = 0;
  addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => { r.resize(canvas); r.rebind(); reset(); }, 120);
  });

  rebuildLUTs();
  refreshReadouts();
  r.uploadHotSpots(packSpots(state.flareScale));

  // On slower GPUs, cap accumulation to ~15 fps so the tab stays responsive.
  // Lower maxSteps reduces per-frame GPU time; quality is recovered by accumulation over time.
  const TARGET_MS = 67; // ~15 fps cap so the tab stays responsive on modest GPUs
  let lastFrame = 0;
  function loop(now: number) {
    if (now - lastFrame >= TARGET_MS) {
      const dt = lastNow ? (now - lastNow) / 1000 : 0; lastNow = now;
      if (state.playing) simTime += dt * SPEED;   // advance coordinate time only while playing
      lastFrame = now;
      // Playing: fixed EMA (blend==1 on the reset frame to clear). Paused: progressive running mean.
      const blend = state.playing ? (sample === 0 ? 1 : EMA_BLEND) : 1 / (sample + 1);
      const u: UniformValues = {
        resW: r.width, resH: r.height, a: state.a, incl: state.incl * Math.PI / 180,
        rObs: 1000, fovScale: 14, rIn, rOut, Tpeak: T_PEAK, exposure: state.exposure,
        time: simTime, frame: sample, reset: sample === 0 ? 1 : 0, maxSteps: 1200,
        blend, timeScale: state.timeScale, turbAmp: state.turbAmp,
        breatheAmp: state.breatheAmp, nSpots: baseSpots.length,
      };
      r.frame(u);
      sample++;
      if ((sample & 7) === 0 || sample < 4) sppEl.textContent = String(sample);
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}
