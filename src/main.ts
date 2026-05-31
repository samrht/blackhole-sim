import { Renderer } from "./render/gpu";
import type { UniformValues } from "./render/uniforms";
import { buildTempLUT, buildColorLUT } from "./physics/lookups";
import { iscoRadius, photonOrbit } from "./physics/orbits";

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

  const state = { a: 0.9, incl: 72, exposure: 1.6 };
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

  function loop() {
    const u: UniformValues = {
      resW: r.width, resH: r.height, a: state.a, incl: state.incl * Math.PI / 180,
      rObs: 1000, fovScale: 14, rIn, rOut, Tpeak: T_PEAK, exposure: state.exposure,
      time: performance.now() / 1000, frame: sample, reset: sample === 0 ? 1 : 0, maxSteps: 3000,
    };
    r.frame(u);
    sample++;
    if ((sample & 7) === 0 || sample < 4) sppEl.textContent = String(sample);
    requestAnimationFrame(loop);
  }
  loop();
}
