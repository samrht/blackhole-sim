// Screenshot export (2026-10-03): the presented frame at the canvas's full internal resolution, read back by
// Renderer.readbackDisplay and saved as a PNG. toRGBA and screenshotName are pure (tests/screenshot.test.ts);
// savePNG touches the DOM and is covered by the verify:gpu app check.

/** Tightly packed readback of the canvas format -> RGBA8 for ImageData, alpha forced opaque (the canvas is
 *  configured "opaque"; the present pass's alpha is not meaningful). Returns a new array. */
export function toRGBA(px: Uint8Array, format: GPUTextureFormat): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(px.length);
  const bgra = format.startsWith("bgra");
  for (let i = 0; i < px.length; i += 4) {
    out[i] = bgra ? px[i + 2] : px[i];
    out[i + 1] = px[i + 1];
    out[i + 2] = bgra ? px[i] : px[i + 2];
    out[i + 3] = 255;
  }
  return out;
}

const pad = (n: number) => String(n).padStart(2, "0");
/** blackhole-<preset>-a<spin>-i<inclination deg>-<local time, filesystem-safe>.png */
export function screenshotName(preset: string, a: number, inclDeg: number, when: Date): string {
  const stamp = `${when.getFullYear()}-${pad(when.getMonth() + 1)}-${pad(when.getDate())}T${pad(when.getHours())}-${pad(when.getMinutes())}-${pad(when.getSeconds())}`;
  return `blackhole-${preset}-a${a.toFixed(2)}-i${Math.round(inclDeg)}-${stamp}.png`;
}

/** Encode RGBA pixels as a PNG and download it under `name`. */
export async function savePNG(rgba: Uint8ClampedArray<ArrayBuffer>, w: number, h: number, name: string): Promise<void> {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  c.getContext("2d")!.putImageData(new ImageData(rgba, w, h), 0, 0);
  const blob = await new Promise<Blob | null>((res) => c.toBlob(res, "image/png"));
  if (!blob) throw new Error("PNG encoding failed");
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a"); link.href = url; link.download = name;
  document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}
