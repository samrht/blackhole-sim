// Clip export (2026-10-03): record the canvas as it plays (canvas.captureStream + MediaRecorder) and download
// it. The view is what is recorded, never the panel; smoothness is the live frame rate. pickClipFormat,
// clipName and formatElapsed are pure (tests/clip.test.ts); ClipRecorder is covered by the verify:gpu app check.
import { sceneStem } from "./screenshot";

export const CLIP = { fps: 60, bitsPerSecond: 16e6, chunkMs: 1000, maxMs: 300000 };

// H.264 High at level 5.1 (avc1.640033) allows 1080p60 and larger canvases; MP4 plays everywhere, WebM is
// the fallback for browsers whose MediaRecorder has no MP4.
const FORMATS: { mime: string; ext: "mp4" | "webm" }[] = [
  { mime: "video/mp4;codecs=avc1.640033", ext: "mp4" }, { mime: "video/mp4", ext: "mp4" },
  { mime: "video/webm;codecs=vp9", ext: "webm" }, { mime: "video/webm;codecs=vp8", ext: "webm" }, { mime: "video/webm", ext: "webm" },
];
export function pickClipFormat(isSupported: (mime: string) => boolean): { mime: string; ext: "mp4" | "webm" } | null {
  return FORMATS.find((f) => isSupported(f.mime)) ?? null;
}
export function clipName(preset: string, a: number, inclDeg: number, when: Date, ext: string): string {
  return `${sceneStem(preset, a, inclDeg, when)}.${ext}`;
}
export function formatElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** One recording of a canvas: start(), then stop() resolves with the finished video. */
export class ClipRecorder {
  private rec: MediaRecorder; private chunks: Blob[] = []; private done: Promise<Blob>;
  readonly startedAt = performance.now();
  constructor(canvas: HTMLCanvasElement, readonly format: { mime: string; ext: string }) {
    const stream = canvas.captureStream(CLIP.fps);
    this.rec = new MediaRecorder(stream, { mimeType: format.mime, videoBitsPerSecond: CLIP.bitsPerSecond });
    this.rec.ondataavailable = (e) => { if (e.data.size > 0) this.chunks.push(e.data); };
    this.done = new Promise((res) => {
      this.rec.onstop = () => { stream.getTracks().forEach((t) => t.stop()); res(new Blob(this.chunks, { type: format.mime.split(";")[0] })); };
    });
    this.rec.start(CLIP.chunkMs);
  }
  get recording(): boolean { return this.rec.state === "recording"; }
  stop(): Promise<Blob> { if (this.rec.state !== "inactive") this.rec.stop(); return this.done; }
}
