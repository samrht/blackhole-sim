import { describe, it, expect } from "vitest";
import { pickClipFormat, clipName, formatElapsed, CLIP } from "../src/render/clip";
import { screenshotName } from "../src/render/screenshot";

describe("clip export helpers", () => {
  it("prefers H.264 MP4 (plays everywhere), at a level that allows 1080p60", () => {
    const f = pickClipFormat(() => true)!;
    expect(f.ext).toBe("mp4");
    expect(f.mime).toBe("video/mp4;codecs=avc1.640033");
  });
  it("falls back through plain MP4 to WebM, and reports no support as null", () => {
    expect(pickClipFormat((t) => t === "video/mp4")).toEqual({ mime: "video/mp4", ext: "mp4" });
    expect(pickClipFormat((t) => t.startsWith("video/webm;codecs=vp9"))).toEqual({ mime: "video/webm;codecs=vp9", ext: "webm" });
    expect(pickClipFormat((t) => t === "video/webm")).toEqual({ mime: "video/webm", ext: "webm" });
    expect(pickClipFormat(() => false)).toBeNull();
  });
  it("names clips like screenshots, with the video extension", () => {
    const d = new Date(2026, 9, 3, 21, 4, 5);
    expect(clipName("m87", 0.9, 17, d, "mp4")).toBe("blackhole-m87-a0.90-i17-2026-10-03T21-04-05.mp4");
    expect(clipName("m87", 0.9, 17, d, "webm")).toBe(screenshotName("m87", 0.9, 17, d).replace(/\.png$/, ".webm"));
  });
  it("formats elapsed time as m:ss", () => {
    expect(formatElapsed(0)).toBe("0:00");
    expect(formatElapsed(7400)).toBe("0:07");
    expect(formatElapsed(65000)).toBe("1:05");
    expect(formatElapsed(CLIP.maxMs)).toBe("5:00");
  });
  it("records at up to 60 fps, ~16 Mbit/s, 1 s chunks, auto-stop at 5 minutes", () => {
    expect(CLIP).toEqual({ fps: 60, bitsPerSecond: 16e6, chunkMs: 1000, maxMs: 300000 });
  });
});
