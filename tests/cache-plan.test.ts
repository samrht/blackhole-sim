import { describe, it, expect } from "vitest";
import {
  JITTER, BUILD_SLICES, SETTLE_FRAMES, ENTRY_BYTES, BOOKMARK_BYTES, BOOKMARK_FRAC, CACHE_BUDGET_BYTES,
  geometryKey, planCache, BuildScheduler, chooseMode, type GeometryInputs,
} from "../src/render/cache-plan";

const G: GeometryInputs = { a: 0.9, incl: 72, fovScale: 14, rObs: 1000, rIn: 2.32, rOut: 40, maxSteps: 4800, jetLength: 60, displayW: 1920, displayH: 1080, epoch: 0,
  band: 0, hotFlow: 0, flowN0: 0, flowRg: 0 };
const MB = 2 ** 20;

describe("jitter sets", () => {
  it("are 4 distinct offsets inside the pixel with zero mean", () => {
    expect(JITTER.length).toBe(4);
    expect(new Set(JITTER.map((j) => j.join())).size).toBe(4);
    for (const [x, y] of JITTER) { expect(Math.abs(x)).toBeLessThan(0.5); expect(Math.abs(y)).toBeLessThan(0.5); }
    expect(JITTER.reduce((s, j) => s + j[0], 0)).toBeCloseTo(0, 12);
    expect(JITTER.reduce((s, j) => s + j[1], 0)).toBeCloseTo(0, 12);
  });
});

describe("geometry key", () => {
  it("changes with every ray-path input", () => {
    const k = geometryKey(G);
    for (const f of Object.keys(G) as (keyof GeometryInputs)[]) {
      expect(geometryKey({ ...G, [f]: G[f] + 1 })).not.toBe(k);
    }
  });
  it("ignores shading-only inputs (time, exposure, jet strength, sky ...)", () => {
    const withShading = { ...G, time: 99, exposure: 3, jetStrength: 0, jetGamma: 9, fluxVar: 0.1, skyStrength: 0.5, turbAmp: 0 } as GeometryInputs;
    expect(geometryKey(withShading)).toBe(geometryKey(G));
  });
});

describe("bookmark headroom (jet-knots review: the envelope widened to q <= 1.51)", () => {
  it("BOOKMARK_FRAC keeps a 1.5x margin over the measured face-on bookmark fraction, 18.2 %", () => {
    expect(BOOKMARK_FRAC).toBeGreaterThanOrEqual(1.5 * 0.182);
    expect(BOOKMARK_FRAC).toBeLessThanOrEqual(0.3);
  });
});

describe("planCache", () => {
  it("uses 4 sets at 1080p on a 128 MB binding", () => {
    const p = planCache(1920, 1080, 128 * MB);
    expect(p.nSets).toBe(4);
    expect(p.entryBytes).toBe(1920 * 1080 * ENTRY_BYTES);
    expect(p.bookmarkCapacity).toBe(Math.min(Math.ceil(BOOKMARK_FRAC * 1920 * 1080 * 4), Math.floor(128 * MB / BOOKMARK_BYTES)));
    expect(4 * p.entryBytes + p.bookmarkCapacity * BOOKMARK_BYTES).toBeLessThanOrEqual(CACHE_BUDGET_BYTES);
  });
  it("stays live when one set does not fit a binding (5K canvas, 128 MiB)", () => {
    expect(planCache(5120, 2880, 128 * MB).nSets).toBe(0); // 236 MB per set
  });
  it("drops to fewer sets when the 512 MB budget would be exceeded", () => {
    const p = planCache(3840, 2160, 2048 * MB);
    expect([1, 2]).toContain(p.nSets);
    expect(p.nSets * p.entryBytes + p.bookmarkCapacity * BOOKMARK_BYTES).toBeLessThanOrEqual(CACHE_BUDGET_BYTES);
  });
  it("never plans a zero-capacity bookmark buffer when caching", () => {
    expect(planCache(8, 8, 128 * MB).bookmarkCapacity).toBeGreaterThanOrEqual(1);
  });
});

describe("BuildScheduler", () => {
  it("covers every row of every set exactly once, in BUILD_SLICES slices per set", () => {
    const h = 1080, s = new BuildScheduler(4, h);
    const seen = [new Uint8Array(h), new Uint8Array(h), new Uint8Array(h), new Uint8Array(h)];
    let n = 0;
    for (let x = s.next(); x; x = s.next()) { for (let y = x.rowStart; y < x.rowEnd; y++) seen[x.set][y]++; n++; }
    expect(n).toBe(4 * BUILD_SLICES);
    for (const a of seen) expect(a.every((v) => v === 1)).toBe(true);
    expect(s.completedSets).toBe(4);
    expect(s.next()).toBeNull();
  });
  it("marks a set complete on the call that hands out its last slice", () => {
    const s = new BuildScheduler(2, 32);
    for (let k = 0; k < BUILD_SLICES - 1; k++) s.next();
    expect(s.completedSets).toBe(0);
    s.next();
    expect(s.completedSets).toBe(1);
  });
  it("handles fewer rows than slices and resets", () => {
    const s = new BuildScheduler(1, 5);
    const got = []; for (let x = s.next(); x; x = s.next()) got.push(x);
    expect(got.map((g) => [g.rowStart, g.rowEnd])).toEqual([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]);
    s.reset(2, 10);
    expect(s.completedSets).toBe(0);
    for (let k = 0; k < SETTLE_FRAMES; k++) expect(s.next()).toBeNull();
    expect(s.next()).toEqual({ set: 0, rowStart: 0, rowEnd: 1 });
  });
  it("builds nothing while the geometry keeps changing (dragging): every reset restarts the settle wait", () => {
    const s = new BuildScheduler(4, 1080);
    for (let frame = 0; frame < 50; frame++) { s.reset(4, 1080); expect(s.next()).toBeNull(); }
    for (let k = 1; k < SETTLE_FRAMES; k++) expect(s.next()).toBeNull();
    expect(s.next()).toEqual({ set: 0, rowStart: 0, rowEnd: 68 });
  });
  it("never builds anything with 0 sets", () => {
    expect(new BuildScheduler(0, 100).next()).toBeNull();
  });
});

describe("chooseMode", () => {
  it("forceLive (a 1.3 mm hotspot can be in view) traces live even with a complete cache", () => {
    expect(chooseMode(true, 4, true, true)).toBe("live");
    expect(chooseMode(true, 4, true, false)).toBe("cached");
  });
  it("is cached only while playing, enabled, with a complete set", () => {
    expect(chooseMode(true, 1, true)).toBe("cached");
    expect(chooseMode(true, 0, true)).toBe("live");   // still building set 0
    expect(chooseMode(false, 4, true)).toBe("live");  // paused: progressive still
    expect(chooseMode(true, 4, false)).toBe("live");  // ?nocache / pinned scale
  });
});
