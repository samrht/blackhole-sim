import { describe, it, expect } from "vitest";
import { SHARE_FIELDS, encodeShare, decodeShare, type ShareLimit } from "../src/share";

// The panel's real ranges (index.html) for the fields under test.
const LIMITS: Record<string, ShareLimit> = {
  p: { kind: "select", options: ["default", "custom", "m87", "sgra"] },
  a: { kind: "range", min: 0, max: 0.998, step: 0.001 },
  i: { kind: "range", min: 1, max: 89, step: 1 },
  m: { kind: "range", min: 0, max: 10, step: 0.01 },
  x: { kind: "range", min: -3, max: 4, step: 0.1 },
  fv: { kind: "range", min: 0, max: 1.4, step: 0.05 },
  eta: { kind: "range", min: -4, max: 0, step: 0.05 },
  jet: { kind: "check" },
  play: { kind: "check" },
};
const DEFAULTS = { p: "default", a: "0.9", i: "72", m: "8", x: "-1", fv: "1", eta: "-2.7", jet: "1", play: "1" };

describe("shareable links (#hash)", () => {
  it("every field maps to a panel control once, preset first and pause last (apply order)", () => {
    const keys = SHARE_FIELDS.map((f) => f.key), ids = SHARE_FIELDS.map((f) => f.id);
    expect(new Set(keys).size).toBe(keys.length);
    expect(new Set(ids).size).toBe(ids.length);
    expect(keys[0]).toBe("p");
    expect(keys[keys.length - 1]).toBe("play");
    expect(SHARE_FIELDS.filter((f) => f.customOnly).map((f) => f.key)).toEqual(["a", "i", "m", "acc"]);
  });
  it("the default view encodes to an empty hash", () => {
    expect(encodeShare(DEFAULTS, DEFAULTS)).toBe("");
  });
  it("writes only what differs; spin/inclination/mass/accretion only for a Custom view", () => {
    expect(encodeShare({ ...DEFAULTS, p: "m87", a: "0.9", i: "17", x: "0.5", play: "0" }, DEFAULTS)).toBe("p=m87&x=0.5&play=0");
    expect(encodeShare({ ...DEFAULTS, p: "custom", a: "0.5", i: "30" }, DEFAULTS)).toBe("p=custom&a=0.5&i=30");
  });
  it("round-trips through decode", () => {
    const v = { ...DEFAULTS, p: "custom", a: "0.732", i: "41", m: "9.81", fv: "1.4", eta: "-1.5", jet: "0" };
    const h = encodeShare(v, DEFAULTS);
    expect(decodeShare("#" + h, LIMITS)).toEqual({ p: "custom", a: "0.732", i: "41", m: "9.81", fv: "1.4", eta: "-1.5", jet: "0" });
  });
  it("clamps to the slider's range and snaps to its step, written without trailing zeros like slider values", () => {
    expect(decodeShare("#a=1.5&i=-3&x=0.4999&fv=0.33&m=7.0000001", LIMITS)).toEqual({ a: "0.998", i: "1", x: "0.5", fv: "0.35", m: "7" });
  });
  it("ignores unknown keys, non-numbers, unknown presets and malformed pairs; accepts with or without #", () => {
    expect(decodeShare("#zz=3&a=abc&p=andromeda&x&=5&jet=maybe&i=12", LIMITS)).toEqual({ i: "12" });
    expect(decodeShare("i=12", LIMITS)).toEqual({ i: "12" });
    expect(decodeShare("", LIMITS)).toEqual({});
    expect(decodeShare("#a=Infinity&x=1e400", LIMITS)).toEqual({});
  });
  it("checkboxes are 0/1", () => {
    expect(decodeShare("#jet=0&play=1", LIMITS)).toEqual({ jet: "0", play: "1" });
  });
  it("percent-encoded values decode", () => {
    expect(decodeShare("#x=%2D2.5", LIMITS)).toEqual({ x: "-2.5" });
  });
});
