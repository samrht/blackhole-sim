// Shareable links (2026-10-03): the view lives in the URL hash (#p=m87&x=0.5&play=0), never the query string,
// which the validation routes match by substring (?parity, ?golden, ?record, ...). Each field is a panel
// control; values are the control's own value (sliders in their own units, e.g. log10 mass), written only when
// they differ from the page defaults. Pure functions; main.ts reads/writes the controls (tests/share.test.ts).

export type ShareKind = "range" | "check" | "select";
export interface ShareField { key: string; id: string; kind: ShareKind; customOnly?: boolean }
export type ShareLimit = { kind: "range"; min: number; max: number; step: number } | { kind: "check" } | { kind: "select"; options: string[] };

/** Apply order: the preset first (it sets spin, inclination, mass, accretion and the jet), pause last.
 *  customOnly fields are implied by a preset, so they are written only for a Custom view. */
export const SHARE_FIELDS: ShareField[] = [
  { key: "p", id: "preset", kind: "select" },
  { key: "a", id: "spin", kind: "range", customOnly: true },
  { key: "i", id: "incl", kind: "range", customOnly: true },
  { key: "m", id: "mass", kind: "range", customOnly: true },
  { key: "acc", id: "acc", kind: "range", customOnly: true },
  { key: "x", id: "exp", kind: "range" },
  { key: "t", id: "ts", kind: "range" },
  { key: "ld", id: "ldelay", kind: "check" },
  { key: "fl", id: "turb", kind: "range" },
  { key: "hs", id: "flare", kind: "range" },
  { key: "jet", id: "jeton", kind: "check" },
  { key: "eta", id: "jeteff", kind: "range" },
  { key: "g", id: "jg", kind: "range" },
  { key: "fv", id: "fv", kind: "range" },
  { key: "sky", id: "sky", kind: "range" },
  { key: "d", id: "detail", kind: "range" },
  { key: "play", id: "playpause", kind: "check" },
];

/** Hash body (no '#') of the fields whose value differs from `defaults`; "" for the default view. */
export function encodeShare(values: Record<string, string>, defaults: Record<string, string>): string {
  const parts: string[] = [];
  for (const f of SHARE_FIELDS) {
    if (f.customOnly && values.p !== "custom") continue;
    const v = values[f.key];
    if (v !== undefined && v !== defaults[f.key]) parts.push(`${f.key}=${encodeURIComponent(v)}`);
  }
  return parts.join("&");
}

const decimals = (step: number) => (String(step).split(".")[1] ?? "").length;
/** Validated values by key from a hash (with or without '#'): unknown keys, non-numbers, unknown options and
 *  malformed pairs are dropped; numbers are clamped to the range and snapped to the step. */
export function decodeShare(hash: string, limits: Record<string, ShareLimit>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const pair of hash.replace(/^#/, "").split("&")) {
    const eq = pair.indexOf("=");
    if (eq <= 0) continue;
    const key = pair.slice(0, eq), lim = limits[key];
    let raw: string;
    try { raw = decodeURIComponent(pair.slice(eq + 1)); } catch { continue; }
    if (!lim || !SHARE_FIELDS.some((f) => f.key === key)) continue;
    if (lim.kind === "select") { if (lim.options.includes(raw)) out[key] = raw; continue; }
    if (lim.kind === "check") { if (raw === "0" || raw === "1") out[key] = raw; continue; }
    const n = raw.trim() === "" ? NaN : Number(raw);
    if (!Number.isFinite(n)) continue;
    const snapped = lim.min + Math.round((Math.min(lim.max, Math.max(lim.min, n)) - lim.min) / lim.step) * lim.step;
    out[key] = String(Number(Math.min(lim.max, snapped).toFixed(decimals(lim.step))));
  }
  return out;
}
