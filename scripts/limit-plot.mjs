// Error against compute cost, one panel per configuration: docs/diagnostics/2026-11-06-accuracy-limit/limit-cost.svg
// Hue = method (fixed order), solid = f32, dashed = f64. Hover a marker for its values.
import { readFileSync, writeFileSync } from "node:fs";

const dir = "docs/diagnostics/2026-11-06-accuracy-limit";
const TOL = ["0.001", "0.0001", "0.00001", "0.000001", "1e-7"], F64 = [...TOL, "1e-8", "1e-9"], SCALE = ["1", "0.5", "0.25", "0.125"];
const METHODS = [
  { name: "DP5(4), 5th order, Mino", color: "#2a78d6", rhs: 6, f32: TOL.map((t) => `mino f32 ${t}`), f64: F64.map((t) => `mino f64 ${t}`) },
  { name: "BS3(2), 3rd order, Mino", color: "#eb6834", rhs: 3, f32: TOL.map((t) => `bs3 f32 ${t}`), f64: TOL.map((t) => `bs3 f64 ${t}`) },
  { name: "RK4, 4th order, affine, linear landing", color: "#1baf7a", rhs: 4, f32: SCALE.map((k) => `rk4 f32 ${k}`), f64: SCALE.map((k) => `rk4 f64 ${k}`) },
  { name: "DP5(4) compensated", color: "#6250d6", rhs: 6, f32: TOL.map((t) => `comp f32 ${t}`), f64: [] },
];
const PANELS = [["edge-998", "a = 0.998, i = 85, disk rays", "disk"], ["edge-998-nodisk", "a = 0.998, i = 85, no disk, sky rays", "sky"], ["default-9", "a = 0.9, i = 72, disk rays", "disk"], ["schw", "a = 0, i = 30, disk rays", "disk"]];
const W = 520, H = 330, L = 62, R = 16, T = 34, B = 44, X0 = 2, Y0 = 30;
const xMin = 1e2, xMax = 3e4, yMin = 1e-8, yMax = 1e-1;
const sx = (v) => L + (Math.log10(v / xMin) / Math.log10(xMax / xMin)) * (W - L - R);
const sy = (v) => T + (Math.log10(yMax / v) / Math.log10(yMax / yMin)) * (H - T - B);
const ink = "#1a1a19", muted = "#6b6b64", grid = "#e7e6e1";
const fmt = (v) => v.toExponential(1);
let body = "";
const decade = (a, b) => { const o = []; for (let e = Math.round(Math.log10(a)); e <= Math.round(Math.log10(b)); e++) o.push(e); return o; };
PANELS.forEach(([cfg, title, group], n) => {
  const r = JSON.parse(readFileSync(`${dir}/results/${cfg}.json`, "utf8"));
  const ox = X0 + (n % 2) * (W + 8), oy = Y0 + Math.floor(n / 2) * (H + 6);
  let g = `<g transform="translate(${ox},${oy})"><text x="${L}" y="20" font-size="14" font-weight="600" fill="${ink}">${title}</text>`;
  for (const e of decade(yMin, yMax)) if (e % 2 === 0 || e === -1) { const y = sy(10 ** e); g += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="${grid}"/><text x="${L - 6}" y="${y + 4}" font-size="11" text-anchor="end" fill="${muted}">1e${e}</text>`; }
  for (const e of [2, 3, 4]) { const x = sx(10 ** e); g += `<line x1="${x}" x2="${x}" y1="${T}" y2="${H - B}" stroke="${grid}"/><text x="${x}" y="${H - B + 16}" font-size="11" text-anchor="middle" fill="${muted}">${10 ** e}</text>`; }
  g += `<text x="${(L + W - R) / 2}" y="${H - 8}" font-size="11" text-anchor="middle" fill="${muted}">force evaluations per ray</text>`;
  g += `<text transform="translate(14,${(T + H - B) / 2}) rotate(-90)" font-size="11" text-anchor="middle" fill="${muted}">median on-screen error (px at 720p)</text>`;
  for (const m of METHODS) for (const [arms, dash] of [[m.f64, "5 4"], [m.f32, ""]]) {
    const pts = arms.map((k) => r.arms[k]).filter(Boolean).map((a, i) => ({ x: a.meanAttempts * m.rhs, y: a.groups[group].px.p50, k: arms[i] })).filter((p) => p.y > 0 && p.x >= xMin && p.x <= xMax);
    if (!pts.length) continue;
    g += `<polyline fill="none" stroke="${m.color}" stroke-width="2" ${dash ? `stroke-dasharray="${dash}"` : ""} points="${pts.map((p) => `${sx(p.x).toFixed(1)},${sy(Math.max(p.y, yMin)).toFixed(1)}`).join(" ")}"/>`;
    for (const p of pts) g += `<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(Math.max(p.y, yMin)).toFixed(1)}" r="4" fill="${dash ? "#fff" : m.color}" stroke="${m.color}" stroke-width="2"><title>${p.k}: ${fmt(p.y)} px, ${Math.round(p.x)} force evaluations</title></circle>`;
  }
  body += g + "</g>";
});
let lx = X0 + L, leg = "";
for (const m of METHODS) { leg += `<line x1="${lx}" x2="${lx + 22}" y1="14" y2="14" stroke="${m.color}" stroke-width="2"/><text x="${lx + 28}" y="18" font-size="12" fill="${ink}">${m.name}</text>`; lx += 44 + m.name.length * 6; }
leg += `<line x1="${lx}" x2="${lx + 22}" y1="14" y2="14" stroke="${muted}" stroke-width="2"/><text x="${lx + 28}" y="18" font-size="12" fill="${ink}">f32</text><line x1="${lx + 58}" x2="${lx + 80}" y1="14" y2="14" stroke="${muted}" stroke-width="2" stroke-dasharray="5 4"/><text x="${lx + 86}" y="18" font-size="12" fill="${ink}">f64</text>`;
const tw = 2 * W + 12, th = Y0 + 2 * H + 10;
writeFileSync(`${dir}/limit-cost.svg`, `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${tw} ${th}" width="${tw}" height="${th}" font-family="system-ui, sans-serif"><rect width="100%" height="100%" fill="#fff"/>${leg}${body}</svg>\n`);
console.log("wrote limit-cost.svg");
