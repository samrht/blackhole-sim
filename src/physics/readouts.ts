import { iscoRadius, photonOrbit } from "./orbits";
import { gravRadius, gravTime, iscoPeriod, peakTemperature } from "./units";
import { lumNormFor } from "./lookups";

export interface Readouts {
  tPeakK: number; lumNorm: number;
  horizonM: number; iscoM: number; photonM: number;   // metres
  iscoPeriodS: number;                                // seconds
  realSecondsPerScreenSecond: number;                 // playback scale
}
/** Everything the panel shows in physical units, from the physical state. speedMPerSecond is the
 *  animation's coordinate time (in M) per real second at timeScale 1 (main.ts SPEED). */
export function computeReadouts(p: { massSun: number; a: number; lambda: number; timeScale: number }, speedMPerSecond: number): Readouts {
  const rg = gravRadius(p.massSun), tPeakK = peakTemperature(p.massSun, p.a, p.lambda);
  return {
    tPeakK, lumNorm: lumNormFor(tPeakK),
    horizonM: (1 + Math.sqrt(Math.max(0, 1 - p.a * p.a))) * rg,
    iscoM: iscoRadius(p.a, true) * rg,
    photonM: photonOrbit(p.a, true) * rg,
    iscoPeriodS: iscoPeriod(p.massSun, p.a),
    realSecondsPerScreenSecond: speedMPerSecond * p.timeScale * gravTime(p.massSun),
  };
}
