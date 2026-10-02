// Prints FLICKER_TABLE for src/physics/emission.ts. Run: npx vite-node scripts/calibrate-turbulence.ts
// (~10 min, single thread). Re-run whenever TURB changes.
import { diskFlickerRms, FULL_GRID } from "../src/physics/turbulence-calibration";
const S = [0.05, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];
const rms = diskFlickerRms(S, FULL_GRID);
console.log(`export const FLICKER_TABLE: readonly (readonly [number, number])[] = [[0, 0], ${S.map((s, i) => `[${s}, ${rms[i].toFixed(5)}]`).join(", ")}];`);
