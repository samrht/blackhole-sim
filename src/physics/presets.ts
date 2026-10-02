import { lambdaFromMdot, lambdaForPeakTemperature } from "./units";

export interface Preset {
  id: string; name: string;
  massSun: number; a: number; inclDeg: number;
  /** Accretion: fraction of the Eddington luminosity (units.ts mdotFromLambda). */
  lambda: number;
  /** Jet on/off; its brightness follows from the energy budget (spec 2026-10-02). */
  jet: boolean;
  caption: string;
}

/** Real-object presets (spec 2026-10-01 §3). Sources in each caption. */
export const PRESETS: readonly Preset[] = [
  {
    id: "m87", name: "M87*", massSun: 6.5e9, a: 0.9, inclDeg: 17,
    // Mdot (3-20)e-4 M_sun/yr (EHT M87* Paper VIII); log-midpoint 7.7e-4.
    lambda: lambdaFromMdot(6.5e9, 0.9, 7.7e-4),
    jet: true,
    caption: "EHT 2019: 6.5 billion solar masses (Paper VI); seen 17° from its jet (Mertens et al. 2016, Walker et al. 2018); accretion (3–20)×10⁻⁴ M☉/yr (Paper VIII). Spin is not measured: 0.9 is a common model value. Caveat: M87*'s real flow is hot, thick and radio-bright, not the thin disk drawn here; the colour is what a thin disk at this accretion rate would emit. Its jet's brightness follows from energy conservation: it radiates ε (≈ 0.2 % by default, from M87's optical nucleus vs its jet power) of its Blandford–Znajek power as visible light. The EHT's ring is radio emission from a hot flow this renderer does not model.",
  },
  {
    id: "sgra", name: "Sagittarius A*", massSun: 4.3e6, a: 0.94, inclDeg: 30,
    // Mdot ~1e-8 M_sun/yr (radiatively inefficient flow; L ~ 2e-9 L_Edd).
    lambda: lambdaFromMdot(4.3e6, 0.94, 1e-8), jet: false,
    caption: "Our galaxy's centre: 4.3 million solar masses (GRAVITY). Spin 0.94 and a 30° view are EHT 2022 Paper V's best-bet model values, preferences rather than measurements; accretion ~10⁻⁸ M☉/yr. Caveat: like M87*, the real flow is not a thin disk.",
  },
  {
    id: "cygx1", name: "Cygnus X-1", massSun: 21.2, a: 0.998, inclDeg: 27, lambda: 0.02, jet: true,
    caption: "The first known stellar black hole: 21.2 ± 2.2 solar masses and spin above 0.9985 (Miller-Jones et al. 2021), seen at ~27° (Orosz et al. 2011), accreting at ~2 % of Eddington. Spin sits at the slider's 0.998 cap (the Thorne limit); the measured value is higher. The real disk peaks in X-rays; you see its blue-white visible tail. Its jet uses the same energy budget; the jet's brightness this close to the hole has not been measured.",
  },
  {
    id: "grs1915", name: "GRS 1915+105", massSun: 12.4, a: 0.98, inclDeg: 60, lambda: 0.3, jet: true,
    caption: "A microquasar with a powerful jet: 12.4 solar masses, jet seen at 60° ± 5°, spin ~0.98 (Reid et al. 2014), accreting at ~30 % of Eddington. The real disk peaks in X-rays; you see its blue-white visible tail. Its jet uses the same energy budget; the jet's brightness this close to the hole has not been measured.",
  },
  {
    id: "gargantua", name: "Gargantua (Interstellar)", massSun: 1e8, a: 0.6, inclDeg: 85,
    // "Anemic" disk about as hot as the Sun's surface (Thorne, The Science of Interstellar).
    lambda: lambdaForPeakTemperature(1e8, 0.6, 5800), jet: false,
    caption: "Fictional. 100 million solar masses (Thorne, The Science of Interstellar); the film's disk was rendered at spin 0.6 for the visuals (James, von Tunzelmann, Franklin & Thorne 2015), an 'anemic' disk about as hot as the Sun's surface. The 85° view is our choice to resemble the film. The film removed Doppler colour and brightness shifts; this render keeps them, which is why one side is brighter here.",
  },
];

/** Custom mode's starting mass and accretion: today's 30,000 K disk at a = 0.9. */
export const CUSTOM_DEFAULT = { massSun: 1e8, lambda: lambdaForPeakTemperature(1e8, 0.9, 3e4) };
