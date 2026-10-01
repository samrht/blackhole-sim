import { lambdaFromMdot, lambdaForPeakTemperature } from "./units";

export interface Preset {
  id: string; name: string;
  massSun: number; a: number; inclDeg: number;
  /** Accretion: fraction of the Eddington luminosity (units.ts mdotFromLambda). */
  lambda: number;
  /** Jet strength (the Jet slider). The jet is phenomenological, so where an observation bounds
   *  its brightness near the hole the value is calibrated to it; otherwise the model default 1. */
  jetStrength: number;
  caption: string;
}

/** Real-object presets (spec 2026-10-01 §3). Sources in each caption. */
export const PRESETS: readonly Preset[] = [
  {
    id: "m87", name: "M87*", massSun: 6.5e9, a: 0.9, inclDeg: 17,
    // Mdot (3-20)e-4 M_sun/yr (EHT M87* Paper VIII); log-midpoint 7.7e-4.
    lambda: lambdaFromMdot(6.5e9, 0.9, 7.7e-4),
    // At horizon scale the EHT sees the ring dominate: no jet within this view, and the jet base
    // (2021 data, ~85 r_g out) at <= ~60 mJy, ~10 % of the ~0.6 Jy ring. Measured in this render at
    // strength 1 (GPU, 640x360, 5 animation times): jet light = 29.8x the disk's. So 0.1 / 29.8.
    jetStrength: 0.1 / 29.8,
    caption: "EHT 2019: 6.5 billion solar masses (Paper VI); seen 17° from its jet (Mertens et al. 2016, Walker et al. 2018); accretion (3–20)×10⁻⁴ M☉/yr (Paper VIII). Spin is not measured: 0.9 is a common model value. Caveat: M87*'s real flow is hot, thick and radio-bright, not the thin disk drawn here; the colour is what a thin disk at this accretion rate would emit. The jet is dimmed to ~10 % of the disk's light: at horizon scale the EHT sees the ring dominate and the jet base at no more than ~10 % of the ring (2021 data).",
  },
  {
    id: "sgra", name: "Sagittarius A*", massSun: 4.3e6, a: 0.94, inclDeg: 30,
    // Mdot ~1e-8 M_sun/yr (radiatively inefficient flow; L ~ 2e-9 L_Edd).
    lambda: lambdaFromMdot(4.3e6, 0.94, 1e-8), jetStrength: 0,
    caption: "Our galaxy's centre: 4.3 million solar masses (GRAVITY). Spin 0.94 and a 30° view are EHT 2022 Paper V's best-bet model values, preferences rather than measurements; accretion ~10⁻⁸ M☉/yr. Caveat: like M87*, the real flow is not a thin disk.",
  },
  {
    id: "cygx1", name: "Cygnus X-1", massSun: 21.2, a: 0.998, inclDeg: 27, lambda: 0.02, jetStrength: 1,
    caption: "The first known stellar black hole: 21.2 ± 2.2 solar masses and spin above 0.9985 (Miller-Jones et al. 2021), seen at ~27° (Orosz et al. 2011), accreting at ~2 % of Eddington. Spin sits at the slider's 0.998 cap (the Thorne limit); the measured value is higher. The real disk peaks in X-rays; you see its blue-white visible tail. The jet's brightness this close to the hole has not been measured; it is shown at the model's default.",
  },
  {
    id: "grs1915", name: "GRS 1915+105", massSun: 12.4, a: 0.98, inclDeg: 60, lambda: 0.3, jetStrength: 1,
    caption: "A microquasar with a powerful jet: 12.4 solar masses, jet seen at 60° ± 5°, spin ~0.98 (Reid et al. 2014), accreting at ~30 % of Eddington. The real disk peaks in X-rays; you see its blue-white visible tail. The jet's brightness this close to the hole has not been measured; it is shown at the model's default (it points away from us, so it is faint).",
  },
  {
    id: "gargantua", name: "Gargantua (Interstellar)", massSun: 1e8, a: 0.6, inclDeg: 85,
    // "Anemic" disk about as hot as the Sun's surface (Thorne, The Science of Interstellar).
    lambda: lambdaForPeakTemperature(1e8, 0.6, 5800), jetStrength: 0,
    caption: "Fictional. 100 million solar masses (Thorne, The Science of Interstellar); the film's disk was rendered at spin 0.6 for the visuals (James, von Tunzelmann, Franklin & Thorne 2015), an 'anemic' disk about as hot as the Sun's surface. The 85° view is our choice to resemble the film. The film removed Doppler colour and brightness shifts; this render keeps them, which is why one side is brighter here.",
  },
];

/** Custom mode's starting mass and accretion: today's 30,000 K disk at a = 0.9. */
export const CUSTOM_DEFAULT = { massSun: 1e8, lambda: lambdaForPeakTemperature(1e8, 0.9, 3e4) };
