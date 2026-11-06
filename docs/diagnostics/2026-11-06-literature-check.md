# Literature check for the f32 accuracy paper (2026-11-06)

Scope: web search plus arXiv abstract and HTML pages, read through a summarising tool, and the WGSL specification read directly. **Not done: a NASA ADS full-text search,** the paywalled versions, or a citation trace. So "not found" below means not found in this search, not that it does not exist. Re-read every quotation in the paper itself before relying on it.

## Is the work new?

| Claim in the paper | What the search found | Standing |
|---|---|---|
| On-screen f32 against f64 error in a Kerr GPU ray tracer | GRay2 compares single and double for **speed only** (its Section IV). No paper found that measures the accuracy difference. | Looks like a real gap. Confirm on ADS. |
| Browser (WebGPU) Kerr ray tracing | Only hobby projects and notebooks: [Singularity](https://mal0ware.github.io/Singularity/about/), [Kerr geodesics notebook](https://rreusser.github.io/notebooks/kerr-geodesics). No paper found. | Looks new as a published setting. |
| Hamiltonian drift misleads as a diagnostic | Monitoring it is confirmed as common practice (below). No paper found that breaks it into f32 evaluation error versus real drift. | Looks new in this field. |
| f32 noise switching off a first-integral check in a step controller | Not found anywhere. | Looks new, but it is specific to a controller with such a check (see Al-Sayed Ali below). |
| Which rounding location sets the wall | **Partial overlap:** Al-Sayed Ali et al. 2026. They do not attribute it to a location. | New as a measurement, not as an idea. |
| Compensated summation helping an ODE integrator | **Not new:** Rein & Spiegel 2015; Antoñana et al. 2017; Hairer et al. | Cite it. The paper's contribution is the size here and why it is smaller. |
| Double-single on GPUs | **Not new:** Da Graça & Defour 2006; Thall 2006. Cost is usually quoted as about 10x or more, mostly from old hardware or forum posts. | Cite it. Our 25x to 85x is for a whole integrator on a current GPU, not single operations. |

## The closest prior work: read this one

**Al-Sayed Ali, Bernard, Marzorati, Rouzaud-Cornabas 2026, "Mixed-Precision in adaptive Runge-Kutta method for large ODE systems", [arXiv:2605.23727](https://arxiv.org/abs/2605.23727)** (submitted to SISC). Not about ray tracing. It uses BS3(2), the same 3rd-order pair as our order test.

- **It supports the wall:** single precision's "averaged local error plateaus around 10^-7 for tolerances lower than 10^-6". Our DP5 f32 wall also starts at tol 1e-6.
- **It supports the store finding, indirectly:** their mixed solvers store the solution in double and compute parts of the force in single, and Benchmark 1 has "an accuracy equivalent to Double" (Mixed2 is about 10x worse on Benchmark 2 at the strictest tolerances). That fits a store-set wall, but they do not test the location.
- **It sharpens the controller finding:** single's median step ratio against double is 1.0 on all three benchmarks. So f32 does not generally change an adaptive solver's steps. Ours changes because of its first-integral check with a noise floor. The paper should say this is a property of that check, not of adaptive RK in general.

Check the author list on arXiv: the abstract page and the HTML disagreed in the tool's output.

## How other codes use conserved quantities

| Code | What it does | Source |
|---|---|---|
| TARTARUS | Calls the residual "a direct, gauge-independent measure of the error" from "truncation and round-off". It "grows for every scheme as the photon approaches the event horizon" (a = 0), and the paper does not explain the growth. | [2609.08989](https://arxiv.org/html/2609.08989) |
| OSIRIS | Tracks it "to ensure physical results"; best scheme about 1e-11, worst case near the hole about 1e-10. | [2202.00086](https://ar5iv.labs.arxiv.org/html/2202.00086) |
| OCTOPUS | "An additional diagnostic for assessing numerical accuracy". Captured rays 1e-8 to 1e-6 against about 1e-13 for others: the same pattern as ours. | [2510.12585](https://arxiv.org/html/2510.12585) |
| GRay2 | u^2 is "a good measure of the numerical errors", measured in Cartesian Kerr-Schild coordinates, which avoid the horizon's coordinate singularity. | [1706.07062](https://arxiv.org/html/1706.07062v1) |
| GYOTO | **Imposes** conservation: it "modif[ies] the value of" the velocities, limited to 1 %, rather than only monitoring. | [1109.4769](https://ar5iv.labs.arxiv.org/html/1109.4769) |
| Skylight | Uses four constants of motion "as a test of the accuracy". | [2206.06429](https://ar5iv.labs.arxiv.org/html/2206.06429) |

The near-horizon growth in TARTARUS and OCTOPUS may be the same effect we measured, but neither paper states the coordinates or arithmetic in a way that lets me claim it. Offer it as a possible explanation, nothing stronger.

## WGSL, read directly ([W3C CRD, 26 October 2026](https://www.w3.org/TR/2026/CRD-WGSL-20261026/))

- **Types (6.2.4):** f32, plus f16 behind `enable f16`. No f64.
- **Reassociation (15.7.5):** "An implementation may reassociate operations." Fusion is allowed "if the transformed expression is at least as accurate".
- **fma (17.5.32):** the accuracy rule "allows an implementation which performs an ordinary multiply ... followed by an ordinary addition", so `fma` cannot be relied on for an exact product.
- **Rounding (15.7.2):** "No rounding mode is specified. An implementation may round an intermediate result up or down."
- **Accuracy:** x + y, x - y and x * y "Correctly rounded"; x / y 2.5 ULP.

**Why this matters:** two-sum's exactness assumes round-to-nearest, which the spec does not promise. Our probes passed on two GPUs, so that is an empirical result for those GPUs and compilers, not a guarantee. Anyone reusing this needs to run such a probe on their own hardware.

## Analytic alternatives a reviewer may raise

Kerr null geodesics have closed forms in elliptic functions, so "why integrate at all?" is a fair question. Have an answer ready.

- Gralla & Lupsasca 2020, [Phys. Rev. D 101, 044032](https://link.aps.org/accepted/10.1103/PhysRevD.101.044032); companion [arXiv:1910.12873](https://arxiv.org/abs/1910.12873)
- Dexter & Agol 2009, geokerr, [ApJ 696, 1616](https://iopscience.iop.org/article/10.1088/0004-637X/696/2/1616)
- YNOGK, [arXiv:1305.1250](https://arxiv.org/abs/1305.1250)
- AART: only a [meeting abstract](https://meetings-archive.aps.org/apr/2022/x13/9) found; find the software paper

## Reference details found

| Key in refs.bib | Fix or confirm |
|---|---|
| Tartarus2026 | Authors David F. Bambague, Alexis Larrañaga. Revista eSPECTRA 4(2), 2026. |
| Octopus2025 | Authors Shiyang Hu, Shijie Tan, Dan Li, Lina Zhang, Chen Deng, Wenfu Cao. No journal listed. |
| Chan2018GRay2 | [ApJ 867, 59](https://ui.adsabs.harvard.edu/abs/2018ApJ...867...59C) (2018). doi:10.3847/1538-4357/aadfe5 |
| Pu2016Odyssey | ApJ 820, 105 (2016). doi:10.3847/0004-637X/820/2/105 |
| Moscibrodzka2023 | Accepted to ApJS. doi:10.3847/1538-4365/acb6f9. Volume and pages still to find. |
| Osiris2022 | Eur. Phys. J. C 82 (2022). doi:10.1140/epjc/s10052-022-10054-0. Article number still to confirm. |
| Rein2015IAS15 | [MNRAS 446, 1424-1437](https://ui.adsabs.harvard.edu/abs/2015MNRAS.446.1424R/abstract) (2015). Quotation confirmed: "decreases E_rand by one to two orders of magnitude at almost no additional cost". |
| Chan2013GRay | Title and doi:10.1088/0004-637X/777/1/13 confirmed. |
| Not checked | Carter, Bardeen, Mino, Dormand-Prince, Bogacki-Shampine, Kahan, Knuth, Higham, Hairer, Hairer 2008. Classics, but still to verify. |

**To add:** Al-Sayed Ali et al. 2026; Antoñana, Makazaga & Murua 2017, Numer. Algorithms 76, 861-880, doi:10.1007/s11075-017-0287-z; Joldes, Muller & Popescu 2017, ACM TOMS 44(2), article 15 (double-word error bounds; it reports a previously published addition bound was wrong); [Da Graça & Defour 2006](https://arxiv.org/abs/cs/0603115); [Thall 2006](https://hgpu.org/?p=1806), SIGGRAPH poster; Gralla & Lupsasca 2020; Dexter & Agol 2009. Dekker 1971 is the classic source for the exact product; I did not check its details.
