# Citation check — shard 1 of 3

Source file: `/home/mfho/Latex/Knowledge-Distillation-using-PySR-with-PRIYA-suite/oja_template.bib`

Sampling rule: entries where `(index mod 3) == 1` over the 28 entries in file order
(1, 4, 7, 10, 13, 16, 19, 22, 25, 28). `Sobol1993` (index 27) was additionally
verified because the task flagged it explicitly alongside `Owen2003` (index 28).

Verification method: WebSearch + WebFetch against ADS / arXiv / publisher (OUP, IOP,
APS, EDP, SciRP, Google Scholar). Every entry below was found to be a real paper with
the stated authors/title. Discrepancies are noted in the correction column.

| # | bibkey | status | correction (if WRONG) / note | source URL |
|---|--------|--------|------------------------------|------------|
| 1  | `2007MNRAS.382.1657K` | OK | Kim, Bolton, Viel, Haehnelt, Carswell (2007), MNRAS 382(4):1657-1674, DOI 10.1111/j.1365-2966.2007.12406.x, arXiv:0711.1862. All fields match. | https://academic.oup.com/mnras/article/382/4/1657/1147101 |
| 4  | `2023simsuite` | OK | Bird, Fernandez, Ho et al. (2023), "PRIYA: a new suite of Lyman-α forest simulations for cosmology", JCAP 2023(10):037, DOI 10.1088/1475-7516/2023/10/037, arXiv:2306.05471. ADS bibcode 2023JCAP...10..037B matches. | https://iopscience.iop.org/article/10.1088/1475-7516/2023/10/037 |
| 7  | `cranmer2023interpretablemachinelearningscience` | OK | Cranmer (2023), "Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl", arXiv:2305.01582 (astro-ph.IM). Real, correct eprint. | https://arxiv.org/abs/2305.01582 |
| 10 | `Hernquist` | OK | Hernquist, Katz, Weinberg, Miralda-Escudé (1996), ApJ 457:L51, DOI 10.1086/309899, arXiv:astro-ph/9509105. Authors/year/vol/DOI all match. NOTE: this is an ApJ Letters paper (true page L51-L55); bib gives `number={2}` and omits the page — minor incompleteness, DOI resolves correctly. | https://iopscience.iop.org/article/10.1086/309899 |
| 13 | `2014MNRAS.438.2499B` | OK | Bolton, Becker, Haehnelt, Viel (2014), MNRAS 438(3):2499-2507, DOI 10.1093/mnras/stt2374, arXiv:1308.4411. All fields match. | https://academic.oup.com/mnras/article/438/3/2499/972997 |
| 16 | `Jin_2024` | OK | Jin, Wolfson, Hennawi, González-Hernández, MNRAS 536(3):2277-2293, DOI 10.1093/mnras/stae2741, arXiv:2410.06505. Vol/issue/pages/DOI/authors match. NOTE: OUP issue date is Jan 2025; arXiv year is 2024, so `year=2024` is defensible (arXiv-year convention). | https://academic.oup.com/mnras/article/536/3/2277/7923533 |
| 19 | `Davis_2023` | OK | Davis & Jin (2023), "Discovery of a Planar Black Hole Mass Scaling Relation for Spiral Galaxies", ApJL 956(1):L22, DOI 10.3847/2041-8213/acfa98, arXiv:2309.08986. ADS 2023ApJ...956L..22D. All match. | https://iopscience.iop.org/article/10.3847/2041-8213/acfa98 |
| 22 | `Yang.2026PhRvD.113d3508Y` | OK | Yang, Bird, Ho, Qezlou (2026), "Design and optimization of neural networks for multifidelity cosmological emulation", PRD 113:043508, DOI 10.1103/cqrc-k8wq, arXiv:2507.07184. ADS 2026PhRvD.113d3508Y. All match (eid 043508 = ADS page d3508). | https://journals.aps.org/prd/abstract/10.1103/cqrc-k8wq |
| 25 | `Ma_2025` | OK | Ma, Bolton, Iršič, Gaikwad, Puchwein, "An improved model for the effect of correlated Si III absorption…", MNRAS 546(1), DOI 10.1093/mnras/staf2262, arXiv:2509.08613. Vol/issue/DOI/authors match. NOTE: journal publication is 2026 (ADS bibcode 2026MNRAS.546f2262M); `year=2025` / bibkey `Ma_2025` reflect the arXiv submission year — minor year mismatch, not an error in identity. | https://academic.oup.com/mnras/article/546/1/staf2262/8404157 |
| 27 | `Sobol1993` (flagged) | OK | Sobol', I. M. (1993), "Sensitivity estimates for nonlinear mathematical models", Math. Modelling & Computational Experiments **1(4):407-414**. The bib's `volume={1}, number={4}, pages={407--414}` is the canonical "MMCE, 1(4) (1993) 407–414" form used in Sobol/Saltelli's own later papers. (SciRP lists it as "volume 4" by conflating issue 4 with the volume — that is the erroneous variant; the bib entry is the correct one.) English translation of Matem. Modelirovanie 2(1), 1990. | http://www.andreasaltelli.eu/file/repository/Sobol_2001.pdf |
| 28 | `Owen2003` (flagged) | OK | Owen, Art B. (2003), "The dimension distribution and quadrature test functions", Statistica Sinica **13(1):1-17**. Bib gives volume 13, pages 1-17, year 2003 — all match. | https://www3.stat.sinica.edu.tw/statistica/j13n1/j13n11/j13n11.html |

## Summary

All 11 sampled entries (including both explicitly-flagged recently-added entries
`Sobol1993` and `Owen2003`) resolve to real, correctly-attributed papers. No fabricated,
mis-authored, or wrong-venue entries found in this shard. The only blemishes are minor:
two arXiv-vs-journal year conventions (`Jin_2024`, `Ma_2025`) and one missing Letters
page number (`Hernquist`) — none rise to WRONG. The flagged `Sobol1993` volume/number,
which is cited inconsistently across the web, is in fact the canonical MMCE 1(4) form.
