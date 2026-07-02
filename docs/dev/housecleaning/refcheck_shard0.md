# Citation check — shard 0 of 3

Verified by web search (ADS / arXiv / publisher). Entries selected from
`oja_template.bib` where (entry index mod 3) == 0, i.e. the 1st, 4th, 7th, ...
entries, plus the two recently-added stats references (`Sobol1993`, `Owen2003`)
which were flagged for special attention.

Entry indexing (0-based) used for selection:
0 `2007MNRAS.382.1657K`, 3 `2023simsuite`, 6 `cranmer2023...`, 9 `Hernquist`,
12 `2014MNRAS.438.2499B`, 15 `Jin_2024`, 18 `Davis_2023`,
21 `Yang.2026PhRvD.113d3508Y`, 24 `Ma_2025`. Plus `Sobol1993`, `Owen2003`.

| bibkey | status | correction if wrong | source URL |
|---|---|---|---|
| `2007MNRAS.382.1657K` | OK | none — Kim, Bolton, Viel, Haehnelt, Carswell 2007, MNRAS 382, 1657-1674, doi 10.1111/j.1365-2966.2007.12406.x, arXiv:0711.1862 all match | https://academic.oup.com/mnras/article/382/4/1657/1008613 ; https://arxiv.org/abs/0711.1862 |
| `2023simsuite` | OK | none — Bird, Fernandez, Ho, Qezlou, Monadi, Ni, Chen, Croft, Di Matteo 2023, JCAP 10(2023)037, doi 10.1088/1475-7516/2023/10/037, arXiv:2306.05471, ADS 2023JCAP...10..037B all match | https://iopscience.iop.org/article/10.1088/1475-7516/2023/10/037 ; https://arxiv.org/abs/2306.05471 |
| `cranmer2023interpretablemachinelearningscience` | OK | none — Miles Cranmer 2023, "Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl", arXiv:2305.01582 (astro-ph.IM) all match | https://arxiv.org/abs/2305.01582 |
| `Hernquist` | OK | none — Hernquist, Katz, Weinberg, Miralda-Escude 1996, ApJ 457 (no. 2), L51, doi 10.1086/309899 all match. (Bib omits a `pages` field; it is a Letter, L51-L56. Title/vol/number/year/DOI/authors correct.) | https://iopscience.iop.org/article/10.1086/309899 ; https://arxiv.org/abs/astro-ph/9509105 |
| `2014MNRAS.438.2499B` | OK | none — Bolton, Becker, Haehnelt, Viel 2014, MNRAS 438(3), 2499-2507, doi 10.1093/mnras/stt2374, arXiv:1308.4411 all match | https://academic.oup.com/mnras/article/438/3/2499/972997 ; https://arxiv.org/abs/1308.4411 |
| `Jin_2024` | OK (minor year ambiguity) | Authors/title/DOI(10.1093/mnras/stae2741)/vol 536/issue 3/pages 2277-2293 all correct. Bib `year=2024, month=dec` reflects the online-publication date; Oxford assigns the formal issue to Jan 2025. Not an error — advance-access vs issue date. arXiv:2410.06505. | https://academic.oup.com/mnras/article/536/3/2277/7923533 ; https://arxiv.org/abs/2410.06505 |
| `Davis_2023` | OK | none — Davis & Jin 2023, "Discovery of a Planar Black Hole Mass Scaling Relation for Spiral Galaxies", ApJL 956(1), L22, doi 10.3847/2041-8213/acfa98, arXiv:2309.08986 all match | https://iopscience.iop.org/article/10.3847/2041-8213/acfa98 ; https://arxiv.org/abs/2309.08986 |
| `Yang.2026PhRvD.113d3508Y` | OK | none — Yang, Bird, Ho, Qezlou 2026, "Design and optimization of neural networks for multifidelity cosmological emulation", PRD 113, 043508, doi 10.1103/cqrc-k8wq, arXiv:2507.07184, ADS 2026PhRvD.113d3508Y all match (bibcode "d3508" = eid 043508 in ADS volume coding) | https://journals.aps.org/prd/abstract/10.1103/cqrc-k8wq ; https://arxiv.org/abs/2507.07184 |
| `Ma_2025` | OK (minor year ambiguity) | Ma, Bolton, Irsic, Gaikwad, Puchwein, "An improved model for the effect of correlated Si III absorption ...", MNRAS 546(1), doi 10.1093/mnras/staf2262, arXiv:2509.08613 all match. Bib `year=2025` reflects online publication; ADS assigns the issue to 2026 (2026MNRAS.546f2262M). Advance-access vs issue date, not an error. | https://academic.oup.com/mnras/article/546/1/staf2262/8404157 ; https://arxiv.org/abs/2509.08613 |
| `Sobol1993` | OK | none — Sobol' I. M. 1993, "Sensitivity estimates for nonlinear mathematical models", Math. Modelling Comput. Exp. (MMCE), Vol. 1, No. 4, pp. 407-414. Bib's vol 1 / no. 4 / pp. 407-414 / 1993 is the canonical English-translation citation. (Some secondary sources mislabel it "vol. 4" by conflating the issue number; the bib is the correct form. Russian original: Mat. Model. 1990, 2(1), 112-118.) | https://scholar.google.com/scholar_lookup?title=Sensitivity+estimates+for+nonlinear+mathematical+models&author=I.+M.+Sobol&publication_year=1993&journal=Math.+Model.+Comput.+Exp.&volume=1&pages=407-414 |
| `Owen2003` | OK | none — Owen, Art B. 2003, "The dimension distribution and quadrature test functions", Statistica Sinica, Vol. 13(1), pp. 1-17. Bib's vol 13 / pp. 1-17 / 2003 all match. | https://www.jstor.org/stable/24307017 ; https://artowen.su.domains/reports/ddqtf.pdf |

## Notes
- No fabricated, mis-attributed, or wrong-year entries found in this shard.
- The two flagged stats references (`Sobol1993`, `Owen2003`) are both REAL and
  correctly cited. `Sobol1993` is the well-known global-sensitivity / Sobol'-index
  paper; `Owen2003` is the effective-dimension / dimension-distribution paper.
  Both are appropriate supporting cites for an ANOVA / Sobol-decomposition or
  effective-dimensionality argument.
- The only nits are two advance-access vs issue-date year mismatches (`Jin_2024`,
  `Ma_2025`), both internally consistent (bibkey year == bib `year` field) and
  not factual errors. If strict ADS-issue-year consistency is desired, they could
  be bumped to 2025 and 2026 respectively, but this is cosmetic.
