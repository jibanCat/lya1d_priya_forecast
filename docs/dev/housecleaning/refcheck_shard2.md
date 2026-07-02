# Citation check — shard 2 of 3

Scope: every 3rd entry of `oja_template.bib` (0-based index where `index mod 3 == 2`),
plus the explicitly-flagged recently-added entries `Sobol1993` and `Owen2003`.
Verified against ADS, arXiv, and journal pages. Date of check: 2026-06-09.

Entry ordering (0-based, as they appear in the .bib):
0 Kim · 1 ASTRID · **2 2022MNRAS.517.3200F** · 3 PRIYA · 4 Chabanier ·
**5 2021JCAP...05..033P** · 6 Cranmer · 7 Gunn&Peterson · **8 Croft_1998** ·
9 Hernquist · 10 Miralda_Escude · **11 2022MNRAS.509.2551H** · 12 Bolton ·
13 walther · **14 Cabayol_Garcia_2023** · 15 Jin · 16 Yang(PRL) · **17 Delgado_2022** ·
18 Davis · 19 Fernandez.2024 · **20 Ho.2025arXiv250918271H** · 21 Yang(PRD) ·
22 Planck · **23 Rogers_2021** · 24 Ma · 25 Bird_2019 · **26 Sobol1993** · 27 Owen2003

## Results

| bibkey | status | correction if wrong | source URL |
|---|---|---|---|
| `2022MNRAS.517.3200F` | OK | — Fernandez, Ho, Bird, "A multifidelity emulator for the Lyman-α forest flux power spectrum", MNRAS 517(3), 3200–3211, Dec 2022. Authors/year/vol/pages/DOI/eprint(2207.06445) all match. Bibkey == ADS bibcode. | https://ui.adsabs.harvard.edu/abs/2022MNRAS.517.3200F |
| `2021JCAP...05..033P` | OK | — Pedersen, Font-Ribera, Rogers, McDonald, Peiris, Pontzen, Slosar, "An emulator for the Lyman-α forest in beyond-ΛCDM cosmologies", JCAP 2021(05), 033, May 2021. eprint 2011.15127. Bibkey == ADS bibcode. | https://ui.adsabs.harvard.edu/abs/2021JCAP...05..033P |
| `Croft_1998` | OK | — Croft, Weinberg, Katz, Hernquist, "Recovery of the Power Spectrum of Mass Fluctuations from Observations of the Lyα Forest", ApJ 495(1), 44–62, 1998. DOI 10.1086/305289. Matches ADS 1998ApJ...495...44C. | https://ui.adsabs.harvard.edu/abs/1998ApJ...495...44C |
| `2022MNRAS.509.2551H` | OK | — Ho, Bird, Shelton, "Multifidelity emulation for the matter power spectrum using Gaussian processes", MNRAS 509(2), 2551–2565, Jan 2022. eprint 2105.01081. Bibkey == ADS bibcode. | https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.2551H |
| `Cabayol_Garcia_2023` | OK | — Cabayol-Garcia, Chaves-Montero, Font-Ribera, Pedersen, "A neural network emulator for the Lyman-α forest 1D flux power spectrum", MNRAS 525(3), 3499–3515, 2023. DOI 10.1093/mnras/stad2512, eprint 2305.19064. Authors/year/vol/pages match. | https://academic.oup.com/mnras/article/525/3/3499/7246913 |
| `Delgado_2022` | OK | — Delgado, Wadekar, Hadzhiyska, Bose, Hernquist, Ho, "Modelling the galaxy–halo connection with machine learning", MNRAS 515(2), 2733–2746, 2022. DOI 10.1093/mnras/stac1951. Matches ADS 2022MNRAS.515.2733D. | https://ui.adsabs.harvard.edu/abs/2022MNRAS.515.2733D/abstract |
| `Ho.2025arXiv250918271H` | OK | — Ho, Qezlou, Bird, Yang, Avestruz, Fernandez, Iršič, "Small-scale Lyman alpha forest cosmology with PRIYA: Constraints from XQ100 and KODIAQ-SQUAD…", arXiv:2509.18271, Sep 2025. eprint/DOI/bibcode all match. | https://arxiv.org/abs/2509.18271 |
| `Rogers_2021` | OK | — Rogers, Peiris, "Strong Bound on Canonical Ultralight Axion Dark Matter from the Lyman-Alpha Forest", PRL 126(7), 071302, Feb 2021. DOI 10.1103/PhysRevLett.126.071302. Matches ADS 2021PhRvL.126g1302R. | https://ui.adsabs.harvard.edu/abs/2021PhRvL.126g1302R/abstract |
| `Sobol1993` | OK | — Sobol', "Sensitivity estimates for nonlinear mathematical models", Math. Modelling and Computational Experiments, vol 1, no 4, pp 407–414, 1993. The bib's vol=1/no=4/pp=407–414/1993 is the **canonical** citation (Saltelli textbooks, Wikipedia). NOTE: some databases (SciRP, scispace) collapse this to "vol 4, 407–414" by putting the issue number in the volume field — that is a database artifact, not an error in the .bib. English translation of a 1990 Russian original (Matem. Modelirovanie 2(1), 112–118). Entry is correct. | https://scispace.com/papers/sensitivity-estimates-for-nonlinear-mathematical-models-3txyo9j7gg |
| `Owen2003` | OK | — Owen, "The dimension distribution and quadrature test functions", Statistica Sinica 13 (2003), 1–17. Journal/vol/pages/year confirmed verbatim from the official Statistica Sinica PDF ("Statistica Sinica 13(2003), 1-17"). bib omits the issue number (1) but that is optional, not an error. | https://www3.stat.sinica.edu.tw/statistica/oldpdf/A13n11.pdf |

## Notes / cross-checks
- All ADS-bibcode-style bibkeys (`2022MNRAS.517.3200F`, `2021JCAP...05..033P`,
  `2022MNRAS.509.2551H`, `Ho.2025arXiv250918271H`) are internally consistent: the
  bibkey reproduces the ADS bibcode and matches the `adsurl`/`eprint`/`doi` fields.
- No fabricated, mis-attributed, or wrong-year entries found in this shard.
- `Sobol1993` was singled out for special attention: it is real and the volume/issue
  are correct as written, despite a common third-party "vol 4" mis-listing.
- `Owen2003` was singled out for special attention: real, verified against the
  publisher's own PDF; all fields correct.
