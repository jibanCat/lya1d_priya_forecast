# Paper `\mfho` decision sheet — finishing `oja_template.tex`

**Paper:** `/home/mfho/Latex/Knowledge-Distillation-using-PySR-with-PRIYA-suite/oja_template.tex`
(branch `paper-additions`, HEAD `b83385b`). **44 live `\mfho{}` markers** (L63 is the
`\newcommand`; L848's is inside a `%`-comment). Every marker renders as red `[MFH: …]`, so
**all 44 must be removed before submission** — the question per item is whether removal needs a
*substantive* change or just accepting the already-drafted `\additions{}`.

Built 2026-07-01 from a transcript-recovery + 2-referee cross-check of the disconnected
2026-06-30 session. The paper is already **quantitatively consistent** with the production
knee-cut CSVs (every reframed table/figure/eq re-derived and matched); what remains is
editorial + a few correctness items below. Companion: `docs/HANDOFF_2026-06-30_perz_sobolev.md`.

---

## 0. Load-bearing context — the `grad_err` metric is now provably Fisher-consistent

Resolved 2026-07-01 (code-verified). The Fisher covariance + data vector are **linear P_F**
(no `np.log` in `fisher.py`; all three likelihood paths). The deployed emulator is the
**fiducial-anchored additive combine**, for which

    ∂P_F/∂θ |_fid  =  P_F^GP(fid) · ∂logP_eq/∂θ            (refit_taylor.py:262,280,456,474)

so in the ratio to the GP the anchor `P_F^GP(fid)` **cancels bin-by-bin**, and the equation's
**log**-slope ratio equals the deployed combine's **linear** (Fisher-space) slope ratio — the
exact quantity a Fisher analysis on a linear-P_F covariance consumes. Therefore the current
log-space `grad_err = median_k |∂logP_eq/∂θ ÷ ∂logP_GP/∂θ − 1|` is **exactly Fisher-consistent
for the deployed model, not a proxy**. `docs/pr_review/VERDICT.md:19-23` (which called it
linear-standalone) is **superseded** — it scored the un-deployed equation (banner added there).

**→ Paper action (strengthening, not a fix):** add one sentence at the `grad_err` definition:

> Because the deployed emulator is the fiducial-anchored additive combine, its linear
> (Fisher-space) slope at the fiducial point is
> $\partial P_F/\partial\theta|_{\rm fid} = P_F^{\rm GP}({\rm fid})\,\partial\log P_F^{\rm eq}/\partial\theta$;
> the anchor $P_F^{\rm GP}({\rm fid})$ cancels in the ratio to the GP, so the equation's
> log-slope ratio equals the combine's linear-slope ratio — the quantity a Fisher analysis on
> the linear-$P_F$ covariance would consume.

This also **settles Group B**: keep the one-sentence Fisher *motivation* (L522≡L683) — `grad_err`
genuinely *is* the Fisher-relevant slope faithfulness; there is no σ-claim, so nothing to retract
beyond the two stray σ-magnitude quantities (L252, L794).

---

## BLOCKING vs optional — at a glance

**BLOCKING (correctness / headline integrity — must be actively resolved):**
- Reframe: abstract **L95**, conclusion **L825** (drafts exist at L826).
- Retracted σ/Fisher magnitudes still live: **L252** (σ-ratio + footnote) and its **unmarked twin L794** (`10^6`–`10^19`).
- Eq conventions (answer once, identically): anchor **L229** ≡ combine **L315** (→ log-P).
- Table 1 priors **L174** (z_Hei/z_Hef/A_P vs production hypercube).
- Stale student numbers: **L627, L677** (+ unmarked **L601, L577-578, L634, L639**).
- Internal inconsistency: **§cost wall-time L807** still quotes niter=50 (production = 200; flagged from L859).
- Empty placeholder **L518** (resolution-correction subsection) — write or delete.
- Add the `grad_err` anchor-cancellation sentence (§0).

**OPTIONAL (editorial — accept-and-delete):** keep/drop dropped floats (L429/432/438, L586, L605/608, L641/644, L729, L741, L849, L864, L917); drafted-prose reads (L109, L546, L781, L940, L331, L359); formatting/dangling-sentence FYIs (L261, L445, L553, L565, L574, L579, L599, L628, L632, L637, L671, L675, L871).

---

## Group A — Reframe (abstract / conclusion) — BLOCKING

| Line | Decision | Recommendation |
|---|---|---|
| **95** | Abstract still sells value-accuracy, not derivative-faithfulness/Sobolev. | **Rewrite** to the actual headline; scope the count "nine of eleven … at z=3.6" (match L686/L872). |
| **825** | Conclusion tells the pre-reframe "dτ0 best / n_S worst" story; a reframed para is drafted at L826. | **Adopt the L826 draft, trim old L813-823;** keep one "value accuracy is not sufficient" setup sentence. |
| **109** | Intro is a bulleted outline; marker proposes a 5-part structure. | **Editorial.** Tighten toward the outline; add one contributions sentence naming Fisher's-Mirage/Sobolev. Delete marker regardless. |

## Group B — Retracted σ/Fisher passages still live — BLOCKING (see §0)

Keep the Fisher *motivation* (L522≡L683); remove only the two stray retracted *magnitudes*.

| Line | Decision | Recommendation + draft |
|---|---|---|
| **252** | Operator-motivation quotes `σ_PySR/σ_GP = 10^3–10^6×` + Fisher footnote (retracted quantity). | **Rewrite in grad_err terms, drop the footnote.** Draft: *"…the discovered equations' parameter slopes $\partial\log P_F/\partial\theta$ departed from the GP by more than an order of magnitude (grad\_err $\gg1$), because per-parameter gradients were dominated by oscillatory components rather than the smooth $\theta$ response encoded by the GP."* |
| **L794** *(no marker!)* | §multid_failure: joint fits "produced unreliable Fisher forecasts … exceeding the GP by factors of $10^6$ to $10^{19}$." | **Recast as Jacobian rank-collapse.** Draft: *"…it is unusable for inference: the parameter Jacobian $\partial P_F/\partial\theta$ was numerically rank-deficient, so several parameters share a single gradient direction and their joint response cannot be disentangled. The cause is structural."* |
| **522 ≡ 683** | Keep or drop the one-sentence Fisher motivation? | **KEEP** (per §0 — grad_err *is* the Fisher-relevant slope faithfulness). Answer both identically. |
| **871** | Old σ-ratio appendix table commented, replaced by `tab:per1d_eqs`. | **Accept (delete marker).** |

## Group C — Equation / normalization conventions — BLOCKING

| Line | Decision | Recommendation |
|---|---|---|
| **229** | `eq:norm` written linear-P + at-fid anchor; production trains log-P + empirical sweep-mean anchor. | **Match production: log-P, state the anchor explicitly** (the anchor is the mechanism the paper credits for retaining θ-dependence). |
| **315** | Combine eqs written linear-P; production sums in log-P then exponentiates. | **Rewrite in log-P** — same convention as L229 (answer the pair identically). |
| **331** | Old Phase-1.5 τ0/A_P ablation forms (dropped operators) commented, replaced by production forms. | **Accept.** Confirm L349/L355 = what Fig.`tau0_ap_pred` plots (source stamp L345 says yes). |
| **359** | Printed τ0/A_P eqs are maxsize=20 best-loss forms; more compact faithful members exist. | **Keep maxsize=20 forms** unless a cleaner illustration is worth a Fig.1 regen (both faithful). If swapped, Fig.1 MUST be regenerated with the same eq. |
| **261** | ANOVA-loss para superseded by Sobolev; commented. | **Accept.** |

## Group D — Table 1 priors — BLOCKING

| Line | Decision | Recommendation |
|---|---|---|
| **174** | Table 1 has z_Hei max **4.1** (hypercube **[3.5,4.5]**), z_Hef min **2.6** (hypercube **[2.2,3.2]**); confirm A_P range/units. | **Reconcile to the production hypercube** (`param_priors_table.tex` is the emulator-box truth). Load-bearing: defines the emulator validity domain + Sobol ranges. |

## Group E — Stale student numbers — BLOCKING (truth = `tab:stats_table`, z=3.6)

| Line | Decision | Recommendation + draft |
|---|---|---|
| **627** | Claims n_S HF **2.03%** / cross-param HF avg **0.6%** + "such small error … accurately predicted" (now false). | **Update + soften.** Draft: *"…and 5.16\% for the high fidelity with an average of $\sim$1.4\% across all. Thus $n_S$ has the largest deviation between the true and predicted values, and its higher high-fidelity error makes it the most challenging parameter to emulate accurately in 1 dimension."* |
| **677** | "2.04% for z=3.6" HF max stale; new HF max 10.78% (n_S)/10.50% (h); "clear precision" framing false. | **Update numbers + soften framing.** |
| **644 / 608** | Table 5 (z=2.8, off-grid) + Table 4 (stale hand-picked subsets). | **Cut both** (see Group F); superseded by `tab:multid`. |
| **L601, L577-578, L634, L639** *(no markers)* | Body de-norm RMSE/%-error student numbers (0.0021/2.035%, 0.0029/2.23%, 0.0032/2.26%, 0.3679/214.09%). | **Verify or cut** the specific numerics when rewriting the well-/challenging-parameter subsections — none appear in a refreshed table; easy to overlook. |

## Group F — Keep/drop dropped floats — OPTIONAL (confirm; all commented, restorable)

| Line | Float | Recommendation |
|---|---|---|
| 429/432/438 | Fig.2 holdout_validation | **Drop** (no per-z source; superseded by `multid_bestworst`+`tab:multid`). |
| 586 | Fig.5 denorm_dtau0-ap | **Drop.** |
| 605/608 | Table 4 rmse_pe_table | **Cut** (superseded by `tab:multid`). |
| 641/644 | Table 5 stats_28 (z=2.8) | **Cut** (z=2.8 off-grid). |
| 729 | Fig.10 maxsize_sens | **Lean restore** if space allows — cleanest single visual for "objective, not budget"; else drop (numbers survive in §4.4 + Fig.`ns_budget`). |
| 741 | Fig.11 crossz | **Drop** (z-robustness adequate in prose + the 8/9/6 counts). |
| 849 / 864 / 917 | App figure dump / Phase-1.5 ablation para / joint-multiparam appendix | **Drop** all (rank-collapse point stays in §multid_failure). |

## Group G — Missing content

| Line | Decision | Recommendation |
|---|---|---|
| **518** | Empty placeholder: resolution-correction subsection (P1D vs r, 0–1). | **Must resolve — cannot ship a red placeholder.** If the result exists, write a short subsection; else delete the placeholder AND drop the resolution-correction promise from intro/conclusion. |

## Group H — Drafted prose to read/approve — OPTIONAL

| Line | Recommendation |
|---|---|
| **546** | Read & approve; confirm the "same central finite-difference stencil" wording (corrected vs the earlier false "same stencil the Fisher matrix consumes" — see L940). |
| **781** | Read & approve, esp. the h basis-test wording (corr ≈ −0.25, ~6% variance). |
| **940** | **Approve** — correctness fix (retired the false "same stencil" claim + the budget-control overclaim; keep the hedged wording). |

## Group I — Formatting / dangling-sentence FYIs — OPTIONAL (accept-and-delete)

L445, L565 (figure widened); L553 ("read up to here" — stray bookmark, delete); L574/579/599/628/632/637 (sentences referencing deleted figures commented — verify no dangling `\ref`); L671/675 (rewrite the §hi_fid opener around `tab:stats_table`/`tab:multid` — ties into Group E); L871 (accept). **L859**: accept the production kwargs BUT **reconcile §cost wall-time L807 (niter=50 → 200)** — this half is BLOCKING (internal-consistency).

---

## Cross-cutting flags
1. **Two retracted-Fisher landmines have NO marker:** L794 (`10^6–10^19`) and the wall-time at L807 (L859 marks only the kwargs). Resolving only the marked lines leaves them.
2. **Two consistency chains — answer each pair identically:** L229 ≡ L315 (log-P); L522 ≡ L683 (keep Fisher motivation).
3. **Standardize numbers on:** z=3.6, 9/11 faithful; per-z 8/9/6 at z=2.6/3.6/4.2; five faithful at all three z; `tab:stats_table` for %-errors; `tab:faith_taxonomy` (knee) for grad_err. NOT the old best-loss `taxonomy_table.tex`/`per_param_equations.tex` on disk.
