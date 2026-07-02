# arXiv Preprint — Design Spec

**Status:** design spec (exploratory deliverable) · **Date:** 2026-07-02 · **Supersedes scope of:** `docs/superpowers/plans/2026-07-02-narrative-rarr-update.md` (that Act-11 update is now a sub-part of this larger preprint).

**One-line goal:** Produce a beautiful, publication-quality arXiv preprint that tells the full story of the OWASP Top 10 for LLM Applications (2026) incident-data analysis — the corpus, the ranking method, the 2025→2026 changes, what the data says, and how robust the ranking is — written so a cybersecurity professional with no data-science background can follow every step.

**Authors:** Kyriakos "Rock" Lambros (OWASP GenAI Security Project — Top 10 for LLM Applications, Co-Lead) and Steve Wilson (OWASP GenAI Security Project — Top 10 for LLM Applications, Founder & Co-Lead). *(Steve reviews and approves the final text before release — owner-managed, outside this spec.)*

---

## 1. Audience & voice
Primary readers: **cybersecurity professionals**, spanning DS-novice to DS-expert. Policy: **every data-science concept is explained in plain language on first use**, in a boxed sidebar, and defined again in a glossary appendix. The main argument (the "spine") stays readable end-to-end without the boxes; the boxes are optional depth. No jargon without an inline gloss.

## 2. What this is — and is NOT (scope/authority statement) — *[remediation #5]*
The preprint MUST state, up front and unambiguously:
- This is an **incident-data analysis** authored by two working-group members. It is **not the official OWASP Top 10 release**, does not supersede the official list or process, and does not speak for OWASP.
- "The 2026 list" in this paper refers to the working group's expert-driven candidate ranking; this paper **stress-tests** that ranking against incident data — it does not set it.
- Framing throughout is **exploratory and internally rigorous**, not a peer-reviewed finding.

## 3. Contribution & claims (calibrated) — *[remediations #2, #3, #4, #7]*
**Retitled** away from "validating"/"first":
> **"Incident-Data Robustness Analysis of the OWASP Top 10 for LLM Applications (2026): How a Community-Expert Ranking Holds Up Against a Large-Scale LLM Incident Corpus."**

Honest contribution, stated plainly:
1. We assembled a **large-scale** (7,714-incident) LLM-security incident corpus and bound it to the OWASP LLM taxonomy. *(Claim is "large-scale," not "first-ever"; if a first-of-kind claim is made at all, it is narrowed to "first to bind incident data to the OWASP LLM taxonomy at this scale" and only if defensible — otherwise dropped.)*
2. We derived an **incident-based ranking** via a Bayesian measurement-error model.
3. **Honest finding = weak but robust.** Incident data agrees only weakly with the expert ranking (Cohen's κ ≈ 0.20, wide CI) — it neither strongly confirms nor overturns it. But the **expert ranking is robust**: a pre-registered 4-frontier-model bake-off plus a ground-truth stress test (Spearman ρ ≈ 0.918, bootstrap median Δ ≈ 0) does not move it. The data functions as a **bounded corrective at 25% weight**, consistent with a weak-but-stable signal.
4. The paper's value is **transparency + robustness**, not "the data proves the experts right." This framing must be consistent across the abstract, Part I, and the conclusion (no section may imply the data "validated" the list).

**Thesis-coherence rule:** the paper must explicitly reconcile "weak agreement (κ≈0.20) + robust ranking + 25% data weight" — the 0.75/0.25 blend uses data as a quarter-weight corrective (strong enough to move a tier on a large gap, not to overturn consensus on one noisy corpus), and robustness means the ranking is *stable*, which is the finding, not a disappointment.

## 4. Document structure
Expands the existing notebook (`notebooks/2026_top_10_llm_update_what_the_data_says.ipynb`); the current Acts become Part II.

- **Front matter:** title, authors + affiliations, abstract (calibrated per §3), and a short **"How to read this report"** (audience note + pointer to the glossary).
- **Part I — The List and How It's Made:** what the OWASP LLM Top 10 is; the incident corpus (7,714; CVE/GHSA/OSV/AIAAIC + Corpus B corroboration); the **0.75 expert / 0.25 data blend** explained from scratch and *why* that weighting (source: `docs/BLENDED-TOP10-METHODOLOGY.md`); the **2025 → 2026 changes** — rank moves + biggest movers, the **6 NEW-\*** candidate categories, the **4 ROLL-\*** rollups (source: `taxonomy.json` + methodology doc).
- **Part II — What the Incident Data Says** (the existing Acts, reframed): corpus → classification → classifier accuracy → Bayesian model → incident-derived rankings → expert-vs-incident agreement (κ) → disagreements → what the data can't see.
- **Part III — Robustness (RARR):** the 4-frontier-model bake-off (winner=None), ground-truth validation, recall-correction test → the ranking holds up.
- **Limitations & Independent-Review Status** *(decision (a); [remediation #4])*: single-author goldset (κ, override rate), interim reviewers, corpus caveats (stratum imbalance, OOS blind spot), the "exploratory, pending independent adjudication" framing, and what full external publication would require. Candid, not alarmist — the old `NON-PUBLISHABLE` banner is replaced by this section.
- **Reproducibility + Appendices:** data card (corpus provenance, temporal scope, inclusion), Bayesian model spec, threats register, and a **glossary** (precision/recall, gold set, prior/posterior, credible interval, MCMC, κ, balanced accuracy, bootstrap, Spearman, out-of-scope, negative-binomial measurement-error model).

## 5. Content sources & number integrity — *[remediation #6]*
- **Every statistic is computed in-notebook from committed cycle artifacts** — never hand-typed or transcribed from prose. Sources: `cycles/2026/` (ranking, κ, posteriors, concordance), `cycles/2026-rarr/results/` (RARR + `robustness_validation.json`), `baselines/2026/` (frozen previous ranking + votes), `docs/BLENDED-TOP10-METHODOLOGY.md` values re-derived from `respondent_rankings.npy` + λ where possible.
- **Reconcile the stale narrative:** the current narrative numbers predate the RARR/calibration fixes (Jun 22); re-executing the notebook against the current engine produces the authoritative numbers. Any change from the previously-rendered narrative is expected and must be the *current* value.
- **Consistency check cell:** a notebook cell asserts the displayed 2025→2026 table and the 0.75/0.25 blend match the committed artifacts (fails loudly on drift).

## 6. Architecture / build — *[Approach ①, remediation #9]*
- **Notebook-driven, single source of truth.** The `.ipynb` holds all authored content + chart code, reading committed data (reproducible). Workflow (your directive): **update notebook → execute end-to-end → pull fresh figures → build PDF.**
- **Figures** regenerate in-notebook at **print resolution (300 dpi)** into `notebooks/narrative/figures/`, sized to text width.
- **PDF build:** export the executed notebook to markdown, then compile with a **custom arXiv LaTeX template** (author/affiliation block, running header, numbered sections + TOC, print fonts, figure-width control) via the existing pandoc + xelatex toolchain.
- **Stub-first de-risk:** compile a 2-page stub through the arXiv template (author block + one figure + one section) and confirm it renders *before* pouring in full content.
- **Reproducibility note:** because the notebook reads committed data, `execute → build` reproduces the PDF; the exact toolchain (pandoc version, template, xelatex) is pinned in the reproducibility appendix.

## 7. New figures (executed in-notebook, 300 dpi)
- A **2025→2026 rank-change** figure (bump/dumbbell of the blended list, biggest movers highlighted).
- An **entry-expansion map** (10 incumbents / 6 NEW-\* / 4 ROLL-\*).
- A **RARR robustness** figure (ranking-fidelity ρ vs truth, or the bake-off table rendered as a figure).
- Re-render the existing 16 figures at print resolution.

## 8. Novice-explanation policy & scope discipline — *[remediation #8]*
- Concepts get a **boxed sidebar** on first use + a **glossary** entry. Findings stay in the spine; pedagogy in boxes.
- **Target length:** a focused preprint (roughly 20–30 pages incl. figures/appendices) — comprehensive but not a textbook. "More detail" lives in appendices/boxes, not the argument spine.

## 9. Acceptance criteria (verification)
1. Notebook executes end-to-end with no errors; all figures regenerate.
2. PDF compiles through the arXiv template; visually inspected — figures sized to width, no overflow, author block correct, TOC/section numbers aligned.
3. **No hand-typed statistics** — every number traces to a committed artifact; the consistency-check cell passes.
4. Title/abstract/conclusion are consistent with the calibrated framing (§3) — no "validated"/"first-ever" overclaim; robustness + weak-signal stated honestly.
5. Scope/authority statement (§2) present up front; Limitations section (§5-doc) present; glossary present.
6. Passes the AI-slop guard; human voice throughout; no AI attribution anywhere.
7. Steve's review/approval is obtained by the owner before any submission (out-of-band).

## 10. Non-goals (out of scope)
- Not resolving the single-author goldset via a new independent adjudication (disclosed as a limitation instead; may be a future cycle).
- Not automating the 0.75/0.25 blend into the engine (documented + reproduced in-notebook; engine automation is future work).
- Not the official OWASP release or process.
- Not peer review — this is a preprint.

## 11. Open items for user review
- Title wording (§3) — approve or tweak.
- Target length (§8) — acceptable?
- Any additional story beats to include or cut from §4.
