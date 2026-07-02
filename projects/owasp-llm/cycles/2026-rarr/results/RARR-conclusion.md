# RARR (Recall-Aware Robustness Re-analysis) — Conclusion

**Cycle:** 2026-rarr · **Date:** 2026-07-01 · **Status:** ground-truth-validated

## Question
Does re-analyzing the 2026 OWASP GenAI incidence ranking with frontier LLM classifiers
(vs the 2026 Stage-1 classifier, llama-3.3-70B) change the ranking?

## Pre-registered result (primary, confirmatory)
The pre-registered bake-off selection metric is **balanced accuracy** on a stratified 361-incident
lockbox, gated by beating the 2026 floor + Benjamini-Hochberg significance (rule
`winner_none_rule: empty_eligible_set`). Four frontier models were scored on the 1,200-incident
adjudicated goldset:

| config | balanced accuracy (lockbox) |
|---|---|
| **2026 floor** | **0.863** |
| llama-405b | 0.744 |
| qwen3-235b | 0.733 |
| deepseek-v3 | 0.711 |
| mistral-large-2411 | 0.691 |

**Winner = None.** No frontier model significantly beat the 2026 floor. Per the pre-registration,
**the 2026 ranking stands.** (The injection gate was treated as advisory — user-approved deviation,
non-adversarial corpus; resist-rates disclosed in `gate_advisory_disclosure.json`.)

## Robustness validation (why "2026 stands" is a positive result, not just a null)
Against the goldset's adjudicated **ground truth**, we tested whether *any* classifier produces a
class-incidence ranking closer to truth than the 2026 floor. Four independent methods
(Kendall-τ + top-N Jaccard, multi-label/rate, corpus-mix reweighting, and a 13-metric adversarial
sweep — run `wf_d45e5c61`) converge:

- The 2026 floor's ranking is already **Spearman ρ = 0.918** with the true ranking.
- The best frontier edge (llama, +0.027 ρ) **dissolves under paired bootstrap** (median Δ = 0.000;
  P(better) = 0.47).
- **Reweighting the goldset to the corpus class mix raises the floor to ρ = 0.971** and makes the
  frontier models *significantly worse* — the raw goldset had flattered them.
- A bad classifier (qwen3-235b) *significantly degrades* the ranking; none improves it.
- Magnitudes: the ensemble's raw proportion advantage (the floor is false-positive-inflated on
  LLM02/LLM09/ROLL-CMSB) is **removed by the pipeline's recall/precision correction**
  (CV-corrected neg-L2 Δ = −0.002, CI [−0.019, +0.015] — indistinguishable). This validates the
  recall-aware design: the correction makes the output robust to classifier quality.

**Conclusion: the 2026 incidence ranking is robust to classifier choice, on both the ordinal order
and the recall-corrected magnitudes.**

## Exploratory findings (secondary, NOT confirmatory — registered as a deviation)
Reported honestly as exploratory; they did **not** clear the pre-registered bar and do **not**
change the ranking:
1. **Frontier models are better per-incident classifiers on precision-aware metrics** — deepseek
   accuracy 0.618 vs floor 0.515 (bootstrap CI excludes 0); out-of-scope recall 0.49 vs floor 0.00.
2. **The 2026 classifier has a 0%-out-of-scope-recall blind spot** — it assigns a specific class to
   all ~38% of incidents that are truly out-of-scope. This inflates raw per-class counts but is
   corrected by the recall/precision layer and does not reorder the ranking.
3. The floor's false-positive inflation is concentrated in LLM02/LLM09/ROLL-CMSB.

## What did NOT happen (integrity notes)
- We did **not** override winner=None by metric-shopping. An adversarial premortem
  (6 perspectives) flagged that selecting a model on post-hoc deployment metrics after a
  pre-registered null is HARKing; the ranking-fidelity + recall-correction tests then showed the
  apparent "improvement" was an out-of-scope-abstention artifact that the pipeline already corrects.
- We did **not** run a corpus reclassification (no winner to deploy; no pods spent).

## Provenance / reproducibility
- Goldset scoring: `classify/seq/predictions_{llama-405b,deepseek-v3,mistral-large-2411,qwen3-235b}.json`
  + gate files. Bake-off: `results/bakeoff_seq/{classify_provenance.json, bakeoff_crosscheck.json,
  gate_advisory_disclosure.json, multi_metric_analysis.json}`.
- Both rankings preserved: the 2026 ranking (`cycles/2026/infer/`) stands as the result; this cycle
  (`cycles/2026-rarr/`) holds the re-analysis.

## Caveats
- The adjudicated goldset (n=1,200) is single-author (blind-vs-consensus disagreement 0.75;
  553/1,200 overrides). A second annotator would harden the truth target; the ranking-fidelity
  bootstrap median of exactly 0.000 makes a hidden improvement unlikely regardless.
- Goldset→corpus TV divergence is 0.468; goldset metrics transfer to the corpus only to the extent
  the calibration transfers (the standing limitation of the method, not new to this analysis).
- The recall-correction neutralization test is a point-estimate CV proxy for the pipeline's Bayesian
  Beta-posterior correction; directionally faithful.
