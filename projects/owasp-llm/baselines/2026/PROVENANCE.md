# PROVENANCE — 2026 OWASP-LLM Baselines

Generated: 2026-07-01T07:02:24+00:00

## Cycle source files (SHA256 at freeze)

| Artifact | Repo-relative path | SHA256 (first 16) |
|----------|-------------------|-------------------|
| lambda_samples.npy | projects/owasp-llm/cycles/2026/infer/lambda_samples.npy | 5f655cc687c321f0... |
| labeled_incidents.json | projects/owasp-llm/cycles/2026/classify/labeled_incidents.json | 61736c0890c1dc64... |
| respondent_rankings.npy | projects/owasp-llm/baselines/2026/respondent_rankings.npy | cff05ee4b80ac79e... |
| concordance.json | projects/owasp-llm/cycles/2026/results/concordance.json | 202c36aeca83b7aa... |

## F1 — Incidence-kappa fact

The frozen kappa (0.2028985507246377) is computed over **all 20** inference∩vote
entries (total_count=20, NOT the measurable subset of 17).  This mirrors how
`concordance.py:193` operates: the draw loop runs over all common entries without
a measurability filter.

## Method-delta 0.0 (never credited)

On 2026 OWASP-LLM data, bare-lambda ranking (`_ranks_from_lambda`, dead code in
concordance.py) and lambda*size incidence ranking (`_ranks_from_incidence`) produce
**identical kappa medians** (method_kappa_delta=+0.000000000).  This
coincidence is DISCLOSED and NEVER credited as a method gain.  Individual draw
rankings differ on 1927/5000 draws; the medians happen to coincide.

## CI spans zero

The 95% paired-draw CI [-0.1594, 0.5652] spans zero.  Cannot reject kappa=0 at
the 2026 sample size.  This does NOT indicate the engine is non-functional; it
reflects the structural inadequacy of n=20 (see prospective power block).

## STANDING_CAVEAT contradiction

`concordance.py:48` (STANDING_CAVEAT) claims "computed over the measurable subset
only," but the as-shipped kappa is over 20 entries.  The secondary measurable-subset
kappa (~0.1221) differs from the shipped 0.2029.  This contradiction
is surfaced as a standing disclosure; it is NOT silently resolved.

## Surrogate-variance caveat (prospective power)

The Fleiss-Cohen-Everitt asymptotic variance (sigma²=0.936) is a COARSE
DESIGN-STAGE SURROGATE, DISTINCT from and NOT governing the reported paired-draw
bootstrap CI.  Normal-approximation at n≈20 with ranked/stratum dependence violates
the iid assumption.  The n_required=46 is a structural-adequacy verdict, not a
"collect more taxonomy entries" instruction (n is a fixed ~20-entry taxonomy).

## Omnibus bridge

The previous-vs-new comparison in the U9 report is labeled **OMNIBUS**: the 2026
pre-RARR posteriors and the new run differ in ALL of data + method + recall-correction
+ config.  No clean method-only bridge exists.

## Byte-pin

The frozen kappa values in rankings_baselines.json come from reading
`projects/owasp-llm/cycles/2026/results/concordance.json` at freeze time.  They are NOT hard-coded constants.
Re-running `reproduce.py` re-derives them from the RAW respondent matrix
(non-circular: bootstraps from respondent_rankings.npy, NOT vote_rank_samples.npy).
