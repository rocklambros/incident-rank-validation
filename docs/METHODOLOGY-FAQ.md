# Methodology FAQ

## Why Kruskal-Wallis for selection bias? (M14)

The selection-bias test checks whether the distribution of incident verdict labels differs between the corpus subset and the full population. Verdict labels (`measurable`, `frame_blind`, `ambiguous`) are **nominal**, not ordinal — there is no natural ordering between them. Spearman's rho requires ordinal or continuous variables; applying it to nominal verdict labels would be a misuse that produces a number without a valid interpretation.

Kruskal-Wallis H tests whether the rank distributions of a continuous or ordinal variable differ across nominal groups — or equivalently, whether the group-membership distribution differs from expected under the null. It is non-parametric and does not assume normality. The H statistic is chi-squared distributed under the null, giving a valid p-value for the comparison we actually care about: "does this corpus subset have a verdict-label distribution consistent with independent sampling from the full population?"

## Why is the post-hoc register Merkle-chained? (M16)

Post-hoc analysis entries must be append-only. If an analyst can silently edit or delete a prior post-hoc entry, the register loses its evidentiary value — you cannot tell whether the analysis record is complete. Merkle-chaining each entry to the hash of the previous entry makes tampering loud: any deletion or edit breaks the chain, and the `verify-posthoc-chain` check (run at `decide` time) raises `MerkleChainError`.

The chain does not prevent fabrication of an entirely new sequence (an attacker with full repository access could rewrite history), but it does prevent quiet, local edits. Combined with git history, the two controls together provide a high bar for undetected tampering. The chain also makes the order of entries verifiable — you cannot reorder entries without breaking the chain.

## What is the cost ceiling for Plan 5? (M9)

**Default ceiling: $500 USD per cycle.** This is set by M9 and is documented in `docs/PROVISIONING-PLAN.md`. The ceiling covers GPU cost for Stage-2 LLM classification only; it does not include human reviewer time.

To exceed the ceiling, you must:
1. Add an explicit override to `docs/PROVISIONING-PLAN.md` with the new ceiling and the name of the person authorizing it.
2. Add a checklist item to the REVIEWERS.md PRE-PUBLISH CHECKLIST noting the overage: "cost ceiling exceeded by $X, authorized by <name> on <date>".

The engine does not enforce the ceiling programmatically — it is a discipline control, not a mechanism control. The monitoring setup (RunPod billing API polled every 10 minutes with auto-shutoff at 1.2x ceiling) provides a soft enforcement backstop.

## Why Beta posteriors for calibration?

The Beta distribution is the **conjugate prior** for the Binomial likelihood. When you have a gold set of N incidents and observe k correct classifier assignments, the posterior for recall is Beta(alpha + k, beta + N - k) — closed-form, no sampling required. This makes calibration computation fast and exact.

Beyond computational convenience: Beta is interpretable. `Beta(a, b)` has mean `a/(a+b)` and concentration `a+b`. You can directly read off the posterior mean recall and the effective number of gold-set observations the prior contributes. When no gold set exists, `Beta(1,1)` is the uniform prior — the most honest expression of complete ignorance about classifier quality.

## Why CPU-only for NUTS?

NUTS (No-U-Turn Sampler) via NumPyro/JAX is run on CPU for **reproducibility across platforms**. JAX on GPU introduces non-determinism from floating-point reduction order (GPU warp scheduling is hardware-level non-deterministic). Even with a fixed PRNG seed, two runs on the same GPU can produce slightly different NUTS trajectories.

On CPU with X64 precision (`jax.config.update("jax_enable_x64", True)`), NUTS is deterministic given the same seed. This is required for the prereg discipline: if you cannot reproduce the inference result from the committed seed, you cannot verify that the result matches the pre-registered parameters.

CPU NUTS is slower than GPU NUTS, but for the model sizes in Plan 1 (tens of entries, thousands of incidents), CPU wall time is acceptable (typically < 5 minutes). Stage-2 LLM classification — the actual GPU-intensive workload — runs on RunPod separately (see PROVISIONING-PLAN.md).

## Why transparency-first publication?

The engine's stated purpose is to validate whether community vote rankings concord with incident-derived prevalence. If you suppress findings (e.g., "kappa was low, so we didn't publish"), you are **selecting on the outcome** — only publishing cycles where the vote ranking looks good. This selection bias undermines the entire project's credibility.

Transparency-first means: publish the report regardless of kappa, coverage, or measurability. Report limitations prominently. If the cycle has `non_publishable=True`, publish internally and document what would be required for external publication. If coverage is below the pre-registered minimum, say so explicitly in the report — the `below_prereg_minimum` tag is the control.

## What does "non-publishable" mean?

A cycle is stamped `non_publishable=True` when it fails the independent-review requirement:
- `PreregManifest.rubric_reviewer` is None (no external rubric reviewer), OR
- `PreregManifest.statistical_reviewer` is None (no external statistical reviewer), OR
- Either reviewer's `viewed_results_before_signoff` is True (reviewer saw results before attesting)

"Non-publishable" means the report cannot be shared externally as a methodology-validated finding. It is still valid for internal use, system testing, and iterative development. Every synthetic cycle in Plan 1 is non-publishable by design — synthetic data cannot substitute for independent human review of a real corpus.

The path to publishable is documented in REVIEWERS.md section "Path to Publishable."

## Why not just use the vote ranking directly?

The community vote ranking is a **convenience sample** — it measures the opinions of people who happened to participate in the vote, not the actual distribution of incidents in the wild. It has its own measurement error:
- **Selection bias:** voters are not representative of all practitioners.
- **Framing effects:** ballot wording and examples influence votes.
- **Temporal lag:** the vote reflects historical beliefs, not current incident rates.

The engine's purpose is to check whether the vote ranking is *consistent* with empirical incident prevalence, given measurement uncertainty on both sides. The kappa statistic measures this concordance. If kappa is high, the vote ranking is a good proxy for incident-derived ranking — but only within the uncertainty bounds the model quantifies. Using the vote ranking directly skips this validation step and treats practitioners' beliefs as ground truth, which is the exact claim the project exists to test.
