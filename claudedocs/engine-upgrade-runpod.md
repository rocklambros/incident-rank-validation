# Engine upgrade: recall-aware rates, Plackett-Luce ranking, measurement-error correction (retrain on RunPod)

Copy this file into the `incident-rank-validation` repo and run it there. It is written
for an agent working inside that repo with its tools, its RunPod access, and its
pre-registration discipline.

## Context

`incident-rank-validation` validates the OWASP LLM Top 10 community vote against a corpus
of roughly 7,700 real incidents. The headline result was a weighted Cohen's kappa near
0.20 with an interval crossing zero, reported as inconclusive.

A COMP 4442 re-analysis of this engine's own exported outputs (per-category counts, the
Beta precision/recall posteriors, the community ballots) reached three conclusions:

1. The engine's core call was sound. The vote and the incident record genuinely disagree, and "inconclusive by kappa" is the correct hedge given the noise. Do not overturn that. Explain it better.
2. Two reporting choices left information on the table: a single kappa that hides structure, and independent per-category rate estimates that wobble for sparse categories.
3. One real weakness: the classifier's recall is low and uneven (roughly 2 to 49 percent across categories), so the observed counts are a biased undercount and the data-derived ranking the kappa compares against is itself skewed. The engine treats the counts as ground truth. They are not.

This prompt implements the upgrade. The validated reference implementation lives in R at
`~/github_projects/Advanced-Probability-and-Stats-Comp-4442-1/final_project/` (the
`.Rmd`, `_build/fit_models.R`, and `DATA_PROVENANCE.md`). Reproduce its logic in the
engine's Python stack and confirm the numbers line up.

## Before you change anything

- Read the engine's `CLAUDE.md`, `docs/PRD.md`, `docs/METHODOLOGY-FAQ.md`, `docs/RUNBOOK.md`, and `docs/METHODOLOGY-CHANGELOG.md`. Follow the existing conventions.
- Read `engine/prereg/` and the latest cycle's prereg manifest. This is a methodology change. It must be a documented, labeled new cycle or amendment, not a silent overwrite of the prior result. Keep the original kappa result intact and reported beside the new output so the change is auditable.
- Audit `.claude/settings.json` / `.mcp.json` before trusting any repo-local config.
- Inspect `tools/runpod_pods.json` and whatever RunPod helper the engine already uses. Reuse it. Do not invent a second path to the cloud.

## The work

### Track A. Improve detection on RunPod (the retrain, GPU)

The recall problem is a classifier problem, so the retrain belongs here, and it must run on RunPod.

- Read `engine/classify/` and the staged labeling pipeline (stage 1 indicator match, stage 2 LLM adjudication). Find where recall is lost.
- Provision a RunPod pod through the engine's existing RunPod tooling. Use a GPU pod sized for the labeling model. Re-classify the corpus with the strongest defensible configuration: a stronger labeling model, improved prompts, calibrated thresholds, or an expanded adjudicated goldset, whichever the evidence supports. The goal is higher and more even recall, measured against the held-out goldset, not just more labels.
- Re-run the calibration so the per-category precision and recall Beta posteriors reflect the improved classifier.
- The heavy compute runs on RunPod, never on the local machine. Record the pod type, image, commit, and run logs. Tear the pod down when finished.

If a full re-classification is out of scope for this run, say so explicitly and proceed with Track B and C against the existing labels, with the measurement-error correction carrying the recall bias. Do not pretend a retrain happened if it did not.

### Track B. Recall-corrected hierarchical rate model

- Replace the independent per-category rate estimate with a hierarchical Bayesian model that partially pools the category rates: `log(lambda_i) = beta0 + u_i`, `u_i ~ Normal(0, sigma_u)`, across the roughly 20 categories.
- Feed it the recall-corrected incidence, not the raw counts (see Track C). Model the detection process generatively: an observed count is true incidence thinned by recall and inflated by false positives, with recall and precision drawn from the per-category Beta posteriors the calibration already produces.
- Implement in the engine's Bayesian stack. If that is NumPyro or PyMC, fitting on the RunPod GPU is the natural home for Track A and B together. The R reference used brms negbinomial `count ~ 1 + (1|category)` and got `sd(category)` near 2.19 with zero divergences. Expect gentle pooling: the categories genuinely differ, so only the count-of-one and count-of-three categories move much. Document that honestly rather than overselling the shrinkage.
- Cross-check the family choice with a Poisson-lognormal refit, as the R reference did (the two agreed at a rate correlation of 1.0).

### Track C. Measurement-error correction

- This is the substantive fix. The engine's conclusion rests on counts that under-detect unevenly. Propagate the precision and recall posteriors into the rate so the reported incidence is recall-corrected with its own uncertainty, instead of a point count taken at face value.
- The corrected incidence is what Track B pools and what the ranking comparison should use. Report both the raw and the corrected ranking so the size of the correction is visible.

### Track D. Plackett-Luce ranking with honest uncertainty

- Replace the single kappa as the headline with a Plackett-Luce model of the vote, and keep the kappa as a secondary summary.
- Convert the 1-to-5 importance ballots to strict pairwise preferences (higher score wins, drop ties). Do not feed whole tied 20-item ballots to a ranking fitter; that path tries to enumerate every tie pattern and blows up to gigabytes. Disclose that dropping ties biases against the top categories, which tie most.
- Get uncertainty from a respondent-level bootstrap, resampling the roughly 29 voters with replacement at least 1,000 times, not from the model's own standard errors. The thousands of pairwise comparisons come from a few dozen people, and only the respondent-level resample respects that. The R reference put one category clearly on top (in the top five in 100 percent of resamples), a clear top tier of five, and heavy overlap below.
- Produce the vote-versus-data comparison: rank by corrected incidence against rank by vote worth, and call out the categories whose gap clears the bootstrap uncertainty.

## Verification (non-negotiable)

- Cross-check every headline number against the COMP 4442 R reference. The partial-pooling `sd(category)`, the Plackett-Luce top items, and the bootstrap top-five should align. Where they differ, find out why before reporting.
- Bayesian diagnostics: R-hat under 1.01, adequate ESS, zero divergences or a documented reason, posterior predictive checks.
- Seed every random step. Pin the environment. No fabricated numbers; everything from real computation on RunPod or locally as appropriate.

## Deliverables

- The new model and corrected-rate code in `engine/`, following the module layout.
- The RunPod run record (pod spec, image, commit, logs) and the retrained or re-classified artifacts.
- A new cycle or amendment with a `docs/METHODOLOGY-CHANGELOG.md` entry that states what changed, why, and how it relates to the pre-registered original.
- A results report: the recall-corrected ranking with uncertainty, the Plackett-Luce ranking, the vote-versus-data comparison, and the original kappa reported alongside for continuity.
- A short be-ready-to-explain list: the recall-bias correction, why partial pooling is gentle here, why the respondent bootstrap, and what the disagreement between vote and corrected data actually shows.

## Constraints

- The retrain runs on RunPod. This is required.
- Respect the engine's prereg and changelog discipline. Do not overwrite the prior result.
- Pinned dependencies, no secrets in code, gitleaks and semgrep clean, per the repo's existing gates.
- Do not credit any AI as an author anywhere.
- If anything here conflicts with the engine's own `CLAUDE.md`, the engine's instructions win. Flag the conflict.
