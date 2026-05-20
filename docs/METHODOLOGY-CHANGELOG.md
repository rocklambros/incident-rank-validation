# Methodology Changelog

## Plan 1 v5 (2026-05-20)

HANDOFF v2.5 compliant. Closures: R1-R33, L1-L11, M1-M23.

Key methodology decisions:
- Negative-Binomial measurement-error model with NUTS
- Beta-parameterized recall/precision from gold-set calibration
- Frame-blind censoring (unmeasurable, never low-prevalence)
- Quadratic-weighted Cohen's kappa for tiered concordance
- Kruskal-Wallis H for selection-bias quantification (nominal verdict labels)
- Non-Bayesian robustness twin for model-independence
- Transparency-first publication (never suppression)
- Pre-registration discipline with hash-locked manifests

Implementation details:
- Lambda prior: HalfNormal(scale=0.5), rate-per-unit-stratum interpretation.
- NB concentration prior: Gamma(5.0, 0.1), weakly informative toward Poisson.
- Hyperparameters + PRNG seed sourced from `<project>/project.toml`, hash-locked in PreregManifest.
- Two synthetic projects (`synthetic`, `synthetic-stress`) exercise distinct kappa regimes + untuned hyperparameters.
- GPU pinned to CPU for NUTS; Stage-2 LLM classification on RunPod (Plan 5) with committed PROVISIONING-PLAN.md.
- No methodology claims are made until a real cycle runs in Plan 5.

### Plan 5 prerequisites (M17)
- Two-cycle parity: 30-day reviewer audit comparing independent cycle results before any external publication
- Reviewer independence verified via REVIEWERS.md PRE-PUBLISH CHECKLIST
- External rubric reviewer + statistical reviewer identified, attested, signed_at precedes infer
- docs/PROVISIONING-PLAN.md committed before Stage-2 run
