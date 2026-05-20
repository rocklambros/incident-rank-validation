# Methodology changelog

## 0.1.0 (Plan 1 v5, 2026-05-20)
- HANDOFF v2.5 compliant.
- Lambda prior: HalfNormal(scale=0.5), rate-per-unit-stratum interpretation.
- NB concentration prior: Gamma(5.0, 0.1), weakly informative toward Poisson.
- Selection-bias statistic: Kruskal-Wallis (nominal verdict labels, not ordinal — v2.4 correction).
- Hyperparameters + PRNG seed sourced from `<project>/project.toml`, hash-locked in PreregManifest.
- Two synthetic projects (`synthetic`, `synthetic-stress`) exercise distinct kappa regimes + untuned hyperparameters.
- GPU pinned to CPU; Stage-2 LLM classification on RunPod (Plan 5) with committed PROVISIONING-PLAN.md.
- No methodology claims are made until a real cycle runs in Plan 5.

## Plan 5 publication prerequisites (v2.5 §6 control 11 + §7.5 GPU rule + M17 two-cycle parity)
- External rubric reviewer + statistical reviewer identified, attested, signed_at precedes infer.
- docs/PROVISIONING-PLAN.md committed before Stage-2 run.
- Cycle output held for 30 days for reviewer audit before any external sharing (M17 two-cycle parity).
