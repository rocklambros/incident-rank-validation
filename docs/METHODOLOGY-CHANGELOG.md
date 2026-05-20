# Methodology Changelog

## 0.2.0 (Plan 2, 2026-05-20)

GenAI Agentic corpus A adapter, per-stratum bias profiles, snapshot vendoring.

Key deliverables:
- `engine/adapters/genai_agentic.py`: concrete adapter reading vendored snapshot, emitting canonical IncidentRecord instances.
- `engine/adapters/genai_agentic_bias.py`: per-stratum BiasProfile declarations for "security" and "ai-harm" strata with quarantine rules for bare-LLM03 contamination (HANDOFF §3 F2).
- `engine/cli/snapshot.py`: vendor-snapshot CLI command with content-addressed hashing and provenance.json (6 fields per HANDOFF §5.1).
- Severity-default detection: source-ingest "Medium" default → `severity=None` (HANDOFF §3 Mixture).
- Future-dated row drop per HANDOFF §4 Temporal.
- Drift detector integration with vendored snapshot (JSONL format for drift.py compatibility).
- Vendored snapshot at `projects/owasp-llm/cycles/2026/corpora/genai_agentic/<hash>/`.

Methodology decision: this plan defines the stratum boundaries for Corpus A as the two values of the source `corpus` field ("security", "ai-harm"), each with a declared BiasProfile. This is a structural decision that constrains the Bayesian model's stratification in Plans 3–5. HANDOFF §4's "Corpus A is a mixture" row requires per-stratum bias profiles for corpus AND category — Plan 2 implements the corpus-level stratification; category-level stratification (per HANDOFF §3 Mixture) is deferred to Plan 3 where the rubric defines how categories interact with the measurement model. Entry definitions and overlap weights are provisional (2025 taxonomy placeholders) and MUST be replaced by the frozen 2026 rubric in Plan 3.

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

## Plan 1 v5.1 erratum (2026-05-20)

**Acceptance-criteria-verification erratum, no methodology change.**

Plan 1 v5 tag `v0.1.0-plan1` (commit `7a91f51`) declared acceptance against criteria 1-29. Criteria 6 (CI matrix green on ubuntu-latest AND macos-latest), 7 (cross-platform diff job green M5), 8 (SBOM generated and cosign-signed), and 9 (synthetic e2e + synthetic-stress e2e both pass) were claimed met but were never actually verified by CI: a YAML flow-style mapping in `.github/workflows/ci.yml` (`with: { name: foo-${{ matrix.os }}, ... }`) contained a `}` from the matrix expression that prematurely closed the outer mapping. GitHub Actions's parser rejected the entire file at startup, producing zero jobs across every CI run since the workflow landed. PyYAML accepted the file leniently which is why no local validation caught it.

Wave 0 CI rehabilitation (PR #4, merged as `a3ddb06`) fixed five distinct bugs in sequence, each hidden by the prior:
1. YAML flow-style `}` collision (the parse failure that suppressed everything)
2. `cyclonedx-py -o foo` missing the required subcommand (correct: `cyclonedx-py environment -o foo`)
3. `cosign sign-blob` keyless missing `id-token: write` permission
4. `on: [push, pull_request]` racing itself; replaced with `pull_request: + push: branches: [main]` plus a concurrency group keyed on `head_ref`
5. `gitleaks/gitleaks-action@v2` requires a paid `GITLEAKS_LICENSE` secret for the `rocklambros` account; removed (the versioned pre-push gitleaks hook per commit `2063f39` remains the primary control)

Criteria 6, 7, 8, 9 are now actually CI-verified on main as of commit `a3ddb06`. Tag `v0.1.1-plan1` re-anchors the acceptance claim at the commit where the verification is true. Tag `v0.1.0-plan1` is preserved unchanged for git-history integrity; readers reaching it via `git log` or `git show` should consult this erratum.

Class lesson, recorded for future phases: a workflow file's mere presence does not prove its execution. Every phase that adds or modifies CI must verify CI actually runs the new logic to completion, not just that the workflow file parses or that local tests pass. `actionlint` against the file is necessary but not sufficient (it catches YAML/schema issues, not runtime issues like permission gaps or step-script errors). The class is documented in `docs/PRD.md` §3.8 Risks and in `claudedocs/IMPLEMENTATION_PROMPTS.md` Phase 2 lessons.
