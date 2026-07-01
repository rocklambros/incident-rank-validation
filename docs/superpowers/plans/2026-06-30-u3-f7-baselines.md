# U3: F7 Baselines + Prospective Power Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Reproduce + FREEZE the two "previous" rankings (bare-λ 0.20 baseline; λ·size-on-ORIGINAL-labels baseline) as committed, deterministically-reproducible artifacts, plus a prospective power statement — the concrete realization of the **preserve-previous-rankings mandate** and the "previous" side of the U9 previous-vs-new report.

**Architecture:** Pure functions produce each baseline value (inputs→value, no I/O); a thin freeze CLI persists them to a NEW committed tree `projects/owasp-llm/baselines/2026/` (NEVER inside byte-immutable `cycles/2026/`). Baseline B reuses the independent oracle (`oracle_incidence_ranking`) verbatim as a free cross-check. Offline only — reads the existing frozen `lambda_samples.npy`, invokes no sampler.

**Tech Stack:** numpy/scipy (pinned; no new deps), pytest. No R, ever.

## Global Constraints
- No AI/Claude/Anthropic attribution. CI gate: `uv run ruff check .` + `uv run mypy engine tests` before every commit; FULL `uv run pytest -q` before any push. F4 pin green. Two-stage review + opus whole-increment; push PR #22; CI green.
- **NEVER write under `cycles/2026/`** (byte-immutable). All new artifacts land in `projects/owasp-llm/baselines/2026/`. The freeze CLI MUST `raise` if the output path resolves inside any `cycles/` dir. Verify `git status` is clean under `cycles/2026/` after freezing.
- **Offline:** no live run, no NUTS. Only READ existing posteriors/labels. **No new third-party dependencies** (power math is closed-form numpy/scipy). No R.
- **Provisional gate:** output stays provisional (per the no-R/Python-oracle memory) until the oracle math spot-check (T10) agrees.
- **Preserve previous rankings:** this unit IS that mandate — the frozen baselines are the "previous" side; nothing here recomputes/mutates the new primary.

## Design decisions (user away — authorized; recommendation adopted, documented for review)
- **D1 — artifact location:** `projects/owasp-llm/baselines/2026/` with `rankings_baselines.json` as the single U9-consumed manifest (+ `lambda_median.npy`, `vote_rank_samples.npy`, `PROVENANCE.md`, `reproduce.py`).
- **D2 — prospective-power target κ = 0.40, confidence 0.95** (fair/minimal-policy-relevance, Landis-Koch; two-sided CI lower bound > 0). **PRE-REGISTERED, distinct from the observed 0.20.** [FOR THE HUMAN: this is a methodology choice; 0.40 is the lowest defensible "decision-relevant" bar. Adjust at the U5 lock if desired.]
- **D3 — power method:** closed-form quadratic-weighted-kappa asymptotic variance (Fleiss-Cohen-Everitt) + normal approximation → sample-size solver (matches the oracle's analytic-surrogate philosophy; no MCMC).
- **D4 — "original labels" for Baseline B** = the exact `cycles/2026/classify/labeled_incidents.json` that produced the frozen `lambda_samples.npy` (pin by SHA256). The data-vs-method bridge becomes informative only when a FUTURE cycle re-labels; for 2026 the "data" side is fixed by definition. Record any pre-relabel commit SHA in PROVENANCE as `labels_prebakeoff_ref` but do NOT use it as the frozen input (would be incoherent with the posterior).
- **D5 — bare-λ determinism:** the kappa depends on BOOTSTRAPPED vote ranks (`VoteRankPosterior.rank_samples` from `engine/vote/bootstrap.py`, seeded from raw votes). FREEZE `vote_rank_samples.npy` (16000×20) as a committed sibling AND record the bootstrap seed; `reproduce.py` re-derives kappa and asserts SHA256 byte-match.
- **D6 — manifest power fields at `schema_version >= 4`** (additive, defaulted, excluded from `<4` canonical form → lock-safe, v1/v2/v3 unchanged).

## Verified facts (from U3 research)
- `lambda_samples.npy` = **16000×20** (not 8000). Frozen numbers (from `concordance.json`): `kappa_median=0.2028985507246377`, `ci=[-0.1594202898550725, 0.5652173913043478]`, `measurable_count=17`.
- Bare-λ kappa = NOT byte-deterministic from λ alone (bootstrap-dependent — hence D5).
- Baseline B reuses `engine.verify.oracle.oracle_incidence_ranking` + `oracle_incidence_intervals` + `engine.verify.check._build_strata` verbatim.
- `meaningful_kappa_n` / target-κ are NOT persisted anywhere (introduced here via D6).

## Artifact tree (new, committed, outside cycles/)
```
projects/owasp-llm/baselines/2026/
  rankings_baselines.json     # THE U9 contract
  lambda_median.npy           # (20,) float64 median over 16000 draws
  vote_rank_samples.npy       # (16000,20) frozen vote ranks (D5)
  PROVENANCE.md               # cycle source paths + SHA256 at freeze time
  reproduce.py                # standalone verifier (no cycles/ needed for the math)
```
`rankings_baselines.json` schema (U9 consumes): `{artifact_type, schema_version, cycle, generated_from{lambda_samples/inference_summary/labeled_incidents/vote_rank_samples: path+shape+sha256(+seed)}, entry_ids[20], measurable_entry_ids[17], not_measurable[3], baselines{bare_lambda{method,function,tier_boundaries,n_common,bootstrap_draws,kappa_median,kappa_ci,kappa_ci_method}, lambda_size_original_labels{method,function,ranking,incidence_median,incidence_ci}}, prospective_power{target_kappa,confidence,method,n_required,current_n,excludes_zero_at_current_n,stage}}`.

## Tasks (TDD, bite-sized)
### T0 — Synthetic fixtures (no cycle files)
- [ ] `tests/unit/fixtures/` builder yielding a tiny `lambda_samples (200×4)`, `vote_rank_samples (200×4)`, `entry_ids`, `entry_strata`, `stratum_sizes` with hand-computed incidence/kappa. Enables all pure tests offline. Commit.
### T1 — Baseline A: `compute_bare_lambda_baseline(...)` in `engine/baselines/bare_lambda.py`
- [ ] RED: unit test on fixture asserts kappa median/CI == hand-computed; reuses `_ranks_from_lambda` + `quadratic_weighted_kappa`. GREEN: thin loop over draws mirroring `compute_concordance` (~lines 200-210) but with `_ranks_from_lambda`. Signature `(lambda_samples, vote_rank_samples, entry_ids, measurable_ids, tier_boundaries) -> (median, ci_lo, ci_hi)`. Commit.
### T2 — Power solver `kappa_sample_size_required(...)` in `engine/decide/prospective_power.py`
- [ ] RED: pin `n_required` for `(target_kappa, confidence, variance_factor)` vs hand-computed `n = ceil((z_{1-α/2}/κ_target)² · σ₁²)`, σ₁² = per-item asymptotic weighted-kappa variance (Fleiss-Cohen-Everitt) under the pre-registered marginal. Edge: κ_target≤0 raises; confidence∈(0,1); monotonic in κ_target/confidence. GREEN: closed-form (`scipy.stats.norm.ppf`). Commit.
### T3 — Power statement wrapper `prospective_power_statement(...)`
- [ ] Returns the `prospective_power` dict block; asserts `excludes_zero_at_current_n=False` at n=17, `stage="design"`, keys present. Commit.
### T4 — Baseline B: `compute_lambda_size_baseline(...)` in `engine/baselines/lambda_size.py`
- [ ] RED: unit test asserts `ranking`/`incidence_median`/`incidence_ci` on fixture. GREEN: REUSE `oracle_incidence_ranking` + `oracle_incidence_intervals` + `_build_strata` verbatim (built-in cross-check). Restrict to the 17 measurable ids. Commit.
### T5 — Assembler `build_rankings_baselines(...)` in `engine/baselines/freeze.py` (pure, no I/O)
- [ ] RED: schema test asserts every required key + the frozen bare-λ constants (`0.2028985507246377`, CI) present + equal. GREEN. Commit.
### T6 — Integration reproduction (read-only on cycles/2026/)
- [ ] RED: `tests/integration/test_f7_baseline_repro.py` loads real `lambda_samples.npy` (16000×20) + `inference_summary.json` + `labeled_incidents.json` + frozen `vote_rank_samples.npy`; asserts bare-λ kappa == `0.2028985507246377` / CI == `[-0.1594…, 0.5652…]` and `n_common == 17`. GREEN once T1/T4 correct. Guard: assert no test writes under `cycles/`. Commit.
### T7 — Freeze CLI `engine/cli/freeze_baselines.py`
- [ ] RED: writes to temp dir, asserts `rankings_baselines.json` + siblings created, SHA256 fields populated; **asserts the CLI RAISES if the output path resolves inside any `cycles/` dir** (safeguard). GREEN. Then run it once to materialize `projects/owasp-llm/baselines/2026/`. Commit (incl. the materialized artifacts).
### T8 — `reproduce.py` + `PROVENANCE.md`
- [ ] RED: test invokes `reproduce.py` against the committed artifacts, asserts it re-derives the frozen kappa + validates SHA256s (self-contained, no cycles/ needed). GREEN. Commit.
### T9 — Manifest power fields (D6)
- [ ] RED: `PreregManifest(schema_version=4, prospective_power_target_kappa=0.40, prospective_power_confidence_level=0.95)` round-trips; a non-default power field at schema_version<4 raises (mirror the F6/sigma_u `__post_init__` guard); v<4 canonical form byte-unchanged (golden-hash test like U2-2). GREEN: additive fields + validation. Commit.
### T10 — Oracle spot-check of power math
- [ ] RED: `test_power_oracle.py` recomputes `n_required` by an INDEPENDENT arrangement of the same closed form (solve for SE then invert), asserts agreement within tolerance. Scope = formula math only (a design-stage value can't be re-derived from a run). GREEN. Commit.
### T11 — U9 contract test (consumer guard) + loader
- [ ] RED: `test_u9_contract.py` opens `rankings_baselines.json` via a loader and asserts the exact keys U9 reads (`baselines.bare_lambda.kappa_ci`, `baselines.lambda_size_original_labels.ranking`, `prospective_power.n_required`, `measurable_entry_ids`). GREEN: loader in `engine/report/` (where U9 will import it). Commit.

## Definition of done
ruff/mypy/pytest/semgrep green; T0-T11 pass; `projects/owasp-llm/baselines/2026/` materialized + committed; `cycles/2026/` byte-unchanged (git status clean under it); no R; no new deps; oracle spot-check (T10) green; output provisional until the oracle math check agrees.

## Premortem note
Run `adversarial-premortem-complete` before execution. Likely probes: the bare-λ bootstrap determinism (D5 — is freezing vote_rank_samples sufficient, or does the seed+raw-votes path drift?); the power σ₁² model (D3 — is the Fleiss-Cohen variance the right one for a paired-draw percentile CI?); the byte-immutability guard (does the freeze CLI truly refuse cycles/ paths?); the manifest v4 lock-safety (golden-hash for v1/v2/v3); whether "original labels" (D4) is coherent for the 2026 bridge.
