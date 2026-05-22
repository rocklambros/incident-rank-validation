# Plan 6: Pipeline Execution Fixes + WandB Integration

**Goal:** Fix all 10 runtime-breaking bugs in the `--execute` code path discovered by the Plan 5 adversarial premortem, wire RunPod and WandB credentials from `pass`, add WandB monitoring for NUTS inference, and deliver the integration test that would have caught every one of these bugs.

**Architecture:** Remediate bugs in `engine/cli/pipeline.py`, `engine/cli/pipeline_executor.py`, `engine/model/inference.py`, and `engine/model/robustness.py`. Add a new `engine/monitoring/wandb_logger.py` module for WandB integration. Update security tests to target the real `Stage2Classifier` instead of the stub `Stage2Protocol`. Add a fixture-driven integration test that exercises the full `--execute` path with a 5-record synthetic corpus.

**Tech Stack:** Python 3.12, wandb, pass (Unix password manager), existing engine modules (numpyro, jax, click, httpx).

---

## F1: IncidentRecord Constructor — Missing 5 of 9 Required Fields

**File:** `engine/cli/pipeline.py:78-83`
**Bug:** `IncidentRecord(id=, text=, corpus_stratum=, native_labels=)` — only 4 of 9 positional fields supplied. Missing: `date`, `severity`, `source_class`, `quality`, `source_url`.
**Fix:** Parse all 9 fields from the JSONL corpus records. Use sensible defaults for optional fields when the source omits them (`severity=rec.get("severity")`, `quality=rec.get("quality", "auto")`).

## F2: Stage-2 Execution Dead Code

**File:** `engine/cli/pipeline.py:92-95`
**Bug:** `route_to_stage2()` is called and `low_confidence` computed, but `Stage2Classifier` is never instantiated. `stage2_results` stays `()`. The entire Stage-2 path is dead code.
**Fix:** When `--stage2-config` is provided, load the Stage-2 manifest, instantiate `HttpRunPodClient` (credentials from `pass`), build `CostTracker`, instantiate `Stage2Classifier`, load the incidents routed to Stage-2 by ID, call `classify_batch()`, merge Stage-2 results with Stage-1 using `merge_classifications()`.

## F3: Wrong Vote Loader Import

**File:** `engine/cli/pipeline.py:206`
**Bug:** `from engine.vote.xlsx_loader import load_vote_xlsx` — module `engine.vote.xlsx_loader` does not exist. The real module is `engine.vote.loader` with function `load_vote_data`.
**Fix:** `from engine.vote.loader import load_vote_data`

## F4: VoteData Has No `.rows` Attribute

**File:** `engine/cli/pipeline.py:209`
**Bug:** `len(vote_data.rows)` — `VoteData` has attributes `rankings`, `entry_ids`, `n_respondents`. No `.rows`.
**Fix:** `vote_data.n_respondents`

## F5: Wrong `compute_concordance()` Signature

**File:** `engine/cli/pipeline.py:223-227`
**Bug:** Called as `compute_concordance(lambda_samples=, entry_ids=, vote_data=)`. Actual signature requires `(inference_result, vote_posterior, tier_boundaries, flag_threshold_tau, measurable_count, total_count, meaningful_kappa_n, measurability_minimum)`.
**Fix:** Reconstruct `InferenceResult` from saved artifacts, call `bootstrap_vote_ranks()` to get `VoteRankPosterior`, load `PreregManifest` for threshold params, then call `compute_concordance()` with the correct 8-parameter signature. Also compute `measurability_map`, `selection_bias`, `rollup_results`, `robustness_spread`, `twin_agreement`, and `prereg_diff` — the full decide layer.

## F6: Hardcoded `confidence_threshold=0.3`

**Files:** `engine/cli/pipeline.py:49,94`
**Bug:** `confidence_threshold=0.3` is hardcoded twice instead of reading from the pre-registered manifest.
**Fix:** Load `PreregManifest` from `prereg/manifest.json` and use `manifest.confidence_threshold`.

## F7: `execute_infer_phase()` Is a Validation-Only Stub

**File:** `engine/cli/pipeline_executor.py:96-128`
**Bug:** `execute_infer_phase()` checks 3 gates, then returns `None`. It never calls `run_inference()`, never writes artifacts, and never calls `write_infer_artifacts()` or `write_nuts_failure()`.
**Fix:** After passing gates: load labeled incidents, deserialize calibration posteriors, build observation arrays, call `run_inference()` with `num_chains` parameter, handle `DiagnosticsFailure` by calling `write_nuts_failure()`, on success call `write_infer_artifacts()`.

## F8: `num_chains=1` Hardcoded in `run_inference()`

**File:** `engine/model/inference.py:213`
**Bug:** `num_chains=1` hardcoded. With 1 chain, R-hat dict is empty (line 259: fallback to 1.0), so the R-hat diagnostic gate is silently bypassed. The premortem R9 remediation specified `num_chains=4`.
**Fix:** Add `num_chains: int = 4` parameter to `run_inference()`. Pass it through to `MCMC()`. The ESS/R-hat diagnostics then operate on real multi-chain data.

## F9: Robustness Inference Returns Empty Diagnostics

**File:** `engine/model/robustness.py:94-99`
**Bug:** Returns `InferenceResult(r_hat={}, ess={}, divergences=0)`. No diagnostics computed at all. The downstream diagnostic gates in `run_inference()` fallback to "everything is fine" when dicts are empty.
**Fix:** Extract diagnostics from MCMC the same way `run_inference()` does: `get_samples(group_by_chain=True)`, `numpyro.diagnostics.summary()`, extract R-hat and ESS, count divergences. Apply the same diagnostic gates.

## F10: `route_to_stage2()` Returns `set[str]` (IDs Only)

**File:** `engine/cli/pipeline_executor.py:19-27`
**Bug:** Returns `set[str]` (incident IDs). For multi-label classification where one incident matches multiple entries, the caller needs to know which incident records to re-classify, not which (incident, entry) pairs.
**Analysis:** This is actually correct — Stage-2 re-classifies the entire incident, not a specific (incident, entry) pair. The set of incident IDs is what the caller needs to filter the original incident list for Stage-2 input. No fix needed.

## F11: `str.format()` Template Injection Risk

**File:** `engine/classify/stage2_prompt.py:35-40`
**Bug:** `incident.text` is interpolated via `str.format()`. If incident text contains `{braces}`, it raises `KeyError`. This is a reliability bug (not a security vulnerability — the text is attacker-controlled data, but format string injection cannot execute code in Python).
**Fix:** Escape braces in incident text before interpolation, or switch to a template approach that doesn't use `str.format()` for user data.

## F12: Security Tests Target Stub, Not Real Classifier

**File:** `tests/security/test_stage2_injection_fixture.py:73-79`
**Bug:** `test_stage2_rejects_injection` tests instantiate `Stage2Protocol()` (the stub that raises `NotImplementedError`), not `Stage2Classifier` (the real implementation). All 10 tests are `xfail(strict=True)` — they pass because `NotImplementedError` is raised, confirming only that the stub exists.
**Fix:** Create new tests that target `Stage2Classifier` with a mock `RunPodClient` that returns attacker-controlled JSON. Verify that injection payloads in incident text cannot change the classification outcome. Keep the xfail tests as documentation of the stub state, add new passing tests.

## F13: Zero Integration Test Coverage on `--execute` Path

**File:** `tests/unit/test_pipeline_cli.py:101-164`
**Bug:** `TestExecuteFlags` tests only check negative assertions (`"prerequisites satisfied" not in output`). They verify the command entered the execute branch but never check if it actually worked. Zero coverage of the actual execution logic.
**Fix:** Add a fixture-driven integration test that creates a 5-record synthetic corpus, a valid manifest, a rubric, calibration posteriors, runs `classify-real --execute`, `infer-real --execute`, `decide-real --execute`, and verifies output artifacts exist and are well-formed.

## F14: WandB Integration (New Feature)

**Requirement:** Add Weights & Biases monitoring for NUTS inference runs. Track:
- Lambda samples summary statistics (per-entry median, CI)
- R-hat values per parameter
- ESS values per parameter
- Divergence count
- Wall-clock inference time
- RunPod cost (when Stage-2 is active)
- Concordance kappa (at decide phase)

**Credential source:** `pass wandb/api-key`
**WandB project:** `incident-rank-validation`
**Design:** A `WandBLogger` class in `engine/monitoring/wandb_logger.py` that wraps the wandb SDK. Optional — if wandb is not installed or credentials are not available, the pipeline runs without monitoring. No hard dependency.

## F15: RunPod Credentials from `pass`

**Requirement:** The `--execute` path for `classify-real` with `--stage2-config` must load RunPod credentials from `pass` when env vars are not set.
**Credential source:** `pass runpod/api-key`
**Design:** A `load_secret(name: str) -> str` helper in `engine/cli/secrets.py` that calls `pass show <name>` via subprocess. Falls back to environment variables. Used by the pipeline executor when instantiating `HttpRunPodClient`.

## Dependency Changes

- Add `wandb>=0.19,<1.0` as an optional dependency (`[project.optional-dependencies] monitoring = ["wandb>=0.19,<1.0"]`)
- No new hard dependencies

## Version

Bump to `1.1.0` — this is a feature release (WandB integration) with critical bug fixes.
