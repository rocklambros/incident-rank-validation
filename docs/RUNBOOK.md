# Runbook — fail-loudly states

## R-hat > 1.01 / ESS below threshold
- **Symptom:** `DiagnosticsFailure` raised; report not emitted.
- **Diagnosis:** NUTS chain not converged; could be insufficient warmup, pathological geometry, or mis-specified prior.
- **Remediation:** (1) bump `num_warmup` in `run_inference` to 2000; (2) inspect divergences — if > 0, the model has a funnel or the prior is conflicting with the data; (3) consider re-parameterization. Do NOT loosen the threshold without statistical-reviewer signoff.
- **Escalation owner:** statistical reviewer (see REVIEWERS.md).

## Post-warmup divergences > 0
- **Symptom:** Same as above; `divergences` field non-zero.
- **Diagnosis:** sampler hit a region of the posterior with bad geometry.
- **Remediation:** increase `target_accept_prob` in NUTS init (default 0.8 → 0.95); re-parameterize if persistent. Check for prior-data conflict.

## NUTS timeout
- **Symptom:** `TimeoutError` raised at `signal.alarm` deadline.
- **Diagnosis:** chain too slow; usually model dimension exploded or backend not on CPU (check `jax.default_backend()`).
- **Remediation:** verify `JAX_PLATFORM_NAME=cpu`; reduce model parameters (fewer entries x strata); if real-corpus scale, raise `--timeout-seconds` but expect long runs.

## LockMismatchError
- **Symptom:** prereg lock fails verification.
- **Diagnosis:** manifest mutated after lock written, OR lock file edited.
- **Remediation:** never edit `prereg.lock.json` directly. Re-run `prereg` if intent was to change a field; commit the new lock.

## AttestationError
- **Symptom:** lock file not committed to git, or working tree differs from HEAD.
- **Diagnosis:** uncommitted edits to the lock or to a reviewer attestation file.
- **Remediation:** `git status` -> review changes -> commit explicitly. Never bypass with `--no-verify`.

## DriftSignoffRequired
- **Symptom:** `infer` refuses; report says drift anomalies detected on N labels.
- **Diagnosis:** corpus snapshot has shifted per-entry counts beyond threshold.
- **Remediation:** review the drift report; if benign (upstream re-categorization, vendor disclosure batch), pass `--accept-drift-signoff "<>=30-char rationale>"`. The rationale persists to `cycle/drift_signoffs/<timestamp>.txt`.
- **Escalation:** if cause is suspected adversarial ingestion, file an erratum.

## Reviewer signoff missing (non_publishable=True)
- **Symptom:** report carries `non_publishable` stamp.
- **Diagnosis:** `PreregManifest.rubric_reviewer` or `.statistical_reviewer` is None, or `viewed_results_before_signoff=True` on either.
- **Remediation:** either get fresh attestations from external reviewers (see REVIEWERS.md path-to-publishable), or accept the internal-only stamp.
- **Reminder:** per v2.4 (M8), `signed_at` is derived from `git log` — backdating is detected by mismatch.

## Twin-Bayesian top-tier disagreement
- **Symptom:** `TwinAgreement.disagreements` non-empty.
- **Diagnosis:** point-estimate twin (de-biased counts) and NUTS posterior disagree on the direction of a top-tier comparison.
- **Remediation:** DO NOT reconcile silently. Report the disagreement as a finding per HANDOFF section 5.5. Typical causes: (1) overlap weights misspecified, (2) tail posterior dominated by prior, (3) twin's point estimate naive about leakage.

## Below pre-registered measurability minimum
- **Symptom:** `coverage.below_prereg_minimum = True` in the report.
- **Diagnosis:** the cycle did not reach the target subset coverage.
- **Remediation:** publish anyway (transparency-first). The report tag is the control. Consider running the staged frame-coverage audit (Plan 5+ extension) to upgrade unmeasurable entries.

## Frame-blind verdict surprise
- **Symptom:** an entry you expected to be measurable shows as frame-blind.
- **Diagnosis:** rubric author set `frame_blind=True` on the entry.
- **Remediation:** review the rubric drafting attestation; if the verdict is wrong, freeze a new rubric (committed, attested) and re-run.

## CorpusModeViolation
- **Symptom:** CLI refuses `--corpus-mode real` against synthetic provenance, or vice versa.
- **Diagnosis:** the corpus snapshot's `provenance.json` adapter field doesn't match the declared corpus mode, OR `--corpus-mode real` was attempted with `non_publishable=True` manifest.
- **Remediation:** match the mode to the data; or upgrade attestation to publishable.

## CrossCycleComparisonError
- **Symptom:** any code path comparing two cycles' results raises this.
- **Diagnosis:** cycle_id or taxonomy_hash differs between the cycles.
- **Remediation:** do not bypass. Per HANDOFF section 5.1, per-entry prevalence does not trend across cycles because entries get renamed/renumbered.

## Cross-platform JAX variance (within MCSE)
- **Symptom:** CI `cross-platform-diff` job passes but local NUTS run differs by ~0.001 in lambda median between macOS-arm64 and Linux-x86_64.
- **Diagnosis:** BLAS-level non-determinism even at X64. Expected.
- **Remediation:** None required if within MCSE. The `cross-platform-diff` job validates categorical fields (measurable set, coverage_ratio); kappa medians may drift slightly. If drift exceeds 0.01, investigate.

## Defense-in-depth false confidence (F-defenseindepth)
- **Symptom:** a finding surfaces and the first instinct is "the engine has many controls, the bug must be elsewhere."
- **Diagnosis:** cognitive trap — see SUCCESSOR-PRIMER.
- **Remediation:** treat the engine as a hypothesis. Check whether the relevant control's *assumptions* hold for the case at hand before assuming the control fired correctly.

---

## RARR Live-Run: Pod-Leak Recovery (SIGKILL / crash)

Every pod provisioned by `orchestrate_live_run` is written to `tools/runpod_pods.json`
atomically (write-temp + fsync + os.replace) **before** the readiness wait.
The file survives `SIGKILL` and `os._exit` — it is the backstop when both
atexit and the try/finally teardown are bypassed.

### Detect leaks

```bash
cat tools/runpod_pods.json            # registry snapshot
python tools/terminate_runpod.py      # dry-run reconcile — no DELETEs issued
```

Reconcile output categories:
- `live_and_ours`: in registry AND live → will be terminated with `--execute`.
- `orphans`: live but NOT in registry → reported loudly; NOT auto-terminated.
- `registered_gone`: in registry but already gone → safe to ignore.

### Terminate via registry (primary path)

```bash
# 1. Dry-run — inspect the plan:
python tools/terminate_runpod.py

# 2. Execute:
python tools/terminate_runpod.py --execute

# 3. Verify clean:
python tools/terminate_runpod.py      # should show 0 live_and_ours, 0 orphans
```

Name-guard: only pods whose name starts with `qwen3-235b`, `llama-405b`,
`deepseek-v3`, or `mistral-large-2411` are auto-terminated.  Others are skipped and
logged — this prevents accidentally deleting unrelated pods on the account.

### Manual RunPod console fallback (API down or registry corrupt)

1. Open <https://www.runpod.io/console/pods>
2. Identify pods named `classify-<model>` (e.g. `classify-qwen3-235b`)
3. Stop/terminate each manually
4. Remove their entries from `tools/runpod_pods.json` (or delete the file if empty)

### Terminate a specific pod (with force override)

```bash
python tools/terminate_runpod.py --pod-id <ID> --execute         # name-guard enforced
python tools/terminate_runpod.py --pod-id <ID> --execute --force  # bypass guard
```

### Preflight fail-closed (R2)

`live_run_cli` calls `reconcile(execute=False)` before provisioning and raises
if **any** live pod or orphan is detected.  Always clear leaked pods before re-running:

```bash
python tools/terminate_runpod.py --execute
# then confirm clean, then re-run the live-run CLI
```

### Post-run HF token rotation (R6)

The HF token (`HF_TOKEN`) is injected into pod environment at pod-creation time
and is no longer needed once pods are terminated.  After each live run:

1. Rotate at <https://huggingface.co/settings/tokens>
2. Update `pass`: `pass edit huggingface/token`
3. Prefer injecting via RunPod pod-secrets (console → Pod → Secrets) rather than
   env var to limit exposure window.

### Grid-lock and manifest-lock

`live_run_cli` verifies before provisioning:
- `cycles/2026-rarr/prereg/manifest.lock` — SHA-256 of the pre-registration manifest
- `cycles/2026-rarr/prereg/grid_lock.json` — SHA-256 of `bakeoff_grid.json`
  (created on first preflight; must be committed alongside the grid file)

If a lock mismatch is reported after an intentional change, delete `grid_lock.json`,
re-run preflight to re-lock, run a new premortem, then commit.

### All outputs are PROVISIONAL until U9 oracle agrees

`orchestrate_live_run` returns `LiveRunResult(provisional=True)`.  Do not use
outputs for any decision before the independent Python oracle (U9) confirms.
