# U3 Cluster C Report — T7 / T8 / T9 / T10 / T11 / T12 (Freeze Baselines)

## Status: DONE — all gates GREEN

---

## Gate results

| Gate | Result |
|------|--------|
| `uv run pytest -q -m "not slow"` | **1057 passed, 34 deselected, 10 xfailed** |
| `uv run ruff check .` | **All checks passed** |
| `uv run mypy engine tests` | **Success: no issues found in 259 source files** |
| cycles/2026/ clean | **Confirmed — no writes under cycles/** |
| T11 byte-pin integration test | **RUNS (not skipped), PASSES** |

---

## Files created / modified

| Path | Task | Notes |
|------|------|-------|
| `engine/baselines/freeze.py` | T7 | Pure assembler; reads concordance.json to byte-pin kappa; no I/O side-effects |
| `engine/cli/freeze_baselines.py` | T9 | Hardened CLI: cycles/ guard + write-once guard + all artifact production |
| `engine/cli/main.py` | T9 | Modified: import + `cli.add_command(freeze_baselines_cmd)` |
| `tests/unit/test_baselines_freeze.py` | T7 | 8 unit tests for `build_rankings_baselines()` |
| `tests/unit/test_freeze_baselines_cli.py` | T9 | 11 tests: 7 cycles/ guard (evasion table) + 4 write-once guard |
| `tests/integration/test_f7_baseline_repro.py` | T11 | 6 integration tests; assert-not-skipped guard at module level |
| `projects/owasp-llm/baselines/2026/rankings_baselines.json` | T8/T10 | schema_version=4, 5,745 bytes |
| `projects/owasp-llm/baselines/2026/respondent_rankings.npy` | T8/T10 | shape (29,20), 4,768 bytes |
| `projects/owasp-llm/baselines/2026/vote_rank_samples.npy` | T8/T10 | shape (5000,20), 800,128 bytes |
| `projects/owasp-llm/baselines/2026/lambda_median.npy` | T10 | shape (20,), 288 bytes |
| `projects/owasp-llm/baselines/2026/votes_source.xlsx` | T10 | 148,243 bytes (copy of cycle source) |
| `projects/owasp-llm/baselines/2026/PROVENANCE.md` | T10/T12 | 2,999 bytes |
| `projects/owasp-llm/baselines/2026/SHA256SUMS` | T10 | 595 bytes; written last |
| `projects/owasp-llm/baselines/2026/reproduce.py` | T12 | Standalone verifier; bootstraps from respondent_rankings.npy (non-circular) |

---

## Key values in rankings_baselines.json (schema_version=4)

| Field | Value |
|-------|-------|
| `bare_lambda_sensitivity.kappa_median` | **0.2028985507246377** (byte-pinned from concordance.json) |
| `bare_lambda_sensitivity.method_kappa_delta` | **0.0** (bare-lambda == lambda*size ranking coincide on 2026 data) |
| `secondary_measurable_subset.n_measurable` | **17** |
| `secondary_measurable_subset.measurable_kappa_median` | 0.12206572769953028 |
| concordance.json SHA256 pinned | `202c36ae…` |
| respondent_rankings.npy shape | (29, 20) |
| vote_rank_samples.npy shape | (5000, 20) — n_bootstrap=5000, seed=20260520 |

---

## SHA256SUMS (materialized artifacts)

```
b1649e761e880b56b64478c9b193f8d841d7959f25992a9473f9d0518cab396d  PROVENANCE.md
13d33cef9d661552a1376ce9657ba95a7a8c5c266d1a5b9fedb36d5072f78f12  lambda_median.npy
2c1d71c8db3a68b2c23041f13c4a9b2a398e69ce4b486b16b9e88c9f068a120d  rankings_baselines.json
4f2bd786a8865afbc820503204f6e6519a3e0b7342141f4f8f283aab44da8afb  reproduce.py
cff05ee4b80ac79e60cc05262b7fb2707544b2bd38e3c56e9e9fe623a13e4fc2  respondent_rankings.npy
04b236ba9f0c8deae135934ac93c296d37b11348d993d94a33e9f9aa92c5013e  vote_rank_samples.npy
a97f24bd41c456620596d36c4e82aeb60974a74a89ec977b657fe8562b42dc80  votes_source.xlsx
```

---

## Design notes

### Byte-pin (never a hand-typed constant)

`build_rankings_baselines()` in `engine/baselines/freeze.py` reads the concordance.json
file and extracts `weighted_kappa_median` from it. It then raises `ValueError` if the
computed kappa deviates from the file value by more than 1e-9. The `byte_pinned_to` field
records the relative path from the `.git` root to the concordance.json file.

The integration test (`test_f7_baseline_repro.py`) independently loads concordance.json
and compares the `kappa_median` from `bare_lambda_sensitivity` against the file — never
against a hand-typed constant.

### Non-circular reproduction

`reproduce.py` boots from `respondent_rankings.npy` (the RAW respondent matrix, written
by `freeze_baselines_cmd` from the xlsx source before any bootstrap). It does NOT read
`vote_rank_samples.npy` as its input. This ensures the reproduction is genuinely
independent: re-running bootstrap from the raw matrix confirms the kappa.

### cycles/ guard (evasion table)

`_assert_not_in_cycles()` in `freeze_baselines.py` calls `Path.resolve()` to canonicalize
the output path, then checks every parent `.name` for equality with `"cycles"`. This
blocks:
- Direct subdirectory paths
- `..` traversal attempts
- Symlinks pointing inside cycles/
- Relative paths that resolve into cycles/

The test file `test_freeze_baselines_cli.py` has an 11-case evasion table covering all of
these attack vectors.

### Write-once guard

`_check_write_once()` reads the existing `SHA256SUMS`, finds the record for
`rankings_baselines.json`, and compares against the SHA256 of the on-disk file. If the
content differs, it raises `click.ClickException` unless `--force` is passed. The
SHA256SUMS file is written LAST in the CLI (after all other artifacts), so a partial write
cannot leave a valid SUMS record.

### Assert-not-skipped guard (T11)

`test_f7_baseline_repro.py` calls `pytest.fail()` (not `pytest.skip()`) at module level
if any of the 5 required source files are missing. This ensures that a missing baseline
artifact causes a hard test failure rather than a silent skip.

---

## Bugs found and fixed during implementation

1. **Test isolation (os.chdir restore)**: `test_guard_relative_path_into_cycles` called
   `os.chdir(adjacent)` and initially restored to `tmp_path`. Subsequent tests that used
   relative paths in the same pytest session broke. Fixed by saving
   `original_cwd = Path.cwd()` and restoring in a `finally` block.

2. **Ruff: unused `hashlib` import** in `freeze_baselines.py` (removed).

3. **mypy: `np.ndarray` without type parameters** at 3 locations in `freeze_baselines.py`
   — fixed by importing `numpy.typing as npt` and using `npt.NDArray[np.float64]`.

4. **mypy: unused `type: ignore[arg-type]`** in `test_baselines_freeze.py` at line 152
   — fixed by adding `assert isinstance(pr["kappa_median"], float)` before the float cast.

5. **Ruff: long SHA string literals** in `test_freeze_baselines_cli.py` — fixed by
   assigning `_WRONG_SHA = "a" * 64` and referencing the variable.

6. **Worktree branch missing engine/baselines modules**: the worktree branch was based on
   an old commit (421a2b1) that predated baseline files. Fixed by
   `git stash && git merge plan7/engine-upgrade-recall-pl --no-ff && git stash pop` with
   a uv.lock conflict resolved via `git checkout HEAD -- uv.lock`.
