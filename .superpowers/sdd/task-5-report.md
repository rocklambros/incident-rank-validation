# Task 5 Report: Stage-2 Prompt Delimiter Injection Fix (Plan 8e T5)

## Status: DONE

## Commit SHA
`6bfcf6d`

## Files Modified

- **Modified:** `engine/classify/stage2_prompt.py`
  - Added `_neutralize_delimiters(text: str) -> str` above `build_messages`
  - Changed `incident_text=incident.text` to `incident_text=_neutralize_delimiters(incident.text)` in `build_messages`

- **Created:** `tests/security/test_stage2_delimiter_escape.py`
  - Two tests as specified in the brief
  - Added `# type: ignore[arg-type]` on the two `build_messages` call sites because mypy STRICT rejects the local `_Inc` dataclass where `IncidentRecord` is expected

## Commands and Output

### ruff check .
```
$ uv run ruff check .
All checks passed!
```

### mypy engine tests
```
$ uv run mypy engine tests
Success: no issues found in 225 source files
```

### pytest tests/security/test_stage2_delimiter_escape.py -v
```
============================= test session starts ==============================
platform linux -- Python 3.12.2, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/rock/github_projects/incident-rank-validation
configfile: pyproject.toml
plugins: xdist-3.6.1, anyio-4.13.0, env-1.1.5
collected 2 items

tests/security/test_stage2_delimiter_escape.py ..                        [100%]

============================== 2 passed in 0.05s ===============================
```

### uv run pytest -q (full suite)
Exit code: 0 (three independent runs confirmed). No failures, no errors.
10 expected failures (XFAIL) for `test_stage2_injection_fixture.py` — pre-existing,
marked as Plan 5 deliverable.

## Concerns

1. **`# type: ignore[arg-type]` in test:** The brief prescribes `_Inc(id, text)` as
   the incident stub but `build_messages` is typed `incident: IncidentRecord`. Mypy
   strict rejects the structural mismatch. Added `# type: ignore[arg-type]` on both
   call sites — minimal and correct; runtime behavior is identical to the brief spec.
   An alternative would be to widen the production signature to a `Protocol` with
   `.text: str`, but that is a larger production-interface change beyond this task's
   scope.

2. **`build_prompt` is NOT neutralized:** The legacy `build_prompt` function (used by
   `tests/security/test_stage2_delimiter.py`) still inserts `incident.text` un-escaped.
   The brief scopes the fix to `build_messages` only, so `build_prompt` is left
   unchanged. The existing delimiter tests only assert on clean text or that forged
   delimiters appear inside the fenced region — no regression. However, `build_prompt`
   remains exploitable if called with attacker-controlled text. This is a known gap
   outside this task's scope.

## Fix wave

**Applied by Plan 8e T5 review (code-review findings).**

Closed concern #2 above: `build_prompt` was a latent injection path because it called
`_neutralize_delimiters` for `build_messages` but not for itself.

### Files changed

- `engine/classify/stage2_prompt.py` — `incident_text=incident.text` → `incident_text=_neutralize_delimiters(incident.text)` in `build_prompt` (line 102)
- `tests/security/test_stage2_delimiter.py` — `test_injection_via_fake_delimiter_close` assertion updated from "merely contained" (weak) to "exactly one delimiter" (strong/neutralized)
- `tests/security/test_stage2_delimiter_escape.py` — `test_build_prompt_also_neutralizes_delimiters` appended

### Commands and output

```
$ uv run ruff check .
All checks passed!

$ uv run mypy engine tests
Success: no issues found in 225 source files

$ uv run pytest tests/security/test_stage2_delimiter.py tests/security/test_stage2_delimiter_escape.py -v
collected 8 items
tests/security/test_stage2_delimiter.py .....        [ 62%]
tests/security/test_stage2_delimiter_escape.py ...   [100%]
8 passed in 0.08s

$ uv run pytest -q
[100%] — exit code 0
10 XFAIL (pre-existing Plan 5 markers), 0 failures, 0 errors
```

### Commit SHA

`0e96093` — fix(classify): neutralize delimiters in build_prompt too (Plan 8e T5 review)
