## Summary

<!-- 1-3 bullet points describing what this PR does and why. -->

## Integrity Checklist

- [ ] **Vote-blindness**: this PR does not read, import, or reference vote data
  outside `engine/decide/` and `engine/vote/`.
- [ ] **Corpus B isolation**: this PR does not import corpus B artifacts in
  `engine/model/inference.py`.
- [ ] **Pre-registration**: if this PR modifies the manifest, rubric, taxonomy,
  or hyperparameters after a lock step, a pre-reg diff entry is included.
- [ ] **Methodology changelog**: if this PR changes methodology (likelihood,
  hyperparameters, statistics, gates), `docs/METHODOLOGY-CHANGELOG.md` has a
  new semver entry.
- [ ] **Tests pass**: `uv run pytest`, `uv run mypy engine tests`,
  `uv run ruff check .` all green.

## Test Plan

<!-- How was this change tested? Which test files cover it? -->
