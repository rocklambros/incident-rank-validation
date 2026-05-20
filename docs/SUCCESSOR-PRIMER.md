# Successor Primer

## Project purpose
This engine validates ranked taxonomies (like OWASP Top 10 lists) against incident corpora using Bayesian inference. It answers: does the community vote ranking concord with incident-derived prevalence, with measurement uncertainty modeled on both sides?

## Key principles
1. **Pre-registration discipline:** all parameters locked before seeing results
2. **Frame-blind honesty:** entries invisible to the corpus are reported unmeasurable, never low-prevalence
3. **Transparency-first publication:** never suppress findings; report limitations prominently
4. **Mechanism over discipline:** where silent error is possible, use mechanism; where not, use discipline + disclosure

## Architecture overview
- `engine/` — the analysis engine (taxonomy-neutral)
- `projects/` — per-project data (taxonomy, corpus, results)
- `tests/` — unit tests, proof tests, security fixtures
- CLI entry point: `incident-rank run-synthetic`, `incident-rank infer`, `incident-rank decide`

## If something looks wrong (F-defenseindepth warning, v2.4)

The engine has many integrity controls (prereg lock, drift signoff, cross-cycle refusal, post-hoc Merkle chain, transparency-first publication, etc.). This defense-in-depth can create FALSE CONFIDENCE: when a finding surfaces, the first instinct is "the engine has all those controls, the bug must be upstream."

**Treat the engine as a hypothesis, not a guarantee.** Before assuming the controls fired correctly:

1. Identify the specific control(s) relevant to the finding.
2. Check the control's *assumptions* — not whether it ran, but whether its premises hold for this case. (Example: drift detection assumes a "previous" snapshot exists; first-run cycles bypass it silently.)
3. Read the test that exercises the control. If the test doesn't exist or exercises a different regime than the one that failed, the control hasn't been validated for the failure mode you're looking at.

When in doubt, check the assumption rather than the data. Premortem 3 closed several findings of the form "the control runs but doesn't measure what its name implies" (M14 selection bias, M18 robustness consistency). The same class of bug will recur if you trust the names.

## Where to start
1. Read `docs/HANDOFF.md` — the approved design specification
2. Run `uv run pytest -v` to verify tests pass
3. Run `uv run incident-rank run-synthetic --cycle projects/synthetic/cycles/2026 --corpus-mode synthetic` to see the full pipeline
4. Read `docs/RUNBOOK.md` for failure modes
