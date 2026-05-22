# Gold-Set Staffing Plan

## Status: SCAFFOLDING — labels not yet coded

Per HANDOFF §5.3: "The gold set is the load-bearing artifact and the project critical path.
Before any engine build on real data, a staffing plan, a time budget, and a power calculation
are produced and committed."

## Staffing

- **Coder 1:** <TBD — domain expert, independent of ranking author>
- **Coder 2:** <TBD — domain expert, independent of ranking author>
- **Adjudicator:** Rock Lambros (ranking author; adjudicates disagreements on the record)

## Coding protocol

Dual independent coding against the byte-identical frozen rubric
(`projects/owasp-llm/cycles/2026/prereg/rubric.json`, hash: 2383f398...`).
Blind to classifier label. Blind to vote data.

Krippendorff's alpha reported. Disagreements adjudicated by the adjudicator.
Adjudication log is a committed artifact at `projects/owasp-llm/cycles/2026/goldset/adjudication_log.json`.

## Sample size

Power calculation targeting usable CI width on rare entries.
Order of magnitude: several hundred to ~1,000 labels (HANDOFF §5.3).

- Precision frame: stratified sample of classifier-positive assignments per entry.
- Recall frame: independent of classifier label, stratified by corpus_stratum, source_class,
  confidence, oversampling rare entries and contamination stratum.

Exact sample size from power calculation: <TBD — computed before coding begins>.

## Time budget

- Estimated coding time per label: <TBD>
- Total estimated coding time: <TBD>
- Target completion: <TBD>

## Snapshot binding

Gold-set labels are bound to snapshot hash: `24806f1a4f0917f85f7509d6cb2a34b12e56eb902714b37bc2b03a2cf1a246bb`

## Gate

This document exists as a scaffolding prerequisite. The gold-set labels must be
human-coded before the classify phase runs on real data. Plan 4's calibration
pipeline is the engineering infrastructure; the labels are the human artifact.
