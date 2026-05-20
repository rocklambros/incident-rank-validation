---
name: publication-formality-calibration
description: This project ships methodology to the OWASP working group, not an academic journal. Mechanical enforcement of academic-publication-grade formalities is excessive when discipline + standing disclosure are the right control. Includes a decision tree for ambiguous cases.
metadata:
  type: feedback
---

When deciding between mechanical enforcement and documented discipline, calibrate to the audience:

- **Mechanism (enforce in code):** a defect would cause silent methodology error. Examples: NUTS leakage math, frame-blind censoring, drift-signoff gating, cross-cycle lineage refusal, prereg hash-lock.
- **Discipline + disclosure (document, do not enforce):** the control is academic-publication formality and the audience is the OWASP working group operating on transparency + standing caveats. Examples: reviewer-≠-ranking-author, peer-review SOPs, manual checklists that gate publication.

## Decision tree

Apply **mechanism** if the defect would cause silent methodology error:
- Gold-set partition disjoint from rule-development incidents (silent leakage bias)
- NUTS leakage matrix overcount (silent rank bias)
- Prereg hash-lock (silent post-hoc tuning)
- Drift-signoff gate (silent acceptance of contaminated snapshot)
- Cross-cycle lineage refusal (silent cross-cycle false comparison)
- Frame-blind censoring (silent ranking of unmeasurable entries)
- Hyperparameter pre-registration in manifest (silent post-hoc tuning)
- Overlap-weights column-stochastic check (silent FP double-counting)
- Quadratic-kappa zero-denominator handling (silent nonsense headline)
- Post-hoc register integrity chain (silent deletion of inconvenient analyses)

Apply **discipline + disclosure** if the defect would cause visible process irregularity, not silent error:
- Reviewer-≠-ranking-author (HANDOFF §4 single-author stamp IS the disclosure control)
- Reviewer signed off before viewing results (`viewed_results_before_signoff: bool` self-declared)
- Rubric author viewed corpus before drafting (`RubricDraftingAttestation` self-declared)
- Peer-review SOPs (OWASP-WG audience, not academic journal)
- 30-day publication holdout (process commitment, not engine refusal)
- `--accept-drift-signoff` reason string content (engine enforces length floor + persistence, not semantics)

## Ambiguous-case rule

When the category isn't obvious, ask: *"if this defect occurs and nobody notices, does the published ranking become wrong?"*
- Yes → mechanism.
- No (the published ranking is fine but the process is irregular) → discipline + disclosure.

**Why:** Stated 2026-05-20 in `incident-rank-validation`. Rock said "we aren't publishing this in some sort of academic journal so relax the enforcement gap" in response to a proposal to mechanically enforce reviewer ≠ ranking author. Decision tree added 2026-05-20 after Premortem 3 surfaced ambiguity (M10).

**How to apply:** Pairs with [[correctness-over-speed]]. Correctness-over-speed applies to methodology integrity (use the decision tree to find these). This calibration applies to *process* formality. Default: if HANDOFF specifies "discipline + disclosure," do that; do not escalate to mechanical enforcement unless explicitly asked. When ambiguous, apply the decision-tree question.
