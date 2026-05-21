# Reviewers

HANDOFF v2.3 §6 control 5 + §4 Crosswalk-authorship row require both reviewers to be *independent* of the ranking author for cycles to be publishable. This file records the current state of reviewer identification per Plan 1 v4 acceptance gate (criterion 19).

## Current state (2026-05-20): INTERIM — single-reviewer mode

The ranking author is **Rock Lambros**. Per HANDOFF §4, when no independent reviewer is available, internal runs may proceed but reports are stamped "single-author rubric, uncontrolled" and are **not publishable** to the OWASP working group.

### Rubric reviewer

- **Name:** Rock Lambros
- **Affiliation:** OWASP GenAI Top 10 effort lead (ranking author of this project)
- **Status:** SELF-REVIEW — same identity as ranking author; non-publishable per HANDOFF §4
- **Attestation file:** docs/REVIEWERS/rock-lambros-rubric.txt
- **Attestation SHA-256:** 1f8eebe6f8e2ac853d72be88de94e6986eecf29b9ea5b0dc0a351613d23cbd7b
- **Methodological consequence:** the rubric is single-authored. Manifests signed by Rock as rubric reviewer are non-publishable per HANDOFF §4, regardless of what the `non_publishable` field derives.

### Statistical reviewer

- **Name:** Rock Lambros
- **Affiliation:** OWASP GenAI Top 10 effort lead (ranking author of this project)
- **Status:** INTERIM — same identity as ranking author
- **Methodological consequence:** the inference model has not been audited by an independent statistician. Manifests signed by Rock as statistical reviewer are non-publishable per HANDOFF §4.

## What this configuration enables

- Plan 1 / Plan 2 / Plan 3 / Plan 4 implementation work proceeds. These are internal-only prerequisites of a publishable cycle; reviewer independence is not required to execute them.
- Synthetic e2e cycles run normally (already non-publishable by construction in Plan 1).
- Plan 5 cycles may run as **internal-only** artifacts marked `non_publishable=True`.
- All engine integrity controls (prereg lock, drift signoff, cross-cycle refusal, transparency-first publication) continue to operate.

## What this configuration blocks

- **Publishable LLM 2026 cycle.** Cannot publish results to the OWASP working group without:
  - external rubric reviewer (an OWASP working-group member who is not Rock and does not report to Rock in the OWASP work), AND
  - external statistical reviewer (independent of Rock and of the rubric reviewer),
  - both identified, attested, and signed off BEFORE the first `infer` run for the cycle.

## Path to publishable

When external reviewers are identified, replace the entries above with:

```
### Rubric reviewer
- Name: <reviewer name>
- Affiliation: <affiliation; MUST NOT be Rock or report to Rock in OWASP work>
- Availability commitment: <signoff schedule>
- Attestation file: docs/REVIEWERS/<reviewer-name-slug>-rubric.txt
- Attestation SHA-256: <computed at commit time>
```

Once both external reviewers complete attestation (committed signed text + SHA-256, per HANDOFF §6.5), a new manifest can be signed with `non_publishable=False` derivation.

## Mechanical-enforcement posture

This project ships to the OWASP working group, not to an academic journal. The reviewer-≠-ranking-author check is intentionally **discipline-based**, not mechanically enforced. Rationale: the audience operates on transparency + standing disclosure, not on peer-review SOPs. HANDOFF §4 marks single-author rubrics as non-publishable; that disclosure is the control. The PRE-PUBLISH CHECKLIST below is the procedural artifact that anchors the discipline.

## PRE-PUBLISH CHECKLIST

Every item must hold before any external publication of cycle results:

- [ ] Rubric reviewer identity ≠ ranking author (Rock).
- [ ] Statistical reviewer identity ≠ ranking author (Rock).
- [ ] Rubric reviewer and statistical reviewer are different identities.
- [ ] Both reviewers' attestation files are committed with SHA-256 recorded in the manifest.
- [ ] Both `signed_at` timestamps precede the first `infer` run for the cycle (verifiable from `inference_provenance.json`).
- [ ] Both `viewed_results_before_signoff` = False.
- [ ] Manifest `non_publishable` derives to False (presence + clean attestations; reviewer-≠-author verified manually here, not by the engine).
- [ ] Plan 5 cycle output is reviewed against this entire checklist before any external sharing.

## History

- 2026-05-20: Initial REVIEWERS.md created. Rock identified as sole reviewer for interim. External reviewers TBD before Plan 5 publishable cycle.
