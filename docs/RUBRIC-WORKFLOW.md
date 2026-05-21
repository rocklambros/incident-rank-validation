# Rubric Drafting and Freeze Workflow

This document describes the procedural steps for drafting, adjudicating, and
freezing the classification rubric for an incident-rank-validation cycle.

## Prerequisites

- Engine v0.3.0+ with rubric data model, CLI commands, and gate logic.
- Entry definitions vendored in `projects/<project>/cycles/<cycle>/taxonomy/`.
- `docs/REVIEWERS.md` consulted for reviewer state (INTERIM vs EXTERNAL).

## Vote-Blindness Rule (HANDOFF §6 control 2)

The rubric drafter (Claude + Rock) MUST NOT view vote results during drafting.
This means:

1. Do not open the `2026/polling/` directory in the source repo
   (`https://github.com/GenAI-Security-Project/GenAI-LLM-Top10`) or any
   file containing vote results.
2. Do not read the `Analysis` or `Results` sheets from the voting spreadsheet.
3. The CASE 2 ranking order in HANDOFF §2 is metadata about *what* the entries
   are and how they were named, not vote-influence data. Reading HANDOFF §2
   for entry names is permitted; reading it to infer relative importance is not.

The `rubric_attestation.json` records whether vote-blindness was maintained.
If violated, the cycle is non-publishable regardless of other controls.

## Step-by-Step Procedure

### 1. Vendor Taxonomy

Copy the 20 entry definitions to `projects/<project>/cycles/<cycle>/taxonomy/`
and create `taxonomy.json`. Use the `vendor-snapshot` pattern from Plan 2.

### 2. Draft Rubric Entries

For each of the 20 entries, produce a `RubricEntry` with all 8 required fields:

- **entry_id**: Canonical identifier (e.g., `LLM01`).
- **canonical_name**: Official name from the taxonomy.
- **in_scope**: What incidents this entry covers.
- **exclusions**: What this entry explicitly does NOT cover.
- **boundary_rules**: Pairwise rules against adjacent/confusable entries.
- **positive_indicators**: Keywords, patterns, or signals that suggest classification.
- **negative_indicators**: Signals that suggest classification elsewhere.
- **co_occurrence_pairs**: Entry pairs expected to co-occur on the same incident.

Rolled-up candidates (4 entries) get their own rubric entry with
`is_rollup_candidate=true` and `rolled_into` pointing to the parent.

### 3. Identify Boundary Cells

For every pair of entries that could be confused:

- Write a boundary rule on BOTH sides (rules must be paired).
- If the boundary is genuinely ambiguous (50/50): mark `is_ambiguous=true`.
  This propagates as label uncertainty in the measurement model.
- If the boundary is clear: mark `is_ambiguous=false` and state the rule.

### 4. Rock Adjudicates

Rock reviews every boundary rule and either:

- Confirms the rule as written, or
- Marks it `ambiguous-both-labels` with rationale.

Adjudication decisions go into `adjudication_log.json`.

### 5. Populate Attestation

Create `rubric_attestation.json`:

```json
{
  "viewed_corpus_before_drafting": false,
  "viewed_corpus_details": "",
  "viewed_vote_data_before_drafting": false,
  "viewed_vote_data_details": ""
}
```

If the drafter viewed corpus samples, set `viewed_corpus_before_drafting` to `true` and list which samples.
The report will carry a "corpus-informed rubric" caveat (HANDOFF §6 control 11(d)).

If the drafter viewed any vote results (including rank ordinals from HANDOFF §2), set
`viewed_vote_data_before_drafting` to `true` and describe what was seen. A vote-data
exposure makes the cycle non-publishable per HANDOFF §6 control 2.

### 6. Validate

```bash
uv run python -m engine.cli.main validate-rubric \
  --rubric projects/owasp-llm/cycles/2026/prereg/rubric.json \
  --taxonomy projects/owasp-llm/cycles/2026/taxonomy/taxonomy.json
```

### 7. Freeze (requires external reviewer)

```bash
uv run python -m engine.cli.main freeze-rubric \
  --rubric projects/owasp-llm/cycles/2026/prereg/rubric.json \
  --cycle-dir projects/owasp-llm/cycles/2026
```

This emits the rubric hash. Add it to the prereg manifest and lock.

### 8. External Reviewer Signoff

Per HANDOFF §6 control 5 and REVIEWERS.md:

1. External reviewer reads the rubric.
2. Reviewer writes attestation file at `docs/REVIEWERS/<name>-rubric.txt`.
3. Compute SHA-256, update REVIEWERS.md, commit atomically.
4. Update the cycle manifest with reviewer identity and hash.

If no external reviewer is available, the run proceeds as `non_publishable=True`.

## Rollup Sub-Test Consumption Spec (Premortem4 R5)

The rubric contains 4 rolled-up candidate entries (`is_rollup_candidate=true`, `rolled_into` pointing to
a parent incumbent). This section specifies how Plans 4-5 should consume them:

1. **Independent classification:** Rollup candidates are classified independently alongside their parent.
   The classifier evaluates each incident against BOTH the parent entry and the rollup entry.
2. **Dual labeling:** If an incident matches a rollup candidate, it receives BOTH the rollup label AND
   the parent label. The rollup classification is a sub-test of the parent, not a competing label.
3. **Parent contribution:** The concordance statistic for the parent entry includes incidents that matched
   via the rollup sub-test. Rollup matches contribute to the parent's evidence pool.
4. **No independent rank:** Rollup candidates do NOT have their own rank position in the final Top 10.
   They exist to test whether the parent entry's scope is too broad (i.e., whether the rolled-up concept
   should have been a separate entry). This sub-test result is reported as a methodology finding.

## Adapter Compatibility Warning (Premortem4 R11)

The frozen rubric covers 20 entries (10 incumbents + 6 standalone new + 4 rollup). The provisional
adapter (`engine/adapters/genai_agentic.py:_PROVISIONAL_2025_ENTRIES`) classifies against only 10
incumbent entries. Do NOT run the classify pipeline with this rubric until Plan 5 updates the adapter.
A 20-entry rubric with a 10-entry classifier will produce incomplete classifications that silently
pass the `require_rubric_hash_match()` gate (which checks the rubric hash, not classifier coverage).

## Rubric Amendments

Pre-registration is a commitment, not a prison. If a boundary rule deficiency is discovered
after freeze (e.g., during gold-set labeling in Plan 4), the rubric may be amended with
full disclosure:

1. **Version bump:** Increment `Rubric.version` (e.g., `"1.0.0"` → `"1.1.0"`).
2. **Rationale:** Document the deficiency and the change in the pre-registration diff
   artifact (HANDOFF §5.5). Include which boundary rules changed and why.
3. **Re-freeze:** Run `freeze-rubric` to produce a new rubric hash.
4. **Re-lock manifest:** Update `rubric_hash` in the manifest and re-run `write_lock`.
5. **Disclosure:** The report's pre-reg diff section shows original vs amended rubric
   hashes and the rationale. The amendment is disclosed, not hidden.
6. **Re-review:** If the run is targeted for publication, the amended rubric requires
   external reviewer re-signoff. If the reviewer is unavailable, the report carries an
   "amendment not independently reviewed" disclosure alongside the pre-reg diff. An
   un-reviewed amendment does not make the run non-publishable by itself (the original
   rubric WAS reviewed), but the disclosure is mandatory.

The default is "do not amend" — the rubric is frozen for a reason. Amendments are for
genuine deficiencies discovered during execution, not for preference changes or
optimization. Each amendment requires Rock's sign-off and is recorded in the
adjudication log.
