# Gold-Set LLM-Assisted Adjudication Guide

Human adjudication workflow for expanded gold-set labeling.
LLMs pre-label incidents on RunPod; you review, accept, or override.

## Why This Exists

The premortem on Approach A (2026-05-23) found that using 3 LLMs as
independent gold-set labelers (2-of-3 majority vote) creates correlated
labels that inflate calibration posteriors. LLMs trained on similar corpora
share systematic biases — their agreement demonstrates model homogeneity,
not label validity.

This guide restructures the workflow: LLMs do the heavy lifting as
pre-labelers, but a human adjudicator is the authority on every gold label.

## Prerequisites

1. **LLM pre-labeling complete** — RunPod batch has run 3 models on the
   target incidents. Output is a JSONL file at:
   `projects/owasp-llm/cycles/2026/calibration/llm_prelabels.jsonl`
2. **Rubric frozen** — you must have the frozen rubric open for reference:
   `projects/owasp-llm/cycles/2026/prereg/rubric.json`
3. **Boundary cases doc** — keep `docs/BOUNDARY-CASES.md` open for edge cases

## What the LLM Pre-Labels Look Like

Each line in `llm_prelabels.jsonl` contains:

```json
{
  "incident_id": "GA-04821",
  "text": "A researcher demonstrated that...",
  "model_votes": [
    {"model_id": "model-A", "entry_id": "LLM01", "confidence": 0.92, "rationale": "..."},
    {"model_id": "model-B", "entry_id": "LLM01", "confidence": 0.88, "rationale": "..."},
    {"model_id": "model-C", "entry_id": "LLM05", "confidence": 0.71, "rationale": "..."}
  ],
  "consensus": "LLM01",
  "agreement": "2-of-3",
  "triage_tier": "agree"
}
```

The `triage_tier` field sorts incidents into three buckets for you:

| Tier | Meaning | Your action |
|------|---------|-------------|
| `agree` | All 3 models chose the same entry | Quick-verify: read text, confirm or override |
| `split` | 2-of-3 agree, 1 dissents | Focused review: read text + all 3 rationales |
| `disagree` | All 3 chose different entries, or ≥1 said out-of-scope | Full review: read text, apply rubric from scratch |

## The Workflow

### Step 0: Blind-first read (critical)

For every incident, **read the incident text first and form your own
preliminary label before looking at the LLM votes.** This prevents
anchoring bias — the premortem identified LLM pre-labels anchoring
human judgment as a residual risk.

### Step 1: Process `agree` tier (fastest)

These are incidents where all 3 LLMs agreed. Expected: ~50-60% of incidents.

1. Read the incident text
2. Form your own label (before looking at votes)
3. Check the consensus label
4. If your label matches the consensus: **accept** — write `"adjudicated": "accept"`
5. If your label differs:
   - Re-read the rubric entry for both your label and the consensus label
   - Check `BOUNDARY-CASES.md` for relevant boundary rules
   - Make your final call and write `"adjudicated": "override"` with your label

**Time estimate:** ~30 seconds per incident for accepts, ~2 minutes for overrides.

### Step 2: Process `split` tier (focused review)

These are 2-of-3 agreements. Expected: ~25-35% of incidents.

1. Read the incident text — form your own label first
2. Read all 3 rationales (majority and dissent)
3. Check: does the dissenting model raise a valid point? Often the
   dissent surfaces a boundary rule the majority missed
4. Apply the rubric's `boundary_rules` for the entries in play
5. Write your adjudication:
   - `"adjudicated": "accept"` if you agree with the majority
   - `"adjudicated": "override"` if you agree with the dissent or have a different label
   - `"adjudicated": "multi-label"` if the incident genuinely matches 2+ entries

**Time estimate:** ~2-3 minutes per incident.

### Step 3: Process `disagree` tier (full review)

All 3 models disagreed or flagged out-of-scope. Expected: ~10-15% of incidents.

1. Read the incident text — form your own label
2. Skim the 3 rationales for useful signal (but don't weight them heavily —
   the models couldn't agree, so their reasoning is suspect)
3. Apply the rubric from scratch:
   - Walk through the entry table (see quick reference below)
   - Check `in_scope`, `exclusions`, and `boundary_rules` for candidate entries
   - If no entry fits: `"entry_id": "out-of-scope"` is valid
4. Write your adjudication with `"adjudicated": "override"`

**Time estimate:** ~3-5 minutes per incident.

### Step 4: Flag uncertain cases

If you genuinely can't decide after applying the rubric:

```json
{
  "adjudicated": "uncertain",
  "candidate_entries": ["LLM01", "LLM06"],
  "uncertainty_reason": "injection vector is present but primary impact is excessive agency"
}
```

Uncertain cases get a second pass after all other incidents are coded.
If still uncertain after the second pass, label with the entry whose
`in_scope` definition is the closest match and add a note.

## Output Format

Write your adjudication into a JSONL file at:
`projects/owasp-llm/cycles/2026/calibration/adjudicated_goldset.jsonl`

Each line:

```json
{
  "incident_id": "GA-04821",
  "llm_consensus": "LLM01",
  "adjudicated": "accept",
  "labels": ["LLM01"],
  "notes": null
}
```

For overrides:

```json
{
  "incident_id": "GA-07312",
  "llm_consensus": "LLM05",
  "adjudicated": "override",
  "labels": ["LLM01", "LLM05"],
  "notes": "Injection vector (LLM01) + downstream XSS (LLM05) per boundary rule LLM01↔LLM05"
}
```

For out-of-scope:

```json
{
  "incident_id": "GA-02198",
  "llm_consensus": "LLM09",
  "adjudicated": "override",
  "labels": [],
  "notes": "Not an LLM incident — traditional software supply chain issue with no LLM component"
}
```

## The 20 Entries (Quick Reference)

| ID | Name |
|----|------|
| LLM01 | Prompt Injection |
| LLM02 | Sensitive Information Disclosure |
| LLM03 | Supply Chain Vulnerabilities |
| LLM04 | Data and Model Poisoning |
| LLM05 | Improper Output Handling |
| LLM06 | Excessive Agency |
| LLM07 | System Prompt Leakage |
| LLM08 | Vector and Embedding Weaknesses |
| LLM09 | Misinformation |
| LLM10 | Unbounded Consumption |
| NEW-PMP | Prompt Management and Pipelines |
| NEW-MTIE | Multi-Tenant Isolation Erosion |
| NEW-MA | Model Abuse |
| NEW-ITSCD | Insufficient Training/Serving Configuration and Defaults |
| NEW-WLA | Weak LLM Agent Authorization |
| NEW-MSDA | Model Supply and Dependency Attacks |
| ROLL-CMSB | Cross-Modal Safety Bypass |
| ROLL-LAPTF | Lack of Adversarial Prompt Testing Frameworks |
| ROLL-SICG | Stale or Insufficient Content Guardrails |
| ROLL-CFAS | Cascading Failures in Agentic Systems |

Frame-blind entries (LLM04, LLM08, LLM10) have no classifier rules.
Label them if present — they contribute to recall calibration even
though they are excluded from the kappa computation.

## Session Discipline

- **Max 2 hours per session.** Fatigue causes label drift.
- **Take breaks between tiers.** Do all `agree` first, then `split`, then `disagree`.
- **Track your override rate.** If you're overriding >50% of `agree`-tier
  labels, the LLM consensus is systematically biased and you should note
  which entries are most affected.
- **Log session start/end times** in a `coding_sessions.md` file alongside
  the adjudicated output. This feeds into the methodology changelog.

## After Adjudication

The `adjudicated_goldset.jsonl` file feeds into the calibration pipeline:

```bash
# Convert adjudicated gold labels to calibration tally format
incident-rank cal-tally \
  --cycle projects/owasp-llm/cycles/2026 \
  --gold-override projects/owasp-llm/cycles/2026/calibration/adjudicated_goldset.jsonl

# Recompute calibration posteriors
incident-rank cal-calibrate \
  --cycle projects/owasp-llm/cycles/2026

# Check diagnostic for improved adequacy
cat projects/owasp-llm/cycles/2026/calibration/diagnostic.json | python -m json.tool
```

## Quality Metrics to Report

After adjudication, compute and record:

1. **Override rate by tier:** What % of `agree`/`split`/`disagree` labels did you change?
2. **Override rate by entry:** Which entries did LLMs get wrong most often?
3. **Krippendorff's alpha (LLM vs human):** Computed between the LLM consensus
   and your final labels. This measures how well the LLMs approximate human judgment.
4. **Multi-label rate:** What % of incidents got 2+ entry labels?
5. **Out-of-scope rate:** What % of incidents matched no entry?

These metrics are recorded in the methodology changelog and inform whether
future cycles can increase LLM autonomy (high alpha) or need more human
review (low alpha).

## Time Estimate

| Tier | Expected % | Per-incident time | Subtotal |
|------|-----------|-------------------|----------|
| `agree` | 55% of ~4,000 = ~2,200 | 30s accept / 2min override | ~25-30 hours |
| `split` | 30% of ~4,000 = ~1,200 | 2-3 min | ~40-60 hours |
| `disagree` | 15% of ~4,000 = ~600 | 3-5 min | ~30-50 hours |
| **Total** | | | **~95-140 hours** |

This is a significant commitment. Strategies to reduce it:

1. **Prioritize entries with low classifier-positive counts** — rare entries
   need gold labels most urgently (see premortem F7)
2. **Start with high-confidence `agree` labels** (confidence > 0.9 from all
   3 models) — these have the highest accept rate
3. **Batch by entry** — coding all incidents for one entry at a time reduces
   context-switching between rubric definitions
4. **Target 30 gold labels per entry minimum** — stop at the floor if time
   is limited, prioritizing coverage over depth

Minimum viable adjudication: ~1,200 incidents (30 per entry × 20 entries × 2
strata) at ~2 min average = ~40 hours. This is the floor that produces
usable calibration for all 20 entries.
