# Gold-Set Coding Guide

Step-by-step walkthrough for manually coding the gold-set calibration batches.
This produces the `calibration.json` artifact that feeds into NUTS inference (Plan 5).

## Prerequisites

Before starting, verify:

1. **Plans 1-5 merged to main** (tags `v0.1.0-plan1` through `v1.0.0-plan5`)
2. **Rubric frozen** — hash `2383f39825e4949251d514df49d1fed586874714aa9396fa24f3c0774534b120` (20 entries)
3. **Manifest locked** — `manifest.lock` present in cycle directory
4. **Snapshot available** — corpus snapshot at a known directory with a snapshot date

## Overview

```
Stage 1: cal-classify     — automated, ~30 seconds
Stage 2: cal-sample       — automated, ~5 seconds
Stage 3: MANUAL CODING    — you do this, ~33-50 hours
Stage 4: cal-tally        — automated, ~10 seconds
Stage 5: cal-calibrate    — automated, ~5 seconds
Stage 6: cal-cv-stability — automated, ~5 seconds
```

---

## Stage 1: Classify the Corpus

Run the deterministic keyword/indicator classifier on the full corpus:

```bash
incident-rank cal-classify \
  --cycle projects/owasp-llm/cycles/2026 \
  --rubric projects/owasp-llm/cycles/2026/prereg/rubric.json \
  --manifest projects/owasp-llm/cycles/2026/prereg/manifest.json \
  --snapshot-dir projects/owasp-llm/cycles/2026/corpora \
  --snapshot-date 2026-01-15
```

**Output:** `projects/owasp-llm/cycles/2026/calibration/classifications.json`

Verify: check the classification count and rule hash match the manifest.

---

## Stage 2: Draw Samples

Draw precision-frame and recall-frame samples:

```bash
incident-rank cal-sample \
  --cycle projects/owasp-llm/cycles/2026
```

**Output:** Sample results + batch files written to `calibration/batches/`

This produces two types of batches:

### Precision-frame batches

- One batch per non-frame-blind entry × stratum
- Each contains ~40 incidents that the classifier labeled as matching that entry
- **Your question for each incident:** "Does this incident actually match [entry_id]?"
- Possible labels: `["LLM01"]` (yes, it matches) or `[]` (no, false positive)

### Recall-frame batches

- One batch per stratum
- Each contains ~100 incidents sampled from the full corpus
- **Your question for each incident:** "Which entries (from all 20) does this incident match?"
- Possible labels: `["LLM01", "LLM05"]` (multi-label) or `[]` (matches nothing)
- The batch header includes a `coding_checklist` with all 20 entry names for reference

---

## Stage 3: Manual Coding (YOU DO THIS)

### Setup

1. Navigate to `projects/owasp-llm/cycles/2026/calibration/batches/`
2. You'll find JSON batch files — each contains an `incidents` array
3. Open each file in your editor

### How to code a precision-frame batch

The batch header tells you: `"frame": "precision"`, `"entry_id": "LLM01"`, `"stratum": "security"`

For **each incident** in the batch:

1. Read the incident text
2. Decide: Does this incident genuinely describe [entry_id]?
3. Fill in the `labels` field:
   - **Yes, it matches:** `"labels": ["LLM01"]`
   - **No, false positive:** `"labels": []`
   - **It matches AND another entry too:** `"labels": ["LLM01", "LLM06"]`

### How to code a recall-frame batch

The batch header tells you: `"frame": "recall"`, `"entry_id": null`, `"stratum": "security"`

For **each incident** in the batch:

1. Read the incident text
2. Read the `coding_checklist` in the batch header for the 20 entry definitions
3. Decide: Which entries does this incident match? (could be zero, one, or many)
4. Fill in the `labels` field:
   - **No match:** `"labels": []`
   - **One entry:** `"labels": ["LLM09"]`
   - **Multiple entries:** `"labels": ["LLM01", "LLM05", "NEW-PMP"]`

### The 20 entries (quick reference)

| ID | Name | Frame-Blind? |
|----|------|:---:|
| LLM01 | Prompt Injection | |
| LLM02 | Sensitive Information Disclosure | |
| LLM03 | Supply Chain Vulnerabilities | |
| LLM04 | Data and Model Poisoning | YES |
| LLM05 | Improper Output Handling | |
| LLM06 | Excessive Agency | |
| LLM07 | System Prompt Leakage | |
| LLM08 | Vector and Embedding Weaknesses | YES |
| LLM09 | Misinformation | |
| LLM10 | Unbounded Consumption | YES |
| NEW-PMP | Prompt Management and Pipelines | |
| NEW-MTIE | Multi-Tenant Isolation Erosion | |
| NEW-MA | Model Abuse | |
| NEW-ITSCD | Insufficient Training/Serving Configuration and Defaults | |
| NEW-WLA | Weak LLM Agent Authorization | |
| NEW-MSDA | Model Supply and Dependency Attacks | |
| ROLL-CMSB | Cross-Modal Safety Bypass | |
| ROLL-LAPTF | Lack of Adversarial Prompt Testing Frameworks | |
| ROLL-SICG | Stale or Insufficient Content Guardrails | |
| ROLL-CFAS | Cascading Failures in Agentic Systems | |

Frame-blind entries (LLM04, LLM08, LLM10) are **excluded** from calibration — they have no classifier rules and no precision frames. You may still encounter them in recall frames; label them if present.

### Coding rules

1. **Do not modify `incident_id`, `text`, or `batch_header` fields** — the tally stage verifies integrity
2. **Every incident must have `labels` set** — `null` means uncoded and will be skipped with a warning
3. **`[]` is a valid label** — it means "no match" (different from null/uncoded)
4. **Multi-label is expected** — one incident can match multiple entries
5. **Use only valid entry IDs** — the tally stage rejects unknown IDs

### If you make a mistake

If you need to correct a label after initial coding:

1. Update the `labels` field to the correct value
2. Add an `amendment` note:

```json
{
  "incident_id": "GA-00123",
  "text": "...",
  "labels": ["LLM01", "LLM06"],
  "notes": "corrected from [LLM01] — missed LLM06 co-occurrence",
  "amendment": "corrected from [LLM01] — missed LLM06 co-occurrence"
}
```

### Time estimate

- ~1,000 incident codings total
- ~2-3 minutes per incident
- **Total: 33-50 hours**
- Recommend breaking into sessions of 1-2 hours to avoid fatigue-driven label drift

---

## Stage 4: Tally

After all batches are coded, aggregate the results:

```bash
incident-rank cal-tally \
  --cycle projects/owasp-llm/cycles/2026
```

**Output:** `calibration/tally.json`

This validates all batch files (hash integrity, valid entry IDs, no missing labels) and aggregates into precision/recall counts per entry per stratum.

**If validation fails:** the tally stage reports ALL errors across ALL batches in one pass. Fix them all, then re-run.

---

## Stage 5: Compute Calibration Posteriors

```bash
incident-rank cal-calibrate \
  --cycle projects/owasp-llm/cycles/2026
```

**Output:**
- `calibration/posteriors.json` — Beta posteriors (this is what Plan 5 reads)
- `calibration/diagnostic.json` — calibration adequacy report

Review the diagnostic:
- **"adequate"** entries: 90% CI width < 0.30, good to go
- **"wide: small-sample"** entries: posteriors are wide due to few labels, not poor classifier
- **"wide: recall-frame-only"** entries: new/rollup entries with no precision data
- **"no-data"** entries: frame-blind or no classifier positives

---

## Stage 6: Cross-Validation Stability

```bash
incident-rank cal-cv-stability \
  --cycle projects/owasp-llm/cycles/2026
```

**Output:** `calibration/cv_result.json`

k=5 cross-validation of the calibration posteriors. Check:
- Fold variance < 0.05 for high-count entries → "stable"
- min_per_fold < 5 → "unstable" annotation (expected for low-count entries)

---

## After Calibration

With `posteriors.json` produced, the Plan 5 pipeline is unblocked:

```bash
# Classify real corpus
incident-rank classify-real --cycle projects/owasp-llm/cycles/2026 --execute

# Run NUTS inference (CPU-pinned, ~10 minutes)
incident-rank infer-real --cycle projects/owasp-llm/cycles/2026 --execute

# Decision layer (after vote data enters)
incident-rank decide-real --cycle projects/owasp-llm/cycles/2026 --execute
```

## Single-Coder Waiver

You are coding solo. This means:
- No inter-rater reliability is computed
- Results are **non-publishable** per HANDOFF §4
- Your `coder_id` is preserved so future coders' work can be added alongside yours
- When external coders are recruited, inter-rater reliability will be computed across all coders
