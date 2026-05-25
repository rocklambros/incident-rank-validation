# Human Adjudication Instructions

## Attached Files

> **The following files are attached and referenced throughout this document.**

| File | Description |
|------|-------------|
| `llm_prelabels.jsonl` | 5,972 incidents with full text + 3-model votes, sorted by triage tier (agree → split → disagree). This is your primary working file for Frame 1. |
| `rubric.json` | Frozen rubric with 20 entry definitions, scope rules, exclusions, and boundary rules. Your authority for label decisions. |
| `BOUNDARY-CASES.md` | Edge-case disambiguation rules for entries that overlap (e.g., LLM01 vs LLM05, LLM06 vs NEW-WLA). |
| `labeled_incidents.json` | 6,639 classifier-assigned labels (hybrid consensus). Your working file for Frame 2 precision verification. |
| `diagnostic.json` | Calibration diagnostic showing which entries have precision data and which don't. Reference only. |
| `GOLDSET-LLM-ADJUDICATION-GUIDE.md` | Full adjudication protocol with examples, anti-bias rules, and session discipline. |

---

## Overview

You are reviewing LLM-classified incidents to build a gold-standard calibration dataset. This has two frames:

- **Frame 1 — Recall Adjudication:** Read incident text, form your own label blind, then compare to LLM consensus. Accept or override. **~40 hours.**
- **Frame 2 — Precision Verification:** For entries with zero/low precision data, answer "is this correctly classified?" Yes/no. **~1.5 hours.**

Both frames produce JSONL files that feed into the calibration pipeline. The pipeline uses these to correct precision posteriors for each entry, which directly affects kappa.

---

## Output Files You Create

| File | Frame | Location |
|------|-------|----------|
| `adjudicated_goldset.jsonl` | Frame 1 | `projects/owasp-llm/cycles/2026/calibration/` |
| `precision_verification.jsonl` | Frame 2 | `projects/owasp-llm/cycles/2026/calibration/` |

---

## Entry Quick Reference

| ID | Name |
|----|------|
| LLM01 | Prompt Injection |
| LLM02 | Sensitive Information Disclosure |
| LLM03 | Supply Chain Vulnerabilities |
| LLM04 | Data and Model Poisoning |
| LLM05 | Improper Output Handling |
| LLM06 | Excessive Agency |
| LLM07 | Hidden Context Exposure |
| LLM08 | Vector and Embedding Weaknesses |
| LLM09 | Misinformation |
| LLM10 | Unbounded Consumption |
| NEW-PMP | Persistent Memory Poisoning |
| NEW-MTIE | MCP Tool Interface Exploitation |
| NEW-MA | Model Misalignment |
| NEW-ITSCD | Inference-Time Side-Channel Disclosure |
| NEW-WLA | Weaponized LLM Abuse |
| NEW-MSDA | Model Scheming and Deceptive Alignment |
| ROLL-CMSB | Cross-Modal Safety Bypass |
| ROLL-LAPTF | LLM Artifact Promotion Trust Failure |
| ROLL-SICG | Systemic Insecure Code Generation |
| ROLL-CFAS | Compositional Fine-tuning Alignment Subversion |

---

# Phase 1: Frame 1 — Recall Adjudication

## 1.1 What You're Working From

Open `llm_prelabels.jsonl`. Each line is one incident:

```json
{
  "incident_id": "INC-01335",
  "text": "LibreChat — Vulnerability (CVE-2026-31950)\nLibreChat is a ChatGPT clone...",
  "model_votes": [
    {"model_id": "Qwen/Qwen3-235B-A22B", "entry_id": "LLM02", "confidence": 0.95, "rationale": "..."},
    {"model_id": "meta-llama/Llama-3.1-405B-Instruct-FP8", "entry_id": "LLM02", "confidence": 1.0, "rationale": "..."},
    {"model_id": "deepseek-ai/DeepSeek-V3", "entry_id": "LLM02", "confidence": 0.95, "rationale": "..."}
  ],
  "consensus": "LLM02",
  "agreement": "3-of-3",
  "triage_tier": "agree"
}
```

The file is pre-sorted: all `agree` incidents first (2,568), then `split` (2,973), then `disagree` (431).

## 1.2 The Blind-First Rule

**For every single incident, you MUST:**

1. Read the `text` field
2. Form your own label in your head (or write it down) BEFORE looking at `model_votes` or `consensus`
3. Record this as `blind_label` in your output

This prevents anchoring bias. If you find yourself peeking at the consensus first, stop and reset.

## 1.3 Tier-by-Tier Instructions

### Tier A: `agree` (2,568 incidents — do these first)

All 3 models chose the same entry. Expected time: ~30 seconds each for accepts, ~2 minutes for overrides.

**Steps for each incident:**

1. Read the `text` field — title, description, impact
2. Decide your own label (this is your `blind_label`)
3. Now look at the `consensus` field
4. **If your label matches the consensus:**
   - Record: `"adjudicated": "accept"`, `"labels": ["<consensus>"]`
5. **If your label differs from the consensus:**
   - Open `rubric.json` — read the entry definitions for both your label and the consensus label
   - Check `BOUNDARY-CASES.md` for any relevant disambiguation rules
   - Make your final call
   - Record: `"adjudicated": "override"`, `"labels": ["<your label>"]`

### Tier B: `split` (2,973 incidents — do these second)

2-of-3 models agree, 1 dissents. Expected time: ~2–3 minutes each.

**Steps for each incident:**

1. Read the `text` field — form your own label first
2. Look at all 3 entries in `model_votes` — read the majority rationale AND the dissenting rationale
3. Ask: does the dissenting model raise a valid point? Often the dissent surfaces a boundary rule the majority missed
4. Check `rubric.json` boundary rules for the entries in play
5. Make your final call:
   - `"adjudicated": "accept"` if you agree with the majority
   - `"adjudicated": "override"` if you agree with the dissent or have a different label entirely
   - `"adjudicated": "multi-label"` if the incident genuinely matches 2+ entries (use sparingly)

### Tier C: `disagree` (431 incidents — do these last)

All 3 models chose different entries, or none reached consensus. Expected time: ~3–5 minutes each.

**Steps for each incident:**

1. Read the `text` field — form your own label first
2. Skim the 3 rationales for useful signal, but don't weight them heavily — the models couldn't agree, so their reasoning is suspect
3. Apply the rubric from scratch:
   - Walk through the entry table above
   - Check `in_scope`, `exclusions`, and `boundary_rules` in `rubric.json` for candidate entries
   - If no entry fits: `"labels": []` with `"adjudicated": "override"` (out-of-scope is valid)
4. Record your decision

## 1.4 Handling Uncertainty

If you genuinely can't decide after applying the rubric:

```json
{
  "incident_id": "INC-XXXXX",
  "llm_consensus": "LLM05",
  "adjudicated": "uncertain",
  "labels": ["LLM01", "LLM05"],
  "blind_label": "LLM01",
  "notes": "injection vector present but primary impact is excessive agency"
}
```

Set these aside. After finishing all three tiers, do a second pass on uncertain cases. If still uncertain after the second pass, label with the entry whose `in_scope` definition is the closest match and add a note explaining why.

## 1.5 Output Format

Write one JSON line per incident to `adjudicated_goldset.jsonl`:

**Accept (your label matches consensus):**
```json
{"incident_id": "INC-01335", "llm_consensus": "LLM02", "adjudicated": "accept", "labels": ["LLM02"], "blind_label": "LLM02", "notes": null}
```

**Override (your label differs):**
```json
{"incident_id": "INC-07312", "llm_consensus": "LLM05", "adjudicated": "override", "labels": ["LLM01"], "blind_label": "LLM01", "notes": "Injection vector per boundary rule LLM01↔LLM05"}
```

**Multi-label:**
```json
{"incident_id": "INC-04500", "llm_consensus": "LLM06", "adjudicated": "multi-label", "labels": ["LLM06", "NEW-WLA"], "blind_label": "LLM06", "notes": "Agent with both excessive permissions and weaponized abuse vector"}
```

**Out-of-scope:**
```json
{"incident_id": "INC-02198", "llm_consensus": "LLM09", "adjudicated": "override", "labels": [], "blind_label": "out-of-scope", "notes": "Traditional supply chain issue, no LLM component"}
```

**Required fields for every line:**
- `incident_id` — copy from the prelabel record
- `llm_consensus` — copy from the prelabel record (can be `null` for disagree tier)
- `adjudicated` — one of: `accept`, `override`, `multi-label`, `uncertain`
- `labels` — list of entry IDs (empty list for out-of-scope)
- `blind_label` — the label you formed BEFORE seeing model votes
- `notes` — free text or `null`

## 1.6 Sampling Priority

You don't need to review all 5,972 incidents. Minimum target: **1,200** (30 per entry × 20 entries × 2 strata).

Prioritize by incident volume — these entries move kappa the most:

| Priority | Entry | Agree | Split | Disagree | Total |
|----------|-------|-------|-------|----------|-------|
| 1 | LLM09 | 541 | 327 | — | 868 |
| 2 | NEW-WLA | 547 | 316 | — | 863 |
| 3 | out-of-scope | 796 | 1,598 | — | 2,394 |
| 4 | LLM02 | 183 | 152 | — | 335 |
| 5 | LLM05 | 171 | 88 | — | 259 |
| 6 | LLM03 | 92 | 158 | — | 250 |
| 7 | LLM06 | 66 | 92 | — | 158 |
| 8 | NEW-MA | 22 | 93 | — | 115 |
| 9 | LLM04 | 49 | 62 | — | 111 |
| 10 | LLM01 | 28 | 18 | — | 46 |
| 11 | LLM10 | 27 | 18 | — | 45 |
| 12–20 | remaining | 46 | 69 | — | 115 |
| — | no consensus | — | — | 431 | 431 |

**For each entry, aim for 30 from agree/split and as many from disagree as are available.**

## 1.7 Session Discipline

- **Max 2 hours per session.** Fatigue causes label drift.
- **Take breaks between tiers.** Finish all agree before starting split.
- **Track your override rate.** If you're overriding >50% of agree-tier labels for a specific entry, note it — the LLM consensus is systematically biased for that entry.
- **Save often.** The interactive tool and the pipeline both resume from partial files.

---

# Phase 2: Frame 2 — Precision Verification

## 2.1 What You're Doing

For entries where the calibration pipeline has zero or very low precision data, you review incidents that the classifier assigned to that entry and answer one question: **"Is this correctly classified as [entry]?"**

This is a binary yes/no — much faster than Frame 1.

## 2.2 Entries That Need Precision Data

**Zero precision data (must do — 9 entries × 30 incidents = 270):**

| Entry | Current Precision Samples |
|-------|--------------------------|
| LLM04 — Data and Model Poisoning | 0 (frame-blind) |
| LLM06 — Excessive Agency | 0 |
| LLM08 — Vector and Embedding Weaknesses | 0 (frame-blind) |
| LLM10 — Unbounded Consumption | 0 (frame-blind) |
| NEW-MTIE — MCP Tool Interface Exploitation | 0 |
| NEW-ITSCD — Inference-Time Side-Channel Disclosure | 0 |
| ROLL-CMSB — Cross-Modal Safety Bypass | 0 |
| ROLL-LAPTF — LLM Artifact Promotion Trust Failure | 0 |
| ROLL-CFAS — Compositional Fine-tuning Alignment Subversion | 0 |

**Low precision data (should do if time permits — 7 entries):**

| Entry | Current Precision Samples |
|-------|--------------------------|
| LLM03 — Supply Chain Vulnerabilities | 9 |
| LLM07 — Hidden Context Exposure | 2 |
| NEW-PMP — Persistent Memory Poisoning | 4 |
| NEW-MA — Model Misalignment | 1 |
| NEW-WLA — Weaponized LLM Abuse | 2 |
| NEW-MSDA — Model Scheming and Deceptive Alignment | 1 |
| ROLL-SICG — Systemic Insecure Code Generation | 2 |

## 2.3 What You're Working From

Open `labeled_incidents.json`. Find incidents assigned to each target entry by filtering on the `entry_id` field. For each one, read the incident text (you'll need to look up the text in `llm_prelabels.jsonl` by matching `incident_id`).

## 2.4 Steps for Each Incident

1. Find the incident's text (look it up by `incident_id` in `llm_prelabels.jsonl`)
2. Read the incident text
3. Read the entry definition in `rubric.json` for the claimed entry
4. Answer: **Is this correctly classified as [entry]?**
   - `true` — yes, it belongs to this entry
   - `false` — no, it's misclassified

## 2.5 Output Format

Write one JSON line per incident to `precision_verification.jsonl`:

```json
{"incident_id": "INC-04821", "claimed_entry_id": "LLM06", "is_correct": true, "source": "stage2-verified", "adjudicator_id": "RL"}
```

```json
{"incident_id": "INC-03199", "claimed_entry_id": "LLM06", "is_correct": false, "source": "stage2-verified", "adjudicator_id": "RL"}
```

**Required fields for every line:**
- `incident_id` — the incident ID
- `claimed_entry_id` — the entry the classifier assigned
- `is_correct` — `true` or `false`
- `source` — always `"stage2-verified"`
- `adjudicator_id` — always `"RL"`

## 2.6 Target: 30 Per Entry

For each of the 9 zero-precision entries, review 30 incidents. For the 7 low-precision entries, review enough to bring each to ~30 total (i.e., 30 minus current sample size).

**Estimated time:** ~30 seconds per incident × 270 mandatory + ~150 optional = ~1.5–3 hours.

---

# Phase 3: Quality Checks Before Handoff

Before handing the files back, do a quick self-check:

## 3.1 Frame 1 Checks

- [ ] Every line in `adjudicated_goldset.jsonl` has all 6 required fields (`incident_id`, `llm_consensus`, `adjudicated`, `labels`, `blind_label`, `notes`)
- [ ] `labels` is always a JSON list (even for single labels: `["LLM02"]`, not `"LLM02"`)
- [ ] `labels` is an empty list `[]` for out-of-scope, never `["out-of-scope"]`
- [ ] `adjudicated` is one of: `accept`, `override`, `multi-label`, `uncertain`
- [ ] `blind_label` is filled in for every incident (this is your pre-vote label)
- [ ] No duplicate `incident_id` values
- [ ] You covered at least 30 incidents per entry for the top 10 entries by volume

## 3.2 Frame 2 Checks

- [ ] Every line in `precision_verification.jsonl` has all 5 required fields (`incident_id`, `claimed_entry_id`, `is_correct`, `source`, `adjudicator_id`)
- [ ] `is_correct` is a boolean (`true`/`false`), not a string
- [ ] `source` is `"stage2-verified"` for all lines
- [ ] You covered all 9 zero-precision entries with ~30 incidents each
- [ ] No duplicate `incident_id` values within the same `claimed_entry_id`

## 3.3 Anchoring Bias Check

After you're done, scan your Frame 1 results:
- What percentage of agree-tier incidents did you accept vs override?
- If accept rate is >95% across the board, you may have been anchored by consensus — review a random 10% sample where you accepted
- If override rate is >50% for a specific entry, note which entry — the LLM consensus is systematically wrong for it

---

# Phase 4: Handoff

Place these two files at:

```
projects/owasp-llm/cycles/2026/calibration/adjudicated_goldset.jsonl
projects/owasp-llm/cycles/2026/calibration/precision_verification.jsonl
```

Then return to a Claude Code session and say **"run B5 and B6"**. The pipeline will:

1. Feed both files into the calibration pipeline (`cal-tally` + `cal-calibrate`)
2. Re-run NUTS inference with corrected precision posteriors
3. Recompute kappa

---

# Appendix: Using the Interactive Tool (Optional)

Instead of hand-writing JSONL, you can use the built-in CLI tool:

**Frame 1:**
```bash
python -m tools.adjudicate recall \
  projects/owasp-llm/cycles/2026/calibration/llm_prelabels.jsonl \
  projects/owasp-llm/cycles/2026/calibration/adjudicated_goldset.jsonl \
  projects/owasp-llm/cycles/2026/prereg/rubric.json
```

**Frame 2:**
```bash
python -m tools.adjudicate precision \
  projects/owasp-llm/cycles/2026/classify/labeled_incidents.json \
  projects/owasp-llm/cycles/2026/calibration/precision_verification.jsonl \
  LLM04,LLM06,LLM08,LLM10,NEW-MTIE,NEW-ITSCD,ROLL-CMSB,ROLL-LAPTF,ROLL-CFAS
```

Both tools resume from where you left off — safe to quit and restart between sessions.
