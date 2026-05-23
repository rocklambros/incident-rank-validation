# Kappa Improvement: Measurement Baseline + Gold Set Expansion

## Problem Statement

Weighted Cohen's kappa between incident-derived ranks and vote-derived ranks is
0.275 with CI [-0.01, 0.57]. This is too weak to support data-driven ranking
decisions for the OWASP LLM Top 10 or Agentic Top 10.

Root causes identified:

1. **Draw cap:** `concordance.py` caps kappa computation at 500 of the 8,000
   available NUTS draws, artificially widening the CI.
2. **Calibration poverty:** Only 3 of 20 entries have "adequate" calibration
   posteriors (LLM01, LLM05, LLM09). The remaining 17 entries have posteriors
   too wide to constrain inference, inflating uncertainty in lambda estimates.
3. **Zero-recall entries:** 5 entries have zero recall true-positives
   (NEW-ITSCD, NEW-MSDA, NEW-MTIE, ROLL-CFAS, ROLL-LAPTF), meaning the
   classifier never correctly identifies them.
4. **Uninformative priors:** Unmeasured entries default to Beta(1,1) for
   precision, centering the expected value at 50% regardless of true precision.

## Architecture

Two sequential phases. Phase 1 establishes a trustworthy baseline by removing
an artificial constraint. Phase 2 attacks the calibration poverty that drives
the wide CI, via LLM-assisted gold set expansion with human adjudication.

Phase 1 output feeds into Phase 2. Phase 2 output requires Phase 1 to re-run
(NUTS + kappa recomputation with updated posteriors).

**Tech Stack:** Python, NumPyro/JAX, RunPod serverless endpoints, httpx

---

## Phase 1: Measurement Baseline

**Goal:** Remove the n_draws=500 cap and recompute kappa using all 8,000 NUTS
posterior draws, establishing the true baseline before any data improvements.

### What Changes

One line in `engine/decide/concordance.py`:

```python
# REMOVE this line (currently line 106):
n_draws = min(n_samples, 500)  # cap for speed

# REPLACE with:
n_draws = n_samples
```

The per-entry flag computation (lines 129-134) also uses `n_draws` in its loop,
so it automatically benefits from the same change.

### What Stays the Same

- NUTS inference: no re-run needed (8,000 draws already exist)
- Calibration posteriors: unchanged
- Vote posterior: unchanged
- All other decide-phase modules: unchanged

### Expected Outcome

The kappa CI will narrow because 8,000 draws produce a more stable bootstrap
distribution than 500. The median kappa may shift slightly. This gives us the
true measurement of where we stand before Phase 2 intervenes.

### Compute

Runs locally in under 2 minutes. No RunPod needed.

### Verification

```bash
incident-rank decide --cycle projects/owasp-llm/cycles/2026
cat projects/owasp-llm/cycles/2026/results/concordance.json | python -m json.tool
```

Compare old CI width vs. new CI width. The median should be similar; the CI
should be narrower.

---

## Phase 2: Gold Set Expansion + Pipeline Hardening

**Goal:** Expand the gold set from 3 adequate entries to 20 by using 3 LLM
models as pre-labelers on RunPod, with a human adjudicator as the authority on
every gold label. Then feed the gold labels into the calibration pipeline,
recompute posteriors, re-run NUTS, and recompute kappa.

Phase 2 has two tracks executed in order: **Track A** hardens the pipeline
(code changes), then **Track B** runs the gold set expansion (data work). This
ordering prevents the "stranded deliverable" failure mode where human labor
produces output the pipeline cannot consume.

### Track A: Pipeline Hardening

All code changes ship and pass tests before any data work begins.

#### A1. System/user role split in Stage-2 prompt

Split `stage2_prompt.py`'s monolithic `_SYSTEM_TEMPLATE` into two messages:

- **System message:** Classifier identity, safety rules, rubric
- **User message:** Incident text within delimiters

`build_prompt()` returns a list of two message dicts instead of a single
string. `runpod_client.py` payload changes from
`[{"role": "user", "content": prompt}]` to the two-message list.

This is a prerequisite for multi-model pre-labeling — the system/user split is
the standard interface for OpenAI-compatible endpoints on RunPod.

**Files:** `engine/classify/stage2_prompt.py`, `engine/classify/runpod_client.py`

#### A2. Stage-2 retry + fallback rate tracking

In `stage2.py`:

- Single retry on `RunPodError` with 5-second backoff before falling back to
  out-of-scope
- `fallback_count` and `total_count` counters on `Stage2Classifier`
- Abort gate: if fallback rate exceeds 10% over a rolling window of 100
  incidents, halt the batch and report

**Files:** `engine/classify/stage2.py`

#### A3. Multi-model pipeline wiring

New module `engine/classify/multi_model.py`:

```
MultiModelPreLabeler:
    models: list of (HttpRunPodClient, model_identity) tuples

    pre_label(incident) -> PreLabelResult:
        runs each model on the same incident
        collects (entry_id, confidence, rationale) per model
        computes triage_tier:
            "agree"    = all models chose same entry
            "split"    = 2-of-N agree, rest dissent
            "disagree" = all different or >=1 out-of-scope
        returns PreLabelResult with model_votes + consensus + triage_tier

    pre_label_batch(incidents) -> writes llm_prelabels.jsonl
```

Each `HttpRunPodClient` gets its own `endpoint_id` from cycle config. The
`MultiModelPreLabeler` is independent of the single-model `Stage2Classifier` —
it is a new code path for pre-labeling, not a modification of the existing
classification pipeline.

**Depends on:** A1 (needs role-split interface)
**Files:** `engine/classify/multi_model.py` (new), tests

#### A4. Gold-override integration path

Add to `engine/calibrate/tally.py`:

```python
def apply_gold_override(
    base_tally: TallyResult,
    gold_labels: list[AdjudicatedLabel],
    all_entry_ids: set[str],
) -> TallyResult:
    """Merge adjudicated gold labels into existing tally counts.

    For each gold label:
    - If adjudicated label matches classifier label: recall TP++
    - If adjudicated label differs: recall FN++ for true label,
      precision FP++ for classifier label
    - If classifier said out-of-scope but gold says entry X: recall FN++
    """
```

Add `--gold-override` flag to the `cal-tally` CLI command in
`engine/cli/pipeline.py`. The flag takes a path to
`adjudicated_goldset.jsonl` and calls `apply_gold_override()`.

**Files:** `engine/calibrate/tally.py`, `engine/cli/pipeline.py`

#### A5. Define `lambda_min` and pre-register

Add `lambda_min: float` to `PreregManifest` in `engine/prereg/manifest.py`.
Default: `prior_scale * 0.02` — an entry's posterior lambda must exceed this
to be distinguishable from noise.

The gate is applied post-inference in the decide phase, not during inference.

**Files:** `engine/prereg/manifest.py`. The consuming gate will be placed in
the decide phase during implementation planning — the manifest field and
default value are specified here.

#### A6. Empirical precision prior for unmeasured entries

After gold labels are applied and posteriors recomputed, entries that still
have no precision data get `Beta(mean_alpha, mean_beta)` derived from all
measured entries in the same stratum, replacing the default `Beta(1,1)`.

Implementation: a post-calibration step in `calibrate.py` that fills
unmeasured precision priors from the measured distribution.

**Files:** `engine/calibrate/calibrate.py`

### Track B: Gold Set Expansion

All Track B steps execute after Track A ships.

#### B1. Deploy 3 models on RunPod

Provision 3 serverless endpoints on RunPod, each running a different
open-weight model (e.g., Llama-3-70B, Mixtral-8x22B, Qwen-2-72B — final
model selection based on availability and cost at execution time).

Each endpoint gets its own `RUNPOD_ENDPOINT_ID` in cycle config.

#### B2. Run multi-model pre-labeling

Use `MultiModelPreLabeler` (A3) to classify all ~4,000 target incidents
through all 3 models.

Output: `llm_prelabels.jsonl` with per-incident `model_votes`, `consensus`,
`agreement`, and `triage_tier`.

Cost estimate: ~4,000 incidents x 3 models x $0.01/job = ~$120 RunPod spend.
Time estimate: ~2-4 hours with serverless scaling.

#### B3. Human adjudication with blind-first tool

Build `tools/adjudicate.py` — a CLI tool that:

1. Reads `llm_prelabels.jsonl`
2. Presents incident text only (model votes hidden)
3. Prompts for blind label via `input()`
4. Reveals model votes and consensus
5. Prompts for final adjudication (accept / override / multi-label / uncertain)
6. Writes `adjudicated_goldset.jsonl` with both `blind_label` and final
   `labels`

The `blind_label` field is the audit trail for anchoring detection:
- `blind_label == labels` in >95% of cases where adjudicator matches LLM
  consensus → blind-first protocol is working
- `blind_label != labels` but `labels == consensus` frequently → anchoring
  bias detected

Workflow follows `docs/GOLDSET-LLM-ADJUDICATION-GUIDE.md`.

Minimum viable target: 1,200 incidents (30 per entry x 20 entries x 2 strata)
at ~2 min average = ~40 hours.

#### B4. Feed gold labels into calibration pipeline

```bash
incident-rank cal-tally \
  --cycle projects/owasp-llm/cycles/2026 \
  --gold-override projects/owasp-llm/cycles/2026/calibration/adjudicated_goldset.jsonl

incident-rank cal-calibrate \
  --cycle projects/owasp-llm/cycles/2026

cat projects/owasp-llm/cycles/2026/calibration/diagnostic.json | python -m json.tool
```

#### B5. Re-run Phase 1 (NUTS + kappa)

After gold labels update the posteriors, re-run NUTS inference with the new
calibration, then recompute kappa with the full 8,000 draws.

```bash
incident-rank infer --cycle projects/owasp-llm/cycles/2026
incident-rank decide --cycle projects/owasp-llm/cycles/2026
```

### Dependency Chain

```
A1 (role split) --> A3 (multi-model) --> B1 (deploy) --> B2 (pre-label)
A2 (retry)     --/                                          |
                                                        B3 (adjudicate)
A4 (gold-override) ----------------------------------------> B4 (feed gold)
A5 (lambda_min) --------------------------------------------> B5 (re-run)
A6 (empirical prior) --------------------------------------> B5 (re-run)
```

Track A parallelism: A1+A2 together, A4+A5+A6 together. A3 depends on A1.
Track B is sequential: B1 -> B2 -> B3 -> B4 -> B5.

### Quality Metrics (computed after B5)

1. **Override rate by tier and by entry** — which entries did LLMs get wrong
   most often?
2. **Krippendorff's alpha (LLM consensus vs. human labels)** — how well do
   LLMs approximate human judgment?
3. **Anchoring audit:** compare `blind_label` agreement with LLM consensus vs.
   final `labels` agreement. If final agreement is significantly higher than
   blind agreement, anchoring bias is present.
4. **Calibration adequacy improvement:** how many entries moved from "wide" or
   "no-data" to "adequate"?
5. **Kappa change:** compare pre-gold-set kappa CI with post-gold-set kappa CI.
6. **Signal-to-noise gate:** how many entries pass `lambda_min`?

### What This Does Not Fix

- **Kappa may not improve.** If gold labels confirm the classifier is wrong on
  many entries, posteriors tighten but ranks may shift in ways that increase
  disagreement with vote-derived ranks. Phase 2 gives a trustworthy kappa, not
  necessarily a higher one.
- **40+ hours of human adjudication.** No shortcut. LLM-only gold labels
  produce correlated, inflated calibration (see premortem on Approach A,
  2026-05-23).
- **3 frame-blind entries remain unmeasurable.** LLM04, LLM08, LLM10 have no
  classifier rules by design. Gold labels contribute to recall calibration but
  not to the kappa computation.

---

## Premortem Findings Addressed

| ID | Finding | Remediation | Phase 2 Step |
|----|---------|-------------|-------------|
| R1 | `--gold-override` integration path missing | Add `apply_gold_override()` + CLI flag | A4 |
| R2 | System/user role split missing in prompt | Split `_SYSTEM_TEMPLATE` into two messages | A1 |
| R3 | Blind-first anchoring unenforceable | Build `tools/adjudicate.py` with blind-first flow | B3 |
| R4 | Single-endpoint architecture for 3 models | `MultiModelPreLabeler` with per-model clients | A3 |
| R5 | `lambda_min` undefined | Add to `PreregManifest` with default | A5 |
| R6 | No retry, no fallback tracking | Single retry + fallback rate gate | A2 |
| R7 | Beta(1,1) prior for unmeasured entries | Empirical prior from measured distribution | A6 |

## Residual Risks

1. **Informativeness floor may exclude entries with genuinely zero recall.**
   Accepted: entries absent from the corpus should be excluded. Monitor how
   many entries pass the floor after gold expansion.

2. **Blind-first anchoring partially unenforceable.** Even with the tool, the
   adjudicator could pre-scan the JSONL file. The `blind_label` audit trail
   detects this after the fact.

3. **Single adjudicator bias.** Mitigated by Krippendorff's alpha and the
   `blind_label` audit trail. If alpha is suspiciously high (>0.95),
   investigate anchoring.

4. **Unmeasured entries after Phase 2.** R7 provides a partial fix. If fewer
   than 3 entries remain unmeasured, the empirical prior is based on a small
   sample.
