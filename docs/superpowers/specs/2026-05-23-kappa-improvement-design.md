# Kappa Improvement: Measurement Baseline + Gold Set Expansion

## Problem Statement

Weighted Cohen's kappa between incident-derived ranks and vote-derived ranks is
0.275 with CI [-0.01, 0.57]. This is too weak to support data-driven ranking
decisions for the OWASP LLM Top 10 or Agentic Top 10.

Root causes identified:

1. **Draw cap:** `concordance.py` caps kappa computation at 500 of the 8,000
   available NUTS draws, artificially widening the CI.
2. **Vote draw recycling:** When NUTS draws (8,000) exceed vote bootstrap
   samples (5,000), `concordance.py` line 111 uses modular indexing
   (`s % len(vote_posterior.rank_samples)`), recycling 37.5% of vote draws.
   This creates spuriously narrow CI on the vote side.
3. **Calibration poverty:** Only 3 of 20 entries have "adequate" calibration
   posteriors (LLM01, LLM05, LLM09). The remaining 17 entries have posteriors
   too wide to constrain inference, inflating uncertainty in lambda estimates.
4. **Precision poverty (dominant):** 6 entries have zero precision data
   (LLM06, NEW-MTIE, NEW-ITSCD, ROLL-CMSB, ROLL-LAPTF, ROLL-CFAS). Their
   precision posteriors default to Beta(1,1), centering expected precision at
   50% regardless of true precision. This is the **primary driver** of kappa
   CI width — wider than the recall poverty contribution.
5. **Zero-recall entries:** 5 entries have zero recall true-positives
   (NEW-ITSCD, NEW-MSDA, NEW-MTIE, ROLL-CFAS, ROLL-LAPTF), meaning the
   classifier never correctly identifies them.
6. **Uninformative priors:** Unmeasured entries default to Beta(1,1) for
   precision, centering the expected value at 50% regardless of true precision.

## Architecture

Two sequential phases. Phase 1 establishes a trustworthy baseline by removing
artificial constraints and fixing measurement bugs. Phase 2 attacks the
calibration poverty that drives the wide CI, via Two-Frame Gold Calibration
with LLM-assisted pre-labeling and human adjudication.

Phase 1 output feeds into Phase 2. Phase 2 output requires Phase 1 to re-run
(NUTS + kappa recomputation with updated posteriors).

**Tech Stack:** Python, NumPyro/JAX, RunPod serverless endpoints, httpx

---

## Phase 1: Measurement Baseline

**Goal:** Remove the n_draws=500 cap, fix vote draw recycling, and recompute
kappa using all 8,000 NUTS posterior draws, establishing the true baseline
before any data improvements.

### What Changes

#### 1a. Remove draw cap

One line in `engine/decide/concordance.py`:

```python
# REMOVE this line (currently line 106):
n_draws = min(n_samples, 500)  # cap for speed

# REPLACE with:
n_draws = n_samples
```

The per-entry flag computation (lines 129-134) also uses `n_draws` in its loop,
so it automatically benefits from the same change.

#### 1b. Fix vote draw recycling

In `engine/decide/concordance.py` line 111:

```python
# CURRENT (recycling bug):
vote_draw = vote_posterior.rank_samples[s % len(vote_posterior.rank_samples)]

# REPLACE with:
n_common = min(n_draws, len(vote_posterior.rank_samples))
```

Use `n_common` as the loop bound instead of `n_draws`. Both the kappa
bootstrap loop and the per-entry flag loop must use this shared bound.
This ensures each vote draw is used exactly once, eliminating the spurious
correlation from recycled draws.

### What Stays the Same

- NUTS inference: no re-run needed (8,000 draws already exist)
- Calibration posteriors: unchanged
- Vote posterior: unchanged
- All other decide-phase modules: unchanged

### Expected Outcome

The kappa CI will narrow because 8,000 draws (capped to ~5,000 by common
bound) produce a more stable bootstrap distribution than 500. The median
kappa may shift slightly. The vote-side CI contribution will widen slightly
(no longer artificially narrowed by recycling), giving a more honest estimate.

### Compute

Runs locally in under 2 minutes. No RunPod needed.

### Verification

```bash
incident-rank decide --cycle projects/owasp-llm/cycles/2026
cat projects/owasp-llm/cycles/2026/results/concordance.json | python -m json.tool
```

Compare old CI width vs. new CI width. The median should be similar; the CI
should be narrower overall but more honest on the vote side.

---

## Phase 2: Two-Frame Gold Calibration + Pipeline Hardening

**Goal:** Attack both recall AND precision poverty simultaneously using a
two-frame gold calibration approach. Frame 1 (recall) identifies what the
classifier misses. Frame 2 (precision) verifies what the classifier claims.
Together they expand calibration from 3 adequate entries to 20.

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

    pre_label_batch(incidents, checkpoint_path) -> writes llm_prelabels.jsonl
        Checkpoint/resume: writes each result as it completes.
        On restart, reads existing checkpoint, skips already-processed
        incident IDs, and appends new results.
```

Each `HttpRunPodClient` gets its own `endpoint_id` from cycle config. The
`MultiModelPreLabeler` is independent of the single-model `Stage2Classifier` —
it is a new code path for pre-labeling, not a modification of the existing
classification pipeline.

**Depends on:** A1 (needs role-split interface)
**Files:** `engine/classify/multi_model.py` (new), tests

#### A4. Two-Frame Gold Calibration

Replace the original single-function `apply_gold_override()` with a two-frame
calibration system that attacks both recall and precision poverty.

**Why two frames:** The original gold-override improved recall only. But
precision poverty is the **dominant** source of kappa CI width — 6 entries
have zero precision data, meaning their Beta(1,1) priors center expected
precision at 50% regardless of truth. A recall-only gold set leaves this
untouched.

##### Frame 1: Gold Recall

Gold recall labels answer: "For this incident, which entry(ies) does it
actually belong to?" This is the same question the original gold-override
addressed. Each gold recall label produces:

- **Recall TP** for the true entry if the classifier also found it
- **Recall FN** for the true entry if the classifier missed it
- **Precision FP** for the classifier's entry if the classifier was wrong

##### Frame 2: Gold Precision

Gold precision labels answer: "For this classifier output claiming entry X,
is the claim correct?" This is new. The adjudicator reviews incidents that
the classifier (or LLM pre-labeler) assigned to each entry and verifies
whether the assignment is correct. Each gold precision label produces:

- **Precision TP** if the classifier's claim is correct
- **Precision FP** if the classifier's claim is wrong

Frame 2 applies to BOTH Stage-2 classifier labels AND LLM pre-labels, so any
entry that at least one LLM detects gets precision data — even entries the
original Stage-2 classifier never assigned.

##### Schema

```python
@dataclass(frozen=True, slots=True)
class GoldRecallLabel:
    incident_id: str
    true_entry_ids: list[str]   # multi-label: each label independent
    classifier_entry_id: str | None
    source: str                 # "manual-curated" | "llm-adjudicated"

@dataclass(frozen=True, slots=True)
class GoldPrecisionLabel:
    incident_id: str
    claimed_entry_id: str       # what classifier/LLM said
    is_correct: bool            # adjudicator verdict
    source: str                 # "stage2-verified" | "llm-prelabel-verified"

@dataclass(frozen=True, slots=True)
class GoldCalibration:
    recall_labels: list[GoldRecallLabel]
    precision_labels: list[GoldPrecisionLabel]
    provenance_hash: str        # SHA-256 of input file
    rubric_hash: str            # must match frozen rubric hash
    adjudicator_id: str
    session_count: int
```

##### Integration

Add to `engine/calibrate/tally.py`:

```python
def calibrate_with_gold(
    base_tally: TallyResult,
    gold: GoldCalibration,
    base_incident_ids: set[str],
    all_entry_ids: set[str],
) -> TallyResult:
    """Merge gold labels into existing tally as separate strata.

    1. Deduplicate: skip any gold incident already in base_incident_ids
    2. Frame 1 (recall): for each GoldRecallLabel:
       - For each true_entry_id: recall TP++ if classifier matched, FN++ if not
       - If classifier was wrong: precision FP++ for classifier's entry
       - Stratum: "gold-recall"
    3. Frame 2 (precision): for each GoldPrecisionLabel:
       - Precision TP++ if is_correct, FP++ if not
       - Stratum: "gold-precision"
    4. Multi-label: each label in true_entry_ids is independent
    5. Gold precision data is stratum-independent: precision is a property
       of the classifier, not the incident's origin stratum. Gold-precision
       counts apply to the entry globally, not per-stratum.
    6. Recall denominators are only charged against entries present in each
       stratum — do not charge against all entries uniformly.
    7. Return combined tally with gold strata appended
    """
```

Add `--gold-calibration` flag (not `--gold-override`) to the `cal-tally` CLI
command in `engine/cli/pipeline.py`. The flag takes a path to
`adjudicated_goldset.jsonl` and calls `calibrate_with_gold()`.

**Schema validation:** The function validates that `gold.rubric_hash` matches
the frozen rubric hash before proceeding. Mismatched hashes halt with a clear
error.

**Files:** `engine/calibrate/tally.py`, `engine/cli/pipeline.py`

#### A5. Manual curation input path

Add a loader for manually curated incidents at
`projects/owasp-llm/cycles/2026/calibration/manual_curated_incidents.json`.

The loader:
1. Reads the JSON file and validates against `IncidentRecord` schema
2. Deduplicates against existing corpus A incident IDs
3. Derives `true_entry_ids` using this priority:
   a. If `native_labels` is non-empty, use it directly
   b. If `native_labels` is empty (all 42 current incidents), parse the entry
      ID from the incident ID prefix: `MANUAL-LLM06-001` → `LLM06`,
      `MANUAL-ROLL-CFAS-001` → `ROLL-CFAS`. The prefix convention is
      `MANUAL-{ENTRY_ID}-{NNN}`.
   c. If neither source yields an entry ID, reject the record with an error.
4. Converts each curated incident into `GoldRecallLabel` records
   (source="manual-curated") using the derived `true_entry_ids`
5. Validates that all derived entry IDs reference valid entry IDs in the
   frozen rubric

This provides a first-class path for human-curated incidents to enter the
gold calibration pipeline without going through LLM pre-labeling. The 42
incidents already committed at
`projects/owasp-llm/cycles/2026/calibration/manual_curated_incidents.json`
are the initial seed for the 6 zero-precision entries.

The loader also reads Frame 2 output
(`precision_verification.jsonl`) and populates
`GoldCalibration.precision_labels`. Both files feed through the same
`--gold-calibration` flag on `cal-tally`. The flag accepts either:
- A single JSONL file (Frame 1 only — recall labels)
- A directory containing `adjudicated_goldset.jsonl` and/or
  `precision_verification.jsonl` (both frames)

When a directory is provided, the loader reads both files if present and
merges them into a single `GoldCalibration` object.

**Files:** `engine/calibrate/gold_loader.py` (new)

#### A5b. Fix ESS gate denominator in inference.py

`inference.py` lines 273-279 compute ESS ratio as
`min(v / num_samples for v in ess_dict.values())` where `num_samples` is
the per-chain count (2000). The correct denominator is
`num_samples * num_chains` (total draws = 8000). The current gate is 4x too
loose — it accepts ESS ratios that would fail with the correct base.

```python
# CURRENT (bug):
ess_ratio = min(v / num_samples for v in ess_dict.values())

# REPLACE with:
total_draws = num_samples * num_chains
ess_ratio = min(v / total_draws for v in ess_dict.values())
```

The threshold stays the same; only the denominator changes.

**Files:** `engine/model/inference.py`

#### A6. Define `lambda_min` and pre-register

Add `lambda_min: float` to `PreregManifest` in `engine/prereg/manifest.py`.
Default: `prior_scale * 0.02` — an entry's posterior lambda must exceed this
to be distinguishable from noise.

The gate is applied post-inference in the decide phase, not during inference.

**Files:** `engine/prereg/manifest.py`. The consuming gate will be placed in
the decide phase during implementation planning — the manifest field and
default value are specified here.

#### A7. Empirical precision prior for unmeasured entries

After gold labels are applied and posteriors recomputed, entries that still
have no precision data get `Beta(mean_alpha, mean_beta)` derived from all
measured entries in the same stratum, replacing the default `Beta(1,1)`.

Implementation: a post-calibration step in `calibrate.py` that fills
unmeasured precision priors from the measured distribution.

**Files:** `engine/calibrate/calibrate.py`

### Track B: Gold Set Expansion

All Track B steps execute after Track A ships.

#### B1. Ingest manual curation

Load the 42 manually curated incidents (committed at
`projects/owasp-llm/cycles/2026/calibration/manual_curated_incidents.json`)
through the A5 loader. These become the first gold recall labels for the 6
zero-precision entries:

| Entry | Count | Coverage |
|-------|-------|----------|
| LLM06 | 7 | Excessive Agency |
| NEW-MTIE | 9 | Multi-Tenant Isolation Erosion |
| NEW-ITSCD | 9 | Insufficient Training/Serving Configuration and Defaults |
| ROLL-CMSB | 10 | Cross-Modal Safety Bypass |
| ROLL-LAPTF | 5 | Lack of Adversarial Prompt Testing Frameworks |
| ROLL-CFAS | 2 | Cascading Failures in Agentic Systems |

**Data quality notes for implementation:**

- **ROLL-LAPTF (5 incidents):** All 5 are model supply chain attacks
  (pickle backdoors, namespace hijacking). ROLL-LAPTF has
  `rolled_into: "LLM03"` — the classifier architecturally routes all LAPTF
  detections to LLM03 (Supply Chain Vulnerabilities). These incidents are
  conceptually LAPTF but will be classified as LLM03 in practice. The gold
  loader should use the curated label (ROLL-LAPTF), not the classifier
  routing target. This tests whether the classifier correctly routes LAPTF
  concepts to LLM03.
- **ROLL-CFAS (2 incidents):** Both describe compositional fine-tuning
  alignment subversion (CoLoRA, MergeBackdoor), not cascading failures in
  agentic systems. These are adjacent but not on-target for CFAS. During
  B3 (LLM pre-labeling), additional incidents that describe actual
  multi-agent cascading failures should be prioritized for CFAS to
  supplement these 2. Minimum target: 5 CFAS incidents total.

These 42 incidents provide recall calibration immediately. Precision
calibration for these entries comes from Frame 2 (B4).

#### B2. Deploy 3 models on RunPod

Provision 3 serverless endpoints on RunPod, each running a different
open-weight model (e.g., Llama-3-70B, Mixtral-8x22B, Qwen-2-72B — final
model selection based on availability and cost at execution time).

Each endpoint gets its own `RUNPOD_ENDPOINT_ID` in cycle config.

#### B3. Run multi-model pre-labeling

Use `MultiModelPreLabeler` (A3) to classify all ~4,000 target incidents
through all 3 models. Checkpoint/resume ensures partial runs are recoverable.

Output: `llm_prelabels.jsonl` with per-incident `model_votes`, `consensus`,
`agreement`, and `triage_tier`.

Cost estimate: ~4,000 incidents x 3 models x $0.01/job = ~$120 RunPod spend.
Time estimate: ~2-4 hours with serverless scaling.

#### B4. Human adjudication with two-frame tool

Build `tools/adjudicate.py` — a CLI tool with two modes:

**Mode 1: Recall adjudication (Frame 1)**

1. Reads `llm_prelabels.jsonl`
2. Presents incident text only (model votes hidden)
3. Prompts for blind label via `input()`
4. Reveals model votes and consensus
5. Prompts for final adjudication (accept / override / multi-label / uncertain)
6. Writes `adjudicated_goldset.jsonl` with both `blind_label` and final
   `labels`

**Mode 2: Precision verification (Frame 2)**

1. Reads classifier output (`classifications.json`) and/or
   `llm_prelabels.jsonl`
2. For each entry with low/zero precision data, samples incidents that the
   classifier or any LLM assigned to that entry
3. Presents: incident text + claimed entry
4. Prompts: "Is this correctly classified as [entry]?" (yes/no)
5. Writes `precision_verification.jsonl` with `GoldPrecisionLabel` records

Each line in `precision_verification.jsonl`:

```json
{
  "incident_id": "GA-04821",
  "claimed_entry_id": "LLM06",
  "is_correct": true,
  "source": "stage2-verified",
  "adjudicator_id": "RL",
  "session_timestamp": "2026-06-15T14:30:00Z"
}
```

Frame 2 is fast (~30 seconds per incident) because the adjudicator only
needs to answer a binary question, not determine the correct label from
scratch. For 6 zero-precision entries × ~30 verified per entry = ~180
incidents at ~30s each = **~1.5 hours**.

The `blind_label` field in Frame 1 is the audit trail for anchoring detection:
- `blind_label == labels` in >95% of cases where adjudicator matches LLM
  consensus → blind-first protocol is working
- `blind_label != labels` but `labels == consensus` frequently → anchoring
  bias detected

Workflow follows `docs/GOLDSET-LLM-ADJUDICATION-GUIDE.md`.

Minimum viable target for Frame 1: 1,200 incidents (30 per entry x 20 entries
x 2 strata) at ~2 min average = ~40 hours.

#### B5. Feed gold labels into calibration pipeline

```bash
# Feed both frames into calibration
incident-rank cal-tally \
  --cycle projects/owasp-llm/cycles/2026 \
  --gold-calibration projects/owasp-llm/cycles/2026/calibration/adjudicated_goldset.jsonl

incident-rank cal-calibrate \
  --cycle projects/owasp-llm/cycles/2026

cat projects/owasp-llm/cycles/2026/calibration/diagnostic.json | python -m json.tool
```

#### B6. Re-run Phase 1 (NUTS + kappa)

After gold labels update the posteriors, re-run NUTS inference with the new
calibration, then recompute kappa with the full draws and recycling fix.

```bash
# 4 chains x 2000 samples = 8,000 total draws.
# If chain count changes, adjust --num-samples to maintain >= 8,000 total.
incident-rank infer --cycle projects/owasp-llm/cycles/2026 --num-samples 2000
incident-rank decide --cycle projects/owasp-llm/cycles/2026
```

### Dependency Chain

```
A1 (role split) --> A3 (multi-model) --> B2 (deploy) --> B3 (pre-label)
A2 (retry)     --/                                          |
                                                        B4 (adjudicate)
A4 (two-frame gold) ----------------------------------------> B5 (feed gold)
A5 (manual curation) --> B1 (ingest manual) -----------------> B5
A6 (lambda_min) --------------------------------------------> B6 (re-run)
A7 (empirical prior) --------------------------------------> B6 (re-run)
```

Track A parallelism: A1+A2 together, A4+A5+A6+A7 together. A3 depends on A1.
Track B is sequential: B1 -> B2 -> B3 -> B4 -> B5 -> B6.

### Kappa Decision Framework

After B6, the recomputed kappa falls into one of three regimes:

| Kappa range | Interpretation | Action |
|-------------|---------------|--------|
| ≥ 0.40 with CI excluding 0 | Fair-to-moderate agreement | Incident-derived ranks contribute to final ranking with disclosed weight |
| 0.20–0.40 or CI includes 0 | Weak agreement | Incident-derived ranks published as context only, not as ranking input |
| < 0.20 | Poor agreement | Incident-derived ranks do not inform ranking; investigate structural causes |

This framework is pre-registered before B6 runs so the interpretation is not
chosen after seeing the result.

### Quality Metrics (computed after B6)

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
7. **Precision coverage:** how many of the 6 formerly zero-precision entries
   now have precision data? Target: all 6.
8. **Frame 2 precision TP rate:** what fraction of classifier claims were
   confirmed correct? Low rates indicate systematic classifier bias.
9. **Inter-rater reliability (aspirational):** if a second adjudicator labels
   a random 10% subsample (~120 incidents), compute Krippendorff's alpha
   between adjudicators. The minimum viable path uses a single adjudicator
   with the `blind_label` audit trail as the anchoring check.

### What This Does Not Fix

- **Kappa may not improve.** If gold labels confirm the classifier is wrong on
  many entries, posteriors tighten but ranks may shift in ways that increase
  disagreement with vote-derived ranks. Phase 2 gives a trustworthy kappa, not
  necessarily a higher one.
- **40+ hours of human adjudication for Frame 1.** No shortcut. LLM-only gold
  labels produce correlated, inflated calibration (see premortem on Approach A,
  2026-05-23).
- **~1.5 hours of precision verification for Frame 2.** This is new work but
  fast (binary yes/no per incident).
- **3 frame-blind entries remain unmeasurable.** LLM04, LLM08, LLM10 have no
  classifier rules by design. Gold labels contribute to recall calibration but
  not to the kappa computation.

---

## Premortem Findings Addressed

| ID | Finding | Remediation | Phase 2 Step |
|----|---------|-------------|-------------|
| R1 | Vote draw recycling inflates CI | Fix modular indexing, use common bound | Phase 1 (1b) |
| R2 | System/user role split missing in prompt | Split `_SYSTEM_TEMPLATE` into two messages | A1 |
| R3 | Blind-first anchoring unenforceable | Build `tools/adjudicate.py` with blind-first flow | B4 |
| R4 | Single-endpoint architecture for 3 models | `MultiModelPreLabeler` with per-model clients | A3 |
| R5 | `lambda_min` undefined | Add to `PreregManifest` with default | A6 |
| R6 | No retry, no fallback tracking | Single retry + fallback rate gate | A2 |
| R7 | Beta(1,1) prior for unmeasured entries | Empirical prior from measured distribution | A7 |
| R8 | Gold-override improves recall only, leaves precision poverty untouched | Two-Frame Gold Calibration with Frame 2 precision verification | A4 |
| R9 | No checkpoint/resume for pre-labeling batch | Checkpoint file in `pre_label_batch()` | A3 |
| R10 | No kappa decision framework pre-registered | Three-regime interpretation table | Kappa Decision Framework |
| R11 | Zero-precision entries have no input path for manual curation | Manual curation loader + 42 committed incidents | A5, B1 |
| R12 | Empty native_labels on all 42 curated incidents (F1.1) | Loader derives entry ID from incident ID prefix when native_labels is empty | A5 |
| R13 | ESS gate divides by per-chain count, 4x too loose (F2.1) | Fix denominator to total_draws = num_samples × num_chains | A5b |
| R14 | Frame 2 output has no CLI ingestion path (F4.2) | gold_loader.py reads precision_verification.jsonl; --gold-calibration accepts directory | A5 |
| R15 | Frame 2 JSONL schema unspecified (F1.2) | Explicit JSON example in B4 | B4 |
| R16 | Precision calibration stratum-dependent (F1.6) | Gold precision is stratum-independent; documented in A4 | A4 |
| R17 | Recall denominator inflated (F1.8) | Per-stratum recall denomination documented in A4 | A4 |
| R18 | B6 chain-count assumption undocumented (F4.1) | Explicit --num-samples with chain annotation in B6 | B6 |
| R19 | No inter-rater reliability baseline (F1.7) | Aspirational 10% subsample protocol in Quality Metrics | Quality Metrics |

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

5. **Manual curation sample size.** 42 incidents across 6 entries (2-10 per
   entry) is a floor, not a ceiling. ROLL-CFAS has only 2 incidents — its
   posterior will remain wide. The LLM pre-labeling in B3 will surface
   additional incidents for these entries.

6. **Frame 2 precision verification depends on classifier/LLM coverage.**
   Entries that neither the classifier nor any LLM ever assigns still get
   zero precision data. This risk is mitigated by using 3 diverse LLMs —
   the probability that all 3 miss an entry that exists in the corpus is low.
