# Design Spec — Recall-Aware Robustness Re-analysis (RARR)

- **Status:** DRAFT (pre-implementation; pending user review + adversarial premortem of this spec)
- **Date:** 2026-06-22
- **Branch:** `plan7/engine-upgrade-recall-pl`
- **Provisional phase label:** *Plan 8 — Recall-Aware Robustness Re-analysis* (the PRD already assigns "Plan 7" to the frame-coverage audit; the final number is a project-owner decision — see §11)
- **Source prompt:** `claudedocs/engine-upgrade-runpod.md`
- **Supersedes the informal recommendations** revised by the two-round adversarial premortem of 2026-06-22 (14 remediations RM1–RM14, traced in §13).

---

## 1. Context

`incident-rank-validation` validates the OWASP LLM Top-10 community vote against a corpus of ~7,700 real incidents. The frozen 2026 cycle reported a quadratic-weighted Cohen's kappa of **0.20 (CI [-0.16, 0.57], 17 of 20 entries measurable)** — inconclusive. A COMP-4442 re-analysis concluded: (1) the engine's core call was sound — "inconclusive by kappa" is the correct hedge; **do not overturn it, explain it better**; (2) a single kappa hides structure and independent per-category rates wobble for sparse categories; (3) classifier recall is low and uneven, so the observed counts are a biased undercount.

This spec implements the upgrade **as a correctness-and-robustness exercise that preserves the original conclusion's primacy**, not a headline swap. It is the product of a two-round, six-perspective adversarial premortem that found the naive version of this work would have rested on a broken calibration foundation, locked a pre-registration manifest that misdescribes the model actually run, swapped the project's research question, and opened a wide researcher-degrees-of-freedom surface. Every design choice below is shaped by closing those gaps.

### 1.1 What the engine already is (verified against code)
- **Bayesian stack:** NumPyro on JAX, **CPU-hard-asserted** for reproducibility (`engine/model/inference.py:121`; `pyproject.toml` `JAX_PLATFORM_NAME=cpu`).
- **Measurement-error likelihood already exists** (`inference.py:172-192`): `true_rate = λ·size`; `tp = true_rate·recall`; false positives via overlap leakage `W·true_rate·(1-precision)`; `NegativeBinomial2` likelihood. The latent `λ` is recall-corrected incidence; `concordance.py:53-63` ranks the vote against `λ`.
- **Classifier:** Stage-1 deterministic indicator match (667 incidents) → Stage-2 LLM adjudication for the rest (5,972), runnable as a 3-model RunPod consensus (`engine/cli/reclassify.py`).
- **Prereg discipline:** frozen `PreregManifest`, SHA-256 lock (`engine/prereg/{manifest.py,lock.py}`); Merkle-chained `post_hoc_register` and external-reviewer attestations are *defined* in HANDOFF §6 but **not instantiated** for the 2026 cycle (`statistical_reviewer:null`, `post_hoc_register_path:null`).

### 1.2 Premortem findings that reshaped this spec
The naive recommendations failed adversarial review on four root causes, all of which this spec must fix **before** producing any new number:
1. **Broken calibration foundation** — the recall tally uses a frame-size denominator and counts classifier emissions rather than recovered truth, manufacturing falsely-precise near-zero recall (`Beta(1,101)`) for the sparse entries the upgrade targets (`tally.py:99-116`).
2. **A manifest that would lie** — `primary_spec` dispatches no model (`pipeline.py:581-582` compares it to itself); locking `primary_spec=hierarchical` would assert a model the engine never runs.
3. **Question substitution + outcome-switching** — Plackett-Luce summarizes the *vote*; making it the headline abandons the vote-vs-*data* concordance question and switches the headline statistic after the kappa result is known.
4. **Illusory verification + open gaming surface** — a same-author oracle with no agreement tolerance, plus unregistered σ_u prior / tie rule / bootstrap seed / OOS definition / bake-off best-of-N, which jointly could manufacture the prompt's pre-stated conclusion.

---

## 2. Research question (unchanged) and framing decision

**The research question stays exactly what it was:** *Does the OWASP LLM Top-10 community vote ranking concord with the incident-derived prevalence ranking?* The headline statistic stays the **quadratic-weighted Cohen's kappa over the measurable subset** (the engine's `primary_spec`).

**Framing decision (RM1, the load-bearing structural choice):** Hierarchical partial pooling, the recall-corrected ranking, and Plackett-Luce enter as **declared robustness specifications under the unchanged kappa primary** — per the engine's own HANDOFF §6 control 4 ("one pre-registered primary specification; everything else is a declared robustness spec"). The report presents them as a **spread around the primary**, never as a replacement headline. This single decision:
- keeps the COMP-4442 verdict intact ("explain it better," not "overturn");
- removes outcome-switching (the headline statistic does not change);
- collapses the "single move" gaming chain (you cannot manufacture a new headline if the headline does not move).

Plackett-Luce is retained because it is a genuinely better *vote-side* summary and a better input to the vote-vs-data comparison — but it answers the vote question, not the concordance question, and is labeled as such.

---

## 3. Goals & non-goals

**Goals**
- G1. **Fix the recall calibration** so per-entry recall/precision posteriors are honest (wide where data is sparse), then rebuild the chain on a single, reproducible classifier.
- G2. **Improve and re-measure recall** via a pre-registered RunPod classifier bake-off, because recall is the only lever that raises `measurable_count` and can tighten the concordance CI (RM3).
- G3. Add **hierarchical partial pooling**, the **raw-vs-recall-corrected ranking**, and a **tie-aware Plackett-Luce** vote model as **robustness specs**, each with mandatory sensitivity analysis.
- G4. Provide an **independent Python consistency-check** (no R) gated by a tamper-evident, non-author-signed mechanism.
- G5. Do all of it under a **locked-before-numbers pre-registration** that leaves the original 2026 cycle byte-immutable and discloses the post-hoc nature.

**Non-goals**
- N1. Changing the headline statistic or the research question (explicitly forbidden — §2).
- N2. Overturning the kappa=0.20 conclusion. The corrected analysis may confirm, refine, or widen it; it is reported, not engineered.
- N3. Running any Bayesian fit on GPU (forbidden by `inference.py:121` / HANDOFF §7.5).
- N4. Adding heavy new dependencies (PyMC) unless §10's supply-chain gate is satisfied; prefer the already-pinned `scipy`/`numpy`.

---

## 4. Compute policy — RunPod always for heavy work

**Standing directive: all training and heavy inference run on RunPod; the local Jetson is never used for heavy compute.** Reconciled with the CPU-determinism invariant by treating **RunPod as the *location* and CPU as the *backend*** where determinism requires it:

| Workload | RunPod pod | Backend | Why |
|---|---|---|---|
| Track A classifier bake-off (LLM labeling, ~7,700 × models × configs) | **GPU** (H200, vLLM) | GPU | LLM inference; pinned image + **pinned HF model revisions** |
| Track B/C hierarchical NUTS + measurement-error correction + σ_u sensitivity sweeps | **CPU** (high-vCPU) | **CPU** (JAX, X64, seeded) | Heavy (specs × priors) but must be CPU for bit-reproducibility (`inference.py:121`) |
| Track D tie-aware Plackett-Luce + ≥1000 respondent bootstrap | **CPU** (high-vCPU) | CPU | Embarrassingly parallel; determinism preserved |
| Independent verification oracle | **separate CPU pod** | CPU | Isolated, independently-pinned environment |

Linux x86_64 RunPod CPU pods match the CI `ubuntu-latest` determinism reference, so the cross-platform parity gate continues to hold. **Provenance:** record pod type, image digest, HF model revisions, engine commit, seeds, and run logs for every pod; tear pods down on completion (RM4, RM13).

---

## 5. Architecture & components

The work is a sequence of engine modules, each with one purpose and a versioned artifact contract. New/changed modules are marked.

### 5.1 Foundation — recall calibration fix `[CHANGE: engine/calibrate/tally.py]` (RM2)
**Problem:** the recall branch increments `recall_hits` on classifier emissions (no adjudication) and sets every entry's denominator to the full recall-frame size, so a 1-true-positive entry becomes `Beta(1,101)` — confident ≈1% recall.
**Design:** recompute recall per entry against **adjudicated truth in the recall frame**:
- `recall_X = (# recall-frame incidents truly X that the classifier labeled X) / (# recall-frame incidents truly X)`.
- Denominator is the **truth cell** for X, not the frame size. Numerator requires comparing the classifier label to the adjudicated `labels[]`.
- Beta posterior `Beta(1 + TP, 1 + FN)` with the corrected counts → sparse entries become **wide**, not falsely precise.
- Add a proof/unit test exercising `tally_batches` recall path (currently untested).
**Contract:** `posteriors.json` schema unchanged; values corrected. Calibration provenance now hashes the classifier label file (§5.7).

### 5.2 Track A — classifier bake-off `[NEW: engine/classify/bakeoff.py + engine/cli/bakeoff.py]` (RM3, RM4)
RunPod-GPU re-classification to raise and even out recall, selected by a **pre-registered** procedure:
- **Config grid** (committed to the manifest before any RunPod run): model set (e.g. {Qwen3-235B, Llama-3.1-405B, DeepSeek-V3, + one candidate 4th}), consensus rule, prompt variant (OOS-calibrated), and Stage-1/Stage-2 thresholds. The grid and its size *N* are frozen up front.
- **Selection metric:** a single pre-declared primary metric (macro-F1 over in-scope entries with truth cell ≥ 5), evaluated on a **held-back lockbox split touched once** — not the reused `random.Random(0)` fold.
- **Multiple-comparisons control:** Benjamini-Hochberg across grid × entry; a config is kept only if it beats the floor after correction.
- **Sparse-entry rule:** entries with adjudicated truth cell **n < 5** (ROLL-CFAS, ROLL-LAPTF, LLM08, NEW-ITSCD, NEW-PMP, …) are declared **recall-unmeasurable**; they are never the basis of a "win" and are flagged unmeasurable downstream.
- **Reproducible floor:** the multimodel/active label provenance is currently an unscripted hybrid; the bake-off code recomputes the F1 floor from a clean checkout with a documented truth field, so "beats the floor" is auditable.
**Output:** `classify/labeled_incidents.json` for the chosen config + `classify_provenance.json` hashing the label file and recording HF model revisions.

### 5.3 Track B — hierarchical pooling robustness spec `[NEW: engine/model/hierarchical.py; CHANGE: robustness dispatch]` (RM6, RM9, RM10)
A **declared robustness spec**, not the primary:
- Model: `log λ_i = β0 + u_i`, `u_i ~ Normal(0, σ_u)`, **non-centered parameterization** to avoid funnel pathologies, over the ~20 entries; same NegBin2 measurement-error likelihood as the primary.
- **σ_u is pre-registered:** new manifest field `sigma_u_hyperprior`; the R reference's `sd≈2.19` is **not** imported as a prior (different link/family/response). Mandatory **sensitivity sweep across ≥3 pre-declared priors**; if any disagreement flag flips across the band, the robustness report is labeled "prior-sensitive."
- `λ` is recorded as a real site (`numpyro.deterministic("lambda", exp(β0+u))`) so the downstream shape contract (`lambda_samples`, `concordance._ranks_from_lambda`) is preserved; σ_u is **persisted** in the inference summary JSON.
- **ESS/R-hat gate fix (RM10):** `_AUX_PARAMS` is parameterized rather than hardcoded to `{"concentration"}`, so adding `σ_u`/`u` does not silently mis-target the gate; the gate must explicitly cover `λ` and the new scale.
- **Dispatch fix:** the robustness runner selects model by spec name; `primary_spec`/robustness-spec identity is recorded as the *executed* value (§5.6).

### 5.4 Track C — measurement-error correction reporting `[CHANGE: engine/report; engine/cli/pipeline_executor.py]` (RM12)
The correction already exists in the likelihood; this track makes it honest and visible:
- **Report raw-count ranking beside recall-corrected (λ) ranking** so the size of the correction is explicit.
- **Rank by incidence `λ·size`, not bare `λ`**, matching "rank by corrected incidence."
- **False-positive term:** the production executor currently passes `OverlapWeights(weights={})` (W=0 → precision inert). Either populate W from the measured cross-entry confusion, or **explicitly document precision-correction as unused** for this cycle. Decision required (§12).

### 5.5 Track D — tie-aware Plackett-Luce robustness spec `[NEW: engine/vote/plackett_luce.py]` (RM8, RM11)
A **vote-side robustness spec** feeding the vote-vs-data comparison; kappa stays primary:
- **Tie-aware model (Davidson or Rao-Kupper)** — *not* "drop ties." ~32% of ballot pairs tie, concentrated in the top tier; dropping them biases exactly the ranked tier. The strict-drop variant is reported only as a sensitivity comparison.
- **Uncertainty from respondent-level bootstrap ≥1000** over the ~29 voters, seed bound in `manifest.prng_seed` (not a function default). Report top-tier stability across a **seed × tie-rule grid**, plus per-item worth SEs; frame "top-five in 100% of resamples" as a **dominance check at n=29**, not a precision claim.
- **Separation handling:** when a category is ranked top by (nearly) all voters, the PL worth MLE diverges; use a regularized/penalized fit so bootstrap resamples that drop the rare dissenters remain defined.
- **Implementation:** engine uses `scipy`/`numpy` (no new heavy dep). The oracle (§5.6) uses an *independent* implementation.

### 5.6 Verification — independent Python consistency-check + enforceable gate `[NEW: engine/verify/oracle.py]` (RM7, RM11) — **no R, ever**
- The oracle re-derives σ_u, PL worths/ranks, and the corrected ranking from a **frozen spec, without reading engine code**, using a **different implementation** (different algorithm/library; if a new lib is used it must pass §10's supply-chain gate, else hand-rolled on pinned `scipy`).
- **Per-deliverable agreement tolerances, pre-declared in the manifest** (a single "agrees" flag is incoherent):
  - Ranks (PL, corrected): **Kendall-τ ≥ τ₀** and exact agreement on the headline tier;
  - σ_u (continuous): **|Δσ_u| ≤ 2 × combined MCSE** of both chains;
  - Corrected incidence: **credible-interval overlap** per entry.
- **Enforceable gate:** the oracle decision is an **append-only entry in the Merkle-chained `post_hoc_register`** with `signed_at` derived from `git log` (backdating-detectable) and a **non-author signer**. `decide` refuses a publishable report unless the chain verifies. Because the oracle shares author + conceptual source, it is labeled a **consistency check**, not independent verification (RM7).

### 5.7 Provenance & reproducibility `[CHANGE: engine/repro, engine/snapshot, engine/prereg/manifest.py]` (RM5, RM13)
- Add `goldset_hash` to `PreregManifest` and verify it at `infer`/`decide` (the goldset parameterizes every prior and is currently unbound).
- Calibration provenance hashes the **classifier label file**; the reproduction bundle records a real `snapshot_hash` (currently `"none"`), HF model revisions, seeds, and pod metadata.
- Commit the currently-untracked judgment artifacts (`curation_review.md`, backups) or move them out of the cycle dir; no asymmetric durability.

### 5.8 `primary_spec` dispatch + drift integrity `[CHANGE: engine/cli/pipeline.py, engine/report/diff.py, tests/proofs]` (RM5)
- Fix the tautology: `compute_prereg_diff` receives the **executed** spec as `actual_primary_spec`, not `manifest.primary_spec` on both sides.
- An unknown/unrunnable spec **raises**; no silent fallback to HalfNormal.
- `test_two_cycle_parity` asserts **which model ran**, not just output parity.

---

## 6. Foundation-first execution sequence

Strict ordering — each gate blocks the next (⛔ = no downstream number may be computed until done):

1. ⛔ **Fix recall calibration** (§5.1) + tests → recompute `posteriors.json`; confirm sparse-entry posteriors widen.
2. ⛔ **Reproducible classifier provenance + F1 floor** (§5.2) from a clean checkout.
3. ⛔ **Lock the pre-registration** (§9): manifest (kappa primary unchanged; robustness specs + grid + σ_u prior + seeds + tolerances + `goldset_hash` declared), Merkle `post_hoc_register` opened, reviewers lined up.
4. **Track A bake-off** on RunPod GPU → chosen classifier; recalibrate (§5.1 estimator).
5. **Primary re-run** (kappa concordance, negbin) on the new labels → the headline number.
6. **Robustness specs** (Track B hierarchical, Track C corrected ranking, Track D tie-aware PL) on RunPod CPU → spread.
7. **Oracle consistency check** (§5.6) → Merkle gate.
8. **Report** (raw vs corrected; primary + robustness spread; PL vote ranking + vote-vs-data gap; original kappa preserved; power statement) → `decide`.

---

## 7. Pre-registration & anti-gaming controls

Every researcher degree of freedom the premortem found is closed by a **pre-committed** value in the locked manifest, before any new number:

| Degree of freedom | Pre-commitment |
|---|---|
| Classifier config | Frozen grid + size *N*, single primary metric, lockbox split, BH correction (RM3) |
| Sparse-entry recall | n<5 → unmeasurable; never a selection basis (RM3) |
| σ_u prior | `sigma_u_hyperprior` field + ≥3-prior sensitivity band; flip ⇒ "prior-sensitive" (RM6) |
| PL tie rule | Tie-aware (Davidson/Rao-Kupper) registered; strict-drop only as sensitivity (RM8) |
| Bootstrap seed | Bound in `manifest.prng_seed`; report over seed grid (RM8) |
| OOS / measurable set | Frozen measurable entry set; any drop is a logged amendment (RM3) |
| Oracle "agreement" | Per-deliverable numeric tolerances declared up front (RM7) |
| Disagreement claim | Direction + threshold pre-registered before any fit |

---

## 8. Cycle structure & the original result

- **New cycle directory** (e.g. `projects/owasp-llm/cycles/2026-rarr/`) binding the **same `snapshot_hash`, `taxonomy_hash`, and `goldset_hash`** as the 2026 cycle — auditably the same data, new classifier + robustness lenses.
- **`primary_spec` and `statistic` unchanged** from the original (kappa concordance over negbin) — this is a re-analysis, not a new headline.
- **Original `cycles/2026/` left byte-immutable.** Before freezing the comparison, reconcile the original report's internal kappa inconsistency (0.20 in `report.md:12` vs 0.275 in `report.md:37`) so the baseline anchors a single number (RM14).
- Old-vs-new presented via an explicit, documented bridge in the report (not the engine's auto cross-cycle comparison, which is deliberately refused).

---

## 9. Governance package (ordered; tiers gate publishability) (RM11)

**Before the first new number:**
1. Resolve the "Plan 7" name collision — assign a distinct phase label (§11).
2. Lock the new manifest (all §7 pre-commitments) — lock timestamp must precede the first new number.
3. Open and populate the Merkle `post_hoc_register.json`, tagging the re-analysis **EXPLORATORY**.
4. Identify external **rubric** and **statistical** reviewers (≠ ranking author), attestations signed before `infer`.

**Before external sharing:**
5. `METHODOLOGY-CHANGELOG.md` entry with a **major** semver bump (inference-model-family change), stating what changed, why, and its relation to the pre-registered original.
6. Two-cycle parity + the reviewer audit window.

Until 1–4 hold, output is `non_publishable=True` / EXPLORATORY by the repo's own rules — that is the honest state for single-author work, not a flipped flag.

---

## 10. Supply-chain & security (RM13)

- Prefer implementing PL + oracle on already-pinned `scipy`/`numpy`. Any new dependency (e.g. `choix`) must be **pinned + added to the SBOM + CVE-scanned** before use; the repo already carries two deferred HIGH transitive CVEs, so PyMC's large closure is avoided unless justified.
- **Pin RunPod model revisions** (not floating HF names); record the image digest.
- Remove or explicitly justify `--trust-remote-code` on token-bearing pods; do not expose public SSH on a pod holding `HF_TOKEN`.
- **Escape the Stage-2 injection delimiters** against corpus text that contains them; re-run the injection fixture against any new bake-off model before it is kept.

---

## 11. Naming & accountability

The PRD assigns "Plan 7" to the frame-coverage audit. This work needs a distinct phase slot — proposed **"Plan 8 — Recall-Aware Robustness Re-analysis"** — with its own PRD phase-map entry and acceptance criteria. The branch `plan7/engine-upgrade-recall-pl` may be renamed for clarity. *Final phase number is a project-owner decision (§12).*

---

## 12. Open decisions (need the user / project owner)

1. **Phase number** — adopt "Plan 8," or another slot? (§11)
2. **Precision/false-positive term (§5.4)** — populate the overlap matrix `W` from measured confusion this cycle, or explicitly document precision-correction as unused and defer? (Populating it is more correct but adds modeling + a confusion-estimation step.)
3. **External statistical reviewer** — who, and by when? Without one, the cycle is structurally `non_publishable` (§9). For an internal tool this may be acceptable as a logged EXPLORATORY state.
4. **Bake-off breadth** — minimum (re-bless the existing multimodel with clean provenance + fixed calibration) vs. a full grid with a candidate 4th model. Both honor RunPod-always; the full grid costs more GPU for the evidence that we're at the recall frontier.

---

## 13. Remediation traceability (premortem → spec)

| RM | Premortem finding | Closed by |
|---|---|---|
| RM1 | Outcome-switching / PL-as-headline / "single move" | §2 framing; §5.5 (PL = robustness) |
| RM2 | Recall denominator bug (Beta(1,101)) | §5.1 |
| RM3 | Unreproducible floor / bake-off gaming / tiny cells | §5.2; §7 |
| RM4 | RunPod provenance & teardown | §4; §5.2; §5.7 |
| RM5 | `primary_spec` no-op / drift tautology / parity | §5.8 |
| RM6 | σ_u prior-dominated / 2.19 not transferable | §5.3; §7 |
| RM7 | Oracle non-independent / no tolerance | §5.6 |
| RM8 | Drop-ties bias / seed lottery / n=29 stability | §5.5; §7 |
| RM9 | Hierarchical breaks downstream shape | §5.3 |
| RM10 | ESS gate site-name fragility / σ_u not persisted | §5.3 |
| RM11 | Governance package absent/out of order | §9 |
| RM12 | Track C FP term inert / ranks bare λ | §5.4 |
| RM13 | Supply chain + pod security | §10 |
| RM14 | Ambiguous baseline kappa; untracked artifacts | §8; §5.7 |

---

## 14. Acceptance criteria

- Recall posteriors for n<5 entries are **wide**, not `Beta(1,101)`; the recall-tally proof test passes.
- The F1 floor regenerates byte-stable from a clean checkout; classifier provenance hashes the label file + records model revisions.
- The locked manifest's `primary_spec`/`statistic` equal the original; robustness specs, grid, σ_u prior, seeds, tolerances, and `goldset_hash` are all present; lock timestamp precedes the first new number.
- Bayesian diagnostics: R-hat < 1.01, adequate ESS (gate covers `λ` and `σ_u`), zero divergences or a documented reason, posterior predictive checks.
- σ_u sensitivity table across ≥3 priors is reported; prior-sensitivity is disclosed if flags flip.
- PL is tie-aware; the report shows the seed × tie-rule stability grid and per-item worth SEs.
- The oracle consistency check passes its **pre-declared** per-deliverable tolerances; the Merkle `post_hoc_register` chain verifies at `decide` with a non-author signer.
- Report shows: original kappa (preserved), the re-run primary kappa, the robustness spread (raw vs corrected, hierarchical, PL vote ranking + vote-vs-data gap), and a **power statement** on what n would be needed to exclude zero.
- CI green: ruff, mypy, pytest, semgrep, gitleaks; SBOM clean; cross-platform parity holds.

---

## 15. Strategic note

The conclusion rests on a concordance over **n_common = 17 with a CI crossing zero** — an *n problem*. The single highest-value outcome of this work is **Track A recall raising `measurable_count`**, which is the only lever that can tighten that CI. Tracks B/C/D are robustness lenses that explain the result better; they do not, by themselves, move the headline. The report leads with that honestly (§14 power statement) rather than dressing an unchanged conclusion in a confident new statistic.
