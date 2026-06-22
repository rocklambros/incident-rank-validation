# Design Spec — Recall-Aware Robustness Re-analysis (RARR)

- **Status:** DRAFT · **Revision 2** (post adversarial-premortem-of-the-spec; supersedes R1 commit `f1cbc45`)
- **Date:** 2026-06-22
- **Branch:** `plan7/engine-upgrade-recall-pl`
- **Provisional phase label:** *Plan 8 — Recall-Aware Robustness Re-analysis* (PRD already assigns "Plan 7" to the frame-coverage audit; final number is a project-owner decision — §11)
- **Source prompt:** `claudedocs/engine-upgrade-runpod.md`
- **Lineage:** Revised by two adversarial premortems — one on the original recommendations (14 remediations RM1-RM14), one on Revision 1 of this spec (17 spec-deltas SD1-SD17). Both traced in §13.

### Revision 2 changes (what the spec-premortem caught in R1)
The premortem of Revision 1 found that several remediations were unbuildable or self-defeating. R2 fixes them: the recall fix now targets the correct code path (the adjudicated goldset, not the recall-frame batches); `goldset_hash` no longer mutates the frozen manifest (it would have broken the immutable cycle's lock); the robustness-reporting and Merkle-gate machinery are scoped as **net-new work** rather than assumed to exist; the σ_u oracle is re-scoped to a buildable surrogate; the lock-before-numbers ordering is fixed; governance is right-sized to an internal tool; and three user decisions are encoded (carry sparse entries, λ·size primary ranking, internal-tool governance).

---

## 1. Context

`incident-rank-validation` validates the OWASP LLM Top-10 community vote against ~7,700 real incidents. The frozen 2026 cycle reported quadratic-weighted Cohen's kappa **0.20 (CI [-0.16, 0.57], 17/20 measurable)** — inconclusive. A COMP-4442 re-analysis concluded: the core call was sound ("inconclusive by kappa" is the correct hedge) — **explain it better, do not overturn**; a single kappa hides structure and sparse-category rates wobble; classifier recall is low and uneven, so counts are a biased undercount.

This is a **correctness-and-robustness re-analysis that keeps the original conclusion's primacy**, not a headline swap. Every choice is shaped by two rounds of adversarial premortem.

### 1.1 What the engine already is (verified against code)
- **Bayesian stack:** NumPyro/JAX, **CPU-hard-asserted** (`inference.py:121`; `pyproject.toml` `JAX_PLATFORM_NAME=cpu`). The cross-platform CI gate checks only *categorical* sets, not Bayesian numerics (`ci.yml:60-62`); two-cycle parity uses a 0.15 kappa tolerance on one machine (`test_two_cycle_parity.py:74`) — so reproducibility is **within-MCSE**, not bit-exact.
- **Measurement-error likelihood already exists** (`inference.py:172-192`): `true_rate=λ·size`; `tp=true_rate·recall`; FP leakage `W·true_rate·(1-precision)`; `NegativeBinomial2`. **But the production executor passes `OverlapWeights(weights={})`** (`pipeline_executor.py:253`) → `W=0` → the precision/FP term is inert. `concordance.py:53-63` ranks the vote against **bare λ**, not incidence λ·size.
- **Recall calibration:** the correct truth-vs-prediction estimator is `calibrate_with_gold` (`tally.py:227-242`) over the **adjudicated goldset** (`adjudicated_goldset.jsonl`, which carries both `classifier_entry_id` and `true_entry_ids`). The separate `tally_batches` recall path (`tally.py:94-116`) operates on recall-frame batches that **store only truth labels, no classifier prediction**, and inflates denominators to frame size.
- **Classifier:** Stage-1 indicator match (667) → Stage-2 LLM for the rest (5,972), runnable as a 3-model RunPod consensus (`reclassify.py`).
- **Robustness machinery is unwired:** `RobustnessSpread` (`robustness_multiplicity.py:31-56`) is **kappa-only**; `run_robustness_inference` (`robustness.py:26`) has **no CLI caller**; both report callers pass `robustness=None`. Adding hierarchical/PL/corrected-ranking robustness specs is **net-new wiring**, not a config flip.
- **Prereg/governance primitives exist but are partly unwired:** `PreregManifest` + SHA-256 lock (`prereg/{manifest.py,lock.py}`); Merkle `post_hoc_register` (`erratum/{merkle.py,post_hoc.py}`) whose verifier `read_and_verify_register` has **no callers** and whose `PostHocAnalysis` has **no signer / no git-timestamp** field; git-derived `signed_at` exists only for `ReviewerSignoff` (`prereg/git_timestamp.py`). For single-author work the repo's posture is **discipline-over-mechanism** (`REVIEWERS.md:56`).

### 1.2 Premortem root causes this spec must fix
1. **Calibration foundation** — recall must be computed truth-vs-prediction per incident over the goldset (not frame-size denominators), or sparse-entry posteriors are falsely-precise near-zero.
2. **A manifest that would lie** — `primary_spec` dispatches no model; robustness dispatch is unwired; the prereg must not assert a model the engine never runs.
3. **Question substitution / outcome-switching** — Plackett-Luce summarizes the *vote*; it stays a robustness lens, not the concordance headline.
4. **Illusory verification + open gaming surface** — the oracle and the pre-registration must close (not relocate) researcher degrees of freedom, with buildable mechanisms.

---

## 2. Research question (unchanged) and framing decision

**The research question and headline statistic are unchanged:** *Does the community vote ranking concord with incident-derived prevalence?*, measured by the **quadratic-weighted Cohen's kappa over the measurable subset**.

**Framing decision (RM1):** hierarchical partial pooling and Plackett-Luce enter as **declared robustness specifications under the unchanged kappa primary** (HANDOFF §6 control 4). The recall-corrected incidence ranking is the **primary's ranking input** (see §2.1). This keeps the COMP-4442 verdict intact, avoids statistic-switching, and collapses the "single-move" gaming chain — **provided the robustness spread is mechanically reported** (§5.5), which is net-new work, not an inherited property.

### 2.1 Ranking quantity — primary ranks by incidence λ·size (user decision, SD8)
The primary kappa's ranking input moves from bare latent λ to **incidence `λ·size`** (summed over strata), the decision-relevant quantity. This is a **correctness fix to the ranking input, not a change of statistic or question**, and it is applied **symmetrically**: the byte-immutable 2026 cycle is untouched, but RARR derives a **λ·size baseline kappa from the original labels** and reports it **beside** the as-published bare-λ 0.20 (for continuity) and the new corrected result. Disclosed in the changelog. *Residual risk: this does move the headline number; the symmetric recomputation + disclosure is the mitigation (§16).*

---

## 3. Goals & non-goals

**Goals**
- G1. Fix recall calibration (truth-vs-prediction over the goldset) so per-entry posteriors are honest — **wide where data is sparse** — then rebuild the chain on a single reproducible classifier.
- G2. Improve/re-measure recall via a pre-registered RunPod bake-off (the only lever that raises `measurable_count`).
- G3. Add hierarchical pooling + tie-aware Plackett-Luce as **mechanically-reported robustness specs**; make the FP/precision term live and rank by incidence.
- G4. Provide an independent Python **consistency check** (no R) with a buildable, honestly-scoped gate.
- G5. Lock a pre-registration **before any new number** (incl. calibration), leaving the original cycle byte-immutable.

**Non-goals**
- N1. Changing the headline statistic or the question (§2).
- N2. Overturning the kappa conclusion (report it; do not engineer it).
- N3. Any Bayesian fit on GPU (forbidden by `inference.py:121`).
- N4. Heavy new deps (PyMC) unless §10's supply-chain gate is met; prefer pinned `scipy`/`numpy`.
- N5. **External-publication ceremony as a blocking gate** — this is an internal tool (§9, user decision SD11).

---

## 4. Compute policy — RunPod always for heavy work

All training and heavy inference run on RunPod; the local Jetson is never used for heavy compute. **RunPod is the *location*; CPU is the *backend* where determinism requires it.**

| Workload | RunPod pod | Backend |
|---|---|---|
| Track A classifier bake-off (LLM labeling) | **GPU** (H200, vLLM, pinned image **+ pinned HF model revisions**) | GPU |
| Track B/C hierarchical NUTS + measurement-error + σ_u sweeps | **CPU** (high-vCPU) | CPU (JAX X64, seeded) |
| Track D tie-aware PL + ≥1000 respondent bootstrap | **CPU** | CPU |
| Verification consistency-check oracle | **separate CPU pod** | CPU |

**Determinism is within declared MCSE, not bit-exact** (SD10): pin `OMP_NUM_THREADS` and `XLA_FLAGS` in the pod provenance; the two-cycle parity tolerance and the oracle σ_u tolerance are sized to **cross-pod** MCSE. Record pod type, image digest, **resolved HF model commit SHAs**, engine commit, seeds, and logs per pod; tear pods down on completion.

---

## 5. Architecture & components

### 5.1 Foundation — recall calibration fix `[CHANGE: route recall through engine/calibrate (calibrate_with_gold path)]` (RM2, SD1)
**Compute recall truth-vs-prediction per incident over the adjudicated goldset**, not the recall-frame batches:
- Use the `calibrate_with_gold` estimator (`tally.py:227-242`), which compares `classifier_entry_id` to `true_entry_ids` per incident.
- `recall_X = TP_X / (TP_X + FN_X)` with **per-entry truth-cell denominator** (incidents truly X), → `Beta(1+TP, 1+FN)`. Sparse entries become **wide**, not falsely precise.
- Route the cycle's recall calibration through this path; deprecate the frame-size-denominator `tally_batches` recall branch for production use. Add a proof/unit test (currently none exercises this).
- **Do not** target `tally.py:99-116` (the recall-frame branch has no classifier prediction to compare).

### 5.2 Track A — classifier bake-off `[NEW: engine/classify/bakeoff.py + engine/cli/bakeoff.py]` (RM3, SD12)
RunPod-GPU re-classification, selected by a **pre-registered** procedure (locked before any run — §6):
- **Config grid + size N** committed to the locked manifest: model set, consensus rule, OOS-calibrated prompt variant, thresholds.
- **Selection metric: balanced accuracy that INCLUDES the out-of-scope class** (not in-scope macro-F1 only) — 37% of the goldset is OOS, so a metric that ignores OOS rewards over-assignment. Evaluated on a **held-back lockbox split touched once**; declare the per-entry minimum lockbox cell size.
- **Multiple-comparisons control:** Benjamini-Hochberg across grid × entry; keep a config only if it beats the floor after correction.
- **Sparse-entry rule (SD2):** entries with truth cell **n<5** are excluded from the **selection metric only** (you may not pick a classifier on a 1-3-sample cell). They are **NOT dropped** from calibration or the concordance ranking — they are carried with honest wide posteriors (§5.1, §5.3). Recompute the n<5 list from the goldset at lock time (note: NEW-PMP truth cell = 6, i.e. measurable).
- **Reproducible floor** recomputed from a clean checkout with a documented truth field, so "beats the floor" is auditable.
**Output:** chosen-config labels + `classify_provenance.json` hashing the label file and recording **resolved HF model commit SHAs**.

### 5.3 Track B — hierarchical pooling robustness spec `[NEW: engine/model/hierarchical.py + robustness dispatch + CLI wiring]` (RM6, RM9, RM10, SD13)
A **declared, mechanically-reported** robustness spec:
- `log λ_i = β0 + u_i`, `u_i ~ Normal(0, σ_u)`, **non-centered**, ~20 entries, same NegBin2 likelihood; `λ` recorded as a deterministic site so the `lambda_samples` contract and downstream consumers survive; **σ_u persisted** in the inference summary JSON (new field).
- **σ_u prior pre-registered** (new manifest hyperparameter); the R reference's `sd≈2.19` is **not** imported as a prior. Sweep ≥3 pre-declared priors. **Pre-committed decision rule (SD13):** feature the most-conservative-pooling prior's result; **if σ_u's posterior is dominated by its prior** (declared prior/posterior-overlap criterion), **abandon pooling** and report independent per-entry rates with wide intervals instead (pooling may be the wrong tool at ~16-20 groups).
- **ESS/R-hat gate fix (RM10):** parameterize `_AUX_PARAMS` (don't hardcode `{"concentration"}`); the gate must cover `λ` and `σ_u`. Expect the hierarchical scale to mix harder than the primary — that is information, not a bug.
- **Dispatch + wiring (SD4):** extend `run_robustness_inference` to dispatch `"hierarchical_pooling"` **and wire it into the CLI** (`run_robustness_inference` currently has no caller). Record the *executed* spec name in provenance.

### 5.4 Track C — measurement-error correction reporting `[CHANGE: engine/report; pipeline_executor.py]` (RM12, SD8)
- **Rank the primary by incidence `λ·size`** (summed over strata) — §2.1. Thread `sizes` into `compute_concordance` (currently receives none).
- **Report raw-count ranking beside recall-corrected ranking** so the correction's size is explicit.
- **FP/precision term:** populate the overlap matrix `W` from the measured cross-entry confusion so the precision posteriors are live, **or** explicitly document precision-correction as unused this cycle (decision — §12).

### 5.5 Robustness reporting mechanism `[NEW: heterogeneous robustness spread]` (SD4 — the load-bearing fix for §2)
The existing `RobustnessSpread` is kappa-only; making §2's anti-gaming claim real requires a **new mechanical report contract** that holds heterogeneous robustness outputs together:
- A typed structure carrying, per robustness spec: the kappa under that spec (where applicable), σ_u + sensitivity band, the PL worth ranking, and the corrected-incidence ranking — rendered as a single **spread/comparison block** the report cannot omit (a `decide`-time check refuses a report missing declared robustness outputs).
- Without this, "declared robustness spec" is an author-discretion label and the gaming surface merely moved to the narrative (premortem S-D/S-L).

### 5.6 Track D — tie-aware Plackett-Luce robustness spec `[NEW: engine/vote/plackett_luce.py]` (RM8)
- **Tie-aware model (Davidson/Rao-Kupper)**, not "drop ties" (~32% of pairs tie, concentrated in the top tier). Strict-drop is reported only as sensitivity.
- **Respondent bootstrap ≥1000** over ~29 voters; **seed bound in `manifest.prng_seed`**; report top-tier stability across a **seed × tie-rule grid** + per-item worth SEs; frame "top-five 100%" as a **dominance check at n=29**, not precision.
- **Separation handling:** penalized/regularized fit so resamples that drop rare dissenters stay defined.
- **Implementation:** engine on pinned `scipy`/`numpy`.

### 5.7 Verification — independent Python consistency check + gate `[NEW: engine/verify/oracle.py + post_hoc gate wiring]` (RM7, RM11, SD5, SD6) — **no R, ever**
- The oracle re-derives the deliverables from a frozen spec, **without reading engine code**, using a **different optimizer family** (e.g. MM/fixed-point for PL, not `scipy.optimize`):
  - PL worths/ranks and corrected ranking: independent implementation on pinned numpy.
  - **σ_u: an analytic/optimization surrogate** (REML / Laplace marginal-likelihood point estimate + interval) — **not** a matched MCMC posterior (scipy has no NUTS; matching two NUTS chains within MCSE is infeasible). Tolerance is declared against the surrogate.
- **Per-deliverable tolerances, pre-declared, sized to cross-pod MCSE:** Kendall-τ ≥ τ₀ + exact headline-tier agreement (ranks); |Δσ_u| ≤ band (surrogate); CI-overlap (corrected incidence).
- **Gate is net-new build (SD6):** add a **signer** and **git-`log`-derived `signed_at`** to `PostHocAnalysis`; **wire `read_and_verify_register` into `decide`** with a **signer≠author** check; `decide` refuses a publishable report unless the chain verifies. Labeled a **consistency check, not independent verification** (shared author/conceptual source), and **discipline-based for single-author work** (the author can self-sign; disclosed, not oversold as tamper-proof — Merkle detects edits, not full-chain fabrication).
- **Oracle environment (SD5):** committed as a distinct `oracle.uv.lock` (or hand-rolled on the pinned stack); its lockfile hash recorded in the register so "independent env" is reproducible.

### 5.8 Robustness wiring + drift integrity `[CHANGE: pipeline.py, robustness.py, tests/proofs]` (RM5, SD4)
The real gap is not the harmless `primary_spec` self-comparison (inert while the primary is unchanged) but that **robustness dispatch is unwired**. Wire `run_robustness_inference` into the CLI; record the executed spec; add a parity-proof assertion that the executed model matches the declared spec. Fix the `compute_prereg_diff` self-comparison opportunistically.

### 5.9 Provenance & reproducibility `[CHANGE: engine/repro, engine/snapshot; schema-versioned manifest]` (RM13, SD3, SD9, SD16)
- **Do NOT mutate the frozen `PreregManifest` dataclass.** Adding a field rehashes every prior manifest and breaks the 2026 lock (`lock.py` hashes all `dataclasses.fields`). Carry `goldset_hash` via a **manifest schema-version** (old locks verify under v1 rules) or a **sidecar provenance file**.
- Calibration provenance hashes the **classifier label file**; the reproduction bundle records a real `snapshot_hash` (currently `"none"`), resolved HF model SHAs, seeds, pod metadata.
- **Pin HF model revisions** end-to-end (`--revision` in `vllm serve`; capture resolved SHA into `runpod_pods.json` + provenance + bundle) — net-new (SD16).
- **Reconcile the pre-existing 2026 provenance rot** (engine_version 0.3.0/1.1.0/1.2.0 across files; `manifest_hash ≠ lock`) before anchoring RARR to it.
- **Curation artifacts (SD9):** the currently-untracked judgment artifacts go to a **separate `projects/owasp-llm/curation/2026/` dir**, NOT committed under the byte-immutable `cycles/2026/`. Run gitleaks before committing any artifact.

---

## 6. Foundation-first execution sequence (lock-before-numbers fixed — SD7)

⛔ = blocks downstream numbers. **Calibration is a pre-registered number** (HANDOFF §11a), so the lock precedes it.

1. ⛔ **Land the estimator + provenance code changes** (recall path §5.1, schema-versioned `goldset_hash` §5.9, robustness wiring §5.8) — code, not numbers.
2. ⛔ **Lock the pre-registration** (§9): kappa primary unchanged; λ·size ranking declared; robustness specs + bake-off grid/metric/lockbox + σ_u prior + seeds + oracle tolerances + `goldset_hash` all committed; Merkle register opened. Lock timestamp (git-derived) precedes every number below.
3. ⛔ **Recompute recall calibration** via §5.1 → write to `posteriors.precheck.json`; confirm sparse-entry posteriors widen.
4. **Track A bake-off** on RunPod GPU → chosen classifier; **recalibrate** → canonical `posteriors.json` (record `output_hash`).
5. **Primary re-run** (kappa concordance, negbin, **λ·size ranking**) on new labels → headline; also compute the **λ·size baseline from original labels** (§2.1).
6. **Robustness specs** (Track B hierarchical, Track C raw-vs-corrected, Track D tie-aware PL) on RunPod CPU → mechanical spread (§5.5).
7. **Oracle consistency check** (§5.7) → Merkle gate at `decide`.
8. **Report** (original 0.20 + λ·size baseline + new primary + robustness spread + PL vote ranking + vote-vs-data gap + prospective power statement) → `decide`.

---

## 7. Pre-registration & anti-gaming controls

Every degree of freedom is a **pre-committed, git-timestamped** value in the locked manifest, before any number:

| Degree of freedom | Pre-commitment |
|---|---|
| Classifier config | Frozen grid + N, OOS-inclusive balanced-accuracy metric, once-touched lockbox, BH correction |
| Sparse-entry recall | n<5 excluded from **selection metric only**; carried in ranking with wide posteriors |
| σ_u prior | Pre-registered prior + ≥3-prior sweep + **pre-committed decision rule** (conservative prior / abandon-pooling criterion) |
| PL tie rule | Tie-aware (Davidson/Rao-Kupper) registered; strict-drop only as sensitivity |
| Bootstrap seed | Bound in `manifest.prng_seed`; reported over a seed grid |
| Ranking quantity | λ·size, declared; baseline recomputed symmetrically |
| OOS / measurable set | Frozen; any drop is a logged amendment |
| Oracle tolerances | Per-deliverable, declared, cross-pod-MCSE-sized |
| Disagreement claim | Direction + threshold pre-registered before any fit |

*Honest limit (S-L):* for single-author work these are **discipline-based** — the lock and signer can be self-applied. Disclosed, not oversold.

---

## 8. Cycle structure & the original result

- **New cycle directory** (e.g. `cycles/2026-rarr/`) binding the same snapshot/taxonomy hashes (and `goldset_hash` via schema-version) as 2026.
- **`primary_spec`/`statistic` unchanged** (kappa concordance over negbin); λ·size is a ranking-input change, declared.
- **Original `cycles/2026/` byte-immutable.** RARR derives the λ·size baseline from the original labels *in the RARR cycle* (not by editing 2026). Reconcile the original report's internal kappa inconsistency (0.20 vs 0.275) in the RARR write-up, not by mutating the original.
- **Old-vs-new bridge (SD15):** report **old-labels kappa under the new λ·size estimator** alongside the new-labels result, to separate *data change* (new classifier) from *method change* (estimator) — disclose the instrument confound.

---

## 9. Governance — right-sized to an internal tool (user decision SD11)

**Keep all MECHANICAL methodology-integrity controls** (they are cheap and they are the point): recall-calibration fix, leakage firewall (lock-before-numbers), pre-registered grid/priors/seeds/tolerances, mechanical robustness spread, oracle consistency gate, provenance/seed hygiene.

**Make external-publication ceremony OPTIONAL and non-blocking:** external rubric/statistical reviewers, the audit window, and publication-grade SBOM are **not** blocking gates for this internal deliverable. The honest deliverable state is **"EXPLORATORY, internally rigorous."** `non_publishable=True` is acceptable and expected for single-author internal work; it is disclosed, not a defect to engineer around.

Still required, in order, before the first number: (1) resolve the phase-name collision (§11); (2) lock the manifest (§6 step 2); (3) open the Merkle register. Changelog: a methodology-changelog entry with an appropriate semver bump.

---

## 10. Supply-chain & security (RM13, SD16)

- Prefer PL + oracle on pinned `scipy`/`numpy`; any new dep (e.g. `choix`) must be pinned + SBOM'd + CVE-scanned (repo already carries 2 deferred HIGH CVEs).
- **Pin RunPod model revisions** (commit SHA, not tag); record image digest. `--trust-remote-code` permitted **only with a pinned revision recorded in provenance**; **no public SSH on token-bearing pods**; scope `HF_TOKEN` read-only.
- **Escape the Stage-2 injection delimiters** (currently un-escaped — `stage2_prompt.py:79-83`); build a **live/recorded-response injection gate** against any new bake-off model (today's fixture is mock-only, `xfail`).

---

## 11. Naming & accountability

"Plan 7" is taken (frame-coverage audit). Use a distinct slot — proposed **"Plan 8 — Recall-Aware Robustness Re-analysis"** — with its own PRD phase-map entry. Final number is a project-owner decision (§12).

---

## 12. Open decisions (project owner)

1. **Phase number** — adopt "Plan 8"? (§11)
2. **Precision/FP term (§5.4)** — populate `W` from measured confusion this cycle, or document precision-correction as unused and defer?
3. **Bake-off breadth** — minimum (re-bless multimodel with clean provenance + fixed calibration) vs. a full grid with a candidate 4th model. Both honor RunPod-always.

*(Governance level, sparse-entry handling, and ranking quantity are resolved — §9, §5.2, §2.1.)*

---

## 13. Traceability

**Premortem #1 → spec:** RM1 §2; RM2 §5.1; RM3 §5.2/§7; RM4 §4/§5.9; RM5 §5.8; RM6 §5.3/§7; RM7 §5.7; RM8 §5.6; RM9 §5.3; RM10 §5.3; RM11 §9/§5.7; RM12 §5.4; RM13 §10; RM14 §8/§5.9.

**Premortem #2 (of spec R1) → R2:** SD1 §5.1; SD2 §5.2 (carry, user); SD3 §5.9; SD4 §5.5/§5.8; SD5 §5.7; SD6 §5.7; SD7 §6; SD8 §2.1/§5.4 (λ·size, user); SD9 §5.9; SD10 §4; SD11 §9 (internal-tool, user); SD12 §5.2; SD13 §5.3; SD14 §15; SD15 §8; SD16 §5.9/§10; SD17 §6.

---

## 14. Acceptance criteria

- Recall posteriors for sparse entries are **wide** (not `Beta(1,101)`); recall computed via the goldset truth-vs-prediction path; the new proof test passes.
- F1/balanced-accuracy floor regenerates from a clean checkout; classifier provenance hashes the label file + records resolved HF SHAs.
- Locked manifest: `primary_spec`/`statistic` equal the original; λ·size ranking, robustness specs, grid, OOS-inclusive metric, σ_u prior + decision rule, seeds, oracle tolerances, `goldset_hash` (schema-versioned) all present; **git-derived lock timestamp precedes every number**; the 2026 lock still verifies.
- Robustness spread is **mechanically rendered** (decide-time check); `run_robustness_inference` is wired and the executed spec is recorded.
- Bayesian diagnostics within MCSE: R-hat < 1.01, ESS gate covers λ and σ_u, divergences 0 or documented, PPCs; determinism reproduces **within declared cross-pod MCSE** (threads/XLA pinned).
- σ_u sensitivity table + pre-committed decision applied; PL tie-aware with seed×tie-rule stability grid + worth SEs.
- Oracle passes pre-declared per-deliverable tolerances; Merkle register verifies at `decide` with a recorded signer; consistency-check framing disclosed.
- Report shows: original 0.20 (bare-λ), λ·size baseline from original labels, new primary (λ·size), robustness spread, PL vote ranking + vote-vs-data gap, old-labels-under-new-estimator bridge, and a **prospective** power statement.
- CI green (ruff/mypy/pytest/semgrep/gitleaks); SBOM clean; categorical cross-platform parity holds; curation artifacts under `curation/2026/`, not `cycles/2026/`.

---

## 15. Strategic note & power (SD14)

The conclusion rests on a concordance over **n_common = 17 with a CI crossing zero** — an *n problem*. The highest-value outcome is **Track A recall raising `measurable_count`**, the only lever that tightens the CI. Report a **prospective** power statement: for a pre-specified, decision-relevant kappa (declared at lock, *not* the observed 0.20), the n required to exclude zero — labeled design-stage. Do **not** report retrospective/observed power on the realized estimate (it is a monotone restatement of the CI).

## 16. Residual risk

- **λ·size moves the headline (§2.1).** Mitigation: same statistic + symmetric baseline recomputation + disclosure. Accept that the number changes; that is a correctness improvement, reported transparently.
- **Discipline-over-mechanism for single-author work** (`REVIEWERS.md:56`): lock/signer can be self-applied; the gate reduces but cannot eliminate this without an external party — acceptable for an internal tool, disclosed.
- **σ_u may be unidentifiable at ~16-20 groups** — §5.3's decision rule makes "abandon pooling" an explicit, pre-committed branch.
- **Pre-existing 2026 provenance rot** is inherited; reconcile before anchoring (§5.9).
- **The conclusion may stay inconclusive** even done perfectly — reported as the finding (transparency-first), not papered over.
