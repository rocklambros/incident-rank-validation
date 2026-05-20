# Boundary cases and edge behavior

## Multi-target overlap weights (M11)

**Declaration:** overlap weights are declared in the rubric as a sparse matrix `overlap_weights[entry_i][entry_j] = w` where `w` is the fraction of an incident matching entry_i that is also attributable to entry_j. The full matrix for a given entry must be column-stochastic: all weights from a source entry sum to <= 1.0 (the remainder is "unshared").

**Column-stochasticity constraint:** `sum(overlap_weights[i][*]) <= 1.0` for each source entry `i`. The engine rejects any rubric where a source entry's outbound weights exceed 1.0 with a `OverlapWeightsError`.

**What "fractional" means in practice:** an incident classified under entry A with `overlap_weights[A][B] = 0.3` contributes 1.0 count to A and 0.3 fractional-count to B. The de-biased count for B is incremented by 0.3. The twin uses these fractional counts directly; the NUTS model samples integer-valued allocations via a Dirichlet-Multinomial draw over the weight vector. "Fractional" never means a single incident is split — it means the *expected contribution* to the overlapping entry is 0.3 incidents.

## Reviewer signoff timing under M8

**How to attest:** create a file at `cycle/attestations/<reviewer_role>.txt` with the reviewer's name, affiliation, and a statement that they have reviewed the rubric/methodology (per REVIEWERS.md). Commit the file. The engine reads `signed_at` from `git log --format="%aI" -1 -- <attestation_file>`.

**How `signed_at` is derived:** the engine calls `git log` on the attestation file path; it does NOT trust the timestamp inside the file. This means:
- Backdating by editing the file timestamp is detected (git commit timestamp is authoritative).
- Amending a commit that contains an attestation file changes the `signed_at` derived value; this is treated as a new signoff event, not a retroactive one.
- If the attestation file is re-committed (e.g., to fix a typo), `signed_at` updates to the new commit time. If this moves `signed_at` to after `infer` started, `viewed_results_before_signoff` becomes True and `non_publishable` is set.

**Re-committing attestation files:** re-committing after results are visible permanently stamps `non_publishable=True` on that cycle. There is no undo path short of discarding the cycle and re-running from a clean manifest.

## `--accept-drift-signoff` rationale composition

**Minimum length:** 30 characters. This is a mechanical floor, not a quality guarantee.

**What makes a useful rationale:**
- Name the source of the drift: "NVD re-categorized 47 buffer overflow entries from CWE-119 to CWE-787 on 2026-04-15"
- Confirm benign intent: "re-categorization is vendor-driven, not corpus manipulation"
- Reference a verifiable artifact if available: "see NVD feed diff commit abc123"

**What makes a useless rationale (examples to reject):**
- "benign changes" (12 chars; fails floor)
- "counts shifted slightly, accepting" (passes floor, gives no diagnostic value)
- "upstream update" (15 chars; fails floor)

**Adversarial ingestion:** if the drift cause is unknown or suspected adversarial, do NOT pass `--accept-drift-signoff`. File an erratum per HANDOFF section 6 and pause the cycle.

## Multi-tier quadratic kappa interpretation

**`tier_boundaries=[5,10]` meaning:** entries ranked 1-5 are "Top-5 tier", entries ranked 6-10 are "6-10 tier", entries ranked 11+ are the lower tier. The kappa computation treats these as ordered categories.

**Quadratic weight matrix:** disagreements are penalized proportional to the square of the distance between tier indices. A Top-5 vs 6-10 disagreement (adjacent tiers) incurs weight `(1)^2 / (max_distance)^2`. A Top-5 vs 11-20 disagreement (two tiers apart) incurs weight `(2)^2 / (max_distance)^2` — four times the penalty.

**Interpretation in context:** the kappa measures agreement between the vote ranking's tier assignment and the posterior-derived tier assignment. A high kappa with a Top-5 vs 11-20 disagreement is impossible by construction — such a disagreement drives kappa down sharply. If kappa is reported as N/A, see the N/A branch below.

## Frame-blind vs classifier-blind distinction

**Frame-blind:** the rubric author declared `frame_blind=True` on the entry. This means the entry's concept is definitionally outside the corpus's observable frame — no amount of classifier improvement would make it measurable in this corpus. Example: an entry covering insider threats in a corpus of external disclosures only.

**Classifier-blind:** the corpus contains incidents that could match the entry, but the Stage-1 classifier assigns them all to other entries (or "unknown"). This is a classifier quality issue, not a fundamental observability issue. Classifier-blind entries are not frame-blind; they may become measurable with an improved classifier or Stage-2 adjudication.

**Why the distinction matters:** frame-blind entries are reported as `verdict=frame_blind` and are excluded from measurability counts. Classifier-blind entries are included in the measurable set but may have low observed prevalence. Conflating the two would undercount measurable entries and overstate corpus limitations.

## Beta(1,1) uninformative prior behavior

**Beta(1,1)** is the uniform distribution over [0,1]. When used as the prior for recall or precision in the calibration model, it expresses complete ignorance — any value from 0% to 100% recall is equally plausible before seeing gold-set data.

**Practical consequence:** with a small gold set (N < 20), the posterior is dominated by the prior near its boundaries. A classifier with 0/5 correct gold-set matches will have a posterior that still assigns meaningful probability to recall > 0.5. This is correct behavior — the prior is uninformative — but it means small gold sets produce wide posteriors for recall/precision, which propagates to wide uncertainty in de-biased counts.

**When to use a more informative prior:** if you have calibration data from a prior cycle on the same classifier, use it to construct a Beta(a, b) with a > 1 and b > 1. Commit the prior parameters to `PreregManifest.calibration_prior` before the cycle starts.

## Kappa N/A branch (fewer measurable entries than `meaningful_kappa_n`)

**Trigger:** if the number of measurable entries (entries with `verdict != frame_blind`) is fewer than `meaningful_kappa_n` (default: 5), kappa is reported as `None` with `kappa_na_reason="insufficient_measurable_entries"`.

**Why not compute kappa anyway:** with fewer than ~5 entries, quadratic-weighted kappa is numerically unstable and statistically uninterpretable. The tier-boundary structure loses meaning when there are fewer entries than the first tier boundary.

**What to report:** the report surface shows `kappa: null` and the reason string. This is not a failure — it is an honest statement that concordance cannot be quantified for this corpus/taxonomy combination. The coverage and measurability statistics are still valid and reported.

**Recovery path:** add more measurable entries (by improving the classifier, expanding the corpus scope, or revising the rubric) until `n_measurable >= meaningful_kappa_n`.

## OverlapWeights self-loop rejection

**Rule:** `overlap_weights[i][i]` (an entry overlapping with itself) is always rejected with `OverlapWeightsError: self-loop`. Self-loops are undefined in the de-biasing model — they would count an incident twice for the same entry.

**Common mistake:** declaring `overlap_weights["A1"]["A1"] = 1.0` to mean "all A1 incidents are fully A1" is wrong. The absence of an overlap weight row for A1 already means all A1 incidents are 100% attributed to A1. Only cross-entry weights are meaningful.
