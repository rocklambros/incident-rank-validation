# Design: relock the blended Top-10 to the probabilistic method

Date: 2026-07-05
Version: 3 (hardened through two adversarial-premortem rounds)
Status: approved design, pre-implementation
Scope owner: Rock Lambros (project Co-Lead)

Two premortem rounds shaped this spec. Round 1 (on v1) surfaced the fit-to-target
transform and the authority overclaim; round 2 (on v2) confirmed the engineering
remediations closed and found that v2's tier fix collided with the movers table, its
transform justification was a non-sequitur, its committed "accepted" language made the
authority problem more citable, and its integrity check covered the arrays but not the
labels/crosswalk. Scope-owner decisions folded in: keep the linear-lambda order on an
honest defended-reconstruction basis (not a false "principled" claim); present the
ranking as three tiers with the Misinformation finding relocated to the data-witness
section; describe the method on the public preprint as an AUTHORIAL ADOPTION, not an
OWASP institutional sign-off, with the executive decision recorded in an internal note;
keep every standing disclosure. Section 13 traces each premortem finding to its fix.

## 1. Context and decision

The project leads adopt the probabilistic blend as the analytical METHOD for the 2026
blended Top-10, an exploratory report. Rock, as Co-Lead, made this an executive decision
on 2026-07-05 to finalize the analysis; the working group's standing pre-publication
review is the backstop. The public preprint describes this as the authors' adoption of a
method, NOT an OWASP endorsement of the method or the ranking. Every standing disclosure
holds unchanged: single-author gold set, interim reviewers, non-publishable status, weak
agreement (Cohen's kappa ~0.20, interval crossing zero). The internal decision note
(workstream 5.3) records the executive decision; the external artifacts never claim
institutional sign-off.

The prior probabilistic analysis ran in a since-deleted session scratchpad and was never
committed. Per `STYLE-GUIDE.md`, published numbers come from the notebook's own
computation, never hand-typed. The method is reimplemented as a tested engine module, and
the reconstruction prototype plus its knob-sweep are committed as provenance (section 11),
so the algorithm is auditable rather than remembered.

## 2. Guardrails (do not touch)

- Figure styling, sizing, layout attributes; `figure-layout.lua`; `arxiv-template.latex`.
- The build pipeline (`tools/build_preprint.py`); generated `.tex` / `.pdf` never hand-edited.
- Every Part II data-witness figure (dumbbell, bump/slope, ridge, tier donut, confusion
  heatmap, precision bars/posteriors, paired dots, theme bars, out-of-scope treemap,
  Sankey, 3x3 matrix). The blend scale does not touch the incident-data-alone analysis.
- The calibrated thesis framing: weak agreement, robust ranking. Never "validates" / "first-ever".
- Every standing disclosure (preprint Limitations and Scope, `REVIEWERS.md`, the prereg
  manifest keep their interim / single-author / non-publishable state).

## 3. The algorithm (authoritative spec)

Inputs, all committed:
- `projects/owasp-llm/cycles/2026/infer/lambda_samples.npy` — (16000, 20) latent
  incidence (lambda) posterior draws, the recall/precision-corrected rates from the
  negative-binomial measurement-error model.
- `projects/owasp-llm/baselines/2026/vote_rank_samples.npy` — (5000, 20) vote-rank
  posterior draws, integer ranks 1..20 (lower = more important).
- Column order for BOTH arrays is `inference_summary.json["entry_ids"]` ==
  `rankings_baselines.json["entry_ids"]` == the vote fixture
  `tests/unit/fixtures/vote_entry_ids_2026.json` (assert all three equal, element-for-
  element, at runtime). Order: `LLM01..LLM10, NEW-ITSCD, NEW-MA, NEW-MSDA, NEW-MTIE,
  NEW-PMP, NEW-WLA, ROLL-CFAS, ROLL-CMSB, ROLL-LAPTF, ROLL-SICG`. **Landmine: NOT
  taxonomy.json's raw order.** Map by the manifest; never by position.
- Rollup crosswalk read from `taxonomy.json` `rolled_into`, and value-pinned in the module
  against a committed reference (assert the four edges): ROLL-CMSB->LLM01, ROLL-LAPTF->LLM03,
  ROLL-CFAS->LLM04, ROLL-SICG->LLM05.
- Frame-blind set = `rankings_baselines.json["not_measurable"]` = {LLM04, LLM08, LLM10}.
- Weights: 0.75 vote, 0.25 data. Seed: `project.toml` `prng_seed = 20260520`.

Steps:
1. Pair draws. Exact scheme, pinned so the numbers are reproducible off the lockfile:
   `rng = numpy.random.default_rng(20260520); Li = rng.integers(0, 16000, size=16000);
   Vi = rng.integers(0, 5000, size=16000)` (Li drawn before Vi, two `.integers` calls,
   not a single 2-D call). Pair `lambda[Li]` with `vote[Vi]`. The order is stable to N and
   seed; the exact P(top-k) depend on this scheme, so it is pinned here and cross-checked
   against the committed prototype in CI.
2. Fold to the 10 incumbents:
   - Data axis (sum-prevalence, corrected rates add): `lambda_folded[parent] += lambda[child]`.
   - Vote axis (min-rank): `vote_folded[parent] = min(vote[parent], vote[child])`.
3. Per draw, z-score across all 10 incumbents on each axis:
   - `data_score = zscore(lambda_folded)`, linear scale. Honest justification (defended
     reconstruction, not a purity claim): the corrected incidence rates are standardized on
     their native additive scale, so the 0.25 data tug reflects absolute prevalence
     differences; a log transform would encode a different, unstated multiplicative-
     importance assumption. Linear is also the transform that reproduces the recorded
     result; the original computation is unrecoverable, so this is a defended reconstruction,
     disclosed as such. The committed knob-sweep (section 11) shows a log transform swaps
     only positions 7-8, both inside the unordered tail tier (section 4), so no ordered
     claim in the report depends on the choice.
   - `vote_score = zscore(-vote_folded)`.
4. Blend per draw: `blended = 0.75*vote_score + 0.25*data_score`. Frame-blind entries drop
   the data term (`0.75->1.0` vote, `0.25->0.0` data) for {LLM04, LLM08, LLM10}, so they are
   placed by vote alone. Two disclosures the methodology text must carry (premortem F9):
   the frame-blind entries' corrected rates still enter the all-10 z-score normalization of
   the measurable entries (verified order-neutral in the sweep, and the P(top-k) shift ~0.03
   under the alternative measurable-only population — disclose this sensitivity), and
   dropping the data term is placement-affecting (it seats LLM04 at #5 rather than the #4 it
   holds if kept at 0.25), not neutral. Show the keep-at-0.25 alternative for comparison.
5. Rank the 10 by `blended` descending per draw, deterministic `(score, entry_id)` tie-break.
6. Aggregate over the 16000 draws: position distribution per entry; point order by mean
   position with an `entry_id` secondary sort key (stable); P(top-3), P(top-5);
   5th-95th-percentile position interval.

## 3a. Integrity and provenance requirements

- The module verifies the sha256 of EVERY load-bearing input against a committed manifest
  before computing (reuse `engine/snapshot/hashing.py::verify_snapshot_hash`): both `.npy`
  arrays, `taxonomy.json` (crosswalk), `inference_summary.json` and `rankings_baselines.json`
  (labels + frame-blind set), and the vote entry-id fixture. Add the vote array and the
  label/crosswalk sources to a `generated_from`-style manifest with `sha256` and `shape`.
- The module asserts, at runtime, the inference<->vote<->taxonomy entry-id triangle equal,
  the four crosswalk edges by value, and `shape[1] == 20`, before mapping. A permuted-input
  unit test feeds a shuffled array/crosswalk and asserts failure.
- Commit a golden output (order, P(top-k) array, position intervals at the pinned seed/N)
  and a multi-seed stability test that loops >= 20 seeds and asserts (a) top-5 ordinal
  invariance and (b) the 6-10 tail as a set, matching the tier framing (section 4).
- Wire the committed prototype into CI as a second implementation; assert `engine.order ==
  prototype.order`, a cross-implementation anchor the author does not edit alongside the engine.
- The manifest and golden are commit-anchored, not signed. This closes the accident and
  naive-tamper classes; a commit-access adversary is out of scope for an internal research
  tool, disclosed here (consistent with the Plan-8d "no signer gate, right-sized" decision).
- Compute and build are offline and pure (no network egress). Keep it so.

## 4. Verification oracle and the three-tier presentation

The engine reproduces, from the committed arrays:
- Point order (sort key and engine result): `LLM01, LLM02, LLM06, LLM03, LLM04, LLM10,
  LLM09, LLM07, LLM08, LLM05`.
- Top-5 ordinal order stable across >= 20 seeds; the 6-10 tail stable as a set.
- P(top-3): LLM01 ~0.99, LLM02 ~0.95 (recompute the golden as authoritative; the ~ values
  are orientation and carry the section-3-step-4 population sensitivity).

Presentation as THREE TIERS (premortem R2-1, R2-2, F4, F5; scope-owner decision):
- **Tier 1, the co-leading pair:** LLM01 and LLM02. State that the two blends disagree on
  which is #1 (rank-space LLM02, probabilistic LLM01) and their intervals overlap; present
  them as co-leading, not a crisp #1 over #2.
- **Tier 2, the tied band:** LLM06, LLM03, LLM04, with overlapping intervals. LLM06 (+3 from
  its published position) is the one defensible mover and may be stated as such.
- **Tier 3, the unordered tail:** LLM10, LLM09, LLM07, LLM08, LLM05 (P(top-5) ~ 0 for all).
  No prose, table, or figure assigns a definite rank or "+N" mover to a tail entry. The
  Misinformation-underrated finding does NOT live here as a blended-rank claim; it is a
  data-witness result in Part II (section 7). Frame-blind movement (LLM10) is never narrated
  as a data signal, since the data term is dropped for it.
The point order remains the engine's sort key and internal result; the narrative and the
figure present the three tiers, not ten crisp ranks and not a "+N" movers table.

## 5. Workstreams and file changes

1. **Engine module** — `engine/decide/blend.py` (new). Pure functions; typed; seed and N
   injected; the exact draw scheme of 3 step 1; the integrity checks of 3a; a frozen result
   dataclass returning order, per-entry mean position, position distribution, P(top-k),
   position interval, tier assignment.
2. **Tests** — `tests/unit/test_probabilistic_blend.py` (new). Assert sections 4 and 3a:
   point order, top-5 ordinal + tail-set multi-seed invariance, P(top-k) within a stated
   Monte-Carlo tolerance, all-10 z-score population with the disclosed alternative, fold
   correctness on a toy input, permuted-input and permuted-crosswalk failure, entry-id
   triangle equality, deterministic tie-break, crosswalk read+value-pinned from taxonomy,
   cross-implementation order match vs the committed prototype. `uv run mypy engine tests`
   + `uv run ruff check .` before every commit.
3. **Internal decision note** — commit a dated record (`docs/` or a `REVIEWERS.md` addendum)
   stating that Rock, as Co-Lead, made the executive decision on 2026-07-05 to adopt the
   probabilistic blend as the analytical method for this exploratory report, that this is an
   authorial method choice and NOT an OWASP institutional endorsement, and that the standing
   pre-publication working-group review applies. This is an internal record; the external
   artifacts do not cite it as institutional sign-off (premortem R2-3, F2).
4. **Notebook** — replace the rank-space compute (cell 45) with an `engine.decide.blend`
   call; regenerate `rank_change_2025_2026.png` (or a tier figure) with the new tiers and
   identical styling; add the two uncertainty figures (section 6); derive all computed cell
   outputs and figure annotations from the engine result, never re-typed literals; rewrite
   the prose (markdown) cells, including the inverted #1/#2 ordinals and the tier framing.
   The preprint `.md` exports from these markdown cells, so their numbers are static prose,
   protected by the guard in 5.5.
5. **Consistency guard** — expand cell 46 to assert, against the engine result, the tier
   assignments and every load-bearing number in any prose cell or figure, with Monte-Carlo
   tolerance. The build executes the notebook, so a drifted number fails the build.
6. **CI** — add a job that installs `uv sync --extra preprint`, executes the notebook, and
   asserts the guard cell ran, so the guard is enforced in CI, not only locally (premortem
   R2-6). Skip pandoc/xelatex in CI (heavy); the load-bearing check is notebook-execute + guard.
7. **New figures** (section 6).
8. **Preprint `.md`** (section 8).
9. **Methodology doc** — `docs/BLENDED-TOP10-METHODOLOGY.md`: rewrite header status, 4.3
   (probabilistic adopted as the method; do NOT claim the RARR result "resolved" the
   magnitude deferral — state that RARR established ordinal robustness to classifier choice,
   that proceeding on the current committed lambda is the scope-owner's decision, and that
   retrain-sensitivity of magnitudes is the parked tail risk of section 12; reconcile with
   section 12, no self-contradiction), 4.4 (sum-prevalence fold), 4.5 (frame-blind-drop as
   placement-affecting, alternative shown), 5 (the three-tier ranking with the uncertainty
   layer), AND section 9 (the Gamma deck prompt — update its tables/prose to the new tiers;
   it currently keeps the old order). Label the change a documented amendment (premortem
   R2-4, R2-7, F8).
10. **STYLE-GUIDE.md** — add the two writing rules (section 9-of-this-spec) AND update the
    "Key numbers" block (still states the old rank-space formula and movers).
11. **Sync the mirror** — update or regenerate `notebooks/narrative/2026_top_10_*.md` so it
    does not keep a stale order; add a CI grep-consistency check across the external docs
    (premortem R2-7).
12. **Provenance + rollback** — stamp `method: probabilistic-blend` + short commit in
    `front_matter.md` (title page); tag the last-known-good interim state before merge; add a
    one-line rollback note to section 10 (premortem R2-9).
13. **Rebuild + verify** — confirm the notebook executes clean end to end under
    `build_preprint.py`; the plotly/kaleido refactor to `narrative_charts._plotly_write_image`
    is IMPERATIVE (BUILD.md records the break persists), not conditional; record
    pandoc/xelatex/TeXLive versions in BUILD.md; add a build check that greps the generated
    `.tex` for the engine-emitted tier/number strings AND the expected Greek/math glyphs
    (premortem R2-6, R2-9). Do not declare success on green engine tests alone.

## 6. New figures (existing figure style, saved to `notebooks/preprint/figures/`)

- `blend_position_intervals.png` — each risk as a dot at its mean position with a bar
  spanning the 5th-95th-percentile position, visually grouped into the three tiers.
- `blend_top_k_probs.png` — companion grouped bars of P(top-3) and P(top-5) per risk.

## 8. Preprint prose change map

- Part I "The 0.75 / 0.25 blend" — the blend combines two posterior distributions in score
  space; the 0.75/0.25 weighting is unchanged, the SCALE (rank -> score) and the data-axis
  fold (min-rank -> sum-prevalence) change; say so plainly.
- "The 2026 blended Top 10" — present the THREE TIERS (section 4), the uncertainty layer as
  the lead, the two new figures, sum-prevalence fold, frame-blind-drop as placement-
  affecting, the co-leading-pair #1 framing, and a one-paragraph rank-space robustness lens
  (Kendall tau 0.87 for the bulk order) WITHOUT pairing it with a "so the headline is
  robust" reassurance (the headline #1 is the method-dependent exception; state that
  plainly). No "+N" movers table for tail entries.
- Misinformation — the "incident data ranks Misinformation high / the record says it is
  underrated" finding lives in Part II (Act 8) as a data-witness result (the transform-
  independent lambda-vs-vote gap and the disagreement flag), NOT as a blended-rank mover
  (premortem R2-1, R2-9c).
- Describe the position distribution as independent draws from the two posteriors, not a
  "joint posterior." State that P(top-k) reflects input sampling under a fixed method, is
  dominated by the 0.75 vote weight where that holds, and carries the ~0.03 z-population
  sensitivity of section 3 step 4.
- Point-of-use hedges: the "99 percent" Misinformation flag is P(the two signals disagree),
  and LLM09's signal rests on the ai-harm stratum with a flat precision prior; carry the
  hedge where the number is stated.
- Authority: the preprint states the authors ADOPT the method for this exploratory report;
  it never claims OWASP working-group sign-off. The Scope and Limitations sections are
  unchanged (premortem R2-3).
- Glossary — add probabilistic blend, distribution over positions, credible interval over
  rank (noting it presumes a fixed target rank, softer for frame-blind entries), P(top-k)
  (vote-weight caveat), sum-prevalence fold, frame-blind drop, Kendall tau, tied tier.
  Update the "0.75 / 0.25 blend" entry to score-space.

New novice sidebars on first use: distribution over positions; credible interval over a
rank; P(top-k); why a distribution beats a single rank number.

## 9. Writing standards

Binding `STYLE-GUIDE.md` governs. Add and apply: no sentence starts with a conjunction; no
antithesis / contrast framing. Keep the existing banned-slop list, the no-fluff rule, one
sidebar per term, lead-with-the-claim structure.

## 10. Out of scope

- Re-running inference or the vote bootstrap. Committed posterior samples are the inputs.
- Changing the 0.75/0.25 weights, the taxonomy, or the RARR conclusions.
- Any Part II data-witness figure or the incident-derived ranking.
- Flipping the report to publishable, or removing any interim / single-author disclosure.
Rollback: the interim state is tagged last-known-good before merge; recovery is a revert to
that tag plus a rebuild that reproduces the tagged artifact.

## 11. Committed provenance

Move the reconstruction prototype into the repo (a `docs/` provenance note or an
`engine/decide/` companion) as the record of how the method was reconstructed, including the
knob-sweep that shows which orderings each transform/fold/drop setting produces. It is NOT
deleted. The engine module is a clean TDD reimplementation; the prototype is evidence and
the CI cross-implementation anchor (3a), not a runtime dependency.

## 12. Residual and tail risk

- Transform provenance: a defended reconstruction, not recovered ground truth. Mitigated by
  honest disclosure (section 3 step 3), the unordered tail, and the committed sweep. Accepted.
- Manifest/golden self-certification: closes the accident and naive-tamper classes, not a
  commit-access adversary; out of scope for an internal tool, disclosed (3a).
- Tail risk (parked): a future recall-corrected retrain moves lambda and the order shifts.
  Critical, currently Unlikely on the RARR ordinal result. RARR tests classifier robustness,
  not magnitude stability under retrain, so this is NOT claimed resolved; proceeding now is
  the scope-owner's decision. Re-evaluate before the engine-upgrade cycle runs. Do not delete.

## 13. Premortem traceability (both rounds)

Round 1: F1 fit-to-target -> 3 step 3, 4 (tiers), 11. F2 authority -> 1, 5.3, 8. F3 drift ->
5.4, 5.5, 5.10, 11-of-workstreams. F4 uncertainty-vs-list -> 4, 6, 8. F5 #1-hinges-on-method
-> 4, 8. F6 integrity -> 3a. F7 build -> 5.13. F8 methodology record -> 5.9. F9 frame-blind
-> 3 step 4, 5.9. F10 joint-posterior -> 8. F11 novice point-of-use -> 8.
Round 2: R2-1 tier/movers -> 4, 8. R2-2 justification non-sequitur -> 3 step 3. R2-3
authority language -> 1, 5.3, 8. R2-4 RARR category error -> 5.9, 12. R2-5 integrity
completeness -> 3a. R2-6 CI guard -> 5.6. R2-7 drift copies (deck prompt, mirror) -> 5.9,
5.11. R2-8 z-population P(top-k) sensitivity -> 3 step 4, 8. R2-9 provenance/rollback/build/
mover-label/draw-scheme/toolchain/tail-test -> 5.12, 5.13, 3 step 1, 4.
Cleared in round 2: linear-axis-single-entry-dominance (three-way cluster), point-order
aggregation consistency (verified consistent).
