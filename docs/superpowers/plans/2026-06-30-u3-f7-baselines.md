# U3: F7 Baselines + Prospective Power Implementation Plan (v2 — post-premortem)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Reproduce + FREEZE the **as-shipped 2026 "previous ranking"** (the λ·size *incidence* ranking over all 20 inference∩vote entries — literally what `cycles/2026/results/concordance.json` contains) as a committed, deterministically-reproducible artifact, plus a bare-λ **sensitivity disclosure** and a **prospective** power statement — the concrete realization of the **preserve-previous-rankings mandate** and the "previous" side of the U9 previous-vs-new report.

**Architecture:** Pure functions produce each value (inputs→value, no I/O); a thin freeze CLI persists them to a NEW committed tree `projects/owasp-llm/baselines/2026/` (NEVER inside byte-immutable `cycles/2026/`). The frozen previous ranking + kappa is **byte-pinned to `cycles/2026/results/concordance.json`** (not to a hand-typed constant). A committed RAW vote source (xlsx + a frozen `respondent_rankings.npy` 29×20) makes reproduction **non-circular**. Offline only — reads existing frozen `lambda_samples.npy`, invokes no sampler.

**Tech Stack:** numpy/scipy (pinned; no new deps), pytest. No R, ever.

---

## ⚑ PREMORTEM CORRECTION (F1 — CODE-CONFIRMED CRITICAL; changes the deliverable)

The v1 plan **misread the engine.** Verified by reading `engine/decide/concordance.py` + `cycles/2026/results/concordance.json`:

- `compute_concordance` builds `common` at **concordance.py:178 with NO measurability filter** → the frozen kappa is computed over **all 20** inference∩vote entries (`concordance.json`: `total_count=20`, `measurable_count=17`).
- Ranking is via **`_ranks_from_incidence`** (concordance.py:198), i.e. **λ·Σstratum_sizes**. **`_ranks_from_lambda` (concordance.py:54) is DEAD CODE** — it never produced a published number.
- Tier boundaries are recomputed from `n_common`: `(n_common//3, 2·n_common//3)` = **(6, 12)** for n=20 (concordance.py:191). **NOT (5,10).**
- On 2026 data, bare-λ and λ·size rankings **COINCIDE** → **0.0 method-kappa delta**.
- Vote posterior is bootstrapped at **n_bootstrap=5000, seed=`manifest.prng_seed` (=20260520)** (`pipeline.py`), so the frozen vote array is **5000×20**, NOT 16000×20. (`lambda_samples.npy` is 16000×20; `n_draws=min(16000,5000)=5000`.)

**Consequences that invalidated v1:** there is **no separate "bare-λ 0.20 baseline over 17 entries."** The v1 T6 assertion (`kappa==0.2029` AND `n_common==17`) is **code-confirmed UNSATISFIABLE**. The v1 "two baselines" framing is wrong.

**Tail-risk mitigation (NON-NEGOTIABLE):** the four tracked cycle inputs make an integration test *run* in CI while the v1 assertions are unsatisfiable — under schedule pressure the path of least resistance is to loosen the float pin to `0.1185` or inconsistently drop the 17-restriction, silently canonizing a "previous ranking" that was **never shipped** into `rankings_baselines.json`, which then flows immutably into the U9 report the user reads. **Therefore: pin the frozen values by asserting BYTE-EQUALITY against the committed `cycles/2026/results/concordance.json`.** The "previous ranking" *is* what that file contains.

---

## 🧑 FOR-THE-HUMAN (firm resolutions of the premortem's human-calls — documented for later review)

- **F14 — which kappa to freeze.** Freeze the **AS-SHIPPED 20-entry** ranking + `kappa=0.2028985507246377`, `CI=[-0.1594…, 0.5652…]` — it is what was PUBLISHED and is the "previous ranking" to preserve for the user's compare/contrast. Byte-pin to `concordance.json`. **ALSO** compute + disclose the **17-measurable-subset kappa (~0.1185)** as a SECONDARY footnote, and record the **STANDING_CAVEAT CONTRADICTION**: `concordance.py:48` says "computed over the measurable subset only," which contradicts the 20-entry shipped number. Do not resolve silently — surface both.
- **F2/F1 — baseline identity.** Reframe from "two baselines (bare-λ vs λ·size)" to **ONE frozen "previous ranking" = the as-shipped 2026 λ·size incidence ranking**, PLUS a **DISCLOSED bare-λ sensitivity** (compute `_ranks_from_lambda`; disclose it coincides = **0.0 method delta** on 2026 — the honest "size-weighting changed nothing on 2026" finding). **U9 MUST NOT credit any kappa gain to the ranking method.**
- **F5 — power β-term.** Pre-register **1−β = 0.80**; `n_required = ceil(σ²·(z_{1-α/2}+z_{1-β})²/κ²)`. The v1 formula (`z_{1-α/2}` only) was **50% power mislabeled "confidence 0.95."**
- **F6/F21/F25 — surrogate variance.** DISCLOSE (in PROVENANCE + a schema disclosure field) that the closed-form Fleiss-Cohen-Everitt weighted-kappa variance is a **COARSE DESIGN-STAGE SURROGATE**, DISTINCT from and **does NOT govern** the reported paired-draw bootstrap CI. The power statement is a rough sizing guide, not a claim about the reported CI. Normal-approx at n≈20 with ranked/stratum dependence violates iid — flag the violated assumptions.
- **F15/F17 — criterion + provenance.** Label **κ=0.40 as a PROSPECTIVE target for FUTURE cycles, REGISTERED 2026-06-30, explicitly NOT a 2026 pre-registration** (0.20 was observed first). The criterion is **"would the design-n CI exclude 0" (κ>0)**, honestly labeled — **NOT** κ≥0.40 non-inferiority. **Remove the "adjust at U5" escape hatch** (any change requires a dated amendment note).
- **F18 — bridge coherence.** Relabel the previous-vs-new compare as an **OMNIBUS** (data + method + recall-correction + config ALL differ between the pre-RARR 2026 posteriors and the new run) — **NOT** a clean method-only bridge.
- **F7 — power "n".** `n` is a **fixed ~20-entry taxonomy, not growable**; report `n_required` as a **structural-adequacy verdict**, not a "collect more taxonomy entries" instruction.

---

## Global Constraints
- No AI/Claude/Anthropic attribution. CI gate: `uv run ruff check .` + `uv run mypy engine tests` before every commit; FULL `uv run pytest -q` before any push. F4/label pin green. Two-stage review + opus whole-increment; push PR #22; CI green.
- **NEVER write under `cycles/2026/`** (byte-immutable). All new artifacts land in `projects/owasp-llm/baselines/2026/`. The freeze CLI MUST `raise` if the output path **`Path.resolve()`s inside any `cycles/` dir** (collapse `..`, follow symlinks; `commonpath`/parent-name check). Verify `git status` clean under `cycles/2026/` after freezing.
- **Offline:** no live run, no NUTS. Only READ existing posteriors/labels. **No new third-party dependencies** (power math is closed-form numpy/scipy). No R.
- **Provisional gate:** output stays provisional (per the no-R/Python-oracle memory) until **both** non-vacuous cross-checks agree — the engine-vs-oracle incidence-ranking check (T4) **and** the Monte-Carlo power-variance check (T6). Only then is the freeze publishable.
- **Preserve previous rankings:** this unit IS that mandate — the frozen previous ranking is the "previous" side, byte-pinned to `concordance.json`; nothing here recomputes/mutates the new primary.

## Design decisions (user away — authorized; recommendation adopted, documented for review)
- **D1 — artifact location:** `projects/owasp-llm/baselines/2026/` with `rankings_baselines.json` as the single U9-consumed manifest (+ `lambda_median.npy`, `vote_rank_samples.npy`, `respondent_rankings.npy`, the raw xlsx, `SHA256SUMS`, `PROVENANCE.md`, `reproduce.py`).
- **D2 — prospective power:** target **κ = 0.40**, **α two-sided at 0.95 (z_{0.975})**, **power 1−β = 0.80 (z_{0.80})**, criterion **"design-n CI excludes 0" (κ>0)**. **PROSPECTIVE / FUTURE-CYCLE, registered 2026-06-30, NOT a 2026 pre-registration** (0.20 observed first). No U5 escape hatch — changes require a dated amendment. [FOR-THE-HUMAN, above.]
- **D3 — power method:** closed-form quadratic-weighted-kappa asymptotic variance (name the exact **Fleiss-Cohen-Everitt** variant + register the full **H1 joint cell-probability model**, since σ² depends on the joint, not marginals) + normal-approx sample-size solver. **DISCLOSED as a coarse design-stage surrogate** distinct from the paired-draw bootstrap CI (F6/F16/F21). Cross-checked by a **Monte-Carlo variance estimate** (T6), not only an algebraic re-arrangement.
- **D4 — "previous ranking" identity:** the exact as-shipped 2026 **λ·size incidence ranking over 20 entries**, byte-pinned to `concordance.json`. The pin is against `cycles/2026/classify/labeled_incidents.json` (SHA256) that produced the frozen `lambda_samples.npy`. Record any pre-relabel commit SHA in PROVENANCE as `labels_prebakeoff_ref` but do NOT use it as the frozen input. Bridge is labeled **OMNIBUS** (F18): the previous posteriors are recall-UNcorrected/pre-RARR and differ from the new run in data+method+recall+config.
- **D5 — vote determinism (non-circular):** freeze `vote_rank_samples.npy` at **5000×20** (`n_bootstrap=5000`) and record **seed=20260520**. **COMMIT the RAW votes** — the xlsx AND a frozen `respondent_rankings.npy` (29×20) + SHA256 — so `reproduce.py` **bootstraps from the RAW source** (non-circular). SHA256 byte-match of the frozen `.npy` is a **SECONDARY** check only.
- **D6 — manifest power fields at `schema_version >= 4`:** implement as a **NEW `if schema_version < 4:` pop-block placed ABOVE the `== 1` block** (do NOT extend `== 1` and do NOT extend the `< 3` block). Additive, defaulted, active-but-unlocked guard mirrors the F6 `__post_init__` pattern. Requires **v1 + v2 + v3** golden-hash tests (only a v2 golden exists today).

## Verified facts (from U3 research + premortem code-confirmation)
- `lambda_samples.npy` = **16000×20**. Vote bootstrap = **5000×20** (`pipeline.py`: `n_bootstrap=5000, seed=manifest.prng_seed=20260520`). `n_draws = min(16000, 5000) = 5000`.
- Frozen number (from `concordance.json`): `weighted_kappa_median=0.2028985507246377`, `weighted_kappa_ci=[-0.1594202898550725, 0.5652173913043478]`, `measurable_count=17`, `total_count=20`, `ci_method="paired_draw_percentile"`.
- The frozen kappa is the **incidence (λ·size) kappa over 20 entries** via `_ranks_from_incidence` (concordance.py:198); `_ranks_from_lambda` (concordance.py:54) is **dead code**. Tiers **(6,12)**.
- On 2026 data, bare-λ ranking == λ·size ranking → **0.0 method delta** (disclosed, never credited).
- Baseline cross-check reuses `engine.verify.oracle.oracle_incidence_ranking` + `oracle_incidence_intervals` + `engine.verify.check._build_strata` verbatim (engine-vs-oracle, F12 — a REAL gate, not oracle-vs-itself).
- `meaningful_kappa_n` / target-κ / power fields are NOT persisted anywhere (introduced here via D6).
- The `vote/` dir (xlsx) is currently **UNTRACKED** in git; `lambda_samples.npy`, `inference_summary.json`, `labeled_incidents.json`, `concordance.json` ARE tracked → committing the raw votes (D5/T8) is what makes reproduction non-circular and CI-runnable.

## Artifact tree (new, committed, outside cycles/)
```
projects/owasp-llm/baselines/2026/
  rankings_baselines.json     # THE U9 contract (schema below)
  lambda_median.npy           # (20,) float64 median over 16000 draws
  vote_rank_samples.npy       # (5000,20) frozen vote ranks (D5)
  respondent_rankings.npy     # (29,20) RAW respondent matrix — non-circular source (D5)
  votes_source.xlsx           # committed raw vote workbook (D5/F4)
  SHA256SUMS                  # write-once integrity manifest (F10)
  PROVENANCE.md               # cycle source paths + SHA256 + all disclosures
  reproduce.py                # standalone verifier (+ --verify-provenance)
```
`rankings_baselines.json` schema (U9 consumes):
```
{
  artifact_type, schema_version, cycle,
  generated_from{ lambda_samples, inference_summary, labeled_incidents,
                  respondent_rankings, concordance_json: {path(repo-relative), shape, sha256(, seed)} },
  entry_ids[20], measurable_entry_ids[17], not_measurable[3],
  previous_ranking{                       # THE frozen as-shipped ranking (byte-pinned)
    method:"incidence_lambda_size", function:"_ranks_from_incidence",
    tier_boundaries:[6,12], n_common:20, bootstrap_draws:5000, bootstrap_seed:20260520,
    ranking[20], kappa_median:0.2028985507246377, kappa_ci:[-0.1594…,0.5652…],
    kappa_ci_method:"paired_draw_percentile",
    byte_pinned_to:"cycles/2026/results/concordance.json"
  },
  bare_lambda_sensitivity{                # DISCLOSED, never credited
    method:"_ranks_from_lambda", function:"_ranks_from_lambda",
    ranking[20], method_kappa_delta:0.0,
    disclosure:"size-weighting changed nothing on 2026; not credited as a method gain"
  },
  secondary_measurable_subset{            # F14 footnote
    measurable_kappa_median:~0.1185, n_measurable:17,
    standing_caveat_contradiction:"concordance.py:48 claims 'measurable subset only' but shipped kappa is over 20"
  },
  prospective_power{
    target_kappa:0.40, alpha_two_sided:0.05, power_1_minus_beta:0.80,
    criterion:"design_n_ci_excludes_zero", registered:"2026-06-30",
    scope:"prospective_future_cycles", h1_joint_model:{…}, method:"fleiss_cohen_everitt_wk_variance",
    variance_disclosure:"coarse design-stage surrogate; distinct from and does not govern the reported paired-draw bootstrap CI",
    n_required, current_n:20, structural_note:"n is a fixed ~20-entry taxonomy, not growable — structural-adequacy verdict",
    excludes_zero_at_current_n, stage:"design"
  },
  disclosures{ incidence_kappa:true, method_delta_zero:true, ci_spans_zero:true, omnibus_bridge:true }
}
```

## Tasks (TDD, bite-sized)
### T0 — Synthetic fixtures (no cycle files)
- [ ] `tests/unit/fixtures/` builder yielding a tiny `lambda_samples (200×4)`, `respondent_rankings (5×4)`, `vote_rank_samples (200×4)`, `entry_ids`, `entry_strata`, `stratum_sizes` with **hand-computed incidence/kappa AND a case where bare-λ ≠ λ·size** (so the delta path is exercised, not just the 0.0 coincidence). Commit.
### T1 — Previous-ranking reproducer `compute_previous_ranking(...)` in `engine/baselines/previous_ranking.py`
- [ ] RED: unit test on fixture asserts ranking + kappa median/CI == hand-computed, using **`_ranks_from_incidence`** + `quadratic_weighted_kappa` over the **full common set (no measurability filter)**, tiers `(n//3, 2n//3)`. GREEN: thin loop mirroring `compute_concordance` (concordance.py:193-213), `n_draws=min(len(lambda),len(vote))`, `np.argsort` with `kind='stable'`. Signature `(lambda_samples, vote_rank_samples, inf_entry_ids, vote_entry_ids, entry_strata, stratum_sizes) -> (ranking, median, ci_lo, ci_hi, n_common, tier_boundaries)`. Commit.
### T2 — Bare-λ sensitivity + method-delta `compute_bare_lambda_sensitivity(...)` in `engine/baselines/bare_lambda.py`
- [ ] RED: unit test asserts the `_ranks_from_lambda` ranking AND `method_kappa_delta` (bare-λ kappa − incidence kappa); on the fixture's ≠ case delta≠0, and a second fixture where they coincide asserts delta==0.0. GREEN: reuse `_ranks_from_lambda` + T1's loop. Disclosure text baked into the return. **Never credited as a gain.** Commit.
### T3 — Secondary measurable-subset kappa `compute_measurable_subset_kappa(...)`
- [ ] RED: unit test on fixture (with a measurable-id subset) asserts the subset-restricted kappa median. GREEN: T1 loop restricted to measurable ids (slice both lambda cols and vote cols to the subset — F13 alignment). Returns `~0.1185`-analog + the standing-caveat-contradiction note. Commit.
### T4 — Engine-vs-oracle incidence cross-check (non-vacuous gate — F12/F13)
- [ ] RED: `test_baseline_oracle_crosscheck.py` asserts engine `_ranks_from_incidence` (median-λ) **equals** `oracle_incidence_ranking` over the **20-entry** set; a deliberately-perturbed lambda makes them DISAGREE (proves the gate can fail). GREEN: REUSE `oracle_incidence_ranking` + `oracle_incidence_intervals` + `_build_strata` verbatim; confirm 20-column alignment. Commit.
### T5 — Power solver `kappa_sample_size_required(...)` in `engine/decide/prospective_power.py`
- [ ] RED: pin `n_required` vs hand-computed **`n = ceil(σ²·(z_{1-α/2}+z_{1-β})²/κ²)`** (β-term present — F5), σ² = Fleiss-Cohen-Everitt per-item weighted-kappa variance under the registered **H1 joint cell-probability model** (F16). Edge: κ_target≤0 raises; α,β∈(0,1); monotone in κ_target/α/β. GREEN: closed-form (`scipy.stats.norm.ppf`). Commit.
### T6 — Power statement + Monte-Carlo variance cross-check `prospective_power_statement(...)`
- [ ] RED: returns the `prospective_power` dict (β-term, disclosure text, `structural_note`, `stage="design"`); asserts `excludes_zero_at_current_n` at n=20 matches the closed form. **Second RED: Monte-Carlo estimate of σ² from simulated H1 draws agrees with the closed form within tolerance** (F12/F21 — the real independent check; keep output provisional until this passes). GREEN. Commit.
### T7 — Assembler `build_rankings_baselines(...)` in `engine/baselines/freeze.py` (pure, no I/O)
- [ ] RED: schema test asserts every required key incl. `previous_ranking` (kappa `0.2028985507246377` + CI, `n_common:20`, tiers `[6,12]`, `bootstrap_draws:5000`), `bare_lambda_sensitivity.method_kappa_delta`, `secondary_measurable_subset`, `prospective_power`, and **all four `disclosures`**. `generated_from.*.path` stored **repo-relative** (F24). GREEN. Commit.
### T8 — Commit RAW votes + make reproduction non-circular (F4/F11)
- [ ] Copy the raw xlsx → `baselines/2026/votes_source.xlsx`; materialize `respondent_rankings.npy` (29×20) via the existing loader; record SHA256 of both. RED: test loads `respondent_rankings.npy`, bootstraps with `n_bootstrap=5000, seed=20260520`, and reproduces `vote_rank_samples` **from the raw source** (SHA byte-match of the frozen `.npy` is secondary). Commit the raw source + arrays.
### T9 — Freeze CLI `engine/cli/freeze_baselines.py` (hardened)
- [ ] RED: writes to temp dir; asserts all siblings + `SHA256SUMS` created, `generated_from` sha fields populated. **cycles/ guard uses `Path.resolve()`** — T7-style **evasion table** asserts RAISE on `..` traversal, symlink-into-cycles, and relative paths (F8). **Write-once guard:** refuses to overwrite an existing `baselines/2026/` whose content SHA differs from `SHA256SUMS` (F10). Register the command in `main.py`'s CLI group (or test the `python -m engine.cli.freeze_baselines` path — F23). GREEN. Commit.
### T10 — Materialize `baselines/2026/` (freeze BEFORE integration — F19 ordering)
- [ ] Run the T9 CLI once to write the real artifacts; add a CI `SHA256SUMS` compare step. Commit the materialized tree. **This precedes T11 so the integration test never depends on an un-materialized file.**
### T11 — Integration reproduction (byte-equal to concordance.json — F1 tail-risk)
- [ ] RED: `tests/integration/test_f7_baseline_repro.py` loads real `lambda_samples.npy` (16000×20) + `inference_summary.json` + `labeled_incidents.json`, **self-builds** `vote_rank_samples` from committed `respondent_rankings.npy` (seed 20260520, 5000×20 — F19 self-contained). Asserts **`n_common==20`**, tiers **`(6,12)`**, and previous-ranking `kappa`/`ci` **byte-equal to `cycles/2026/results/concordance.json`** (parse the JSON, compare fields — NOT a hand-typed constant), within `atol=1e-9` with `argsort kind='stable'` (F22). Asserts `bare_lambda_sensitivity.method_kappa_delta == 0.0` on 2026. **assert-not-skipped guard** + assert no test writes under `cycles/`. GREEN once T1/T2/T8 correct. Commit.
### T12 — `reproduce.py` + `--verify-provenance` + `PROVENANCE.md`
- [ ] RED: test invokes `reproduce.py` against committed artifacts, re-derives the frozen kappa (bootstrapping from `respondent_rankings.npy`) + validates the internal SHA256SUMS. **`--verify-provenance`** re-hashes the LIVE `cycles/2026` source files vs the pinned `generated_from.*.sha256` (skip-warn if a source is absent — F9). PROVENANCE.md records: cycle source paths+SHA at freeze, the F1 incidence-kappa fact, method-delta 0.0, CI-spans-0, the STANDING_CAVEAT contradiction, the surrogate-variance caveat, and the omnibus-bridge label. GREEN. Commit.
### T13 — Manifest power fields (D6 — F20)
- [ ] RED: `PreregManifest(schema_version=4, prospective_power_target_kappa=0.40, prospective_power_alpha=0.05, prospective_power_1_minus_beta=0.80)` round-trips; a non-default power field at `schema_version<4` RAISES (mirror the F6 `__post_init__` guard); implement a **NEW `if schema_version < 4:` pop-block ABOVE the `== 1` block** (do not extend `== 1` or `< 3`). Golden-hash tests for **v1, v2, AND v3** canonical forms byte-unchanged. GREEN. Commit.
### T14 — Power-math oracle spot-check
- [ ] RED: `test_power_oracle.py` recomputes `n_required` by an INDEPENDENT algebraic arrangement (solve for SE then invert) AND checks it against the T6 Monte-Carlo σ². Scope = formula math only. GREEN. Commit.
### T15 — U9 contract test (consumer guard) + loader
- [ ] RED: `test_u9_contract.py` opens `rankings_baselines.json` via a loader; asserts the exact keys U9 reads: `previous_ranking.kappa_ci`, `previous_ranking.ranking`, `bare_lambda_sensitivity.method_kappa_delta`, `prospective_power.n_required`, `measurable_entry_ids`, `disclosures`. **Asserts U9 cannot credit a kappa gain to the ranking method** (delta is surfaced == 0.0). GREEN: loader in `engine/report/`. Commit.

## Definition of done
ruff/mypy/pytest/semgrep green; T0-T15 pass; `projects/owasp-llm/baselines/2026/` materialized + committed (incl. raw xlsx + `respondent_rankings.npy` + `SHA256SUMS`); `cycles/2026/` byte-unchanged (git status clean under it); no R; no new deps; the previous-ranking kappa/CI **byte-equal to `concordance.json`** (`n_common==20`, tiers `(6,12)`); `method_kappa_delta==0.0` disclosed and never credited; power block carries the β-term + surrogate-variance disclosure + structural-adequacy note; manifest v1/v2/v3 golden hashes unchanged and v4 round-trips; **output provisional until BOTH the engine-vs-oracle incidence check (T4) AND the Monte-Carlo power-variance check (T6) agree.**

## Premortem note
This plan is the v2 revision folding the `adversarial-premortem-complete` run (6 perspectives, F1 Critical code-confirmed). All surviving findings F1–F25 are folded above (F1 architecture correction; F2/F14/F18 disclosures; F3/F4/F11 vote-source; F5/F6/F16/F21/F25 power; F7 structural-n; F8/F10 CLI hardening; F9 verify-provenance; F12/F13 non-vacuous gates; F15/F17 criterion+provenance; F19 ordering; F20 manifest; F22 float-pin; F23/F24 CLI+paths). Human-calls resolved in the FOR-THE-HUMAN block for later review.
