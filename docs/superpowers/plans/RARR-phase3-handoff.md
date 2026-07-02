# RARR Phase-3 Hand-off (autonomous campaign checkpoint — 2026-07-01)

The autonomous campaign drove **all offline-completable Phase-2 work to CI-green**; it has reached the Phase-3 boundary, which is gated on (a) a provisioned RunPod endpoint and (b) genuine pre-registration decisions that are the user's to confirm. This doc is the resume contract for Phase 3.

## What landed this campaign (all CI-green on PR #22)
| Unit | Range | What |
|---|---|---|
| F-B | `68dcc59..27e8161` | labeled_incidents completeness guard (pinned corpus snapshot; hard-raise on truncation) |
| F-C | `12c8bf9..4baf0f9` | CLI integration coverage (shared real-snapshot fixture; premortem caught the universe-drift Critical) |
| U2 | `23a7bbd..3f5e261` | integrity hardening — **F6 redesigned to flag-not-widen** (premortem saved it), goldset_hash threading, robustness §14, F8 strata-disjoint, report gate |
| U3 | `ee42599..a066689` | **previous rankings FROZEN** byte-pinned to `concordance.json` in `projects/owasp-llm/baselines/2026/` + prospective power (premortem caught F1: the shipped κ is λ·size-over-20, not bare-λ-over-17) |
| U4 | `4a35429..45c17c3` | hygiene (docs tracked; curation relocated to `curation/2026/`; engine_version reconciled 1.2.0) |

**Both pre-Phase-3 co-blockers (F-C + F6) are cleared; the gate is open.** The **previous ranking** you asked to preserve is frozen, reproducible (`baselines/2026/reproduce.py`), and integrity-checked (`SHA256SUMS`).

## ⛔ The hard-stop: U8 (the live RunPod bake-off run)
Confirmed unreachable here: `RUNPOD_ENDPOINT_ID` is unset and no serverless endpoint is provisioned (a `runpod` api-key exists in `pass`, but there is no deployed endpoint with the 4 models). **To unblock:** deploy a RunPod serverless endpoint with the model weights loaded and set `RUNPOD_ENDPOINT_ID` (per `engine/classify/runpod_client.py:55`). The live `predict_fn` wiring itself is still a `NotImplementedError` stub at `engine/cli/bakeoff.py:103-108` (inventory #16) — that is buildable offline (below).

## Remaining units (U5–U9) + what each needs
- **U5 — Pre-registration LOCK [needs USER pre-registration decisions].** Lock the RARR manifest with the config grid + the **4th labeling model** (alongside Qwen3-235B / Llama-3.1-405B / DeepSeek-V3); F3b lock bindings (goldset hash, floor-label file, the 4 bake-off constants LOCKBOX_FRACTION/BAKEOFF_ALPHA/MIN_CELL/seed, the winner=None rule, scored-classifier identity in provenance); create the NEW cycle dir `projects/owasp-llm/cycles/2026-rarr/` (never write into byte-immutable `cycles/2026/`); wire the D6 manifest v4 power fields (κ=0.40/0.95/0.80) into the live manifest + the **F6 lock-acceptance check** (inventory-I: the live manifest must be schema≥3 and deliberately set `recall_min_denominator`, or record a written "thin cells left bare" rationale); land the deferred **U2-7 (F3 σ_u-sweep wiring)** if `hierarchical_pooling` is added to `robustness_specs`.
  - **★ RECOMMENDATION-FIRST (for the user): the 4th model.** Recommend a strong, RunPod-deployable open model distinct from the existing three — e.g. **Qwen2.5-72B-Instruct** or **Mixtral-8x22B** (open-weight, deployable; adds capacity diversity without a closed-API dependency). Must pass the live injection gate (U6) before inclusion. *This is a pre-registration choice — confirm before the lock.*
  - **F6 lock decision (inventory-I):** recommend enabling F6 on the RARR cycle (`recall_min_denominator = 8`, `recall_floor_epsilon` small, `gate=False` keep-but-flag) so thin/under-detected recall cells are flagged + disclosed, with a uniform λ-floor for numerical stability. Confirm at the lock.
- **U6 — Live injection gate [buildable offline].** The live/recorded-response injection gate the 4th model must pass (Stage-2 delimiter escaping is done; the live gate is not). Testable with recorded responses.
- **U7 — Live-run wiring [buildable offline].** Fill `bakeoff_cmd`'s `NotImplementedError` — build a `config_name → {incident_id: entry_id}` `predict_fn` over RunPod (compose `HttpRunPodClient` + the per-model classify path), then call the built `run_bakeoff`. Testable with a MOCK predict_fn; the LIVE run is U8. **Winner-classify producer MUST call `write_classify_coverage(...)`** (the F-B contract) or the recall-flip sites raise.
- **U8 — THE RUN [HARD-STOP: needs the endpoint].** Execute on RunPod: bake-off → winner → winner-classify → recall calibrate → primary/robustness/oracle, in `cycles/2026-rarr/`. Carries B3 selection-quality gates (F2 goal-alignment, I2 BH power, F6(8e) winner's-curse, F8(decide) TV, F10 stability). Output PROVISIONAL until the oracle agrees.
- **U9 — Report [needs U8's new rankings].** Renders **previous-vs-new SIDE BY SIDE** — the previous side is already frozen (`baselines/2026/rankings_baselines.json`, the U9 loader is built + contract-tested); the new side is U8's output. Plus the F6 thin-cell disclosure (inventory D→U9) + the power/sensitivity render.

## Cross-unit obligations already tracked (from the premortems)
- D→U9: render the F6 thin-cell flags + a λ sensitivity panel; gate the report to require it when `recall_min_denominator>0`.
- F→U9: commit the ranking sensitivity grid (K∈{6,8,10}) as a report artifact.
- I→U5: the F6 lock-acceptance check (above).
- U3 FOR-THE-HUMAN block (in `2026-06-30-u3-f7-baselines.md`): the target-κ, entry-set, power-framing, and bridge-coherence resolutions for review.

## Resume protocol
Read this file + `RARR-campaign-execution.md` + `RARR-remaining-work-inventory.md` + `.superpowers/sdd/progress.md` + `LESSONS-rarr.md`. Confirm the U5 pre-registration decisions (4th model, F6 lock, constants). Provision the RunPod endpoint + set `RUNPOD_ENDPOINT_ID`. Then execute U5→U9 per the per-unit loop (plan → adversarial-premortem-complete → remediate → subagent-driven → verify → push → CI). NEVER write into `cycles/2026/`; NEVER fabricate a run; keep the previous rankings frozen; output stays provisional until the oracle agrees.
