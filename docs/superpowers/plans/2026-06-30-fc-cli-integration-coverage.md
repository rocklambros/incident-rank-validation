# F-C: CLI integration coverage (shared real-snapshot minimal-cycle fixture) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).
> **This plan was hardened by an adversarial premortem (Round 1, converged) — every task block carries a "Premortem fixes (BINDING)" list. Apply them; they prevent pass-for-the-wrong-reason and false-RED failures the bare plan would have shipped.**

**Goal:** ONE shared, override-able real-snapshot minimal-cycle pytest fixture + integration tests closing the pre-Phase-3 CLI-integration gap: recall-flip-reflects-classifier, decide_real→run_oracle (8d-I3), the two F-B raises routed here (cal_tally completeness + infer goldset-guard), and live non-empty overlap W — all through the REAL adapter with a future-dated record (the universe-drift regression class synthetic fixtures cannot catch).

**Architecture:** A builder-factory fixture in `tests/integration/conftest.py` writes a tmp cycle whose `corpora/genai_agentic/<H>/incidents.json` (RAW universe, incl. a future-dated record the adapter drops) is content-addressed by `snapshot_hash(incidents.json)`; `manifest.snapshot_hash=<H>`; the F-B marker is written via the production `write_classify_coverage`. Tests drive executor functions (precise raise/return assertions) and, for 8d-I3, the `decide-real` CLI then call `run_oracle(cycle)` directly. **No live RunPod, no real NUTS** — `run_inference` is spied at its source module; decide/oracle use pre-staged `lambda_samples.npy`. **Test-only:** adds `tests/integration/conftest.py` + `tests/integration/test_fc_real_cycle.py` (+ one pyproject marker registration); changes NO engine behavior.

**Tech Stack:** pytest, numpy, openpyxl, click CliRunner, the engine's own writers.

## Global Constraints
- **No AI/Claude/Anthropic attribution** anywhere.
- **CI gate (exact):** `uv run ruff check .` + `uv run mypy engine tests` before every commit; FULL `uv run pytest -q` before any push. mypy strict (every test fn `-> None`, helpers typed).
- **F4 pin** `tests/unit/test_recall_single_label_semantics.py` MUST stay green.
- **Byte-immutability:** the fixture builds tmp cycles ONLY; NEVER read-for-mutation or write under `projects/owasp-llm/cycles/2026/`. The builder asserts `out_dir.is_absolute()` and every cycle path is under `tmp_path`; the fixture is **function-scoped**; T6 asserts `tmp_path in cycle.parents` before the direct `run_oracle`.
- **Two-stage review per task** + final whole-increment review (opus); push PR #22; CI green.

## Verified facts (build to these — resolved premortem risks)
- Adapter reads a SINGLE `incidents.json` (dict `{"incidents":[...]}`, genai_agentic.py:101); the F-B guard reads the SAME file via `read_snapshot_universe_ids`. One corpus file; no `.jsonl`.
- `<H> = engine.snapshot.hashing.snapshot_hash(incidents.json)` computed AFTER writing; snapshot dir named `<H>`; `manifest.snapshot_hash=<H>`. Write `incidents.json` ONCE with `json.dumps(data, sort_keys=True)` (cross-platform hash stability); never re-serialize.
- F-B marker = `cycle/classify/classify_coverage.json` via `write_classify_coverage(cycle/"classify", snapshot_hash=<H>, corpus_incident_ids=<all raw ids>, in_scope_incident_ids=<labeled ids>)`.
- **Spy target = `engine.model.inference.run_inference`** (it is a LOCAL import inside `execute_infer_phase` at pipeline_executor.py:328, so the name is NOT on `engine.cli.pipeline_executor`; patching the source module intercepts the call-time `from … import run_inference`). Always call `execute_infer_phase(..., num_warmup=0, num_samples=1)` so an un-intercepted real fit fails fast instead of hanging.
- **Fixture manifest sets `overlap_min_fp=1`** (default is 2, manifest.py:65; the fixture's single FP would otherwise yield an EMPTY W and T3 would be a permanent false-RED). Tests read `overlap_min_fp` from the manifest, never hard-code a different value.
- `infer-real`/`decide-real` do NOT enforce `require_classifier_rule_hash_match` (only `cal-classify` does); `cal_tally` DOES check rubric_hash + lock_hash via `validate_and_tally` (so batch hashes must match `manifest.lock` + `rubric.json`).
- `classify-real` AND `infer-real` require `calibration/posteriors.json` pre-existing → fixture stages it.
- The adapter's `_transform` drops `date > snapshot_date` → the future-dated record is in the raw universe (counted in `n_corpus`) but absent from `iter_incidents()` and labeled (absorbed into `n_oos`).
- `decide-real` requires `--vote-xlsx` (`required=True`, pipeline.py:530); `cal_tally` requires `--cycle --manifest --rubric` (`required=True`, calibration.py:329-337); the vote loader expects sheet `"Raw Results (Anonymized)"` (loader.py:77).

## File Structure
- **Create** `tests/integration/conftest.py` — `real_minimal_cycle` builder factory (function-scoped).
- **Create** `tests/integration/test_fc_real_cycle.py` — tests T0–T7.
- **Modify** `pyproject.toml` — register the `integration` marker (T0, NOT T7 — `filterwarnings=["error"]` turns an unknown mark into a CI error).

---

### Task 0 (T0): Builder skeleton + snapshot-hash coupling + marker registration (the shared asset)
**Files:** Create `tests/integration/conftest.py`; create `tests/integration/test_fc_real_cycle.py` (test_builder_happy_path); modify `pyproject.toml`.
**Produces:** function-scoped fixture `real_minimal_cycle` → `build(tmp_path, *, truncate_labeled=False, ghost_recall_id=None, with_batches=False, with_infer=False, with_vote=False) -> Path`.

**Premortem fixes (BINDING):**
- Register `integration` in `pyproject.toml [tool.pytest.ini_options] markers` NOW (one line: `"integration: real-adapter CLI integration tests (tmp-only, no NUTS)"`).
- Builder writes `incidents.json` ONCE via `json.dumps(data, sort_keys=True)`; computes `<H>=snapshot_hash(that file)`; names the dir `<H>`; sets `manifest.snapshot_hash=<H>`. No re-serialization anywhere.
- Builder asserts `out_dir.is_absolute()` before any write; the returned `cycle` is under `tmp_path`. Fixture is **function-scoped**.
- Fixture manifest sets `overlap_min_fp=1`.
- T0 test asserts (in addition to "verify does not raise"): `_resolve_snapshot_incidents(cycle, H) is not None` (the guard path is LIVE, not a vacuous no-op); and reads `classify/classify_coverage.json` DIRECTLY asserting `n_corpus==6, n_in_scope==4, n_oos==2` (independent of the verifier — breaks write/verify circularity).

- [ ] **Step 1 (failing test):**
```python
def test_builder_happy_path(real_minimal_cycle, tmp_path) -> None:
    import json
    from engine.snapshot.hashing import snapshot_hash
    from engine.calibrate.coverage import (
        verify_labeled_completeness, _resolve_snapshot_incidents, COVERAGE_FILENAME,
    )
    cycle = real_minimal_cycle(tmp_path)
    assert tmp_path in cycle.parents or cycle == tmp_path
    manifest = json.loads((cycle / "prereg" / "manifest.json").read_text())
    snap_dirs = list((cycle / "corpora" / "genai_agentic").iterdir())
    assert len(snap_dirs) == 1
    H = snap_dirs[0].name
    assert manifest["snapshot_hash"] == H == snapshot_hash(snap_dirs[0] / "incidents.json")
    assert manifest["overlap_min_fp"] == 1
    assert _resolve_snapshot_incidents(cycle, H) is not None  # guard is LIVE
    marker = json.loads((cycle / "classify" / COVERAGE_FILENAME).read_text())
    assert (marker["n_corpus"], marker["n_in_scope"], marker["n_oos"]) == (6, 4, 2)
    labeled = json.loads((cycle / "classify" / "labeled_incidents.json").read_text())
    labeled_ids = {str(r["incident_id"]) for r in labeled}
    verify_labeled_completeness(cycle, H, labeled_ids)  # must NOT raise
```
- [ ] **Step 2:** run → FAIL. **Step 3:** implement the builder (6-incident universe below; use real writers `write_classify_coverage`, `snapshot_hash`).

Corpus `incidents.json` = `{"incident_count": 6, "incidents": [...]}`, each record carrying exactly the keys `GenAIAgenticAdapter._transform` requires (READ genai_agentic.py first; include precisely those — id, date, title, description, impact, corpus, category, quality_tier, severity, references, owasp_llm, year, …):

| id | date | classifier label (labeled) | goldset (adjudicated) | role |
|----|------|----------------------------|-----------------------|------|
| INC-01 | 2025-12-01 | LLM01 | — | clean in-scope |
| INC-02 | 2025-12-02 | LLM02 | llm_consensus=LLM02, labels=[LLM02], adjudicated=accept | goldset-agree |
| INC-FLIP | 2025-12-03 | LLM01 | llm_consensus=LLM02, labels=[LLM02], adjudicated=accept | recall-flip + FP for W |
| INC-03 | 2025-12-04 | LLM01 | — | filler (≥2 LLM01) |
| INC-OOS | 2025-12-05 | (absent from labeled) | — | genuine OOS, IN snapshot |
| INC-FUTURE | 2026-05-21 | (dropped by adapter) | — | universe-drift: date>pull_date 2026-05-20 |

`prereg/manifest.json` + `manifest.lock` (byte-identical): minimal constructible PreregManifest (mirror `tests/unit/test_prereg.py::_make_manifest`), `snapshot_hash=<H>`, `cycle_id="2026fc"`, `robustness_specs=[]`, `overlap_min_fp=1`. `prereg/rubric.json`: LLM01, LLM02 (non-frame-blind), entry_ids matching the adapter's provisional set; `rubric_hash` via `hash_file`. `calibration/posteriors.json`: Beta(1,1) for `LLM01::security`,`LLM02::security` (recall+precision). `calibration/adjudicated_goldset.jsonl`: INC-02 + INC-FLIP rows EXACTLY as the table (INC-FLIP MUST carry `"llm_consensus":"LLM02"` and `"labels":["LLM02"]`). `classify/labeled_incidents.json`: 4 rows (INC-01,INC-02,INC-FLIP,INC-03) `stratum:"security"`; `truncate_labeled` drops INC-03 (corpus + marker UNCHANGED — n_in_scope stays 4). `classify/classify_coverage.json` via `write_classify_coverage(corpus_incident_ids={6 raw ids}, in_scope_incident_ids={4 labeled ids})`. Conditionals: `with_infer` (T6/T3-staged) → see T3/T6; `with_vote` → see T6; `with_batches` → see T4; `ghost_recall_id` → see T5.
- [ ] **Step 4:** run → PASS. **Step 5:** `ruff` + `mypy engine tests`. **Step 6:** commit `test(integration): real-snapshot minimal-cycle builder + integration marker (F-C T0)`.

### Task 1 (T1): adapter drops the future-dated record (universe-drift self-guard)
**Premortem fixes:** the `n_oos` count is builder arithmetic — assert the SPECIFIC OOS set, not just the count.
- [ ] Test: `GenAIAgenticAdapter(snap_dir, snapshot_date="2026-05-20")` → assert `"INC-FUTURE" not in {i.id for i in adapter.iter_incidents()}`; `"INC-FUTURE" in read_snapshot_universe_ids(incidents.json)`; and `read_snapshot_universe_ids(incidents.json) - labeled_ids == {"INC-OOS","INC-FUTURE"}`. Run/commit.

### Task 2 (T2): recall-flip reflects the classifier, not consensus
**Premortem fixes:** assert the DELTA attributable to INC-FLIP (not a vacuous `tp>=1` that INC-02 satisfies).
- [ ] Test: build gold from the fixture. (a) WITH `classifier_labels=load_classifier_labels(labeled)`: `calibrate_with_gold(...)` → `recall_counts[("LLM02","security")]` has `total_in_sample==2, false_negatives==1, true_positives==1` (INC-FLIP missed because classifier said LLM01; INC-02 hit). (b) WITHOUT classifier (`classifier_labels=None`, consensus path): same cell has `false_negatives==0, true_positives==2` (INC-FLIP credited via consensus LLM02). Assert the FN delta of exactly 1 is the flip. Drive via `engine/calibrate/tally.py` + `gold_loader.py`. Run/commit.

### Task 3 (T3): overlap W non-empty end-to-end (W routed to the model call site)
**Premortem fixes:** spy `engine.model.inference.run_inference`; read `overlap_min_fp` from the manifest; fully specify the spy return; assert `spy.called`. **Disclosure:** this proves W REACHES the model call, not that NUTS consumes it — see the ledger's W-consumption note.
- [ ] Test: `build_overlap_from_confusion(gold, ("LLM01","LLM02"), min_fp_count=manifest["overlap_min_fp"])` → assert non-empty (the LLM01-claimed/LLM02-true confusion populates W). Then `monkeypatch.setattr("engine.model.inference.run_inference", spy)` where `spy` captures its `overlap` kwarg and returns:
```python
InferenceResult(entry_ids=("LLM01","LLM02"), lambda_samples=np.ones((200,2), dtype=np.float64),
                r_hat={"LLM01":1.0,"LLM02":1.0}, ess={"LLM01":200.0,"LLM02":200.0},
                divergences=0, num_warmup=0, num_samples=1)  # match the real InferenceResult ctor
```
Call `execute_infer_phase(cycle, num_warmup=0, num_samples=1)`; assert `spy.called` AND the captured `overlap.weights` is non-empty. Run/commit.

### Task 4 (T4): cal_tally RAISES on truncated labeled_incidents (routed F-B cal_tally raise)
**Premortem fixes:** `--manifest` MUST be `manifest.lock`; pass ALL required CLI args; batch hashes computed from the written `manifest.lock`+`rubric.json`; pre-flight that the NON-truncated cycle's `validate_and_tally` does NOT raise; assert the EXACT exception/message.
- [ ] Builder `with_batches=True`: write `prereg/rubric.json`+`manifest.lock` FIRST, then `generate_batch` (one coded RECALL batch covering the labeled incidents with `labels`) into `calibration/batches/<id>.json` + `calibration/generate_batches_provenance.json` (via `write_provenance`, `manifest_lock_hash=hash_file(manifest.lock)`). Build exactly the batches `validate_and_tally` needs to pass (matching sample_hashes/rubric_hash/lock_hash) — do NOT stub validation.
- [ ] Pre-flight test: `build(with_batches=True)` (NOT truncated) → `cal_tally` via CliRunner `["--cycle",cycle,"--manifest",manifest_lock,"--rubric",rubric_json,"--gold-calibration",cal_dir]` exits 0 (proves the batch chain is clean).
- [ ] Raise test: `build(truncate_labeled=True, with_batches=True)` → same invocation → `result.exit_code != 0` AND `type(result.exception).__name__ == "LabeledIncidentsIncompleteError"` AND `"does not reconcile" in str(result.exception)` (unique to check 5; NOT a batch-validation ValueError, NOT check-4's "pinned snapshot size"). Run/commit.

### Task 5 (T5): infer goldset-guard RAISES on a scored recall incident absent from the snapshot
**Premortem fixes:** spy at `engine.model.inference.run_inference`; ghost row `labels` ∈ {LLM01,LLM02} (else a rubric ValueError gets re-wrapped to RuntimeError); ghost id is NOT added to `labeled_incidents.json` (else check-1 fires, not check-6); assert `match="goldset"`.
- [ ] Builder `ghost_recall_id="INC-GHOST"`: add an adjudicated goldset row `{"incident_id":"INC-GHOST","llm_consensus":"LLM01","labels":["LLM01"],"adjudicated":"accept"}` (id NOT in corpus, NOT in labeled). Test: `monkeypatch.setattr("engine.model.inference.run_inference", spy)`; `execute_infer_phase(cycle, num_warmup=0, num_samples=1)` → `pytest.raises(LabeledIncidentsIncompleteError, match="goldset")` raised BEFORE inference (spy NOT called). Run/commit.

### Task 6 (T6): decide_real → run_oracle, 3 deliverables, none erroring (8d-I3)
**Premortem fixes:** pass `--vote-xlsx`; vote.xlsx sheet `"Raw Results (Anonymized)"`; assert decide's OWN `oracle_report.json` write BEFORE the direct call; assert the in-memory verdict from the direct call; `provisional == False`; seed a clear-winner λ so D1 doesn't FAIL on a 2-entry tie; `assert tmp_path in cycle.parents` before the direct call. Read `concordance._ranks_from_incidence` to confirm tie-break alignment with the oracle.
- [ ] Builder `with_infer=True` → `infer/lambda_samples.npy` shape (200,2) seeded with a CLEAR winner: `rng=np.random.default_rng(42); col0=rng.beta(5,1,(200,1)); col1=rng.beta(1,5,(200,1)); np.concatenate([col0,col1],axis=1)` + `inference_summary.json` (`entry_ids=["LLM01","LLM02"]`, dummy r_hat/ess/divergences). `with_vote=True` → `vote/vote.xlsx` via openpyxl with `ws.title="Raw Results (Anonymized)"` and the columns/rows `engine/vote/loader.py::load_vote_data` expects (READ loader.py for the exact schema; ≥1 numeric-rank data row over [LLM01,LLM02]).
- [ ] Test: `build(with_infer=True, with_vote=True)`; CliRunner `["decide-real","--execute","--cycle",cycle,"--vote-xlsx",cycle/"vote"/"vote.xlsx"]` → assert exit 0 AND `(cycle/"results"/"oracle_report.json").exists()` with 3 deliverables (decide's OWN write). Then `assert tmp_path in cycle.parents`; call `run_oracle(cycle)` directly → on the returned in-memory verdict assert the 3 deliverables `incidence`/`plackett_luce`/`sigma_u` each `status in {"PASS","SKIP"}` (D3 SKIPs with `robustness_specs=[]`), none errored, and `provisional == False`. Run/commit.

### Task 7 (T7): suite hygiene + disclosures
- [ ] Tag T0–T6 `@pytest.mark.integration` (marker already registered in T0). Module docstring: fixture is tmp-only, never touches `cycles/2026/`. Add a `pyproject.toml` comment above `addopts`: "DO NOT add `-m` filters — integration tests must run in CI." Run FULL `uv run pytest -q` (confirm sub-minute, no live RunPod/NUTS, F4 pin green) + ruff + mypy. Commit.

---

## Premortem Round 1 — convergence + disclosure ledger
Converged after 1 round: all findings were remediable construction details / coverage disclosures (folded into the tasks above), not design flaws. **Dropped:** secrets/network (dead — `--no-wandb` default, classify-real not invoked); snapshot_hash path-traversal (hex digest); `pytest-timeout` dep (mitigated by spy → no real-NUTS hang). **Disclosures (also add to LESSONS-rarr.md under F-C):**
- **W-consumption:** T3 proves W reaches the model CALL SITE, not that the NUTS likelihood consumes it. Before finalizing, verify an existing engine/model test asserts W changes the likelihood/log-prob; if none exists, add a tiny unit test that `overlap=W` vs `overlap=None` changes the model's computation. (Else re-label the claim "W routed to model call site.")
- **D3 SKIP:** with `robustness_specs=[]`, the oracle's D3 (σ_u) always SKIPs → D3 correctness is untested by F-C until a `hierarchical_pooling`-bearing fixture exists (tied to inventory #23, a U5 lock decision).
- **Frame-blind:** the fixture uses only non-frame-blind LLM01/LLM02; the decide→oracle path for frame-blind entries (zero goldset overlap) is untested — track for the U5 4th-model selection.
- **Residual:** single-fixture monoculture — mitigated by T0's direct-marker assertion + T1's set-level OOS check (independent of the verifier).
