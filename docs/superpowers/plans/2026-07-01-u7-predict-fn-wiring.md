# RARR U7 — Live Bake-off predict_fn Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the `NotImplementedError` in `bakeoff_cmd` with a real, label-blind RunPod `predict_fn` (single-config classification of goldset incidents) plus cost tracking, retry, and checkpoint — fully mock-testable offline ($0). The LIVE run is U8; U7 ships the wiring + mock tests only.

**Architecture:** A factory `build_live_predict_fn(...)` returns `predict_fn(config_name) -> dict[incident_id, entry_id]` that, for each goldset incident, builds the Stage-2 messages, calls that config's pod via `HttpRunPodClient`, parses the entry_id, with exponential-backoff retry, `CostTracker` accounting, and a resume-on-crash checkpoint. `bakeoff_cmd` loads the locked grid + goldset + snapshot + floor labels, wires pod URLs from env, builds the predict_fn, and calls the existing `run_bakeoff`. The predict_fn NEVER sees adjudicated labels — the lockbox split + scoring happen downstream in `run_bakeoff` (seed 42).

**Tech Stack:** Python 3.11, `engine.classify.{bakeoff,runpod_client,stage2,stage2_prompt,cost_tracker}`, `engine.prereg.bakeoff_grid`, pytest with mocked HttpRunPodClient.

## Global Constraints
- **Leakage firewall:** `predict_fn` classifies incident TEXT only; it receives NO goldset labels. Lockbox split (LOCKBOX_FRACTION=0.3, seed=42, drawn once) + scoring stay in `run_bakeoff`. Do not thread labels into prediction.
- **Offline/$0 in U7:** every test mocks `HttpRunPodClient` (no network, no pod). The live run is U8.
- **NEVER write into `cycles/2026/`.** RARR outputs go to `cycles/2026-rarr/`.
- **Cost safety for the real run:** `CostTracker(ceiling_usd, abort_factor=1.2)`; `check_or_abort()` before each batch; a bad cost estimate must be conservative (over-estimate, never under).
- No AI attribution. Per commit: `uv run ruff check .` + `uv run mypy engine tests` clean; targeted tests green. Controller runs the full suite. Do NOT push (controller pushes).
- **Contract (from surface map):** `PredictFn = Callable[[str], dict[str,str]]` (bakeoff.py:28); `run_bakeoff(goldset_path, config_names, predict_fn, floor_predictions, model_configs, out_dir, label_file, lockbox_fraction=0.3, seed=42, alpha=0.05, min_cell=5, checkpoint_dir=None, corpus_class_counts=None)`. `HttpRunPodClient(base_url=...).run_sync(messages, seed) -> RunPodResponse(output_text, job_id, execution_time_ms)`; `RunPodError` on failure (NO built-in retry). `build_messages(incident, rubric_json) -> [system,user]`; `Stage2Classifier` parses response → entry_id (validated ∈ rubric else out-of-scope). `_load_model_configs(pod_mode=True)` → list[(name, url)] from RUNPOD_MODEL_N_URL/NAME.

---

### Task 1: Retry + cost-aware single-incident classify call

**Files:**
- Create: `engine/classify/bakeoff_predict.py`
- Test: `tests/unit/test_bakeoff_predict.py`

**Interfaces:**
- Produces: `classify_one(client, incident, rubric_json, seed, cost_tracker, cost_per_call_usd, *, max_retries=3) -> str` — returns entry_id; retries `RunPodError` with exponential backoff; records cost; calls `cost_tracker.check_or_abort()` after recording. Uses `Stage2Classifier`-equivalent parsing (reuse `engine.classify.stage2` parse) so a subverted/garbled response falls back to "out-of-scope".

- [ ] **Step 1: Failing test** — mock a client returning a valid JSON response → `classify_one` returns the entry_id and records one cost entry.

```python
# tests/unit/test_bakeoff_predict.py
import pytest
from engine.classify.bakeoff_predict import classify_one, RetryExhaustedError
from engine.classify.cost_tracker import CostTracker
from engine.classify.runpod_client import RunPodError, RunPodResponse
from engine.corpus.types import IncidentRecord  # confirm actual IncidentRecord import path

RUBRIC = '{"entries":[{"entry_id":"LLM01"}]}'  # implementer: use a minimal valid rubric_json the parser accepts

class _Client:
    def __init__(self, responses):  # responses: list of RunPodResponse or Exception
        self._r = list(responses); self.calls = 0
    def run_sync(self, messages, seed):
        self.calls += 1
        r = self._r.pop(0)
        if isinstance(r, Exception): raise r
        return r

def _inc(i="INC-1", text="some incident"):
    return IncidentRecord(incident_id=i, text=text)  # match real constructor

def test_classify_one_returns_entry_id_and_records_cost():
    c = _Client([RunPodResponse('{"entry_id":"LLM01","confidence":0.9,"rationale":"x"}', "job1", 12.0)])
    ct = CostTracker(ceiling_usd=100.0)
    out = classify_one(c, _inc(), RUBRIC, seed=42, cost_tracker=ct, cost_per_call_usd=0.05)
    assert out == "LLM01"
    assert ct.total_cost_usd == pytest.approx(0.05)
    assert c.calls == 1

def test_classify_one_retries_then_succeeds():
    c = _Client([RunPodError("timeout"), RunPodResponse('{"entry_id":"LLM01","confidence":0.9,"rationale":"x"}',"j",1.0)])
    ct = CostTracker(ceiling_usd=100.0)
    out = classify_one(c, _inc(), RUBRIC, seed=42, cost_tracker=ct, cost_per_call_usd=0.05, max_retries=3)
    assert out == "LLM01" and c.calls == 2

def test_classify_one_raises_after_max_retries():
    c = _Client([RunPodError("e"), RunPodError("e"), RunPodError("e")])
    ct = CostTracker(ceiling_usd=100.0)
    with pytest.raises(RetryExhaustedError):
        classify_one(c, _inc(), RUBRIC, seed=42, cost_tracker=ct, cost_per_call_usd=0.05, max_retries=3)
```

- [ ] **Step 2: Run → fail (module missing).**
- [ ] **Step 3: Implement `classify_one`.** Reuse the existing Stage-2 response parsing (import the parse path from `engine/classify/stage2.py`; if it is a private method, factor a tiny module-level `parse_stage2_response(output_text, valid_entry_ids) -> str` OR instantiate the classifier's parser — pick the least-invasive; do NOT duplicate the JSON-parse logic). Backoff sleep must be injectable/patchable (e.g. `sleep_fn=time.sleep` param) so tests don't actually sleep. Record cost BEFORE the retry decision so a failed call that still cost money is counted (confirm RunPodResponse gives execution_time; cost_per_call_usd is passed in). Raise `RetryExhaustedError(RunPodError)` after `max_retries`.
- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(bakeoff): retry+cost-aware single-incident classify for live predict_fn`.

---

### Task 2: The `build_live_predict_fn` factory (label-blind)

**Files:**
- Modify: `engine/classify/bakeoff_predict.py`
- Test: `tests/unit/test_bakeoff_predict.py`

**Interfaces:**
- Produces: `build_live_predict_fn(pod_urls: dict[str,str], model_names: dict[str,str], goldset_incidents: dict[str, IncidentRecord], rubric_json: str, cost_tracker: CostTracker, cost_per_call: dict[str,float], *, seed: int, checkpoint_dir: Path | None = None, client_factory=HttpRunPodClient, max_workers: int = 18) -> PredictFn`. The returned `predict_fn(config_name)` classifies EVERY goldset incident via that config's pod and returns `{incident_id: entry_id}`. It NEVER receives labels.

- [ ] **Step 1: Failing test** — inject a `client_factory` returning a canned client per config; assert `predict_fn("qwen25-72b")` returns a full `{incident_id: entry_id}` dict covering all goldset incidents, and a checkpoint file is written; a second call resumes from checkpoint (client not re-invoked for done ids).

```python
def test_build_predict_fn_classifies_all_incidents_and_checkpoints(tmp_path):
    from engine.classify.bakeoff_predict import build_live_predict_fn
    from engine.classify.cost_tracker import CostTracker
    from engine.classify.runpod_client import RunPodResponse
    incs = {f"INC-{i}": _inc(f"INC-{i}", f"text {i}") for i in range(3)}
    class _C:
        def __init__(self, *a, **k): self.calls = 0
        def run_sync(self, messages, seed):
            self.calls += 1
            return RunPodResponse('{"entry_id":"LLM01","confidence":0.9,"rationale":"x"}', "j", 1.0)
    made = {}
    def factory(*a, **k):
        c = _C(); made[k.get("base_url","")] = c; return c
    ct = CostTracker(ceiling_usd=100.0)
    pf = build_live_predict_fn(
        pod_urls={"qwen25-72b":"http://pod"}, model_names={"qwen25-72b":"Qwen/Qwen2.5-72B-Instruct"},
        goldset_incidents=incs, rubric_json=RUBRIC, cost_tracker=ct, cost_per_call={"qwen25-72b":0.01},
        seed=42, checkpoint_dir=tmp_path, client_factory=factory,
    )
    out = pf("qwen25-72b")
    assert set(out) == set(incs) and all(v == "LLM01" for v in out.values())
    assert (tmp_path / "predict_qwen25-72b.jsonl").exists()
```

- [ ] **Step 2: Run → fail. Step 3: Implement.** Use `ThreadPoolExecutor(max_workers)` over the goldset incidents for one config; build messages via `build_messages`; call `classify_one`; write each `{incident_id, entry_id}` to the per-config checkpoint (`predict_<config>.jsonl`) as completed; on re-entry, load done ids and skip. `cost_per_call[config]` supplies the cost. If `cost_tracker.check_or_abort()` raises mid-sweep, propagate (the run aborts — safety). Raise `KeyError`/`ValueError` if `config_name` has no pod_url (never silently skip a config).
- [ ] **Step 4: Run → pass. Step 5: Commit** — `feat(bakeoff): label-blind live predict_fn factory (threaded, checkpointed)`.

---

### Task 3: Cost model helper

**Files:**
- Modify: `engine/classify/bakeoff_predict.py`
- Test: `tests/unit/test_bakeoff_predict.py`

**Interfaces:**
- Produces: `estimate_cost_per_call(gpu_count: int, hourly_rate_per_gpu_usd: float, est_calls_per_hour: float) -> float` — a CONSERVATIVE (over-estimating) per-call cost = `gpu_count * hourly_rate_per_gpu_usd / est_calls_per_hour`. Replaces the hardcoded `$0.01` placeholder. Default `hourly_rate_per_gpu_usd=4.0` (H200 upper bound), and a low `est_calls_per_hour` to over-estimate.

- [ ] **Step 1: Failing test** — `estimate_cost_per_call(2, 4.0, 3600)` == 2*4/3600; and a higher gpu_count yields higher per-call cost. **Step 2: fail. Step 3: implement** (pure arithmetic; document that over-estimation is intentional so CostTracker aborts EARLY not late). **Step 4: pass. Step 5: Commit** — `feat(bakeoff): conservative H200 per-call cost estimate (replaces $0.01 placeholder)`.

---

### Task 4: Wire `bakeoff_cmd` + floor/corpus loaders (mock-tested end-to-end)

**Files:**
- Modify: `engine/cli/bakeoff.py` (fill the `NotImplementedError` at ~line 105)
- Create: `engine/classify/bakeoff_inputs.py` (floor_predictions + corpus_class_counts + goldset-incident loaders)
- Test: `tests/unit/test_bakeoff_cli.py` (extend the existing mock-predict_fn tests)

**Interfaces:**
- `load_floor_predictions(labeled_incidents_path) -> dict[incident_id, entry_id]` (Stage-1 status-quo labels).
- `load_goldset_incidents(goldset_path, snapshot_path) -> dict[incident_id, IncidentRecord]` (goldset ids → text from snapshot).
- `compute_corpus_class_counts(labeled_incidents_path) -> dict[entry_id, int]`.
- `bakeoff_cmd(cycle_dir: Path, *, execute: bool, predict_fn: PredictFn | None = None) -> BakeoffResult` — if `predict_fn` is None AND `execute`, build the LIVE one from env pod URLs (U8); if a `predict_fn` is injected (tests / dry-run), use it. Loads grid via `load_bakeoff_grid`, reads `cost_ceiling_usd` from `stage2_manifest.json`, calls `run_bakeoff`, writes result under `cycle_dir/classify/`.

- [ ] **Step 1: Failing test** — call `bakeoff_cmd(cycle_dir, execute=False, predict_fn=<mock>)` on a tiny fixture cycle (reuse the `_write_goldset` pattern + a minimal snapshot + a floor file); assert it returns a `BakeoffResult`, picks the perfect config, and writes provenance — all offline, no network. Also assert that with `predict_fn=None, execute=False` it does NOT hit the network (raises a clear error telling the caller to inject or pass env pod URLs).
- [ ] **Step 2: Run → fail (NotImplementedError / missing loaders). Step 3: Implement** the loaders + wire `bakeoff_cmd`. The injected-predict_fn seam is what makes it offline-testable; the live path (predict_fn=None, execute=True, env pod URLs present) is exercised only in U8. Keep the `NotImplementedError` replaced with real logic. Register the CLI args in `engine/cli/main.py` if bakeoff is a subcommand (check + wire).
- [ ] **Step 4: Run → pass. Step 5: Commit** — `feat(bakeoff): wire bakeoff_cmd with injectable predict_fn + floor/corpus loaders (mock-tested)`.

---

## Self-Review
- Coverage: retry+cost call (T1), label-blind factory+checkpoint (T2), cost model (T3), CLI wiring + loaders (T4). The live path is wired but only mock-exercised (U8 runs it live).
- Leakage: predict_fn signature takes NO labels; goldset_incidents carry only id+text. ✅
- **Premortem targets** (a)-(g): all addressed via the remediations below.

---

## Premortem remediations (MANDATORY — folded from workflow wgmh71num, verdict proceed-with-remediations)

The adjudicator confirmed the plan is sound and DROPPED as non-issues: check_or_abort-between-configs (our design places it FINER, per-incident inside classify_one), lockbox seed/fraction-from-manifest (these are locked design constants LOCKBOX_FRACTION=0.3/seed=42, NOT manifest-configurable), sleep-injectable (already in T1), delimiter-neutralization-bypass (build_messages is mandated), execution_time-cost-unreliability (CostTracker sums the fixed per-call estimate, not exec_time). The 7 survivors below are MANDATORY:

**R1 (High) → Task 1 — cost on EVERY attempt, not just success.** `RunPodError` carries NO response (no job_id/exec_time), and the existing `stage2.py` records cost only on success. `classify_one` MUST record `cost_per_call_usd` on EVERY attempt including each `RunPodError` (use a synthetic attempt-scoped job_id like `f"{incident_id}-attempt{n}"` when no response exists), then call `cost_tracker.check_or_abort()` after each record. TEST: mock a client that fails N times then succeeds → assert `total_cost_usd == (N+1)*cost_per_call` and job_count == N+1 (NOT 1). This is the anti-retry-storm guarantee.

**R2 (High) → Task 2 — fail-loud config→pod validation BEFORE threads.** Before spawning the ThreadPoolExecutor, validate `set(config_names) <= set(pod_urls)` AND flag extras; raise `ValueError` with the full missing/extra diff. Await EVERY future's `.result()` (per `reclassify.py:178` as_completed+result precedent) so a worker exception propagates instead of being swallowed. TEST: `build_live_predict_fn(config_names=["missing"], pod_urls={})` raises ValueError naming the config, with ZERO client calls.

**R3 (Medium) → Task 4 — cost ceiling from the LOCKED manifest, no override.** Read `cost_ceiling_usd` AND `abort_factor` from `stage2_manifest.json` (reuse the `Stage2Manifest.read()` pattern at `pipeline.py:372`) → `CostTracker(ceiling_usd=..., _abort_factor=...)`. Do NOT expose a `--cost-ceiling` CLI flag. Fail LOUD if the manifest is missing or `cost_ceiling_usd` is null. **NOTE for implementer:** the U5 stage2_manifest.json added fields (`abort_factor`, `selected_from`, `injection_gate_*`) — verify `Stage2Manifest.read()` tolerates them; if the dataclass is strict and rejects extras, read `cost_ceiling_usd`/`abort_factor` directly from the JSON instead (do NOT loosen a shared strict loader mid-campaign). TEST: tracker ceiling == 500.0 from the fixture manifest.

**R4 (Medium) → Task 3 — LOW cost-estimate default (the 3600 example was wrong).** Replace the illustrative `est_calls_per_hour=3600` with a defensibly-LOW default (e.g. ~200 calls/hr for a 72B on H200 — over-estimating per-call cost so `check_or_abort` fires EARLY/safe, never late). Document that over-estimation is intentional. `CostTracker.reconcile()` is the U8 post-hoc discrepancy backstop (>10% flag), NOT the abort driver.

**R5 (Medium) → Task 2/Task 4 — two-level checkpoint contract.** Document in BOTH docstrings: the OUTER `run_bakeoff` checkpoint (`checkpoint_dir/{name}.json`, full-dict per-config cache) short-circuits `predict_fn` entirely on resume; the INNER `predict_<config>.jsonl` resumes a crashed mid-config sweep. They are distinct files; to be safe, isolate the inner ones under `checkpoint_dir/predict/`. TEST: crash mid-config, re-enter → already-done incident ids are NOT re-sent to the mock client and coverage still passes.

**R6 (Medium) → Task 4 — mock test BINDS bakeoff_cmd's LIVE predict_fn construction.** Add an offline test calling `bakeoff_cmd(cycle_dir, execute=True, predict_fn=None)` with an injected `client_factory` (mocked `HttpRunPodClient` returning canned `RunPodResponse`s) so the glue where bakeoff_cmd itself builds the live predict_fn (env pod URLs → build_live_predict_fn → classify_one) is exercised WITHOUT network — not only the stub-predict_fn path. (Thread `client_factory`/pod-url source through bakeoff_cmd as an injectable seam.)

**R7 (Medium) → Task 4 — leakage firewall by test.** Add a test asserting `load_goldset_incidents` returns records exposing ONLY `incident_id` + `text` (NO label/confidence/adjudicated fields) and `set(returned.keys()) == set(goldset_ids)` — locking the label-blind invariant by test, not just by construction.
