# RARR U8 — Live Run Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A single orchestrator that runs the Phase-3 sequence on RunPod — provision 4 pods → wait ready → injection-gate → bake-off → tear down losers → winner-classify → **tear down ALL pods (guaranteed)** → reconcile cost — with pod-leak prevention as the load-bearing invariant. Built + MOCK-tested fully OFFLINE ($0); the live run is a separate deliberate execution of the SAME code with real env.

**Architecture:** `engine/cli/live_run.py::orchestrate_live_run(cycle_dir, *, provisioner, terminator, gate_fn, bakeoff_fn, classify_fn, cost_tracker, clock, ...)` — every external effect (provision, terminate, gate, bake-off, classify, time) is an INJECTED seam so the whole sequence is deterministic + mock-testable with NO GPU. The real entrypoint wires the live implementations (deploy_runpod, terminate_runpod, run_injection_gate, bakeoff_cmd, classify pipeline). Teardown of every provisioned pod is guaranteed by THREE independent mechanisms: an `atexit` handler registered at provision time, a `try/finally` around the whole run, and the standalone `terminate_runpod.py` as the manual backstop.

**Tech Stack:** Python 3.11, `tools.deploy_runpod`, `tools.terminate_runpod`, `engine.classify.injection_gate`, `engine.cli.bakeoff`, `engine.classify.cost_tracker`, the real-cycle pipeline (classify-real), pytest with injected fakes.

## Global Constraints (safety charter — every step honors these)
- **NO POD LEAK, EVER (Sev-1):** every provisioned pod_id is registered to the teardown set IMMEDIATELY at creation (before the readiness wait), torn down in a `finally`, and re-verified gone via `GET /pods` at the end. Teardown is idempotent (404 = success). If the orchestrator raises anywhere, `finally` + `atexit` still tear down.
- **Cost ceiling from the LOCKED manifest:** `CostTracker(ceiling_usd, _abort_factor)` from `stage2_manifest.json` (500/1.2 → hard stop $600); `check_or_abort` throughout; a background cost monitor; abort → teardown → stop.
- **Wall-time cap:** a hard cap (e.g. 6h total, 60-min readiness sub-cap); on breach → teardown ALL → stop + report.
- **Fail-closed gate:** every candidate model passes `run_injection_gate` → `filter_eligible_by_gate` BEFORE the bake-off; an ungated/failed model is excluded (never reaches the bake-off). (R9 from U6.)
- **Winner-classify writes the F-B marker:** the winner-classify producer MUST call `write_classify_coverage(...)` over the full corpus universe, or the recall-flip sites raise "marker missing" (F-B contract; now live after R5 restored it).
- **NEVER write into `cycles/2026/`.** All output → `cycles/2026-rarr/`. Output stays PROVISIONAL until the oracle agrees (U9).
- **Hard-stop, never fabricate:** if a pod won't deploy after retries, or a model errors out, or the gate excludes so many models that the bake-off is meaningless → STOP, tear down, report; do NOT synthesize numbers.
- No AI attribution. Per commit: ruff + mypy clean on touched files (engine AND test files); controller runs the full suite + `mypy engine tests`. Do NOT push (controller pushes).

---

### Task 1: Teardown-set registry + guaranteed-teardown context manager

**Files:**
- Create: `engine/cli/live_run.py`
- Test: `tests/unit/test_live_run.py`

**Interfaces:**
- Produces: `PodLeasePool(terminator)` — `register(pod_id)` adds to the live set (and to an atexit handler), `terminate_all()` idempotently tears down every registered pod and verifies via `terminator.list_live`. A context manager `guaranteed_teardown(pool)` that calls `pool.terminate_all()` in `__exit__` REGARDLESS of exception.

- [ ] **Step 1: Failing test** — pods registered then an exception raised inside the context → `terminate_all` still called for EVERY registered pod (the leak-prevention core).

```python
# tests/unit/test_live_run.py
import pytest
from engine.cli.live_run import PodLeasePool, guaranteed_teardown

class _FakeTerminator:
    def __init__(self): self.terminated = []; self.live = {"p1","p2"}
    def terminate_pod(self, pod_id): self.terminated.append(pod_id); self.live.discard(pod_id); return True
    def list_live_ids(self): return set(self.live)

def test_teardown_fires_on_exception():
    t = _FakeTerminator(); pool = PodLeasePool(t)
    with pytest.raises(RuntimeError):
        with guaranteed_teardown(pool):
            pool.register("p1"); pool.register("p2")
            raise RuntimeError("bakeoff blew up")
    assert set(t.terminated) == {"p1", "p2"}      # BOTH torn down despite the raise
    assert t.list_live_ids() == set()              # verified gone

def test_teardown_idempotent_on_already_gone():
    t = _FakeTerminator(); t.live = set(); pool = PodLeasePool(t)
    pool.register("p1")
    with guaranteed_teardown(pool):
        pass
    assert "p1" in t.terminated                    # attempted even if already gone (idempotent)
```

- [ ] **Step 2: fail. Step 3: implement** `PodLeasePool` + `guaranteed_teardown` (register to an `atexit` handler at first register; `terminate_all` iterates a COPY of the set, calls `terminator.terminate_pod` for each tolerating errors, then asserts none of ours remain in `list_live_ids`; log every teardown). **Step 4: pass. Step 5: commit** — `feat(live): guaranteed pod-teardown pool (atexit + finally, idempotent)`.

---

### Task 2: Readiness wait with wall-time cap

**Files:** Modify `engine/cli/live_run.py`; Test `tests/unit/test_live_run.py`

**Interfaces:** `wait_until_ready(pod_urls, *, is_ready_fn, clock, readiness_cap_s, poll_interval_s) -> None` — polls each pod's readiness until all ready or the cap elapses; on timeout raises `ReadinessTimeout` (caller tears down). `clock` injected (no real sleep in tests).

- [ ] **Step 1: Failing test** — all-ready before cap → returns; never-ready → raises `ReadinessTimeout` after the cap (with a fake clock, no real wait).
- [ ] **Step 2: fail. Step 3: implement** (injected `clock.now`/`clock.sleep`, injected `is_ready_fn(url)->bool`). **Step 4: pass. Step 5: commit** — `feat(live): readiness wait with injected clock + wall-time cap`.

---

### Task 3: The orchestration sequence (fully injected, mock-tested)

**Files:** Modify `engine/cli/live_run.py`; Test `tests/unit/test_live_run.py`

**Interfaces:** `orchestrate_live_run(cycle_dir, *, provisioner, terminator, gate_fn, bakeoff_fn, classify_fn, cost_tracker, clock, readiness_cap_s, wall_cap_s) -> LiveRunResult`. Sequence: (1) `guaranteed_teardown` opens; (2) provision 4 pods via `provisioner.deploy()` → register each to the pool IMMEDIATELY; (3) `wait_until_ready`; (4) `gate_fn` each model → `filter_eligible_by_gate` → if <2 eligible, STOP (raise, teardown fires); (5) `bakeoff_fn(eligible)` → winner; (6) tear down the NON-winner pods immediately; (7) `classify_fn(winner)` (writes the F-B coverage marker); (8) `finally`: `pool.terminate_all()`; (9) return `LiveRunResult(winner, gate_results, bakeoff_result, cost, pods_torn_down, provisional=True)`.

- [ ] **Step 1: Failing test — the HAPPY path with fakes** — a fake provisioner returns 4 pod ids/urls; fake gate passes all; fake bakeoff picks a winner; fake classify succeeds → assert the winner is returned, ALL pods torn down at the end, non-winners torn down before classify, and `provisional=True`.
- [ ] **Step 2: More failing tests — the SAFETY paths:**
  - bakeoff raises → all pods torn down, error propagated (no leak).
  - gate excludes all-but-one (<2 eligible) → STOP, all pods torn down, no bakeoff.
  - `cost_tracker.check_or_abort` raises mid-run → all pods torn down.
  - readiness timeout → all pods torn down, no bakeoff.
- [ ] **Step 3: implement the sequence** honoring the safety charter; every external effect via the injected seams; `finally: pool.terminate_all()`. **Step 4: all pass. Step 5: commit** — `feat(live): orchestrate provision→gate→bakeoff→teardown-losers→classify→teardown-all (mock-tested safety paths)`.

---

### Task 4: Live wiring + preflight (the real entrypoint — NO auto-run)

**Files:** Modify `engine/cli/live_run.py` (+ register CLI in `engine/cli/main.py`); Test `tests/unit/test_live_run.py`

**Interfaces:** `live_run_cli(cycle_dir, *, execute: bool)` — builds the REAL seams (provisioner=deploy_runpod wrapper, terminator=terminate_runpod, gate_fn=run_injection_gate, bakeoff_fn=bakeoff_cmd, classify_fn=classify-real, cost_tracker from stage2_manifest) and calls `orchestrate_live_run`. **A preflight (offline) runs FIRST and unconditionally:** assert `cycles/2026-rarr/prereg/manifest.lock` verifies, the grid revision SHAs resolve (R2 verify_grid_revisions), `terminate_runpod` reaches the API and reports 0 leaked pods, the cost ceiling is read from the manifest. If `execute` is False → run preflight + a DRY-RUN (print the plan, provision NOTHING) and exit. Only `execute=True` provisions real pods.

- [ ] **Step 1: Failing test** — `live_run_cli(cycle_dir, execute=False)` runs preflight + dry-run with NO provision call (assert the real provisioner is never invoked); a preflight failure (bad lock) raises before any provision.
- [ ] **Step 2: fail. Step 3: implement** (preflight gate; `execute=False` ⇒ dry-run; wire real seams for `execute=True` but do NOT call them in the test). **Step 4: pass. Step 5: commit** — `feat(live): live_run CLI with mandatory offline preflight + dry-run default`.

---

## Self-Review
- Coverage: teardown pool (T1), readiness cap (T2), the full sequence + 4 safety paths (T3), live wiring + preflight/dry-run (T4). The LIVE run is executing `live_run_cli(execute=True)` — done by the controller in a monitored window, NOT in tests.
- The leak-prevention invariant is proven by T1 (teardown on exception) + T3's safety-path tests (teardown on bakeoff-raise / gate-stop / cost-abort / readiness-timeout).
- **Premortem targets** (a)-(g): all resolved via the mandatory remediations below.

---

## Premortem remediations (MANDATORY — folded from workflow we8cfkuv0, verdict proceed-with-remediations)

The adjudicator verified the teardown DESIGN is sound (register-before-wait, try/finally + atexit, idempotent 404=success, post-teardown re-verify) and DROPPED as non-defects: the "not implemented yet" findings (that IS the deliverable), the lockbox-seed-leakage worry (verified NOT exploitable — `lockbox_split(seed=42)` is deterministic, called ONCE, `select_winner` on the lockbox only), the FD-leak (bounded), and the "bakeoff must write the F-B marker" (misread — the winner-CLASSIFY producer writes it, not run_bakeoff). The 6 survivors are MANDATORY before `execute=True`:

**R1 (High) → Task 1: DURABLE incrementally-persisted registry (closes the SIGKILL leak hole).** `PodLeasePool.register(pod_id, name)` MUST atomically append `{pod_id, name}` to `tools/runpod_pods.json` (write-temp + `os.fsync` + `os.replace`) IMMEDIATELY at pod creation, BEFORE the readiness wait — so after `kill -9`/`os._exit` (which bypass BOTH atexit and finally), the standalone `terminate_runpod.py --execute` (registry-based reconcile) can still recover every pod. Add `docs/RUNBOOK.md` with the detect+terminate procedure. TEST: register writes the pod to the on-disk registry synchronously (readable by a second process before terminate).

**R2 (High) → Task 4: FAIL-CLOSED preflight (no double-burn, no lockbox re-draw).** The preflight MUST call `terminator.reconcile(execute=False)` and RAISE if `orphans` is non-empty OR any live pod exists (reconcile itself only PRINTS, never raises — a naive preflight would provision on top of a prior leak → 8 live pods / $130/hr). ALSO: verify `manifest.lock` AND compute+check `sha256(bakeoff_grid.json)` (the grid carries seed/lockbox_fraction; `bakeoff_cmd` reads them via raw json.loads and never verifies the lock — a post-lock grid edit would re-draw the holdout = p-hacking). Re-verify both before the bake-off; assert `bakeoff_cmd` gets the same verified `cycle_dir`. (Consider `chmod 444` grid+manifest during the run.)

**R3 (High) → Task 2/3: interrupting background monitor.** The cost/wall-cap checks are between-steps only and cannot interrupt a blocked `bakeoff_fn`/`classify_fn` (the ThreadPoolExecutor uses `shutdown(wait=True)`). Add a background daemon thread that samples `cost_tracker.total_cost_usd` + elapsed wall time every 15-30s and, on ceiling or `wall_cap_s` breach, calls `pool.terminate_all()` then forces process exit — firing DURING a blocked phase. Use `cancel_futures=True` / `shutdown(wait=False)` when `CostCeilingExceeded` propagates. Confirm `HttpRunPodClient` enforces a finite per-call read timeout (it does — 300s) so no single call hangs unbounded. TEST (mock): a fake clock advancing past `wall_cap_s` during a blocking `bakeoff_fn` → the monitor calls `terminate_all` and the run aborts.

**R4 (Medium) → Task 1: partial teardown RAISES loudly + preserves the registry.** The underlying `terminate_all_registered`/`reconcile` SWALLOW errors into `summary['errors']` and continue without raising — an implementer could return "success" with a pod still live. `PodLeasePool.terminate_all` MUST, after issuing DELETEs, re-query `list_live_ids` and RAISE a loud `CompoundTerminateError` naming the surviving pod_ids if ANY registered pod is still live; do NOT clear the durable registry until the re-verify confirms empty; the orchestrator `finally` logs Sev-1. TEST: one pod's `terminate_pod` raises → `terminate_all` raises AND the registry still lists that pod.

**R5 (Medium) → Task 3: verify the F-B full-corpus marker IN-WINDOW before teardown.** After `classify_fn(winner)` and BEFORE the final teardown, call `verify_labeled_completeness` over the FULL corpus snapshot and assert `classify_fn` was given the full universe (NOT the goldset) — re-raise on `LabeledIncidentsIncompleteError` so a wrong-corpus / incomplete marker is caught while the winner pod is still up (re-runnable), not at U9 after teardown. TEST (mock): a classify_fn that writes a short marker → the post-classify verify raises before teardown.

**R6 (Low) → Task 4: secret hygiene.** Add a 5s timeout + redacted error to `load_secret`; scrub `Bearer ...` from any logged exception; prefer RunPod pod-secret injection over an HF_TOKEN env var, and note post-run token rotation in the RUNBOOK. (Lowest $-impact on a single-tenant account, but cheap defense-in-depth.)

## Parked tail risks (documented, not blocking the build)
- **Live run before Tasks 1-4 are mock-tested** → mitigated by discipline: the controller builds + full-suite-greens the orchestrator OFFLINE before ANY `execute=True`. This is the campaign's stated order.
- **RunPod account-wide API outage during teardown** → even correct code cannot DELETE if the API is down; mitigated by the loud-raise + preserved durable registry + the RUNBOOK's manual RunPod-console fallback. If teardown DELETEs all fail, the orchestrator raises Sev-1 with the pod_ids and I terminate via the console.
