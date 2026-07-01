"""Unit tests for engine/cli/live_run.py — Tasks 1-4.

ALL tests use FAKE terminators and clocks — NO network, NO RunPod pod,
NO real sleep (except the R3 monitor test which sleeps ~150 ms real time).
The real tools/runpod_pods.json is NEVER touched; every test that exercises
the durable registry uses a tmp_path fixture.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from engine.calibrate.coverage import LabeledIncidentsIncompleteError
from engine.classify.injection_gate import InjectionGateResult
from engine.cli.live_run import (
    CompoundTerminateError,
    InsufficientEligibleModels,
    LiveRunResult,
    PodInfo,
    PodLeasePool,
    ReadinessTimeout,
    RealClock,
    _pool_terminate_one,
    guaranteed_teardown,
    live_run_cli,
    orchestrate_live_run,
    wait_until_ready,
)

# ---------------------------------------------------------------------------
# Fake helpers
# ---------------------------------------------------------------------------


class _FakeTerminator:
    """Working terminator: terminate_pod discards from live set, returns True."""

    def __init__(self, live: set[str] | None = None) -> None:
        self.live: set[str] = live if live is not None else {"p1", "p2"}
        self.terminated: list[str] = []

    def terminate_pod(self, pod_id: str) -> bool:
        self.terminated.append(pod_id)
        self.live.discard(pod_id)
        return True

    def list_live_ids(self) -> set[str]:
        return set(self.live)


class _FakeTerminatorPartialFail:
    """Terminator where pod 'p1' raises on terminate_pod (e.g. 500 error)."""

    def __init__(self) -> None:
        self.live: set[str] = {"p1", "p2"}
        self.terminated: list[str] = []

    def terminate_pod(self, pod_id: str) -> bool:
        if pod_id == "p1":
            raise RuntimeError("terminate_pod failed for p1")
        self.terminated.append(pod_id)
        self.live.discard(pod_id)
        return True

    def list_live_ids(self) -> set[str]:
        return set(self.live)


class _FakeTerminatorLieAboutSuccess:
    """Terminator that returns True but never removes pods from the live set."""

    def __init__(self) -> None:
        self.live: set[str] = {"p1", "p2"}
        self.terminated: list[str] = []

    def terminate_pod(self, pod_id: str) -> bool:
        self.terminated.append(pod_id)
        # Intentionally does NOT discard from live — lies about success.
        return True

    def list_live_ids(self) -> set[str]:
        return set(self.live)


class _FakeClock:
    """Injectable fake clock — sleep() advances internal time; no real wait."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start
        self.slept: float = 0.0

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        self._t += seconds
        self.slept += seconds


# ---------------------------------------------------------------------------
# Task 1 — Test 1: teardown fires on exception
# ---------------------------------------------------------------------------


def test_teardown_fires_on_exception(tmp_path: Path) -> None:
    """Both pods are torn down even when the body of guaranteed_teardown raises.

    This is the load-bearing leak-prevention test: a raise inside the context
    must NEVER prevent teardown from firing.
    """
    t = _FakeTerminator()
    pool = PodLeasePool(t, registry_path=tmp_path / "pods.json")

    with pytest.raises(RuntimeError, match="bakeoff blew up"), guaranteed_teardown(pool):
        pool.register("p1", "pod-p1")
        pool.register("p2", "pod-p2")
        raise RuntimeError("bakeoff blew up")

    assert set(t.terminated) == {"p1", "p2"}, "BOTH pods must be terminated"
    assert t.list_live_ids() == set(), "no pods should remain live after teardown"


# ---------------------------------------------------------------------------
# Task 1 — Test 2: teardown idempotent on already-gone pod
# ---------------------------------------------------------------------------


def test_teardown_idempotent_on_already_gone(tmp_path: Path) -> None:
    """terminate_all() is attempted even when the pod is already gone.

    The real RunPod API returns 404 for a deleted pod; that 404 is treated as
    success (idempotent).  Here the FakeTerminator starts with an empty live
    set, simulating the 404 path.
    """
    t = _FakeTerminator(live=set())  # pod already gone
    pool = PodLeasePool(t, registry_path=tmp_path / "pods.json")
    pool.register("p1", "pod-p1")

    with guaranteed_teardown(pool):
        pass  # body succeeds

    assert "p1" in t.terminated, "terminate must be attempted even if already gone"


# ---------------------------------------------------------------------------
# Task 1 — Test 3 (R1): register writes durable registry synchronously
# ---------------------------------------------------------------------------


def test_register_writes_durable_registry(tmp_path: Path) -> None:
    """R1: register() writes the pod to the on-disk registry file synchronously.

    A second process reading the file BEFORE any terminate_all() call must see
    the newly registered pod.  This closes the SIGKILL leak hole: if the
    process is killed between register() and the readiness wait, the standalone
    terminate_runpod.py --execute can still recover the pod from the registry.
    """
    registry = tmp_path / "pods.json"
    t = _FakeTerminator(live=set())
    pool = PodLeasePool(t, registry_path=registry)

    pool.register("p1", "qwen3-235b")

    # Read registry BEFORE any terminate call — must already contain p1.
    data: list[dict[str, object]] = json.loads(registry.read_text())
    assert any(
        e.get("pod_id") == "p1" and e.get("name") == "qwen3-235b" for e in data
    ), f"registry should contain p1 immediately after register(); got: {data}"


# ---------------------------------------------------------------------------
# Task 1 — Test 4 (R4): partial teardown raises + preserves registry
# ---------------------------------------------------------------------------


def test_partial_teardown_raises_and_preserves_registry(tmp_path: Path) -> None:
    """R4: if one terminate_pod raises, terminate_all raises CompoundTerminateError
    AND the durable registry still lists the surviving pod.

    The registry must NOT be cleared on partial failure so that the standalone
    terminate_runpod.py can recover the surviving pod.
    """
    registry = tmp_path / "pods.json"
    t = _FakeTerminatorPartialFail()
    pool = PodLeasePool(t, registry_path=registry)
    pool.register("p1", "pod-one")
    pool.register("p2", "pod-two")

    with pytest.raises(CompoundTerminateError):
        pool.terminate_all()

    # Registry must NOT have been cleared — the surviving pod must still be listed.
    data: list[dict[str, object]] = json.loads(registry.read_text())
    pod_ids_in_registry = {e.get("pod_id") for e in data}
    assert "p1" in pod_ids_in_registry, (
        f"registry must preserve p1 (the survivor) after partial teardown; got: {data}"
    )


# ---------------------------------------------------------------------------
# Task 1 — Test 5 (R4 re-verify): terminate_all re-queries list_live_ids
# ---------------------------------------------------------------------------


def test_terminate_all_raises_if_pod_still_live_after_terminate_pod_returns(
    tmp_path: Path,
) -> None:
    """R4 re-verify: CompoundTerminateError is raised if list_live_ids shows a
    registered pod is still live even after terminate_pod returned True.

    This catches silent API lies (e.g. 200 OK from a buggy endpoint that
    did not actually terminate the pod).
    """
    registry = tmp_path / "pods.json"
    t = _FakeTerminatorLieAboutSuccess()
    pool = PodLeasePool(t, registry_path=registry)
    pool.register("p1", "pod-one")
    pool.register("p2", "pod-two")

    with pytest.raises(CompoundTerminateError, match="SEV-1"):
        pool.terminate_all()

    # Both terminate calls were attempted
    assert set(t.terminated) == {"p1", "p2"}


# ---------------------------------------------------------------------------
# Task 2 — Test 6: wait_until_ready returns when all pods ready
# ---------------------------------------------------------------------------


def test_wait_until_ready_returns_when_all_ready() -> None:
    """wait_until_ready returns without sleeping when is_ready_fn is True for all."""
    clock = _FakeClock(start=0.0)

    wait_until_ready(
        {"pod1": "http://p1:8000", "pod2": "http://p2:8000"},
        is_ready_fn=lambda url: True,
        clock=clock,
        readiness_cap_s=60.0,
        poll_interval_s=5.0,
    )
    # No sleep necessary when all pods are immediately ready.
    assert clock.slept == 0.0


# ---------------------------------------------------------------------------
# Task 2 — Test 7: wait_until_ready raises ReadinessTimeout
# ---------------------------------------------------------------------------


def test_wait_until_ready_raises_timeout() -> None:
    """wait_until_ready raises ReadinessTimeout when pods are never ready.

    Uses a fake clock so no real time elapses — sleep() just advances
    the internal counter.
    """
    clock = _FakeClock(start=0.0)

    with pytest.raises(ReadinessTimeout, match="pod1"):
        wait_until_ready(
            {"pod1": "http://p1:8000"},
            is_ready_fn=lambda url: False,
            clock=clock,
            readiness_cap_s=10.0,
            poll_interval_s=5.0,
        )

    # Fake clock must have advanced at or past the readiness cap.
    assert clock.now() >= 10.0


# ===========================================================================
# Task 3 helpers — fake seams for orchestrate_live_run
# ===========================================================================


def _make_gate_result(name: str, *, passed: bool) -> InjectionGateResult:
    """Build a minimal InjectionGateResult for test gate_fn fakes."""
    return InjectionGateResult(
        model_name=name,
        revision_sha="abc123",
        passed=passed,
        pass_rate=1.0 if passed else 0.0,
        threshold=1.0,
        error_count=0,
        probe_results=(),
    )


class _FakeProvisioner:
    """Returns a fixed list of PodInfo; tracks whether deploy() was called."""

    def __init__(self, pods: list[PodInfo]) -> None:
        self.pods = pods
        self.deploy_called = False

    def deploy(self, on_pod_created: Any = None) -> list[PodInfo]:
        self.deploy_called = True
        for pod in self.pods:
            if on_pod_created is not None:
                on_pod_created(pod.pod_id, pod.name)
        return self.pods


class _FakeBakeoffResult:
    """Minimal duck-type for BakeoffResult; _extract_winner_name reads .winner.name."""

    class _Winner:
        def __init__(self, name: str) -> None:
            self.name = name

    def __init__(self, winner_name: str) -> None:
        self.winner = self._Winner(winner_name)


class _FakeCostTracker:
    """CostTracker that can be set to abort on check_or_abort()."""

    def __init__(self, should_abort: bool = False) -> None:
        self.total_cost_usd: float = 0.0
        self._should_abort = should_abort

    def check_or_abort(self) -> None:
        if self._should_abort:
            raise RuntimeError("SEV: cost ceiling exceeded (fake)")


class _TrackingTerminator:
    """FakeTerminator that tracks the order of terminate_pod() calls."""

    def __init__(self, live: set[str]) -> None:
        self.live = set(live)
        self.terminated: list[str] = []

    def terminate_pod(self, pod_id: str) -> bool:
        self.terminated.append(pod_id)
        self.live.discard(pod_id)
        return True

    def list_live_ids(self) -> set[str]:
        return set(self.live)


def _make_4_pods() -> list[PodInfo]:
    return [
        PodInfo("p1", "model1", "http://p1:8000"),
        PodInfo("p2", "model2", "http://p2:8000"),
        PodInfo("p3", "model3", "http://p3:8000"),
        PodInfo("p4", "model4", "http://p4:8000"),
    ]


# ===========================================================================
# Task 3 — Test 8 (a): happy path
# ===========================================================================


def test_orchestrate_happy_path(tmp_path: Path) -> None:
    """Happy path: winner returned; non-winners torn down before classify; all torn down at end."""
    pods = _make_4_pods()
    all_ids = {"p1", "p2", "p3", "p4"}
    term = _TrackingTerminator(all_ids)
    winner_name = "model1"  # pod p1

    terminated_at_classify_time: list[str] = []

    def classify_fn(winner: str) -> None:
        # Snapshot which pods are terminated AT classify time.
        terminated_at_classify_time.extend(term.terminated)

    result = orchestrate_live_run(
        tmp_path,
        provisioner=_FakeProvisioner(pods),
        terminator=term,
        gate_fn=lambda name: _make_gate_result(name, passed=True),
        bakeoff_fn=lambda eligible: _FakeBakeoffResult(winner_name),
        classify_fn=classify_fn,
        cost_tracker=_FakeCostTracker(),
        clock=_FakeClock(),
        readiness_cap_s=60.0,
        wall_cap_s=3600.0,
        on_fatal=lambda r: None,  # should never fire
        poll_s=0.01,
        is_ready_fn=lambda url: True,
        registry_path=tmp_path / "pods.json",
    )

    assert isinstance(result, LiveRunResult)
    assert result.winner == winner_name
    assert result.provisional is True

    # Non-winners torn down BEFORE classify (steps 5 then 6 in the sequence).
    terminated_before_classify = set(terminated_at_classify_time)
    assert "p2" in terminated_before_classify, "model2 must be torn down before classify"
    assert "p3" in terminated_before_classify, "model3 must be torn down before classify"
    assert "p4" in terminated_before_classify, "model4 must be torn down before classify"
    assert "p1" not in terminated_before_classify, "winner must NOT be torn down before classify"

    # ALL pods torn down by the end (winner terminated in guaranteed_teardown finally).
    assert set(term.terminated) == all_ids, "all 4 pods must be terminated by end"


# ===========================================================================
# Task 3 — Test 9 (b): bakeoff_fn raises → all torn down, error propagates
# ===========================================================================


def test_orchestrate_bakeoff_raises_tears_down_all(tmp_path: Path) -> None:
    """bakeoff_fn raises → guaranteed_teardown fires; all pods torn down; error propagates."""
    pods = _make_4_pods()
    term = _TrackingTerminator({"p1", "p2", "p3", "p4"})

    def bakeoff_fn(eligible: list[str]) -> Any:
        raise RuntimeError("bakeoff exploded")

    bakeoff_called = False

    def tracking_bakeoff(eligible: list[str]) -> Any:
        nonlocal bakeoff_called
        bakeoff_called = True
        return bakeoff_fn(eligible)

    with pytest.raises(RuntimeError, match="bakeoff exploded"):
        orchestrate_live_run(
            tmp_path,
            provisioner=_FakeProvisioner(pods),
            terminator=term,
            gate_fn=lambda name: _make_gate_result(name, passed=True),
            bakeoff_fn=tracking_bakeoff,
            classify_fn=lambda w: None,
            cost_tracker=_FakeCostTracker(),
            clock=_FakeClock(),
            readiness_cap_s=60.0,
            wall_cap_s=3600.0,
            on_fatal=lambda r: None,
            poll_s=0.01,
            is_ready_fn=lambda url: True,
            registry_path=tmp_path / "pods.json",
        )

    assert bakeoff_called, "bakeoff_fn must have been called"
    assert set(term.terminated) == {"p1", "p2", "p3", "p4"}, "all pods must be torn down"


# ===========================================================================
# Task 3 — Test 10 (c): <2 eligible → STOP, all torn down, no bakeoff
# ===========================================================================


def test_orchestrate_less_than_2_eligible_stops(tmp_path: Path) -> None:
    """Only 1 model passes the gate → InsufficientEligibleModels; bakeoff never called."""
    pods = _make_4_pods()
    term = _TrackingTerminator({"p1", "p2", "p3", "p4"})
    bakeoff_calls: list[list[str]] = []

    def _bakeoff_tracking_10(eligible: list[str]) -> _FakeBakeoffResult:
        bakeoff_calls.append(eligible)
        return _FakeBakeoffResult("model1")

    with pytest.raises(InsufficientEligibleModels):
        orchestrate_live_run(
            tmp_path,
            provisioner=_FakeProvisioner(pods),
            terminator=term,
            # Only model1 passes; 1 < 2 → STOP.
            gate_fn=lambda name: _make_gate_result(name, passed=(name == "model1")),
            bakeoff_fn=_bakeoff_tracking_10,
            classify_fn=lambda w: None,
            cost_tracker=_FakeCostTracker(),
            clock=_FakeClock(),
            readiness_cap_s=60.0,
            wall_cap_s=3600.0,
            on_fatal=lambda r: None,
            poll_s=0.01,
            is_ready_fn=lambda url: True,
            registry_path=tmp_path / "pods.json",
        )

    assert bakeoff_calls == [], "bakeoff must NOT be called when <2 eligible"
    assert set(term.terminated) == {"p1", "p2", "p3", "p4"}, "all pods torn down despite STOP"


# ===========================================================================
# Task 3 — Test 11 (d): cost_tracker.check_or_abort raises → all torn down
# ===========================================================================


def test_orchestrate_cost_abort_tears_down(tmp_path: Path) -> None:
    """check_or_abort raises mid-run → all provisioned pods torn down."""
    pods = _make_4_pods()
    term = _TrackingTerminator({"p1", "p2", "p3", "p4"})

    with pytest.raises(RuntimeError, match="cost ceiling"):
        orchestrate_live_run(
            tmp_path,
            provisioner=_FakeProvisioner(pods),
            terminator=term,
            gate_fn=lambda name: _make_gate_result(name, passed=True),
            bakeoff_fn=lambda eligible: _FakeBakeoffResult("model1"),
            classify_fn=lambda w: None,
            # Abort on the first check_or_abort call (fires after provisioning).
            cost_tracker=_FakeCostTracker(should_abort=True),
            clock=_FakeClock(),
            readiness_cap_s=60.0,
            wall_cap_s=3600.0,
            on_fatal=lambda r: None,
            # Large poll_s keeps the monitor asleep for the entire fast test;
            # _stop.wait() returns immediately when monitor.stop() is called on
            # completion.  Prevents a race where the monitor fires mid-registration
            # and only terminates a subset of pods.
            poll_s=3600.0,
            is_ready_fn=lambda url: True,
            registry_path=tmp_path / "pods.json",
        )

    assert set(term.terminated) == {"p1", "p2", "p3", "p4"}, "all pods torn down on cost abort"


# ===========================================================================
# Task 3 — Test 12 (e): readiness timeout → all torn down, no bakeoff
# ===========================================================================


def test_orchestrate_readiness_timeout_tears_down(tmp_path: Path) -> None:
    """ReadinessTimeout → guaranteed_teardown fires; no bakeoff."""
    pods = _make_4_pods()
    term = _TrackingTerminator({"p1", "p2", "p3", "p4"})
    bakeoff_calls: list[list[str]] = []

    def _bakeoff_tracking_12(eligible: list[str]) -> _FakeBakeoffResult:
        bakeoff_calls.append(eligible)
        return _FakeBakeoffResult("model1")

    with pytest.raises(ReadinessTimeout):
        orchestrate_live_run(
            tmp_path,
            provisioner=_FakeProvisioner(pods),
            terminator=term,
            gate_fn=lambda name: _make_gate_result(name, passed=True),
            bakeoff_fn=_bakeoff_tracking_12,
            classify_fn=lambda w: None,
            cost_tracker=_FakeCostTracker(),
            clock=_FakeClock(),
            readiness_cap_s=1.0,   # short cap
            wall_cap_s=3600.0,     # large wall cap — monitor must not fire
            on_fatal=lambda r: None,
            poll_s=0.01,
            poll_interval_s=2.0,   # > readiness_cap → times out in first poll
            is_ready_fn=lambda url: False,  # never ready
            registry_path=tmp_path / "pods.json",
        )

    assert bakeoff_calls == [], "bakeoff must not be called after readiness timeout"
    assert set(term.terminated) == {"p1", "p2", "p3", "p4"}, (
        "all pods torn down on readiness timeout"
    )


# ===========================================================================
# Task 3 — Test 13 (f): R3 monitor fires during blocked bakeoff_fn
# ===========================================================================


def test_monitor_fires_during_blocked_bakeoff(tmp_path: Path) -> None:
    """R3: cost/wall monitor fires while bakeoff_fn is blocked on an Event.

    Uses RealClock with a short wall_cap_s (150 ms) and a real blocking
    bakeoff_fn.  The monitor fires in its daemon thread, calls
    pool.terminate_all(), then on_fatal (which unblocks the main thread).
    Test duration: ~poll_s real seconds (≈ 50 ms).
    """

    class _Sentinel(Exception):
        pass

    unblock_event: threading.Event = threading.Event()
    on_fatal_reasons: list[str] = []

    def bakeoff_fn(eligible: list[str]) -> Any:
        # Block until the monitor fires and sets the event.
        unblock_event.wait(timeout=10.0)
        raise _Sentinel("main-thread: unblocked by monitor")

    def fake_on_fatal(reason: str) -> None:
        on_fatal_reasons.append(reason)
        unblock_event.set()  # unblock the main thread's bakeoff_fn
        # Return (do NOT raise here) — the Sentinel comes from bakeoff_fn.

    pods = [PodInfo("p1", "m1", "http://p1"), PodInfo("p2", "m2", "http://p2")]
    term = _TrackingTerminator({"p1", "p2"})

    with pytest.raises(_Sentinel, match="unblocked by monitor"):
        orchestrate_live_run(
            tmp_path,
            provisioner=_FakeProvisioner(pods),
            terminator=term,
            gate_fn=lambda name: _make_gate_result(name, passed=True),
            bakeoff_fn=bakeoff_fn,
            classify_fn=lambda w: None,
            cost_tracker=_FakeCostTracker(),
            clock=RealClock(),     # real clock so monitor measures real elapsed time
            readiness_cap_s=60.0,
            wall_cap_s=0.15,       # 150 ms — fires after ~1 poll
            on_fatal=fake_on_fatal,
            poll_s=0.05,           # 50 ms poll — fast enough for a unit test
            is_ready_fn=lambda url: True,
            registry_path=tmp_path / "pods.json",
        )

    assert on_fatal_reasons, "monitor must have called on_fatal"
    # Monitor called pool.terminate_all() BEFORE on_fatal.
    assert term.terminated, "monitor must have terminated pods before on_fatal"


# ===========================================================================
# Task 3 — Test 14 (g): R5 incomplete marker raises before final teardown
# ===========================================================================


def _setup_r5_cycle(base: Path) -> Path:
    """Create a minimal 2026-rarr-shaped cycle with a corpus snapshot."""
    cycle = base / "cycles" / "2026-rarr"
    (cycle / "prereg").mkdir(parents=True)
    (cycle / "prereg" / "stage2_manifest.json").write_text(
        json.dumps({"snapshot_hash": "testhash", "cost_ceiling_usd": 500})
    )
    # Corpus snapshot with 3 incidents.
    snap_dir = cycle / "corpora" / "owasp" / "testhash"
    snap_dir.mkdir(parents=True)
    (snap_dir / "incidents.json").write_text(
        json.dumps({"incidents": [{"id": "i1"}, {"id": "i2"}, {"id": "i3"}]})
    )
    return cycle


def test_r5_incomplete_marker_raises_before_teardown(tmp_path: Path) -> None:
    """R5: classify_fn that omits the coverage marker → verify raises; all pods still torn down."""
    cycle = _setup_r5_cycle(tmp_path)
    pods = [PodInfo("p1", "m1", "http://p1"), PodInfo("p2", "m2", "http://p2")]
    term = _TrackingTerminator({"p1", "p2"})

    def bad_classify_fn(winner: str) -> None:
        # Write labeled_incidents.json but intentionally omit classify_coverage.json.
        classify_dir = cycle / "classify"
        classify_dir.mkdir(exist_ok=True)
        (classify_dir / "labeled_incidents.json").write_text(
            '{"i1": "LLM02", "i2": "LLM01"}'
        )
        # classify_coverage.json is NOT written — verify_labeled_completeness will raise.

    with pytest.raises(LabeledIncidentsIncompleteError):
        orchestrate_live_run(
            cycle,
            provisioner=_FakeProvisioner(pods),
            terminator=term,
            gate_fn=lambda name: _make_gate_result(name, passed=True),
            bakeoff_fn=lambda eligible: _FakeBakeoffResult("m1"),
            classify_fn=bad_classify_fn,
            cost_tracker=_FakeCostTracker(),
            clock=_FakeClock(),
            readiness_cap_s=60.0,
            wall_cap_s=3600.0,
            on_fatal=lambda r: None,
            poll_s=0.01,
            is_ready_fn=lambda url: True,
            registry_path=tmp_path / "pods.json",
        )

    # ALL pods must still be torn down even though verify raised.
    assert set(term.terminated) == {"p1", "p2"}, (
        "guaranteed_teardown must terminate all pods even after R5 verify raises"
    )


# ===========================================================================
# Task 4 helpers — fake terminator module + cycle fixture
# ===========================================================================


class _FakeTerminatorModuleOK:
    """Terminator module that reports a clean account (no live pods, no orphans)."""

    def reconcile(self, registry_path: Any = None, *, execute: bool) -> dict[str, Any]:
        return {"orphans": [], "live_and_ours": [], "registered_gone": []}

    def terminate_pod(self, pod_id: str) -> bool:
        return True

    def list_live_pods(self) -> list[dict[str, Any]]:
        return []


class _FakeTerminatorModuleOrphan:
    """Terminator module that reports an orphan pod — preflight must raise."""

    def reconcile(self, registry_path: Any = None, *, execute: bool) -> dict[str, Any]:
        return {
            "orphans": [{"id": "orphan-xyz", "name": "unknown-pod"}],
            "live_and_ours": [],
            "registered_gone": [],
        }

    def list_live_pods(self) -> list[dict[str, Any]]:
        return [{"id": "orphan-xyz", "name": "unknown-pod"}]


def _make_cycle_fixture(base: Path) -> Path:
    """Minimal 2026-rarr cycle fixture that passes all preflight checks."""
    from engine.prereg.lock import write_lock
    from engine.prereg.manifest import PreregManifest

    cycle = base / "cycles" / "2026-rarr"
    prereg = cycle / "prereg"
    prereg.mkdir(parents=True)

    # Build a minimal schema-4 manifest with no reviewer fields set.
    manifest = PreregManifest(
        engine_version="0.1.0",
        engine_version_range_min="0.1.0",
        engine_version_range_max="0.1.99",
        cycle_id="2026-rarr",
        taxonomy_hash="a" * 64,
        snapshot_hash="synthetic-no-snapshot",
        primary_spec="negative_binomial_per_stratum",
        robustness_specs=(),
        flag_threshold_tau=0.5,
        statistic="weighted_cohens_kappa",
        measurability_minimum=10,
        prior_scale=0.5,
        concentration_shape=5.0,
        concentration_rate=0.1,
        ess_fraction=0.4,
        meaningful_kappa_n=4,
        prng_seed=42,
        confidence_threshold=0.3,
        rubric_drafting_attestation=None,
        rubric_reviewer=None,
        statistical_reviewer=None,
        classifier_rule_hash=None,
        rubric_hash=None,
        post_hoc_register_path=None,
        schema_version=4,
        goldset_hash="c" * 64,
        prospective_power_target_kappa=0.40,
        prospective_power_confidence_level=0.95,
        prospective_power_1_minus_beta=0.80,
    )

    # Persist manifest as JSON (use to_dict which is JSON-safe).
    (prereg / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2))

    # Write the valid manifest lock.
    write_lock(manifest, prereg / "manifest.lock")

    # Minimal bakeoff_grid.json (content doesn't matter for preflight, just must exist).
    grid_content = json.dumps({
        "configs": [
            {
                "name": "model1",
                "model_id": "test/model1",
                "revision_sha": "abc123",
                "gpu_type": "NVIDIA H200",
                "gpu_count": 4,
            }
        ],
        "selection": {"lockbox_fraction": 0.5, "seed": 42, "alpha": 0.05, "min_cell": 5},
    })
    (prereg / "bakeoff_grid.json").write_text(grid_content)

    # Pre-create grid_lock.json so preflight doesn't fail on missing lock.
    import hashlib as _hashlib
    grid_sha = _hashlib.sha256(grid_content.encode()).hexdigest()
    (prereg / "grid_lock.json").write_text(
        json.dumps({"bakeoff_grid_sha256": grid_sha}, indent=2) + "\n"
    )

    # stage2_manifest.json for bakeoff cost ceiling (dry-run reads it).
    (prereg / "stage2_manifest.json").write_text(
        json.dumps({"cost_ceiling_usd": 500, "abort_factor": 1.2,
                    "snapshot_hash": "synthetic-no-snapshot"})
    )

    return cycle


# ===========================================================================
# Task 4 — Test 15 (a): dry-run never invokes the provisioner
# ===========================================================================


def test_live_run_cli_dry_run_no_provision(tmp_path: Path) -> None:
    """execute=False → preflight + dry-run; the real provisioner is NEVER invoked."""
    cycle = _make_cycle_fixture(tmp_path)

    class _SpyProvisioner:
        deploy_called = False

        def deploy(self) -> list[PodInfo]:
            _SpyProvisioner.deploy_called = True
            return []

    live_run_cli(
        cycle,
        execute=False,
        _terminator_module=_FakeTerminatorModuleOK(),
        _provisioner=_SpyProvisioner(),
    )

    assert not _SpyProvisioner.deploy_called, (
        "provisioner.deploy() must NEVER be called when execute=False"
    )


# ===========================================================================
# Task 4 — Test 16 (b): bad manifest.lock raises before any provision
# ===========================================================================


def test_live_run_cli_bad_lock_raises_before_provision(tmp_path: Path) -> None:
    """Preflight raises on a tampered manifest.lock; provisioner is never called."""
    cycle = _make_cycle_fixture(tmp_path)

    # Tamper the manifest.lock with a wrong hash.
    (cycle / "prereg" / "manifest.lock").write_text(
        json.dumps({"manifest_hash": "deadbeef" * 8})
    )

    class _SpyProvisioner:
        deploy_called = False

        def deploy(self) -> list[PodInfo]:
            _SpyProvisioner.deploy_called = True
            return []

    with pytest.raises((ValueError, RuntimeError)):
        live_run_cli(
            cycle,
            execute=False,
            _terminator_module=_FakeTerminatorModuleOK(),
            _provisioner=_SpyProvisioner(),
        )

    assert not _SpyProvisioner.deploy_called, (
        "provisioner.deploy() must not be called when manifest.lock is bad"
    )


# ===========================================================================
# Task 4 — Test 17 (c): orphan pod raises before any provision
# ===========================================================================


def test_live_run_cli_orphan_raises_before_provision(tmp_path: Path) -> None:
    """Preflight raises when reconcile reports an orphan; provisioner never called."""
    cycle = _make_cycle_fixture(tmp_path)

    class _SpyProvisioner:
        deploy_called = False

        def deploy(self) -> list[PodInfo]:
            _SpyProvisioner.deploy_called = True
            return []

    with pytest.raises(RuntimeError, match="orphan"):
        live_run_cli(
            cycle,
            execute=False,
            _terminator_module=_FakeTerminatorModuleOrphan(),
            _provisioner=_SpyProvisioner(),
        )

    assert not _SpyProvisioner.deploy_called, (
        "provisioner.deploy() must not be called when orphan pods are detected"
    )


# ===========================================================================
# Safety-review fixes — Test 18: CRITICAL-1 incremental registration
# ===========================================================================


def test_incremental_registration_survives_mid_provision_crash(tmp_path: Path) -> None:
    """CRITICAL-1: on_pod_created callback writes p1 to durable registry before
    deploy() raises; guaranteed_teardown terminates p1 (pool is not empty)."""
    registry = tmp_path / "pods.json"
    term = _FakeTerminator(live={"p1"})
    registry_at_crash: list[str] = []

    class _CrashAfterFirstPodProvisioner:
        def deploy(self, on_pod_created: Any = None) -> list[PodInfo]:
            # Register p1 via callback — incremental, before p2 is created.
            if on_pod_created is not None:
                on_pod_created("p1", "n1")
            # Snapshot durable registry right after callback, before the raise.
            if registry.exists():
                data: list[dict[str, Any]] = json.loads(registry.read_text())
                registry_at_crash.extend(e["pod_id"] for e in data if "pod_id" in e)
            # Crash before provisioning p2.
            raise RuntimeError("crash mid-provision after p1")

    with pytest.raises(RuntimeError, match="crash mid-provision"):
        orchestrate_live_run(
            tmp_path,
            provisioner=_CrashAfterFirstPodProvisioner(),
            terminator=term,
            gate_fn=lambda name: _make_gate_result(name, passed=True),
            bakeoff_fn=lambda eligible: _FakeBakeoffResult("n1"),
            classify_fn=lambda w: None,
            cost_tracker=_FakeCostTracker(),
            clock=_FakeClock(),
            readiness_cap_s=60.0,
            wall_cap_s=3600.0,
            on_fatal=lambda r: None,
            poll_s=0.01,
            is_ready_fn=lambda url: True,
            registry_path=registry,
        )

    # (a) p1 was written to the durable registry at crash time via on_pod_created.
    assert "p1" in registry_at_crash, (
        "p1 must be in durable registry immediately after on_pod_created callback "
        "(before deploy() raises) — binds CRITICAL-1 incremental-registration path"
    )
    # (b) guaranteed_teardown tore down p1 — pool was NOT empty at crash time.
    assert "p1" in term.terminated, (
        "guaranteed_teardown must terminate p1 when deploy() raises mid-way "
        "(if pool were empty, p1 would have leaked)"
    )


# ===========================================================================
# Safety-review fixes — Test 19: CRITICAL-2 preflight uses real terminator
# ===========================================================================


def test_preflight_calls_real_reconcile_not_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CRITICAL-2: live_run_cli without _terminator_module passes _tmod (resolved)
    to _preflight, not None; reconcile is called and no AttributeError occurs."""
    import sys

    reconcile_calls: list[bool] = []

    class _SpyTmod:
        def reconcile(self, registry_path: Any = None, *, execute: bool) -> dict[str, Any]:
            reconcile_calls.append(True)
            return {"orphans": [], "live_and_ours": [], "registered_gone": []}

        def terminate_pod(self, pod_id: str) -> bool:
            return True

        def list_live_pods(self) -> list[Any]:
            return []

    spy = _SpyTmod()
    # Inject the spy as tools.terminate_runpod so live_run_cli picks it up when
    # _terminator_module is None (the production path).
    monkeypatch.setitem(sys.modules, "tools.terminate_runpod", spy)
    tools_mod = sys.modules.get("tools")
    if tools_mod is not None:
        monkeypatch.setattr(tools_mod, "terminate_runpod", spy, raising=False)

    cycle = _make_cycle_fixture(tmp_path)
    # Production path: no _terminator_module injected → live_run_cli resolves it.
    live_run_cli(cycle, execute=False)

    assert reconcile_calls, (
        "preflight must call reconcile on the resolved _tmod, not on None "
        "(the bug: _preflight was passed terminator_module=_terminator_module=None)"
    )


# ===========================================================================
# Safety-review fixes — Test 20: IMPORTANT-3 lied-success non-winner stays tracked
# ===========================================================================


def test_non_winner_lied_success_stays_in_registry(tmp_path: Path) -> None:
    """IMPORTANT-3: _pool_terminate_one re-verifies after terminate_pod; a lied-success
    pod stays in the durable registry so the final terminate_all can catch it."""
    registry = tmp_path / "pods.json"
    term = _FakeTerminatorLieAboutSuccess()
    pool = PodLeasePool(term, registry_path=registry)
    pool.register("p1", "pod-one")
    pool.register("p2", "pod-two")

    # p1 is the "non-winner" being early-terminated; the terminator lies about success.
    with pytest.raises(RuntimeError):
        _pool_terminate_one(pool, "p1")

    # p1 must remain in in-memory pool — NOT silently dropped.
    assert "p1" in pool._registered, (
        "p1 must remain in pool._registered when terminate_pod lied about success"
    )
    # p1 must remain in durable registry so the final terminate_all can catch it.
    data: list[dict[str, Any]] = json.loads(registry.read_text())
    pod_ids_in_registry = {e.get("pod_id") for e in data}
    assert "p1" in pod_ids_in_registry, (
        "p1 must remain in durable registry when terminate_pod lied about success"
    )
