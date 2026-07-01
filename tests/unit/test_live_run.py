"""Unit tests for engine/cli/live_run.py — Task 1 (PodLeasePool +
guaranteed_teardown).

ALL tests use FAKE terminators — NO network, NO RunPod pod.
The real tools/runpod_pods.json is NEVER touched; every test that exercises
the durable registry uses a tmp_path fixture.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.cli.live_run import (
    CompoundTerminateError,
    PodLeasePool,
    guaranteed_teardown,
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
