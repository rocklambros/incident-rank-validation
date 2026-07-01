"""SEV-1-CRITICAL pod-teardown core of the RARR live-run orchestrator.

Three independent teardown mechanisms guarantee no H200 pod is leaked
(a leak burns ~$65/hr):

  1. ``guaranteed_teardown`` context manager  — try/finally fires even on raise.
  2. ``atexit`` handler registered at first ``register()`` call — fires on
     process exit including uncaught exceptions.
  3. Durable on-disk registry (R1) — survives SIGKILL/os._exit; the standalone
     ``terminate_runpod.py --execute`` uses this registry as its backstop.

Usage
-----
    pool = PodLeasePool(terminator)
    with guaranteed_teardown(pool):
        pool.register(pod_id, name)   # writes durable registry immediately
        wait_until_ready(pod_urls, is_ready_fn=..., clock=clock, ...)
        # ... bakeoff / classify ...
    # guarantee: terminate_all() was called in __exit__ no matter what
"""
from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CompoundTerminateError(RuntimeError):
    """Raised when one or more registered pods survive after terminate_all().

    Sev-1 — a surviving pod burns ~$65/hr.  The durable registry (R1) is
    intentionally NOT cleared when this is raised, so ``terminate_runpod.py
    --execute`` can recover the surviving pods.
    """


class ReadinessTimeout(RuntimeError):
    """Raised by wait_until_ready when pods fail to become ready within the cap."""


# ---------------------------------------------------------------------------
# Protocols — injectable seams (no real network in tests)
# ---------------------------------------------------------------------------


class Terminator(Protocol):
    """Minimal protocol implemented by the real terminate_runpod module and by
    fake test doubles.  All methods must be safe to call with no network when
    faked.
    """

    def terminate_pod(self, pod_id: str) -> bool:  # 404 = success
        ...

    def list_live_ids(self) -> set[str]:
        ...


class Clock(Protocol):
    """Injected clock — prevents any real time.sleep() inside tests."""

    def now(self) -> float:
        ...

    def sleep(self, seconds: float) -> None:
        ...


# ---------------------------------------------------------------------------
# PodLeasePool
# ---------------------------------------------------------------------------


class PodLeasePool:
    """Track provisioned pods and guarantee teardown via three mechanisms.

    Parameters
    ----------
    terminator:
        Injectable terminator satisfying the ``Terminator`` protocol.
    registry_path:
        Durable on-disk registry for SIGKILL/os._exit recovery.
        Defaults to ``tools/runpod_pods.json`` (the real RunPod registry).
    """

    def __init__(
        self,
        terminator: Terminator,
        registry_path: Path = Path("tools/runpod_pods.json"),
    ) -> None:
        self._terminator = terminator
        self._registry_path = registry_path
        # pod_id -> name for every pod registered in this process instance.
        self._registered: dict[str, str] = {}
        self._atexit_registered = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, pod_id: str, name: str) -> None:
        """Register *pod_id* for teardown.

        Idempotent: re-registering the same ``pod_id`` is a no-op.

        R1 (durable registry): atomically appends ``{"name": name,
        "pod_id": pod_id}`` to the on-disk registry via write-temp +
        ``os.fsync`` + ``os.replace`` BEFORE updating in-memory state, so a
        second process reading the file before the readiness wait (or after a
        SIGKILL) can still recover the pod.
        """
        if pod_id in self._registered:
            return

        # R1: persist to disk FIRST, before touching in-memory state.
        self._append_to_registry(pod_id, name)
        self._registered[pod_id] = name

        if not self._atexit_registered:
            atexit.register(self._atexit_handler)
            self._atexit_registered = True

        logger.info("Registered pod %s (%s) for teardown", pod_id, name)

    def terminate_all(self) -> None:
        """Tear down all registered pods and verify none survive.

        Iterates a snapshot of the registered set, calls
        ``terminator.terminate_pod`` for each entry (collecting errors rather
        than raising immediately so all pods are attempted).

        R4: re-queries ``terminator.list_live_ids()`` after all DELETEs and
        raises ``CompoundTerminateError`` naming surviving pod_ids if any of
        our registered pods are still live.  The durable registry is NOT
        cleared unless the re-verify confirms all are gone.
        """
        pods_snapshot = dict(self._registered)
        errors: list[tuple[str, Exception]] = []

        for pod_id, name in pods_snapshot.items():
            try:
                result = self._terminator.terminate_pod(pod_id)
                logger.info(
                    "Teardown: pod %s (%s) -> %s",
                    pod_id,
                    name,
                    "ok" if result else "noop",
                )
            except Exception as exc:
                logger.error(
                    "Teardown error for pod %s (%s): %s", pod_id, name, exc
                )
                errors.append((pod_id, exc))

        # R4: re-verify via list_live_ids — catches silent API lies and
        # pods that terminate_pod failed to delete.
        live_ids = self._terminator.list_live_ids()
        survivors = set(pods_snapshot.keys()) & live_ids

        if survivors:
            err_summary = (
                f"; terminate errors: {[(pid, str(e)) for pid, e in errors]}"
                if errors
                else ""
            )
            msg = (
                f"SEV-1: {len(survivors)} registered pod(s) still live after"
                f" teardown: {sorted(survivors)}{err_summary}"
            )
            logger.critical(msg)
            # R4: do NOT clear the durable registry while pods are still live.
            raise CompoundTerminateError(msg)

        # All confirmed gone — safe to remove our entries from the registry
        # and clear in-memory state.
        self._remove_our_pods_from_registry()
        self._registered.clear()

    # ------------------------------------------------------------------
    # Durable registry helpers (R1)
    # ------------------------------------------------------------------

    def _read_registry(self) -> list[dict[str, Any]]:
        """Return current registry entries; empty list if missing or empty."""
        if not self._registry_path.exists():
            return []
        text = self._registry_path.read_text().strip()
        return list(json.loads(text)) if text else []

    def _atomic_write_registry(self, entries: list[dict[str, Any]]) -> None:
        """Write *entries* atomically: temp-file + fsync + os.replace."""
        dir_path = self._registry_path.parent
        dir_path.mkdir(parents=True, exist_ok=True)
        tmp_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=dir_path,
                delete=False,
                suffix=".tmp",
            ) as f:
                json.dump(entries, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
                tmp_name = f.name
            os.replace(tmp_name, str(self._registry_path))
        except Exception:
            if tmp_name and os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

    def _append_to_registry(self, pod_id: str, name: str) -> None:
        """Append {name, pod_id} to the registry (defensive dedup by pod_id)."""
        existing = self._read_registry()
        if not any(e.get("pod_id") == pod_id for e in existing):
            existing.append({"name": name, "pod_id": pod_id})
        self._atomic_write_registry(existing)

    def _remove_our_pods_from_registry(self) -> None:
        """Remove only OUR registered pod_ids from the registry.

        Other entries (e.g. from a previous process) are preserved.
        """
        our_ids = set(self._registered.keys())
        if not our_ids:
            return
        existing = self._read_registry()
        filtered = [e for e in existing if e.get("pod_id") not in our_ids]
        self._atomic_write_registry(filtered)

    # ------------------------------------------------------------------
    # atexit handler
    # ------------------------------------------------------------------

    def _atexit_handler(self) -> None:
        """Called at process exit — last-resort teardown (mechanism 2)."""
        logger.warning(
            "SEV-1: atexit teardown handler triggered — tearing down all registered pods"
        )
        try:
            self.terminate_all()
        except Exception as exc:
            logger.error("SEV-1: atexit teardown raised: %s", exc)


# ---------------------------------------------------------------------------
# guaranteed_teardown context manager
# ---------------------------------------------------------------------------


@contextmanager
def guaranteed_teardown(pool: PodLeasePool) -> Generator[None, None, None]:
    """Context manager that calls ``pool.terminate_all()`` in a finally block.

    Teardown is called REGARDLESS of whether the body raised or not.
    If the body raised an exception, that exception propagates.
    If teardown ALSO fails, the teardown error is attached as ``__context__``
    on the body exception (body exception remains the primary exception).
    """
    body_exc: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        body_exc = exc
    finally:
        teardown_exc: BaseException | None = None
        try:
            pool.terminate_all()
        except BaseException as te:
            teardown_exc = te

        if body_exc is not None:
            if teardown_exc is not None:
                # Chain teardown failure onto body exception as context so both
                # appear in the traceback.  Body exception is primary.
                body_exc.__context__ = teardown_exc
            raise body_exc
        elif teardown_exc is not None:
            raise teardown_exc
        # else: both succeeded — normal return.


# ---------------------------------------------------------------------------
# wait_until_ready
# ---------------------------------------------------------------------------


def wait_until_ready(
    pod_urls: dict[str, str],
    *,
    is_ready_fn: Callable[[str], bool],
    clock: Clock,
    readiness_cap_s: float,
    poll_interval_s: float,
) -> None:
    """Poll pod URLs until all report ready or the wall-time cap elapses.

    Parameters
    ----------
    pod_urls:
        ``{name: url}`` mapping for each pod to check.
    is_ready_fn:
        ``(url) -> bool`` — called for each not-yet-ready pod.  Must NOT
        perform real I/O in tests; inject a fake.
    clock:
        Injected clock with ``.now() -> float`` and ``.sleep(s) -> None``.
        Using an injected clock ensures no real ``time.sleep`` in tests.
    readiness_cap_s:
        Maximum wall-clock seconds before ``ReadinessTimeout`` is raised.
    poll_interval_s:
        Seconds (fake or real) to sleep between poll rounds.

    Raises
    ------
    ReadinessTimeout
        If one or more pods are not ready within *readiness_cap_s* seconds.
        The caller is responsible for tearing down the pool.
    """
    deadline = clock.now() + readiness_cap_s
    while True:
        not_ready = {
            name: url for name, url in pod_urls.items() if not is_ready_fn(url)
        }
        if not not_ready:
            return
        if clock.now() >= deadline:
            raise ReadinessTimeout(
                f"Pods not ready within {readiness_cap_s}s:"
                f" {sorted(not_ready.keys())}"
            )
        clock.sleep(poll_interval_s)


# ---------------------------------------------------------------------------
# RealClock — production implementation of Clock
# ---------------------------------------------------------------------------

import time as _time_module  # noqa: E402 (local import keeps tests free of real time)


class RealClock:
    """Monotonic real-time clock for production use."""

    def now(self) -> float:
        return _time_module.monotonic()

    def sleep(self, seconds: float) -> None:  # noqa: PLR6301
        _time_module.sleep(seconds)


# ---------------------------------------------------------------------------
# Task 3 data types + protocols
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PodInfo:
    """Minimal pod descriptor returned by a Provisioner."""

    pod_id: str
    name: str
    url: str


@dataclass(frozen=True)
class LiveRunResult:
    """Return value of orchestrate_live_run."""

    winner: str
    gate_results: dict[str, Any]
    bakeoff_result: Any
    total_cost: float
    provisional: bool = True


class InsufficientEligibleModels(RuntimeError):
    """Raised when fewer than 2 models pass the injection gate.

    The bake-off is meaningless without at least 2 candidates; STOP and tear
    down rather than crown a winner by default.
    """


class Provisioner(Protocol):
    """Deploy all candidate pods and return their info."""

    def deploy(self) -> list[PodInfo]:
        ...


class CostTrackerLike(Protocol):
    """Injectable cost-tracking seam (satisfied by the real CostTracker)."""

    @property
    def total_cost_usd(self) -> float:
        ...

    def check_or_abort(self) -> None:
        ...


# ---------------------------------------------------------------------------
# Default on_fatal (forces process exit; override in tests)
# ---------------------------------------------------------------------------


def _default_on_fatal(reason: str) -> None:
    """Force process exit on cost/wall-cap breach.

    ``os._exit`` is intentional: it bypasses atexit and finally blocks, so
    the durable registry (R1) is the recovery mechanism.  Override in tests
    to prevent killing pytest.
    """
    logger.critical("SEV-1: cost/wall-cap monitor forcing exit: %s", reason)
    os._exit(1)


# ---------------------------------------------------------------------------
# Background cost / wall-time monitor (R3)
# ---------------------------------------------------------------------------


class _WallMonitor(threading.Thread):
    """Daemon thread that enforces cost and wall-time ceilings.

    Sleeps for *poll_s* real seconds between checks (uses
    ``threading.Event.wait`` so the stop signal is honoured promptly).
    Checks elapsed time via the injected *clock* so tests can supply a
    ``RealClock`` with a short *wall_cap_s* without touching any fake clock.

    On breach:
      1. ``pool.terminate_all()`` — tear down all registered pods.
      2. ``on_fatal(reason)`` — default forces ``os._exit``; tests inject a
         recorder that sets an Event to unblock the blocked main thread.
    """

    def __init__(
        self,
        pool: PodLeasePool,
        cost_tracker: CostTrackerLike,
        clock: Clock,
        wall_cap_s: float,
        poll_s: float,
        on_fatal: Callable[[str], None],
        start_t: float,
    ) -> None:
        super().__init__(daemon=True, name="cost-wall-monitor")
        self._pool = pool
        self._cost_tracker = cost_tracker
        self._clock = clock
        self._wall_cap_s = wall_cap_s
        self._poll_s = poll_s
        self._on_fatal = on_fatal
        self._start_t = start_t
        self._stop = threading.Event()

    def stop(self) -> None:
        """Signal the monitor to stop after its current sleep."""
        self._stop.set()

    def run(self) -> None:
        while True:
            # Real sleep — interruptible by stop().  Returns True if stopped.
            if self._stop.wait(timeout=self._poll_s):
                return

            elapsed = self._clock.now() - self._start_t
            breach_reason: str | None = None

            if elapsed >= self._wall_cap_s:
                breach_reason = (
                    f"wall-time cap {self._wall_cap_s}s breached"
                    f" (elapsed={elapsed:.2f}s)"
                )

            if breach_reason is None:
                try:
                    self._cost_tracker.check_or_abort()
                except Exception as exc:  # noqa: BLE001
                    breach_reason = f"cost ceiling breached: {exc}"

            if breach_reason is not None:
                logger.critical(
                    "SEV-1 cost/wall monitor: %s — forcing teardown + on_fatal",
                    breach_reason,
                )
                try:
                    self._pool.terminate_all()
                except Exception as te:  # noqa: BLE001
                    logger.error(
                        "SEV-1 monitor: terminate_all raised: %s", te
                    )
                self._on_fatal(breach_reason)
                return  # on_fatal may not return (os._exit), but if it does, stop.


# ---------------------------------------------------------------------------
# PodLeasePool — additional methods for Task 3
# ---------------------------------------------------------------------------


def _pool_terminate_one(pool: PodLeasePool, pod_id: str) -> None:
    """Terminate a single registered pod and release it from the pool.

    On success the pod is removed from ``pool._registered`` and the durable
    registry.  On failure the pod is KEPT in the pool so ``terminate_all()``
    can retry in the ``finally`` block.

    This is a module-level function rather than a method to keep the class
    definition in Task 1 self-contained while still being tested via the same
    pool instance.
    """
    name = pool._registered.get(pod_id, pod_id)
    try:
        result = pool._terminator.terminate_pod(pod_id)
        logger.info(
            "Early teardown (non-winner): pod %s (%s) -> %s",
            pod_id,
            name,
            "ok" if result else "noop",
        )
    except Exception as exc:
        logger.error(
            "Early teardown error for pod %s (%s): %s"
            " — pod retained in pool for final teardown",
            pod_id,
            name,
            exc,
        )
        raise

    # Only release on success — failure leaves the pod in pool for retry.
    del pool._registered[pod_id]
    _pool_remove_single_from_registry(pool, pod_id)


def _pool_remove_single_from_registry(pool: PodLeasePool, pod_id: str) -> None:
    existing = pool._read_registry()
    filtered = [e for e in existing if e.get("pod_id") != pod_id]
    pool._atomic_write_registry(filtered)


# ---------------------------------------------------------------------------
# Helper functions for R5 snapshot-hash / labeled-ids reading
# ---------------------------------------------------------------------------


def _read_snapshot_hash_from_cycle(cycle_dir: Path) -> str:
    """Read snapshot_hash from prereg/stage2_manifest.json; sentinel if absent."""
    manifest_path = cycle_dir / "prereg" / "stage2_manifest.json"
    if not manifest_path.exists():
        return "synthetic-no-snapshot"
    data: dict[str, Any] = json.loads(manifest_path.read_text())
    return str(data.get("snapshot_hash", "synthetic-no-snapshot"))


def _read_labeled_ids_from_cycle(cycle_dir: Path) -> set[str]:
    """Read the set of incident IDs that have an in-scope label."""
    labeled_path = cycle_dir / "classify" / "labeled_incidents.json"
    if not labeled_path.exists():
        return set()
    data: Any = json.loads(labeled_path.read_text())
    if isinstance(data, dict):
        return set(data.keys())
    if isinstance(data, list):
        return {str(r["id"]) for r in data if isinstance(r, dict) and "id" in r}
    return set()


def _extract_winner_name(bakeoff_result: Any) -> str:
    """Extract the winner model name from a BakeoffResult (or duck-typed fake)."""
    if hasattr(bakeoff_result, "winner"):
        w = bakeoff_result.winner
        if hasattr(w, "name"):
            return str(w.name)
        return str(w)
    raise AttributeError(
        f"bakeoff_result {type(bakeoff_result).__name__!r} has no .winner attribute"
    )


# ---------------------------------------------------------------------------
# orchestrate_live_run (Task 3)
# ---------------------------------------------------------------------------


def orchestrate_live_run(
    cycle_dir: Path,
    *,
    provisioner: Provisioner,
    terminator: Terminator,
    gate_fn: Callable[[str], Any],
    bakeoff_fn: Callable[[list[str]], Any],
    classify_fn: Callable[[str], None],
    cost_tracker: CostTrackerLike,
    clock: Clock,
    readiness_cap_s: float,
    wall_cap_s: float,
    on_fatal: Callable[[str], None] | None = None,
    poll_s: float = 20.0,
    poll_interval_s: float = 5.0,
    is_ready_fn: Callable[[str], bool] | None = None,
    registry_path: Path | None = None,
) -> LiveRunResult:
    """Orchestrate Phase-3: provision → gate → bakeoff → classify → teardown.

    All external effects (provision, terminate, gate, bakeoff, classify, time)
    are INJECTED seams so the full sequence is deterministic and testable with
    NO GPU, NO network, and NO real sleep.

    Teardown is guaranteed by the ``guaranteed_teardown`` context manager
    (try/finally) and the pre-registered atexit handler.  The durable registry
    (R1) backstops SIGKILL/``os._exit`` scenarios.

    R3: a background daemon monitor fires ``on_fatal`` on cost or wall-time
    ceiling breach EVEN DURING a blocked bakeoff_fn/classify_fn.

    R5: after ``classify_fn`` and BEFORE the final teardown,
    ``verify_labeled_completeness`` is called to confirm the classify producer
    ran over the full corpus universe; a wrong-corpus marker raises
    ``LabeledIncidentsIncompleteError`` while the winner pod is still up.
    """
    from engine.calibrate.coverage import (
        LabeledIncidentsIncompleteError,  # noqa: F401 (re-raised, not caught)
        verify_labeled_completeness,
    )
    from engine.classify.injection_gate import filter_eligible_by_gate

    if on_fatal is None:
        on_fatal = _default_on_fatal

    if is_ready_fn is None:
        try:
            import httpx as _httpx

            def _http_ready_fn(url: str) -> bool:
                try:
                    resp = _httpx.get(f"{url}/health", timeout=5.0)
                    return resp.status_code == 200
                except Exception:  # noqa: BLE001
                    return False

            is_ready_fn = _http_ready_fn
        except ImportError:
            def _never_ready(url: str) -> bool:  # noqa: ARG001
                return False

            is_ready_fn = _never_ready

    _registry_path = (
        registry_path if registry_path is not None else Path("tools/runpod_pods.json")
    )
    pool = PodLeasePool(terminator, registry_path=_registry_path)
    start_t = clock.now()

    monitor = _WallMonitor(
        pool=pool,
        cost_tracker=cost_tracker,
        clock=clock,
        wall_cap_s=wall_cap_s,
        poll_s=poll_s,
        on_fatal=on_fatal,
        start_t=start_t,
    )
    monitor.start()

    winner_name: str = ""
    gate_results: dict[str, Any] = {}
    bakeoff_result: Any = None
    pods: list[PodInfo] = []

    try:
        with guaranteed_teardown(pool):
            # Step 1: Provision — register each pod IMMEDIATELY (R1 durable).
            pods = provisioner.deploy()
            pod_urls: dict[str, str] = {}
            for pod in pods:
                pool.register(pod.pod_id, pod.name)  # durable registry write
                pod_urls[pod.name] = pod.url
            logger.info(
                "Provisioned %d pod(s): %s",
                len(pods),
                [p.name for p in pods],
            )

            cost_tracker.check_or_abort()

            # Step 2: Wait until all pods are ready.
            wait_until_ready(
                pod_urls,
                is_ready_fn=is_ready_fn,
                clock=clock,
                readiness_cap_s=readiness_cap_s,
                poll_interval_s=poll_interval_s,
            )
            logger.info("All pods ready.")
            cost_tracker.check_or_abort()

            # Step 3: Gate each model; exclude any that fail injection-resistance.
            model_names = [pod.name for pod in pods]
            for name in model_names:
                gate_results[name] = gate_fn(name)

            eligible, excluded = filter_eligible_by_gate(model_names, gate_results)
            logger.info(
                "Gate: %d eligible, %d excluded: %s",
                len(eligible),
                len(excluded),
                excluded,
            )

            if len(eligible) < 2:
                raise InsufficientEligibleModels(
                    f"Only {len(eligible)} model(s) passed the injection gate"
                    f" (need ≥2 for a meaningful bake-off): {eligible}"
                )

            cost_tracker.check_or_abort()

            # Step 4: Bake-off → winner.
            bakeoff_result = bakeoff_fn(eligible)
            winner_name = _extract_winner_name(bakeoff_result)
            logger.info("Bake-off winner: %s", winner_name)

            # Step 5: Tear down non-winner pods immediately (pool-aware).
            for pod in pods:
                if pod.name != winner_name:
                    try:
                        _pool_terminate_one(pool, pod.pod_id)
                    except Exception:  # noqa: BLE001
                        # Pool retains the pod; terminate_all() will retry.
                        logger.warning(
                            "Non-winner pod %s teardown failed; retained for final teardown",
                            pod.pod_id,
                        )

            cost_tracker.check_or_abort()

            # Step 6: Classify winner (writes F-B classify_coverage.json marker).
            logger.info("Classifying winner: %s", winner_name)
            classify_fn(winner_name)

            # Step 7 (R5): Verify coverage marker IN-WINDOW while winner pod is up.
            snapshot_hash = _read_snapshot_hash_from_cycle(cycle_dir)
            labeled_ids = _read_labeled_ids_from_cycle(cycle_dir)
            verify_labeled_completeness(cycle_dir, snapshot_hash, labeled_ids)
            logger.info("Coverage marker verified (snapshot=%s).", snapshot_hash)

        # guaranteed_teardown has called pool.terminate_all() in __exit__.
        total_cost = cost_tracker.total_cost_usd
        logger.info(
            "Live run complete: winner=%s total_cost=$%.2f provisional=True",
            winner_name,
            total_cost,
        )
        return LiveRunResult(
            winner=winner_name,
            gate_results=gate_results,
            bakeoff_result=bakeoff_result,
            total_cost=total_cost,
            provisional=True,
        )
    finally:
        monitor.stop()


# ---------------------------------------------------------------------------
# Task 4: preflight (R2) + live_run_cli + CLI command
# ---------------------------------------------------------------------------


def _preflight(
    cycle_dir: Path,
    *,
    terminator_module: Any,
) -> None:
    """Offline preflight — runs unconditionally before any pod is provisioned.

    R2 checks (all FAIL-CLOSED):
      1. cycle_dir must be under cycles/2026-rarr (never bare cycles/2026).
      2. manifest.lock must verify against prereg/manifest.json.
      3. bakeoff_grid.json sha256 must match grid_lock.json (creating it on
         first run; the lock file must then be committed alongside).
      4. reconcile() must report 0 orphans AND 0 live registered pods.
    """
    from engine.prereg.lock import verify_lock
    from engine.prereg.rarr_lock import load_manifest

    # (1) Cycle-dir guard: must be under 2026-rarr, never bare 2026.
    resolved = cycle_dir.resolve()
    parts = resolved.parts
    if "2026-rarr" not in parts:
        raise ValueError(
            f"cycle_dir must be under a cycles/2026-rarr directory, got {cycle_dir}. "
            "NEVER write into cycles/2026/ (the locked 2026 cycle)."
        )
    if "2026" in parts and "2026-rarr" not in parts:
        raise ValueError(
            f"cycle_dir appears to be under cycles/2026/ (not 2026-rarr): {cycle_dir}"
        )

    # (2) Manifest lock verification.
    manifest_path = cycle_dir / "prereg" / "manifest.json"
    lock_path = cycle_dir / "prereg" / "manifest.lock"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found at {manifest_path}. "
            "Ensure the RARR pre-registration manifest is committed."
        )
    if not lock_path.exists():
        raise FileNotFoundError(
            f"manifest.lock not found at {lock_path}. "
            "Run the lock step (write_lock) before the live run."
        )
    manifest = load_manifest(manifest_path)
    verify_lock(manifest, lock_path)  # raises ValueError on mismatch
    logger.info("Manifest lock verified: %s", lock_path)

    # (3) Bakeoff-grid sha256 lock.
    grid_path = cycle_dir / "prereg" / "bakeoff_grid.json"
    grid_lock_path = cycle_dir / "prereg" / "grid_lock.json"
    if not grid_path.exists():
        raise FileNotFoundError(
            f"bakeoff_grid.json not found at {grid_path}."
        )
    grid_sha = hashlib.sha256(grid_path.read_bytes()).hexdigest()
    if not grid_lock_path.exists():
        # First preflight: record the sha and write the lock.
        grid_lock_path.write_text(
            json.dumps({"bakeoff_grid_sha256": grid_sha}, indent=2) + "\n"
        )
        logger.info(
            "Grid lock CREATED at %s (sha=%s…). "
            "Commit this file alongside bakeoff_grid.json.",
            grid_lock_path,
            grid_sha[:16],
        )
    else:
        stored: dict[str, Any] = json.loads(grid_lock_path.read_text())
        stored_sha = str(stored.get("bakeoff_grid_sha256", ""))
        if stored_sha != grid_sha:
            raise ValueError(
                f"bakeoff_grid.json sha256 mismatch:\n"
                f"  stored={stored_sha}\n"
                f"  actual={grid_sha}\n"
                "The grid has been modified after locking. "
                "Restore the original grid or delete grid_lock.json to re-lock "
                "(requires a new premortem)."
            )
        logger.info("Grid lock verified (sha=%s…).", grid_sha[:16])

    # (4) Reconcile — fail-closed on any live pods.
    result: dict[str, Any] = terminator_module.reconcile(execute=False)
    orphans: list[Any] = result.get("orphans", [])
    live_and_ours: list[Any] = result.get("live_and_ours", [])
    if orphans or live_and_ours:
        raise RuntimeError(
            f"Preflight FAIL-CLOSED: {len(orphans)} orphan pod(s) and "
            f"{len(live_and_ours)} live registered pod(s) detected. "
            "Manual cleanup required BEFORE the live run to avoid a double-burn. "
            "Run: python tools/terminate_runpod.py --execute  "
            "See docs/RUNBOOK.md for recovery procedure."
        )
    logger.info("Reconcile: 0 orphans, 0 live pods. Preflight passed.")


def live_run_cli(
    cycle_dir: Path,
    *,
    execute: bool,
    _terminator_module: Any = None,
    _provisioner: Any = None,
) -> None:
    """Live-run entry point: offline preflight → optional DRY-RUN or real run.

    If ``execute=False`` (the default-safe mode): run preflight and print the
    dry-run plan; provision NOTHING.

    If ``execute=True``: run preflight, then call ``orchestrate_live_run`` with
    the real seams (RunPod API, injection gate, bakeoff, classify pipeline).

    Parameters
    ----------
    cycle_dir:
        Root of the locked 2026-rarr cycle (must contain prereg/ with
        manifest.json, manifest.lock, bakeoff_grid.json, grid_lock.json).
    execute:
        ``False`` (default): preflight + dry-run, no pods provisioned.
        ``True``: real run (provisions H200 pods; costs ~$65/hr per pod).
    _terminator_module:
        Test injection point.  Defaults to ``tools.terminate_runpod``.
    _provisioner:
        Test injection point for the provisioner (spy for execute=False tests).
        Defaults to the real ``_RealProvisioner`` when execute=True.
    """
    # Lazy import of real seams to avoid secret-loading at import time.
    _tmod: Any = _terminator_module
    if _tmod is None:
        from tools import terminate_runpod as _tmod

    # --- Preflight (unconditional, offline) ---
    _preflight(cycle_dir, terminator_module=_terminator_module)

    if not execute:
        # DRY-RUN: print the plan and stop.
        from engine.prereg.bakeoff_grid import load_bakeoff_grid
        from engine.prereg.rarr_lock import load_manifest

        manifest = load_manifest(cycle_dir / "prereg" / "manifest.json")
        grid_path = cycle_dir / "prereg" / "bakeoff_grid.json"
        try:
            model_configs = load_bakeoff_grid(grid_path)
            model_list = [mc.name for mc in model_configs]
        except Exception:  # noqa: BLE001
            model_list = ["(grid unreadable)"]

        manifest_data: dict[str, Any] = json.loads(
            (cycle_dir / "prereg" / "stage2_manifest.json").read_text()
        ) if (cycle_dir / "prereg" / "stage2_manifest.json").exists() else {}
        cost_ceiling = manifest_data.get("cost_ceiling_usd", "unknown")

        print(
            f"\n=== DRY-RUN: live Phase-3 orchestration plan ===\n"
            f"  cycle_dir    : {cycle_dir}\n"
            f"  manifest     : cycle_id={manifest.cycle_id}\n"
            f"  models       : {model_list}\n"
            f"  cost_ceiling : ${cost_ceiling}\n"
            f"  execute=False: NOTHING will be provisioned.\n"
            f"  Pass --execute to launch the real run.\n"
        )
        return

    # --- Real run (execute=True) ---
    from engine.classify.cost_tracker import CostTracker
    from engine.prereg.bakeoff_grid import load_bakeoff_grid
    from tools.deploy_runpod import IMAGE, MODELS, create_pod_rest

    # Build cost tracker from stage2_manifest.json
    manifest_data_real: dict[str, Any] = json.loads(
        (cycle_dir / "prereg" / "stage2_manifest.json").read_text()
    )
    ceiling = float(manifest_data_real.get("cost_ceiling_usd", 500))
    abort_factor = float(manifest_data_real.get("abort_factor", 1.2))
    cost_tracker = CostTracker(ceiling_usd=ceiling, _abort_factor=abort_factor)

    # Shared URL store — provisioner writes, gate_fn reads.
    _url_store: dict[str, str] = {}

    # Real provisioner
    if _provisioner is None:
        from tools.deploy_runpod import load_secret as _dp_load_secret

        api_key = _dp_load_secret("runpod/api-key", "RUNPOD_API_KEY")
        hf_token = _dp_load_secret("huggingface/token", "HF_TOKEN")

        class _RealProvisioner:
            def deploy(self) -> list[PodInfo]:
                pods: list[PodInfo] = []
                for model in MODELS:
                    pod_name = f"classify-{model['name']}"
                    env = {
                        "HF_TOKEN": hf_token,
                        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                    }
                    data = create_pod_rest(
                        api_key=api_key,
                        name=pod_name,
                        image=IMAGE,
                        gpu_type=model["gpu_type"],
                        gpu_count=model["gpu_count"],
                        container_disk_gb=model["container_disk_gb"],
                        vllm_cmd=model["vllm_cmd"],
                        env=env,
                    )
                    pod_id = str(data["id"])
                    url = f"https://{pod_id}-8000.proxy.runpod.net"
                    _url_store[model["name"]] = url
                    pods.append(PodInfo(pod_id=pod_id, name=model["name"], url=url))
                return pods

        provisioner: Any = _RealProvisioner()
    else:
        provisioner = _provisioner

    # Real terminator adapter
    class _TerminatorAdapter:
        def terminate_pod(self, pod_id: str) -> bool:
            return bool(_tmod.terminate_pod(pod_id))

        def list_live_ids(self) -> set[str]:
            return {p["id"] for p in _tmod.list_live_pods()}

    terminator = _TerminatorAdapter()

    # Real gate_fn (reads pod URL from _url_store, populated by provisioner)
    grid_path = cycle_dir / "prereg" / "bakeoff_grid.json"
    model_configs = load_bakeoff_grid(grid_path)
    name_to_config = {mc.name: mc for mc in model_configs}
    rubric_json = (cycle_dir / "prereg" / "rubric.json").read_text()
    manifest_data_seed: dict[str, Any] = json.loads(
        (cycle_dir / "prereg" / "bakeoff_grid.json").read_text()
    )
    seed = int(
        manifest_data_seed.get("selection", {}).get("seed", 42)
    )

    def gate_fn(name: str) -> Any:
        from engine.classify.injection_gate import run_injection_gate
        from engine.classify.runpod_client import HttpRunPodClient

        url = _url_store[name]
        mc = name_to_config[name]
        client = HttpRunPodClient(url, model_name=mc.model_id)
        return run_injection_gate(
            client, name, mc.revision_sha, rubric_json, seed=seed
        )

    # Real bakeoff_fn
    from engine.cli.bakeoff import bakeoff_cmd

    def bakeoff_fn(eligible: list[str]) -> Any:
        return bakeoff_cmd(cycle_dir, execute=True)

    # Real classify_fn (wraps classify_real pipeline)
    def classify_fn(winner: str) -> None:
        # Set env vars so the classify pipeline finds the winner pod.
        winner_url = _url_store.get(winner, "")
        os.environ["RUNPOD_WINNER_URL"] = winner_url
        os.environ["RUNPOD_WINNER_NAME"] = winner
        from engine.cli.pipeline import classify_real as _classify_real_cmd

        ctx = _classify_real_cmd.make_context(
            "classify-real", [str(cycle_dir)]
        )
        with ctx:
            _classify_real_cmd.invoke(ctx)

    clock = RealClock()

    orchestrate_live_run(
        cycle_dir,
        provisioner=provisioner,
        terminator=terminator,
        gate_fn=gate_fn,
        bakeoff_fn=bakeoff_fn,
        classify_fn=classify_fn,
        cost_tracker=cost_tracker,
        clock=clock,
        readiness_cap_s=3600.0,   # 60-min readiness cap
        wall_cap_s=21600.0,        # 6-hour total wall cap
        poll_s=20.0,               # monitor polls every 20s
        poll_interval_s=30.0,      # readiness re-poll every 30s
    )


# ---------------------------------------------------------------------------
# Click CLI command (registered in engine/cli/main.py)
# ---------------------------------------------------------------------------

try:
    import click as _click

    @_click.command("live-run")
    @_click.argument(
        "cycle_dir",
        type=_click.Path(exists=False, file_okay=False, path_type=Path),
    )
    @_click.option(
        "--execute/--no-execute",
        default=False,
        help=(
            "Actually provision RunPod pods and run Phase-3 "
            "(default: dry-run + preflight only, NO pods)."
        ),
    )
    def live_run_cmd(cycle_dir: Path, execute: bool) -> None:
        """Provision → gate → bake-off → classify → teardown (live Phase-3 run).

        Always runs offline preflight first.  Without --execute, prints a
        dry-run plan and exits without provisioning anything.
        """
        live_run_cli(cycle_dir, execute=execute)

except ImportError:
    # click not installed in test-only environments; define a stub.
    live_run_cmd = None  # type: ignore[assignment]
