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
        # ... bakeoff / classify ...
    # guarantee: terminate_all() was called in __exit__ no matter what
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
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
