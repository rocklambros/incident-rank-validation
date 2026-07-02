"""Unit tests for tools/terminate_runpod.py — all HTTP calls are mocked.

The module exposes two module-level primitives that tests monkeypatch:
    tools.terminate_runpod._http_get
    tools.terminate_runpod._http_delete

load_secret is also monkeypatched so the `pass` binary is never invoked.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

import tools.terminate_runpod as trm
from tools.terminate_runpod import (
    TerminateError,
    reconcile,
    terminate_all_registered,
    terminate_pod,
)

FAKE_KEY = "test-runpod-api-key-xyz"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeResponse:
    """Minimal stand-in for an httpx.Response."""

    status_code: int
    _body: Any = field(default_factory=dict)

    def json(self) -> Any:
        return self._body

    @property
    def text(self) -> str:
        return json.dumps(self._body)


def _fake_load_secret(pass_name: str, env_var: str) -> str:  # noqa: ARG001
    return FAKE_KEY


def _write_registry(tmp_path: Path, entries: list[dict[str, Any]]) -> Path:
    p = tmp_path / "runpod_pods.json"
    p.write_text(json.dumps(entries, indent=2))
    return p


# ---------------------------------------------------------------------------
# Test 1 — terminate_pod issues DELETE to the correct URL with Bearer auth
# ---------------------------------------------------------------------------


class TestTerminatePodDeleteCall:
    def test_correct_url_and_bearer_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, dict[str, str]]] = []

        def mock_delete(
            url: str, headers: dict[str, str]
        ) -> FakeResponse:
            calls.append((url, dict(headers)))
            return FakeResponse(status_code=200, _body={})

        monkeypatch.setattr(trm, "load_secret", _fake_load_secret)
        monkeypatch.setattr(trm, "_http_delete", mock_delete)

        result = terminate_pod("mypodid123")

        assert result is True
        assert len(calls) == 1
        url, headers = calls[0]
        assert url == "https://rest.runpod.io/v1/pods/mypodid123"
        assert headers["Authorization"] == f"Bearer {FAKE_KEY}"


# ---------------------------------------------------------------------------
# Test 2 — terminate_pod treats HTTP 404 as success (idempotent)
# ---------------------------------------------------------------------------


class TestTerminatePod404:
    def test_404_returns_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(trm, "load_secret", _fake_load_secret)
        monkeypatch.setattr(
            trm,
            "_http_delete",
            lambda url, headers: FakeResponse(
                status_code=404, _body={"error": "pod not found"}
            ),
        )

        result = terminate_pod("already-gone-pod")

        assert result is True


# ---------------------------------------------------------------------------
# Test 3 — terminate_pod raises TerminateError on HTTP 500
# ---------------------------------------------------------------------------


class TestTerminatePod500:
    def test_500_raises_terminate_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(trm, "load_secret", _fake_load_secret)
        monkeypatch.setattr(
            trm,
            "_http_delete",
            lambda url, headers: FakeResponse(
                status_code=500, _body={"error": "internal server error"}
            ),
        )

        with pytest.raises(TerminateError, match="server error"):
            terminate_pod("some-pod-id")


# ---------------------------------------------------------------------------
# Test 4 — terminate_all_registered(execute=False) issues ZERO delete calls
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_no_delete_calls_and_correct_would_terminate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        delete_calls: list[str] = []

        def spy_delete(url: str, headers: dict[str, str]) -> FakeResponse:
            delete_calls.append(url)
            return FakeResponse(status_code=200, _body={})

        monkeypatch.setattr(trm, "load_secret", _fake_load_secret)
        monkeypatch.setattr(trm, "_http_delete", spy_delete)

        registry = _write_registry(
            tmp_path,
            [
                {
                    "name": "qwen3-235b",
                    "model_id": "Qwen/Qwen3-235B-A22B",
                    "pod_id": "pod-aaa",
                },
                {
                    "name": "llama-405b",
                    "model_id": "meta-llama/Llama-3.1-405B",
                    "pod_id": "pod-bbb",
                },
            ],
        )

        result = terminate_all_registered(registry, execute=False)

        # No HTTP calls must have been made
        assert delete_calls == []
        assert len(result["would_terminate"]) == 2
        assert result["terminated"] == []
        assert result["errors"] == []
        pod_ids_would = {e["pod_id"] for e in result["would_terminate"]}
        assert pod_ids_would == {"pod-aaa", "pod-bbb"}


# ---------------------------------------------------------------------------
# Test 5 — terminate_all_registered(execute=True) calls DELETE once per
#           guarded pod and SKIPS a pod whose name is not in allow-prefixes
# ---------------------------------------------------------------------------


class TestSafeDeleteGuard:
    def test_execute_deletes_guarded_skips_unguarded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        delete_calls: list[str] = []

        def mock_delete(url: str, headers: dict[str, str]) -> FakeResponse:
            delete_calls.append(url)
            return FakeResponse(status_code=200, _body={})

        monkeypatch.setattr(trm, "load_secret", _fake_load_secret)
        monkeypatch.setattr(trm, "_http_delete", mock_delete)

        registry = _write_registry(
            tmp_path,
            [
                {
                    "name": "qwen3-235b",
                    "model_id": "Qwen/Qwen3-235B-A22B",
                    "pod_id": "pod-guarded",
                },
                {
                    # Name not in DEFAULT_ALLOW — must be skipped
                    "name": "customer-prod-db",
                    "model_id": "some/model",
                    "pod_id": "pod-unguarded",
                },
            ],
        )

        result = terminate_all_registered(registry, execute=True)

        # Exactly one DELETE, for the guarded pod
        assert len(delete_calls) == 1
        assert "pod-guarded" in delete_calls[0]

        assert len(result["terminated"]) == 1
        assert result["terminated"][0]["pod_id"] == "pod-guarded"

        assert len(result["skipped_by_guard"]) == 1
        assert result["skipped_by_guard"][0]["pod_id"] == "pod-unguarded"


# ---------------------------------------------------------------------------
# Test 6 — reconcile flags a live pod NOT in registry as orphan; does NOT
#           terminate it even when execute=True
# ---------------------------------------------------------------------------


class TestReconcileOrphan:
    def test_orphan_flagged_and_not_terminated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        delete_calls: list[str] = []

        def mock_delete(url: str, headers: dict[str, str]) -> FakeResponse:
            delete_calls.append(url)
            return FakeResponse(status_code=200, _body={})

        registered_id = "reg-pod-111"
        orphan_id = "orphan-pod-999"

        live_pods = [
            {
                "id": registered_id,
                "name": "qwen3-235b",
                "desiredStatus": "RUNNING",
                "costPerHr": 14.0,
            },
            {
                "id": orphan_id,
                "name": "some-unrelated-workload",
                "desiredStatus": "RUNNING",
                "costPerHr": 3.5,
            },
        ]

        def mock_get(url: str, headers: dict[str, str]) -> FakeResponse:
            return FakeResponse(status_code=200, _body=live_pods)

        monkeypatch.setattr(trm, "load_secret", _fake_load_secret)
        monkeypatch.setattr(trm, "_http_get", mock_get)
        monkeypatch.setattr(trm, "_http_delete", mock_delete)

        registry = _write_registry(
            tmp_path,
            [
                {
                    "name": "qwen3-235b",
                    "model_id": "Qwen/Qwen3-235B-A22B",
                    "pod_id": registered_id,
                }
            ],
        )

        result = reconcile(registry, execute=True)

        # Orphan is flagged
        assert len(result["orphans"]) == 1
        assert result["orphans"][0]["id"] == orphan_id

        # Orphan is NEVER terminated — DELETE only for registered+guarded pod
        terminated_ids = {e["pod_id"] for e in result["terminated"]}
        assert orphan_id not in terminated_ids

        # No delete call touches the orphan pod URL
        for url in delete_calls:
            assert orphan_id not in url

    def test_orphan_not_terminated_in_dry_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Orphan is never auto-terminated regardless of execute flag."""
        delete_calls: list[str] = []

        monkeypatch.setattr(
            trm,
            "_http_delete",
            lambda url, h: delete_calls.append(url) or FakeResponse(200),  # type: ignore[func-returns-value]
        )
        monkeypatch.setattr(trm, "load_secret", _fake_load_secret)
        monkeypatch.setattr(
            trm,
            "_http_get",
            lambda url, h: FakeResponse(
                status_code=200,
                _body=[{"id": "orphan-xyz", "name": "external", "desiredStatus": "RUNNING"}],
            ),
        )

        registry = _write_registry(tmp_path, [])  # empty — everything live is orphan

        result = reconcile(registry, execute=False)

        assert len(result["orphans"]) == 1
        assert delete_calls == []


# ---------------------------------------------------------------------------
# Test 7 — registry parsing tolerates missing / empty file (no crash)
# ---------------------------------------------------------------------------


class TestRegistryEdgeCases:
    def test_missing_registry_returns_empty_plan_no_crash(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(trm, "load_secret", _fake_load_secret)
        missing = tmp_path / "does_not_exist.json"

        result = terminate_all_registered(missing, execute=False)

        assert result["would_terminate"] == []
        assert result["terminated"] == []
        assert result["skipped_by_guard"] == []
        assert result["errors"] == []

    def test_empty_registry_file_returns_empty_plan_no_crash(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(trm, "load_secret", _fake_load_secret)
        empty = tmp_path / "pods.json"
        empty.write_text("")

        result = terminate_all_registered(empty, execute=False)

        assert result["would_terminate"] == []
        assert result["terminated"] == []

    def test_whitespace_only_registry_file_no_crash(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(trm, "load_secret", _fake_load_secret)
        ws_only = tmp_path / "pods.json"
        ws_only.write_text("   \n\n  ")

        result = terminate_all_registered(ws_only, execute=False)

        assert result["would_terminate"] == []


# ---------------------------------------------------------------------------
# Test 8 — CLI: --pod-id with unregistered pod blocks without --force (C-1)
# ---------------------------------------------------------------------------


class TestCliUnregisteredPodGuard:
    """Covers C-1: untracked pod must not be deleted without --force."""

    def test_unregistered_pod_without_force_exits1_no_delete(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--pod-id <unregistered> --execute (no --force) → SystemExit(1), DELETE never called."""
        delete_calls: list[str] = []

        def spy_delete(url: str, headers: dict[str, str]) -> FakeResponse:
            delete_calls.append(url)
            return FakeResponse(status_code=200, _body={})

        registry = _write_registry(tmp_path, [])  # empty — pod is untracked
        monkeypatch.setattr(trm, "load_secret", _fake_load_secret)
        monkeypatch.setattr(trm, "_http_delete", spy_delete)
        monkeypatch.setattr(
            "sys.argv",
            [
                "terminate_runpod.py",
                "--pod-id", "untracked-pod-xyz",
                "--execute",
                "--registry", str(registry),
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            trm._cli_main()

        assert exc_info.value.code == 1
        assert delete_calls == [], "DELETE must NOT be called when guard fires"

    def test_unregistered_pod_with_force_calls_delete_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--pod-id <unregistered> --execute --force → guard bypassed, DELETE called once."""
        delete_calls: list[str] = []

        def spy_delete(url: str, headers: dict[str, str]) -> FakeResponse:
            delete_calls.append(url)
            return FakeResponse(status_code=200, _body={})

        def mock_get(url: str, headers: dict[str, str]) -> FakeResponse:
            return FakeResponse(status_code=200, _body=[])

        registry = _write_registry(tmp_path, [])
        monkeypatch.setattr(trm, "load_secret", _fake_load_secret)
        monkeypatch.setattr(trm, "_http_delete", spy_delete)
        monkeypatch.setattr(trm, "_http_get", mock_get)
        monkeypatch.setattr(
            "sys.argv",
            [
                "terminate_runpod.py",
                "--pod-id", "untracked-pod-xyz",
                "--execute",
                "--force",
                "--registry", str(registry),
            ],
        )

        trm._cli_main()  # must not raise

        assert len(delete_calls) == 1
        assert "untracked-pod-xyz" in delete_calls[0]

    def test_registered_allowed_pod_executes_without_force(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--pod-id <registered-allowed> --execute → guard passes, DELETE called once."""
        delete_calls: list[str] = []

        def spy_delete(url: str, headers: dict[str, str]) -> FakeResponse:
            delete_calls.append(url)
            return FakeResponse(status_code=200, _body={})

        def mock_get(url: str, headers: dict[str, str]) -> FakeResponse:
            return FakeResponse(status_code=200, _body=[])

        registry = _write_registry(
            tmp_path,
            [
                {
                    "name": "qwen3-235b",
                    "model_id": "Qwen/Qwen3-235B-A22B",
                    "pod_id": "reg-pod-abc",
                }
            ],
        )
        monkeypatch.setattr(trm, "load_secret", _fake_load_secret)
        monkeypatch.setattr(trm, "_http_delete", spy_delete)
        monkeypatch.setattr(trm, "_http_get", mock_get)
        monkeypatch.setattr(
            "sys.argv",
            [
                "terminate_runpod.py",
                "--pod-id", "reg-pod-abc",
                "--execute",
                "--registry", str(registry),
            ],
        )

        trm._cli_main()  # must not raise

        assert len(delete_calls) == 1
        assert "reg-pod-abc" in delete_calls[0]

    def test_dry_run_pod_id_makes_no_http_calls(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--pod-id without --execute → dry-run, zero DELETE and zero GET calls."""
        delete_calls: list[str] = []
        get_calls: list[str] = []

        monkeypatch.setattr(
            trm,
            "_http_delete",
            lambda url, h: delete_calls.append(url) or FakeResponse(200),  # type: ignore[func-returns-value]
        )
        monkeypatch.setattr(
            trm,
            "_http_get",
            lambda url, h: get_calls.append(url) or FakeResponse(200, []),  # type: ignore[func-returns-value]
        )

        registry = _write_registry(tmp_path, [])
        monkeypatch.setattr(trm, "load_secret", _fake_load_secret)
        monkeypatch.setattr(
            "sys.argv",
            [
                "terminate_runpod.py",
                "--pod-id", "some-pod",
                "--registry", str(registry),
                # --execute intentionally absent
            ],
        )

        trm._cli_main()  # dry-run, no execute

        assert delete_calls == []
        assert get_calls == []


# ---------------------------------------------------------------------------
# Test 9 — terminate_all_registered continues past httpx.ConnectError (I-1)
# ---------------------------------------------------------------------------


class TestTerminateAllContinuesOnNetworkError:
    """Covers I-1: raw httpx network errors must not abort the per-pod loop."""

    def test_connect_error_recorded_loop_continues_to_remaining_pods(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Pod raising httpx.ConnectError is recorded in errors; subsequent pods still terminate."""
        call_order: list[str] = []

        def mock_delete(url: str, headers: dict[str, str]) -> FakeResponse:
            if "pod-bad" in url:
                raise httpx.ConnectError("simulated network failure")
            call_order.append(url)
            return FakeResponse(status_code=200, _body={})

        monkeypatch.setattr(trm, "load_secret", _fake_load_secret)
        monkeypatch.setattr(trm, "_http_delete", mock_delete)

        registry = _write_registry(
            tmp_path,
            [
                {
                    "name": "qwen3-235b",
                    "model_id": "Qwen/Qwen3-235B-A22B",
                    "pod_id": "pod-bad",
                },
                {
                    "name": "llama-405b",
                    "model_id": "meta-llama/Llama-3.1-405B",
                    "pod_id": "pod-good",
                },
            ],
        )

        result = terminate_all_registered(registry, execute=True)

        # Failing pod is recorded in errors, not silently dropped
        assert len(result["errors"]) == 1
        assert result["errors"][0]["pod_id"] == "pod-bad"
        assert "simulated network failure" in result["errors"][0]["error"]

        # Second pod still terminates despite the first failing
        assert len(result["terminated"]) == 1
        assert result["terminated"][0]["pod_id"] == "pod-good"
        assert any("pod-good" in url for url in call_order)
