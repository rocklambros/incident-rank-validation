from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engine.classify.runpod_client import (
    RunPodClient,
    RunPodError,
    RunPodResponse,
)


@dataclass
class MockHttpResponse:
    status_code: int
    _json: dict[str, Any]

    def json(self) -> dict[str, Any]:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class MockRunPodClient(RunPodClient):
    """In-process mock that returns canned responses without HTTP."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    def run_sync(self, prompt: str, seed: int) -> RunPodResponse:
        if not self._responses:
            raise RunPodError("No more mock responses")
        resp = self._responses.pop(0)
        self._call_count += 1
        if resp.get("status") == "FAILED":
            raise RunPodError(f"Job failed: {resp}")
        return RunPodResponse(
            output_text=resp["output"],
            job_id=resp.get("id", f"mock-{self._call_count}"),
            execution_time_ms=resp.get("executionTime", 100.0),
        )

    def close(self) -> None:
        pass


class TestRunPodClient:
    def test_successful_response(self) -> None:
        client = MockRunPodClient([{
            "output": '{"entry_id": "LLM01", "confidence": 0.9}',
            "id": "job-1",
            "executionTime": 500.0,
        }])
        resp = client.run_sync("classify this incident", seed=42)
        assert resp.output_text == '{"entry_id": "LLM01", "confidence": 0.9}'
        assert resp.job_id == "job-1"
        assert resp.execution_time_ms == 500.0

    def test_failed_response_raises(self) -> None:
        client = MockRunPodClient([{"status": "FAILED", "error": "GPU OOM"}])
        try:
            client.run_sync("classify this", seed=42)
            raise AssertionError("should raise")
        except RunPodError:
            pass

    def test_empty_responses_raises(self) -> None:
        client = MockRunPodClient([])
        try:
            client.run_sync("classify this", seed=42)
            raise AssertionError("should raise")
        except RunPodError:
            pass

    def test_response_dataclass(self) -> None:
        r = RunPodResponse(output_text="test", job_id="j1", execution_time_ms=100.0)
        assert r.output_text == "test"
        assert r.job_id == "j1"
