"""RunPod API client for Stage-2 LLM classification."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RunPodResponse:
    output_text: str
    job_id: str
    execution_time_ms: float


class RunPodError(RuntimeError):
    pass


class RunPodClient(Protocol):
    def run_sync(self, prompt: str, seed: int) -> RunPodResponse: ...
    def close(self) -> None: ...


class HttpRunPodClient:
    """Production RunPod client using httpx."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint_id: str | None = None,
        timeout_seconds: float = 300.0,
    ) -> None:
        import httpx

        self._api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        self._endpoint_id = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID", "")
        if not self._api_key:
            raise RunPodError("RUNPOD_API_KEY not set")
        if not self._endpoint_id:
            raise RunPodError("RUNPOD_ENDPOINT_ID not set")
        self._base_url = f"https://api.runpod.ai/v2/{self._endpoint_id}"
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=timeout_seconds,
        )

    def run_sync(self, prompt: str, seed: int) -> RunPodResponse:
        import httpx

        payload = {
            "input": {
                "prompt": prompt,
                "max_tokens": 256,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": seed,
            }
        }
        try:
            resp = self._client.post(f"{self._base_url}/runsync", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise RunPodError(f"RunPod HTTP error: {e}") from e

        data = resp.json()
        if data.get("status") == "FAILED":
            raise RunPodError(f"RunPod job failed: {data.get('error', 'unknown')}")
        return RunPodResponse(
            output_text=str(data.get("output", "")),
            job_id=str(data.get("id", "")),
            execution_time_ms=float(data.get("executionTime", 0.0)),
        )

    def close(self) -> None:
        self._client.close()
