"""RunPod API client for Stage-2 LLM classification."""
from __future__ import annotations

import contextlib
import os
import threading
from dataclasses import dataclass
from typing import Any, Protocol


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
    """Thread-safe RunPod client using OpenAI-compatible chat API."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint_id: str | None = None,
        model_name: str = "",
        timeout_seconds: float = 300.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        self._endpoint_id = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID", "")
        if not self._api_key:
            raise RunPodError("RUNPOD_API_KEY not set")
        if not self._endpoint_id:
            raise RunPodError("RUNPOD_ENDPOINT_ID not set")
        self._base_url = f"https://api.runpod.ai/v2/{self._endpoint_id}"
        self._model_name = model_name
        self._timeout = timeout_seconds
        self._local = threading.local()
        self._clients: list[Any] = []
        self._lock = threading.Lock()

    def _get_client(self) -> Any:
        import httpx

        client = getattr(self._local, "client", None)
        if client is None:
            client = httpx.Client(
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
            self._local.client = client
            with self._lock:
                self._clients.append(client)
        return client

    def run_sync(self, prompt: str, seed: int) -> RunPodResponse:
        import httpx

        client = self._get_client()
        payload = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.0,
            "seed": seed,
        }
        try:
            resp = client.post(
                f"{self._base_url}/openai/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise RunPodError(f"RunPod HTTP error: {e}") from e

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise RunPodError(f"RunPod returned no choices: {data}")

        output_text = choices[0].get("message", {}).get("content", "").strip()
        return RunPodResponse(
            output_text=output_text,
            job_id=str(data.get("id", "")),
            execution_time_ms=0.0,
        )

    def close(self) -> None:
        with self._lock:
            for client in self._clients:
                with contextlib.suppress(Exception):
                    client.close()
            self._clients.clear()
