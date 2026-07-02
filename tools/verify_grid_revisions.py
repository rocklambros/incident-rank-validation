#!/usr/bin/env python3
"""R2: Verify each bakeoff_grid.json revision SHA is a real HF commit (network, one-time).

GETs https://huggingface.co/api/models/{model_id}/revision/{sha} with Bearer auth
for each of the 4 model configs and asserts HTTP 200.  On success prints a
revision_provenance block to add to bakeoff_grid.json.

NOT for CI — this script has network access and calls the HF API.
The offline CI test (test_bakeoff_grid.py) only checks 40-hex format.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx

GRID = Path("projects/owasp-llm/cycles/2026-rarr/prereg/bakeoff_grid.json")
HF_API = "https://huggingface.co/api/models"


def load_secret(pass_name: str, env_var: str) -> str:
    val = os.environ.get(env_var, "")
    if val:
        return val
    result = subprocess.run(
        ["pass", "show", pass_name], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def main() -> None:
    hf_token = load_secret("huggingface/token", "HF_TOKEN")
    data = json.loads(GRID.read_text())
    configs = data["configs"]

    headers = {"Authorization": f"Bearer {hf_token}"}
    failures: list[str] = []

    for cfg in configs:
        name = cfg["name"]
        model_id = cfg["model_id"]
        sha = cfg["revision_sha"]
        url = f"{HF_API}/{model_id}/revision/{sha}"
        print(f"  Checking {name} ({model_id} @ {sha[:12]}...)...", end=" ", flush=True)
        resp = httpx.get(url, headers=headers, timeout=30.0)
        if resp.status_code == 200:
            print("OK 200")
        else:
            print(f"FAILED {resp.status_code}")
            failures.append(f"{name}: {model_id}@{sha} → HTTP {resp.status_code}")

    if failures:
        print("\nFAILED — the following SHAs did not return HTTP 200:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        print(
            "\nSTOP: do NOT proceed. Re-resolve the SHAs from HF and never invent one.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nAll 4 revision SHAs verified (HTTP 200).")
    provenance = {
        "resolved_utc": "2026-07-01",
        "method": "HF api model_info sha (main HEAD at resolution), pinned as immutable commit id",
        "verified": True,
    }
    print("\nrevision_provenance block to add to bakeoff_grid.json:")
    print(json.dumps({"revision_provenance": provenance}, indent=2))


if __name__ == "__main__":
    main()
