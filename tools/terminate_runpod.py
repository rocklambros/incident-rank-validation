#!/usr/bin/env python3
"""Idempotent teardown for RunPod vLLM pods.

Programmatic API (importable by the U8 orchestrator):
    list_live_pods()                       -> list[dict]
    terminate_pod(pod_id)                  -> bool  (404 = success, 5xx = TerminateError)
    terminate_all_registered(registry, *, name_allow_prefixes, execute) -> dict
    reconcile(registry, *, execute)        -> dict

HTTP layer is exposed via module-level functions _http_get / _http_delete so
tests can monkeypatch them without any real network traffic:

    monkeypatch.setattr("tools.terminate_runpod._http_get",    fake_get)
    monkeypatch.setattr("tools.terminate_runpod._http_delete", fake_delete)
    monkeypatch.setattr("tools.terminate_runpod.load_secret",  lambda *_: "key")
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import httpx

REST_BASE = "https://rest.runpod.io/v1"
DEFAULT_ALLOW: tuple[str, ...] = (
    "qwen3-235b",
    "llama-405b",
    "deepseek-v3",
    "qwen25-72b",
)


# ---------------------------------------------------------------------------
# Typed exception
# ---------------------------------------------------------------------------


class TerminateError(RuntimeError):
    """Raised on non-recoverable HTTP errors during pod termination."""


# ---------------------------------------------------------------------------
# Secret loading — identical to deploy_runpod.py; no import at module level
# ---------------------------------------------------------------------------


def load_secret(pass_name: str, env_var: str) -> str:
    """Check env-var first, fall back to `pass show <pass_name>`."""
    val = os.environ.get(env_var, "")
    if val:
        return val
    result = subprocess.run(
        ["pass", "show", pass_name], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Internal helpers — build auth headers (reads secret lazily)
# ---------------------------------------------------------------------------


def _headers() -> dict[str, str]:
    api_key = load_secret("runpod/api-key", "RUNPOD_API_KEY")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Mockable HTTP primitives — monkeypatch these in tests
# ---------------------------------------------------------------------------


def _http_get(url: str, headers: dict[str, str]) -> httpx.Response:
    return httpx.get(url, headers=headers, timeout=30.0)


def _http_delete(url: str, headers: dict[str, str]) -> httpx.Response:
    return httpx.delete(url, headers=headers, timeout=30.0)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def _load_registry(registry_path: Path) -> list[dict[str, Any]]:
    """Return registry entries; empty list if file is missing or empty."""
    if not registry_path.exists():
        return []
    text = registry_path.read_text().strip()
    if not text:
        return []
    data: list[dict[str, Any]] = json.loads(text)
    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_live_pods() -> list[dict[str, Any]]:
    """GET /pods — return parsed pod list (id, name, desiredStatus, …)."""
    h = _headers()
    resp = _http_get(f"{REST_BASE}/pods", h)
    if resp.status_code != 200:
        raise TerminateError(
            f"list_live_pods failed ({resp.status_code}): {resp.text}"
        )
    data: Any = resp.json()
    if isinstance(data, list):
        return cast(list[dict[str, Any]], data)
    # Some API versions wrap in {"pods": [...]}
    return cast(list[dict[str, Any]], data.get("pods", []))


def terminate_pod(pod_id: str) -> bool:
    """DELETE /pods/{pod_id}.

    Idempotent: HTTP 404 (already gone) is treated as SUCCESS (returns True).
    Network/5xx errors raise TerminateError.
    """
    h = _headers()
    resp = _http_delete(f"{REST_BASE}/pods/{pod_id}", h)

    if resp.status_code == 404:
        # Pod already gone — idempotent success
        return True

    if 200 <= resp.status_code < 300:
        return True

    if resp.status_code >= 500:
        raise TerminateError(
            f"terminate_pod({pod_id}) server error"
            f" ({resp.status_code}): {resp.text}"
        )

    # 4xx other than 404 (e.g. 401, 403)
    raise TerminateError(
        f"terminate_pod({pod_id}) client error"
        f" ({resp.status_code}): {resp.text}"
    )


def terminate_all_registered(
    registry_path: Path = Path("tools/runpod_pods.json"),
    *,
    name_allow_prefixes: tuple[str, ...] = DEFAULT_ALLOW,
    execute: bool,
) -> dict[str, Any]:
    """Terminate registry pods that pass the safe-delete name guard.

    Args:
        registry_path: Path to the JSON pod registry.
        name_allow_prefixes: Only pods whose ``name`` starts with one of
            these prefixes may be terminated.  Anything else is skipped and
            logged — we never nuke an unrelated pod on the account.
        execute: If False (DRY-RUN, the default you should force callers to
            opt into) no DELETE call is issued; the plan is returned.

    Returns:
        {would_terminate, terminated, skipped_by_guard, already_gone, errors}
    """
    pods = _load_registry(registry_path)
    summary: dict[str, Any] = {
        "would_terminate": [],
        "terminated": [],
        "skipped_by_guard": [],
        "already_gone": [],
        "errors": [],
    }

    for entry in pods:
        name: str = entry.get("name", "")
        pod_id: str = entry.get("pod_id", "")
        allowed = any(name.startswith(pfx) for pfx in name_allow_prefixes)

        if not allowed:
            print(
                f"  SKIP (guard): name={name!r} not in allow-prefixes"
                f" — refusing to touch pod {pod_id}"
            )
            summary["skipped_by_guard"].append({"name": name, "pod_id": pod_id})
            continue

        if not execute:
            summary["would_terminate"].append({"name": name, "pod_id": pod_id})
            continue

        # execute=True path
        try:
            terminate_pod(pod_id)
            summary["terminated"].append({"name": name, "pod_id": pod_id})
            print(f"  TERMINATED: {name} ({pod_id})")
        except (TerminateError, httpx.HTTPError) as exc:
            summary["errors"].append(
                {"name": name, "pod_id": pod_id, "error": str(exc)}
            )
            print(f"  ERROR: {name} ({pod_id}): {exc}", file=sys.stderr)

    return summary


def reconcile(
    registry_path: Path = Path("tools/runpod_pods.json"),
    *,
    execute: bool,
) -> dict[str, Any]:
    """Cross-check registry against live RunPod pods.

    Categories:
    - registered_gone: in registry but NOT live (already cleaned up)
    - live_and_ours:   in registry AND live (will terminate if execute=True,
                       subject to name guard)
    - orphans:         live but NOT in registry — REPORTED LOUDLY, never
                       auto-terminated; a human decides

    Returns a summary dict.
    """
    pods = _load_registry(registry_path)
    registered_by_id: dict[str, dict[str, Any]] = {
        e["pod_id"]: e for e in pods
    }

    live = list_live_pods()
    live_ids: set[str] = {p["id"] for p in live}

    registered_gone: list[dict[str, Any]] = []
    live_and_ours: list[dict[str, Any]] = []
    orphans: list[dict[str, Any]] = []

    for entry in pods:
        pid = entry["pod_id"]
        if pid in live_ids:
            live_and_ours.append(entry)
        else:
            registered_gone.append(entry)

    for pod in live:
        pid = pod["id"]
        if pid not in registered_by_id:
            orphans.append(pod)

    # Loud orphan warning — these must never be auto-terminated
    if orphans:
        print(
            "\n*** ORPHAN PODS DETECTED — NOT auto-terminated ***",
            file=sys.stderr,
        )
        for o in orphans:
            print(
                f"    ORPHAN  id={o['id']}"
                f"  name={o.get('name', '?')}"
                f"  status={o.get('desiredStatus', '?')}",
                file=sys.stderr,
            )
        print(
            "*** A human must decide what to do with orphans ***\n",
            file=sys.stderr,
        )

    summary: dict[str, Any] = {
        "registered_gone": registered_gone,
        "live_and_ours": live_and_ours,
        "orphans": orphans,
        "terminated": [],
        "skipped_by_guard": [],
        "errors": [],
    }

    if execute:
        for entry in live_and_ours:
            name_e: str = entry.get("name", "")
            pod_id_e: str = entry["pod_id"]
            if not any(name_e.startswith(pfx) for pfx in DEFAULT_ALLOW):
                print(
                    f"  SKIP (guard): name={name_e!r} not in allow-prefixes",
                    file=sys.stderr,
                )
                summary["skipped_by_guard"].append(
                    {"name": name_e, "pod_id": pod_id_e}
                )
                continue
            try:
                terminate_pod(pod_id_e)
                summary["terminated"].append(
                    {"name": name_e, "pod_id": pod_id_e}
                )
                print(f"  TERMINATED: {name_e} ({pod_id_e})")
            except (TerminateError, httpx.HTTPError) as exc:
                summary["errors"].append(
                    {"name": name_e, "pod_id": pod_id_e, "error": str(exc)}
                )
                print(
                    f"  ERROR: {name_e} ({pod_id_e}): {exc}", file=sys.stderr
                )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Terminate RunPod vLLM pods safely.\n"
            "Default (no flags): dry-run reconcile — prints plan, NO deletes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually terminate pods (default: dry-run, no DELETEs issued).",
    )
    parser.add_argument(
        "--pod-id",
        dest="pod_id",
        metavar="ID",
        help="Terminate a specific pod by ID.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Bypass the name guard (only valid with --pod-id). "
            "Prints a loud warning before proceeding."
        ),
    )
    parser.add_argument(
        "--registry",
        default="tools/runpod_pods.json",
        metavar="PATH",
        help="Path to pod registry JSON (default: tools/runpod_pods.json).",
    )
    args = parser.parse_args()
    registry = Path(args.registry)

    if args.pod_id:
        # Single-pod path
        if not args.execute:
            print(
                f"DRY-RUN: would terminate pod {args.pod_id}"
                " (pass --execute to actually do it)"
            )
        else:
            if args.force:
                print(
                    f"WARNING: --force bypasses name guard for pod {args.pod_id}",
                    file=sys.stderr,
                )
            else:
                pods = _load_registry(registry)
                entry = next(
                    (p for p in pods if p["pod_id"] == args.pod_id), None
                )
                if entry is None:
                    # Pod not in registry — block unless --force was given.
                    # Untracked pods must never be deleted silently.
                    print(
                        f"ERROR: pod {args.pod_id} is not in the registry."
                        " Use --force to terminate an untracked pod.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                name = entry.get("name", "")
                if not any(name.startswith(pfx) for pfx in DEFAULT_ALLOW):
                    print(
                        f"ERROR: pod {args.pod_id} has name {name!r} which"
                        " is not in allow-prefixes. Use --force to override.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            try:
                ok = terminate_pod(args.pod_id)
                print(f"{'OK' if ok else 'FAILED'}: pod {args.pod_id}")
            except TerminateError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                sys.exit(1)
    else:
        # Reconcile path (default)
        if not args.execute:
            print("=== DRY-RUN RECONCILE (no DELETE calls will be made) ===")
        result = reconcile(registry, execute=args.execute)
        print(f"  registered_gone : {len(result['registered_gone'])}")
        print(f"  live_and_ours   : {len(result['live_and_ours'])}")
        print(f"  orphans         : {len(result['orphans'])}")
        if args.execute:
            print(f"  terminated      : {len(result['terminated'])}")
            print(f"  errors          : {len(result['errors'])}")
        else:
            print(
                f"  would_terminate : {len(result['live_and_ours'])}"
                "  (pass --execute to act)"
            )

    # Final live count — only re-query when we actually issued deletes; skip on
    # dry-run (execute=False) to avoid a confusing network call that has no
    # bearing on a plan-only run.
    if args.execute:
        try:
            live_after = list_live_pods()
            print(f"\nverify: {len(live_after)} pods still live")
        except (TerminateError, httpx.HTTPError) as exc:
            print(
                f"\nWARNING: post-run verify query failed (degraded): {exc}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    _cli_main()
