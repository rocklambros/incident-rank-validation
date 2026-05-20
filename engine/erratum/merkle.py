"""Merkle chain for append-only integrity (M16 / HANDOFF v2.5 §6.11)."""

from __future__ import annotations

import hashlib
import json

GENESIS = "0" * 64


def chain_link(prev_hash: str, payload: dict[str, object]) -> str:
    """Compute SHA-256 hash for a single chain entry."""
    canonical = json.dumps(
        {"prev": prev_hash, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_chain(entries: list[dict[str, object]]) -> str:
    """Walk the chain, raising on any break. Returns terminal hash."""
    prev = GENESIS
    for i, e in enumerate(entries):
        payload = {k: v for k, v in e.items() if k != "chain_hash"}
        expected = chain_link(prev, payload)
        actual = e.get("chain_hash")
        if actual != expected:
            raise ValueError(
                f"chain break at entry {i}: expected {expected}, got {actual}"
            )
        prev = expected
    return prev
