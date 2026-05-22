"""Credential loader: env var first, then `pass` (Unix password manager)."""
from __future__ import annotations

import os
import subprocess


def load_secret(pass_name: str, env_var: str) -> str:
    value = os.environ.get(env_var)
    if value:
        return value

    try:
        result = subprocess.run(
            ["pass", "show", pass_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except FileNotFoundError:
        pass

    raise RuntimeError(
        f"Secret '{pass_name}' not found. "
        f"Set {env_var} or run `pass insert {pass_name}`."
    )
