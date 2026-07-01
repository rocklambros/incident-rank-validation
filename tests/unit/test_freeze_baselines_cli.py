"""Unit tests for freeze_baselines CLI hardening (T9, U3 Cluster C).

Tests:
1. cycles/ guard: raises on paths that resolve inside any cycles/ dir.
2. Evasion table: .. traversal, symlink into cycles/, relative paths.
3. Write-once guard: refuses to overwrite differing SHA256SUMS.
4. Write-once guard: --force allows overwrite.
"""

from __future__ import annotations

import os
from pathlib import Path

import click
import pytest

from engine.cli.freeze_baselines import _assert_not_in_cycles, _check_write_once

# ---------------------------------------------------------------------------
# Cycles/ guard tests (evasion table)
# ---------------------------------------------------------------------------


def test_guard_direct_cycles_subdir(tmp_path: Path) -> None:
    """Directly addressing a path inside cycles/ raises."""
    cycles_dir = tmp_path / "projects" / "cycles" / "2026"
    cycles_dir.mkdir(parents=True)
    with pytest.raises(click.ClickException, match="cycles"):
        _assert_not_in_cycles(cycles_dir)


def test_guard_cycles_itself(tmp_path: Path) -> None:
    """A path whose name is 'cycles' raises."""
    cycles = tmp_path / "cycles"
    cycles.mkdir()
    with pytest.raises(click.ClickException, match="cycles"):
        _assert_not_in_cycles(cycles)


def test_guard_dotdot_traversal(tmp_path: Path) -> None:
    """.. traversal that resolves into cycles/ is blocked."""
    # Create: tmp/projects/cycles/2026/baselines/
    cycles_deep = tmp_path / "projects" / "cycles" / "2026" / "baselines"
    cycles_deep.mkdir(parents=True)
    # Build a path with .. that resolves into cycles/
    escaped = cycles_deep / ".." / ".." / "cycles" / "evasion"
    # This resolves to tmp/projects/cycles/evasion — still inside cycles/
    with pytest.raises(click.ClickException, match="cycles"):
        _assert_not_in_cycles(escaped)


def test_guard_symlink_into_cycles(tmp_path: Path) -> None:
    """Symlink pointing into cycles/ is resolved and blocked."""
    cycles_target = tmp_path / "projects" / "cycles" / "2026"
    cycles_target.mkdir(parents=True)
    # Create a symlink that points inside cycles/
    symlink_path = tmp_path / "innocent_looking_dir"
    os.symlink(cycles_target, symlink_path)
    with pytest.raises(click.ClickException, match="cycles"):
        _assert_not_in_cycles(symlink_path)


def test_guard_relative_path_into_cycles(tmp_path: Path) -> None:
    """Relative path that resolves into cycles/ is blocked."""
    original_cwd = Path.cwd()
    cycles_dir = tmp_path / "repo" / "cycles" / "2026"
    cycles_dir.mkdir(parents=True)
    # Change to a dir adjacent to cycles and reference it relatively
    adjacent = tmp_path / "repo" / "baselines"
    adjacent.mkdir()
    os.chdir(adjacent)
    try:
        # "../cycles/2026" relative to adjacent => resolves to tmp/repo/cycles/2026
        with pytest.raises(click.ClickException, match="cycles"):
            _assert_not_in_cycles(Path("../cycles/2026"))
    finally:
        os.chdir(original_cwd)  # restore original CWD


def test_guard_safe_path_not_blocked(tmp_path: Path) -> None:
    """A safe path outside any cycles/ dir does not raise."""
    safe = tmp_path / "projects" / "owasp-llm" / "baselines" / "2026"
    safe.mkdir(parents=True)
    # Should not raise
    _assert_not_in_cycles(safe)


def test_guard_nested_safe_path(tmp_path: Path) -> None:
    """A path with 'cycles' in file NAME but not as dir component is OK."""
    safe = tmp_path / "projects" / "not-a-cycles-dir" / "2026"
    safe.mkdir(parents=True)
    # Should not raise — 'cycles' is not a directory component name
    _assert_not_in_cycles(safe)


# ---------------------------------------------------------------------------
# Write-once guard tests
# ---------------------------------------------------------------------------


def test_write_once_guard_no_existing_files(tmp_path: Path) -> None:
    """No existing SHA256SUMS -> no raise (first run)."""
    output_dir = tmp_path / "baselines"
    output_dir.mkdir()
    # No SHA256SUMS, no rankings_baselines.json -> should not raise
    _check_write_once(output_dir, force=False)


def test_write_once_guard_same_sha_no_raise(tmp_path: Path) -> None:
    """Existing SHA256SUMS with matching SHA -> no raise."""
    output_dir = tmp_path / "baselines"
    output_dir.mkdir()
    rankings = output_dir / "rankings_baselines.json"
    rankings.write_text('{"x": 1}')

    import hashlib
    sha = hashlib.sha256(rankings.read_bytes()).hexdigest()
    (output_dir / "SHA256SUMS").write_text(f"{sha}  rankings_baselines.json\n")

    # Same content -> should not raise
    _check_write_once(output_dir, force=False)


def test_write_once_guard_different_sha_raises(tmp_path: Path) -> None:
    """Existing SHA256SUMS with tampered SHA raises without --force (integrity check)."""
    output_dir = tmp_path / "baselines"
    output_dir.mkdir()
    rankings = output_dir / "rankings_baselines.json"
    rankings.write_text('{"x": 1}')
    # Write a DIFFERENT sha in SHA256SUMS — simulates tampering
    _WRONG_SHA = "a" * 64  # 64 'a' chars — differs from the actual file hash
    (output_dir / "SHA256SUMS").write_text(
        f"{_WRONG_SHA}  rankings_baselines.json\n"
    )

    with pytest.raises(click.ClickException, match="[Ii]ntegrity check|tamper|modified since"):
        _check_write_once(output_dir, force=False)


def test_write_once_guard_force_overrides(tmp_path: Path) -> None:
    """--force allows overwrite even when SHA differs."""
    output_dir = tmp_path / "baselines"
    output_dir.mkdir()
    rankings = output_dir / "rankings_baselines.json"
    rankings.write_text('{"x": 1}')
    _WRONG_SHA = "a" * 64
    (output_dir / "SHA256SUMS").write_text(
        f"{_WRONG_SHA}  rankings_baselines.json\n"
    )

    # With --force: should not raise
    _check_write_once(output_dir, force=True)


# ---------------------------------------------------------------------------
# Write-once guard: tamper detection covers all artifacts in SHA256SUMS
# ---------------------------------------------------------------------------

_ALL_ARTIFACT_NAMES = [
    "PROVENANCE.md",
    "lambda_median.npy",
    "rankings_baselines.json",
    "reproduce.py",
    "respondent_rankings.npy",
    "vote_rank_samples.npy",
    "votes_source.xlsx",
]


def _make_full_artifact_dir(base: Path) -> Path:
    """Create a fake baselines dir with all 7 artifacts and a matching SHA256SUMS."""
    import hashlib

    out = base / "baselines"
    out.mkdir()
    sums: list[str] = []
    for name in _ALL_ARTIFACT_NAMES:
        artifact = out / name
        artifact.write_bytes(name.encode())  # unique content per file
        sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        sums.append(f"{sha}  {name}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return out


@pytest.mark.parametrize("artifact_name", _ALL_ARTIFACT_NAMES)
def test_write_once_guard_tamper_each_artifact_raises(
    tmp_path: Path, artifact_name: str
) -> None:
    """Tampering any single artifact in SHA256SUMS raises naming that artifact."""
    output_dir = _make_full_artifact_dir(tmp_path)

    # Overwrite the artifact with different content (simulates tampering)
    (output_dir / artifact_name).write_bytes(b"tampered content")

    with pytest.raises(click.ClickException) as exc_info:
        _check_write_once(output_dir, force=False)

    assert artifact_name in exc_info.value.format_message(), (
        f"Exception message should name the tampered artifact '{artifact_name}', "
        f"got: {exc_info.value.format_message()!r}"
    )


def test_write_once_guard_all_artifacts_clean_no_raise(tmp_path: Path) -> None:
    """All artifacts matching SHA256SUMS -> no raise (clean tree, re-run allowed)."""
    output_dir = _make_full_artifact_dir(tmp_path)
    # All hashes correct — should not raise
    _check_write_once(output_dir, force=False)
