"""Unit tests for engine.prereg — pre-registration manifest, locking, and git attestation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import pytest

from engine.prereg.attestation import AttestationError, verify_committed
from engine.prereg.git_timestamp import GitTimestampError, attestation_signed_at
from engine.prereg.lock import compute_lock_hash, verify_lock, write_lock
from engine.prereg.manifest import PreregManifest
from engine.prereg.rubric_attestation import RubricDraftingAttestation
from engine.prereg.signoff import ReviewerSignoff

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_signoff(
    *,
    viewed: bool = False,
    name: str = "Alice",
    path: str = "docs/REVIEWERS/alice-rubric.txt",
    sha: str = "abc123",
    ts: str = "2025-01-15T10:00:00+00:00",
) -> ReviewerSignoff:
    return ReviewerSignoff(
        reviewer_name=name,
        reviewer_affiliation="Example Org",
        attestation_relative_path=path,
        attestation_sha256=sha,
        signed_at=ts,
        viewed_results_before_signoff=viewed,
    )


def _make_manifest(
    *,
    rubric_reviewer: ReviewerSignoff | None = None,
    statistical_reviewer: ReviewerSignoff | None = None,
    rubric_attestation: RubricDraftingAttestation | None = None,
    **overrides: Any,
) -> PreregManifest:
    defaults: dict[str, Any] = {
        "engine_version": "0.1.0",
        "engine_version_range_min": "0.1.0",
        "engine_version_range_max": "0.2.0",
        "cycle_id": "test-cycle-001",
        "taxonomy_hash": "aaa",
        "snapshot_hash": "bbb",
        "primary_spec": "negative_binomial_per_stratum",
        "robustness_specs": ("poisson_flat",),
        "flag_threshold_tau": 0.8,
        "statistic": "weighted_cohens_kappa",
        "measurability_minimum": 10,
        "prior_scale": 0.5,
        "concentration_shape": 5.0,
        "concentration_rate": 0.1,
        "ess_fraction": 0.4,
        "meaningful_kappa_n": 4,
        "prng_seed": 42,
        "rubric_drafting_attestation": rubric_attestation,
        "rubric_reviewer": rubric_reviewer,
        "statistical_reviewer": statistical_reviewer,
        "classifier_rule_hash": None,
        "post_hoc_register_path": None,
    }
    defaults.update(overrides)
    return PreregManifest(**defaults)


def _init_git_repo(repo: Path) -> None:
    """Initialise a bare git repo at *repo* with an initial commit."""
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, capture_output=True, check=True,
    )
    # Initial commit so HEAD exists
    readme = repo / "README.md"
    readme.write_text("init\n")
    subprocess.run(["git", "add", "README.md"], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo, capture_output=True, check=True,
    )


# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------


class TestPreregManifest:
    """Tests for PreregManifest dataclass."""

    def test_non_publishable_when_rubric_reviewer_none(self) -> None:
        m = _make_manifest(
            rubric_reviewer=None,
            statistical_reviewer=_make_signoff(name="Bob", path="docs/REVIEWERS/bob.txt"),
        )
        assert m.non_publishable is True

    def test_non_publishable_when_statistical_reviewer_none(self) -> None:
        m = _make_manifest(
            rubric_reviewer=_make_signoff(),
            statistical_reviewer=None,
        )
        assert m.non_publishable is True

    def test_non_publishable_when_rubric_reviewer_viewed(self) -> None:
        m = _make_manifest(
            rubric_reviewer=_make_signoff(viewed=True),
            statistical_reviewer=_make_signoff(
                name="Bob", path="docs/REVIEWERS/bob.txt", sha="def456"
            ),
        )
        assert m.non_publishable is True

    def test_non_publishable_when_statistical_reviewer_viewed(self) -> None:
        m = _make_manifest(
            rubric_reviewer=_make_signoff(),
            statistical_reviewer=_make_signoff(
                name="Bob", path="docs/REVIEWERS/bob.txt", sha="def456", viewed=True
            ),
        )
        assert m.non_publishable is True

    def test_publishable_when_both_reviewers_clean(self) -> None:
        m = _make_manifest(
            rubric_reviewer=_make_signoff(),
            statistical_reviewer=_make_signoff(
                name="Bob", path="docs/REVIEWERS/bob.txt", sha="def456"
            ),
        )
        assert m.non_publishable is False

    def test_to_dict_roundtrips_through_json(self) -> None:
        m = _make_manifest(
            rubric_attestation=RubricDraftingAttestation(
                viewed_corpus_before_drafting=False,
                viewed_corpus_details="",
            ),
            rubric_reviewer=_make_signoff(),
            statistical_reviewer=_make_signoff(
                name="Bob", path="docs/REVIEWERS/bob.txt", sha="def456"
            ),
        )
        d = m.to_dict()
        serialized = json.dumps(d, sort_keys=True, separators=(",", ":"))
        roundtripped = json.loads(serialized)
        assert roundtripped == d

    def test_to_dict_deterministic(self) -> None:
        m = _make_manifest(
            rubric_reviewer=_make_signoff(),
            statistical_reviewer=_make_signoff(
                name="Bob", path="docs/REVIEWERS/bob.txt", sha="def456"
            ),
        )
        json1 = json.dumps(m.to_dict(), sort_keys=True, separators=(",", ":"))
        json2 = json.dumps(m.to_dict(), sort_keys=True, separators=(",", ":"))
        assert json1 == json2


# ---------------------------------------------------------------------------
# Lock tests
# ---------------------------------------------------------------------------


class TestLock:
    """Tests for hash-locking."""

    def test_compute_lock_hash_deterministic(self) -> None:
        m = _make_manifest()
        assert compute_lock_hash(m) == compute_lock_hash(m)

    def test_write_lock_verify_lock_roundtrip(self, tmp_path: Path) -> None:
        m = _make_manifest()
        lock_path = tmp_path / "locks" / "prereg.lock"
        write_lock(m, lock_path)
        verify_lock(m, lock_path)  # should not raise

    def test_verify_lock_raises_on_mutation(self, tmp_path: Path) -> None:
        """Parametrize over every manifest field — mutate one, verify rejection."""
        m = _make_manifest(
            rubric_attestation=RubricDraftingAttestation(
                viewed_corpus_before_drafting=False,
                viewed_corpus_details="",
            ),
            rubric_reviewer=_make_signoff(),
            statistical_reviewer=_make_signoff(
                name="Bob", path="docs/REVIEWERS/bob.txt", sha="def456"
            ),
            classifier_rule_hash="original_hash",
            post_hoc_register_path="original/path.jsonl",
        )
        lock_path = tmp_path / "prereg.lock"
        write_lock(m, lock_path)

        # Build mutation table: one alternative value per field
        mutations: dict[str, Any] = {
            "engine_version": "9.9.9",
            "engine_version_range_min": "9.0.0",
            "engine_version_range_max": "9.9.9",
            "cycle_id": "mutated-cycle",
            "taxonomy_hash": "zzz",
            "snapshot_hash": "zzz",
            "primary_spec": "poisson_flat",
            "robustness_specs": ("alt_spec",),
            "flag_threshold_tau": 0.999,
            "statistic": "fleiss_kappa",
            "measurability_minimum": 99,
            "prior_scale": 1.5,
            "concentration_shape": 99.0,
            "concentration_rate": 99.0,
            "ess_fraction": 0.99,
            "meaningful_kappa_n": 99,
            "prng_seed": 999,
            "rubric_drafting_attestation": RubricDraftingAttestation(
                viewed_corpus_before_drafting=True,
                viewed_corpus_details="mutated",
            ),
            "rubric_reviewer": _make_signoff(name="Mutated"),
            "statistical_reviewer": _make_signoff(name="Mutated"),
            "classifier_rule_hash": "mutated_hash",
            "post_hoc_register_path": "mutated/path.jsonl",
        }

        manifest_fields = {f.name for f in fields(m)}
        assert manifest_fields == set(mutations.keys()), (
            f"mutation table is missing fields: {manifest_fields - set(mutations.keys())}"
        )

        for field_name, alt_value in mutations.items():
            mutated = replace(m, **{field_name: alt_value})
            with pytest.raises(ValueError, match="lock hash mismatch"):
                verify_lock(mutated, lock_path)


# ---------------------------------------------------------------------------
# Git timestamp tests
# ---------------------------------------------------------------------------


class TestGitTimestamp:
    """Tests for git-derived signed_at (M8)."""

    def test_attestation_signed_at_returns_commit_timestamp(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        # Create and commit an attestation file
        att = repo / "attestation.txt"
        att.write_text("I attest.\n")
        subprocess.run(["git", "add", "attestation.txt"], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add attestation"],
            cwd=repo, capture_output=True, check=True,
        )

        # Get the expected timestamp via git directly
        res = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", "attestation.txt"],
            cwd=repo, capture_output=True, text=True, check=True,
        )
        expected_ts = res.stdout.strip()

        actual_ts = attestation_signed_at(att, repo)
        assert actual_ts == expected_ts
        # Ensure it looks like ISO 8601
        assert "T" in actual_ts

    def test_git_timestamp_error_for_uncommitted_file(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        untracked = repo / "untracked.txt"
        untracked.write_text("not committed\n")

        with pytest.raises(GitTimestampError, match="could not determine"):
            attestation_signed_at(untracked, repo)


# ---------------------------------------------------------------------------
# Attestation tests (verify_committed)
# ---------------------------------------------------------------------------


class TestVerifyCommitted:
    """Tests for verify_committed git working-tree check."""

    def test_passes_for_committed_file(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        f = repo / "tracked.txt"
        f.write_text("tracked content\n")
        subprocess.run(["git", "add", "tracked.txt"], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add tracked"],
            cwd=repo, capture_output=True, check=True,
        )

        verify_committed(f, repo)  # should not raise

    def test_raises_for_untracked_file(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        f = repo / "untracked.txt"
        f.write_text("untracked\n")

        with pytest.raises(AttestationError, match="not tracked by git"):
            verify_committed(f, repo)

    def test_raises_for_uncommitted_changes(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        f = repo / "changing.txt"
        f.write_text("original\n")
        subprocess.run(["git", "add", "changing.txt"], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add changing"],
            cwd=repo, capture_output=True, check=True,
        )

        # Modify after commit
        f.write_text("modified\n")

        with pytest.raises(AttestationError, match="has uncommitted changes"):
            verify_committed(f, repo)


# ---------------------------------------------------------------------------
# Signoff verify tests
# ---------------------------------------------------------------------------


class TestSignoffVerify:
    """Tests for ReviewerSignoff.verify() — full path including M8."""

    def test_full_verify_passes(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        # Create and commit an attestation file
        att_dir = repo / "docs" / "REVIEWERS"
        att_dir.mkdir(parents=True)
        att_file = att_dir / "alice-rubric.txt"
        att_content = b"I, Alice, attest that the rubric is sound.\n"
        att_file.write_bytes(att_content)

        rel_path = "docs/REVIEWERS/alice-rubric.txt"
        subprocess.run(["git", "add", rel_path], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add alice attestation"],
            cwd=repo, capture_output=True, check=True,
        )

        # Get the git timestamp and file hash
        git_ts = attestation_signed_at(att_file, repo)
        file_hash = hashlib.sha256(att_content).hexdigest()

        signoff = ReviewerSignoff(
            reviewer_name="Alice",
            reviewer_affiliation="Example Org",
            attestation_relative_path=rel_path,
            attestation_sha256=file_hash,
            signed_at=git_ts,
            viewed_results_before_signoff=False,
        )

        signoff.verify(repo)  # should not raise

    def test_hash_mismatch_raises(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        att_file = repo / "att.txt"
        att_file.write_bytes(b"real content\n")
        subprocess.run(["git", "add", "att.txt"], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add att"],
            cwd=repo, capture_output=True, check=True,
        )

        git_ts = attestation_signed_at(att_file, repo)

        signoff = ReviewerSignoff(
            reviewer_name="Alice",
            reviewer_affiliation="Example Org",
            attestation_relative_path="att.txt",
            attestation_sha256="0000000000000000000000000000000000000000000000000000000000000000",
            signed_at=git_ts,
            viewed_results_before_signoff=False,
        )

        with pytest.raises(ValueError, match="attestation hash mismatch"):
            signoff.verify(repo)

    def test_signed_at_mismatch_raises(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_git_repo(repo)

        att_file = repo / "att.txt"
        att_content = b"real content\n"
        att_file.write_bytes(att_content)
        subprocess.run(["git", "add", "att.txt"], cwd=repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add att"],
            cwd=repo, capture_output=True, check=True,
        )

        file_hash = hashlib.sha256(att_content).hexdigest()

        signoff = ReviewerSignoff(
            reviewer_name="Alice",
            reviewer_affiliation="Example Org",
            attestation_relative_path="att.txt",
            attestation_sha256=file_hash,
            signed_at="1999-01-01T00:00:00+00:00",  # wrong timestamp
            viewed_results_before_signoff=False,
        )

        with pytest.raises(ValueError, match="signed_at mismatch"):
            signoff.verify(repo)
