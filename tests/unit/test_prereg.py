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
        "confidence_threshold": 0.3,
        "rubric_drafting_attestation": rubric_attestation,
        "rubric_reviewer": rubric_reviewer,
        "statistical_reviewer": statistical_reviewer,
        "classifier_rule_hash": None,
        "rubric_hash": None,
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

    def test_confidence_threshold_default(self) -> None:
        m = _make_manifest()
        assert m.confidence_threshold == 0.3

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
                viewed_vote_data_before_drafting=False,
                viewed_vote_data_details="",
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
                viewed_vote_data_before_drafting=False,
                viewed_vote_data_details="",
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
            "confidence_threshold": 0.99,
            "rubric_drafting_attestation": RubricDraftingAttestation(
                viewed_corpus_before_drafting=True,
                viewed_corpus_details="mutated",
                viewed_vote_data_before_drafting=True,
                viewed_vote_data_details="mutated",
            ),
            "rubric_reviewer": _make_signoff(name="Mutated"),
            "statistical_reviewer": _make_signoff(name="Mutated"),
            "classifier_rule_hash": "mutated_hash",
            "rubric_hash": "mutated_rubric_hash",
            "post_hoc_register_path": "mutated/path.jsonl",
            "rollup_threshold": 0.99,
            "rollup_p_supported": 0.5,
            "rollup_p_contradicted": 0.5,
            "lambda_min": 0.99,
            # schema_version=2 triggers v2 canonical form → different hash
            "schema_version": 2,
            # goldset_hash is intentionally excluded from the v1 canonical form;
            # it is listed here only so the coverage assert stays exhaustive.
            "goldset_hash": "intentionally_non_breaking_under_v1",
            # sigma_u_hyperprior_scale is v2-only: excluded from the v1 canonical form
            # exactly like goldset_hash, so mutating it alone does not invalidate a v1 lock.
            "sigma_u_hyperprior_scale": 3.0,
            # overlap_min_fp is v2-only: excluded from the v1 canonical form like the above.
            "overlap_min_fp": 5,
            # F6 fields are schema>=3-only: excluded from BOTH v1 AND v2 canonical forms.
            # Mutating them alone does NOT invalidate a schema<3 lock (by design — U2-2).
            "recall_min_denominator": 0,       # must stay 0 on schema<3 (guard prevents non-0)
            "recall_min_denominator_gate": False,
            "recall_floor_epsilon": 0.0,
            "recall_min_denominator_rationale": "",
            # D6 power fields are schema>=4-only: excluded from v1/v2/v3 canonical forms.
            # Mutating them alone does NOT invalidate a schema<4 lock (by design — D6).
            "prospective_power_target_kappa": 0.0,       # must stay 0.0 on schema<4 (guard)
            "prospective_power_confidence_level": 0.0,   # must stay 0.0 on schema<4 (guard)
            "prospective_power_1_minus_beta": 0.0,       # must stay 0.0 on schema<4 (guard)
        }

        manifest_fields = {f.name for f in fields(m)}
        assert manifest_fields == set(mutations.keys()), (
            f"mutation table is missing fields: {manifest_fields - set(mutations.keys())}"
        )

        # goldset_hash, sigma_u_hyperprior_scale, overlap_min_fp, lambda_min, all four
        # F6 fields, and all three D6 power fields are excluded from the v1 canonical
        # form; mutating them does NOT invalidate a v1 lock.
        lock_invariant_fields = {
            "goldset_hash", "sigma_u_hyperprior_scale", "overlap_min_fp", "lambda_min",
            "recall_min_denominator", "recall_min_denominator_gate",
            "recall_floor_epsilon", "recall_min_denominator_rationale",
            "prospective_power_target_kappa",
            "prospective_power_confidence_level",
            "prospective_power_1_minus_beta",
        }
        for field_name, alt_value in mutations.items():
            mutated = replace(m, **{field_name: alt_value})
            if field_name in lock_invariant_fields:
                # These fields do NOT invalidate a v1 lock — that's by design.
                verify_lock(mutated, lock_path)  # must not raise
            else:
                with pytest.raises(ValueError, match="lock hash mismatch"):
                    verify_lock(mutated, lock_path)

    def test_v2_lock_still_verifies(self, tmp_path: Path) -> None:
        """Golden-hash test: schema-2 manifest lock hash is FROZEN after F6 field addition.

        If any new field leaks into the v2 canonical form (to_dict() for schema<3),
        the computed hash will differ from this frozen value and the test will fail
        immediately, proving byte-immutability was broken.
        """
        # Golden hash captured from the CURRENT code before U2-2 field addition.
        # Computed via: compute_lock_hash(make_v2_manifest()) on 2026-06-30.
        FROZEN_V2_HASH = (
            "408ef9abffdcfacf318d92ec08b16594581a54fe763678d4a20676bb7d6c1527"
        )
        m = PreregManifest(
            schema_version=2,
            engine_version="0.1.0",
            engine_version_range_min="0.1.0",
            engine_version_range_max="0.2.0",
            cycle_id="test-cycle-v2-golden",
            taxonomy_hash="aaa",
            snapshot_hash="bbb",
            primary_spec="negative_binomial_per_stratum",
            robustness_specs=("poisson_flat",),
            flag_threshold_tau=0.8,
            statistic="weighted_cohens_kappa",
            measurability_minimum=10,
            prior_scale=0.5,
            concentration_shape=5.0,
            concentration_rate=0.1,
            ess_fraction=0.4,
            meaningful_kappa_n=4,
            prng_seed=42,
            confidence_threshold=0.3,
            rubric_drafting_attestation=None,
            rubric_reviewer=None,
            statistical_reviewer=None,
            classifier_rule_hash=None,
            rubric_hash=None,
            post_hoc_register_path=None,
            goldset_hash="v2-golden-goldset",
            sigma_u_hyperprior_scale=1.5,
            overlap_min_fp=3,
        )
        # Write + verify via normal lock roundtrip
        lock_path = tmp_path / "v2.lock"
        write_lock(m, lock_path)
        verify_lock(m, lock_path)  # must not raise
        # Frozen hash check: any future field that leaks into v2 canonical form fails here
        assert compute_lock_hash(m) == FROZEN_V2_HASH, (
            f"v2 canonical hash changed — a new field leaked into schema<3 to_dict().\n"
            f"Expected: {FROZEN_V2_HASH}\n"
            f"Actual:   {compute_lock_hash(m)}"
        )

    def test_v3_lock_still_verifies(self, tmp_path: Path) -> None:
        """Golden-hash test: schema-3 manifest lock hash is FROZEN after D6 power field addition.

        If any new field leaks into the v3 canonical form (to_dict() for schema<4),
        the computed hash will differ from this frozen value and the test will fail
        immediately, proving byte-immutability was broken.
        """
        # Golden hash captured from the code before D6 field addition.
        # Computed via: compute_lock_hash(make_v3_manifest()) on 2026-06-30.
        FROZEN_V3_HASH = (
            "665463d6874d31bec0e5dd8f67a40e51c341fd98acb7ef9e4bc61d681a461567"
        )
        m = PreregManifest(
            schema_version=3,
            engine_version="0.1.0",
            engine_version_range_min="0.1.0",
            engine_version_range_max="0.2.0",
            cycle_id="test-cycle-v3-golden",
            taxonomy_hash="aaa",
            snapshot_hash="bbb",
            primary_spec="negative_binomial_per_stratum",
            robustness_specs=("poisson_flat",),
            flag_threshold_tau=0.8,
            statistic="weighted_cohens_kappa",
            measurability_minimum=10,
            prior_scale=0.5,
            concentration_shape=5.0,
            concentration_rate=0.1,
            ess_fraction=0.4,
            meaningful_kappa_n=4,
            prng_seed=42,
            confidence_threshold=0.3,
            rubric_drafting_attestation=None,
            rubric_reviewer=None,
            statistical_reviewer=None,
            classifier_rule_hash=None,
            rubric_hash=None,
            post_hoc_register_path=None,
            goldset_hash="v3-golden-goldset",
            sigma_u_hyperprior_scale=1.5,
            overlap_min_fp=3,
            recall_min_denominator=5,
            recall_min_denominator_gate=True,
            recall_floor_epsilon=0.01,
            recall_min_denominator_rationale="test-rationale",
        )
        # Write + verify via normal lock roundtrip
        lock_path = tmp_path / "v3.lock"
        write_lock(m, lock_path)
        verify_lock(m, lock_path)  # must not raise
        # Frozen hash check: any future field that leaks into v3 canonical form fails here
        assert compute_lock_hash(m) == FROZEN_V3_HASH, (
            f"v3 canonical hash changed — a new field leaked into schema<4 to_dict().\n"
            f"Expected: {FROZEN_V3_HASH}\n"
            f"Actual:   {compute_lock_hash(m)}"
        )

    def test_real_2026_v1_lock_verifies(self) -> None:
        """RM14: the frozen 2026 cycle's v1 lock verifies against the manifest."""
        import dataclasses
        import json

        prereg = Path(__file__).resolve().parents[2] / "projects/owasp-llm/cycles/2026/prereg"
        manifest_path = prereg / "manifest.json"
        lock_path = prereg / "manifest.lock"
        if not manifest_path.exists() or not lock_path.exists():
            pytest.skip("2026 cycle prereg artifacts not present")
        raw = json.loads(manifest_path.read_text())
        field_names = {f.name for f in dataclasses.fields(PreregManifest)}
        kwargs = {k: v for k, v in raw.items() if k in field_names}
        manifest = PreregManifest(**kwargs)
        assert manifest.schema_version == 1
        verify_lock(manifest, lock_path)  # must not raise


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
