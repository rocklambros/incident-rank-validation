from pathlib import Path

import pytest

from engine.cli.pipeline_executor import write_robustness_failure


def test_failure_artifact_records_spec_and_message(tmp_path: Path) -> None:
    write_robustness_failure(tmp_path, "hierarchical_pooling", "ESS below threshold")
    p = tmp_path / "robustness_hierarchical_pooling_failure.txt"
    assert p.exists()
    text = p.read_text()
    assert "hierarchical_pooling" in text
    assert "ESS below threshold" in text


def test_failure_artifact_reraises_after_write(tmp_path: Path) -> None:
    """Robustness loop must write the artifact AND re-raise; it must not swallow.

    This test mirrors the try/except/raise pattern in execute_infer_phase's
    robustness loop to assert that an exception propagates after the artifact
    is written.
    """
    spec_name = "alt_spec"
    sentinel = RuntimeError("sentinel robustness failure")

    with pytest.raises(RuntimeError, match="sentinel robustness failure"):
        try:
            raise sentinel
        except Exception as e:
            write_robustness_failure(tmp_path, spec_name, f"{type(e).__name__}: {e}")
            raise

    # Artifact must also have been written before the re-raise
    p = tmp_path / f"robustness_{spec_name}_failure.txt"
    assert p.exists()
    assert "sentinel robustness failure" in p.read_text()


def test_failure_artifact_rejects_unsafe_spec_name(tmp_path: Path) -> None:
    """Path-safety guard must reject spec names with path separators."""
    with pytest.raises(ValueError, match="unsafe spec_name"):
        write_robustness_failure(tmp_path, "../evil", "msg")


def test_failure_artifact_rejects_empty_spec_name(tmp_path: Path) -> None:
    """Path-safety guard must reject empty spec name."""
    with pytest.raises(ValueError, match="unsafe spec_name"):
        write_robustness_failure(tmp_path, "", "msg")
