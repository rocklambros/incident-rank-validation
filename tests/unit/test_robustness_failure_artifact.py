from pathlib import Path

from engine.cli.pipeline_executor import write_robustness_failure


def test_failure_artifact_records_spec_and_message(tmp_path: Path) -> None:
    write_robustness_failure(tmp_path, "hierarchical_pooling", "ESS below threshold")
    p = tmp_path / "robustness_hierarchical_pooling_failure.txt"
    assert p.exists()
    text = p.read_text()
    assert "hierarchical_pooling" in text
    assert "ESS below threshold" in text
