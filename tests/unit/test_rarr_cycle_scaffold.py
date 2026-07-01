"""T1: Verify that cycles/2026-rarr/ contains byte-identical copies of all 2026 inputs."""
import hashlib
from pathlib import Path

C2026 = Path("projects/owasp-llm/cycles/2026")
CRARR = Path("projects/owasp-llm/cycles/2026-rarr")
SNAP = "24806f1a4f0917f85f7509d6cb2a34b12e56eb902714b37bc2b03a2cf1a246bb"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_copied_inputs_are_byte_identical_to_2026() -> None:
    rels = [
        "prereg/rubric.json",
        "prereg/rubric_attestation.json",
        "taxonomy/taxonomy.json",
        "calibration/posteriors.json",
        "calibration/adjudicated_goldset.jsonl",
        f"corpora/genai_agentic/{SNAP}/incidents.json",
        f"corpora/genai_agentic/{SNAP}/incidents.jsonl",
        f"corpora/genai_agentic/{SNAP}/provenance.json",
    ]
    for rel in rels:
        src, dst = C2026 / rel, CRARR / rel
        assert dst.exists(), f"missing copied input: {rel}"
        assert _sha(dst) == _sha(src), f"hash drift in {rel}"


def test_snapshot_dir_name_equals_snapshot_hash() -> None:
    assert (CRARR / "corpora/genai_agentic" / SNAP).is_dir()
