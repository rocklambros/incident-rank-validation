"""Two-frame human adjudication tool (spec B4).

Mode 1: Recall adjudication (Frame 1)
Mode 2: Precision verification (Frame 2)
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

_TIER_ORDER = {"agree": 0, "split": 1, "disagree": 2}


def load_prelabels(path: Path) -> list[dict]:
    results = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            results.append(json.loads(line))
    results.sort(key=lambda r: _TIER_ORDER.get(r.get("triage_tier", ""), 99))
    return results


def write_recall_adjudication(
    path: Path,
    *,
    incident_id: str,
    llm_consensus: str | None,
    adjudicated: str,
    labels: list[str],
    blind_label: str | None,
    notes: str | None,
) -> None:
    record = {
        "incident_id": incident_id,
        "llm_consensus": llm_consensus,
        "adjudicated": adjudicated,
        "labels": labels,
        "blind_label": blind_label,
        "notes": notes,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def write_precision_verification(
    path: Path,
    *,
    incident_id: str,
    claimed_entry_id: str,
    is_correct: bool,
    source: str,
    adjudicator_id: str,
) -> None:
    record = {
        "incident_id": incident_id,
        "claimed_entry_id": claimed_entry_id,
        "is_correct": is_correct,
        "source": source,
        "adjudicator_id": adjudicator_id,
        "session_timestamp": datetime.now(UTC).isoformat(),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_recall_mode(
    prelabels_path: Path,
    output_path: Path,
    rubric_path: Path,
) -> None:
    """Interactive Mode 1: recall adjudication."""
    prelabels = load_prelabels(prelabels_path)
    rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
    entry_names = {
        e["entry_id"]: e.get("canonical_name", e["entry_id"])
        for e in rubric.get("entries", [])
    }

    done_ids: set[str] = set()
    if output_path.exists():
        for line in output_path.read_text().strip().splitlines():
            done_ids.add(json.loads(line)["incident_id"])

    for record in prelabels:
        iid = record["incident_id"]
        if iid in done_ids:
            continue

        display_id = hashlib.sha256(iid.encode()).hexdigest()[:8]
        print(f"\n{'='*60}")
        print(f"Incident: {display_id} | Tier: {record['triage_tier']}")
        print(f"{'='*60}")
        print(f"\n{record.get('text', '[text not in prelabels — read from corpus]')}\n")

        blind = input("Your label (entry ID, or 'skip'): ").strip()
        if blind == "skip":
            continue

        print(f"\nLLM consensus: {record['consensus']}")
        for vote in record.get("model_votes", []):
            name = entry_names.get(vote["entry_id"], vote["entry_id"])
            print(f"  {vote['model_id']}: {vote['entry_id']} ({name}) "
                  f"conf={vote['confidence']:.2f} — {vote['rationale']}")

        final = input("\nFinal label(s) (comma-separated, or 'accept'): ").strip()
        if final == "accept":
            labels = [record["consensus"]] if record["consensus"] else []
            adj = "accept"
        else:
            labels = [l.strip() for l in final.split(",")]
            adj = "override"

        notes = input("Notes (or Enter to skip): ").strip() or None

        write_recall_adjudication(
            output_path,
            incident_id=iid,
            llm_consensus=record["consensus"],
            adjudicated=adj,
            labels=labels,
            blind_label=blind,
            notes=notes,
        )


def run_precision_mode(
    classifications_path: Path,
    output_path: Path,
    target_entries: list[str],
    adjudicator_id: str,
    n_per_entry: int = 30,
) -> None:
    """Interactive Mode 2: precision verification."""
    cls_data = json.loads(classifications_path.read_text(encoding="utf-8"))
    classifications = (
        cls_data if isinstance(cls_data, list)
        else cls_data.get("classifications", [])
    )

    done_ids: set[str] = set()
    if output_path.exists():
        for line in output_path.read_text().strip().splitlines():
            done_ids.add(json.loads(line)["incident_id"])

    for entry_id in target_entries:
        candidates = [
            c for c in classifications
            if c["entry_id"] == entry_id and c["incident_id"] not in done_ids
        ][:n_per_entry]

        print(f"\n--- Precision verification for {entry_id} ({len(candidates)} incidents) ---")
        for c in candidates:
            display_id = hashlib.sha256(c["incident_id"].encode()).hexdigest()[:8]
            print(f"\nIncident: {display_id}")
            print(f"Claimed: {entry_id} (conf={c.get('confidence', '?')})")
            print(f"Rationale: {c.get('rationale', 'N/A')}")

            answer = input("Correct? (y/n/skip): ").strip().lower()
            if answer == "skip":
                continue

            write_precision_verification(
                output_path,
                incident_id=c["incident_id"],
                claimed_entry_id=entry_id,
                is_correct=(answer == "y"),
                source="stage2-verified",
                adjudicator_id=adjudicator_id,
            )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m tools.adjudicate [recall|precision] ...")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "recall":
        if len(sys.argv) < 5:
            print("Usage: python -m tools.adjudicate recall <prelabels.jsonl> <output.jsonl> <rubric.json>")
            sys.exit(1)
        run_recall_mode(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    elif mode == "precision":
        if len(sys.argv) < 5:
            print("Usage: python -m tools.adjudicate precision <classifications.json> <output.jsonl> <entry1,entry2,...>")
            sys.exit(1)
        entries = sys.argv[4].split(",")
        run_precision_mode(
            Path(sys.argv[2]), Path(sys.argv[3]),
            target_entries=entries, adjudicator_id="RL",
        )
