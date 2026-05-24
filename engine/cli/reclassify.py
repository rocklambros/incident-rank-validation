"""Multi-model re-classification pipeline.

Re-runs Stage-2 classification using MultiModelPreLabeler (3-model consensus)
with the improved system/user prompt split. Produces a new labeled_incidents.json
and a diff report against the previous classifications.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
from pathlib import Path

import click

from engine.schema import IncidentRecord

logger = logging.getLogger(__name__)


@click.command("reclassify")
@click.option("--cycle", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--execute", is_flag=True, default=False,
              help="Execute reclassification (without flag, shows plan only)")
@click.option("--max-concurrent", type=int, default=18,
              help="Max concurrent RunPod requests per model")
def reclassify(cycle: Path, execute: bool, max_concurrent: int) -> None:
    """Re-classify Stage-2 incidents using multi-model consensus."""
    import os

    from engine.cli.secrets import load_secret

    prereg = cycle / "prereg"
    rubric_json = (prereg / "rubric.json").read_text()

    # Load existing classifications to identify Stage-2 incidents
    classify_dir = cycle / "classify"
    labeled = json.loads((classify_dir / "labeled_incidents.json").read_text())
    stage1_by_id = {}
    stage2_ids: set[str] = set()
    for c in labeled:
        if c.get("stage") == 1:
            stage1_by_id[c["incident_id"]] = c
        else:
            stage2_ids.add(c["incident_id"])

    click.echo(f"Existing classifications: {len(stage1_by_id)} Stage-1, {len(stage2_ids)} Stage-2")

    # Load corpus incidents
    corpus_dir = cycle / "corpora"
    snapshot_dirs = sorted(corpus_dir.glob("*/*/provenance.json"))
    if not snapshot_dirs:
        snapshot_dirs = sorted(corpus_dir.glob("*/provenance.json"))
    if not snapshot_dirs:
        raise click.ClickException(f"No provenance.json found under {corpus_dir}")

    prov_path = snapshot_dirs[0]
    prov_data = json.loads(prov_path.read_text())
    snapshot_dir = prov_path.parent

    from engine.adapters.genai_agentic import GenAIAgenticAdapter

    adapter = GenAIAgenticAdapter(snapshot_dir, prov_data["pull_date"])
    all_incidents = {inc.id: inc for inc in adapter.iter_incidents()}
    s2_incidents = [all_incidents[iid] for iid in stage2_ids if iid in all_incidents]
    click.echo(f"Loaded {len(s2_incidents)} Stage-2 incidents from corpus")

    # Discover RunPod endpoints from environment
    api_key = load_secret("runpod/api-key", env_var="RUNPOD_API_KEY")
    model_configs = _load_model_configs()
    if not model_configs:
        raise click.ClickException(
            "No model endpoints configured. Set environment variables:\n"
            "  RUNPOD_MODEL_1_ENDPOINT=<endpoint_id>  RUNPOD_MODEL_1_NAME=<model_name>\n"
            "  RUNPOD_MODEL_2_ENDPOINT=<endpoint_id>  RUNPOD_MODEL_2_NAME=<model_name>\n"
            "  RUNPOD_MODEL_3_ENDPOINT=<endpoint_id>  RUNPOD_MODEL_3_NAME=<model_name>"
        )

    click.echo(f"Models configured: {len(model_configs)}")
    for name, eid in model_configs:
        click.echo(f"  {name} → {eid}")

    total_calls = len(s2_incidents) * len(model_configs)
    click.echo(f"\nPlan: {len(s2_incidents)} incidents × {len(model_configs)} models = {total_calls} API calls")

    if not execute:
        click.echo("Run with --execute to start reclassification.")
        return

    # Build clients
    from engine.classify.runpod_client import HttpRunPodClient

    clients = []
    for model_name, endpoint_id in model_configs:
        client = HttpRunPodClient(
            api_key=api_key,
            endpoint_id=endpoint_id,
            model_name=model_name,
        )
        clients.append((client, model_name))

    from engine.classify.multi_model import MultiModelPreLabeler

    labeler = MultiModelPreLabeler(
        models=clients,
        rubric_json=rubric_json,
        prng_seed=42,
    )

    # Checkpoint support
    checkpoint_path = classify_dir / "reclassify_checkpoint.jsonl"
    done_ids: set[str] = set()
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text().strip().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["incident_id"])
        click.echo(f"Resuming from checkpoint: {len(done_ids)} already done")

    remaining = [inc for inc in s2_incidents if inc.id not in done_ids]
    click.echo(f"Classifying {len(remaining)} incidents ({len(done_ids)} cached)...")

    completed = 0
    lock = threading.Lock()
    total = len(remaining)

    # Write results as we go for checkpoint/resume
    with checkpoint_path.open("a", encoding="utf-8") as f:
        def _classify_one(incident: IncidentRecord) -> dict:
            result = labeler.pre_label(incident)
            return {
                "incident_id": result.incident_id,
                "model_votes": [
                    {
                        "model_id": v.model_id,
                        "entry_id": v.entry_id,
                        "confidence": v.confidence,
                        "rationale": v.rationale,
                    }
                    for v in result.model_votes
                ],
                "consensus": result.consensus,
                "agreement": result.agreement,
                "triage_tier": result.triage_tier,
            }

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent,
        ) as pool:
            future_to_inc = {
                pool.submit(_classify_one, inc): inc
                for inc in remaining
            }
            for future in concurrent.futures.as_completed(future_to_inc):
                record = future.result()
                with lock:
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    completed += 1
                    if completed % 50 == 0 or completed == total:
                        click.echo(f"  Progress: {completed}/{total}")

    # Close clients
    for client, _ in clients:
        client.close()

    # Convert checkpoint to new labeled_incidents.json
    click.echo("Converting multi-model results to labeled_incidents format...")
    new_classifications = _convert_to_labeled(
        checkpoint_path, stage1_by_id, s2_incidents,
    )

    out_path = classify_dir / "labeled_incidents_multimodel.json"
    out_path.write_text(json.dumps(new_classifications, indent=2) + "\n")
    click.echo(f"Written {len(new_classifications)} classifications to {out_path}")

    # Generate diff report
    old_s2 = {c["incident_id"]: c for c in labeled if c.get("stage") == 2}
    _print_diff_report(old_s2, new_classifications, classify_dir)


def _load_model_configs() -> list[tuple[str, str]]:
    import os

    configs = []
    for i in range(1, 10):
        endpoint = os.environ.get(f"RUNPOD_MODEL_{i}_ENDPOINT", "")
        name = os.environ.get(f"RUNPOD_MODEL_{i}_NAME", "")
        if endpoint and name:
            configs.append((name, endpoint))
    return configs


def _convert_to_labeled(
    checkpoint_path: Path,
    stage1_by_id: dict[str, dict],
    s2_incidents: list[IncidentRecord],
) -> list[dict]:
    incident_strata = {inc.id: inc.corpus_stratum for inc in s2_incidents}

    # Start with all Stage-1 results
    results = list(stage1_by_id.values())

    # Add multi-model consensus as Stage-2 results
    for line in checkpoint_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        iid = record["incident_id"]
        consensus = record.get("consensus", "out-of-scope")
        if consensus is None:
            consensus = "out-of-scope"

        votes = record.get("model_votes", [])
        max_conf = max((v["confidence"] for v in votes), default=0.0)
        rationales = [
            f"{v['model_id']}: {v['entry_id']} ({v['confidence']:.2f})"
            for v in votes
        ]

        results.append({
            "incident_id": iid,
            "entry_id": consensus,
            "confidence": max_conf,
            "stage": 2,
            "rationale": f"multi-model {record.get('agreement', '?')}: {'; '.join(rationales)}",
            "stratum": incident_strata.get(iid, "security"),
        })

    return results


def _print_diff_report(
    old_s2: dict[str, dict],
    new_classifications: list[dict],
    classify_dir: Path,
) -> None:
    new_s2 = {c["incident_id"]: c for c in new_classifications if c.get("stage") == 2}
    common = set(old_s2) & set(new_s2)

    changed = 0
    changes_detail: list[dict] = []
    for iid in sorted(common):
        old_entry = old_s2[iid]["entry_id"]
        new_entry = new_s2[iid]["entry_id"]
        if old_entry != new_entry:
            changed += 1
            changes_detail.append({
                "incident_id": iid,
                "old_entry": old_entry,
                "new_entry": new_entry,
            })

    click.echo(f"\n{'='*60}")
    click.echo("CLASSIFICATION DIFF REPORT")
    click.echo(f"{'='*60}")
    click.echo(f"Common incidents: {len(common)}")
    click.echo(f"Changed: {changed} ({100*changed/len(common):.1f}%)" if common else "No common incidents")
    click.echo(f"Unchanged: {len(common) - changed}")

    if changes_detail:
        from collections import Counter

        transitions = Counter(
            (c["old_entry"], c["new_entry"]) for c in changes_detail
        )
        click.echo(f"\nTop transitions (old → new):")
        for (old, new), count in transitions.most_common(20):
            click.echo(f"  {old} → {new}: {count}")

    # Write full diff to file
    diff_path = classify_dir / "reclassify_diff.json"
    diff_path.write_text(json.dumps({
        "common_count": len(common),
        "changed_count": changed,
        "change_rate": changed / len(common) if common else 0,
        "changes": changes_detail,
    }, indent=2) + "\n")
    click.echo(f"\nFull diff written to {diff_path}")
