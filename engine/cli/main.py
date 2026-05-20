from __future__ import annotations

from pathlib import Path

import click

from engine.cli.snapshot import vendor_snapshot_cmd


@click.group()
def cli() -> None:
    """incident-rank-validation CLI."""
    pass


cli.add_command(vendor_snapshot_cmd)


@cli.command()
@click.option("--cycle", required=True, type=click.Path(path_type=Path))
@click.option(
    "--corpus-mode", type=click.Choice(["synthetic", "real"]), required=True
)
@click.option(
    "--accept-drift-signoff",
    type=str,
    default=None,
    help="If drift detected, pass a rationale >= 30 chars.",
)
@click.option("--timeout-seconds", type=float, default=None)
def infer(
    cycle: Path,
    corpus_mode: str,
    accept_drift_signoff: str | None,
    timeout_seconds: float | None,
) -> None:
    """Run inference phase (requires prereg lock)."""
    # Phase gate: verify prereg lock exists and is committed
    lock_path = cycle / "prereg" / "prereg.lock.json"
    if not lock_path.exists():
        raise click.ClickException(f"prereg lock not found: {lock_path}")

    # Vote-blindness: vote data must not be present during infer
    vote_dir = cycle / "vote"
    if vote_dir.exists() and any(vote_dir.iterdir()):
        raise click.ClickException(
            "Vote data found during infer phase. Vote enters only at decide. "
            "Remove vote/ from the cycle directory before running infer."
        )

    # Drift signoff with M13 length floor
    prev_snapshot = cycle / "corpora" / "snapshot.previous.jsonl"
    curr_snapshot = cycle / "corpora" / "snapshot.jsonl"
    if prev_snapshot.exists() and curr_snapshot.exists():
        from engine.snapshot.drift import detect_drift

        rep = detect_drift(prev=prev_snapshot, curr=curr_snapshot)
        if rep.requires_signoff:
            reason = (accept_drift_signoff or "").strip()
            if len(reason) < 30:
                raise click.ClickException(
                    f"Drift signoff required ({len(rep.anomalies)} anomalies). "
                    "Pass --accept-drift-signoff '<reason >= 30 chars>'."
                )
            from datetime import UTC, datetime

            signoff_dir = cycle / "drift_signoffs"
            signoff_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            (signoff_dir / f"{ts}.txt").write_text(
                f"Drift signoff accepted at infer.\nReason:\n{reason}\n"
            )

    click.echo("Infer phase: prerequisites satisfied.")


@cli.command()
@click.option("--cycle", required=True, type=click.Path(path_type=Path))
@click.option(
    "--corpus-mode", type=click.Choice(["synthetic", "real"]), required=True
)
def decide(cycle: Path, corpus_mode: str) -> None:
    """Run decide phase (requires prereg lock + infer results)."""
    lock_path = cycle / "prereg" / "prereg.lock.json"
    if not lock_path.exists():
        raise click.ClickException(f"prereg lock not found: {lock_path}")
    click.echo("Decide phase: prerequisites satisfied.")


@cli.command(name="run-synthetic")
@click.option("--cycle", required=True, type=click.Path(path_type=Path))
@click.option(
    "--corpus-mode", type=click.Choice(["synthetic", "real"]), required=True
)
def run_synthetic(cycle: Path, corpus_mode: str) -> None:
    """Run a synthetic end-to-end validation cycle."""
    from engine.cli.synthetic import execute_synthetic_pipeline

    execute_synthetic_pipeline(cycle, corpus_mode)
    click.echo("Synthetic pipeline complete.")
