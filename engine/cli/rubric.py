"""CLI commands for rubric validation and freeze workflow."""
from __future__ import annotations

import json
from pathlib import Path

import click

from engine.prereg.rubric_io import read_rubric, read_rubric_attestation


@click.command("validate-rubric")
@click.option(
    "--rubric",
    type=click.Path(exists=False),
    required=True,
    help="Path to rubric.json",
)
@click.option(
    "--expected-ids",
    type=str,
    default=None,
    help="Comma-separated expected entry IDs for completeness check.",
)
@click.option(
    "--taxonomy",
    type=click.Path(exists=True),
    default=None,
    help="Path to taxonomy.json — reads entry IDs automatically (mutually exclusive with --expected-ids).",
)
@click.option(
    "--no-adjacency-attested",
    type=str,
    default=None,
    help="Comma-separated entry IDs with attested no-adjacency (allowed empty boundary_rules).",
)
def validate_rubric_cmd(
    rubric: str,
    expected_ids: str | None,
    taxonomy: str | None,
    no_adjacency_attested: str | None,
) -> None:
    """Validate a rubric file: schema, completeness, boundary rules."""
    if expected_ids is not None and taxonomy is not None:
        raise click.ClickException(
            "--expected-ids and --taxonomy are mutually exclusive. "
            "Cannot specify both."
        )

    rubric_path = Path(rubric)
    if not rubric_path.exists():
        raise click.ClickException(f"rubric file does not exist: {rubric_path}")

    r = read_rubric(rubric_path)

    attested: set[str] | None = None
    if no_adjacency_attested is not None:
        attested = {i.strip() for i in no_adjacency_attested.split(",") if i.strip()}

    ids: set[str] | None = None
    if taxonomy is not None:
        tax_data = json.loads(Path(taxonomy).read_text())
        ids = {e["entry_id"] for e in tax_data["entries"]}
        click.echo(f"Loaded {len(ids)} entry IDs from taxonomy.json.")
    elif expected_ids is not None:
        ids = {i.strip() for i in expected_ids.split(",")}

    if ids is not None:
        r.validate_completeness(ids, no_adjacency_attested=attested)
        click.echo(f"Completeness: {len(r.entries)} entries match expected set.")

    r.validate_boundary_rules()
    click.echo("Boundary rules: all paired.")

    r.validate_co_occurrences()
    click.echo("Co-occurrence pairs: all reference valid entries.")

    h = r.compute_hash()
    click.echo(f"Rubric hash: {h}")
    click.echo("Rubric is valid.")


@click.command("freeze-rubric")
@click.option(
    "--rubric",
    type=click.Path(exists=True),
    required=True,
    help="Path to rubric.json",
)
@click.option(
    "--cycle-dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to cycle directory (e.g., projects/owasp-llm/cycles/2026).",
)
@click.option(
    "--no-adjacency-attested",
    type=str,
    default=None,
    help="Comma-separated entry IDs with attested no-adjacency (allowed empty boundary_rules).",
)
def freeze_rubric_cmd(
    rubric: str, cycle_dir: str, no_adjacency_attested: str | None
) -> None:
    """Freeze the rubric: validate, require attestation, verify committed, emit hash."""
    import json as _json
    import subprocess

    from engine.prereg.attestation import verify_committed

    rubric_path = Path(rubric)
    cycle = Path(cycle_dir)

    attested: set[str] | None = None
    if no_adjacency_attested is not None:
        attested = {i.strip() for i in no_adjacency_attested.split(",") if i.strip()}

    # Verify rubric file is committed to git (Premortem2 R7).
    repo_root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )
    verify_committed(rubric_path, repo_root)

    r = read_rubric(rubric_path)

    # Completeness check against taxonomy (Premortem2 R5).
    taxonomy_path = cycle / "taxonomy" / "taxonomy.json"
    if taxonomy_path.exists():
        tax_data = _json.loads(taxonomy_path.read_text())
        ids = {e["entry_id"] for e in tax_data["entries"]}
        r.validate_completeness(ids, no_adjacency_attested=attested)
        click.echo(f"Completeness: {len(r.entries)} entries match taxonomy.")
    else:
        click.echo(
            "WARNING: taxonomy.json not found — skipping completeness check."
        )

    r.validate_boundary_rules()
    r.validate_co_occurrences()

    attestation_path = cycle / "prereg" / "rubric_attestation.json"
    if not attestation_path.exists():
        raise click.ClickException(
            f"rubric attestation not found at {attestation_path} — "
            "populate it before freezing"
        )
    verify_committed(attestation_path, repo_root)
    att = read_rubric_attestation(attestation_path)

    rubric_hash = r.compute_hash()
    click.echo(f"Rubric hash: {rubric_hash}")
    click.echo(
        f"Viewed corpus before drafting: {att.viewed_corpus_before_drafting}"
    )
    if att.viewed_corpus_before_drafting:
        click.echo(
            "WARNING: corpus-informed rubric — report will carry caveat."
        )

    click.echo(f"Entries: {len(r.entries)}")
    click.echo(f"  Standalone: {len(r.standalone_entries())}")
    click.echo(f"  Rollup candidates: {len(r.rollup_candidates())}")
    click.echo("Rubric frozen. Add rubric_hash to prereg manifest to lock.")
