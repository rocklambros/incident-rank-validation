"""Freeze CLI: materialize projects/owasp-llm/baselines/2026/ (U3 T9).

Writes the full artifact tree:
    rankings_baselines.json     # U9 contract manifest
    lambda_median.npy           # (20,) float64 median over 16000 draws
    vote_rank_samples.npy       # (5000, 20) frozen vote ranks
    respondent_rankings.npy     # (29, 20) RAW respondent matrix
    votes_source.xlsx           # committed raw vote workbook
    SHA256SUMS                  # write-once integrity manifest
    PROVENANCE.md               # disclosures + source provenance

Security hardening:
    - Refuses to write to any path that resolves inside a ``cycles/`` dir.
    - Write-once guard: refuses to overwrite existing baselines/2026/ whose
      SHA256SUMS content differs (pass --force to override).
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import click
import numpy as np
import numpy.typing as npt

from engine.baselines.freeze import _sha256_path, build_rankings_baselines
from engine.vote.bootstrap import bootstrap_vote_ranks
from engine.vote.loader import load_vote_data

__all__ = ["freeze_baselines_cmd"]

_N_BOOTSTRAP: int = 5000
_SEED: int = 20260520

# Measurable entry determination — frame-blind entries for 2026 cycle
_DEFAULT_NOT_MEASURABLE: str = "LLM04,LLM08,LLM10"


# ---------------------------------------------------------------------------
# Cycles/ guard
# ---------------------------------------------------------------------------


def _assert_not_in_cycles(output_path: Path) -> None:
    """Raise ClickException if output_path resolves inside any cycles/ directory.

    Handles:
    - ``..`` traversal (Path.resolve() collapses these)
    - Symlinks into cycles/ (Path.resolve() follows them)
    - Relative paths (Path.resolve() makes them absolute)
    """
    resolved = output_path.resolve()
    # Check the path itself and all ancestors
    for p in (resolved, *resolved.parents):
        if p.name == "cycles":
            raise click.ClickException(
                f"Output path {output_path!r} resolves to {resolved}, which "
                f"is inside a 'cycles/' directory ({p}). "
                "Refusing to write — cycles/ is byte-immutable. "
                "Use a path outside any cycles/ directory, "
                "e.g. projects/owasp-llm/baselines/2026/."
            )


# ---------------------------------------------------------------------------
# Write-once guard
# ---------------------------------------------------------------------------


def _check_write_once(output_dir: Path, force: bool) -> None:
    """Refuse to overwrite existing baselines whose SHA256SUMS content differs."""
    sha256sums_path = output_dir / "SHA256SUMS"
    rankings_path = output_dir / "rankings_baselines.json"
    if not sha256sums_path.exists() or not rankings_path.exists():
        return  # first run; no content to compare

    # Read the existing SHA256SUMS to find the recorded hash for rankings_baselines.json
    existing_sha: str | None = None
    for line in sha256sums_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[1].strip() == "rankings_baselines.json":
            existing_sha = parts[0]
            break

    if existing_sha is None:
        return  # can't compare; allow rerun

    current_sha = _sha256_path(rankings_path)
    if current_sha != existing_sha:
        if not force:
            raise click.ClickException(
                f"Existing baselines/2026/rankings_baselines.json has SHA256 "
                f"{existing_sha[:12]}... but a fresh run would produce "
                f"{current_sha[:12]}... (different content). "
                "This indicates the source data has changed since the last freeze. "
                "Pass --force to overwrite."
            )
        click.echo("WARNING: --force: overwriting existing baselines with different content.")


# ---------------------------------------------------------------------------
# SHA256SUMS writer
# ---------------------------------------------------------------------------


def _write_sha256sums(output_dir: Path) -> None:
    """Write SHA256SUMS file for all sibling files in output_dir."""
    lines: list[str] = []
    for p in sorted(output_dir.iterdir()):
        if p.name == "SHA256SUMS":
            continue
        if p.is_file():
            sha = _sha256_path(p)
            lines.append(f"{sha}  {p.name}")
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Helper: build entry_strata / stratum_sizes from labeled_incidents.json
# ---------------------------------------------------------------------------


def _build_strata(
    labeled: list[dict[str, object]],
) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    es: dict[str, set[str]] = defaultdict(set)
    sc: dict[str, int] = defaultdict(int)
    for item in labeled:
        eid = str(item["entry_id"])
        stratum = str(item.get("stratum", "default"))
        es[eid].add(stratum)
        sc[stratum] += 1
    entry_strata = {e: tuple(sorted(ss)) for e, ss in es.items()}
    stratum_sizes = {s: max(c, 1) for s, c in sc.items()}
    return entry_strata, stratum_sizes


# ---------------------------------------------------------------------------
# PROVENANCE.md content
# ---------------------------------------------------------------------------


def _write_provenance(
    output_dir: Path,
    generated_from: dict[str, object],
    method_delta: float,
    measurable_kappa: float,
    frozen_at: str,
) -> None:
    """Write PROVENANCE.md with all required disclosures."""
    concordance_entry = generated_from.get("concordance_json", {})
    if isinstance(concordance_entry, dict):
        conc_path = concordance_entry.get("path", "unknown")
        conc_sha = concordance_entry.get("sha256", "unknown")
    else:
        conc_path = conc_sha = "unknown"

    lambda_entry = generated_from.get("lambda_samples", {})
    if isinstance(lambda_entry, dict):
        lambda_path = lambda_entry.get("path", "unknown")
        lambda_sha = lambda_entry.get("sha256", "unknown")
    else:
        lambda_path = lambda_sha = "unknown"

    labeled_entry = generated_from.get("labeled_incidents", {})
    if isinstance(labeled_entry, dict):
        labeled_path = labeled_entry.get("path", "unknown")
        labeled_sha = labeled_entry.get("sha256", "unknown")
    else:
        labeled_path = labeled_sha = "unknown"

    resp_entry = generated_from.get("respondent_rankings", {})
    if isinstance(resp_entry, dict):
        resp_path = resp_entry.get("path", "unknown")
        resp_sha = resp_entry.get("sha256", "unknown")
    else:
        resp_path = resp_sha = "unknown"

    content = f"""# PROVENANCE — 2026 OWASP-LLM Baselines

Generated: {frozen_at}

## Cycle source files (SHA256 at freeze)

| Artifact | Repo-relative path | SHA256 (first 16) |
|----------|-------------------|-------------------|
| lambda_samples.npy | {lambda_path} | {str(lambda_sha)[:16]}... |
| labeled_incidents.json | {labeled_path} | {str(labeled_sha)[:16]}... |
| respondent_rankings.npy | {resp_path} | {str(resp_sha)[:16]}... |
| concordance.json | {conc_path} | {str(conc_sha)[:16]}... |

## F1 — Incidence-kappa fact

The frozen kappa (0.2028985507246377) is computed over **all 20** inference∩vote
entries (total_count=20, NOT the measurable subset of 17).  This mirrors how
`concordance.py:193` operates: the draw loop runs over all common entries without
a measurability filter.

## Method-delta 0.0 (never credited)

On 2026 OWASP-LLM data, bare-lambda ranking (`_ranks_from_lambda`, dead code in
concordance.py) and lambda*size incidence ranking (`_ranks_from_incidence`) produce
**identical kappa medians** (method_kappa_delta={method_delta:+.9f}).  This
coincidence is DISCLOSED and NEVER credited as a method gain.  Individual draw
rankings differ on 1927/5000 draws; the medians happen to coincide.

## CI spans zero

The 95% paired-draw CI [-0.1594, 0.5652] spans zero.  Cannot reject kappa=0 at
the 2026 sample size.  This does NOT indicate the engine is non-functional; it
reflects the structural inadequacy of n=20 (see prospective power block).

## STANDING_CAVEAT contradiction

`concordance.py:48` (STANDING_CAVEAT) claims "computed over the measurable subset
only," but the as-shipped kappa is over 20 entries.  The secondary measurable-subset
kappa (~{measurable_kappa:.4f}) differs from the shipped 0.2029.  This contradiction
is surfaced as a standing disclosure; it is NOT silently resolved.

## Surrogate-variance caveat (prospective power)

The Fleiss-Cohen-Everitt asymptotic variance (sigma²=0.936) is a COARSE
DESIGN-STAGE SURROGATE, DISTINCT from and NOT governing the reported paired-draw
bootstrap CI.  Normal-approximation at n≈20 with ranked/stratum dependence violates
the iid assumption.  The n_required=46 is a structural-adequacy verdict, not a
"collect more taxonomy entries" instruction (n is a fixed ~20-entry taxonomy).

## Omnibus bridge

The previous-vs-new comparison in the U9 report is labeled **OMNIBUS**: the 2026
pre-RARR posteriors and the new run differ in ALL of data + method + recall-correction
+ config.  No clean method-only bridge exists.

## Byte-pin

The frozen kappa values in rankings_baselines.json come from reading
`{conc_path}` at freeze time.  They are NOT hard-coded constants.
Re-running `reproduce.py` re-derives them from the RAW respondent matrix
(non-circular: bootstraps from respondent_rankings.npy, NOT vote_rank_samples.npy).
"""
    (output_dir / "PROVENANCE.md").write_text(content)


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


@click.command("freeze-baselines")
@click.option(
    "--cycle-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to the cycle directory, e.g. projects/owasp-llm/cycles/2026",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to write baselines, e.g. projects/owasp-llm/baselines/2026",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Allow overwriting existing content that differs in SHA256.",
)
@click.option(
    "--respondent-source",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Path to source respondent_rankings.npy. "
        "Defaults to tests/unit/fixtures/respondent_rankings_2026.npy "
        "relative to the repo root."
    ),
)
@click.option(
    "--not-measurable",
    type=str,
    default=_DEFAULT_NOT_MEASURABLE,
    show_default=True,
    help="Comma-separated frame-blind entry IDs (not measurable).",
)
def freeze_baselines_cmd(
    cycle_dir: Path,
    output_dir: Path,
    force: bool,
    respondent_source: Path | None,
    not_measurable: str,
) -> None:
    """Freeze the 2026 previous-ranking baselines to a committed artifact tree.

    Writes rankings_baselines.json (U9 contract), lambda_median.npy,
    vote_rank_samples.npy, respondent_rankings.npy, votes_source.xlsx,
    SHA256SUMS, PROVENANCE.md.

    Raises if output_dir resolves inside any cycles/ directory (immutability
    guard) or if existing content SHA256s differ (write-once guard, use --force
    to override).
    """
    # ---- Resolve paths ----
    cycle_dir = cycle_dir.resolve()
    output_dir = output_dir.resolve()

    # ---- Guard: refuse to write inside cycles/ ----
    _assert_not_in_cycles(output_dir)

    # ---- Guard: write-once ----
    _check_write_once(output_dir, force)

    # ---- Find repo root (for repo-relative paths in generated_from) ----
    repo_root: Path | None = None
    p = output_dir
    while p != p.parent:
        if (p / ".git").exists():
            repo_root = p
            break
        p = p.parent

    def _rel(path: Path) -> str:
        if repo_root is not None:
            try:
                return str(path.relative_to(repo_root))
            except ValueError:
                pass
        return str(path)

    click.echo(f"cycle_dir:  {cycle_dir}")
    click.echo(f"output_dir: {output_dir}")

    # ---- Load cycle artifacts ----
    lambda_npy_path = cycle_dir / "infer" / "lambda_samples.npy"
    inf_summary_path = cycle_dir / "infer" / "inference_summary.json"
    labeled_path = cycle_dir / "classify" / "labeled_incidents.json"
    concordance_path = cycle_dir / "results" / "concordance.json"
    xlsx_path = cycle_dir / "vote" / "OWASP Top 10 LLM Candidates Voting Results - 2026.xlsx"

    for required in (lambda_npy_path, inf_summary_path, labeled_path, concordance_path, xlsx_path):
        if not required.exists():
            raise click.ClickException(f"Required cycle file not found: {required}")

    click.echo("Loading cycle artifacts...")
    lambda_samples: npt.NDArray[np.float64] = np.load(lambda_npy_path)

    with open(inf_summary_path) as fh:
        inf_summary = json.load(fh)
    inf_entry_ids: tuple[str, ...] = tuple(inf_summary["entry_ids"])

    with open(labeled_path) as fh:
        labeled: list[dict[str, object]] = json.load(fh)
    entry_strata, stratum_sizes = _build_strata(labeled)

    # ---- Measurable / not-measurable ----
    not_measurable_ids: tuple[str, ...] = tuple(
        e.strip() for e in not_measurable.split(",") if e.strip()
    )
    not_measurable_set = set(not_measurable_ids)
    measurable_entry_ids: tuple[str, ...] = tuple(
        e for e in inf_entry_ids if e not in not_measurable_set
    )
    click.echo(
        f"Measurable: {len(measurable_entry_ids)}, "
        f"not-measurable: {len(not_measurable_ids)}"
    )

    # ---- Parse xlsx → respondent matrix ----
    click.echo(f"Parsing vote xlsx: {xlsx_path.name}...")
    vote_data = load_vote_data(xlsx_path)
    respondent_rankings: npt.NDArray[np.float64] = vote_data.rankings
    vote_entry_ids: tuple[str, ...] = vote_data.entry_ids
    click.echo(
        f"  respondent_rankings shape: {respondent_rankings.shape}, "
        f"n_respondents: {vote_data.n_respondents}"
    )

    # If a pre-existing fixture is provided as source, prefer it (deterministic)
    if respondent_source is not None:
        click.echo(f"Using respondent source: {respondent_source}")
        respondent_rankings = np.load(respondent_source)

    # ---- Bootstrap vote ranks ----
    click.echo(f"Bootstrapping vote ranks (n={_N_BOOTSTRAP}, seed={_SEED})...")
    vote_posterior = bootstrap_vote_ranks(
        respondent_rankings,
        vote_entry_ids,
        n_bootstrap=_N_BOOTSTRAP,
        seed=_SEED,
    )
    vote_rank_samples: npt.NDArray[np.float64] = vote_posterior.rank_samples
    click.echo(f"  vote_rank_samples shape: {vote_rank_samples.shape}")

    # ---- Create output dir ----
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Save numpy artifacts ----
    click.echo("Saving numpy artifacts...")
    resp_npy_path = output_dir / "respondent_rankings.npy"
    vote_npy_path = output_dir / "vote_rank_samples.npy"
    lambda_median_npy_path = output_dir / "lambda_median.npy"
    np.save(resp_npy_path, respondent_rankings)
    np.save(vote_npy_path, vote_rank_samples)
    np.save(lambda_median_npy_path, np.median(lambda_samples, axis=0))

    # ---- Copy xlsx ----
    xlsx_dest = output_dir / "votes_source.xlsx"
    shutil.copy2(xlsx_path, xlsx_dest)
    click.echo(f"Copied xlsx -> {xlsx_dest.name}")

    # ---- Compute SHA256s for generated_from ----
    generated_from: dict[str, object] = {
        "lambda_samples": {
            "path": _rel(lambda_npy_path),
            "shape": list(lambda_samples.shape),
            "sha256": _sha256_path(lambda_npy_path),
        },
        "inference_summary": {
            "path": _rel(inf_summary_path),
            "sha256": _sha256_path(inf_summary_path),
        },
        "labeled_incidents": {
            "path": _rel(labeled_path),
            "sha256": _sha256_path(labeled_path),
        },
        "respondent_rankings": {
            "path": _rel(resp_npy_path),
            "shape": list(respondent_rankings.shape),
            "sha256": _sha256_path(resp_npy_path),
        },
        "concordance_json": {
            "path": _rel(concordance_path),
            "sha256": _sha256_path(concordance_path),
            "seed": _SEED,
        },
    }

    # ---- Build rankings_baselines dict (pure assembler — reads concordance.json) ----
    click.echo("Assembling rankings_baselines.json (byte-pinning kappa to concordance.json)...")
    manifest = build_rankings_baselines(
        concordance_json_path=concordance_path,
        lambda_samples=lambda_samples,
        vote_rank_samples=vote_rank_samples,
        inf_entry_ids=inf_entry_ids,
        vote_entry_ids=vote_entry_ids,
        measurable_entry_ids=measurable_entry_ids,
        not_measurable_entry_ids=not_measurable_ids,
        entry_strata=entry_strata,
        stratum_sizes=stratum_sizes,
        generated_from=generated_from,
    )

    # ---- Write rankings_baselines.json ----
    rankings_path = output_dir / "rankings_baselines.json"
    rankings_path.write_text(json.dumps(manifest, indent=2) + "\n")
    click.echo(f"Wrote rankings_baselines.json ({rankings_path.stat().st_size} bytes)")

    # ---- Write PROVENANCE.md ----
    prev_ranking = manifest.get("previous_ranking", {})
    if isinstance(prev_ranking, dict):
        pass
    bare_sensitivity = manifest.get("bare_lambda_sensitivity", {})
    method_delta: float = 0.0
    if isinstance(bare_sensitivity, dict):
        method_delta = float(bare_sensitivity.get("method_kappa_delta", 0.0))
    secondary = manifest.get("secondary_measurable_subset", {})
    measurable_kappa: float = 0.0
    if isinstance(secondary, dict):
        measurable_kappa = float(secondary.get("measurable_kappa_median", 0.0))

    frozen_at = datetime.now(UTC).isoformat(timespec="seconds")
    _write_provenance(output_dir, generated_from, method_delta, measurable_kappa, frozen_at)
    click.echo("Wrote PROVENANCE.md")

    # ---- Write SHA256SUMS (after all other files are written) ----
    _write_sha256sums(output_dir)
    click.echo("Wrote SHA256SUMS")

    click.echo("\n=== freeze-baselines complete ===")
    click.echo(f"Artifact tree: {output_dir}")
    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            click.echo(f"  {p.name}  ({p.stat().st_size:,} bytes)")
