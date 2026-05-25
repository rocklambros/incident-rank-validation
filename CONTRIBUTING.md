# Contributing

## Development Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/rocklambros/incident-rank-validation.git
cd incident-rank-validation
uv sync --group dev
```

## Running Tests

```bash
# Full test suite
uv run pytest

# Single test file
uv run pytest tests/unit/test_inference.py -v

# Skip slow NUTS tests
uv run pytest -m "not slow"

# Type checking
uv run mypy engine tests

# Linting
uv run ruff check .

# Security scanning (semgrep)
uv run semgrep --config .semgrep.yml --error engine/

# All checks (what CI runs)
uv run ruff check . && uv run mypy engine tests && uv run pytest
```

## Code Style

- **Formatter:** ruff, line length 100, target Python 3.12.
- **Type checking:** mypy strict mode. All public functions must have type
  annotations.
- **Lint rules:** E, F, I, B, UP, SIM (see `pyproject.toml [tool.ruff.lint]`).
- **Comments:** avoid unless the *why* is non-obvious. Do not explain *what*
  the code does.

## Branch and Commit Conventions

- Branch names: `plan<N>/<slug>` for phase work, `fix/<slug>` for bug fixes.
- Commit messages: `type(scope): description` where type is `feat`, `fix`,
  `refactor`, `test`, `docs`, or `build`.
- Each plan lands as a series of commits, tagged with a semver version
  (e.g., `v1.2.0-plan6`).

## Adding a New Taxonomy Cycle

To run the engine against a new taxonomy (e.g., OWASP ASI Top 10):

1. **Create the project directory:**
   ```
   projects/<project-name>/
     project.toml          # hyperparameters, taxonomy source, PRNG seed
     cycles/<year>/
       taxonomy/           # entry definition files + taxonomy.json
       prereg/             # manifest, rubric, attestation (populated by CLI)
       corpora/            # vendored snapshot (populated by vendor-snapshot)
       classify/           # classification output (populated by classify)
       calibration/        # gold-set, posteriors (populated by calibrate)
       infer/              # NUTS output (populated by infer)
       results/            # decision layer output (populated by decide/report)
   ```

2. **Write a corpus adapter** under `engine/adapters/` implementing
   `engine.adapters.base.CorpusAdapter`. The adapter reads a vendored snapshot
   and emits `IncidentRecord` instances. Declare a `BiasProfile` per stratum.

3. **Draft a rubric** per entry using `engine/prereg/rubric.py` data structures.
   Freeze with `incident-rank freeze-rubric`.

4. **Run the pipeline** phase by phase (see README for the CLI commands).

5. **Update `docs/METHODOLOGY-CHANGELOG.md`** with a semver entry describing
   the new cycle.

## Methodology Changelog Discipline

Every change that lands on main and affects methodology gets a semver bump in
`docs/METHODOLOGY-CHANGELOG.md`:

- **Major:** changes to the likelihood family, hyperparameter defaults,
  statistic choice, or gate behavior.
- **Minor:** new adapters, new project cycles, report-template additions.
- **Patch:** bug fixes that do not change methodology.

Engine changes made purely for a new cycle's needs still get a changelog entry
if they alter how results are produced.

## Integrity Rules for Contributors

These are non-negotiable constraints from the methodology spec
([`docs/HANDOFF.md`](docs/HANDOFF.md)):

- **Vote data** enters only at the `decide` phase. Do not read, import, or
  reference vote files in any module outside `engine/decide/` and
  `engine/vote/`.
- **Rubric drafting** must be vote-blind. If you draft rubric entries, do not
  look at the vote results spreadsheet.
- **Corpus B** (or any corroboration corpus) is never a posterior input. It
  must not appear in `engine/model/inference.py` imports.
- **GPU inference** is not permitted for NUTS. CPU-only preserves bitwise
  reproducibility across runs. Stage-2 LLM classification on GPU is fine.
- **Pre-registration** artifacts (manifest, rubric hash, taxonomy hash) must
  not be modified after the lock step without producing a pre-reg diff entry.

## Review Process

All changes go through pull requests. CI must pass (ruff, mypy, pytest,
semgrep). For changes that affect the methodology or inference path, expect a
detailed review focused on correctness rather than style.

## Security Vulnerabilities

Do not open a public issue. See [`SECURITY.md`](SECURITY.md) for the
responsible disclosure process.
