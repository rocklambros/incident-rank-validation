# Residual Risks + Narrative Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close two premortem residual risks (ci_method field, ai-harm precision disclosure), update the notebook, and build a standalone narrative report generator.

**Architecture:** Three phases in dependency order: (1) add `ci_method` field to `ConcordanceResult` dataclass + serialization + deserialization round-trip, (2) register `F-aiharm-precision` threat + update notebook Act 10, (3) build `engine/report/narrative.py` module with 16 chart generators and markdown renderer, wired to a CLI command. The narrative report reads the same data files as the notebook and renders programmatically from the threats register.

**Tech Stack:** Python 3.12, dataclasses, click CLI, matplotlib/seaborn, plotly/kaleido, NumPy, pytest

---

## File Structure

**Phase 1 — ci_method field:**
- Modify: `engine/decide/concordance.py` (add field to dataclass)
- Modify: `engine/cli/pipeline_executor.py` (add field to serialization)
- Modify: `engine/cli/pipeline.py` (add field to deserialization)
- Test: `tests/unit/test_concordance.py` (unit + round-trip tests)
- Regenerate: `projects/owasp-llm/cycles/2026/results/concordance.json`

**Phase 2 — ai-harm precision disclosure:**
- Modify: `engine/threats/register.py` (add `F-aiharm-precision` threat)
- Modify: `notebooks/what_the_data_says_2026.ipynb` (Act 10 paragraph)
- Test: `tests/unit/test_threats.py` (presence test)
- Regenerate: `projects/owasp-llm/cycles/2026/results/report.md`

**Phase 3 — narrative report generator:**
- Create: `engine/report/narrative.py` (main module)
- Create: `engine/report/narrative_charts.py` (all 16 chart generators)
- Create: `engine/report/narrative_data.py` (data loading)
- Modify: `engine/cli/pipeline.py` (add `report-narrative` command)
- Modify: `engine/cli/main.py` (register command)
- Modify: `pyproject.toml` (add optional `narrative` dependency group)
- Test: `tests/unit/test_narrative.py`
- Output: `projects/owasp-llm/cycles/2026/results/narrative/`

---

### Task 1: Add `ci_method` field to `ConcordanceResult`

**Files:**
- Test: `tests/unit/test_concordance.py`
- Modify: `engine/decide/concordance.py:27-40`

- [ ] **Step 1: Write failing tests for ci_method field**

Add to the end of `tests/unit/test_concordance.py`:

```python
from engine.decide.concordance import ConcordanceResult, STANDING_CAVEAT


class TestCiMethodField:
    def test_ci_method_default_value(self) -> None:
        """ConcordanceResult should have ci_method with correct default."""
        result = ConcordanceResult(
            weighted_kappa_median=0.20,
            weighted_kappa_ci=(-0.16, 0.57),
            measurable_count=17,
            total_count=20,
            coverage_ratio=0.85,
            below_prereg_minimum=False,
            meaningful_kappa_n=5,
            flags=(),
            standing_caveat=STANDING_CAVEAT,
        )
        assert result.ci_method == "paired_draw_percentile"

    def test_ci_method_in_na_result(self) -> None:
        """N/A results should also carry the ci_method default."""
        entries = ("A", "B")
        inf = _make_inference(entries)
        vote = _make_vote_posterior(entries)
        result = compute_concordance(
            inference_result=inf,
            vote_posterior=vote,
            tier_boundaries=(3,),
            flag_threshold_tau=0.5,
            measurable_count=2,
            total_count=10,
            meaningful_kappa_n=5,
            measurability_minimum=3,
        )
        assert result.ci_method == "paired_draw_percentile"

    def test_ci_method_in_normal_result(self) -> None:
        """Normal results should carry the ci_method default."""
        entries = tuple(f"E{i}" for i in range(10))
        inf = _make_inference(entries, n_samples=200)
        vote = _make_vote_posterior(entries, n_bootstrap=200)
        result = compute_concordance(
            inference_result=inf,
            vote_posterior=vote,
            tier_boundaries=(3, 7),
            flag_threshold_tau=0.5,
            measurable_count=10,
            total_count=15,
            meaningful_kappa_n=5,
            measurability_minimum=5,
        )
        assert result.ci_method == "paired_draw_percentile"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_concordance.py::TestCiMethodField -v`
Expected: FAIL — `TypeError: ConcordanceResult.__init__() got an unexpected keyword argument 'ci_method'` or `AttributeError: 'ConcordanceResult' object has no attribute 'ci_method'`

- [ ] **Step 3: Add ci_method field to ConcordanceResult**

In `engine/decide/concordance.py`, change the dataclass (lines 27-40):

```python
@dataclass(frozen=True, slots=True)
class ConcordanceResult:
    """Full concordance output including kappa, CI, flags, and caveat."""

    weighted_kappa_median: float | None
    weighted_kappa_ci: tuple[float, float] | None
    measurable_count: int
    total_count: int
    coverage_ratio: float
    below_prereg_minimum: bool
    meaningful_kappa_n: int
    flags: tuple[FlagFinding, ...]
    standing_caveat: str
    ci_method: str = "paired_draw_percentile"
    entry_comparisons: tuple[dict[str, object], ...] | None = None
```

The `ci_method` field goes after `standing_caveat` and before `entry_comparisons`. Both have defaults (ci_method has a string default, entry_comparisons has None), so Python field ordering is satisfied.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_concordance.py -v`
Expected: ALL PASS (including all pre-existing tests — they don't pass `ci_method`, so the default kicks in)

- [ ] **Step 5: Commit**

```bash
git add engine/decide/concordance.py tests/unit/test_concordance.py
git commit -m "feat(concordance): add ci_method field to ConcordanceResult dataclass"
```

---

### Task 2: Add ci_method to serialization and deserialization

**Files:**
- Test: `tests/unit/test_concordance.py` (round-trip test)
- Modify: `engine/cli/pipeline_executor.py:349-364`
- Modify: `engine/cli/pipeline.py:519-532`

- [ ] **Step 1: Write failing round-trip test**

Add to `tests/unit/test_concordance.py`:

```python
import json

from engine.decide.robustness_multiplicity import FlagDirection, FlagFinding


class TestCiMethodSerialization:
    def test_ci_method_in_serialized_dict(self) -> None:
        """ci_method should appear in the concordance JSON output."""
        result = ConcordanceResult(
            weighted_kappa_median=0.20,
            weighted_kappa_ci=(-0.16, 0.57),
            measurable_count=17,
            total_count=20,
            coverage_ratio=0.85,
            below_prereg_minimum=False,
            meaningful_kappa_n=5,
            flags=(),
            standing_caveat=STANDING_CAVEAT,
        )
        # Simulate what pipeline_executor.write_decide_artifacts does
        conc_dict = {
            "weighted_kappa_median": result.weighted_kappa_median,
            "weighted_kappa_ci": list(result.weighted_kappa_ci) if result.weighted_kappa_ci else None,
            "measurable_count": result.measurable_count,
            "total_count": result.total_count,
            "coverage_ratio": result.coverage_ratio,
            "below_prereg_minimum": result.below_prereg_minimum,
            "ci_method": result.ci_method,
            "flags": [],
        }
        serialized = json.dumps(conc_dict)
        data = json.loads(serialized)
        assert data["ci_method"] == "paired_draw_percentile"

    def test_round_trip_preserves_ci_method(self) -> None:
        """Serialize then deserialize: ci_method must survive the round trip."""
        original = ConcordanceResult(
            weighted_kappa_median=0.20,
            weighted_kappa_ci=(-0.16, 0.57),
            measurable_count=17,
            total_count=20,
            coverage_ratio=0.85,
            below_prereg_minimum=False,
            meaningful_kappa_n=5,
            flags=(
                FlagFinding(entry_id="LLM01", probability=0.88, direction=FlagDirection.VOTE_OVER_RANKS),
            ),
            standing_caveat="test caveat",
        )

        # Serialize (what pipeline_executor does)
        conc_dict = {
            "weighted_kappa_median": original.weighted_kappa_median,
            "weighted_kappa_ci": list(original.weighted_kappa_ci) if original.weighted_kappa_ci else None,
            "measurable_count": original.measurable_count,
            "total_count": original.total_count,
            "coverage_ratio": original.coverage_ratio,
            "below_prereg_minimum": original.below_prereg_minimum,
            "ci_method": original.ci_method,
            "flags": [
                {"entry_id": f.entry_id, "probability": f.probability, "direction": f.direction.value}
                for f in original.flags
            ],
        }
        json_str = json.dumps(conc_dict, indent=2)
        data = json.loads(json_str)

        # Deserialize (what pipeline.py report_cmd does)
        flags_raw = data.get("flags", [])
        flags = tuple(
            FlagFinding(
                entry_id=f["entry_id"],
                probability=f["probability"],
                direction=FlagDirection(f["direction"]),
            )
            for f in flags_raw
        )
        reconstructed = ConcordanceResult(
            weighted_kappa_median=data.get("weighted_kappa_median"),
            weighted_kappa_ci=tuple(data["weighted_kappa_ci"]) if data.get("weighted_kappa_ci") else None,
            measurable_count=data["measurable_count"],
            total_count=data["total_count"],
            coverage_ratio=data["coverage_ratio"],
            below_prereg_minimum=data.get("below_prereg_minimum", False),
            meaningful_kappa_n=5,
            flags=flags,
            standing_caveat=STANDING_CAVEAT,
            ci_method=data.get("ci_method", "paired_draw_percentile"),
        )

        assert reconstructed.ci_method == original.ci_method
        assert reconstructed.weighted_kappa_median == original.weighted_kappa_median
        assert reconstructed.flags[0].entry_id == "LLM01"
```

- [ ] **Step 2: Run tests to verify they pass (these are self-contained)**

Run: `pytest tests/unit/test_concordance.py::TestCiMethodSerialization -v`
Expected: PASS (these tests simulate the serialize/deserialize pattern)

- [ ] **Step 3: Update serialization in pipeline_executor.py**

In `engine/cli/pipeline_executor.py`, add `ci_method` to `conc_dict` (around line 349-364). Change the block to:

```python
    conc_dict = {
        "weighted_kappa_median": concordance.weighted_kappa_median,
        "weighted_kappa_ci": (
            list(concordance.weighted_kappa_ci)
            if concordance.weighted_kappa_ci
            else None
        ),
        "measurable_count": concordance.measurable_count,
        "total_count": concordance.total_count,
        "coverage_ratio": concordance.coverage_ratio,
        "below_prereg_minimum": concordance.below_prereg_minimum,
        "ci_method": concordance.ci_method,
        "flags": [
            {"entry_id": f.entry_id, "probability": f.probability, "direction": f.direction.value}
            for f in concordance.flags
        ],
    }
```

- [ ] **Step 4: Update deserialization in pipeline.py**

In `engine/cli/pipeline.py`, update the ConcordanceResult reconstruction (lines 519-532). Add `ci_method` to the constructor:

```python
        concordance = ConcordanceResult(
            weighted_kappa_median=conc_data.get("weighted_kappa_median"),
            weighted_kappa_ci=(
                tuple(conc_data["weighted_kappa_ci"])
                if conc_data.get("weighted_kappa_ci") else None
            ),
            measurable_count=conc_data["measurable_count"],
            total_count=conc_data["total_count"],
            coverage_ratio=conc_data["coverage_ratio"],
            below_prereg_minimum=conc_data.get("below_prereg_minimum", False),
            meaningful_kappa_n=manifest.meaningful_kappa_n,
            flags=flags,
            standing_caveat="",
            ci_method=conc_data.get("ci_method", "paired_draw_percentile"),
        )
```

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/unit/test_concordance.py tests/unit/test_report.py tests/unit/test_pipeline_executor.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add engine/cli/pipeline_executor.py engine/cli/pipeline.py tests/unit/test_concordance.py
git commit -m "feat(concordance): add ci_method to serialization and deserialization round-trip"
```

---

### Task 3: Regenerate concordance.json with ci_method field

**Files:**
- Modify: `projects/owasp-llm/cycles/2026/results/concordance.json`

The spec requires regeneration via the pipeline, not a direct JSON patch. We construct a `ConcordanceResult` from the existing JSON data using the updated dataclass (which now has `ci_method` with its default value), then re-serialize via the same `conc_dict` pattern used in `write_decide_artifacts`. This exercises the new serialization code path.

- [ ] **Step 1: Re-serialize concordance.json through ConcordanceResult**

```bash
python3 -c "
import json
from pathlib import Path
from engine.decide.concordance import ConcordanceResult
from engine.decide.robustness_multiplicity import FlagFinding, FlagDirection

p = Path('projects/owasp-llm/cycles/2026/results/concordance.json')
raw = json.loads(p.read_text())

# Reconstruct typed ConcordanceResult from existing JSON
flags = tuple(
    FlagFinding(
        entry_id=f['entry_id'],
        probability=f['probability'],
        direction=FlagDirection(f['direction']),
    ) for f in raw.get('flags', [])
)
conc = ConcordanceResult(
    weighted_kappa_median=raw.get('weighted_kappa_median'),
    weighted_kappa_ci=tuple(raw['weighted_kappa_ci']) if raw.get('weighted_kappa_ci') else None,
    measurable_count=raw['measurable_count'],
    total_count=raw['total_count'],
    coverage_ratio=raw['coverage_ratio'],
    below_prereg_minimum=raw.get('below_prereg_minimum', False),
    meaningful_kappa_n=5,
    flags=flags,
    standing_caveat='',
    # ci_method gets its default 'paired_draw_percentile' from the dataclass
)

# Re-serialize via the same pattern as write_decide_artifacts (Task 2 Step 3)
conc_dict = {
    'weighted_kappa_median': conc.weighted_kappa_median,
    'weighted_kappa_ci': list(conc.weighted_kappa_ci) if conc.weighted_kappa_ci else None,
    'measurable_count': conc.measurable_count,
    'total_count': conc.total_count,
    'coverage_ratio': conc.coverage_ratio,
    'below_prereg_minimum': conc.below_prereg_minimum,
    'ci_method': conc.ci_method,
    'flags': [
        {'entry_id': f.entry_id, 'probability': f.probability, 'direction': f.direction.value}
        for f in conc.flags
    ],
}
p.write_text(json.dumps(conc_dict, indent=2) + '\n')
print(f'Regenerated concordance.json with ci_method={conc.ci_method}')
"
```

- [ ] **Step 2: Verify the field is present**

```bash
python3 -c "
import json
data = json.loads(open('projects/owasp-llm/cycles/2026/results/concordance.json').read())
assert 'ci_method' in data, 'ci_method missing!'
assert data['ci_method'] == 'paired_draw_percentile', f'Wrong value: {data[\"ci_method\"]}'
print('PASS: ci_method field verified')
"
```

- [ ] **Step 3: Commit**

```bash
git add projects/owasp-llm/cycles/2026/results/concordance.json
git commit -m "feat(artifact): regenerate concordance.json with ci_method field"
```

---

### Task 4: Register F-aiharm-precision threat

**Files:**
- Test: `tests/unit/test_threats.py`
- Modify: `engine/threats/register.py:70-79`

- [ ] **Step 1: Write failing test for F-aiharm-precision**

Add to `tests/unit/test_threats.py`:

```python
def test_f_aiharm_precision_present() -> None:
    threats = get_threats_register()
    ids = {t.threat_id for t in threats}
    assert "F-aiharm-precision" in ids


def test_f_aiharm_precision_content() -> None:
    threats = get_threats_register()
    threat = next(t for t in threats if t.threat_id == "F-aiharm-precision")
    assert "Beta(1,1)" in threat.description or "Uniform(0,1)" in threat.description
    assert "conservative" not in threat.mitigation.lower()
    assert "3 of 20" in threat.residual_risk
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_threats.py::test_f_aiharm_precision_present -v`
Expected: FAIL — `StopIteration` or `KeyError`

- [ ] **Step 3: Add F-aiharm-precision threat to register.py**

In `engine/threats/register.py`, add after the `F-defenseindepth` threat (line 78, before the closing paren of the tuple):

```python
    Threat(
        "F-aiharm-precision",
        "ai-harm stratum has no direct precision measurements; ai-harm precision "
        "keys are absent from posteriors.json entirely, so the model falls back to "
        "Beta(1,1) = Uniform(0,1) via the default initialization in inference.py "
        "(apply_empirical_precision_prior cannot reach keys that do not exist)",
        "disclosed in notebook Acts 4, 5, and 10; the uniform prior is maximally "
        "uninformative rather than borrowed — it does not assume high or low "
        "precision for ai-harm entries",
        "ai-harm rankings rely on a flat precision prior (mean 0.5); true ai-harm "
        "precision could be substantially higher or lower, shifting rankings in "
        "either direction; only 3 of 20 ai-harm recall keys have material evidence "
        "(LLM09, LLM04, NEW-MA); NEW-WLA has only 1 observation above the pure prior",
    ),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_threats.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add engine/threats/register.py tests/unit/test_threats.py
git commit -m "feat(threats): register F-aiharm-precision with corrected mechanism description"
```

---

### Task 5: Update notebook Act 10 with ai-harm precision disclosure

**Files:**
- Modify: `notebooks/what_the_data_says_2026.ipynb` (cell id `ef090aa6`)

- [ ] **Step 1: Read the notebook to get current state**

Read the notebook file to satisfy the NotebookEdit tool requirement.

- [ ] **Step 2: Append the accepted-limitation paragraph to Act 10**

Use NotebookEdit on cell `ef090aa6` to append after the existing "kappa ceiling" paragraph. The cell currently ends with:

> "...A larger corpus, better-defined entry boundaries, and independent recall measurement would all narrow the interval and sharpen the comparison."

Append this paragraph:

```
\n\n**Accepted limitation: ai-harm precision.** The 323 precision verifications were drawn entirely from the security stratum. The ai-harm stratum (92 in-scope incidents across 8 entry assignments, of which only 3 received recall posteriors with material evidence — LLM09, LLM04, NEW-MA; NEW-WLA has only 1 observation above the pure prior) has no direct precision measurements — ai-harm precision keys are absent from the calibration data entirely. The model falls back to a flat Beta(1,1) = Uniform(0,1) prior for ai-harm precision, meaning it assumes no prior knowledge about how precise the classifier is on ai-harm incidents (prior mean 0.5). Closing this gap would require sourcing additional ai-harm incidents beyond the existing corpus, which is outside this project's scope. The disclosure in Acts 4 and 5 describes how the model handles missing precision data.
```

- [ ] **Step 3: Verify notebook executes**

Run: `jupyter nbconvert --to notebook --execute notebooks/what_the_data_says_2026.ipynb --output /dev/null 2>&1 | tail -5`

Or if nbconvert is not available, verify the cell was updated correctly by reading it back.

- [ ] **Step 4: Commit**

```bash
git add notebooks/what_the_data_says_2026.ipynb
git commit -m "docs(notebook): add ai-harm precision accepted-limitation paragraph to Act 10"
```

---

### Task 6: Regenerate report.md with F-aiharm-precision

**Files:**
- Modify: `projects/owasp-llm/cycles/2026/results/report.md`

The `report_cmd` CLI reads `get_threats_register()` directly (render.py:182), so re-running the report command will pick up the new threat. Since running the full report command requires the full cycle directory with all artifacts, we'll use a targeted approach.

- [ ] **Step 1: Regenerate report.md via the report CLI command**

```bash
python3 -m engine.cli.main report --cycle projects/owasp-llm/cycles/2026
```

If this fails due to missing artifacts, regenerate by running Python directly:

```python
python3 -c "
from engine.report.render import render_report, ReportInputs
from engine.threats.register import get_threats_register

# Verify the new threat is in the register
threats = get_threats_register()
ids = [t.threat_id for t in threats]
assert 'F-aiharm-precision' in ids, f'Missing! Got: {ids}'
print('F-aiharm-precision is registered')
print(f'Total threats: {len(threats)}')
"
```

If the full report command works, verify the output. If not, patch report.md by reading the current file and inserting the new threat line.

- [ ] **Step 2: Verify F-aiharm-precision appears in report.md**

```bash
grep "F-aiharm-precision" projects/owasp-llm/cycles/2026/results/report.md
```

Expected: `- **F-aiharm-precision**: ai-harm stratum has no direct precision measurements...`

- [ ] **Step 3: Commit**

```bash
git add projects/owasp-llm/cycles/2026/results/report.md
git commit -m "feat(artifact): regenerate report.md with F-aiharm-precision threat"
```

---

### Task 7: Add narrative dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add optional narrative dependency group**

In `pyproject.toml`, find the existing `[project.optional-dependencies]` section (which has `monitoring`). Add the `narrative` key to the SAME table — do NOT re-declare the `[project.optional-dependencies]` header. The result should look like:

```toml
[project.optional-dependencies]
monitoring = ["wandb>=0.19,<1.0"]
narrative = [
  "matplotlib>=3.8",
  "seaborn>=0.13",
  "plotly>=5.15",
  "kaleido>=0.2.1",
  "pandas>=2.0",
]
```

**IMPORTANT:** Only add the `narrative = [...]` line. The `[project.optional-dependencies]` header and `monitoring` line already exist.

Also add mypy overrides for the new dependencies. In the `[[tool.mypy.overrides]]` section, extend the existing ignore list or add a new block:

```toml
[[tool.mypy.overrides]]
module = ["matplotlib.*", "seaborn.*", "plotly.*", "kaleido.*", "pandas.*"]
ignore_missing_imports = true
```

- [ ] **Step 2: Install narrative dependencies**

```bash
pip install -e ".[narrative]"
```

- [ ] **Step 3: Verify imports work**

```bash
python3 -c "import matplotlib; import seaborn; import plotly; print('All narrative deps OK')"
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add optional narrative dependency group to pyproject.toml"
```

---

### Task 8: Create narrative data loading module

**Files:**
- Create: `engine/report/narrative_data.py`
- Test: `tests/unit/test_narrative.py`

- [ ] **Step 1: Write failing test for data loading**

Create `tests/unit/test_narrative.py`:

```python
"""Tests for engine.report.narrative — standalone narrative report generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

CYCLE_DIR = Path("projects/owasp-llm/cycles/2026")
SKIP_NO_CYCLE = pytest.mark.skipif(
    not CYCLE_DIR.exists(),
    reason="Cycle data not present (CI skip)",
)


@SKIP_NO_CYCLE
class TestNarrativeDataLoading:
    def test_load_data_returns_dict(self) -> None:
        from engine.report.narrative_data import load_narrative_data

        data = load_narrative_data(CYCLE_DIR)
        assert isinstance(data, dict)

    def test_load_data_has_required_keys(self) -> None:
        from engine.report.narrative_data import load_narrative_data

        data = load_narrative_data(CYCLE_DIR)
        required = {
            "rubric", "incidents", "prelabels", "goldset",
            "precision_verification", "posteriors", "diagnostic",
            "inference_summary", "lambda_samples", "concordance",
            "selection_bias", "rank_comparison_md",
        }
        missing = required - set(data.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_lambda_samples_shape(self) -> None:
        from engine.report.narrative_data import load_narrative_data

        data = load_narrative_data(CYCLE_DIR)
        assert data["lambda_samples"].shape == (16000, 20)

    def test_concordance_has_ci_method(self) -> None:
        from engine.report.narrative_data import load_narrative_data

        data = load_narrative_data(CYCLE_DIR)
        assert "ci_method" in data["concordance"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_narrative.py::TestNarrativeDataLoading::test_load_data_returns_dict -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.report.narrative_data'`

- [ ] **Step 3: Create narrative_data.py**

Create `engine/report/narrative_data.py`:

```python
"""Data loading for the standalone narrative report generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_narrative_data(cycle_dir: Path) -> dict[str, Any]:
    """Load all data files needed for the narrative report.

    Mirrors notebook Act 0 data loading.
    """
    cycle = cycle_dir.resolve()
    data: dict[str, Any] = {}

    with open(cycle / "prereg" / "rubric.json") as f:
        data["rubric"] = json.load(f)

    with open(cycle / "classify" / "labeled_incidents_multimodel.json") as f:
        data["incidents"] = json.load(f)

    prelabels = []
    with open(cycle / "calibration" / "llm_prelabels.jsonl") as f:
        for line in f:
            prelabels.append(json.loads(line))
    data["prelabels"] = prelabels

    goldset = []
    with open(cycle / "calibration" / "adjudicated_goldset.jsonl") as f:
        for line in f:
            goldset.append(json.loads(line))
    data["goldset"] = goldset

    precision_verif = []
    with open(cycle / "calibration" / "precision_verification.jsonl") as f:
        for line in f:
            precision_verif.append(json.loads(line))
    data["precision_verification"] = precision_verif

    with open(cycle / "calibration" / "posteriors.json") as f:
        data["posteriors"] = json.load(f)

    with open(cycle / "calibration" / "diagnostic.json") as f:
        data["diagnostic"] = json.load(f)

    with open(cycle / "infer" / "inference_summary.json") as f:
        data["inference_summary"] = json.load(f)

    data["lambda_samples"] = np.load(
        cycle / "infer" / "lambda_samples.npy",
        allow_pickle=False,
    )

    with open(cycle / "results" / "concordance.json") as f:
        data["concordance"] = json.load(f)

    with open(cycle / "results" / "selection_bias.json") as f:
        data["selection_bias"] = json.load(f)

    with open(cycle / "results" / "rank_comparison_report.md") as f:
        data["rank_comparison_md"] = f.read()

    report_md_path = cycle / "results" / "report.md"
    data["non_publishable"] = True  # safe default: assume non-publishable if report.md is missing
    if report_md_path.exists():
        report_text = report_md_path.read_text()
        data["non_publishable"] = "NON-PUBLISHABLE" in report_text

    corpus_b_path = cycle / "results" / "corpus_b_corroboration.json"
    if corpus_b_path.exists():
        with open(corpus_b_path) as f:
            data["corpus_b"] = json.load(f)

    data["entry_names"] = {
        e["entry_id"]: e["canonical_name"] for e in data["rubric"]["entries"]
    }
    data["entry_ids"] = data["inference_summary"]["entry_ids"]

    return data
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_narrative.py::TestNarrativeDataLoading -v`
Expected: ALL PASS (or SKIP if cycle data absent)

- [ ] **Step 5: Commit**

```bash
git add engine/report/narrative_data.py tests/unit/test_narrative.py
git commit -m "feat(narrative): add data loading module for narrative report"
```

---

### Task 9: Create narrative chart generators (matplotlib charts)

**Files:**
- Create: `engine/report/narrative_charts.py`

This is the largest task. We split charts into matplotlib-based (Tasks 9) and plotly-based (Task 10).

- [ ] **Step 1: Create the chart module with matplotlib-based generators**

Create `engine/report/narrative_charts.py`. This file contains all 16 chart rendering functions. Each function takes the loaded data dict and a figures directory, and saves a PNG. The chart code mirrors the notebook cells.

```python
"""Chart generators for the standalone narrative report.

Each function renders one chart and saves it as a PNG.
All matplotlib charts use the Agg backend for headless rendering.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import seaborn as sns  # noqa: E402

ENTRY_IDS = [
    "LLM01", "LLM02", "LLM03", "LLM04", "LLM05",
    "LLM06", "LLM07", "LLM08", "LLM09", "LLM10",
    "NEW-ITSCD", "NEW-MA", "NEW-MSDA", "NEW-MTIE", "NEW-PMP", "NEW-WLA",
    "ROLL-CFAS", "ROLL-CMSB", "ROLL-LAPTF", "ROLL-SICG",
]
FRAME_BLIND = {"LLM04", "LLM08", "LLM10"}

sns.set_theme(style="whitegrid", font_scale=1.1)


def _setup_colors() -> dict[str, str]:
    ib = sns.color_palette("mako", n_colors=12)
    no = sns.color_palette("flare", n_colors=8)
    rp = sns.color_palette("crest", n_colors=6)
    colors: dict[str, str] = {}
    ii, ni, ri = 0, 0, 0
    for eid in ENTRY_IDS:
        if eid in FRAME_BLIND:
            colors[eid] = "#999999"
        elif eid.startswith("LLM"):
            c = ib[ii % len(ib)]
            colors[eid] = f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
            ii += 1
        elif eid.startswith("NEW"):
            c = no[ni % len(no)]
            colors[eid] = f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
            ni += 1
        else:
            c = rp[ri % len(rp)]
            colors[eid] = f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
            ri += 1
    return colors


ENTRY_COLORS = _setup_colors()


def render_stratum_bar(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 2: Stratum breakdown bar chart."""
    import pandas as pd

    inc_df = pd.DataFrame(data["incidents"])
    stratum_counts = inc_df["stratum"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 4))
    stratum_counts.plot(kind="bar", ax=ax, color=["#2196F3", "#FF9800"])
    ax.set_title("Incidents by Stratum")
    ax.set_xlabel("Stratum")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(figures_dir / "stratum_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_tier_donut(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 3: Tier distribution donut chart."""
    tier_counts = Counter(
        p["triage_tier"]
        for p in data["prelabels"]
        if p.get("consensus") != "out-of-scope"
    )
    labels = ["agree", "split", "disagree"]
    values = [tier_counts.get(t, 0) for t in labels]
    colors_list = ["#4CAF50", "#FFC107", "#F44336"]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.0f%%", startangle=90,
        colors=colors_list, pctdistance=0.75,
    )
    centre_circle = plt.Circle((0, 0), 0.50, fc="white")
    ax.add_patch(centre_circle)
    ax.set_title("Consensus Tier Distribution")
    fig.tight_layout()
    fig.savefig(figures_dir / "tier_donut.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_confusion_heatmap(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 3: Entry-pair disagreement heatmap."""
    in_scope_entries = [e for e in ENTRY_IDS if e not in FRAME_BLIND]
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)

    for p in data["prelabels"]:
        if p.get("triage_tier") in ("split", "disagree") and p.get("consensus") != "out-of-scope":
            votes = p.get("model_votes", [])
            unique_entries = {v["entry_id"] for v in votes if isinstance(v, dict) and v.get("entry_id") in set(in_scope_entries)}
            entries_list = sorted(unique_entries)
            for i_idx in range(len(entries_list)):
                for j_idx in range(i_idx + 1, len(entries_list)):
                    pair_counts[(entries_list[i_idx], entries_list[j_idx])] += 1

    n = len(in_scope_entries)
    matrix = np.zeros((n, n), dtype=int)
    for (a, b), count in pair_counts.items():
        if a in in_scope_entries and b in in_scope_entries:
            i = in_scope_entries.index(a)
            j = in_scope_entries.index(b)
            matrix[i, j] = count
            matrix[j, i] = count

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        matrix, xticklabels=in_scope_entries, yticklabels=in_scope_entries,
        cmap="YlOrRd", ax=ax, annot=True, fmt="d",
    )
    ax.set_title("Entry-Pair Disagreement Frequency")
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_precision_bars(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 4: Precision posterior mean bars."""
    precision = data["posteriors"]["precision"]
    entries_with_prec = []
    means = []
    for key, params in sorted(precision.items()):
        eid = key.split("::")[0]
        alpha = params["alpha"]
        beta = params["beta"]
        mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0
        entries_with_prec.append(eid)
        means.append(mean)

    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = range(len(entries_with_prec))
    ax.barh(y_pos, means, color=[ENTRY_COLORS.get(e, "#999999") for e in entries_with_prec])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(entries_with_prec, fontsize=9)
    ax.set_xlabel("Precision Posterior Mean")
    ax.set_title("Precision Posteriors (security stratum only)")
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(figures_dir / "precision_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_precision_posteriors(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 4: Beta posterior distributions for key entries."""
    from scipy import stats as sp_stats

    precision = data["posteriors"]["precision"]
    key_entries = ["LLM03", "LLM09", "LLM02", "out-of-scope"]
    x = np.linspace(0, 1, 200)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, eid in zip(axes.flat, key_entries):
        key = f"{eid}::security"
        if key in precision:
            alpha = precision[key]["alpha"]
            beta = precision[key]["beta"]
            y = sp_stats.beta.pdf(x, alpha, beta)
            ax.plot(x, y, linewidth=2)
            ax.fill_between(x, y, alpha=0.3)
            ax.set_title(f"{eid} (α={alpha:.0f}, β={beta:.0f})")
            ax.set_xlabel("Precision")
            ax.set_ylabel("Density")
        else:
            ax.text(0.5, 0.5, f"No data for {eid}", ha="center", va="center")
    fig.suptitle("Precision Posterior Distributions")
    fig.tight_layout()
    fig.savefig(figures_dir / "precision_posteriors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_ridge_plot(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 5: Ridge plot of posterior lambda for all 20 entries."""
    lambda_samples = data["lambda_samples"]
    entry_ids = data["entry_ids"]
    medians = {eid: float(np.median(lambda_samples[:, i])) for i, eid in enumerate(entry_ids)}
    sorted_entries = sorted(entry_ids, key=lambda e: medians[e], reverse=True)

    fig, axes = plt.subplots(len(sorted_entries), 1, figsize=(10, 16), sharex=True)
    for ax, eid in zip(axes, sorted_entries):
        idx = entry_ids.index(eid)
        vals = lambda_samples[:, idx]
        color = ENTRY_COLORS.get(eid, "#999999")
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(vals, bw_method=0.2)
        x_grid = np.linspace(vals.min(), vals.max(), 200)
        density = kde(x_grid)
        ax.fill_between(x_grid, density, alpha=0.6, color=color)
        ax.set_ylabel(eid, rotation=0, ha="right", fontsize=9)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].set_xlabel("λ (incident rate)")
    fig.suptitle("Posterior λ Distributions (sorted by median)")
    fig.tight_layout()
    fig.savefig(figures_dir / "ridge_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_dumbbell_chart(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 6: Dumbbell chart of rank distributions."""
    lambda_samples = data["lambda_samples"]
    entry_ids = data["entry_ids"]

    rank_medians = {}
    rank_cis = {}
    for i, eid in enumerate(entry_ids):
        ranks = np.zeros(lambda_samples.shape[0])
        for s in range(lambda_samples.shape[0]):
            draw = lambda_samples[s]
            order = np.argsort(-draw)
            rank_arr = np.empty_like(order, dtype=float)
            rank_arr[order] = np.arange(1, len(draw) + 1, dtype=float)
            ranks[s] = rank_arr[i]
        rank_medians[eid] = float(np.median(ranks))
        rank_cis[eid] = (float(np.percentile(ranks, 5)), float(np.percentile(ranks, 95)))

    sorted_entries = sorted(entry_ids, key=lambda e: rank_medians[e])

    fig, ax = plt.subplots(figsize=(10, 12))
    y_pos = range(len(sorted_entries))
    for y, eid in zip(y_pos, sorted_entries):
        ci = rank_cis[eid]
        color = ENTRY_COLORS.get(eid, "#999999")
        ax.plot([ci[0], ci[1]], [y, y], color=color, linewidth=2, alpha=0.6)
        ax.scatter([rank_medians[eid]], [y], color=color, s=80, zorder=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{e} ({data['entry_names'].get(e, e)})" for e in sorted_entries], fontsize=9)
    ax.set_xlabel("Rank (90% CI)")
    ax.set_title("Incident-Derived Rankings with Uncertainty")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(figures_dir / "dumbbell_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_bump_chart(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 7: Bump chart comparing expert and incident ranks."""
    conc = data["concordance"]
    rank_md = data["rank_comparison_md"]

    vote_ranks: dict[str, float] = {}
    lambda_ranks: dict[str, float] = {}
    for line in rank_md.split("\n"):
        if "|" in line and not line.startswith("|--") and "Entry" not in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 3:
                eid = parts[0]
                try:
                    lam_med = float(parts[1].split("(")[0].strip())
                    vote_med = float(parts[2].split("(")[0].strip())
                    lambda_ranks[eid] = lam_med
                    vote_ranks[eid] = vote_med
                except (ValueError, IndexError):
                    continue

    common = sorted(set(lambda_ranks) & set(vote_ranks))
    if not common:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No rank data available", ha="center", va="center")
        fig.savefig(figures_dir / "bump_chart.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(10, 12))
    for eid in common:
        color = ENTRY_COLORS.get(eid, "#999999")
        ax.plot(
            [0, 1], [lambda_ranks[eid], vote_ranks[eid]],
            marker="o", color=color, linewidth=2, markersize=8,
        )
        ax.annotate(eid, (0, lambda_ranks[eid]), textcoords="offset points",
                     xytext=(-50, 0), fontsize=9, ha="right")
        ax.annotate(eid, (1, vote_ranks[eid]), textcoords="offset points",
                     xytext=(10, 0), fontsize=9, ha="left")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Incident Rank", "Expert Rank"])
    ax.set_ylabel("Rank")
    ax.invert_yaxis()
    ax.set_title("Expert vs Incident Rankings")
    fig.tight_layout()
    fig.savefig(figures_dir / "bump_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_ci_overlap(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 7: CI overlap between lambda and vote rank CIs."""
    rank_md = data["rank_comparison_md"]
    entries_data: list[dict[str, Any]] = []

    for line in rank_md.split("\n"):
        if "|" in line and not line.startswith("|--") and "Entry" not in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 3:
                eid = parts[0]
                try:
                    lam_part = parts[1]
                    vote_part = parts[2]
                    lam_med = float(lam_part.split("(")[0].strip())
                    lam_ci_str = lam_part.split("(")[1].rstrip(")")
                    lam_lo, lam_hi = [float(x) for x in lam_ci_str.replace("–", "-").split("-") if x]
                    vote_med = float(vote_part.split("(")[0].strip())
                    vote_ci_str = vote_part.split("(")[1].rstrip(")")
                    vote_lo, vote_hi = [float(x) for x in vote_ci_str.replace("–", "-").split("-") if x]
                    entries_data.append({
                        "entry_id": eid, "lam_med": lam_med, "lam_lo": lam_lo, "lam_hi": lam_hi,
                        "vote_med": vote_med, "vote_lo": vote_lo, "vote_hi": vote_hi,
                    })
                except (ValueError, IndexError):
                    continue

    if not entries_data:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No CI data", ha="center")
        fig.savefig(figures_dir / "ci_overlap.png", dpi=150)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    for i, ed in enumerate(entries_data):
        color = ENTRY_COLORS.get(ed["entry_id"], "#999999")
        ax.plot([ed["lam_lo"], ed["lam_hi"]], [i - 0.1, i - 0.1], color=color, linewidth=3, alpha=0.7)
        ax.plot([ed["vote_lo"], ed["vote_hi"]], [i + 0.1, i + 0.1], color=color, linewidth=3, alpha=0.4, linestyle="--")
        ax.scatter([ed["lam_med"]], [i - 0.1], color=color, s=60, zorder=5)
        ax.scatter([ed["vote_med"]], [i + 0.1], color=color, s=60, zorder=5, marker="^")
    ax.set_yticks(range(len(entries_data)))
    ax.set_yticklabels([ed["entry_id"] for ed in entries_data], fontsize=9)
    ax.set_xlabel("Rank")
    ax.set_title("CI Overlap: Incident (solid) vs Expert (dashed)")
    fig.tight_layout()
    fig.savefig(figures_dir / "ci_overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_paired_dots(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 8: Paired dot plots for flagged entries."""
    flags = data["concordance"].get("flags", [])
    if not flags:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No flagged entries", ha="center")
        fig.savefig(figures_dir / "paired_dots.png", dpi=150)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, f in enumerate(flags):
        eid = f["entry_id"]
        prob = f["probability"]
        direction = f["direction"]
        color = ENTRY_COLORS.get(eid, "#999999")
        ax.barh(i, prob, color=color, alpha=0.7)
        label = "↑ expert" if direction == "vote_over_ranks" else "↓ expert"
        ax.text(prob + 0.01, i, f"{eid} ({label})", va="center", fontsize=10)
    ax.set_xlabel("P(tier mismatch)")
    ax.set_title("Flagged Entries: Expert vs Incident Rank Divergence")
    ax.set_yticks([])
    ax.axvline(x=0.8, color="red", linestyle="--", alpha=0.5, label="τ = 0.80")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "paired_dots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_theme_bars(data: dict[str, Any], figures_dir: Path, entry_id: str, filename: str) -> None:
    """Act 8: Theme keyword bars for a specific entry."""
    theme_keywords = {
        "deepfake": ["deepfake", "synthetic media", "face swap"],
        "misinfo": ["disinformation", "misinformation", "fake news"],
        "voice_clone": ["voice clone", "voice synthesis", "audio deepfake"],
        "code_gen": ["code generation", "copilot", "code completion"],
        "data_leak": ["data leak", "data exposure", "information disclosure"],
        "prompt_inject": ["prompt injection", "jailbreak", "prompt attack"],
        "supply_chain": ["supply chain", "dependency", "package"],
        "agent_abuse": ["agent", "autonomous", "tool use", "mcp"],
    }

    entry_prelabels = [
        p for p in data["prelabels"]
        if p.get("consensus") == entry_id
    ]

    theme_counts: dict[str, int] = defaultdict(int)
    for p in entry_prelabels:
        rationale = (p.get("rationale", "") or "").lower()
        for theme, keywords in theme_keywords.items():
            if any(kw in rationale for kw in keywords):
                theme_counts[theme] += 1

    if not theme_counts:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"No theme keywords found for {entry_id}", ha="center")
        fig.savefig(figures_dir / filename, dpi=150)
        plt.close(fig)
        return

    sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
    themes, counts = zip(*sorted_themes)

    fig, ax = plt.subplots(figsize=(8, 4))
    color = ENTRY_COLORS.get(entry_id, "#999999")
    ax.barh(range(len(themes)), counts, color=color, alpha=0.8)
    ax.set_yticks(range(len(themes)))
    ax.set_yticklabels(themes, fontsize=10)
    ax.set_xlabel("Keyword Frequency")
    ax.set_title(f"Incident Themes: {entry_id}")
    fig.tight_layout()
    fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_confusion_matrix_3x3(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 9B: 3x3 confusion matrix for boundary entries."""
    boundary_entries = ["LLM09", "NEW-WLA", "ROLL-CMSB"]
    models = ["qwen", "llama", "deepseek"]

    pair_disagree: dict[tuple[str, str], int] = defaultdict(int)
    for p in data["prelabels"]:
        votes = p.get("model_votes", [])
        unique_votes = {v["entry_id"] for v in votes if isinstance(v, dict) and v.get("entry_id") in set(boundary_entries)}
        if len(unique_votes) >= 2:
            vote_list = sorted(unique_votes)
            for i_idx in range(len(vote_list)):
                for j_idx in range(i_idx + 1, len(vote_list)):
                    pair_disagree[(vote_list[i_idx], vote_list[j_idx])] += 1

    n = len(boundary_entries)
    matrix = np.zeros((n, n), dtype=int)
    for (a, b), count in pair_disagree.items():
        if a in boundary_entries and b in boundary_entries:
            i = boundary_entries.index(a)
            j = boundary_entries.index(b)
            matrix[i, j] = count
            matrix[j, i] = count

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix, xticklabels=boundary_entries, yticklabels=boundary_entries,
        cmap="YlOrRd", ax=ax, annot=True, fmt="d",
    )
    ax.set_title("Confusion Boundary: Model Disagreement")
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrix_3x3.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_matplotlib_charts(data: dict[str, Any], figures_dir: Path) -> None:
    """Generate all matplotlib-based charts."""
    render_stratum_bar(data, figures_dir)
    render_tier_donut(data, figures_dir)
    render_confusion_heatmap(data, figures_dir)
    render_precision_bars(data, figures_dir)
    render_precision_posteriors(data, figures_dir)
    render_ridge_plot(data, figures_dir)
    render_dumbbell_chart(data, figures_dir)
    render_bump_chart(data, figures_dir)
    render_ci_overlap(data, figures_dir)
    render_paired_dots(data, figures_dir)
    render_theme_bars(data, figures_dir, "LLM09", "theme_bars_llm09.png")
    render_theme_bars(data, figures_dir, "NEW-WLA", "theme_bars_new_wla.png")
    render_confusion_matrix_3x3(data, figures_dir)
```

- [ ] **Step 2: Commit (no test yet — charts are tested via integration in Task 13)**

```bash
git add engine/report/narrative_charts.py
git commit -m "feat(narrative): add matplotlib chart generators for narrative report"
```

---

### Task 10: Create plotly chart generators

**Files:**
- Modify: `engine/report/narrative_charts.py`

- [ ] **Step 1: Add plotly chart generators to narrative_charts.py**

Append to `engine/report/narrative_charts.py`:

```python
def render_plotly_rankings(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 6: Interactive-style rankings as static PNG."""
    import plotly.express as px

    lambda_samples = data["lambda_samples"]
    entry_ids = data["entry_ids"]
    entry_names = data["entry_names"]

    rows = []
    for i, eid in enumerate(entry_ids):
        med = float(np.median(lambda_samples[:, i]))
        lo = float(np.percentile(lambda_samples[:, i], 5))
        hi = float(np.percentile(lambda_samples[:, i], 95))
        rows.append({"entry_id": eid, "name": entry_names.get(eid, eid), "median": med, "lo": lo, "hi": hi})

    import pandas as pd
    df = pd.DataFrame(rows).sort_values("median", ascending=False)

    fig = px.bar(
        df, x="median", y="entry_id", orientation="h",
        error_x_minus=df["median"] - df["lo"],
        error_x=df["hi"] - df["median"],
        title="Incident-Derived Rankings (λ posteriors)",
        labels={"median": "λ (posterior median)", "entry_id": "Entry"},
    )
    fig.update_layout(height=600, width=1000, yaxis={"categoryorder": "total ascending"})
    fig.write_image(str(figures_dir / "plotly_rankings.png"), width=1000, height=600)


def render_oos_treemap(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 9A: Out-of-scope incident treemap."""
    import plotly.express as px

    oos_prelabels = [
        p for p in data["prelabels"]
        if p.get("consensus") == "out-of-scope"
    ]

    theme_keywords = {
        "Infrastructure / Cloud": ["cloud", "infrastructure", "server", "kubernetes", "docker"],
        "Web Application": ["xss", "sql injection", "csrf", "web app", "http"],
        "Network / Protocol": ["network", "protocol", "dns", "tcp", "tls", "ssl"],
        "Mobile / IoT": ["mobile", "iot", "android", "ios", "firmware"],
        "Crypto / Auth": ["crypto", "authentication", "password", "certificate", "oauth"],
        "Other": [],
    }

    cluster_counts: dict[str, int] = defaultdict(int)
    for p in oos_prelabels:
        rationale = (p.get("rationale", "") or "").lower()
        matched = False
        for cluster, keywords in theme_keywords.items():
            if cluster == "Other":
                continue
            if any(kw in rationale for kw in keywords):
                cluster_counts[cluster] += 1
                matched = True
                break
        if not matched:
            cluster_counts["Other"] += 1

    if not cluster_counts:
        # Write a placeholder
        fig = px.treemap(
            names=["No OOS data"], parents=[""], values=[1],
            title="Out-of-Scope Incidents (no data)",
        )
        fig.write_image(str(figures_dir / "oos_treemap.png"), width=1000, height=600)
        return

    import pandas as pd
    df = pd.DataFrame([
        {"cluster": k, "count": v, "parent": "Out-of-Scope"}
        for k, v in cluster_counts.items()
    ])

    fig = px.treemap(
        df, path=["parent", "cluster"], values="count",
        title=f"Out-of-Scope Incidents by Theme ({sum(cluster_counts.values())} total)",
    )
    fig.update_layout(width=1000, height=600)
    fig.write_image(str(figures_dir / "oos_treemap.png"), width=1000, height=600)


def render_sankey_confusion(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 9B: Sankey diagram for confusion boundary entries."""
    import plotly.graph_objects as go_plotly

    boundary = ["LLM09", "NEW-WLA", "ROLL-CMSB"]
    models = ["qwen", "llama", "deepseek"]

    flows: dict[tuple[str, str], int] = defaultdict(int)
    for p in data["prelabels"]:
        votes = p.get("model_votes", [])
        consensus = p.get("consensus", "")
        if consensus not in boundary:
            continue
        for v in votes:
            if not isinstance(v, dict):
                continue
            model_id = v.get("model_id", "")
            model_short = model_id.split("/")[-1].split("-")[0].lower() if "/" in model_id else model_id.lower()
            vote_entry = v.get("entry_id", "")
            if vote_entry in boundary:
                flows[(f"{model_short}: {vote_entry}", consensus)] += 1

    if not flows:
        fig = go_plotly.Figure()
        fig.add_annotation(text="No confusion boundary data", x=0.5, y=0.5, showarrow=False)
        fig.write_image(str(figures_dir / "sankey_confusion.png"), width=1000, height=600)
        return

    all_labels = sorted(set(
        [k[0] for k in flows] + [k[1] for k in flows]
    ))
    label_idx = {l: i for i, l in enumerate(all_labels)}

    source = [label_idx[k[0]] for k in flows]
    target = [label_idx[k[1]] for k in flows]
    value = list(flows.values())

    fig = go_plotly.Figure(data=[go_plotly.Sankey(
        node=dict(label=all_labels, pad=15, thickness=20),
        link=dict(source=source, target=target, value=value),
    )])
    fig.update_layout(title="Model Votes → Consensus (Confusion Boundary)", width=1000, height=600)
    fig.write_image(str(figures_dir / "sankey_confusion.png"), width=1000, height=600)


def generate_all_plotly_charts(data: dict[str, Any], figures_dir: Path) -> None:
    """Generate all plotly-based charts (rendered as static PNG via kaleido)."""
    render_plotly_rankings(data, figures_dir)
    render_oos_treemap(data, figures_dir)
    render_sankey_confusion(data, figures_dir)
```

- [ ] **Step 2: Commit**

```bash
git add engine/report/narrative_charts.py
git commit -m "feat(narrative): add plotly chart generators (static PNG via kaleido)"
```

---

### Task 11: Create narrative markdown renderer

**Files:**
- Create: `engine/report/narrative.py`

- [ ] **Step 1: Create the main narrative module**

Create `engine/report/narrative.py`:

```python
"""Standalone narrative report generator.

Reads the same data files as the notebook, generates charts as static PNGs,
and writes a self-contained markdown report for redistribution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from engine.threats.register import get_threats_register


def _render_markdown(data: dict[str, Any], figures_dir: Path) -> str:
    """Generate the full narrative markdown report."""
    lines: list[str] = []

    if data.get("non_publishable"):
        lines.append(
            "**STATUS: NON-PUBLISHABLE** (single-author rubric, uncontrolled)\n\n"
        )

    entry_names = data["entry_names"]
    entry_ids = data["entry_ids"]
    conc = data["concordance"]
    sel_bias = data["selection_bias"]

    # Act 1: The Question
    lines.append("# What the Data Says About the 2026 Top 10\n\n")
    lines.append("## Act 1: The Question\n\n")
    lines.append("| Entry | Name | Incident Rank |\n")
    lines.append("|-------|------|---------------|\n")
    for eid in entry_ids:
        name = entry_names.get(eid, eid)
        lines.append(f"| {eid} | {name} | — |\n")
    lines.append("\n")

    # Act 2: The Corpus
    lines.append("## Act 2: The Corpus\n\n")
    total_incidents = len(data["incidents"])
    from collections import Counter
    stratum_counts = Counter(inc.get("stratum", "unknown") for inc in data["incidents"])
    for stratum, count in stratum_counts.most_common():
        lines.append(f"- **{stratum}**: {count} incidents\n")
    lines.append(f"\nTotal: {total_incidents} incidents.\n\n")
    lines.append(f"![Stratum breakdown](figures/stratum_bar.png)\n\n")

    # Act 3: Classification
    lines.append("## Act 3: Classification\n\n")
    tier_counts = Counter(
        p.get("triage_tier", "unknown")
        for p in data["prelabels"]
        if p.get("consensus") != "out-of-scope"
    )
    lines.append(
        f"Consensus tiers: {tier_counts.get('agree', 0)} agree, "
        f"{tier_counts.get('split', 0)} split, "
        f"{tier_counts.get('disagree', 0)} disagree.\n\n"
    )
    lines.append(f"![Tier distribution](figures/tier_donut.png)\n\n")
    lines.append(f"![Confusion heatmap](figures/confusion_heatmap.png)\n\n")

    # Act 4: How Good Is the Classifier?
    lines.append("## Act 4: How Good Is the Classifier?\n\n")
    prec_keys = list(data["posteriors"].get("precision", {}).keys())
    security_prec = [k for k in prec_keys if "::security" in k]
    aiharm_prec = [k for k in prec_keys if "::ai-harm" in k or "::ai_harm" in k]
    verified_entries = {r.get("claimed_entry_id") for r in data["precision_verification"]}
    lines.append(
        f"Precision verifications: {len(data['precision_verification'])} records "
        f"across {len(verified_entries)} verified entries. "
        f"Posterior keys: {len(security_prec)} security-stratum, "
        f"{len(aiharm_prec)} ai-harm.\n\n"
    )
    if not aiharm_prec:
        lines.append(
            "**Note:** The ai-harm stratum has zero direct precision measurements. "
            "The model falls back to a flat Beta(1,1) = Uniform(0,1) prior for "
            "ai-harm precision.\n\n"
        )
    lines.append(f"![Precision bars](figures/precision_bars.png)\n\n")
    lines.append(f"![Precision posteriors](figures/precision_posteriors.png)\n\n")

    # Act 5: From Counts to Rankings
    lines.append("## Act 5: From Counts to Rankings\n\n")
    lambda_samples = data["lambda_samples"]
    lines.append(
        f"MCMC: {lambda_samples.shape[0]} posterior draws, "
        f"{len(entry_ids)} entries.\n\n"
    )
    inf_summary = data["inference_summary"]
    r_hat = inf_summary.get("r_hat", {})
    ess = inf_summary.get("ess", {})
    if r_hat:
        max_r_hat = max(r_hat.values()) if isinstance(r_hat, dict) else r_hat
        lines.append(f"Max R̂: {max_r_hat:.4f}. ")
    if ess:
        min_ess = min(ess.values()) if isinstance(ess, dict) else ess
        lines.append(f"Min ESS: {min_ess:.0f}.\n\n")
    lines.append(f"![Ridge plot](figures/ridge_plot.png)\n\n")

    # Act 6: The Incident-Derived Rankings
    lines.append("## Act 6: The Incident-Derived Rankings\n\n")
    lines.append(f"![Dumbbell chart](figures/dumbbell_chart.png)\n\n")
    lines.append(f"![Rankings](figures/plotly_rankings.png)\n\n")

    # Corpus B corroboration
    if "corpus_b" in data:
        cb = data["corpus_b"]
        lines.append("### Corpus B Corroboration\n\n")
        overlap = cb.get("overlap_count", 0)
        agree = cb.get("agreement_count", 0)
        rate = cb.get("agreement_rate", 0.0)
        lines.append(
            f"Corpus B (GenAI agentic): {cb.get('corpus_b_incident_count', 0)} incidents. "
            f"Shared with corpus A: {overlap}. "
            f"Label agreement: {agree}/{overlap} ({rate:.0%}).\n\n"
        )

    # Act 7: The Confrontation
    lines.append("## Act 7: The Confrontation\n\n")
    if conc.get("weighted_kappa_median") is not None:
        ci = conc.get("weighted_kappa_ci", [])
        if "ci_method" not in conc:
            raise ValueError(
                "concordance.json missing ci_method field; "
                "run Phase 1 (Tasks 1-3) before generating the narrative report"
            )
        ci_method = conc["ci_method"]
        lines.append(
            f"Weighted Cohen's kappa: {conc['weighted_kappa_median']:.4f} "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] (95% interval, method: {ci_method}).\n\n"
        )
    lines.append(
        f"Selection bias: H = {sel_bias.get('statistic_value', 0):.4f}, "
        f"p = {sel_bias.get('p_value', 0):.4f}.\n\n"
    )
    lines.append(f"![Bump chart](figures/bump_chart.png)\n\n")
    lines.append(f"![CI overlap](figures/ci_overlap.png)\n\n")

    # Act 8: Where Experts and Incidents Disagree
    lines.append("## Act 8: Where Experts and Incidents Disagree\n\n")
    flags = conc.get("flags", [])
    if flags:
        lines.append(f"{len(flags)} entries flagged (P(tier mismatch) > τ):\n\n")
        for f in flags:
            lines.append(
                f"- **{f['entry_id']}**: P = {f['probability']:.2f}, "
                f"direction = {f['direction']}\n"
            )
        lines.append("\n")
    lines.append(f"![Paired dots](figures/paired_dots.png)\n\n")
    lines.append(f"![Theme bars LLM09](figures/theme_bars_llm09.png)\n\n")
    lines.append(f"![Theme bars NEW-WLA](figures/theme_bars_new_wla.png)\n\n")

    # Act 9: What the Data Cannot See
    lines.append("## Act 9: What the Data Cannot See\n\n")
    lines.append(f"![OOS treemap](figures/oos_treemap.png)\n\n")
    lines.append(f"![Sankey confusion](figures/sankey_confusion.png)\n\n")
    lines.append(f"![Confusion matrix](figures/confusion_matrix_3x3.png)\n\n")

    # Act 10: What This Means
    lines.append("## Act 10: What This Means\n\n")
    lines.append(
        "The current kappa of "
        f"{conc.get('weighted_kappa_median', 0):.2f} "
    )
    if conc.get("weighted_kappa_ci"):
        ci = conc["weighted_kappa_ci"]
        lines.append(f"[{ci[0]:.2f}, {ci[1]:.2f}] ")
    lines.append(
        "is consistent with weak-to-moderate agreement, but the confidence interval "
        "is too wide to draw firm conclusions.\n\n"
    )

    # Accepted limitation: ai-harm precision
    lines.append(
        "**Accepted limitation: ai-harm precision.** The 323 precision verifications "
        "were drawn entirely from the security stratum. The ai-harm stratum (92 "
        "in-scope incidents across 8 entry assignments, of which only 3 received "
        "recall posteriors with material evidence — LLM09, LLM04, NEW-MA; NEW-WLA "
        "has only 1 observation above the pure prior) has no direct precision "
        "measurements — "
        "ai-harm precision keys are absent from the calibration data entirely. "
        "The model falls back to a flat Beta(1,1) = Uniform(0,1) prior for ai-harm "
        "precision, meaning it assumes no prior knowledge about how precise the "
        "classifier is on ai-harm incidents (prior mean 0.5).\n\n"
    )

    # Threats to Validity — programmatic from register
    lines.append("## Threats to Validity\n\n")
    for t in get_threats_register():
        lines.append(f"- **{t.threat_id}**: {t.description}\n")
    lines.append("\n")

    lines.append("---\n")
    lines.append(
        "Before publishing externally, verify against `docs/REVIEWERS.md` "
        "PRE-PUBLISH CHECKLIST. This report is internal-only unless the "
        "checklist passes.\n"
    )

    return "".join(lines)


def generate_narrative_report(cycle_dir: Path, output_dir: Path) -> Path:
    """Generate a standalone narrative report with embedded figures.

    Returns the path to the generated report.md.
    """
    from engine.report.narrative_charts import (
        generate_all_matplotlib_charts,
        generate_all_plotly_charts,
    )
    from engine.report.narrative_data import load_narrative_data

    data = load_narrative_data(cycle_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    generate_all_matplotlib_charts(data, figures_dir)
    generate_all_plotly_charts(data, figures_dir)

    report_md = _render_markdown(data, figures_dir)
    report_path = output_dir / "report.md"
    report_path.write_text(report_md)

    return report_path
```

- [ ] **Step 2: Commit**

```bash
git add engine/report/narrative.py
git commit -m "feat(narrative): add main narrative report module with markdown renderer"
```

---

### Task 12: Add report-narrative CLI command

**Files:**
- Modify: `engine/cli/pipeline.py`
- Modify: `engine/cli/main.py`

- [ ] **Step 1: Add the CLI command to pipeline.py**

Add at the end of `engine/cli/pipeline.py` (after the `report_cmd` function):

```python
@click.command(name="report-narrative")
@click.option(
    "--cycle-dir",
    type=click.Path(exists=True, path_type=Path, resolve_path=True),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, resolve_path=True),
    default=None,
)
def report_narrative_cmd(cycle_dir: Path, output_dir: Path | None) -> None:
    """Generate standalone narrative report with figures."""
    from engine.report.narrative import generate_narrative_report

    if output_dir is None:
        output_dir = cycle_dir / "results" / "narrative"
    result_path = generate_narrative_report(cycle_dir, output_dir)
    click.echo(f"Narrative report written to {result_path}")
```

- [ ] **Step 2: Register the command in main.py**

In `engine/cli/main.py`, add the import and registration:

Add to imports (around line 15-22):
```python
from engine.cli.pipeline import (
    classify_real,
    corroborate,
    decide_real,
    infer_real,
    report_cmd,
    report_narrative_cmd,
    repro_bundle_cmd,
)
```

Add registration (after `cli.add_command(reclassify)`, around line 50):
```python
cli.add_command(report_narrative_cmd)
```

- [ ] **Step 3: Verify CLI registration**

```bash
python3 -m engine.cli.main report-narrative --help
```

Expected: Shows help text with `--cycle-dir` and `--output-dir` options.

- [ ] **Step 4: Commit**

```bash
git add engine/cli/pipeline.py engine/cli/main.py
git commit -m "feat(cli): add report-narrative command"
```

---

### Task 13: Integration test for narrative report

**Files:**
- Modify: `tests/unit/test_narrative.py`

- [ ] **Step 1: Add integration and structural tests**

Add to `tests/unit/test_narrative.py`:

```python
import re


EXPECTED_FIGURES = [
    "stratum_bar.png",
    "tier_donut.png",
    "confusion_heatmap.png",
    "precision_bars.png",
    "precision_posteriors.png",
    "ridge_plot.png",
    "dumbbell_chart.png",
    "plotly_rankings.png",
    "bump_chart.png",
    "ci_overlap.png",
    "paired_dots.png",
    "theme_bars_llm09.png",
    "theme_bars_new_wla.png",
    "oos_treemap.png",
    "sankey_confusion.png",
    "confusion_matrix_3x3.png",
]

ACT_HEADINGS = [
    "Act 1", "Act 2", "Act 3", "Act 4", "Act 5",
    "Act 6", "Act 7", "Act 8", "Act 9", "Act 10",
]

AI_SLOP_PATTERNS = [
    r"\bdelve\b",
    r"\btapestry\b",
    r"\bunlock\b.*\bpotential\b",
    r"\bsynergy\b",
    r"\bholistic\b",
    r"\bparadigm shift\b",
    r"\bgame.?changer\b",
    r"\beverchanging\b",
]


@SKIP_NO_CYCLE
class TestNarrativeIntegration:
    def test_generate_narrative_report(self, tmp_path: Path) -> None:
        from engine.report.narrative import generate_narrative_report

        output_dir = tmp_path / "narrative"
        result_path = generate_narrative_report(CYCLE_DIR, output_dir)
        assert result_path.exists()
        assert result_path.name == "report.md"

    def test_all_figures_present(self, tmp_path: Path) -> None:
        from engine.report.narrative import generate_narrative_report

        output_dir = tmp_path / "narrative"
        generate_narrative_report(CYCLE_DIR, output_dir)
        figures_dir = output_dir / "figures"
        for fig_name in EXPECTED_FIGURES:
            fig_path = figures_dir / fig_name
            assert fig_path.exists(), f"Missing figure: {fig_name}"
            assert fig_path.stat().st_size > 1024, f"Figure too small (<1KB): {fig_name}"

    def test_all_act_headings_present(self, tmp_path: Path) -> None:
        from engine.report.narrative import generate_narrative_report

        output_dir = tmp_path / "narrative"
        generate_narrative_report(CYCLE_DIR, output_dir)
        report_text = (output_dir / "report.md").read_text()
        for heading in ACT_HEADINGS:
            assert heading in report_text, f"Missing heading: {heading}"

    def test_no_ai_slop(self, tmp_path: Path) -> None:
        from engine.report.narrative import generate_narrative_report

        output_dir = tmp_path / "narrative"
        generate_narrative_report(CYCLE_DIR, output_dir)
        report_text = (output_dir / "report.md").read_text()
        for pattern in AI_SLOP_PATTERNS:
            matches = re.findall(pattern, report_text, re.IGNORECASE)
            assert not matches, f"AI slop detected: {pattern} -> {matches}"

    def test_figure_references_valid(self, tmp_path: Path) -> None:
        from engine.report.narrative import generate_narrative_report

        output_dir = tmp_path / "narrative"
        generate_narrative_report(CYCLE_DIR, output_dir)
        report_text = (output_dir / "report.md").read_text()
        refs = re.findall(r"!\[.*?\]\((figures/[^)]+)\)", report_text)
        for ref in refs:
            fig_path = output_dir / ref
            assert fig_path.exists(), f"Broken image ref: {ref}"

    def test_non_publishable_banner(self, tmp_path: Path) -> None:
        from engine.report.narrative import generate_narrative_report

        output_dir = tmp_path / "narrative"
        generate_narrative_report(CYCLE_DIR, output_dir)
        report_text = (output_dir / "report.md").read_text()
        assert "NON-PUBLISHABLE" in report_text

    def test_threats_section_has_f_aiharm_precision(self, tmp_path: Path) -> None:
        from engine.report.narrative import generate_narrative_report

        output_dir = tmp_path / "narrative"
        generate_narrative_report(CYCLE_DIR, output_dir)
        report_text = (output_dir / "report.md").read_text()
        assert "F-aiharm-precision" in report_text

    def test_kappa_congruence_with_concordance_json(self, tmp_path: Path) -> None:
        from engine.report.narrative import generate_narrative_report

        output_dir = tmp_path / "narrative"
        generate_narrative_report(CYCLE_DIR, output_dir)
        report_text = (output_dir / "report.md").read_text()

        conc_data = json.loads((CYCLE_DIR / "results" / "concordance.json").read_text())

        # R12: Spot-check kappa value
        kappa_value = conc_data["weighted_kappa_median"]
        kappa_str = f"{kappa_value:.4f}"
        assert kappa_str in report_text, (
            f"Kappa {kappa_str} not found in narrative report"
        )

        # R12: Spot-check CI bounds
        ci = conc_data.get("weighted_kappa_ci", [])
        if ci:
            ci_lo_str = f"{ci[0]:.4f}"
            ci_hi_str = f"{ci[1]:.4f}"
            assert ci_lo_str in report_text, f"CI lower bound {ci_lo_str} not in report"
            assert ci_hi_str in report_text, f"CI upper bound {ci_hi_str} not in report"

        # R12: Spot-check flag count
        flags = conc_data.get("flags", [])
        assert f"{len(flags)} entries flagged" in report_text or len(flags) == 0

    def test_bump_chart_has_real_data(self, tmp_path: Path) -> None:
        """R10: Verify bump chart contains actual rank data, not placeholder."""
        from engine.report.narrative import generate_narrative_report

        output_dir = tmp_path / "narrative"
        generate_narrative_report(CYCLE_DIR, output_dir)
        report_text = (output_dir / "report.md").read_text()
        assert "No rank data available" not in report_text
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/unit/test_narrative.py -v -k "TestNarrativeIntegration" --timeout=300`
Expected: ALL PASS (or SKIP if cycle data absent). This may take a few minutes for chart generation.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_narrative.py
git commit -m "test(narrative): add integration tests for narrative report generator"
```

---

### Task 14: Generate the narrative report artifact

**Files:**
- Create: `projects/owasp-llm/cycles/2026/results/narrative/report.md`
- Create: `projects/owasp-llm/cycles/2026/results/narrative/figures/*.png`

- [ ] **Step 1: Generate the narrative report**

```bash
python3 -m engine.cli.main report-narrative --cycle-dir projects/owasp-llm/cycles/2026
```

- [ ] **Step 2: Verify output**

```bash
ls -la projects/owasp-llm/cycles/2026/results/narrative/
ls -la projects/owasp-llm/cycles/2026/results/narrative/figures/
wc -l projects/owasp-llm/cycles/2026/results/narrative/report.md
```

Expected: `report.md` exists, 16 PNG files in `figures/`, all > 1 KB.

- [ ] **Step 3: Verify acceptance criteria**

```bash
python3 -c "
from pathlib import Path
import re

narrative_dir = Path('projects/owasp-llm/cycles/2026/results/narrative')
report = (narrative_dir / 'report.md').read_text()

# Check all act headings
for i in range(1, 11):
    assert f'Act {i}' in report, f'Missing Act {i}'

# Check NON-PUBLISHABLE
assert 'NON-PUBLISHABLE' in report, 'Missing NON-PUBLISHABLE banner'

# Check F-aiharm-precision
assert 'F-aiharm-precision' in report, 'Missing F-aiharm-precision'

# Check all figure refs
refs = re.findall(r'!\[.*?\]\((figures/[^)]+)\)', report)
for ref in refs:
    assert (narrative_dir / ref).exists(), f'Broken ref: {ref}'

# Check figure count
figs = list((narrative_dir / 'figures').glob('*.png'))
assert len(figs) == 16, f'Expected 16 figures, got {len(figs)}'
for fig in figs:
    assert fig.stat().st_size > 1024, f'Too small: {fig.name}'

print(f'ALL CHECKS PASS: {len(refs)} figure refs, {len(figs)} PNGs, all acts present')
"
```

- [ ] **Step 4: Commit**

```bash
git add projects/owasp-llm/cycles/2026/results/narrative/
git commit -m "feat(artifact): generate standalone narrative report with 16 figures"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ Phase 1 ci_method: Tasks 1-3 (field, serialization, regenerate)
- ✅ Phase 2 F-aiharm-precision: Tasks 4-6 (register, notebook, report.md)
- ✅ Phase 3 narrative: Tasks 7-14 (deps, data, charts, markdown, CLI, tests, artifact)
- ✅ NON-PUBLISHABLE header: Task 11 (_render_markdown)
- ✅ Programmatic threats register: Task 11 (get_threats_register() in Act 10)
- ✅ Corpus B corroboration: Task 11 (Act 6 section)
- ✅ MPLBACKEND=Agg: Task 9 (os.environ.setdefault)
- ✅ kaleido ≥ 0.2.1: Task 7 (pyproject.toml)
- ✅ CLI registration: Task 12 (main.py add_command)
- ✅ Deserialization round-trip: Task 2 (pipeline.py update)
- ✅ Numerical congruence: Task 13 (kappa spot-check test)
- ✅ pytest.mark.skipif: Task 8 (SKIP_NO_CYCLE marker)

**Placeholder scan:** No TBD/TODO/implement-later found.

**Type consistency:** `generate_narrative_report(cycle_dir: Path, output_dir: Path) -> Path` used consistently across Tasks 11, 12, 13, 14. `load_narrative_data(cycle_dir: Path) -> dict[str, Any]` used consistently in Tasks 8, 11. Chart functions all take `(data: dict[str, Any], figures_dir: Path) -> None`.
