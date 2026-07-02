# arXiv Preprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Produce a beautiful, publication-quality arXiv preprint (PDF + LaTeX source) telling the full OWASP-2026-LLM-Top-10 incident-data robustness story, built notebook-first, with every statistic computed from committed data.

**Architecture:** The notebook (`notebooks/2026_top_10_llm_update_what_the_data_says.ipynb`) is the single authored source. New charts + the 0.75/0.25 blend + the 2025→2026 changelog live in tested engine modules the notebook imports. Executing the notebook saves all figures at 300 dpi and runs a consistency check. The PDF is built: execute → `jupyter nbconvert --to markdown --no-input` → prepend calibrated YAML front-matter → `pandoc --template=arxiv-template.latex --pdf-engine=xelatex`. `engine/report/narrative.py` (the internal auto-report) is **left untouched** — the preprint is a separate, notebook-driven artifact.

**Tech Stack:** Python 3.13 (conda-base-py kernel), numpy/pandas/matplotlib/seaborn/plotly/kaleido/scipy, jupyter+nbconvert, pandoc 3.9 + xelatex (TeXLive 2026), pytest/ruff/mypy.

## Global Constraints
- **Compute, never transcribe:** every number in the preprint is computed in-notebook from committed artifacts (`cycles/2026/`, `cycles/2026-rarr/results/`, `baselines/2026/`, `taxonomy.json`). A consistency-check cell fails loudly on drift. (spec §5)
- **Calibrated framing (spec §3):** title = *"Incident-Data Robustness Analysis of the OWASP Top 10 for LLM Applications (2026): How a Community-Expert Ranking Holds Up Against a Large-Scale LLM Incident Corpus."* No "validating"/"first-ever". Thesis = weak agreement (κ≈0.20) **but robust ranking**. Consistent across abstract/Part I/conclusion.
- **Scope statement (spec §2):** front-matter states this is NOT the official OWASP release and does not supersede it.
- **Limitations, not banner (spec §4 / decision a):** replace `NON-PUBLISHABLE` with a candid "Limitations & Independent-Review Status" section; frame findings exploratory.
- **Entry names from `taxonomy.json`** (repo canonical names, e.g. LLM07 = "Hidden Context Exposure") — never hand-typed.
- **Novice-first (spec §1, §8):** every DS concept gets a boxed sidebar on first use + a glossary entry; ~20–30 pp target; findings in the spine, pedagogy in boxes.
- **Human voice, no AI attribution** anywhere (prose, commits, notebook). Avoid AI-slop patterns.
- Gates before every push: `uv run pytest -q`, `uv run ruff check .`, `uv run mypy engine tests`.
- Work on branch `docs/arxiv-preprint`; never write into immutable `cycles/2026/`.

---

### Task 1: Robustness-validation JSON (Act-11 numbers, reproducible)

**Files:** Create `tools/compute_robustness_validation.py`; Test `tests/unit/test_robustness_validation.py`; Output `projects/owasp-llm/cycles/2026-rarr/results/robustness_validation.json`; also `git add` the untracked `cycles/2026-rarr/classify/seq/predictions_*.json` + `gate_*.json` (repro inputs).

**Interfaces — Produces:** `compute_robustness_validation(cycle_dir: Path, floor_path: Path) -> dict` with schema:
`{"goldset_n", "ranked_classes", "bakeoff_balanced_accuracy": {config: float}, "ranking_fidelity_spearman_vs_truth": {name: float}, "ranking_delta_vs_floor_bootstrap": {name: {"mean": float, "ci95": [lo,hi]}}, "corpus_reweight_spearman_vs_truth": {"floor": float, "best_frontier": str, "best_frontier_rho": float, "delta_ci95": [lo,hi]}, "recall_correction_negL2": {"floor_raw","ensemble_raw","floor_cvcorrected","ensemble_cvcorrected","delta_ensemble_minus_floor_ci95"}}`

- [ ] **Step 1: failing test** — `tests/unit/test_robustness_validation.py`:
```python
from pathlib import Path
from tools.compute_robustness_validation import compute_robustness_validation
CYCLE = Path("projects/owasp-llm/cycles/2026-rarr")
FLOOR = Path("projects/owasp-llm/cycles/2026/classify/labeled_incidents.json")

def test_floor_ranks_high_no_frontier_beats_it():
    r = compute_robustness_validation(CYCLE, FLOOR)
    assert r["ranking_fidelity_spearman_vs_truth"]["floor"] > 0.85
    for m, d in r["ranking_delta_vs_floor_bootstrap"].items():
        lo, hi = d["ci95"]; assert lo <= 0 <= hi

def test_recall_correction_closes_gap():
    rc = compute_robustness_validation(CYCLE, FLOOR)["recall_correction_negL2"]
    lo, hi = rc["delta_ensemble_minus_floor_ci95"]; assert lo <= 0 <= hi
```
- [ ] **Step 2: run → FAIL** `uv run pytest tests/unit/test_robustness_validation.py -v` (ModuleNotFoundError).
- [ ] **Step 3: implement** — port the exact RARR-session computations (load truth via `engine.classify.bakeoff.load_bakeoff_truth`; floor via `bakeoff_inputs.load_floor_predictions`; 4 `predictions_*.json`; 4-vote ensemble tie→deepseek). Compute: incidence-ranking Spearman ρ vs truth (primary class = `sorted(truth_set)[0]`, exclude `"out-of-scope"`) for floor+models+ensemble; paired bootstrap `default_rng(42)` B=3000 of ρ(cand)−ρ(floor); corpus-reweighted ρ (post-stratify goldset by floor class to corpus mix via `compute_corpus_class_counts`); 5-fold CV `default_rng(11)` recall/precision correction neg-L2 (corrected=obs·prec/recall, recall floored 0.05); `bakeoff_balanced_accuracy` read from `results/bakeoff_seq/bakeoff_crosscheck.json`. `argparse` main (`--cycle --floor-path --out`).
- [ ] **Step 4: run → PASS.**
- [ ] **Step 5: generate + commit**
```bash
python tools/compute_robustness_validation.py --cycle projects/owasp-llm/cycles/2026-rarr --floor-path projects/owasp-llm/cycles/2026/classify/labeled_incidents.json --out projects/owasp-llm/cycles/2026-rarr/results/robustness_validation.json
uv run ruff check tools/compute_robustness_validation.py tests/unit/test_robustness_validation.py && uv run mypy engine tests
git add tools/compute_robustness_validation.py tests/unit/test_robustness_validation.py projects/owasp-llm/cycles/2026-rarr/results/robustness_validation.json projects/owasp-llm/cycles/2026-rarr/classify/seq/predictions_*.json projects/owasp-llm/cycles/2026-rarr/classify/seq/gate_*.json
git commit -m "feat(rarr): reproducible robustness_validation.json + committed prediction inputs"
```

---

### Task 2: Blend + 2025→2026 changelog compute module

**Files:** Create `engine/report/blend_2025_2026.py`; Test `tests/unit/test_blend_2025_2026.py`.

**Interfaces — Produces:**
- `load_entries(taxonomy_path: Path) -> list[dict]` → each `{entry_id, canonical_name, group}` where group ∈ {"incumbent","new","rollup"} from `is_incumbent`/`is_rollup_candidate`, plus `rolled_into` for rollups.
- `blended_ranking(vote_ranks: dict[str,int], lambda_ranks: dict[str,int], w_vote: float = 0.75) -> list[dict]` → for the 10 incumbents, `{entry_id, vote_rank, lambda_rank, blend, blend_rank}` with `blend = w_vote*vote_rank + (1-w_vote)*lambda_rank`, sorted ascending; `blend_rank` 1..10.
- `rank_moves(published_order: list[str], blended: list[dict]) -> dict[str,int]` → per entry `published_pos - blend_pos` (positive = moved up).

- [ ] **Step 1: failing test**:
```python
from engine.report.blend_2025_2026 import blended_ranking, load_entries, rank_moves
from pathlib import Path
TAX = Path("projects/owasp-llm/cycles/2026/taxonomy/taxonomy.json")

def test_groups_10_6_4():
    g = {}
    for e in load_entries(TAX): g[e["group"]] = g.get(e["group"], 0) + 1
    assert g == {"incumbent": 10, "new": 6, "rollup": 4}

def test_blend_formula_llm01_folded():
    # LLM01 vote_rank 1, folded lambda_rank 9 -> blend 3.00 (methodology doc worked example)
    out = {r["entry_id"]: r for r in blended_ranking({"LLM01":1,"LLM02":2}, {"LLM01":9,"LLM02":2})}
    assert abs(out["LLM01"]["blend"] - 3.00) < 1e-9
    assert abs(out["LLM02"]["blend"] - 2.00) < 1e-9

def test_rank_moves_sign():
    blended = [{"entry_id":"A","blend_rank":1},{"entry_id":"B","blend_rank":2}]
    assert rank_moves(["B","A"], blended) == {"B": -1, "A": 1}
```
- [ ] **Step 2: run → FAIL.**
- [ ] **Step 3: implement** the three pure functions (no I/O beyond `load_entries` reading the JSON). Handle ties deterministically (sort key = (blend, entry_id)).
- [ ] **Step 4: run → PASS.**
- [ ] **Step 5: commit** `git commit -m "feat(report): blend + 2025->2026 changelog compute module"`.

---

### Task 3: New charts + 300-dpi print resolution

**Files:** Modify `engine/report/narrative_charts.py`; Test `tests/unit/test_narrative_charts_new.py`.

**Interfaces — Produces** (signatures mirror existing chart fns; each saves a PNG and returns the Path):
- `render_rank_change_2025_2026(blended: list[dict], entry_names: dict[str,str], figures_dir: Path) -> Path` → slope/bump chart of published→blended positions, biggest movers highlighted; save `rank_change_2025_2026.png`.
- `render_entry_expansion_map(entries: list[dict], figures_dir: Path) -> Path` → grouped diagram: 10 incumbents, 6 NEW-*, 4 ROLL-* with arrows to parents; save `entry_expansion_map.png`.
- `render_rarr_robustness(robustness: dict, figures_dir: Path) -> Path` → bar of `ranking_fidelity_spearman_vs_truth` (floor + models + ensemble) with a 0.918 reference line; save `rarr_robustness.png`.

- [ ] **Step 1: failing test** — assert each fn creates a >1KB PNG at the expected path from minimal fixtures.
- [ ] **Step 2: run → FAIL.**
- [ ] **Step 3: implement** — follow existing idioms: matplotlib `fig.savefig(path, dpi=300, bbox_inches="tight")` + `plt.close(fig)`, `tight_layout()`, `ENTRY_COLORS.get(id,"#999999")`. **Also change existing matplotlib saves `dpi=150`→`dpi=300` and plotly `width/height` `1000/600`→`2000/1200`** across the file (print resolution).
- [ ] **Step 4: run → PASS**; also `uv run pytest tests/unit/test_narrative.py -q` (ensure the dpi bump didn't break existing chart tests — they check size>1KB, unaffected).
- [ ] **Step 5: commit** `git commit -m "feat(charts): 3 preprint charts + 300-dpi print resolution"`.

---

### Task 4: Custom arXiv LaTeX template + stub compile (de-risk the build FIRST)

**Files:** Create `notebooks/preprint/arxiv-template.latex`; Create `notebooks/preprint/_stub.md`; Test `tests/unit/test_preprint_build.py` (stub-compile test).

**Interfaces — Produces:** a pandoc LaTeX template consuming YAML `author: [{name, affiliation}]`, `title`, `abstract`, `date`; `\documentclass[11pt]{article}`, `geometry margin=1in`, `fancyhdr` running header (short title + "OWASP GenAI Security Project"), `hyperref` colorlinks, numbered sections, TOC; author block rendered without `authblk` (custom `\author{...\\ \small affiliation}` loop `$for(author)$…$endfor$`).

- [ ] **Step 1: write the template** — start from `pandoc -D latex > notebooks/preprint/arxiv-template.latex`, then edit: replace the author block with the affiliation-aware loop; add `\usepackage{fancyhdr}` + header; keep `$body$`, `$toc$`, `$if(abstract)$` blocks.
- [ ] **Step 2: write `_stub.md`** — YAML front-matter with the two authors + a title + a 2-line abstract, one `## Section`, one paragraph, and one figure `![Test](../narrative/figures/bump_chart.png){width=85%}`.
- [ ] **Step 3: failing test** `tests/unit/test_preprint_build.py`:
```python
import shutil, subprocess
from pathlib import Path
import pytest
PRE = Path("notebooks/preprint")

@pytest.mark.skipif(shutil.which("pandoc") is None or shutil.which("xelatex") is None, reason="toolchain")
def test_stub_compiles_with_template(tmp_path):
    out = tmp_path / "stub.pdf"
    r = subprocess.run(["pandoc", str(PRE/"_stub.md"), "-o", str(out),
        "--pdf-engine=xelatex", f"--template={PRE/'arxiv-template.latex'}",
        "--from=markdown+yaml_metadata_block", "--toc"],
        capture_output=True, text=True, cwd=PRE)
    assert r.returncode == 0, r.stderr[-2000:]
    assert out.stat().st_size > 10 * 1024
```
- [ ] **Step 4: run → PASS** (fix the template until the stub renders: author block, figure width, header, TOC). *This validates the beautiful-PDF path before any content is poured in (spec §6 stub-first).*
- [ ] **Step 5: commit** `git commit -m "feat(preprint): custom arXiv LaTeX template + stub-compile test"`.

---

### Task 5: Notebook→PDF build pipeline

**Files:** Create `tools/build_preprint.py`; Modify `notebooks/requirements.txt` (add `nbconvert>=7`, `jupyter>=1`); Test: extend `tests/unit/test_preprint_build.py` (build-script unit test on a tiny fixture notebook).

**Interfaces — Produces:** `build_preprint(notebook: Path, out_dir: Path, front_matter_md: Path, template: Path) -> Path` →
1. `jupyter nbconvert --to notebook --execute --inplace <notebook>` (regenerates figures + validates cells);
2. `jupyter nbconvert --to markdown --no-input <notebook> --output-dir <out_dir>` (prose + image refs, code hidden);
3. prepend `front_matter_md` (the YAML block) to the exported `.md`;
4. `pandoc <md> -o <pdf> --pdf-engine=xelatex --template=<template> --from=markdown+yaml_metadata_block --toc --toc-depth=2`;
returns the PDF path. CLI `--notebook --out-dir`.

- [ ] **Step 1: add deps** to `notebooks/requirements.txt`; `uv pip install nbconvert jupyter` (or the project's env manager).
- [ ] **Step 2: failing test** — build from a 3-cell fixture notebook (one markdown cell + one code cell that `savefig`s a tiny PNG + a raw YAML cell) → assert a >10KB PDF is produced.
- [ ] **Step 3: implement** `build_preprint` (subprocess calls, fixed argv, `cwd` = notebook dir so `figures/` resolves).
- [ ] **Step 4: run → PASS.**
- [ ] **Step 5: commit** `git commit -m "feat(preprint): notebook->markdown->pandoc build pipeline"`.

---

### Tasks 6–10: Notebook content (prose + figure-producing code cells)

> These author the preprint's content. Code cells **save** figures (300 dpi) to `notebooks/narrative/figures/` and call `plt.close()` (no inline display, so `--no-input` export is clean). Prose cells are markdown with `![caption](figures/x.png){width=85%}` refs. Every number is pulled from `DATA`/committed files via the Task 1–2 modules. Content requirements are specified; the implementer writes the prose to spec in the project's human voice.

### Task 6: Front-matter + "How to Read" + Scope statement + reframed abstract
**Files:** Modify notebook (insert cells at index 1, before Act 1); Create `notebooks/preprint/front_matter.md` (the YAML block: title per Global Constraints, `author: [{name:"Kyriakos \"Rock\" Lambros", affiliation:"OWASP GenAI Security Project — Top 10 for LLM Applications, Co-Lead"}, {name:"Steve Wilson", affiliation:"…Founder & Co-Lead"}]`, date, and the calibrated abstract).
- [ ] Insert a **scope/authority** markdown cell (spec §2: not the official release; stress-tests, doesn't set the list; exploratory).
- [ ] Insert a **"How to Read This Report"** cell (audience note; pointer to glossary; that boxed sidebars are optional depth).
- [ ] Write the **calibrated abstract** into `front_matter.md` (weak-but-robust thesis; no "validating"/"first"; κ≈0.20 stated honestly; robustness headline).
- [ ] **Verify:** `grep -c "official OWASP" front_matter.md`/scope cell; abstract contains "robust" and does NOT contain "validat" or "first-ever".
- [ ] Commit `git commit -m "feat(preprint): front-matter, scope statement, how-to-read, calibrated abstract"`.

### Task 7: Part I — The List and How It's Made
**Files:** Modify notebook (cells after index 1). Uses Task 2 module + `DATA`.
- [ ] Code cell: load `taxonomy.json` via `load_entries`; compute vote/λ ranks + `blended_ranking` + `rank_moves`; **save** `rank_change_2025_2026.png` and `entry_expansion_map.png` (Task 3 fns).
- [ ] Prose: the incident corpus (7,714; CVE/GHSA/OSV/AIAAIC + Corpus B; **"large-scale"** framing, no "first-ever" unless narrowed+defensible per spec §3); the **0.75/0.25 blend** explained for novices (why expert-led, data as quarter-weight corrective — source `BLENDED-TOP10-METHODOLOGY.md`); the **2025→2026 changes** (10 incumbents / 6 NEW-* / 4 ROLL-* with the expansion map; biggest movers from `rank_moves`).
- [ ] Sidebars: "What a ranking blend is", "What incidence means". (Glossary entries added in Task 9.)
- [ ] **Verify:** the movers rendered match `rank_moves` output (consistency cell, Task 10 enforces globally).
- [ ] Commit `git commit -m "feat(preprint): Part I — corpus, 0.75/0.25 blend, 2025->2026 changes"`.

### Task 8: Reframe Acts 1–10 as Part II + Act 11 (RARR)
**Files:** Modify notebook (existing Act cells 1–38; add Act 11 after cell 38).
- [ ] Light reframing of Acts 1–10 into "Part II — What the Incident Data Says": section-numbered, consistent with the calibrated thesis (no "validated"); keep the existing analysis/figures (ensure their code cells `savefig` at 300 dpi + `plt.close()`).
- [ ] Add **Act 11 (Part III — Robustness)** cells: code cell loads `robustness_validation.json` + saves `rarr_robustness.png` (Task 3); prose covers the bake-off (winner=None table from `bakeoff_balanced_accuracy`), ground-truth validation (ρ, bootstrap Δ crosses 0), recall-correction — framed as "the ranking holds up."
- [ ] **Verify:** Act 11 prose contains the floor ρ (`ranking_fidelity_spearman_vs_truth["floor"]`) and "winner=None"/"no frontier" from the JSON, not literals.
- [ ] Commit `git commit -m "feat(preprint): Part II reframe + Act 11 RARR robustness"`.

### Task 9: Sidebars + Glossary appendix
**Files:** Modify notebook (add glossary appendix cell near end; ensure each concept has a first-use sidebar).
- [ ] Add a **Glossary** appendix defining: precision, recall, gold set, prior/posterior, credible interval, MCMC, Cohen's κ, balanced accuracy, bootstrap, Spearman ρ, out-of-scope, negative-binomial measurement-error model, λ (latent incidence). Plain-language, one entry each.
- [ ] Ensure every listed concept has a boxed first-use sidebar in Parts I–III (audit against the glossary list).
- [ ] Commit `git commit -m "feat(preprint): novice sidebars + glossary appendix"`.

### Task 10: Limitations & Review Status + consistency-check cell + reproducibility/appendices
**Files:** Modify notebook (near end).
- [ ] **Limitations & Independent-Review Status** cell (spec §4/decision a): single-author goldset (κ, override rate from `goldset_provenance`), interim reviewers, corpus caveats (stratum imbalance 6,297 security / 342 ai-harm; OOS blind spot), frame-blind entries (LLM04/08/10), "exploratory pending independent adjudication", and the pre-publish checklist summary — candid, replaces `NON-PUBLISHABLE`.
- [ ] **Consistency-check code cell** (Global Constraint / spec §5) — fails the notebook if any displayed headline number drifts from source:
```python
import json, numpy as np
from pathlib import Path
# blend movers match the module vs the methodology doc's committed baseline
from engine.report.blend_2025_2026 import blended_ranking, load_entries, rank_moves
xc = json.loads((CYCLE.parent/"2026-rarr"/"results"/"robustness_validation.json").read_text())
assert xc["ranking_fidelity_spearman_vs_truth"]["floor"] > 0.85, "floor rho drift"
base = json.loads(Path("projects/owasp-llm/baselines/2026/rankings_baselines.json").read_text())
assert abs(base["previous_ranking"]["kappa_median"] - 0.2028985507246377) < 1e-9, "kappa drift"
print("consistency OK")
```
- [ ] **Reproducibility appendix** cell: exact `python tools/build_preprint.py` command, data provenance (content hashes), toolchain versions (pandoc/xelatex), "recompute by re-executing this notebook."
- [ ] Commit `git commit -m "feat(preprint): limitations, consistency-check, reproducibility appendix"`.

---

### Task 11: Execute end-to-end + build PDF + acceptance verification
**Files:** produces `notebooks/preprint/<stem>.md` + `<stem>.pdf` + refreshed `notebooks/narrative/figures/*.png`.
- [ ] **Step 1:** `python tools/build_preprint.py --notebook notebooks/2026_top_10_llm_update_what_the_data_says.ipynb --out-dir notebooks/preprint` (executes, exports, builds).
- [ ] **Step 2: acceptance checks (spec §9):** notebook executed no-error; PDF >10KB and compiles; **visually inspect** (open PDF) — figures sized to width/no overflow, two-author affiliation block correct, TOC/section numbers aligned, all figures present at 300 dpi; abstract/conclusion free of "validat"/"first-ever" (`! grep -iE "validat|first-ever" <md>` on title/abstract region); scope statement + limitations + glossary present; consistency-check cell printed "consistency OK".
- [ ] **Step 3:** AI-slop scan (reuse `tests/unit/test_narrative.py`'s slop patterns against the exported md); human-voice check.
- [ ] **Step 4: commit** the built artifacts + refreshed figures `git commit -m "docs(preprint): build executed PDF + LaTeX source + 300-dpi figures"`.

### Task 12: Full-suite gate + PR
- [ ] `uv run ruff check .` && `uv run mypy engine tests` && `uv run pytest -q` → all green.
- [ ] `git push -u origin docs/arxiv-preprint`; `gh pr create --title "docs: OWASP 2026 LLM Top-10 incident-data robustness preprint (arXiv)" --body "<summary + acceptance evidence>"`.
- [ ] Note in the PR body: **Steve Wilson reviews/approves the final text before any arXiv submission** (owner-managed, out-of-band). arXiv submission = LaTeX source (.tex from pandoc) + `figures/` — not PDF-only.

---

## Self-Review
**1. Spec coverage:** §1 novice (T9); §2 scope (T6); §3 calibrated title/abstract/thesis (T6,T8, verified T11); §4 limitations-not-banner (T10); §5 compute-not-transcribe + consistency cell (T1,T2,T10); §6 notebook-first build + stub-first (T4 before content, T5, T11); §7 new figures (T3, produced T7/T8); §8 sidebars/glossary/length (T9); §9 acceptance (T11); §10 non-goals respected (narrative.py untouched; blend not engine-automated beyond the report module; not official release). All covered.
**2. Placeholder scan:** prose tasks (T6–T10) specify content requirements + inputs, not literal 25-page prose (correct right-sizing for a deliverable's prose — the implementer writes to spec in-voice); all CODE tasks (T1–T5, T10 cell) carry complete code. No TBDs.
**3. Type consistency:** `load_entries`→`{entry_id,canonical_name,group,rolled_into}` (T2) consumed by `render_entry_expansion_map(entries,…)` (T3, T7). `blended_ranking(...)→[{entry_id,vote_rank,lambda_rank,blend,blend_rank}]` (T2) consumed by `render_rank_change_2025_2026(blended,…)` (T3) + `rank_moves(published_order, blended)` (T2, T7). `robustness_validation.json` schema (T1) consumed by `render_rarr_robustness(robustness,…)` (T3) + Act 11 (T8) + consistency cell (T10). `build_preprint(notebook,out_dir,front_matter_md,template)` (T5) consumes `arxiv-template.latex` (T4) + `front_matter.md` (T6). Consistent.

**Sequencing note:** T4 (stub-compile) runs before content tasks to de-risk the build (premortem #9). T1–T3 (data+charts) precede the notebook cells that call them (T6–T8).
