# Plan 8e Hardening — premortem Bucket A (harness patches) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. These are small, cohesive edits to two files; one implementer applies all four fixes with per-fix commits, then a single whole-range review.

**Goal:** Apply the four premortem "Bucket A" findings that are Phase-1 harness-code changes — F5 (prediction class-vocabulary guard), F3a (provenance completeness + `MIN_CELL` constant), F9 (goldset truth provenance), F8 (goldset↔corpus representativeness diagnostic), F7 (checkpoint hook) — all CI-green, no GPU.

**Why now:** The adversarial premortem of Phase 1 found these *in the merged harness*. F5 is a silent wrong-winner defect; F3a/F9/F8 make the eventual lock verifiable and the truth dependency auditable; F7 prevents catastrophic loss of a multi-hour Phase-3 sweep. F1 (winner→calibration wiring), F4 (RM14), and the lock/decision items (F2/F3b/F6/F10) are NOT in this plan — they are downstream (Phase 2) or need the lock / real numbers (Phase 3).

**Files:** `engine/classify/bakeoff.py`, `engine/cli/bakeoff.py`, `tests/unit/test_bakeoff_select.py`, `tests/unit/test_bakeoff_provenance.py`, `tests/unit/test_bakeoff_cli.py`.

## Global Constraints
- NO new dependencies (numpy/scipy/stdlib only). `engine/classify/bakeoff.py` MUST NOT import any `engine.*` module.
- mypy strict + ruff clean; use `int | float` union form, not `(int, float)`. `np`/`scipy` scalars cast with `float(...)`.
- Run before every commit: `uv run ruff check .` → `uv run mypy engine tests`. Before push: FULL `uv run pytest -q` (not a `-k` subset).
- No AI attribution in commits. Branch `plan7/engine-upgrade-recall-pl` (PR #22).

---

### Fix 1 (F5): Prediction class-vocabulary guard in `select_winner`

**File:** `engine/classify/bakeoff.py` ; **Test:** `tests/unit/test_bakeoff_select.py`

A config whose predictions use a class string outside the goldset vocabulary (e.g. `"LLM-01"` vs `"LLM01"`, a stray OOS token) is silently scored as all-misses → wrong winner or spurious `None`. Guard it.

- [ ] **Step 1: failing test** — append to `tests/unit/test_bakeoff_select.py`:
```python


def test_select_winner_raises_on_unknown_prediction_class() -> None:
    truth: dict[str, frozenset[str]] = {f"a{i}": frozenset({"A"}) for i in range(8)}
    truth.update({f"b{i}": frozenset({"B"}) for i in range(8)})
    lock = frozenset(truth)
    floor = {k: "A" for k in truth}
    bad = {k: ("A" if k.startswith("a") else "B-TYPO") for k in truth}
    import pytest

    with pytest.raises(ValueError, match="goldset vocabulary"):
        select_winner({"bad": bad}, floor, truth, lock)
```

- [ ] **Step 2: run** `uv run pytest tests/unit/test_bakeoff_select.py::test_select_winner_raises_on_unknown_prediction_class -v` → FAIL (no ValueError).

- [ ] **Step 3: implement** — in `select_winner`, AFTER the loop that builds `config_lb`/`config_ba` and `floor_lb`, and BEFORE the per-(config,class) p-value computation, insert:
```python
    # F5: predictions must use the goldset's class vocabulary, else they are
    # silently scored as all-misses (a wrong-winner footgun).
    allowed_classes = set(truth_cell_sizes(truth))
    floor_unknown = {c for c in floor_lb.values() if c not in allowed_classes}
    if floor_unknown:
        raise ValueError(
            f"floor predicts classes absent from the goldset vocabulary: "
            f"{sorted(floor_unknown)}"
        )
    for _name, _lb in config_lb.items():
        unknown = {c for c in _lb.values() if c not in allowed_classes}
        if unknown:
            raise ValueError(
                f"config {_name!r} predicts classes absent from the goldset "
                f"vocabulary: {sorted(unknown)}"
            )
```

- [ ] **Step 4: run** the test → PASS. Run `uv run pytest tests/unit/test_bakeoff_select.py -v` (all select tests still pass).

- [ ] **Step 5: gate + commit**
```bash
uv run ruff check . && uv run mypy engine tests
git add engine/classify/bakeoff.py tests/unit/test_bakeoff_select.py
git commit -m "fix(classify): guard bake-off predictions against unknown classes (Plan 8e F5)"
```

---

### Fix 2 (F3a + F9): `MIN_CELL` constant + goldset provenance block

**File:** `engine/classify/bakeoff.py` ; **Test:** `tests/unit/test_bakeoff_provenance.py`

Provenance omits the goldset hash, `min_cell`, and the truth's adjudication metadata — making the eventual lock unverifiable and the single-author truth dependency invisible. Add a `MIN_CELL` module constant (so all four tolerances are discoverable/lockable) and a goldset provenance block.

- [ ] **Step 1: failing tests** — append to `tests/unit/test_bakeoff_provenance.py`:
```python


def test_goldset_provenance_records_hash_and_disagreement(tmp_path: Path) -> None:
    from engine.classify.bakeoff import goldset_provenance

    rows = [
        {"incident_id": "i1", "llm_consensus": "A", "adjudicated": "accept",
         "labels": ["A"], "blind_label": "A", "notes": None},
        {"incident_id": "i2", "llm_consensus": "A", "adjudicated": "override",
         "labels": ["B"], "blind_label": "B", "notes": None},
    ]
    gp = tmp_path / "gold.jsonl"
    gp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    meta = goldset_provenance(gp)
    import hashlib
    assert meta["sha256"] == hashlib.sha256(gp.read_bytes()).hexdigest()
    assert meta["n_records"] == 2
    # i2 has blind_label "B" != llm_consensus "A" -> 1/2 disagreement
    assert meta["blind_consensus_disagreement_rate"] == 0.5
    assert meta["adjudicated_counts"] == {"accept": 1, "override": 1}
    assert meta["adjudicator"] == "single-author"


def test_provenance_records_min_cell_and_goldset_block(tmp_path: Path) -> None:
    label_file = tmp_path / "labeled_incidents.json"
    label_file.write_text("[]\n")
    out = write_bakeoff_provenance(
        tmp_path, _result(), [], label_file,
        seed=7, lockbox_fraction=0.3, min_cell=5,
        goldset_meta={"sha256": "abc", "n_records": 2},
    )
    data = json.loads(out.read_text())
    assert data["min_cell"] == 5
    assert data["goldset"]["sha256"] == "abc"
```

- [ ] **Step 2: run** `uv run pytest tests/unit/test_bakeoff_provenance.py -v` → FAIL (import error / unexpected kwargs).

- [ ] **Step 3a: add the `MIN_CELL` constant** — beside `BAKEOFF_ALPHA` in `engine/classify/bakeoff.py`:
```python
MIN_CELL: int = 5
```
Change `select_winner`'s signature default from `min_cell: int = 5` to `min_cell: int = MIN_CELL`.

- [ ] **Step 3b: add the `goldset_provenance` helper** — `engine/classify/bakeoff.py` (after `write_bakeoff_provenance`; uses already-imported `hashlib`, `json`, `Path`):
```python
def goldset_provenance(goldset_path: Path) -> dict[str, object]:
    """Audit metadata for the goldset truth file (Plan 8e F3a/F9).

    Records the content hash, record count, the blind-label vs llm-consensus
    disagreement rate (a single-author truth-uncertainty signal), and the
    adjudication breakdown.  Makes the truth the winner is selected against
    auditable and bindable in the lock.
    """
    raw = goldset_path.read_bytes()
    n = 0
    blind_disagree = 0
    adjudicated: dict[str, int] = {}
    for line in raw.decode().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        n += 1
        if rec.get("blind_label") != rec.get("llm_consensus"):
            blind_disagree += 1
        adj = str(rec.get("adjudicated", ""))
        adjudicated[adj] = adjudicated.get(adj, 0) + 1
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "n_records": n,
        "blind_consensus_disagreement_rate": (blind_disagree / n) if n else 0.0,
        "adjudicated_counts": adjudicated,
        "adjudicator": "single-author",
    }
```

- [ ] **Step 3c: extend `write_bakeoff_provenance`** — add two params and two payload keys. Change the signature:
```python
def write_bakeoff_provenance(
    out_dir: Path,
    result: BakeoffResult,
    model_configs: Iterable[ModelConfig],
    label_file: Path,
    seed: int | None = None,
    lockbox_fraction: float | None = None,
    min_cell: int | None = None,
    goldset_meta: dict[str, object] | None = None,
) -> Path:
```
In the payload, add immediately after `"lockbox_fraction": lockbox_fraction,`:
```python
        "min_cell": min_cell,
        "goldset": goldset_meta,
```

- [ ] **Step 4: run** `uv run pytest tests/unit/test_bakeoff_provenance.py -v` → PASS (existing tests still pass; the no-kwargs call defaults the new fields to None).

- [ ] **Step 5: gate + commit**
```bash
uv run ruff check . && uv run mypy engine tests
git add engine/classify/bakeoff.py tests/unit/test_bakeoff_provenance.py
git commit -m "feat(classify): record goldset hash + min_cell + truth provenance (Plan 8e F3a/F9)"
```

---

### Fix 3 (F8): goldset↔corpus representativeness diagnostic

**File:** `engine/classify/bakeoff.py` ; **Test:** `tests/unit/test_bakeoff_provenance.py`

The winner is selected on a 1,200-record goldset but applied to ~7,700 incidents; if the class mixes diverge, the winner may underperform on the bulk corpus. Add a total-variation diagnostic (the corpus distribution is a run-time input; recorded when provided).

- [ ] **Step 1: failing test** — append to `tests/unit/test_bakeoff_provenance.py`:
```python


def test_goldset_corpus_divergence() -> None:
    from engine.classify.bakeoff import goldset_corpus_divergence

    truth = {f"a{i}": frozenset({"A"}) for i in range(5)}
    truth.update({f"b{i}": frozenset({"B"}) for i in range(5)})
    # identical mix (50/50) -> 0
    assert goldset_corpus_divergence(truth, {"A": 50, "B": 50}) == 0.0
    # disjoint -> 1.0
    assert goldset_corpus_divergence(truth, {"C": 100}) == 1.0
    # partial skew in (0,1)
    d = goldset_corpus_divergence(truth, {"A": 90, "B": 10})
    assert 0.0 < d < 1.0
```

- [ ] **Step 2: run** → FAIL (import error).

- [ ] **Step 3: implement** — add to `engine/classify/bakeoff.py`:
```python
def goldset_corpus_divergence(
    goldset_truth: Mapping[str, frozenset[str]],
    corpus_class_counts: Mapping[str, int],
) -> float:
    """Total-variation distance between the goldset and corpus class mixes.

    0 = identical mix, 1 = disjoint.  A high value means the goldset the winner
    is SELECTED on is not representative of the corpus it is APPLIED to (F8).
    """
    gold_counts = truth_cell_sizes(goldset_truth)
    gold_total = sum(gold_counts.values())
    corpus_total = sum(corpus_class_counts.values())
    if gold_total == 0 or corpus_total == 0:
        return 0.0
    classes = set(gold_counts) | set(corpus_class_counts)
    tv = 0.0
    for c in classes:
        p = gold_counts.get(c, 0) / gold_total
        q = corpus_class_counts.get(c, 0) / corpus_total
        tv += abs(p - q)
    return 0.5 * tv
```

- [ ] **Step 4: run** the test → PASS.

- [ ] **Step 5: gate + commit**
```bash
uv run ruff check . && uv run mypy engine tests
git add engine/classify/bakeoff.py tests/unit/test_bakeoff_provenance.py
git commit -m "feat(classify): goldset-vs-corpus representativeness diagnostic (Plan 8e F8)"
```

---

### Fix 4 (F7) + wiring: checkpoint hook + thread provenance/divergence through `run_bakeoff`

**File:** `engine/cli/bakeoff.py` ; **Test:** `tests/unit/test_bakeoff_cli.py`

Wire the new provenance (Fix 2), the optional corpus diagnostic (Fix 3), `min_cell`, and a per-config checkpoint cache (so a mid-sweep failure resumes instead of losing the whole run) into `run_bakeoff`.

- [ ] **Step 1: failing test** — append to `tests/unit/test_bakeoff_cli.py` (the file already has `_write_goldset`, `run_bakeoff`, `ModelConfig`, `import json`, `import pytest` if present — add `import pytest` if missing):
```python


def test_run_bakeoff_checkpoint_resumes(tmp_path: Path) -> None:
    goldset = _write_goldset(tmp_path)
    all_ids = [f"a{i}" for i in range(12)] + [f"b{i}" for i in range(12)]
    floor = {k: "A" for k in all_ids}
    label_file = tmp_path / "labeled_incidents.json"
    label_file.write_text("[]\n")
    ckpt = tmp_path / "ckpt"

    def perfect(name: str) -> dict[str, str]:
        return {k: ("A" if k.startswith("a") else "B") for k in all_ids}

    r1 = run_bakeoff(
        goldset_path=goldset, config_names=["perfect"], predict_fn=perfect,
        floor_predictions=floor, model_configs=[], out_dir=tmp_path / "o1",
        label_file=label_file, seed=7, checkpoint_dir=ckpt,
    )
    assert (ckpt / "perfect.json").exists()

    def boom(name: str) -> dict[str, str]:
        raise RuntimeError("predict_fn must not be called on a cache hit")

    r2 = run_bakeoff(
        goldset_path=goldset, config_names=["perfect"], predict_fn=boom,
        floor_predictions=floor, model_configs=[], out_dir=tmp_path / "o2",
        label_file=label_file, seed=7, checkpoint_dir=ckpt,
    )
    assert r2.winner == r1.winner == "perfect"


def test_run_bakeoff_records_goldset_provenance(tmp_path: Path) -> None:
    goldset = _write_goldset(tmp_path)
    all_ids = [f"a{i}" for i in range(12)] + [f"b{i}" for i in range(12)]
    floor = {k: "A" for k in all_ids}
    label_file = tmp_path / "labeled_incidents.json"
    label_file.write_text("[]\n")

    def perfect(name: str) -> dict[str, str]:
        return {k: ("A" if k.startswith("a") else "B") for k in all_ids}

    run_bakeoff(
        goldset_path=goldset, config_names=["perfect"], predict_fn=perfect,
        floor_predictions=floor, model_configs=[], out_dir=tmp_path / "out",
        label_file=label_file, seed=7,
        corpus_class_counts={"A": 100, "B": 50},
    )
    prov = json.loads((tmp_path / "out" / "classify_provenance.json").read_text())
    assert prov["goldset"]["sha256"]
    assert prov["min_cell"] == 5
    assert "corpus_tv_divergence" in prov["goldset"]
```

- [ ] **Step 2: run** `uv run pytest tests/unit/test_bakeoff_cli.py -v` → FAIL (unexpected kwargs).

- [ ] **Step 3: implement** — in `engine/cli/bakeoff.py`:
  - Add `import json` to the imports if not present.
  - Add to the `from engine.classify.bakeoff import (...)` block: `MIN_CELL`, `goldset_corpus_divergence`, `goldset_provenance`.
  - Change the `run_bakeoff` signature to add the new params (after `alpha`):
```python
    min_cell: int = MIN_CELL,
    checkpoint_dir: Path | None = None,
    corpus_class_counts: Mapping[str, int] | None = None,
```
  - Replace the body. The current body is:
```python
    truth = load_bakeoff_truth(goldset_path)
    _dev, lockbox = lockbox_split(truth, lockbox_fraction=lockbox_fraction, seed=seed)
    config_predictions = {name: predict_fn(name) for name in config_names}
    # Coverage guard: every lockbox incident must have a prediction, else the
    # metric denominator silently shrinks (a Phase-3 footgun).
    missing_floor = lockbox - set(floor_predictions)
    if missing_floor:
        raise ValueError(
            f"floor_predictions missing {len(missing_floor)} lockbox incidents"
        )
    for name, preds in config_predictions.items():
        missing = lockbox - set(preds)
        if missing:
            raise ValueError(
                f"config {name!r} missing {len(missing)} lockbox incidents"
            )
    result = select_winner(
        config_predictions, floor_predictions, truth, lockbox, alpha=alpha
    )
    write_bakeoff_provenance(
        out_dir,
        result,
        model_configs,
        label_file,
        seed=seed,
        lockbox_fraction=lockbox_fraction,
    )
    return result
```
  Replace it with:
```python
    truth = load_bakeoff_truth(goldset_path)
    _dev, lockbox = lockbox_split(truth, lockbox_fraction=lockbox_fraction, seed=seed)

    # F7: per-config checkpoint cache so a mid-sweep failure resumes instead of
    # discarding a multi-hour grid run.
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_predictions: dict[str, dict[str, str]] = {}
    for name in config_names:
        ckpt = checkpoint_dir / f"{name}.json" if checkpoint_dir is not None else None
        if ckpt is not None and ckpt.exists():
            config_predictions[name] = json.loads(ckpt.read_text())
            continue
        preds = dict(predict_fn(name))
        config_predictions[name] = preds
        if ckpt is not None:
            ckpt.write_text(json.dumps(preds, sort_keys=True))

    # Coverage guard: every lockbox incident must have a prediction, else the
    # metric denominator silently shrinks (a Phase-3 footgun).
    missing_floor = lockbox - set(floor_predictions)
    if missing_floor:
        raise ValueError(
            f"floor_predictions missing {len(missing_floor)} lockbox incidents"
        )
    for name, preds in config_predictions.items():
        missing = lockbox - set(preds)
        if missing:
            raise ValueError(
                f"config {name!r} missing {len(missing)} lockbox incidents"
            )

    result = select_winner(
        config_predictions, floor_predictions, truth, lockbox,
        alpha=alpha, min_cell=min_cell,
    )

    goldset_meta = goldset_provenance(goldset_path)
    if corpus_class_counts is not None:
        goldset_meta["corpus_tv_divergence"] = goldset_corpus_divergence(
            truth, corpus_class_counts
        )
    write_bakeoff_provenance(
        out_dir,
        result,
        model_configs,
        label_file,
        seed=seed,
        lockbox_fraction=lockbox_fraction,
        min_cell=min_cell,
        goldset_meta=goldset_meta,
    )
    return result
```

- [ ] **Step 4: run** `uv run pytest tests/unit/test_bakeoff_cli.py -v` → PASS, then the FULL suite `uv run pytest -q` → PASS.

- [ ] **Step 5: gate + commit**
```bash
uv run ruff check . && uv run mypy engine tests
git add engine/cli/bakeoff.py tests/unit/test_bakeoff_cli.py
git commit -m "feat(cli): checkpoint hook + thread goldset provenance/divergence into run_bakeoff (Plan 8e F7)"
```

---

## Self-review
Spec coverage: F5 (Fix 1), F3a (Fix 2 — MIN_CELL + goldset hash + min_cell in provenance), F9 (Fix 2 — disagreement + adjudicator), F8 (Fix 3 + wired in Fix 4), F7 (Fix 4 checkpoint). Out of scope (documented): F1 (downstream calibration wiring), F4 (RM14), F2/F3b/F6/F10 (lock/decisions/Phase-3). No placeholders; full code each step. Type consistency: `goldset_meta`/`min_cell` params match between `write_bakeoff_provenance` (Fix 2) and `run_bakeoff` (Fix 4); `goldset_provenance`/`goldset_corpus_divergence`/`MIN_CELL` defined in `bakeoff.py` and imported in `cli/bakeoff.py`.
