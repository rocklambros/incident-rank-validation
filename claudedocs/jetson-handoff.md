# Jetson handoff — incident-rank-validation

Continuity bundle for moving `incident-rank-validation` development from the Mac harness to the Jetson device.

---

## 1. State at handoff (2026-05-20)

- **HANDOFF.md** is at **v2.4** — the approved methodology spec. Four adversarial premortems have run against it.
- **Plan 1 v5** is in `docs/superpowers/plans/2026-05-19-engine-synthetic-cycle.md`. **30 tasks. No engine code has been written yet.**
- **REVIEWERS.md**: Rock is sole reviewer (interim, single-author non-publishable per HANDOFF §4). PRE-PUBLISH CHECKLIST documented.
- **Memory files** committed under `claudedocs/memory/`. Install instructions below.
- **Premortem 3 closure (M1–M23)** inlined into Plan 1 v5.

Files committed in the repo right now:
```
.githooks/{pre-push, README.md}
docs/HANDOFF.md                                      (v2.4)
docs/REVIEWERS.md
docs/REVIEWERS/.gitkeep
docs/superpowers/plans/2026-05-19-engine-synthetic-cycle.md   (v5)
claudedocs/memory/{MEMORY.md, *.md}                  (5 memory files)
claudedocs/jetson-handoff.md                         (this file)
```

Files NOT yet existing (created by Plan 1 execution):
- `pyproject.toml`, `uv.lock`, `engine/`, `tests/`, `projects/`, `.github/`, `.gitleaks.toml`, `.semgrep.yml`, `NOTICE`, `SECURITY.md`, `README.md`, `docs/METHODOLOGY-CHANGELOG.md`, `docs/RUNBOOK.md`, `docs/BOUNDARY-CASES.md`, `docs/METHODOLOGY-FAQ.md`, `docs/SUCCESSOR-PRIMER.md`, `docs/PROVISIONING-PLAN.md`.

---

## 2. Setup on Jetson (one-time)

### 2.1 Prerequisites

- Python 3.12 (`pyenv` or distribution package)
- `uv` ≥ 0.5.11 — https://github.com/astral-sh/uv
- `git`
- `gitleaks` — `brew install gitleaks` on macOS; on Linux `go install github.com/gitleaks/gitleaks/v8@latest` or download a release binary
- `cosign` (optional, for verifying signed SBOM artifacts from CI) — https://docs.sigstore.dev/cosign/installation/
- Claude Code with the `superpowers` plugin available (this plan depends on `superpowers:subagent-driven-development`)

### 2.2 Clone and activate git hooks

```bash
git clone <repo-url> incident-rank-validation
cd incident-rank-validation
git config core.hooksPath .githooks
```

### 2.3 Install memory files into Claude Code auto-memory

The 5 memory files in `claudedocs/memory/` need to be installed into your local Claude Code auto-memory directory so they load when a session opens in this project. **Open Claude Code in the repo root at least once first** so the project directory under `~/.claude/projects/` gets created — then run the install:

```bash
# Find the auto-memory dir Claude Code created for this project
PROJECT_DIR=$(pwd)
SLUG=$(ls -d ~/.claude/projects/-* 2>/dev/null | grep "$(basename "$PROJECT_DIR")" | head -1)
if [ -z "$SLUG" ]; then
  echo "ERROR: Open Claude Code in this directory once first to create the project dir."
  exit 1
fi
MEMORY_DIR="$SLUG/memory"
mkdir -p "$MEMORY_DIR"
cp claudedocs/memory/*.md "$MEMORY_DIR/"
ls -la "$MEMORY_DIR"   # expect 5 .md files (MEMORY.md + 4 memory files)
```

If `~/.claude/projects/` doesn't show a directory for this project after opening Claude Code, the slug convention may have changed — check `~/.claude/projects/` for any directory containing `incident-rank-validation` in its name and copy the memory files there manually.

### 2.4 Verify

- `cat docs/HANDOFF.md | grep "^Version:"` → `Version: 2.4`
- `ls docs/superpowers/plans/` → `2026-05-19-engine-synthetic-cycle.md` exists
- `wc -l docs/superpowers/plans/2026-05-19-engine-synthetic-cycle.md` → roughly 1500-1900 lines (v5)
- `head -50 docs/REVIEWERS.md` → shows Rock as interim sole reviewer
- `ls .githooks/` → shows `pre-push` and `README.md`
- `git config core.hooksPath` → `.githooks`

---

## 3. Continuing development on Jetson

Open Claude Code in the repo root. **First prompt:**

> Read `docs/HANDOFF.md` (v2.4) and `docs/superpowers/plans/2026-05-19-engine-synthetic-cycle.md` (v5) end to end. These are the approved methodology spec and the engineering plan for this project. Then in ≤ 250 words, confirm:
> 1. What the project does (one paragraph).
> 2. The HANDOFF v2.4 information-firewall discipline (§6 control 11) — what's mechanism, what's discipline.
> 3. The Plan 1 v5 task count and the headline acceptance gate.
> 4. The current REVIEWERS.md state and what it gates.
>
> Also confirm that the memory files at the auto-memory path have loaded (correctness-over-speed, publication-formality-calibration, secret-scanning-posture, decision-style-recommendation-first). If they have not loaded, stop and ask me to fix the install before proceeding.
>
> Once everything is confirmed, launch subagent-driven execution for Plan 1 v5 via the `superpowers:subagent-driven-development` skill, starting with Task 0.

(If you want a paranoid first pass, ask Claude to re-read all 4 memory files explicitly and acknowledge each one before reading the spec.)

---

## 4. What "done" looks like for Plan 1

- All 30 Plan 1 v5 tasks completed (see acceptance criteria in the plan's Task 29).
- CI green on Linux + macOS matrix, including the M5 cross-platform output-diff job.
- `git tag v0.1.0-plan1` applied.
- `docs/PROVISIONING-PLAN.md` carries the default $500/cycle cost ceiling (overridable for Plan 5).
- `docs/RUNBOOK.md`, `BOUNDARY-CASES.md`, `METHODOLOGY-FAQ.md`, `SUCCESSOR-PRIMER.md` complete.
- All four memory files referenced from at least one section of the new docs (cross-linking).
- REVIEWERS.md PRE-PUBLISH CHECKLIST stays as the publication gate.

---

## 5. Plan 2-5 prerequisites (DO NOT start before Plan 1 acceptance)

- **REVIEWERS.md** populated with external reviewers (rubric + statistical, both ≠ Rock). Currently: Rock as both (interim).
- **`docs/PROVISIONING-PLAN.md`** filled in with RunPod commitments before any Stage-2 cycle. Default cost ceiling $500/cycle (M9).
- **M17 two-cycle parity**: any publishable cycle holds for 30 days for reviewer audit before external sharing.
- HANDOFF §9 open items (gold-set staffing, PPL pin, measurability minimum for LLM project, recency window, etc.).

---

## 6. Premortem history (for posterity)

Four adversarial premortems run before any code was written:

| Round | Closure | Plan delta |
|---|---|---|
| Premortem 0 (within spec) | HANDOFF v1.0 → v2.0 | (spec only) |
| Premortem 1 | R1–R25 | Plan v1 → v2 |
| Premortem 2 | R1–R33 + L1–L11 | Plan v2 → v3 → v4; HANDOFF v2.1 → v2.2 → v2.3 |
| Premortem 3 | M1–M23 | Plan v4 → v5; HANDOFF v2.3 → v2.4 |

Severity declined over rounds (Critical: 5 → 1 → 1 → 1). Premortem 3 noted convergence: future premortems likely surface polish-level findings. Recommendation: **execute Plan 1 v5 and find real bugs in real code, rather than running Premortem 4.**

---

## 7. Memory file contents (preview)

The 4 memory files that ship in `claudedocs/memory/` are auto-loaded by Claude Code when a session opens in this project. Brief descriptions so you know what's in them without opening each:

- **`correctness-over-speed.md`** — default to rigorous when comparing options; don't anchor on prior commitments. Triggered 2026-05-19 when I framed a Plan 1 rewrite as "patch vs rescope" — Rock rejected the framing and asked for whichever was most correct regardless of effort.
- **`publication-formality-calibration.md`** — mechanism for methodology integrity, discipline+disclosure for process formality. Includes a decision tree. Triggered 2026-05-20 when I proposed mechanical enforcement of reviewer-≠-ranking-author — Rock said "we aren't publishing this in some sort of academic journal" and relaxed the gap.
- **`secret-scanning-posture.md`** — gitleaks pre-push hook is the standing substitute for GitHub-native scanning (declined). Don't propose CI gate or paid GitHub Secret Protection again.
- **`decision-style-recommendation-first.md`** — when asking decision questions, use AskUserQuestion with the recommended option first, marked "(Recommended)", plus the reasoning and the strongest counterargument against the recommendation.

---

## 8. If something is unclear after handoff

Read in order:
1. `docs/HANDOFF.md` (v2.4) — methodology spec.
2. `docs/superpowers/plans/2026-05-19-engine-synthetic-cycle.md` (v5) — engineering plan.
3. `docs/REVIEWERS.md` — publication gate.
4. (Once Plan 1 runs:) `docs/SUCCESSOR-PRIMER.md`, `BOUNDARY-CASES.md`, `RUNBOOK.md`, `METHODOLOGY-FAQ.md`.

If you find a methodology gap that the premortems missed, that's a Plan 1 finding. Treat it as either a new R/L/M item (close mechanically) or a discipline-disclosure item (document in REVIEWERS or BOUNDARY-CASES) per the decision tree in `publication-formality-calibration.md`.

---

*Created 2026-05-20 on the Mac harness, in the same commit that finalized Plan 1 v5 and HANDOFF v2.4. Last operation before handoff was `git push origin main`.*
