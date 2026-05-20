---
name: secret-scanning-posture
description: Why this repo uses a gitleaks pre-push hook instead of GitHub-native secret scanning
metadata:
  node_type: memory
  type: project
  originSessionId: b8a2bb68-f524-48d7-b5f9-542b3a5013f2
---

GitHub-native secret scanning + push protection is **unavailable** on `rocklambros/incident-rank-validation`: it is a private repo on a personal (User) account, which requires the paid GitHub Secret Protection product (HTTP 422 "Secret scanning is not available for this repository" on the API enable attempt 2026-05-19).

Standing substitute: a versioned client-side hook at `.githooks/pre-push` running gitleaks over the pushed commit range. Activation per clone requires `git config core.hooksPath .githooks` (documented in `.githooks/README.md`).

**Why:** native push protection is plan-gated; making the repo public (which would unlock free scanning) is forbidden by the spec's "private now, public-ready" posture; a paid upgrade is an operator/billing decision.

**How to apply:** the user explicitly chose client-side-only and **declined** both the server-side gitleaks CI gate and the paid GitHub Secret Protection upgrade. Do not add a CI secret-scanning workflow or propose the paid plan again unless the user reopens it. The hook is bypassable with `git push --no-verify` by design (accepted, logged risk). gitleaks 8.30 default rules apply entropy/allowlist filtering, so low-entropy or example-shaped tokens (e.g. `AKIAIOSFODNN7EXAMPLE`) are intentionally not flagged; if stricter coverage is wanted when code lands, add a committed `.gitleaks.toml`. Revisit if the repo goes public per [[finishing-a-development-branch]].
