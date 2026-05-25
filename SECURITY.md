# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please disclose it
responsibly by emailing:

**rock@rockcyber.com**

Please include:

- A description of the issue
- Steps to reproduce
- Any relevant proof-of-concept
- Whether the issue affects methodology integrity (pre-registration, inference,
  or report outputs) or only tooling

You should receive an acknowledgement within 5 business days and a resolution
timeline within 15 business days.

Do not open a public GitHub issue for security vulnerabilities.

## Scope

The following areas are in scope for security reports:

- **Stage-2 LLM prompt injection** -- incident text that escapes the
  delimiter-fenced prompt and alters classification output
- **Pre-registration tampering** -- any path that allows modifying the
  hash-locked manifest, rubric, or hyperparameters after the lock step
- **Supply-chain integrity** -- vendored corpus snapshots, dependency
  vulnerabilities in the inference or classification path, SBOM accuracy
- **Credential exposure** -- API keys for RunPod or other services leaking
  through logs, error messages, or committed artifacts
- **Inference correctness** -- inputs that cause the NUTS sampler to produce
  silently wrong posteriors (not just diagnostics failures, which are gated)

Out of scope: the synthetic test projects (`projects/synthetic/`,
`projects/synthetic-stress/`) contain fabricated data by design and are not
subject to data-integrity reports.

## Safe Harbor

Good-faith security research against this repository is welcome. If you act in
good faith and follow this policy, the maintainer will not pursue legal action
and will credit you in the fix (unless you prefer to remain anonymous).

## Supported Versions

Security fixes are applied to the latest commit on the `main` branch only.
There are no maintained release branches.

| Version | Supported |
|---|---|
| Latest on `main` | Yes |
| Tagged releases | Fixes backported on a case-by-case basis |
| Pre-v1.0.0 tags | No |

## Known Supply-Chain Residuals

See `pyproject.toml` comments for tracked transitive CVEs that are blocked on
upstream dependency upgrades (cyclonedx-bom, semgrep). These are monitored and
will be resolved when upstream releases permit.
