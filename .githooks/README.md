# Git hooks

Versioned hooks for this repository. Activate them once per clone:

```sh
git config core.hooksPath .githooks
```

## pre-push

Runs `gitleaks` against the commits being pushed and blocks the push if a
secret is found. Requires `gitleaks` on `PATH` (`brew install gitleaks`).

This is the client-side secret-scanning control (QC.1 / NIST SP 800-218).
GitHub-native secret-scanning push protection is unavailable on this private
personal repository without a paid plan; this hook is the standing substitute.
It is bypassable with `git push --no-verify` by design — that path is an
accepted, logged risk, not a defect.
