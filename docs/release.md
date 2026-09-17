---
type: Runbook
title: Release
description: How a merge to main becomes a PyPI version through semantic-release, the one-time PyPI and GitHub settings behind it, the first automated release, and recovery.
tags: [release, semantic-release, pypi, github-actions, runbook]
generated:
    by: reference_agent
    at: 2026-09-17T00:00:00Z
---

# Release

Nobody edits the version. A push to `main` runs `.github/workflows/release.yaml`, and
semantic-release decides from the commits since the last tag whether a version exists,
which one, and what its notes say. The commit types and what they cut are in
[ways-of-working.md](ways-of-working.md).

## What one run does

```
push main ──▶ quality (ci.yaml: frankstate matrix, quality, security)
          ──▶ release  (environment: release)
                npx semantic-release
                  1. analyse commits since tag X.Y.Z; stop here if nothing releases
                  2. write the notes on top of CHANGELOG.md
                  3. make release-prepare VERSION=A.B.C
                       uv version A.B.C --no-sync      pyproject.toml + uv.lock
                       make build                      dist/*.whl, dist/*.tar.gz, twine check
                  4. commit "chore(release): A.B.C [skip ci]" (CHANGELOG, pyproject, uv.lock), tag A.B.C
                  5. GitHub Release A.B.C with the two dists attached
                pypa/gh-action-pypi-publish   only if dist/ exists, i.e. a release happened
```

Tags are bare `X.Y.Z`, never `vX.Y.Z`, because the existing tags are. The publish step
runs in the same job as semantic-release on purpose: GitHub starts no workflow for a
push made with `GITHUB_TOKEN`, so a workflow triggered by the new tag would never fire.
The old tag-triggered release had exactly that shape and would have published nothing.

Version equality is by construction: `uv version` writes what semantic-release tags.
`uv.lock` carries its own `frankstate` version entry, which is why it is re-locked and
committed with the release.

## One-time setup

| Where | Setting | Why |
| --- | --- | --- |
| PyPI, project `frankstate` | Trusted publisher: owner `aamaragones`, repo `frankenst-ai`, workflow `release.yaml`, environment `release` | OIDC instead of a token. Renaming the workflow file or the environment breaks publishing silently |
| GitHub, Environments | `release` exists; no required reviewers | A reviewer would block every automatic release |
| GitHub, General | Squash merge only; squash title = **pull request title** | The default takes a single commit's message, which nobody validated |
| GitHub, Branches | Protect `main` with **required status checks only**: `ci / frankstate (3.12)`, `(3.13)`, `(3.14)`, `ci / quality`, `ci / security`, `pr-title / Validate title` | "Require a pull request" or "restrict pushes" rejects the release commit. If either is ever wanted, mint a GitHub App token in the `release` environment and add the app to the bypass list |
| Local | `brew install node` | For the dry run below; nothing in the Python toolchain needs it |

The job names above are the check names; renaming a job orphans its rule.

## First release under this scheme

The last hand-made tag is `0.2.2`, and **it never reached PyPI**: its run
(2026-08-20) failed at "Publish to PyPI" with `invalid-publisher`. `0.2.0` and `0.2.1`
published through `release.yml`; the file was renamed to `release.yaml` on 2026-07-18
and the trusted publisher on PyPI still names `release.yml`, so the claim GitHub sends
(`workflow: release.yaml`) matches nothing. PyPI stops at `0.2.1` and no GitHub Release
`0.2.2` exists. The fix is the publisher row above, on pypi.org; nothing in the repo can be.

The commits above `0.2.2` do not parse as Conventional Commits, so semantic-release
ignores them. Two PRs, in this order:

1. On PyPI, edit the trusted publisher's workflow name from `release.yml` to
   `release.yaml` (table above).
2. `fix(deps): require langchain-core >=1.6,<1.7`, a two-file PR against the old
   workflow. The old `release.yaml` fires only on tags, so merging it cuts nothing; the
   commit waits in range.
3. Switch the repository to squash-only merges with the PR title as the commit title.
4. The adoption PR: `ci: adopt semantic-release, make-driven CI and OKF docs | chore(examples): ...`.
   It carries the new gates and the code they gate, which is why it cannot be split.
   On merge, `release.yaml` runs on `main` for the first time: the analysis finds the
   `fix(deps)` commit, cuts `0.2.3`, publishes, and the `ci:`/`chore` commit is hidden
   from the notes.
5. `gh workflow run release.yaml -f republish_tag=0.2.2` publishes the version that
   missed, so PyPI and the tags agree.
6. Protect `main` with the required checks, picking their names from the run that just
   completed rather than typing them.

`CHANGELOG.md` keeps the hand-written Keep a Changelog history below the generated
entries; a marker line records where the format changed. It is never edited again.

## Dry run locally

```bash
npm install --no-audit --no-fund
GITHUB_TOKEN=$(gh auth token) npx semantic-release --dry-run --no-ci \
    --branches "$(git branch --show-current)"
```

Prints the next version and its notes; writes nothing. `--branches` is needed because
a dry run only analyses a configured release branch, `--no-ci` because the shell is not
a CI environment.

## Recovery

| Symptom | Cause | Action |
| --- | --- | --- |
| Tag and GitHub Release exist, PyPI does not have the version | The upload failed after the tag was pushed; a rerun finds no new commits and releases nothing | `gh workflow run release.yaml -f republish_tag=A.B.C` rebuilds from the tag and uploads |
| `invalid-publisher` at "Publish to PyPI" | PyPI has no trusted publisher whose four claims match: owner, repo, workflow **filename**, environment. The log prints the claims GitHub sent | Create or correct the publisher on pypi.org to those exact values; then `republish_tag` for the version that missed |
| Merged, no release | The squash title was not a releasing type | Intentional for `chore`/`build`/`ci`; otherwise the next PR carries the right type |
| Two merges in a row | Runs queue on the `release-<repo>` concurrency group, never cancel | None; each analyses from the latest tag |
| A `feat!` on 0.x cut a minor | `{"breaking": true, "release": "minor"}` in `.releaserc` | Delete that rule when the API is declared stable |
