---
type: Guide
title: Ways of Working
description: Layers and what may import what, the Makefile as the single interface, Conventional Commits and what each type releases, the PR standard, the OKF documentation rule and the comment-ratio rule.
tags: [ways-of-working, conventional-commits, pull-requests, ci, documentation]
generated:
    by: reference_agent
    at: 2026-09-17T00:00:00Z
---

# Ways of Working

## Layers

`src/frankstate/` is the published package and the only public API. `src/config/`,
`src/utils/`, `src/core_ai_examples/` and `src/services/` are the reference and
integration layers; they may change without a release. The dependency rule and the
one sanctioned exception are in [architecture.md](architecture.md), and a test
enforces them. A change in the reference layers goes in as `chore(<layer>)` unless it
should ship, because any releasing type publishes a wheel, and a wheel whose content did
not change is a version number with nothing behind it.

## The Makefile is the single interface

Every check has a `make` target and CI calls only those targets, so passing locally and
failing CI should not be possible. `make help` lists them.

```bash
make ci              # everything CI runs, in CI order: yaml-check lint format-check type
                     # comment-ratio cov cov-frankstate audit build pre-commit
make test-frankstate # the published slice only
make hooks           # install pre-commit; make pre-commit runs every hook over the tree
```

| Gate | What it prevents |
| --- | --- |
| `lint`, `format-check` | ruff over the whole tree; the examples used to be tab-indented because the formatter only saw `frankstate` |
| `type` | strict mypy over `src`, `tests/support`, `.github/scripts`, `main.py` |
| `cov-frankstate` | the wheel's suite under 90% on Python 3.12, 3.13 and 3.14; this is the release gate |
| `cov` | the whole tree above an explicit floor, ratcheted upward; a lower number beats hiding the untested Functions code with `omit` |
| `comment-ratio` | `#` comment lines above 15% of a file's non-blank lines (docstrings do not count); `src/frankstate` is reported, not failed |
| `yaml-check` | any `.yml`: the house extension is `.yaml` |
| `audit` | `pip-audit` over the locked runtime deps; an accepted advisory is listed in the Makefile with its reason |
| `pre-commit` | hygiene hooks over everything, not only the published slice |

The one step that is not a make target is `gitleaks` in the `security` job: a Go binary
scanning git history, not a project tool.

## Conventional Commits

The PR title becomes the squash commit, and semantic-release reads that commit. A title
that does not parse merges green and cuts nothing, so `pr-title` validates it.

| Type | Release | Use for |
| --- | --- | --- |
| `feat` | minor | a new capability in `src/frankstate` |
| `fix`, `perf` | patch | a defect or a dependency range a consumer resolves |
| `docs`, `refactor`, `test`, `style` | patch | anything that changes the wheel without changing behaviour |
| `<type>!` or `BREAKING CHANGE:` | minor while 0.x | a contract change; becomes major when the API is declared stable |
| `chore`, `build`, `ci` | none | tooling, reference layers, pins |

Scopes are free-form; prefer the layer names (`frankstate`, `config`, `utils`,
`examples`, `services`, `deps`). A compound title `feat: a | fix: b` is accepted for a
PR that carries two themes; the analyzer takes the highest type.

## Pull requests

- One theme per PR; split a refactor from a behaviour change.
- Fill the template (`.github/PULL_REQUEST_TEMPLATE.md`). If the PR touches
  `src/frankstate`, say so in the overview: the type you chose is the version bump.
- `make ci` green before pushing.
- Never edit `CHANGELOG.md`, `[project].version` or the `frankstate` entry in
  `uv.lock`: semantic-release writes all three and anything added by hand sinks to the
  bottom. To change what the changelog says, change the commit message.
- Squash-merge into `main`; there is no other long-lived branch. The release mechanics
  are in [release.md](release.md).

## Comments

Comment lines (`#`) stay at or below 15% of a file's non-blank lines; `make comment-ratio`
measures it. Docstrings are not counted, but they are one sentence that states the claim
and its consequence. A comment that is the only record of a real failure stays.

## Documentation

Markdown under `docs/`, flat, with `README.md` as the index; there is no site to build.
Every page carries [OKF v0.2](https://cloud.google.com/blog/products/data-analytics/okf-v0-2-adds-trust-signals)
frontmatter:

| Field | | |
| --- | --- | --- |
| `type` | required | `Concept`, `Guide`, `Runbook` or `Reference` |
| `title`, `description`, `tags` | descriptive | |
| `generated: {by, at}` | who produced the content, and when | `reference_agent/<model>` or `human:<id>` |
| `verified: [{by, at}]` | optional | absent means unverified; only a person adds their own entry |

Bump `generated.at` when a page is rewritten. Never write a `verified` entry on someone
else's behalf: it is the one field that means a human read the page. One topic, one
page; `README.md` links here instead of restating. `README-pypi.md` is the
package's own long description and stays self-contained because PyPI resolves no
relative link.

All code, comments and docs in English.
