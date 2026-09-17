# Contributing

- Read [docs/ways-of-working.md](docs/ways-of-working.md): layers, commands, commit types and the PR standard.
- The PR title is a Conventional Commit; it becomes the squash commit and decides the release.
- Run `make ci` before pushing: it is exactly what CI runs.
- Never edit `CHANGELOG.md` or `[project].version`; semantic-release writes both ([docs/release.md](docs/release.md)).
- Report vulnerabilities as described in [SECURITY.md](SECURITY.md), not in a public issue.
