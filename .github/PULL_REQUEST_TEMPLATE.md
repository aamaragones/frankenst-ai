### Overview

Why this change exists and what it involves. If it changes the public API under
`src/frankstate`, say so here: the commit type below decides the version bump.

### Changes

-

### Checklist

- [ ] PR title follows Conventional Commits (`<type>[scope][!]: <description>`).
      It becomes the squash commit and the release note; `feat`/`fix` publish to PyPI,
      `chore`/`build`/`ci` do not.
- [ ] `make ci` passes locally (the same targets, in the same order, that CI runs)
- [ ] New or changed behavior is covered by tests
- [ ] Docs updated where relevant (OKF v0.2 frontmatter on new pages; bump `generated.at`
      when a page is rewritten); `README-pypi.md` updated if the package surface changed
- [ ] `#` comment lines stay at or below 15% of code lines in new or changed files
      (`make comment-ratio`)
- [ ] No secrets, tokens, endpoints or real hostnames introduced
- [ ] `CHANGELOG.md` and `[project].version` untouched (semantic-release owns both)
- [ ] If produced with an AI assistant: reviewed line by line

### Related

- Issues / threads / docs:
