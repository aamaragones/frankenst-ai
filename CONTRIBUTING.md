# Contributing

## Scope

This repository contains three different layers:

- `src/frankstate`: reusable public package surface.
- `src/core_examples`: reference package showing how to consume `frankstate`. It may reuse `src/services/foundry` as a repository adapter for shared runtime bootstrap, but should avoid depending on other service entrypoints.
- `src/services`: runtime integrations and deployment-facing adapters.
- `research`: exploratory material that can inform future work but is not part of the repository's contractual surface.

Keep those boundaries explicit in pull requests.

## Local setup

Use `uv` as the single dependency and environment manager for the repository.
From the repository root:

```bash
uv venv .venv
source .venv/bin/activate
uv sync --frozen
uv sync --frozen --extra examples --group dev
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

If you prefer repository shortcuts, `make install-dev` wraps the same editable
sync flow from the repository root.

## Tests

Run the test suite from the repository root:

```bash
uv run pytest -q
```

Repository shortcuts:

- `make test` runs the full mono-repo suite.
- `make test-frankstate` validates only the installable `frankstate` package surface.

Use that split when you want to distinguish package changes from mono-repo-only changes.

If you touch packaging, also validate the distributions:

```bash
uv build
```

`make build` runs the packaging build and `twine check` validation flow with `uv`.

## Dependency policy

- `src/frankstate` is a published library, so its runtime dependencies should use compatible version ranges, not exact `==` pins.
- The runtime floor should reflect versions exercised by the repository test suite.
- The runtime ceiling should stay conservative around fast-moving dependencies such as LangGraph and LangChain so `frankstate` does not claim untested compatibility.
- Exact pins live in `dependency-groups` plus `uv.lock` for repository development and CI environments, not in the published core wheel metadata.

Current core policy:

- `langchain-core>=1.3.0,<1.4`
- `langgraph>=1.1.8,<1.2`
- `pydantic>=2.12.5,<3`

## Versioning policy

- `0.1.x` is for backward-compatible fixes, documentation updates, packaging adjustments, test improvements and compatibility maintenance that do not require user code changes.
- `0.2.0` should be used when the public `frankstate` contract changes in a meaningful way: import paths, required conventions, signatures, behavior of core abstractions or supported dependency families.
- Before any minor bump such as `0.2.0`, document the user-facing change in `CHANGELOG.md` and update package-facing documentation when the recommended usage changes.

## Pull requests

- Keep changes focused and avoid unrelated refactors.
- Add or update tests when modifying reusable contracts in `src/frankstate`.
- Update documentation when changing public behavior, install steps or repository structure.
- Preserve the distinction between public API, examples and service integrations.

## Documentation

- Prefer adding documentation close to the contract it explains: docstrings in `src/frankstate`, comments in YAML and examples in layout classes.
- When a component reads or writes new state keys, document that change in the state schema and in the component docstring.
- Keep project abstractions aligned with official LangGraph terminology to avoid confusion in new layouts.

## Issue reports

When opening a bug report, include:

- Python version
- installation method used
- minimal reproduction
- expected behavior
- actual behavior

For changes that affect providers such as Ollama, Azure AI or Foundry, include the relevant runtime configuration shape without exposing secrets.