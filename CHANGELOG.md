# Changelog

All notable changes to this project will be documented in this file.

The format follows Keep a Changelog and the project currently stays in the `0.x`
phase while the public packaging and repository boundaries continue to mature.

## [0.1.3] - 2026-05-15

### Added

- `PromptMixin` and `RetrieverMixin` cooperative mixins for `RunnableBuilder`, enabling prompt-building and lazy retriever initialization as composable, opt-in capabilities.

### Changed

- `RunnableBuilder` contract is now stricter: `model` is keyword-only, `_configure_runnable` is the single required abstract hook, and `invoke` / `ainvoke` / `get` / `runnable` form the complete public surface.
- `RetrieverMixin` exposes a `retriever` lazy property and a `_build_retriever(**kwargs)` hook; it accepts either a pre-built `BaseRetriever` or a `VectorStore` and raises a descriptive `ValueError` when neither is provided.

## [0.1.2] - 2026-05-14

### Added

- `uv.lock`, `ruff`, and `mypy` are now part of the repository's standard development flow through `pyproject.toml` dependency groups and tool configuration.
- A centralized `CoreSettings` layer plus focused path/config tests now cover repository defaults and environment overrides for the `core_examples` reference package.
- Reusable Azure DevOps test/build templates were expanded for the mono-repo, including a Docker-based test runner flow that mirrors the repository validation sequence.

### Changed

- Repository setup, local development, packaging, and contributor docs now use `uv` as the primary workflow and document `.venv`-first usage in English across the main README, package README, and contributing guide.
- `WorkflowBuilder` now expects a `GraphLayout` subclass directly, and the reference layouts follow a two-step `build_runtime()` plus `layout()` contract so runtime dependencies are resolved before node and edge declaration.
- Repository configuration access now goes through `core_examples.config.settings`, replacing the old constants-based path handling for active source and test consumers.
- `LLMServices` and the Foundry/Azure AI integration layer now align on `azure_ai` naming, support `project_endpoint`-driven configuration, and resolve their shared config path through the centralized settings layer.
- Packaging and CI inputs are now centered in `pyproject.toml`, with static dependency metadata, `uv`-managed sync/build flows, and updated Docker/Azure pipeline templates for the mono-repo.

### Fixed

- Key Vault, blob storage, RAG helpers, and related Azure service integrations now handle current Foundry/runtime configuration patterns more consistently.
- Example layouts, output-path utilities, logging bootstrap, and LLM service tests now reflect the settings-based configuration flow and current repository boundaries more accurately.

## [0.1.1] - 2026-04-22

### Added

- Root `make` shortcuts for editable installs, package-scoped `frankstate` tests, packaging validation, the local MCP server and Azure Functions container workflows.
- Dedicated `frankstate` test modules for `StateEnhancer` / `StateEvaluator` / `StateCommander` and `RunnableBuilder`, plus purpose-specific test doubles modules for builder infrastructure and workflow layout fixtures.

### Changed

- Azure Functions MCP entrypoints now use the dedicated `mcp_tool_trigger(...)` decorator instead of the generic trigger surface.
- Repository docs now advertise the root development shortcuts and the distinction between the full repository suite and the package-scoped `frankstate` suite.
- `frankstate` test support is now organized by responsibility: builder doubles live separately from workflow layout fixtures, and the old mixed `fake.py` / `spy.py` structure is removed.
- Repository development requirements now include `build` and `twine` so the packaging validation flow can run from the local dev environment.

### Fixed

- Package-scoped `frankstate` tests now reflect the package boundary more cleanly by separating state-handler and runnable-builder concerns.

## [0.1.0] - 2026-04-18

### Added

- A package-focused release narrative for `frankstate`, including clearer public API boundaries and a dedicated published-package README.

### Changed

- The `frankstate` root API remains intentionally small, while reusable contracts continue to live in their concrete submodules instead of shortcut barrels.
- Core runtime dependency metadata now uses bounded compatibility ranges for `langchain-core`, `langgraph`, and `pydantic`.
- Structured prompt resources in `src/core_examples` now use Markdown `.md` files and Markdown sectioning instead of legacy `.txt` templates with XML-like tags.
- `display_graph()` and other package-facing docs now describe optional behavior and repository/package boundaries more explicitly.

### Fixed

- Internal and public documentation now consistently distinguish the published `frankstate` package from the broader `Frankenst-AI` repository.
- Runnable/retriever examples and related tests now align with the simplified retriever contract and current prompt resource layout.

## [0.0.5] - 2026-04-17

### Added

- Environment-driven logging controls through `LOG_LEVEL` and `LOG_TO_FILE`, plus a dedicated `config_logging.yml` template for repository defaults.

### Changed

- Repository logging now uses `configure_logging()` plus `logging.config.dictConfig()` instead of embedding logging settings in `config.yml`.
- The local MCP server and notebook examples now target `fastmcp` over HTTP, expose both Oak and adaptive RAG tools from the same entrypoint, and make header/auth testing an explicit part of the research demo flow.
- Module loggers across `frankstate`, example runnables and tools now use full module names via `logging.getLogger(__name__)` for clearer traces and filtering.
- Release automation now uses the current major versions of the GitHub Actions involved in build, artifact upload/download and Python setup.

## [0.0.4] - 2026-04-12

### Changed

- Standardized the source tree on Python 3.12 typing syntax with built-in generics and `|` unions, including the remaining tool properties, runnables, retrievers, indexers and `frankstate` managers.
- Kept `frankstate` visibly aligned with LangGraph by removing wrapper-specific edge and node type aliases and using direct inline unions instead.
- `RunnableBuilder` and the Oak reference runnables now consistently use chat-model typing across the runnable assembly path.
- Azure AI Search schema loading now maps index names directly to schema names through the `AI_SEARCH_INDEX_SCHEMA_MAP` registry.

### Fixed

- `LLMServices` now exposes explicit nullable runtime contracts while preserving the existing `launch()` plus class-attribute access pattern used by the layouts and tools.
- Tool property schemas and runnable signatures now use precise modern annotations, removing several nullable and `args_schema` inconsistencies.
- Key Vault and Azure AI Search indexing helpers now raise more specific configuration/runtime errors instead of generic exceptions.
- Project metadata now points the repository URL back to the repository root.

## [0.0.3] - 2026-04-12

### Added

- Optional node-level `kwargs` passthrough for native `StateGraph.add_node()` options.

### Changed

- `StateCommander` and `CommandNode` now use `destinations` as the public routing contract.
- `WorkflowBuilder` and `NodeManager` now mirror LangGraph's native node registration shape with positional args plus kwargs.
- `GraphLayout` now returns concrete node and edge unions to improve type safety during workflow assembly.
- Node `tags` are normalized into LangGraph `metadata["tags"]` during workflow assembly.
- The human-in-the-loop example and node registry now use `destinations` consistently.

## [0.0.2] - 2026-04-07

### Added

- Root Dockerfile that boots the Streamlit demo with `uv`-managed installs.
- OSS baseline files: `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`.
- Unified `examples` extra that aggregates the Azure and Ollama example dependencies.

### Changed

- The public distribution, package directory and import surface now use `frankstate`.
- `pyproject.toml` now uses SPDX-style `license = "MIT"` metadata.
- Audit updated to remove findings already resolved in the repository root.
- Installation docs now use the standard unquoted extras form `.[examples,dev]`.
- Release automation now targets setuptools-based builds instead of Poetry.
- Minor fix: requirements reference now points from `requirements-frankstate-base.txt` to `requirements-frankstate-core.txt`.