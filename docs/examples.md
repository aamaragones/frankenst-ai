---
type: Guide
title: Reference examples
description: The four GraphLayouts under src/config/graph_layout, how to render and run them locally with Ollama or Azure, the logging variables, and the Azure Functions container.
tags: [examples, layouts, ollama, azure, langgraph, logging]
generated:
    by: reference_agent
    at: 2026-09-17T00:00:00Z
---

# Reference examples

Everything outside `src/frankstate/` is one concrete way to consume the package. It is
not installed by `pip install frankstate`; clone the repository for it. The layer roles
are in [architecture.md](architecture.md); the model configuration in
[llm-services.md](llm-services.md).

## Setup

```bash
uv sync --frozen --extra examples --group dev   # or: make install-dev
cp .env.example .env                            # only the variables your provider needs
```

Pick a provider in `src/config/config_llms.yaml` under `launch:`. The shipped file
points at Ollama, so the examples run with no account:

```bash
ollama pull gemma4:e4b-it-qat && ollama pull embeddinggemma
```

The model names come from the yaml, not from this page: when they diverge the yaml is
right. Document processing for the RAG layouts also needs `poppler-utils` and
`tesseract-ocr` from the system package manager.

## The layouts

| `--layout` | Class | Pattern | Needs |
| --- | --- | --- | --- |
| `simple_oak` | `SimpleOakConfigGraph` | agent + tool loop | `model` |
| `oak_human_loop` | `OakHumanLoopConfigGraph` | agent + human review before a sensitive tool | `model` |
| `local_vectorstore_rag` | `LocalVectorstoreAdaptiveRAGConfigGraph` | adaptive RAG over a local Chroma store | `model`, `embeddings` |
| `ai_search_rag` | `AISearchAdaptiveRAGConfigGraph` | adaptive RAG over Azure AI Search | `model`, `embeddings`, `AZURE_SEARCH_*` |

Every layout follows the two-phase contract: `build_runtime()` asks
`LLMServices.launch().require(...)` for its clients and builds the runnable builders;
`layout()` declares nodes and edges, reading node names from `config_nodes.yaml` through
`load_node_registry()`. A typo in that yaml fails on load, not inside `compile()`.

## Render a layout

`main.py` compiles a layout and prints its Mermaid diagram, which is the fastest way to
see that a topology is what you meant:

```bash
python main.py --layout simple_oak
python main.py --layout oak_human_loop --with-metadata
```

Constructing the Ollama client does not contact the server, so this works offline.
The rendered diagrams under `artifacts/layouts/` were produced this way.

## Compile and invoke from code

```python
from frankstate import WorkflowBuilder
from config.graph_layout.simple_oak_config_graph import SimpleOakConfigGraph
from core_ai_examples.models.stategraph.stategraph import SharedState

graph = WorkflowBuilder(config=SimpleOakConfigGraph, state_schema=SharedState).compile()
graph.invoke({"messages": [("user", "Which Pokémon evolves from Charmander?")]})
```

`graph` is a native LangGraph `CompiledStateGraph`. The notebooks under `research/` walk
through each layout interactively; they are exploratory and not part of the suite.

## Configuration and secrets

| Variable | Read by | Purpose |
| --- | --- | --- |
| `FRANK_CORE_PACKAGE_PATH` | `settings` | Overrides the `src/` root the yaml paths hang off |
| `AZURE_KEY_VAULT_NAME` | `settings` | Enables the Key Vault fallback for every `{secret: NAME}` and every flagged settings field |
| `AZURE_FOUNDRY_PROJECT_ENDPOINT`, `AZURE_INFERENCE_MODEL_NAME`, `AZURE_EMBEDDINGS_ENDPOINT`, `AZURE_EMBEDDINGS_MODEL_NAME` | `config_llms.yaml` | The `azure_ai` provider |
| `DATABRICKS_LLM_ENDPOINT`, `DATABRICKS_EMBEDDINGS_ENDPOINT` | `config_llms.yaml` | The `databricks` provider |
| `AZURE_SEARCH_SERVICE_ENDPOINT`, `AZURE_SEARCH_API_KEY` | `settings.azure`, the retriever tool | Azure AI Search for `ai_search_rag` and the Functions app |
| `AZURE_BLOB_STORAGE_NAME` | `utils.blob_storage` | Source PDFs for the indexer Function |

Values live in `.env` or the environment, never in yaml or Python. Any code that needs
a secret calls `get_settings().resolve_secret(NAME)`.

## Logging

`configure_logging()` completes the `config_logging.yaml` template at runtime.

| Variable | Default | Effect |
| --- | --- | --- |
| `LOG_LEVEL` | `INFO` | Root level |
| `LOG_TO_FILE` | `false` | Also write `logs/application.log` under the repository root; the folder is created on first use |

Prompts and payloads are logged at `DEBUG` only, so a production `INFO` log never
carries a system prompt.

## Services

- **MCP server**: `make mcp-server` runs `src/services/mcp/server_oaklang_agent.py` on
  port 8000. The root `Dockerfile` runs the same process (`make docker-build`,
  `make docker-run`).
- **Azure Functions** (`src/services/functions/`): an EventGrid indexer and two MCP
  tools. `function_app.py` is a container packaging artifact that loads only after the
  build reshapes the tree under `/home/site/wwwroot`, so it is not importable from the
  repo. Local run: `make function-app-run`, then `make function-app-logs`.
