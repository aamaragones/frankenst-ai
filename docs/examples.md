---
type: Guide
title: Reference examples
description: The four GraphLayouts under src/config/graph_layout, the two install profiles (databricks or mcp), tools declared by name, the human-in-the-loop pause and resume, logging, and the MCP and Azure Functions services.
tags: [examples, layouts, ollama, azure, langgraph, interrupt, mcp, logging]
generated:
    by: reference_agent
    at: 2026-10-02T00:00:00Z
---

# Reference examples

Everything outside `src/frankstate/` is one concrete way to consume the package. It is
not installed by `pip install frankstate`; clone the repository for it. The layer roles
are in [architecture.md](architecture.md); the model configuration in
[llm-services.md](llm-services.md).

## Setup

```bash
make install-dev    # examples + databricks extras, dev group: the quality env
make install-mcp    # examples + mcp extras, dev group: the MCP server and client env
cp .env.example .env                            # only the variables your provider needs
```

| Extra | Adds | Combines with |
| --- | --- | --- |
| `examples` | the providers `ollama` and `azure_ai`, RAG tooling, notebooks | always |
| `databricks` | `databricks-langchain`, the `databricks` provider | `examples`; never `mcp` |
| `mcp` | `langchain[mcp]` (`MCPAdapter`) and FastMCP 4 | `examples`; never `databricks` |

`databricks` and `mcp` never share a venv: their `mcp` pins exclude each other, and
`[tool.uv].conflicts` makes `uv sync` refuse the pair. Since `LLMServices.launch()` builds
every `launch` key, the shipped yaml keeps them on `ollama`, which both envs carry.

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
| `oak_human_loop` | `OakHumanLoopConfigGraph` | agent + human review before a sensitive tool | `low_model`, a checkpointer at `compile()` |
| `local_vectorstore_rag` | `LocalVectorstoreAdaptiveRAGConfigGraph` | adaptive RAG over a local Chroma store | `model`, `embeddings` |
| `ai_search_rag` | `AISearchAdaptiveRAGConfigGraph` | adaptive RAG over Azure AI Search | `model`, `embeddings`, `AZURE_SEARCH_*` |

Every layout follows the two-phase contract: `build_runtime()` asks
`LLMServices.launch().require(...)` for its clients and builds the runnable builders;
`layout()` declares nodes and edges, reading node names from `config_nodes.yaml` through
`load_node_registry()`. A typo in that yaml fails on load, not inside `compile()`.

Tools are declared by name in the same file: `OAKTOOLS_NODE.metadata.tools` is what the
agent binds, `HUMAN_REVIEW_NODE.metadata.sensitive_tools` what pauses for review, and
`SIMPLE_OAKTOOLS_NODE.metadata.tools` the review-free subset `simple_oak` uses. The names
are the tools' own (`name` in each `*Property`), and a declared name no tool carries
fails `build_runtime()`.

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

## Human in the loop

`HumanReviewSensitiveToolCall` pauses with `interrupt(HumanReviewRequest, response_schema=
HumanReviewDecision)` when the agent calls a sensitive tool. Both models live in
`src/core_ai_examples/models/interrupt/human_review.py`: the request is what the reviewer
sees, the decision what they answer, validated by LangGraph before the node resumes.

```python
graph = WorkflowBuilder(config=OakHumanLoopConfigGraph, state_schema=SharedState).compile(
    checkpointer=InMemorySaver()                      # a pause needs somewhere to resume from
)
config = {"configurable": {"thread_id": "002"}}
paused = graph.invoke({"messages": [("user", "dominate all the pokemon of Ireland")]}, config)
paused["__interrupt__"][0].value["question"]          # the HumanReviewRequest
paused["__interrupt__"][0].response_schema            # == HumanReviewDecision.model_json_schema()
graph.invoke(Command(resume={"action": "feedback", "data": "I meant Iceland"}), config)
graph.invoke(Command(resume={"action": "maybe"}), config)   # pydantic.ValidationError
```

`continue` releases the pending tool calls; `feedback` answers them with the reviewer's
text and sends the agent round again. `research/demo_human_in_the_loop.ipynb` runs it.

## Configuration and secrets

| Variable | Read by | Purpose |
| --- | --- | --- |
| `FRANK_CORE_PACKAGE_PATH` | `settings` | Overrides the `src/` root the yaml paths hang off |
| `AZURE_KEY_VAULT_NAME` | `settings` | Enables the Key Vault fallback for every `{secret: NAME}` and every flagged settings field; unset, a missing secret is a `LookupError` naming the variable |
| `AZURE_FOUNDRY_PROJECT_ENDPOINT`, `AZURE_INFERENCE_MODEL_NAME`, `AZURE_EMBEDDINGS_ENDPOINT`, `AZURE_EMBEDDINGS_MODEL_NAME` | `config_llms.yaml` | The `azure_ai` provider |
| `DATABRICKS_LLM_ENDPOINT`, `DATABRICKS_EMBEDDINGS_ENDPOINT` | `config_llms.yaml` | The `databricks` provider |
| `AZURE_SEARCH_SERVICE_ENDPOINT`, `AZURE_SEARCH_API_KEY` | `settings.azure`, the retriever tool | Azure AI Search for `ai_search_rag` and the Functions app |
| `AZURE_BLOB_STORAGE_NAME` | `utils.blob_storage` | Source PDFs for the indexer Function |

Values live in the environment or in `.env` at the repository root, which settings finds
from any working directory, never in yaml or Python. Any code that needs a secret calls
`get_settings().resolve_secret(NAME)`.

## Logging

`configure_logging()` completes the `config_logging.yaml` template at runtime.

| Variable | Default | Effect |
| --- | --- | --- |
| `LOG_LEVEL` | `INFO` | Root level |
| `LOG_TO_FILE` | `false` | Also write `logs/application.log` under the repository root; the folder is created on first use |

Prompts and payloads are logged at `DEBUG` only, so a production `INFO` log never
carries a system prompt.

## Services

- **MCP server** (`mcp` env): `make mcp-server` runs `src/services/mcp/server_oaklang_agent.py`
  on port 8000; the root `Dockerfile` runs the same process. A human-review pause ends
  the tool call with the review question, since nothing resumes it over MCP. The client
  side is `langchain.mcp.MCPAdapter`, in `research/demo_graphs_comunication_mpc_tools.ipynb`.
- **Azure Functions** (`src/services/functions/`): an EventGrid indexer and two MCP
  tools. `function_app.py` is a container packaging artifact that loads only after the
  build reshapes the tree under `/home/site/wwwroot`, so it is not importable from the
  repo. Local run: `make function-app-run`, then `make function-app-logs`.
