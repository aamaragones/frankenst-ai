# 🧟 Frankenst-AI | LangGraph Patterns
**Frankenst-AI** is a project that introduces a **modular and scalable structure** based on design patterns and coding best practices, applied to **LangGraph**.

It is not a framework that replaces LangGraph. It is a reusable project layer that helps you organize components, layouts and runtime assembly so you can build LangGraph workflows with less duplication and stronger boundaries.

This README describes the repository as a whole.
The published package in this mono-repo is `frankstate`; package-focused installation and API notes live in `README-pypi.md`.

This project aims to improve **scalability**, **reusability**, **testability**, and **maintainability** through reusable, configurable, and highly decoupled components designed to assemble complex LLM workflows.

By leveraging a well-organized structure, the project enables the creation of composable and extensible AI systems (agent patterns, RAG patterns, MCPs, etc.) while still returning official LangGraph graphs at the end of the build process.

The project has been designed with the following goals:

- **Isolated logic and separation of concerns** between conditional edges, nodes, tools, and runnables, encapsulated as reusable and independent components.

- Key components like **StateEnhancer**, **StateCommander**, and **StateEvaluator** are designed to be reused across multiple workflows.

- Use of YAML, **centralized configurations**, and explicit layout classes managed with builders and managers to define, **modify**, and **scale** different graph architectures **without duplicating logic**.

## Repository Shape

This repository is a mono-repo with four layers and different expectations of stability:

- `src/frankstate` is the reusable pattern layer and the only code published in the base `frankstate` wheel.
- `src/core_examples` is the repository reference package. It shows one concrete way to consume `frankstate`, but it is not the stable public API of the published package. Inside this mono-repo it may reuse shared adapters from `src/services/foundry` to keep example runtime bootstrap centralized.
- `src/services` is the integration layer for repository-specific runtimes and deployment entrypoints such as MCP servers or Azure Functions.
- `research` contains exploratory notebooks and experiments. It is useful context, but it is not part of the repository's contractual surface.

If you are evaluating the repository, start with `src/frankstate`, then move to `src/core_examples`, and only read `src/services` when you need integration-specific entrypoints.

## How Frankenst-AI Maps to LangGraph

Frankenst-AI keeps the official LangGraph runtime model and adds a small layer
of project-specific naming around it:

- **StateEnhancer** wraps the node callable that reads the current state and returns a partial update.
- **StateEvaluator** wraps the callable passed to conditional edges and returns the routing key used in the path map.
- **StateCommander** wraps nodes that return an official LangGraph `Command` when routing and state updates must happen in the same step.

These names are project abstractions, but the compiled graph still relies on
`StateGraph`, `add_node()`, `add_edge()`, `add_conditional_edges()` and
`Command` from LangGraph.

In other words, Frankenst-AI helps you structure and assemble LangGraph workflows; it does not introduce a separate graph runtime.

## Runtime Support

Frankstate currently supports LangGraph as its implemented workflow runtime.

The abstractions in this repository are intentionally being shaped so they can
grow beyond a single runtime, and Microsoft Agent Framework is a planned future
integration direction. That future support does not exist yet in the published
package, examples or tests.

## Frankstate Public API

The published package keeps a very small root API:

```python
from frankstate import WorkflowBuilder
```

That root import is reserved for the main assembly entrypoint.

All other reusable contracts should be imported from subpackages,
not from `frankstate.__init__`.

Most of those contracts still come from concrete modules, while
`frankstate.managers` exposes a small curated shortcut surface.

This repository README keeps only that summary so it stays focused on the mono-repo.
For the package-oriented import guide, installation path, and public API framing, see `README-pypi.md`.

This keeps `frankstate` root stable and prevents it from turning into an absolute import bucket for every internal type.

## Prerequisites

- Python 3.12.3 or higher
- Ollama 0.20.2 or higher (free); or an Azure Foundry Deployment (payment)
- uv 0.8 or higher

Install `uv` with the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

Choose one of these two installation paths depending on what you need.

### Option A. Install the published package `frankstate`

Use this option when you only want the reusable public package published on PyPI.

For package-only documentation, examples, and API framing, prefer `README-pypi.md`.

- With `pip` installer:
```pip install frankstate```
- With `uv` installer:
```uv pip install frankstate```

This option installs only the published `frankstate` wheel.
It does not install the repository reference package under `src/core_examples`, the service layer under `src/services`, or the repository tests.
The `examples` extra only adds optional dependencies; it does not install the repository example code.

### Option B. Clone and install the repository

Use this option when you want the full mono-repo, including `src/core_examples`, prompt assets, tests and local development tooling.

1. Clone the repository.

2. Create a local virtual environment:

    ```bash
    uv venv .venv
    ```

3. Activate it:

    ```bash
    source .venv/bin/activate
    ```

    On Windows PowerShell:

    ```powershell
    .venv\Scripts\Activate.ps1
    ```

4. Sync the base repository environment:

    ```bash
    uv sync --frozen
    ```

5. If you also want the repository examples and development dependencies:

    ```bash
    uv sync --frozen --extra examples --group dev
    ```

6. Run commands from the active virtual environment, or keep using `uv run`:

    ```bash
    uv run pytest -q
    ```

7. (Optional) System packages for the example/document-processing stack:

    ```
    sudo apt update
    sudo apt-get install poppler-utils
    sudo apt install tesseract-ocr
    ```

`uv.lock` is committed, so `uv sync --frozen` recreates the repository environment without resolver drift.

Check optional root shortcuts with `make help`.

## Running the Repository Locally

The steps below are for running the mono-repo reference stack locally, not just the published package:

1. Choose an LLM Services backend

   Choose one of the following options:

   #### 1.1 Using a local model with Ollama
   Start the Ollama service: ```ollama run ministral-3:8b```
  
   #### 1.2 Using Azure AI Foundry Deployment
    Configure your model variables in `.env`: ```cp .env.example .env```
2. Compile graph layouts with `WorkflowBuilder`
    
    The minimal example below uses the reference package under `src/core_examples` to show how the published `frankstate` package is consumed in a real repository.

    Reference layouts now follow a two-step contract:

    - `build_runtime()` resolves runtime dependencies such as LLM services, runnable builders, embeddings or retrievers.
    - The keys returned by `build_runtime()` are projected onto the layout instance and must be declared as annotated attributes in the layout class.
    - `layout()` declares nodes and edges using those already-resolved attributes on the layout instance.

    This keeps imports side-effect free while preserving a declarative layout file.

    In that example:

    - `WorkflowBuilder` is part of the reusable pattern in `src/frankstate`.
    - `SimpleOakConfigGraph` and `SharedState` are concrete reference classes from `src/core_examples`.
    - In your own project, those `src/core_examples` imports would be replaced by your own layouts and state schemas.

    For exploratory examples and experimentation, refer to the `research/demo...ipynb` notebooks.

#### Minimal WorkflowBuilder Example

```python
from frankstate import WorkflowBuilder
from core_examples.config.layouts.simple_oak_config_graph import SimpleOakConfigGraph
from core_examples.models.stategraph.stategraph import SharedState

workflow_builder = WorkflowBuilder(
    config=SimpleOakConfigGraph,
    state_schema=SharedState,
)

graph = workflow_builder.compile()
```

The `graph` is still a LangGraph graph object produced through LangGraph's own runtime.

## Repository Logging

The repository now configures application logging through `configure_logging()` and the
`src/core_examples/config/config_logging.yml` template.

- `LOG_LEVEL` controls the root log level. Default: `INFO`.
- `LOG_TO_FILE` controls whether logs are also persisted under `logs/application.log`. Default: `false`.

Examples:

```bash
LOG_LEVEL=DEBUG python app.py
LOG_LEVEL=DEBUG LOG_TO_FILE=true python app.py
```

## Local Validation

```bash
uv run ruff check
uv run mypy
uv run pytest --cov=src --cov-report=term-missing -v
```

For a quick test-only pass:

```bash
uv run pytest -q
```

If you prefer root shortcuts:

- `make test` runs the full repository suite.
- `make test-frankstate` runs only the installable package suite under `tests/unit_test/frankstate`.

`ruff`, `mypy`, and `pytest` read their configuration from `pyproject.toml`.
The Docker validation pipeline runs this same `uv` sequence before image build steps.

## Local Functions Apps Container Run

This section is repository-specific and does not describe behavior of the published `frankstate` wheel.
- `src/services/functions/function_app.py` is an Azure Functions App Containers packaging
    artifact. It is not a reusable Python module from the source tree and is
    expected to load only after the container build reshapes the filesystem under
    `/home/site/wwwroot`.

- Useful local checks:
    ```bash
    docker build -f src/services/functions/Dockerfile -t mylocalfunction:0.1 .
    docker run -d -p 8080:80 mylocalfunction:0.1
    docker logs <container_id>
    ```

- Docker deep debug recipes: 
    ```bash 
    docker exec -it <container_id> /bin/bash 
    apt-get update 
    apt-get install azure-functions-core-tools-4 
    apt-get install azure-cli 
    cd /home/site/wwwroot 
    az login 
    func start --verbose
    ```

## Repository Structure

```bash
frankenst-ai/
├── main.py                  # Local entry point to assemble and compile graph layouts
├── app.py                   # Optional deployment-facing wrapper entry point
├── pyproject.toml           # Single source of truth for package metadata, extras and dependency groups
├── uv.lock                  # Locked repository environment for reproducible uv sync flows
├── .env                     # Environment variables for local configuration; .env.example for reference
├── README.md                # Main project documentation
├── src/
│   ├── services/            # Service entrypoints plus shared provider adapters used by the repository
│   ├── core_examples/       # Importable reference package showing how to structure a real LangGraph project using `frankstate`
│   │   ├── components/
│   │   │   ├── nodes/
│   │   │   │   ├── enhancers/        # StateEnhancers for simple node logic modifying StateGraph via runnables or custom modules
│   │   │   │   └── commands/         # StateCommander for routing and modifying state through LangGraph commands
│   │   │   ├── edges/
│   │   │   │   └── evaluators/       # StateEvaluator for conditional edge logic
│   │   │   ├── tools/                # Tool definitions and integrations
│   │   │   ├── retrievers/           # Retrievers definitions, builders and integrations
│   │   │   └── runnables/            # Executable LangChain RunnableBuilder modules for invoke or ainvoke logic
│   │   ├── config/
│   │   │   ├── config.yml            # Main runtime configuration file for the project
│   │   │   ├── config_nodes.yml      # Node registry used by the example graph layouts
│   │   │   └── layouts/              # Reference GraphLayout subclasses using build_runtime() + layout()
│   │   ├── constants/           
│   │   ├── models/                   # Structural models: StateGraph, tool properties, structured outputs, etc.
│   │   └── utils/                  
│   └── frankstate/          # Frankstate utilities for assembling and compiling LangGraph
│       ├── entity/
│       │   ├── graph_layout.py       # Base GraphLayout contract: build runtime first, then declare nodes and edges
│       │   ├── runnable_builder.py   # Builder class for LangChain Runnable objects
│       │   ├── statehandler.py       # Core entities for handling StateGraph
│       │   ├── node.py               # Core node-related entities 
│       │   └── edge.py               # Core edge-related entities
│       ├── managers/               
│       └── workflow_builder.py       # Workflow Builder to compile LangGraph from GraphLayout subclasses
├── research/                # Exploratory notebooks and experiments; useful as reference
├── tests/
│   ├── integration_test/      
│   └── unit_test/              
├── artifacts/               # Generated artifacts, static files and outputs
└── logs/                    # Log files and runtime logs
```

## Contributing

See `CONTRIBUTING.md` for repository boundaries, local setup, documentation conventions and pull request expectations.
