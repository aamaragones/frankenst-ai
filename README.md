# 🧟 Frankenst-AI | LangGraph Patterns

**Frankenst-AI** is a modular, scalable structure for LangGraph, built on design
patterns rather than a new runtime. Nodes, conditional edges, command routing and LCEL
builders are independent, reusable contracts; a declarative `GraphLayout` wires them;
`WorkflowBuilder` compiles the result into an official LangGraph graph.

The published package is **`frankstate`** (`src/frankstate/`). Everything else in this
repository, the layouts, components, model services and Azure integrations, is one
concrete way to consume it and is not part of the wheel.

## Install

```bash
pip install frankstate            # the package only; API notes in README-pypi.md
```

```bash
git clone https://github.com/aamaragones/frankenst-ai && cd frankenst-ai
uv sync --frozen --extra examples --group dev   # the whole repository, with the examples
make help                                       # every target
```

Requires Python 3.12.3+ and `uv`. The examples run against Ollama by default; see
[docs/examples.md](docs/examples.md) for the provider switch.

## Documentation

| Page | Type | Description |
| --- | --- | --- |
| [architecture.md](docs/architecture.md) | Concept | Layers, the dependency rule, the `frankstate` assembly lifecycle, entity contracts and invariants |
| [llm-services.md](docs/llm-services.md) | Concept | `config_llms.yaml` as the single provider entry point: one runtime per `launch` key, secrets through settings, `use_responses_api` |
| [examples.md](docs/examples.md) | Guide | The four layouts, `python main.py --layout ...`, Ollama and Azure setup, logging, the Functions container |
| [ways-of-working.md](docs/ways-of-working.md) | Guide | `make ci`, Conventional Commits and what each type releases, the PR standard, the comment-ratio and OKF rules |
| [release.md](docs/release.md) | Runbook | How a merge to `main` becomes a PyPI version, one-time settings, first release, recovery |

Pages under `docs/` carry [OKF v0.2](https://cloud.google.com/blog/products/data-analytics/okf-v0-2-adds-trust-signals)
frontmatter: `type` (the only required field: `Concept`, `Guide`, `Runbook` or
`Reference`), `title`, `description`, `tags`, and `generated: {by, at}` recording who
produced the content. `by` distinguishes an agent (`reference_agent/<model>`) from a
person (`human:<id>`). A human review is recorded as `verified: [{by: human:<id>, at: <ts>}]`,
never on someone else's behalf. No page carries one yet.

`docs/` is flat and this file is the index: one domain, nothing to group.

The package's own API reference is [README-pypi.md](README-pypi.md), which is also the
PyPI long description. Contribution rules are in [CONTRIBUTING.md](CONTRIBUTING.md);
vulnerability reporting in [SECURITY.md](SECURITY.md); license in [LICENSE](LICENSE).

## Repository structure

```
frankenst-ai/
├── main.py                       Render a layout: python main.py --layout simple_oak
├── pyproject.toml                Package metadata, the `examples` extra, dependency groups, tool config
├── Makefile                      The single command interface; CI calls only its targets
├── src/
│   ├── frankstate/               The published wheel (entity/, managers/, workflow_builder.py)
│   ├── config/                   settings.py, config_llms.yaml, config_nodes.yaml, config_logging.yaml,
│   │   └── graph_layout/         the four reference GraphLayouts
│   ├── utils/                    config loader, logger, secrets backend, blob storage, ollama/, rag/
│   ├── core_ai_examples/         components/ (nodes, edges, runnables, tools, retrievers) and models/
│   └── services/                 llm/ (LLMServices), mcp/ (server), functions/ (Azure Functions)
├── tests/                        unit_test/frankstate (the slice) and unit_test/examples (mirrors the rest); integration_test/ freezes the wheel
├── docs/                         OKF v0.2 pages, indexed above
├── research/                     Exploratory notebooks; not part of the suite
└── artifacts/                    Rendered layout diagrams and sample documents
```
