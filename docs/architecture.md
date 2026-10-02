---
type: Concept
title: Architecture
description: Repository layers and the dependency rule, the frankstate assembly lifecycle, entity contracts, the kwargs seam on nodes and on WorkflowBuilder, and the invariants the tests enforce.
tags: [architecture, frankstate, langgraph, layers, contracts]
generated:
    by: reference_agent
    at: 2026-10-02T00:00:00Z
---

# Architecture

`frankstate` is not a runtime. It is a thin assembly layer that turns a declarative
`GraphLayout` into an official LangGraph `CompiledStateGraph`. Everything else in this
repository exists to show one way of consuming it. How to run the examples is in
[examples.md](examples.md); how models are configured in [llm-services.md](llm-services.md).

## Layers and the dependency rule

| Layer | Path | Role | Contract |
| --- | --- | --- | --- |
| Patterns | `src/frankstate/` | The published wheel. Imports nothing from this repo | Public, versioned by the PR title type (see [release.md](release.md)) |
| Config | `src/config/` | `settings.py`, the three YAML files, `graph_layout/` | The single door to configuration and secrets |
| Shared | `src/utils/` | config loader, logger, secrets backend, blob storage, Ollama proxy, RAG indexers | Reads paths from `config.settings`; never imports `services` or a core. `secrets.py` imports nothing from the repo |
| Core examples | `src/core_ai_examples/` | components (nodes, edges, runnables, tools, retrievers) and models | Reference, not public API |
| Services | `src/services/` | `llm/` (LLMServices), `mcp/`, `functions/` (Azure Functions) | Adapters; reach a core only to expose it |

```
core_ai_examples ─┐
services ─────────┼──▶ config, utils, frankstate
                  │
services ─────────┴──▶ core_ai_examples   (adapters only: Functions orchestrators, MCP server)
config/graph_layout ──▶ core_ai_examples, services.llm   (the one sanctioned upward import)
config/settings ──▶ utils.secrets   (and nothing else imports utils.secrets)
```

The layouts are the exception on purpose: a `GraphLayout` wires core components to a
model, so it must see both. Nothing under `core_ai_examples` imports it back. Before
this rule `services` and the examples imported each other in six files, and the only
way to import either was the editable install's `.pth` file: a fresh checkout without
`uv sync` could not run a single example test. `tests/unit_test/examples/config/test_layer_imports.py`
scans the imports and fails the PR that reintroduces a cycle.

The wheel is frozen at the artifact level: `tests/integration_test/test_wheel_contents.py`
builds it and fails if `config/`, `utils/`, `core_ai_examples/` or `services/` leak in.

## The `frankstate` package

The root exports one name, `WorkflowBuilder`; every other contract is imported from
its concrete module. The `entity` and `managers` packages export nothing, so the root
never becomes an import bucket.

```
src/frankstate/
├── workflow_builder.py      WorkflowBuilder(config, state_schema, **StateGraph kwargs): compile(**compile kwargs), to_mermaid()
├── entity/
│   ├── graph_layout.py      GraphLayout: build_runtime() then layout()
│   ├── node.py              BaseNode, SimpleNode, CommandNode, ToolGraphNode
│   ├── edge.py              BaseEdge, SimpleEdge, ConditionalEdge
│   ├── statehandler.py      StateEnhancer, StateEvaluator, StateCommander
│   └── runnable_builder.py  RunnableBuilder, PromptMixin, RetrieverMixin
└── managers/
    ├── node_manager.py      NodeManager
    └── edge_manager.py      EdgeManager
```

`entity` is what you declare; `managers` is how it is serialized into LangGraph;
`WorkflowBuilder` orchestrates.

## Assembly lifecycle

`compile()` runs `get_nodes()` → `NodeManager.add_nodes()` → `StateGraph.add_node()`
per node, then `get_edges()` → `EdgeManager` → `add_edge()` / `add_conditional_edges()`,
then `StateGraph.compile(**kwargs)` with whatever `compile()` received (`checkpointer`,
`interrupt_before`, `store`, `cache`, `name`). The result is a plain LangGraph graph.

A layout runs in two phases, each at most once per instance, so nothing happens at
import time:

1. `build_runtime() -> dict[str, Any]` builds the heavy dependencies (models,
   retrievers, agents). Every returned key must be an annotated class attribute and
   every annotated attribute must be returned, or construction fails there instead of
   at the first `compile()`.
2. `layout() -> None` declares nodes and edges as instance attributes.

Discovery is **by attribute type in declaration order**. Reordering two attributes
reorders the graph; renaming one changes nothing.

## Entity contracts

| Contract | Wraps | Returns | LangGraph call |
| --- | --- | --- | --- |
| `StateEnhancer.enhance(state)` | a node callable | partial state update or `Command` | `add_node()` |
| `StateEvaluator.evaluate(state)` | a routing callable | routing key, a list of keys, or `Send`s | `add_conditional_edges()` router |
| `StateCommander.command(state)` | routing + update | `Command` | node returning `Command` |

`SimpleNode` resolves to `enhancer.enhance`, `CommandNode` to `commander.command` and
injects `destinations` from the commander for rendering, `ToolGraphNode` wraps a native
`ToolNode` so it joins the same pipeline. `SimpleEdge` is `add_edge()`;
`ConditionalEdge(node_source, map_dict, evaluator)` is `add_conditional_edges()`.

`RunnableBuilder` is the LCEL contract: implement `_configure_runnable()`, get a cached
`runnable` with `invoke`/`ainvoke`/`stream`/`astream`, which forward `**kwargs` (such as
`config=`) to the runnable. `PromptMixin` demands a `_build_prompt()` hook;
`RetrieverMixin` builds a retriever lazily from a `vectordb` or takes a pre-built one.

`StateEnhancer` and `StateEvaluator` take keyword arguments beyond `runnable_builder`
and set them as attributes, so a layout configures a node (`retriever=`, `strict=True`)
on the line that declares it. The subclass declares each accepted keyword with a class
annotation, the same rule `GraphLayout` applies to its runtime keys; an undeclared name
is a `TypeError` at the layout line instead of a missing attribute at the first run.

Node callables are the bound `enhance`/`command`/`evaluate` methods, so LangGraph's
injection by parameter name applies: a subclass that declares `runtime`, `config`,
`writer` or `store` receives it. `ToolGraphNode` takes `langgraph.prebuilt.ToolNode`,
the class's canonical home.

## The `kwargs` seam

`frankstate` is an assembler, so it never names a LangGraph option itself. Node
constructors accept native `add_node()` options (`metadata`, `retry_policy`,
`cache_policy`, `timeout`, `defer`, `error_handler`) as `**kwargs` and forward them
verbatim. `BaseNode.__init__` validates every key against `StateGraph.add_node()`'s
signature at construction, because the kwargs are stored and only reach `add_node()`
inside `compile()`, far from the typo. `WorkflowBuilder(config, state_schema, **kwargs)`
forwards to `StateGraph(...)` and validates the same way, for a different reason:
`StateGraph.__init__` accepts `**kwargs` for its deprecated aliases and swallows an
unknown name silently. `compile(**kwargs)` forwards to `StateGraph.compile(...)` without
validating, since that signature has no catch-all and Python already raises at the
call. New LangGraph options therefore work without a core change;
`test_workflow_builder_integration.py` asserts each one reaches the compiled graph.
Handler kwargs follow the same logic for the same reason: the value is read long after
the line that passed it, so the check against the class annotations happens there.

## Runtime constraints

Python `>=3.12.3`; `langchain-core>=1.6,<2`, `langgraph>=1.2,<2`, `pydantic>=2.13,<3`.
Major-bounded ranges, never `==`: the package introspects LangGraph signatures instead
of hardcoding them, so a LangGraph minor does not need a `frankstate` release, and a
library that pins exact versions cannot be installed next to anything else. The `examples` extra adds the providers and
tooling the reference layer needs; it does not ship the reference code.

## Invariants

1. `frankstate` imports nothing from this repository.
2. The root exports only `WorkflowBuilder`.
3. Layout declaration order is the topology; discovery is by type.
4. `build_runtime()` and `layout()` run at most once per instance.
5. Native LangGraph options travel through `**kwargs`: `add_node()`'s on nodes, `StateGraph`'s and `compile()`'s on `WorkflowBuilder`.
6. The build returns an official LangGraph graph; there is no second runtime.
7. `config.settings` is the only entry point to configuration paths and secrets.
