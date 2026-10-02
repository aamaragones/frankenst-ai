# 🧟 frankstate

[![release](https://github.com/aamaragones/frankenst-ai/actions/workflows/release.yaml/badge.svg?branch=main)](https://github.com/aamaragones/frankenst-ai/actions/workflows/release.yaml?query=branch%3Amain)
[![pypi](https://img.shields.io/pypi/v/frankstate.svg)](https://pypi.python.org/pypi/frankstate)
[![license](https://img.shields.io/github/license/aamaragones/frankenst-ai.svg)](https://github.com/aamaragones/frankenst-ai/blob/main/LICENSE)
[![downloads](https://static.pepy.tech/badge/frankstate/month)](https://pepy.tech/project/frankstate)
[![versions](https://img.shields.io/pypi/pyversions/frankstate.svg)](https://github.com/aamaragones/frankenst-ai)

`frankstate` is a lightweight pattern layer for assembling LangGraph workflows with clearer structure, stronger boundaries, and less duplicated graph wiring.

It does not replace LangGraph and it does not introduce a separate runtime. The compiled result is still a native LangGraph graph built with official LangGraph primitives.

## What The Package Provides

The published package focuses on reusable workflow assembly contracts:

- `WorkflowBuilder` to compile a graph from a layout class, and `to_mermaid()` to render it. Its constructor and `compile()` take LangGraph's own `StateGraph` and `StateGraph.compile` options as `**kwargs`.
- `GraphLayout` to separate runtime dependency construction from graph declaration.
- `SimpleNode`, `CommandNode`, `ToolGraphNode`, `SimpleEdge`, and `ConditionalEdge` to model graph structure.
- `StateEnhancer`, `StateEvaluator`, and `StateCommander` to keep node and routing logic aligned with LangGraph concepts.
- `RunnableBuilder` with `PromptMixin` and `RetrieverMixin` to assemble LCEL chains inside a layout.
- `NodeManager` and `EdgeManager` to normalize layout declarations into LangGraph registration calls.

## Public API

The package root intentionally exports only:

```python
from frankstate import WorkflowBuilder
```

All other reusable contracts should be imported from subpackages, usually from
their concrete modules:

```python
from frankstate.entity.graph_layout import GraphLayout
from frankstate.entity.node import SimpleNode, CommandNode, ToolGraphNode
from frankstate.entity.edge import SimpleEdge, ConditionalEdge
from frankstate.entity.statehandler import StateEnhancer, StateEvaluator, StateCommander
from frankstate.entity.runnable_builder import RunnableBuilder, PromptMixin, RetrieverMixin
from frankstate.managers.node_manager import NodeManager
from frankstate.managers.edge_manager import EdgeManager
```


## Installation

With `pip`:

```bash
pip install frankstate
```

With `uv`:

```bash
uv pip install frankstate
```

Optional example dependencies:

```bash
pip install frankstate[examples]
```

The published wheel contains only `frankstate`.
Repository-level reference code, service integrations, notebooks, and tests are not part of the base package.

## Minimal Example

```python
from frankstate import WorkflowBuilder
from langgraph.checkpoint.memory import InMemorySaver
from my_project.layouts.simple_graph import SimpleGraphLayout
from my_project.state import GraphState

workflow_builder = WorkflowBuilder(
    config=SimpleGraphLayout,
    state_schema=GraphState,          # **kwargs go to StateGraph: context_schema, input_schema, output_schema
)

graph = workflow_builder.compile(      # **kwargs go to StateGraph.compile
    checkpointer=InMemorySaver(),
    interrupt_before=["review"],
)
print(workflow_builder.to_mermaid())   # the topology as a Mermaid diagram
```

`graph` is a native LangGraph `CompiledStateGraph`. `frankstate` adds no option of its
own: anything `StateGraph(...)` or `StateGraph.compile(...)` accepts is passed by its
LangGraph name, so a new LangGraph option works without a `frankstate` release. A name
`StateGraph` would silently swallow (its deprecated aliases such as `config_schema`)
raises `TypeError` at the constructor; `compile()` lets LangGraph raise its own.

Migrating from 0.2: `checkpointer`, `input_schema` and `output_schema` are no longer
constructor parameters of their own. `checkpointer` moves to `compile(checkpointer=...)`;
the two schemas keep working as constructor `**kwargs`. Handler keyword arguments must now
be declared by a class annotation on the handler subclass.

## LangGraph Alignment

`frankstate` keeps the official LangGraph execution model:

- `StateEnhancer` wraps node logic that returns partial state updates.
- `StateEvaluator` wraps the callable used by conditional edges.
- `StateCommander` wraps nodes that return official LangGraph `Command` objects.

The compiled graph still relies on LangGraph's own `StateGraph`, `add_node()`, `add_edge()`, `add_conditional_edges()`, and `Command`.

Handlers are configured from the layout line. `StateEnhancer` and `StateEvaluator` take a
`runnable_builder` plus keyword arguments that become attributes, and the subclass declares
each accepted keyword with a class annotation; an undeclared name raises `TypeError` where
the layout passed it, not at the first node run that reads a missing attribute.

```python
class RetrieveContext(StateEnhancer):
    retriever: MyRetriever            # declares the `retriever=` keyword
    strict: bool = False              # optional, with its default

    async def enhance(self, state):
        return {"context": self.retriever.get_context(state["question"])}

RetrieveContext(retriever=my_retriever, strict=True)   # in the layout
```

## Repository Boundaries

If you are browsing the repository instead of the published package:

- `src/frankstate` is the reusable package.
- `src/core_ai_examples` is the repository reference layer.
- `src/services` contains repository integrations and deployment entrypoints.

Those repository layers help demonstrate how `frankstate` can be consumed, but they are not the stable public API of the base wheel.

## Runnable Builders

`RunnableBuilder` is a recommended way to organize LangChain LCEL builders inside a layout.
Optional capabilities can be added through cooperative multiple inheritance using mixins:

| Mixin | Adds |
|---|---|
| `PromptMixin` | Enforces a `_build_prompt(**kwargs)` hook |
| `RetrieverMixin` | Lazily initialized retriever from a `vectordb` or pre-built `retriever` |

Mix and match to get exactly the capability each builder needs — no base class carries attributes it does not use:

```
ChatBuilder   → PromptMixin, RunnableBuilder                   # prompt + model
RAGBuilder    → RetrieverMixin, PromptMixin, RunnableBuilder   # retriever + prompt + model
AgentBuilder  → PromptMixin, RunnableBuilder                   # prompt + tools (tools stored in subclass)
```

Working examples live in the repository under `src/core_ai_examples/components/runnables/`.
