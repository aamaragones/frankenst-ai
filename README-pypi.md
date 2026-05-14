# 🧟 frankstate

`frankstate` is a lightweight pattern layer for assembling LangGraph workflows with clearer structure, stronger boundaries, and less duplicated graph wiring.

It does not replace LangGraph and it does not introduce a separate runtime. The compiled result is still a native LangGraph graph built with official LangGraph primitives.

## What The Package Provides

The published package focuses on reusable workflow assembly contracts:

- `WorkflowBuilder` to compile a graph from a layout class.
- `GraphLayout` to separate runtime dependency construction from graph declaration.
- `SimpleNode`, `CommandNode`, `SimpleEdge`, and `ConditionalEdge` to model graph structure.
- `StateEnhancer`, `StateEvaluator`, and `StateCommander` to keep node and routing logic aligned with LangGraph concepts.
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
from frankstate.entity.node import SimpleNode, CommandNode
from frankstate.entity.edge import SimpleEdge, ConditionalEdge
from frankstate.entity.statehandler import StateEnhancer, StateEvaluator, StateCommander
from frankstate.entity.runnable_builder import RunnableBuilder
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
from my_project.layouts.simple_graph import SimpleGraphLayout
from my_project.state import GraphState

workflow_builder = WorkflowBuilder(
    config=SimpleGraphLayout,
    state_schema=GraphState,
)

graph = workflow_builder.compile()
```

## LangGraph Alignment

`frankstate` keeps the official LangGraph execution model:

- `StateEnhancer` wraps node logic that returns partial state updates.
- `StateEvaluator` wraps the callable used by conditional edges.
- `StateCommander` wraps nodes that return official LangGraph `Command` objects.

The compiled graph still relies on LangGraph's own `StateGraph`, `add_node()`, `add_edge()`, `add_conditional_edges()`, and `Command`.

## Repository Boundaries

If you are browsing the repository instead of the published package:

- `src/frankstate` is the reusable package.
- `src/core_examples` is the repository reference layer.
- `src/services` contains repository integrations and deployment entrypoints.

Those repository layers help demonstrate how `frankstate` can be consumed, but they are not the stable public API of the base wheel.