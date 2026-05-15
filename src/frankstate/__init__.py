"""Public root API for the reusable `frankstate` package.

Only `WorkflowBuilder` is re-exported from the package root.

All other reusable building blocks must be imported from subpackages so that
`frankstate` does not become an absolute import bucket. Reusable contracts come
from their concrete modules. Examples:

- `from frankstate.entity.graph_layout import GraphLayout`
- `from frankstate.entity.node import SimpleNode, CommandNode`
- `from frankstate.entity.edge import SimpleEdge, ConditionalEdge`
- `from frankstate.entity.statehandler import StateEnhancer, StateEvaluator, StateCommander`
- `from frankstate.entity.runnable_builder import PromptMixin, RetrieverMixin, RunnableBuilder`
- `from frankstate.managers.node_manager import NodeManager`
- `from frankstate.managers.edge_manager import EdgeManager`
"""

from frankstate.workflow_builder import WorkflowBuilder

__all__ = ["WorkflowBuilder"]
