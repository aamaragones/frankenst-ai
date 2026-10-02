"""WorkflowBuilder: assembles a LangGraph StateGraph from a GraphLayout."""

import inspect
import logging
from typing import Any

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from frankstate.entity.graph_layout import GraphLayout
from frankstate.managers.edge_manager import EdgeManager
from frankstate.managers.node_manager import NodeManager


class WorkflowBuilder:
    """Assemble a LangGraph `StateGraph` from a `GraphLayout` subclass.

    The builder mirrors LangGraph's own surface instead of wrapping it: constructor
    `**kwargs` go verbatim to `StateGraph(state_schema, **kwargs)` and `compile(**kwargs)`
    goes verbatim to `StateGraph.compile(**kwargs)`, so a checkpointer, `interrupt_before`
    or `context_schema` are passed exactly where LangGraph asks for them. The public flow:

    1. Instantiate the builder with a layout and a state schema.
    2. Call `compile(...)` with any `StateGraph.compile` option.
    3. Invoke the returned compiled graph from notebooks, services or apps.
    """

    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        config: type[GraphLayout],
        state_schema: type[Any],
        **kwargs: Any,
    ):
        """Create a workflow builder for a graph layout.

        Args:
            config: Layout class inheriting from `GraphLayout`.
            state_schema: LangGraph state schema used by `StateGraph`.
            **kwargs: Forwarded verbatim to `StateGraph`, such as `context_schema`,
                `input_schema` or `output_schema`. `StateGraph` swallows unknown
                names for its deprecated aliases, so they are rejected here instead.
        """
        accepted = inspect.signature(StateGraph.__init__).parameters.keys() - {
            "self",
            "state_schema",
            "kwargs",
        }
        if unsupported := kwargs.keys() - accepted:
            raise TypeError(
                f"Unsupported StateGraph option(s) {sorted(unsupported)}; "
                f"supported: {sorted(accepted)}."
            )
        if not isinstance(config, type) or not issubclass(config, GraphLayout):
            raise TypeError(
                "WorkflowBuilder expects `config` to be a GraphLayout subclass"
            )

        self.workflow: StateGraph[Any, Any, Any, Any] = StateGraph(
            state_schema, **kwargs
        )
        self.config: GraphLayout = config()
        self.edge_manager: EdgeManager = EdgeManager()
        self.node_manager: NodeManager = NodeManager()
        self._workflow_configured: bool = False

        self.logger.info(
            "WorkflowBuilder initialized for GraphLayout %s",
            config.__name__,
        )

    def compile(self, **kwargs: Any) -> CompiledStateGraph[Any, Any, Any, Any]:
        """Assemble the layout once, then call `StateGraph.compile(**kwargs)` verbatim.

        `checkpointer`, `interrupt_before`, `store`, `cache`, `name` and every other
        compile option keep LangGraph's names; an unknown one raises LangGraph's own
        `TypeError` at this call.
        """
        self._ensure_workflow_configured()
        return self.workflow.compile(**kwargs)

    def to_mermaid(self, with_metadata: bool = False) -> str:
        """Return the compiled graph as Mermaid text.

        This is the default, dependency-free representation: it relies only on
        LangGraph/LangChain core, needs no network access and produces a
        deterministic string suitable for documentation or diffs.

        Node `metadata` declared through layout `kwargs` would otherwise be
        inlined into every node label by the Mermaid renderer. By default that
        metadata is cleared so the diagram shows only node names.

        Args:
            with_metadata: When `True`, keep node metadata in the rendered
                labels. Defaults to `False` for a clean diagram showing only
                node names.
        """
        graph = self.compile().get_graph()
        if not with_metadata:
            for node_id, node in list(graph.nodes.items()):
                graph.nodes[node_id] = node._replace(metadata=None)
        return graph.draw_mermaid()

    def _ensure_workflow_configured(self) -> None:
        """Configure the workflow once before any compile or visualization step."""
        if not self._workflow_configured:
            self._configure_workflow()

    def _configure_workflow(self) -> None:
        """Assemble the workflow from the nodes and edges discovered in the layout."""
        self._configure_nodes()
        for node_args, node_kwargs in self.node_manager.configs_nodes():
            self.workflow.add_node(*node_args, **node_kwargs)

        self._configure_edges()
        for config in self.edge_manager.configs_edges():
            self.workflow.add_edge(*config)
        for (
            node_source,
            router,
            path_map,
        ) in self.edge_manager.configs_conditional_edges():
            self.workflow.add_conditional_edges(
                node_source,
                router,
                path_map=path_map,
            )

        self._workflow_configured = True

    def _configure_nodes(self) -> None:
        """Load node definitions from the layout into the node manager."""
        self.node_manager.add_nodes(nodes=self.config.get_nodes())

    def _configure_edges(self) -> None:
        """Load edge definitions from the layout into the edge manager."""
        self.edge_manager.add_edges(edges=self.config.get_edges())
