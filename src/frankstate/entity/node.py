"""Node wrapper definitions consumed by GraphLayout and NodeManager."""

import inspect
from typing import Any

from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from frankstate.entity.statehandler import StateCommander, StateEnhancer

_ADD_NODE_KWARGS: frozenset[str] = frozenset(
    name
    for name, parameter in inspect.signature(StateGraph.add_node).parameters.items()
    if parameter.kind is inspect.Parameter.KEYWORD_ONLY
)


class BaseNode:
    """Base named node definition consumed by GraphLayout and NodeManager.

    Extra keyword arguments are collected into `kwargs` and forwarded verbatim to
    `StateGraph.add_node()` (for example `metadata`, `retry_policy`, `cache_policy`,
    `timeout`, `defer` or `destinations`). Accepting them as `**kwargs` keeps call
    sites flush with LangGraph's own `add_node()` surface instead of nesting a
    dictionary argument.

    To keep that ergonomics safe, every key is validated against the keyword-only
    parameters of `StateGraph.add_node()` at construction time. An unknown option
    (including a typo in a node argument that would otherwise be swallowed) raises
    a `TypeError` immediately at the call site rather than failing later during
    `compile()`.
    """

    def __init__(self, name: str, **kwargs: Any):
        unsupported = kwargs.keys() - _ADD_NODE_KWARGS
        if unsupported:
            raise TypeError(
                f"{type(self).__name__} received unsupported add_node option(s) "
                f"{sorted(unsupported)}. Supported options: {sorted(_ADD_NODE_KWARGS)}."
            )
        self.name = name
        self.kwargs: dict[str, Any] = kwargs

class SimpleNode(BaseNode):
    """Node wrapper for a StateEnhancer callable.

    Extra keyword arguments are passed through to `StateGraph.add_node()` by the
    workflow builder.
    """

    def __init__(
        self,
        enhancer: StateEnhancer,
        name: str,
        **kwargs: Any,
    ):
        super().__init__(name, **kwargs)
        self.enhancer = enhancer

class CommandNode(BaseNode):
    """Node wrapper for a StateCommander callable returning Command.

    Extra keyword arguments are passed through to `StateGraph.add_node()`, but
    `destinations` remains controlled by the commander contract so graph
    rendering stays consistent with the `Command.goto` targets.
    """

    def __init__(
        self,
        commander: StateCommander,
        name: str,
        **kwargs: Any,
    ):
        try:
            _ = commander.destinations
        except AttributeError as exc:
            raise ValueError(
                f"{type(commander).__name__} must expose a 'destinations: dict[str, str]' property "
                "or a constructor-populated '_destinations' attribute where values are the "
                "registered names of destination nodes. See StateCommander docstring for the convention."
            ) from exc
        super().__init__(name, **kwargs)
        self.commander = commander

    @property
    def destinations(self) -> tuple[str, ...]:
        """Return route targets for LangGraph graph rendering.

        Reads `commander.destinations` so that `add_node(destinations=...)` can
        draw edges without requiring a `Literal` annotation on the callable.
        This is only used for graph rendering and has no effect on graph execution.
        """
        return tuple(self.commander.destinations.values())

class ToolGraphNode(BaseNode):
    """Node wrapper for a native LangGraph `ToolNode`.

    The wrapped `tool_node` is the action added to the graph, so its own
    `tags` and `name` are preserved by LangGraph. The wrapper exists only to
    give the tool node the same `kwargs` pass-through surface as the other
    node wrappers, allowing `add_node` options such as `metadata`,
    `retry_policy` or `timeout` to travel with the tool node from GraphLayout.
    """

    def __init__(
        self,
        tool_node: ToolNode,
        name: str | None = None,
        **kwargs: Any,
    ):
        if tool_node is None:
            raise ValueError("ToolGraphNode requires a non-null 'tool_node'.")
        super().__init__(name or tool_node.name, **kwargs)
        self.tool_node = tool_node

