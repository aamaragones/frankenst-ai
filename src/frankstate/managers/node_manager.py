"""NodeManager: stores graph nodes and exposes them in StateGraph format."""

import logging
from collections.abc import Iterable
from typing import Any

from frankstate.entity.node import CommandNode, SimpleNode, ToolGraphNode


class NodeManager:
    """Store graph node definitions and expose them in `StateGraph` format.

    The manager accepts `SimpleNode`, `CommandNode` and `ToolGraphNode`
    wrappers. During configuration it resolves each node to the callable or
    runnable consumed by `StateGraph.add_node()`.

    Node names are treated as a LangGraph contract invariant: registration keeps
    insertion order and rejects duplicate names before delegating to LangGraph.
    """

    logger: logging.Logger = logging.getLogger(__name__)
    
    def __init__(self) -> None:
        self.nodes: dict[str, SimpleNode | CommandNode | ToolGraphNode] = {}
        self.logger.info("NodeManager initialized")
    
    def _normalize_nodes(
        self,
        nodes: SimpleNode | CommandNode | ToolGraphNode | Iterable[SimpleNode | CommandNode | ToolGraphNode],
    ) -> list[SimpleNode | CommandNode | ToolGraphNode]:
        """Return nodes as a list while supporting single-node inputs."""
        if isinstance(nodes, SimpleNode | CommandNode | ToolGraphNode):
            return [nodes]

        return list(nodes)

    def _get_node_value(self, node: SimpleNode | CommandNode | ToolGraphNode) -> Any:
        """Resolve a node wrapper to the callable or runnable added to the graph."""
        if isinstance(node, ToolGraphNode):
            return node.tool_node
        elif isinstance(node, SimpleNode):
            return node.enhancer.enhance
        elif isinstance(node, CommandNode):
            return node.commander.command
        else:
            raise TypeError(f"Unexpected node type: {type(node)}")

    def _get_node_kwargs(self, node: SimpleNode | CommandNode | ToolGraphNode) -> dict[str, Any]:
        """Return keyword arguments forwarded verbatim to `StateGraph.add_node()`.

        Wrapper `kwargs` (for example `metadata`, `retry_policy` or `timeout`)
        pass through unchanged. For `CommandNode`, `destinations` is sourced from
        the commander and wins over passthrough kwargs to avoid drifting from the
        runtime routing contract.
        """
        kwargs = dict(node.kwargs)

        if isinstance(node, CommandNode):
            node_destinations = node.destinations
            if "destinations" in kwargs and kwargs["destinations"] != node_destinations:
                raise ValueError(
                    f"Node '{node.name}' defines destinations both in commander and kwargs with different values"
                )
            kwargs["destinations"] = node_destinations

        return kwargs

    def add_nodes(
        self,
        nodes: SimpleNode | CommandNode | ToolGraphNode | Iterable[SimpleNode | CommandNode | ToolGraphNode],
    ) -> None:
        """Add one or more supported node instances to the internal registry.

        Accepted inputs are `SimpleNode`, `CommandNode`, `ToolGraphNode` or a
        list containing any mix of those types.
        """
        for node in self._normalize_nodes(nodes):
            if isinstance(node, SimpleNode | CommandNode | ToolGraphNode):
                if node.name in self.nodes:
                    raise ValueError(f"Node name '{node.name}' is already registered")
                self.nodes[node.name] = node
            else:
                raise TypeError(f"Unexpected node type: {type(node)}")

    def get_nodes(self) -> tuple[SimpleNode | CommandNode | ToolGraphNode, ...]:
        """Retrieve all registered nodes preserving insertion order."""
        return tuple(self.nodes.values())

    def configs_nodes(self) -> tuple[tuple[tuple[str, Any], dict[str, Any]], ...]:
        """Retrieve deterministic `((name, callable), kwargs)` pairs for `add_node()`.

        The first tuple mirrors the positional part of `StateGraph.add_node()`.
        The kwargs dictionary mirrors keyword arguments forwarded to LangGraph,
        including `frankstate`-managed `metadata` and `destinations`.

        This keeps the wrapper close to LangGraph's native calling convention:
        edges stay positional, while nodes can evolve with additional kwargs
        without requiring a new `frankstate` config class for every upstream change.

        The returned callable may be synchronous or asynchronous. LangGraph
        accepts both forms for node execution.
        """
        return tuple(
            (
                (name, self._get_node_value(node)),
                self._get_node_kwargs(node),
            )
            for name, node in self.nodes.items()
        )

    def remove_node(self, node: str | SimpleNode | CommandNode | ToolGraphNode) -> None:
        """Remove a registered node by its runtime name.

        The removal contract is name-based, not object-identity-based. Callers
        may pass either the node instance or the registered node name.
        """
        node_name = node if isinstance(node, str) else node.name

        if node_name in self.nodes:
            self.nodes.pop(node_name)
        else:
            raise ValueError(f"Node name '{node_name}' is not registered")
