import logging
from collections.abc import Iterable
from typing import Any

from langgraph.prebuilt import ToolNode

from frankstate.entity.node import BaseNode, CommandNode, SimpleNode


class NodeManager:
    """Store graph node definitions and expose them in `StateGraph` format.

    The manager accepts project nodes (`SimpleNode`, `CommandNode`) and native
    LangGraph `ToolNode` instances. During configuration it resolves each node
    to the callable consumed by `StateGraph.add_node()`.

    Node names are treated as a LangGraph contract invariant: registration keeps
    insertion order and rejects duplicate names before delegating to LangGraph.
    """

    logger: logging.Logger = logging.getLogger(__name__)
    
    def __init__(self):
        self.nodes: dict[str, SimpleNode | CommandNode | ToolNode] = {}
        self.logger.info("NodeManager initialized")
    
    def _normalize_nodes(
        self,
        nodes: SimpleNode | CommandNode | ToolNode | Iterable[SimpleNode | CommandNode | ToolNode],
    ) -> list[SimpleNode | CommandNode | ToolNode]:
        """Return nodes as a list while supporting single-node inputs."""
        if isinstance(nodes, SimpleNode | CommandNode | ToolNode):
            return [nodes]

        return list(nodes)

    def _get_node_value(self, node: SimpleNode | CommandNode | ToolNode) -> Any:
        """Resolve a node wrapper to the callable or ToolNode added to the graph."""
        if isinstance(node, ToolNode):
            return node
        elif isinstance(node, SimpleNode):
            return node.enhancer.enhance
        elif isinstance(node, CommandNode):
            return node.commander.command
        else:
            raise TypeError(f"Unexpected node type: {type(node)}")

    def _get_node_tags(self, node: SimpleNode | CommandNode | ToolNode) -> list[str] | None:
        """Return node tags for wrappers or native ToolNode instances.

        Wrapper nodes expose `tags` directly. Native `ToolNode` already uses the
        same attribute name in LangGraph.
        """
        return node.tags if isinstance(node, BaseNode) else getattr(node, "tags", None)

    def _get_node_kwargs(self, node: SimpleNode | CommandNode | ToolNode) -> dict[str, Any]:
        """Return keyword arguments mirrored from `StateGraph.add_node()`.

        `frankstate` keeps `tags` as the common layout field even though native
        `StateGraph.add_node()` expects `metadata`. This helper normalizes that
        discrepancy by merging wrapper/tool tags into `metadata["tags"]` so the
        layout API stays uniform across `SimpleNode`, `CommandNode`, and native
        `ToolNode`.

        For `CommandNode`, `destinations` is sourced from the commander and wins
        over passthrough kwargs to avoid drifting from the runtime routing contract.
        """
        kwargs = dict(node.kwargs) if isinstance(node, BaseNode) and node.kwargs else {}
        metadata = kwargs.pop("metadata", None)
        tags = self._get_node_tags(node)

        if tags:
            if metadata is not None and not isinstance(metadata, dict):
                raise TypeError("Node metadata must be a dict when tags are also provided")

            # Layouts always declare `tags`; wrapper nodes must project them into
            # LangGraph metadata because only native ToolNode exposes `tags` directly.
            kwargs["metadata"] = {
                **(metadata or {}),
                "tags": tags,
            }
        elif metadata is not None:
            kwargs["metadata"] = metadata

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
        nodes: SimpleNode | CommandNode | ToolNode | Iterable[SimpleNode | CommandNode | ToolNode],
    ) -> None:
        """
        Add one or more supported node instances to the internal registry.

        Accepted inputs are `SimpleNode`, `CommandNode`, `ToolNode` or a list
        containing any mix of those types.
        """
        for node in self._normalize_nodes(nodes):
            if isinstance(node, SimpleNode | CommandNode | ToolNode):
                if node.name in self.nodes:
                    raise ValueError(f"Node name '{node.name}' is already registered")
                self.nodes[node.name] = node
            else:
                raise TypeError(f"Unexpected node type: {type(node)}")

    def get_nodes(self) -> tuple[SimpleNode | CommandNode | ToolNode, ...]:
        """
        Retrieve all registered nodes preserving insertion order.
        """
        return tuple(self.nodes.values())

    def configs_nodes(self) -> tuple[tuple[tuple[str, Any], dict[str, Any]], ...]:
        """
        Retrieve deterministic `((name, callable), kwargs)` pairs for `add_node()`.

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

    def remove_node(self, node: str | SimpleNode | CommandNode | ToolNode) -> None:
        """Remove a registered node by its runtime name.

        The removal contract is name-based, not object-identity-based. Callers
        may pass either the node instance or the registered node name.
        """
        node_name = node if isinstance(node, str) else node.name

        if node_name in self.nodes:
            self.nodes.pop(node_name)
        else:
            raise ValueError(f"Node name '{node_name}' is not registered")
