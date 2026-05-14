import logging
from collections.abc import Hashable, Iterable
from typing import Any, Literal

from frankstate.entity.edge import ConditionalEdge, SimpleEdge


class EdgeManager:
    """Store graph edges and expose them in the format expected by LangGraph.

    The manager keeps both static and conditional edges, and provides separate
    deterministic sequences for `StateGraph.add_edge()` and
    `StateGraph.add_conditional_edges()`.

    Edge registration intentionally mirrors the declared layout order and does
    not silently deduplicate repeated entries.
    """

    logger: logging.Logger = logging.getLogger(__name__)
    
    def __init__(self):
        self.edges: list[SimpleEdge | ConditionalEdge] = []
        self.logger.info("EdgeManager initialized")

    def _normalize_edges(
        self,
        edges: SimpleEdge | ConditionalEdge | Iterable[SimpleEdge | ConditionalEdge],
    ) -> list[SimpleEdge | ConditionalEdge]:
        """Return edges as a list while supporting single-edge inputs."""
        if isinstance(edges, SimpleEdge | ConditionalEdge):
            return [edges]

        return list(edges)

    def add_edges(self, edges: SimpleEdge | ConditionalEdge | Iterable[SimpleEdge | ConditionalEdge]) -> None:
        """
        Add one or more edges to the registry preserving declaration order.
        """
        for edge in self._normalize_edges(edges):
            if isinstance(edge, SimpleEdge | ConditionalEdge):
                self.edges.append(edge)
            else:
                raise TypeError(f"Each edge must be a SimpleEdge or ConditionalEdge, got {type(edge)}")

    def get_edges(
        self,
        filter_type: type[SimpleEdge] | type[ConditionalEdge] | None = None,
    ) -> tuple[SimpleEdge | ConditionalEdge, ...] | tuple[SimpleEdge, ...] | tuple[ConditionalEdge, ...]:
        """
        Retrieve registered edges, optionally filtered by exact edge class.
        """
        if filter_type is None:
            return tuple(self.edges)
        elif isinstance(filter_type, type) and issubclass(filter_type, SimpleEdge | ConditionalEdge):
            return tuple(edge for edge in self.edges if type(edge) is filter_type)
        else:
            raise TypeError(f"Each edge must be a SimpleEdge or ConditionalEdge, expected {type(filter_type)}")

    def configs_edges(self) -> tuple[tuple[str, str], ...]:
        """
        Return ordered tuples of `(node_source, node_path)` for `StateGraph.add_edge()`.
        """
        return tuple(
            (edge.node_source, edge.node_path)
            for edge in self.edges if isinstance(edge, SimpleEdge)
        )
    
    def configs_conditional_edges(
        self,
    ) -> tuple[tuple[str, Any, dict[Hashable, str | Literal["START", "END"]]], ...]:
        """
        Return ordered tuples for `StateGraph.add_conditional_edges()`.

        The evaluator callable may be synchronous or asynchronous.
        """
        return tuple(
            (edge.node_source, edge.evaluator.evaluate, edge.map_dict)
            for edge in self.edges if isinstance(edge, ConditionalEdge)
        )
