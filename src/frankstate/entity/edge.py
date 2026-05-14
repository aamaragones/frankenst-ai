from collections.abc import Hashable
from typing import Literal

from frankstate.entity.statehandler import StateEvaluator


class BaseEdge:
    """Base edge definition storing the source node name."""

    def __init__(self, node_source: str | Literal["START", "END"]):
        self.node_source = node_source

class SimpleEdge(BaseEdge):
    """Static edge definition used with StateGraph.add_edge."""

    def __init__(
        self, 
        node_source: str | Literal["START", "END"],
        node_path: str | Literal["START", "END"],
    ):
        super().__init__(node_source)
        self.node_path = node_path


class ConditionalEdge(BaseEdge):
    """Conditional edge definition used with StateGraph.add_conditional_edges."""

    def __init__(
        self,
        node_source: str | Literal["START", "END"],
        map_dict: dict[Hashable, str | Literal["START", "END"]],
        evaluator: StateEvaluator,
    ):
        super().__init__(node_source)
        self.map_dict = map_dict
        self.evaluator = evaluator