from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import AnyMessage
from langchain_core.runnables import Runnable
from langgraph.types import Command
from pydantic import BaseModel

from frankstate.entity.runnable_builder import RunnableBuilder


class StateEvaluator(ABC):
    """Base contract for conditional routing in a LangGraph StateGraph.

    Implementations receive the current graph state and must return a routing key
    that exists in the corresponding `ConditionalEdge.map_dict`.

    Typical usage:
    - Read a subset of the current state.
    - Evaluate a condition.
    - Return a short routing key such as `"tools"`, `"rewrite"` or `"end"`.

    This contract mirrors the callable passed to `StateGraph.add_conditional_edges()`.
    Implementations may be synchronous (`def evaluate`) or asynchronous
    (`async def evaluate`) depending on whether they rely on `invoke()` or
    `ainvoke()` semantics from LangChain/LangGraph integrations.
    """

    def __init__(
        self,
        runnable_builder: RunnableBuilder | None = None,
        **kwargs: Any,

        ):
        self.runnable: Runnable[Any, Any] | None = runnable_builder.get() if runnable_builder else None

        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def evaluate(self, state: list[AnyMessage] | dict[str, Any] | BaseModel) -> str:
        """Return the routing key used by a conditional edge path map.

        The returned value must match one of the keys declared in the
        `ConditionalEdge.map_dict` for the edge that uses this evaluator.

        Concrete evaluators may be implemented as synchronous (`def evaluate`)
        or asynchronous (`async def evaluate`) handlers depending on whether
        they use `invoke()` or `ainvoke()`.
        """
        pass


class StateEnhancer(ABC):
    """Base contract for node callables that return partial state updates.

    This wrapper keeps the project API stable while matching the official
    LangGraph pattern where a node receives the current state and returns a
    partial update that LangGraph merges into the state.

    In the example graphs, enhancers commonly append new values to `messages`
    and may also add or update scalar keys required by the graph schema, such as
    `question`, `generation` or `iterations`.

    Implementations may be synchronous (`def enhance`) or asynchronous
    (`async def enhance`) depending on whether they call `invoke()` or
    `ainvoke()` on their runnable dependencies.
    """

    def __init__(
            self,
            runnable_builder: RunnableBuilder | None = None,
            **kwargs: Any,
        ):
        
        self.runnable: Runnable[Any, Any] | None = runnable_builder.get() if runnable_builder else None
         
        for key, value in kwargs.items():
            setattr(self, key, value)

        
        
    @abstractmethod
    def enhance(self, state: list[AnyMessage] | dict[str, Any] | BaseModel) -> dict[str, Any]:
        """Return a partial state update produced by runnable or custom enhance logic.

        The returned mapping is merged by LangGraph into the current state. The
        exact keys must be compatible with the graph state schema used when the
        workflow is compiled.

        Concrete enhancers may be implemented as synchronous (`def enhance`)
        or asynchronous (`async def enhance`) handlers depending on whether
        they use `invoke()` or `ainvoke()`.
        """
        pass

class StateCommander(ABC):
    """Base contract for nodes that route with LangGraph Command.

    Use this abstraction only when routing and state updates must happen in the
    same node callable, such as human-in-the-loop flows or supervisor patterns.

    Unlike `StateEvaluator`, a commander does not rely on a separate
    `ConditionalEdge`. It returns an official `Command` object with a `goto`
    destination and optional state updates.

    Subclasses must implement `command()` and inject their own routing
    information (e.g. via constructor arguments) rather than reading the node
    registry directly.

    Convention — ``destinations`` property:
        Subclasses should expose a ``destinations: dict[str, str]`` mapping
        semantic keys (e.g. ``"tools"``, ``"enhancer"``) to the
        ``BaseNode.name`` values of the destination nodes.

        `CommandNode` reads this property to populate
        ``StateGraph.add_node(destinations=...)`` without depending on dynamic
        attribute access.
    """

    @property
    def destinations(self) -> dict[str, str]:
        """Return the semantic route map used by CommandNode and LangGraph.

        Subclasses may override this property directly, or populate a backing
        `_destinations` attribute from their constructor.
        """
        destinations = getattr(self, "_destinations", None)

        if not isinstance(destinations, dict):
            raise AttributeError(
                f"{type(self).__name__} must expose a 'destinations: dict[str, str]' property "
                "or a '_destinations' attribute populated by the constructor."
            )

        return destinations

    @abstractmethod
    def command(self, state: list[AnyMessage] | dict[str, Any] | BaseModel) -> Command[str]:
        """Return a `Command` that routes the graph and optionally updates state.

        The `goto` value must match a node name registered in the compiled graph.
        Subclasses receive concrete route names through constructor injection so
        that this method stays free of YAML or registry reads.
        """
        pass