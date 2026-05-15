from typing import Any

from frankstate.entity.statehandler import StateCommander, StateEnhancer


class BaseNode:
    """Base named node definition consumed by GraphLayout and NodeManager.

    `tags` is the common user-facing annotation surface for `frankstate` nodes.
    LangGraph's native `ToolNode` already exposes `tags`, while `StateGraph.add_node()`
    expects generic keyword arguments such as `metadata`. `frankstate` keeps
    `tags` as the standard layout field for all node types and later projects it
    into `metadata["tags"]` for wrapper nodes during workflow assembly.

    `kwargs` stores future-facing keyword arguments that should be forwarded to
    `StateGraph.add_node()` without forcing `frankstate` to predefine every
    native option in its own constructor surface.
    """

    def __init__(self, name: str, tags: list[str] | None = None, kwargs: dict[str, Any] | None = None):
        self.name = name
        self.tags = tags
        self.kwargs = dict(kwargs) if kwargs else None

class SimpleNode(BaseNode):
    """Node wrapper for a StateEnhancer callable.

    Optional `kwargs` are passed through to `StateGraph.add_node()` by the
    workflow builder after `frankstate` merges its own `tags` convention.
    """

    def __init__(
        self,
        enhancer: StateEnhancer,
        name: str,
        tags: list[str] | None = None,
        kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(name, tags=tags, kwargs=kwargs)
        self.enhancer = enhancer

class CommandNode(BaseNode):
    """Node wrapper for a StateCommander callable returning Command.

    Optional `kwargs` are passed through to `StateGraph.add_node()`, but
    `destinations` remains controlled by the commander contract so graph
    rendering stays consistent with the `Command.goto` targets.
    """

    def __init__(
        self,
        commander: StateCommander,
        name: str,
        tags: list[str] | None = None,
        kwargs: dict[str, Any] | None = None,
    ):
        try:
            _ = commander.destinations
        except AttributeError as exc:
            raise ValueError(
                f"{type(commander).__name__} must expose a 'destinations: dict[str, str]' property "
                "or a constructor-populated '_destinations' attribute where values are the "
                "registered names of destination nodes. See StateCommander docstring for the convention."
            ) from exc
        super().__init__(name, tags=tags, kwargs=kwargs)
        self.commander = commander

    @property
    def destinations(self) -> tuple[str, ...]:
        """Return route targets for LangGraph graph rendering.

        Reads `commander.destinations` so that `add_node(destinations=...)` can
        draw edges without requiring a `Literal` annotation on the callable.
        This is only used for graph rendering and has no effect on graph execution.
        """
        return tuple(self.commander.destinations.values())
