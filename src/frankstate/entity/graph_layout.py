"""Base GraphLayout contract for declaring nodes, edges and runtime."""

import logging
from abc import ABC, abstractmethod
from typing import Any, get_type_hints

from frankstate.entity.edge import ConditionalEdge, SimpleEdge
from frankstate.entity.node import CommandNode, SimpleNode, ToolGraphNode
from frankstate.entity.runnable_builder import RunnableBuilder


class GraphLayout(ABC):
    """Base contract for all `frankstate` graph layouts.

    A layout is responsible for two concerns only:
    - build runtime dependencies without doing work at import time
    - declare concrete runnable builders, nodes and edges on the instance

    Validation and serialization into LangGraph remain delegated to the
    existing managers.
    """

    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        self.runtime: dict[str, Any] | None = None
        self._runtime_built: bool = False
        self._layout_built: bool = False

        self.logger.info(f"{self.__class__.__name__} initialized")

    @abstractmethod
    def build_runtime(self) -> dict[str, Any]:
        """Return the runtime values required to declare the layout."""

    @abstractmethod
    def layout(self) -> None:
        """Declare runnable builders, nodes and edges on the layout instance."""

    def _get_declared_runtime_keys(self) -> set[str]:
        """Return annotated attribute names declared by the concrete layout."""
        hints = get_type_hints(self.__class__)
        declared_annotations = self.__class__.__dict__.get("__annotations__", {})
        return {key for key in hints.keys() if key in declared_annotations}

    def _build_runtime(self) -> None:
        """Build and project runtime attributes once per layout instance."""
        if not self._runtime_built:
            runtime = self.build_runtime()
            if not isinstance(runtime, dict):
                raise TypeError(
                    f"{self.__class__.__name__}.build_runtime() must return dict[str, Any], got {type(runtime)}"
                )

            declared_keys = self._get_declared_runtime_keys()
            runtime_keys = set(runtime.keys())
            missing_annotations = sorted(runtime_keys - declared_keys)
            if missing_annotations:
                raise ValueError(
                    f"{self.__class__.__name__}.build_runtime() returned keys without class annotations: {missing_annotations}"
                )

            missing_runtime_keys = sorted(declared_keys - runtime_keys)
            if missing_runtime_keys:
                raise ValueError(
                    f"{self.__class__.__name__}.build_runtime() must populate all annotated runtime keys: {missing_runtime_keys}"
                )

            self.runtime = runtime
            for key, value in runtime.items():
                setattr(self, key, value)

            self._runtime_built = True

    def _build_layout(self) -> None:
        """Declare layout objects once per layout instance."""
        self._build_runtime()

        if not self._layout_built:
            result = self.layout()
            if result is not None:
                raise TypeError(
                    f"{self.__class__.__name__}.layout() must not return a value"
                )

            self._layout_built = True

    def _filter_attributes(self, expected_type: type | tuple[type, ...]) -> list[Any]:
        """Return instance attributes matching the requested runtime types."""
        self._build_layout()
        return [
            attr_value
            for attr_value in self.__dict__.values()
            if isinstance(attr_value, expected_type)
        ]

    def get_nodes(self) -> list[SimpleNode | CommandNode | ToolGraphNode]:
        """Return concrete nodes preserving the layout declaration order."""
        return self._filter_attributes((SimpleNode, CommandNode, ToolGraphNode))

    def get_edges(self) -> list[SimpleEdge | ConditionalEdge]:
        """Return concrete edges for the current layout instance."""
        return self._filter_attributes((SimpleEdge, ConditionalEdge))

    def get_runnable_builders(self) -> list[RunnableBuilder]:
        """Return runnable builders exposed by the layout.

        Builders are returned in the same declaration order in which the layout
        projected them onto the instance during `build_runtime()`.
        """
        return self._filter_attributes(RunnableBuilder)

    def get_runnable_builder(self, attribute_name: str) -> RunnableBuilder:
        """Return a named runnable builder exposed by the layout.

        Use this helper when the caller wants an explicit builder by attribute
        name instead of relying on declaration order.
        """
        self._build_layout()
        builder = getattr(self, attribute_name, None)
        if not isinstance(builder, RunnableBuilder):
            raise KeyError(
                f"{self.__class__.__name__} does not expose a RunnableBuilder named '{attribute_name}'"
            )
        return builder
