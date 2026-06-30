from typing import cast

import pytest

from frankstate.entity.edge import SimpleEdge
from frankstate.entity.graph_layout import GraphLayout
from frankstate.entity.node import SimpleNode
from tests.support.frankstate_doubles.builders import FakeRunnableBuilder
from tests.support.frankstate_doubles.stub import StaticMessageEnhancer


class OrderedLayout(GraphLayout):
    PRIMARY_BUILDER: FakeRunnableBuilder
    SECONDARY_BUILDER: FakeRunnableBuilder

    def __init__(self) -> None:
        super().__init__()
        self.runtime_calls = 0
        self.layout_calls = 0

    def build_runtime(self) -> dict[str, FakeRunnableBuilder]:
        self.runtime_calls += 1
        return {
            "PRIMARY_BUILDER": FakeRunnableBuilder(async_result={"content": "first"}),
            "SECONDARY_BUILDER": FakeRunnableBuilder(async_result={"content": "second"}),
        }

    def layout(self) -> None:
        self.layout_calls += 1
        self.FIRST_NODE = SimpleNode(
            enhancer=StaticMessageEnhancer("first"),
            name="first_node",
            metadata={"tags": ["first"]},
        )
        self.SECOND_NODE = SimpleNode(
            enhancer=StaticMessageEnhancer("second"),
            name="second_node",
            metadata={"tags": ["second"]},
        )
        self.FIRST_EDGE = SimpleEdge(node_source="first_node", node_path="second_node")
        self.SECOND_EDGE = SimpleEdge(node_source="second_node", node_path="END")


@pytest.mark.unit
def test_build_runtime_requires_dict() -> None:
    class InvalidRuntimeLayout(GraphLayout):
        def build_runtime(self) -> dict[str, str]:
            return []  # type: ignore[return-value]

        def layout(self) -> None:
            pass

    with pytest.raises(TypeError, match="must return dict"):
        InvalidRuntimeLayout().get_nodes()


@pytest.mark.unit
def test_build_runtime_rejects_runtime_keys_without_annotation() -> None:
    class UnexpectedRuntimeLayout(GraphLayout):
        EXPECTED: str

        def build_runtime(self) -> dict[str, str]:
            return {"EXPECTED": "ok", "EXTRA": "boom"}

        def layout(self) -> None:
            pass

    with pytest.raises(ValueError, match="without class annotations"):
        UnexpectedRuntimeLayout().get_nodes()


@pytest.mark.unit
def test_build_runtime_rejects_missing_annotated_runtime_keys() -> None:
    class MissingRuntimeLayout(GraphLayout):
        EXPECTED: str

        def build_runtime(self) -> dict[str, str]:
            return {}

        def layout(self) -> None:
            pass

    with pytest.raises(ValueError, match="must populate all annotated runtime keys"):
        MissingRuntimeLayout().get_nodes()


@pytest.mark.unit
def test_layout_must_not_return_value() -> None:
    class ReturningLayout(GraphLayout):
        def build_runtime(self) -> dict[str, str]:
            return {}

        def layout(self) -> None:
            return "unexpected"  # type: ignore[return-value]

    with pytest.raises(TypeError, match="must not return a value"):
        ReturningLayout().get_nodes()


@pytest.mark.unit
def test_declared_runtime_keys_ignore_inherited_annotations() -> None:
    layout = OrderedLayout()

    assert layout._get_declared_runtime_keys() == {
        "PRIMARY_BUILDER",
        "SECONDARY_BUILDER",
    }


@pytest.mark.unit
def test_getters_preserve_order_and_build_once() -> None:
    layout = OrderedLayout()

    nodes = layout.get_nodes()
    edges = layout.get_edges()
    builders = layout.get_runnable_builders()

    assert layout.runtime_calls == 1
    assert layout.layout_calls == 1
    assert [node.name for node in nodes] == ["first_node", "second_node"]
    assert [(edge.node_source, cast(SimpleEdge, edge).node_path) for edge in edges] == [
        ("first_node", "second_node"),
        ("second_node", "END"),
    ]
    assert builders == [layout.PRIMARY_BUILDER, layout.SECONDARY_BUILDER]


@pytest.mark.unit
def test_get_runnable_builder_returns_named_builder() -> None:
    layout = OrderedLayout()

    builder = layout.get_runnable_builder("PRIMARY_BUILDER")

    assert builder is layout.PRIMARY_BUILDER


@pytest.mark.unit
def test_get_runnable_builder_rejects_unknown_name() -> None:
    layout = OrderedLayout()

    with pytest.raises(KeyError, match="does not expose a RunnableBuilder"):
        layout.get_runnable_builder("MISSING_BUILDER")
