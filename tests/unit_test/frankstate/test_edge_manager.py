import pytest

from frankstate.entity.edge import ConditionalEdge, SimpleEdge
from frankstate.managers.edge_manager import EdgeManager
from tests.support.frankstate_doubles.stub import FieldRouteEvaluator


@pytest.mark.unit
def test_add_edges_accepts_single_and_iterable_preserving_order() -> None:
    manager = EdgeManager()
    first = SimpleEdge(node_source="a", node_path="b")
    second = SimpleEdge(node_source="b", node_path="c")

    manager.add_edges(first)
    manager.add_edges([second])

    assert manager.get_edges() == (first, second)


@pytest.mark.unit
def test_add_edges_rejects_invalid_types() -> None:
    manager = EdgeManager()

    with pytest.raises(TypeError, match="SimpleEdge or ConditionalEdge"):
        manager.add_edges("invalid")  # type: ignore[arg-type]


@pytest.mark.unit
def test_get_edges_filters_by_exact_edge_type() -> None:
    manager = EdgeManager()
    static_edge = SimpleEdge(node_source="a", node_path="b")
    conditional_edge = ConditionalEdge(
        node_source="b",
        map_dict={"accept": "c"},
        evaluator=FieldRouteEvaluator(),
    )

    manager.add_edges([static_edge, conditional_edge])

    assert manager.get_edges(SimpleEdge) == (static_edge,)
    assert manager.get_edges(ConditionalEdge) == (conditional_edge,)


@pytest.mark.unit
def test_get_edges_rejects_invalid_filter_type() -> None:
    manager = EdgeManager()

    with pytest.raises(TypeError, match="expected"):
        manager.get_edges(str)  # type: ignore[arg-type]


@pytest.mark.unit
def test_configs_edges_and_conditional_edges_match_langgraph_shape() -> None:
    manager = EdgeManager()
    static_edge = SimpleEdge(node_source="a", node_path="b")
    conditional_edge = ConditionalEdge(
        node_source="b",
        map_dict={"accept": "c", "reject": "d"},
        evaluator=FieldRouteEvaluator(),
    )

    manager.add_edges([static_edge, conditional_edge])
    static_configs = manager.configs_edges()
    conditional_configs = manager.configs_conditional_edges()

    assert static_configs == (("a", "b"),)
    assert conditional_configs[0][0] == "b"
    assert conditional_configs[0][2] == {"accept": "c", "reject": "d"}
    assert conditional_configs[0][1]({"route": "accept"}) == "accept"
