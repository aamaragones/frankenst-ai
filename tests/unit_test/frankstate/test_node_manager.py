import asyncio

import pytest
from langgraph.prebuilt import ToolNode

from frankstate.entity.node import CommandNode, SimpleNode, ToolGraphNode
from frankstate.managers.node_manager import NodeManager
from tests.support.frankstate_doubles.stub import (
    MissingDestinationsCommander,
    RoutingCommander,
    StaticMessageEnhancer,
    uppercase_text,
)


@pytest.mark.unit
def test_add_nodes_accepts_single_and_iterable_preserving_order() -> None:
    manager = NodeManager()
    first = SimpleNode(StaticMessageEnhancer("first"), name="first")
    second = SimpleNode(StaticMessageEnhancer("second"), name="second")

    manager.add_nodes(first)
    manager.add_nodes([second])

    assert tuple(node.name for node in manager.get_nodes()) == ("first", "second")


@pytest.mark.unit
def test_add_nodes_rejects_duplicate_names() -> None:
    manager = NodeManager()
    first = SimpleNode(StaticMessageEnhancer("first"), name="same")
    duplicate = SimpleNode(StaticMessageEnhancer("second"), name="same")

    manager.add_nodes(first)

    with pytest.raises(ValueError, match="already registered"):
        manager.add_nodes(duplicate)


@pytest.mark.unit
def test_add_nodes_rejects_invalid_types() -> None:
    manager = NodeManager()

    with pytest.raises(TypeError, match="Unexpected node type"):
        manager.add_nodes("invalid")  # type: ignore[arg-type]


@pytest.mark.unit
def test_command_node_requires_destinations_attribute() -> None:
    with pytest.raises(ValueError, match="destinations"):
        CommandNode(commander=MissingDestinationsCommander(), name="invalid")


@pytest.mark.unit
def test_node_rejects_unsupported_add_node_option_at_construction() -> None:
    with pytest.raises(TypeError, match="unsupported add_node option"):
        SimpleNode(StaticMessageEnhancer("simple"), name="simple_node", retry_polcy=None)


@pytest.mark.unit
def test_configs_nodes_resolve_wrapper_callables_metadata_and_destinations() -> None:
    manager = NodeManager()
    simple = SimpleNode(
        enhancer=StaticMessageEnhancer("simple"),
        name="simple_node",
        defer=True,
        metadata={"owner": "simple"},
    )
    command = CommandNode(
        commander=RoutingCommander(destinations={"accept": "final_node"}),
        name="command_node",
    )
    tool_node = ToolGraphNode(
        tool_node=ToolNode([uppercase_text], name="tool_node", tags=["tool"]),
        metadata={"tags": ["tool"]},
    )

    manager.add_nodes([simple, command, tool_node])
    configs = manager.configs_nodes()

    simple_args, simple_kwargs = configs[0]
    command_args, command_kwargs = configs[1]
    tool_args, tool_kwargs = configs[2]

    simple_name, simple_action = simple_args
    command_name, command_action = command_args
    tool_name, tool_action = tool_args

    assert simple_name == "simple_node"
    assert simple_kwargs == {
        "defer": True,
        "metadata": {"owner": "simple"},
    }
    assert asyncio.run(simple_action({"messages": []}))["messages"][-1].content == "simple"

    assert command_name == "command_node"
    assert command_kwargs == {
        "destinations": ("final_node",),
    }
    assert command_action({"decision": "accept"}).goto == "final_node"

    assert tool_name == "tool_node"
    assert tool_action is tool_node.tool_node
    assert tool_kwargs == {"metadata": {"tags": ["tool"]}}


@pytest.mark.unit
def test_configs_nodes_reject_conflicting_command_destinations_in_kwargs() -> None:
    manager = NodeManager()
    command = CommandNode(
        commander=RoutingCommander(destinations={"accept": "final_node"}),
        name="command_node",
        destinations=("other_node",),
    )

    manager.add_nodes(command)

    with pytest.raises(ValueError, match="destinations both in commander and kwargs"):
        manager.configs_nodes()


@pytest.mark.unit
def test_remove_node_deletes_registered_node() -> None:
    manager = NodeManager()
    node = SimpleNode(StaticMessageEnhancer("simple"), name="simple_node")
    manager.add_nodes(node)

    manager.remove_node(node)

    assert manager.get_nodes() == ()


@pytest.mark.unit
def test_remove_node_rejects_unknown_node() -> None:
    manager = NodeManager()
    node = SimpleNode(StaticMessageEnhancer("simple"), name="simple_node")

    with pytest.raises(ValueError, match="not registered"):
        manager.remove_node(node)


@pytest.mark.unit
def test_remove_node_accepts_registered_name() -> None:
    manager = NodeManager()
    node = SimpleNode(StaticMessageEnhancer("simple"), name="simple_node")
    manager.add_nodes(node)

    manager.remove_node("simple_node")

    assert manager.get_nodes() == ()
