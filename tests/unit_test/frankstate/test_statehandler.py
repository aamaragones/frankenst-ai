import asyncio
import inspect

import pytest

from frankstate.entity.statehandler import StateEnhancer, StateEvaluator
from tests.support.frankstate_doubles.builders import FakeRunnableBuilder
from tests.support.frankstate_doubles.stub import (
    AsyncFieldRouteEvaluator,
    FieldRouteEvaluator,
    MissingDestinationsCommander,
    RoutingCommander,
    RunnableMessageEnhancer,
    SyncRunnableMessageEnhancer,
)


@pytest.mark.unit
def test_state_enhancer_injects_runnable_and_kwargs() -> None:
    builder = FakeRunnableBuilder(async_result={"content": "from-runnable"})
    enhancer = RunnableMessageEnhancer(runnable_builder=builder, marker="seen")

    result = asyncio.run(enhancer.enhance({"messages": []}))

    assert enhancer.runnable is not None
    assert enhancer.marker == "seen"
    assert builder.configure_calls == 1
    assert result["messages"][-1].content == "from-runnable"


@pytest.mark.unit
def test_state_enhancer_supports_sync_handlers() -> None:
    builder = FakeRunnableBuilder(sync_result={"content": "from-sync-runnable"})
    enhancer = SyncRunnableMessageEnhancer(runnable_builder=builder, marker="seen")

    result = enhancer.enhance({"messages": []})

    assert enhancer.runnable is not None
    assert enhancer.marker == "seen"
    assert builder.configure_calls == 1
    assert result["messages"][-1].content == "from-sync-runnable"


@pytest.mark.unit
def test_state_evaluator_injects_kwargs() -> None:
    evaluator = FieldRouteEvaluator(field="decision", marker="seen")

    assert evaluator.field == "decision"
    assert evaluator.marker == "seen"
    assert evaluator.evaluate({"decision": "accept"}) == "accept"


@pytest.mark.unit
def test_state_evaluator_supports_async_handlers() -> None:
    evaluator = AsyncFieldRouteEvaluator(field="decision", marker="seen")

    result = asyncio.run(evaluator.evaluate({"decision": "accept"}))

    assert evaluator.field == "decision"
    assert evaluator.marker == "seen"
    assert result == "accept"


@pytest.mark.unit
def test_statehandler_base_contracts_are_not_async_only() -> None:
    assert not inspect.iscoroutinefunction(StateEvaluator.evaluate)
    assert not inspect.iscoroutinefunction(StateEnhancer.enhance)


@pytest.mark.unit
def test_state_commander_returns_command_with_update() -> None:
    commander = RoutingCommander(destinations={"accept": "accept_node", "reject": "reject_node"})

    command = commander.command({"decision": "reject"})

    assert command.goto == "reject_node"
    assert command.update is not None
    assert command.update["decision"] == "reject"
    assert command.update["messages"][-1].content == "command:reject"


@pytest.mark.unit
def test_state_commander_destinations_returns_backing_mapping() -> None:
    commander = RoutingCommander(destinations={"accept": "accept_node"})

    assert commander.destinations == {"accept": "accept_node"}


@pytest.mark.unit
def test_state_commander_destinations_requires_property_or_backing_attr() -> None:
    commander = MissingDestinationsCommander()

    with pytest.raises(AttributeError, match=r"must expose a 'destinations: dict\[str, str\]' property"):
        _ = commander.destinations