from typing import Any

import pytest
from langchain_core.messages import AIMessage

from core_ai_examples.components.nodes.commands import (
    human_review_sensitive_tool_call as module,
)
from core_ai_examples.components.tools.dominate_pokemon.dominate_pokemon_tool import (
    DominatePokemonTool,
)
from core_ai_examples.models.interrupt.human_review import HumanReviewDecision
from tests.support.core_ai_examples_doubles import tool_call

pytestmark = pytest.mark.unit

DESTINATIONS = {"tools": "OakTools", "enhancer": "OakLangAgent"}


class _InterruptRecorder:
    def __init__(self, decision: HumanReviewDecision) -> None:
        self.decision = decision
        self.calls: list[tuple[Any, Any]] = []

    def __call__(self, value: Any, *, response_schema: Any = None) -> Any:
        self.calls.append((value, response_schema))
        return self.decision


def _node() -> module.HumanReviewSensitiveToolCall:
    return module.HumanReviewSensitiveToolCall(
        sensitive_tools=[DominatePokemonTool()], destinations=DESTINATIONS
    )


def _state(*calls: Any) -> dict[str, Any]:
    return {"messages": [AIMessage(content="", tool_calls=list(calls))]}


def test_non_sensitive_calls_go_straight_to_the_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _InterruptRecorder(HumanReviewDecision(action="continue"))
    monkeypatch.setattr(module, "interrupt", recorder)

    command = _node().command(
        _state(tool_call("get_evolution", {"pokemon_name": "a"}, "c1"))
    )

    assert command.goto == "OakTools"
    assert recorder.calls == []


def test_a_sensitive_call_interrupts_with_the_request_and_the_decision_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _InterruptRecorder(HumanReviewDecision(action="continue"))
    monkeypatch.setattr(module, "interrupt", recorder)
    call = tool_call("dominate_pokemon", {"place": "Ireland"}, "c1")

    command = _node().command(_state(call))

    ((value, schema),) = recorder.calls
    assert schema is HumanReviewDecision
    assert value == {"question": value["question"], "tool_call": call}
    assert "Ireland" in value["question"]
    assert command.goto == "OakTools"


def test_feedback_answers_every_pending_call_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = HumanReviewDecision(action="feedback", data="not that place")
    monkeypatch.setattr(module, "interrupt", _InterruptRecorder(decision))
    calls = [
        tool_call("get_evolution", {"pokemon_name": "a"}, "c1"),
        tool_call("dominate_pokemon", {"place": "Ireland"}, "c2"),
        tool_call("dominate_pokemon", {"place": "Iceland"}, "c3"),
    ]

    command = _node().command(_state(*calls))

    assert command.goto == "OakLangAgent"
    assert command.update is not None
    assert [(m["tool_call_id"], m["content"]) for m in command.update["messages"]] == [
        ("c1", ""),
        ("c2", "not that place"),
        ("c3", ""),
    ]
