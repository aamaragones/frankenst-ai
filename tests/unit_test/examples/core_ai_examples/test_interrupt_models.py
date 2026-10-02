import pytest
from pydantic import ValidationError

from core_ai_examples.models.interrupt.human_review import (
    HumanReviewDecision,
    HumanReviewRequest,
)
from tests.support.core_ai_examples_doubles import tool_call

pytestmark = pytest.mark.unit


def test_continue_needs_no_data() -> None:
    assert HumanReviewDecision(action="continue").data is None


def test_feedback_requires_data() -> None:
    with pytest.raises(ValidationError, match="'feedback' needs 'data'"):
        HumanReviewDecision(action="feedback")


def test_unknown_actions_are_rejected() -> None:
    with pytest.raises(ValidationError):
        HumanReviewDecision.model_validate({"action": "maybe"})


def test_the_schema_exposes_the_two_actions() -> None:
    schema = HumanReviewDecision.model_json_schema()

    assert schema["properties"]["action"]["enum"] == ["continue", "feedback"]
    assert set(schema["properties"]) == {"action", "data"}


def test_the_request_carries_the_question_and_the_tool_call() -> None:
    call = tool_call("dominate_pokemon", {"place": "Ireland"}, "c1")
    request = HumanReviewRequest(question="sure?", tool_call=call)

    assert (
        set(request)
        == set(HumanReviewRequest.__annotations__)
        == {"question", "tool_call"}
    )
