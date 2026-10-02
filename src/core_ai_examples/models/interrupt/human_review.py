"""What the human-review pause shows, and what it accepts back."""

from typing import Literal, Self, TypedDict

from langchain_core.messages.tool import ToolCall
from pydantic import BaseModel, model_validator


class HumanReviewRequest(TypedDict):
    """The interrupt payload: the question asked and the tool call it is about."""

    question: str
    tool_call: ToolCall


class HumanReviewDecision(BaseModel):
    """The resume value; LangGraph validates it against this schema before resuming."""

    action: Literal["continue", "feedback"]
    data: str | None = None

    @model_validator(mode="after")
    def _feedback_carries_data(self) -> Self:
        if self.action == "feedback" and not self.data:
            raise ValueError("'feedback' needs 'data': the message the agent gets back")
        return self
