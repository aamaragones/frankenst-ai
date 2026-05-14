from typing import Any, Literal, cast

from langchain_core.messages import AIMessage, AnyMessage
from pydantic import BaseModel

from frankstate.entity.statehandler import StateEvaluator


class RouteHumanNode(StateEvaluator):
    """Route to human review when the latest message contains tool calls.

    Reads:
        - `messages`

    Returns:
        - `"review"` when the last assistant message includes tool calls
        - `"end"` when the graph can stop without human intervention
    """

    def evaluate(self, state: list[AnyMessage] | dict[str, Any] | BaseModel) -> Literal["end", "review"]:
        state = cast(dict[str, Any], state)
        last_message = cast(AIMessage, state["messages"][-1])
        if len(last_message.tool_calls) == 0:
            return "end"
        else:
            return "review"
