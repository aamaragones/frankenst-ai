from typing import Any, Literal

from langchain_core.messages import AnyMessage
from pydantic import BaseModel

from frankstate.entity.statehandler import StateEvaluator


# NOTE: this is a class 'from langgraph.prebuilt import tools_condition'
class RouteToolCondition(StateEvaluator):
    """Route to the tool node when the latest AI message contains tool calls.

    Reads the most recent message from the provided state and returns one of the
    routing keys expected by the surrounding `ConditionalEdge.map_dict`.
    """

    def evaluate(
        self,
        state: list[AnyMessage] | dict[str, Any] | BaseModel,
        messages_key: str = "messages",
    ) -> Literal["end", "tools"]:
        if isinstance(state, list):
            ai_message = state[-1]
        elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
            ai_message = messages[-1]
        elif messages := getattr(state, messages_key, []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return "end"