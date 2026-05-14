from typing import Any, cast

from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from frankstate.entity.statehandler import StateCommander


class HumanReviewSensitiveToolCall(StateCommander):
    """Commander that inserts a human review step for sensitive tool calls.

    The node inspects the latest assistant message, pauses execution with
    `interrupt()` when a sensitive tool is detected, and then returns a
    LangGraph `Command` pointing either to the tool node or back to the agent
    node with feedback updates.

    Args:
        sensitive_tools: Tool instances that require explicit human approval.
        destinations: Mapping of semantic keys to concrete node names. Expected keys
            are ``"tools"`` and ``"enhancer"``. Injected by the layout so that
            this class stays free of registry reads.
    """

    def __init__(
        self,
        sensitive_tools: list[BaseTool] | None = None,
        destinations: dict[str, str] | None = None,
    ):
        self.sensitive_tool_names = [tool.name for tool in (sensitive_tools or [])]
        self._destinations = destinations or {}

    def command(self, state: list[AnyMessage] | dict[str, Any] | BaseModel) -> Command[str]:
        """Return a `Command` based on the human review decision.

        Reads:
            - `messages`

        Returns:
            - `Command(goto=...)` to continue directly with tools
            - `Command(goto=..., update={...})` to send feedback back to the agent
        """
        state = cast(dict[str, Any], state)
        last_message = cast(AIMessage, state["messages"][-1])
        tool_calls = last_message.tool_calls

        # Separate sensitive and non-sensitive tool calls
        sensitive_calls = [
            tool_call for tool_call in tool_calls
            if tool_call["name"] in self.sensitive_tool_names
        ]
        
        # If no sensitive tools, run all tools immediately
        if not sensitive_calls:
            return Command(goto=self.destinations["tools"])

        # Get the *first* sensitive tool call to ask for review and update
        tool_call = sensitive_calls[0]

        # Interrupt and ask human for feedback
        human_review = interrupt({
            "question": f"Are you sure you want to proceed with this sensitive action for {tool_call["args"]}?",
            "tool_call": tool_call,
        })

        review_action = human_review["action"]
        review_data = human_review.get("data")

        if review_action == "continue":
            return Command(goto=self.destinations["tools"])

        elif review_action == "feedback":
            # ToolMessage for sensitive tool feedback
            feedback_tool_message = {
                "role": "tool",
                "content": review_data,
                "name": tool_call["name"],
                "tool_call_id": tool_call["id"],
            }

            # Return in same order as original tool_calls
            all_tool_messages = []

            for call in tool_calls:
                if call["id"] == tool_call["id"]:
                    all_tool_messages.append(feedback_tool_message)
                else:
                    # passthrough (empty response for untouched tools)
                    all_tool_messages.append({
                        "role": "tool",
                        "content": "",
                        "name": call["name"],
                        "tool_call_id": call["id"],
                    })

            return Command(goto=self.destinations["enhancer"], update={"messages": all_tool_messages})

        raise ValueError(
            f"Unsupported human review action '{review_action}'. Expected: continue, feedback."
        )

