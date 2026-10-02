from typing import Any, cast

from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from core_ai_examples.models.interrupt.human_review import (
    HumanReviewDecision,
    HumanReviewRequest,
)
from frankstate.entity.statehandler import StateCommander


class HumanReviewSensitiveToolCall(StateCommander):
    """Commander that inserts a human review step for sensitive tool calls.

    The node inspects the latest assistant message and, when a sensitive tool is
    called, pauses with `interrupt()` carrying a `HumanReviewRequest`. The resume
    value is validated as a `HumanReviewDecision`; `continue` releases the tool
    calls, `feedback` answers them with the reviewer's message instead. The graph
    must be compiled with a checkpointer or the pause has nowhere to resume from.

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

    def command(
        self, state: list[AnyMessage] | dict[str, Any] | BaseModel
    ) -> Command[str]:
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

        sensitive_calls = [
            tool_call
            for tool_call in tool_calls
            if tool_call["name"] in self.sensitive_tool_names
        ]
        if not sensitive_calls:
            return Command(goto=self.destinations["tools"])

        # Only the first sensitive call is reviewed; the rest ride on the decision.
        tool_call = sensitive_calls[0]
        decision = interrupt(
            HumanReviewRequest(
                question=(
                    "Are you sure you want to proceed with this sensitive action "
                    f"for {tool_call['args']}?"
                ),
                tool_call=tool_call,
            ),
            response_schema=HumanReviewDecision,
        )

        if decision.action == "continue":
            return Command(goto=self.destinations["tools"])

        # Every pending call gets a tool message so the agent sees a complete turn.
        all_tool_messages = [
            {
                "role": "tool",
                "content": decision.data if call["id"] == tool_call["id"] else "",
                "name": call["name"],
                "tool_call_id": call["id"],
            }
            for call in tool_calls
        ]
        return Command(
            goto=self.destinations["enhancer"],
            update={"messages": all_tool_messages},
        )
