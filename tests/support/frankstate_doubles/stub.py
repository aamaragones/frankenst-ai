from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.types import Command

from frankstate.entity.statehandler import StateCommander, StateEnhancer, StateEvaluator


class StaticMessageEnhancer(StateEnhancer):
    def __init__(self, message: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.message = message

    async def enhance(self, state: Any) -> dict[str, list[AIMessage]]:
        return {"messages": [AIMessage(content=self.message)]}


class RunnableMessageEnhancer(StateEnhancer):
    marker: str

    async def enhance(self, state: Any) -> dict[str, list[AIMessage]]:
        assert self.runnable is not None
        result = await self.runnable.ainvoke(state)
        if isinstance(result, dict) and "content" in result:
            content = result["content"]
        else:
            content = str(result)
        return {"messages": [AIMessage(content=content)]}


class SyncRunnableMessageEnhancer(StateEnhancer):
    marker: str

    def enhance(self, state: Any) -> dict[str, list[AIMessage]]:
        assert self.runnable is not None
        result = self.runnable.invoke(state)
        if isinstance(result, dict) and "content" in result:
            content = result["content"]
        else:
            content = str(result)
        return {"messages": [AIMessage(content=content)]}


class FieldRouteEvaluator(StateEvaluator):
    marker: str

    def __init__(self, field: str = "route", **kwargs: Any):
        super().__init__(**kwargs)
        self.field = field

    def evaluate(self, state: Any) -> str:
        if isinstance(state, dict):
            return str(state[self.field])
        return str(getattr(state, self.field))


class AsyncFieldRouteEvaluator(StateEvaluator):
    marker: str

    def __init__(self, field: str = "route", **kwargs: Any):
        super().__init__(**kwargs)
        self.field = field

    async def evaluate(self, state: Any) -> str:
        if isinstance(state, dict):
            return str(state[self.field])
        return str(getattr(state, self.field))


class ToolCallEvaluator(StateEvaluator):
    def evaluate(self, state: Any) -> str:
        last_message = state["messages"][-1]
        return "tools" if getattr(last_message, "tool_calls", None) else "end"


class RoutingCommander(StateCommander):
    def __init__(self, destinations: dict[str, str]):
        self._destinations = destinations

    def command(self, state: Any) -> Command[str]:
        decision = state.get("decision", "accept")
        return Command(
            goto=self.destinations[decision],
            update={
                "messages": [AIMessage(content=f"command:{decision}")],
                "decision": decision,
            },
        )


class MissingDestinationsCommander(StateCommander):
    def command(self, state: Any) -> Command[str]:
        return Command(goto="nowhere")


@tool
def uppercase_text(text: str) -> str:
    """Uppercase the provided text."""

    return text.upper()


class ToolCallingEnhancer(StateEnhancer):
    async def enhance(self, state: Any) -> dict[str, list[AIMessage]]:
        text = state.get("tool_text", "pikachu")
        return {
            "messages": [
                AIMessage(
                    content="tool-request",
                    tool_calls=[
                        {
                            "name": uppercase_text.name,
                            "args": {"text": text},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }


class ToolSummaryEnhancer(StateEnhancer):
    async def enhance(self, state: Any) -> dict[str, list[AIMessage]]:
        last_message = state["messages"][-1]
        return {"messages": [AIMessage(content=f"tool:{last_message.content}")]}
