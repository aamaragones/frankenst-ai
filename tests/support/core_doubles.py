from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


class ToolBindingFakeModel:
    def __init__(self, response_content: str = "oak-response") -> None:
        self.response_content = response_content
        self.bound_tools: list[Any] = []

    def bind_tools(self, tools: Sequence[Any]) -> RunnableLambda[Any, AIMessage]:
        self.bound_tools = list(tools)
        return RunnableLambda(lambda _: AIMessage(content=self.response_content))