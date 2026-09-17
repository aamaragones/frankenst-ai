"""Offline doubles for the core_ai_examples tests."""

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda


class ToolBindingFakeModel:
    def __init__(self, response_content: str = "oak-response") -> None:
        self.response_content = response_content
        self.bound_tools: list[Any] = []

    def bind_tools(self, tools: Sequence[Any]) -> RunnableLambda[Any, AIMessage]:
        self.bound_tools = list(tools)
        return RunnableLambda(lambda _: AIMessage(content=self.response_content))


class FakeStructuredModel:
    """Chat-completions double: only `with_structured_output`, answering a stub schema."""

    def __init__(self, **response_fields: Any) -> None:
        self.response_fields = response_fields or {"binary_score": "yes"}
        self.captured_prompts: list[Any] = []
        self.structured_schema: Any = None
        self.structured_method: str | None = None

    def with_structured_output(self, schema: Any, method: str) -> Runnable[Any, Any]:
        self.structured_schema = schema
        self.structured_method = method

        def _respond(prompt_value: Any) -> Any:
            self.captured_prompts.append(prompt_value)
            return schema(**self.response_fields)

        return RunnableLambda(_respond)


class FakeResponsesStructuredModel:
    """Responses-route double: `bind(text=...)` only, since `with_structured_output` fails there."""

    use_responses_api = True

    def __init__(self, payload: str = '{"binary_score": "yes"}') -> None:
        self.payload = payload
        self.bound_kwargs: dict[str, Any] = {}
        self.captured_prompts: list[Any] = []

    def bind(self, **kwargs: Any) -> Runnable[Any, Any]:
        self.bound_kwargs = kwargs

        def _respond(prompt_value: Any) -> AIMessage:
            self.captured_prompts.append(prompt_value)
            return AIMessage(content=self.payload)

        return RunnableLambda(_respond)
