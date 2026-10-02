import asyncio
from collections.abc import Callable
from typing import cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from services.llm.llm_services import LLMRuntime
from tests.support.core_ai_examples_doubles import ScriptedToolCallingModel, tool_call

pytest.importorskip("fastmcp", reason="needs the mcp extra")
from services.mcp import server_oaklang_agent as server

pytestmark = [pytest.mark.unit, pytest.mark.mcp]

ADMIN = {"x-custom-header": "admin"}


def _publish(published_runtime: Callable[..., LLMRuntime], *script: AIMessage) -> None:
    published_runtime(low_model=cast(BaseChatModel, ScriptedToolCallingModel(*script)))


def test_the_header_check_rejects_other_callers(
    published_runtime: Callable[..., LLMRuntime],
) -> None:
    _publish(published_runtime, AIMessage(content="done"))

    assert asyncio.run(server.run_oaklang_agent("hi", {})).startswith("Unauthorized")


def test_a_plain_answer_is_returned(
    published_runtime: Callable[..., LLMRuntime],
) -> None:
    _publish(published_runtime, AIMessage(content="done"))

    assert asyncio.run(server.run_oaklang_agent("hi", ADMIN)) == "done"


def test_a_sensitive_tool_call_returns_the_review_question(
    published_runtime: Callable[..., LLMRuntime],
) -> None:
    _publish(
        published_runtime,
        AIMessage(
            content="",
            tool_calls=[tool_call("dominate_pokemon", {"place": "Ireland"}, "c1")],
        ),
    )

    answer = asyncio.run(server.run_oaklang_agent("dominate Ireland", ADMIN))

    assert "Ireland" in answer and answer.endswith("?")


def test_the_mcp_tools_are_registered_under_their_names() -> None:
    names = {tool.name for tool in asyncio.run(server.mcp.list_tools())}

    assert names == {"handoff_oaklang_agent", "adaptive_rag_tool"}
