from pathlib import Path
from typing import cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from core_examples.components.runnables.oaklang_agent.oaklang_agent import OakLangAgent
from core_examples.components.tools.get_evolution.get_evolution_tool import (
    GetEvolutionTool,
)
from core_examples.components.tools.random_movements.random_movements_tool import (
    RandomMovementsTool,
)
from core_examples.utils.common import (
    load_and_clean_text_file,
    resolve_package_resource,
)
from tests.support.core_doubles import ToolBindingFakeModel

pytestmark = pytest.mark.unit


def test_oaklang_agent_build_prompt_loads_runtime_prompt_assets() -> None:
    agent = OakLangAgent(
        model=cast(BaseChatModel, ToolBindingFakeModel()),
        tools=[GetEvolutionTool()],
    )

    prompt = agent._build_prompt()
    formatted = prompt.invoke({"messages": [HumanMessage(content="Hi Oak")]})
    messages = formatted.to_messages()

    assert prompt.input_variables == ["messages"]
    assert len(messages) == 3
    assert messages[0].type == "system"
    assert "Professor Oak" in messages[1].content
    assert messages[-1].content == "Hi Oak"


def test_oaklang_agent_build_prompt_loads_runtime_prompt_assets_outside_repo_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    agent = OakLangAgent(
        model=cast(BaseChatModel, ToolBindingFakeModel()),
        tools=[GetEvolutionTool()],
    )

    prompt = agent._build_prompt()
    formatted = prompt.invoke({"messages": [HumanMessage(content="Hi Oak")]})
    messages = formatted.to_messages()

    assert messages[0].type == "system"
    assert "Professor Oak" in messages[1].content
    assert messages[-1].content == "Hi Oak"


def test_oaklang_prompt_resource_can_be_loaded_without_module_file_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    context = load_and_clean_text_file(
        resolve_package_resource(
            "core_examples.components.runnables.oaklang_agent",
            "prompt",
            "context.md",
        ),
    )

    assert "Professor Oak" in context


def test_oaklang_agent_binds_tools_and_returns_model_response() -> None:
    fake_model = ToolBindingFakeModel(response_content="oak-generated-response")
    tools = [GetEvolutionTool(), RandomMovementsTool()]
    agent = OakLangAgent(model=cast(BaseChatModel, fake_model), tools=tools)

    result = agent.invoke({"messages": [HumanMessage(content="Hi Oak")]})

    assert isinstance(result, AIMessage)
    assert result.content == "oak-generated-response"
    assert [tool.name for tool in fake_model.bound_tools] == [
        "GetEvolutionTool",
        "RandomMovementsTool",
    ]
