import importlib
import sys

from langchain_core.messages import AIMessage, HumanMessage

from core_examples.components.runnables.oaklang_agent.oaklang_agent import OakLangAgent
from core_examples.components.tools.get_evolution.get_evolution_tool import (
    GetEvolutionTool,
)
from core_examples.components.tools.random_movements.random_movements_tool import (
    RandomMovementsTool,
)
from core_examples.constants import CONFIG_FILE_PATH
from core_examples.utils.common import (
    load_and_clean_text_file,
    resolve_package_resource,
)
from core_examples.utils.config_loader import read_yaml
from tests.support.core_doubles import ToolBindingFakeModel


def test_oaklang_agent_build_prompt_loads_runtime_prompt_assets() -> None:
    agent = OakLangAgent(
        model=ToolBindingFakeModel(),
        tools=[GetEvolutionTool()],
    )

    prompt = agent._build_prompt()
    formatted = prompt.invoke({"messages": [HumanMessage(content="Hi Oak")]})

    assert prompt.input_variables == ["messages"]
    assert len(formatted.messages) == 3
    assert formatted.messages[0].type == "system"
    assert "Professor Oak" in formatted.messages[1].content
    assert formatted.messages[-1].content == "Hi Oak"


def test_oaklang_agent_build_prompt_loads_runtime_prompt_assets_outside_repo_root(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    agent = OakLangAgent(
        model=ToolBindingFakeModel(),
        tools=[GetEvolutionTool()],
    )

    prompt = agent._build_prompt()
    formatted = prompt.invoke({"messages": [HumanMessage(content="Hi Oak")]})

    assert formatted.messages[0].type == "system"
    assert "Professor Oak" in formatted.messages[1].content
    assert formatted.messages[-1].content == "Hi Oak"


def test_core_config_paths_resolve_outside_repo_root(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("core_examples.constants", None)

    core_constants = importlib.import_module("core_examples.constants")

    assert core_constants.CONFIG_FILE_PATH.is_file()
    assert core_constants.CONFIG_NODES_FILE_PATH.is_file()


def test_core_config_can_be_read_as_package_resource_outside_repo_root(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    config = read_yaml(CONFIG_FILE_PATH)

    assert config["launch"]["model"] == "ollama"
    assert config["ollama"]["model"]["model"] == "gemma4:e4b"


def test_oaklang_prompt_resource_can_be_loaded_without_module_file_paths(monkeypatch, tmp_path) -> None:
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
    agent = OakLangAgent(model=fake_model, tools=tools)

    result = agent.invoke({"messages": [HumanMessage(content="Hi Oak")]})

    assert isinstance(result, AIMessage)
    assert result.content == "oak-generated-response"
    assert [tool.name for tool in fake_model.bound_tools] == [
        "GetEvolutionTool",
        "RandomMovementsTool",
    ]
