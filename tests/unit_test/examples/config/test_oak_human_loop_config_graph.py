import asyncio
from collections.abc import Callable
from typing import Any, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END
from langgraph.types import Command
from pydantic import ValidationError

from config.graph_layout import oak_human_loop_config_graph as layout_module
from config.settings import get_settings
from core_ai_examples.components.edges.evaluators.route_human_node import RouteHumanNode
from core_ai_examples.components.nodes.commands.human_review_sensitive_tool_call import (
    HumanReviewSensitiveToolCall,
)
from core_ai_examples.models.interrupt.human_review import HumanReviewDecision
from core_ai_examples.models.stategraph.stategraph import SharedState
from frankstate import WorkflowBuilder
from frankstate.entity.edge import ConditionalEdge, SimpleEdge
from frankstate.entity.node import CommandNode, SimpleNode, ToolGraphNode
from services.llm.llm_services import LLMRuntime
from tests.support.core_ai_examples_doubles import ScriptedToolCallingModel, tool_call
from utils.config_loader import load_node_registry

pytestmark = pytest.mark.unit

DOMINATE_IRELAND = AIMessage(
    content="",
    tool_calls=[tool_call("dominate_pokemon", {"place": "Ireland"}, "call-1")],
)
DONE = AIMessage(content="done")


def _registry() -> dict[str, dict[str, Any]]:
    return load_node_registry(get_settings().config_nodes_file_path)


def _publish(
    published_runtime: Callable[..., LLMRuntime], *script: AIMessage
) -> ScriptedToolCallingModel:
    model = ScriptedToolCallingModel(*(script or (DONE,)))
    published_runtime(low_model=cast(BaseChatModel, model))
    return model


def _compile(published_runtime: Callable[..., LLMRuntime], *script: AIMessage) -> Any:
    _publish(published_runtime, *script)
    return WorkflowBuilder(
        config=layout_module.OakHumanLoopConfigGraph, state_schema=SharedState
    ).compile(checkpointer=InMemorySaver())


def _run(graph: Any, payload: Any, thread: str) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread}}
    return cast(dict[str, Any], asyncio.run(graph.ainvoke(payload, config)))


def test_build_runtime_binds_the_tools_the_registry_declares(
    published_runtime: Callable[..., LLMRuntime],
) -> None:
    model = _publish(published_runtime)
    registry = _registry()

    runtime = layout_module.OakHumanLoopConfigGraph().build_runtime()

    assert set(runtime) == {"CONFIG_NODES", "OAKLANG_AGENT", "SENSITIVE_TOOLS"}
    assert runtime["OAKLANG_AGENT"].model is model
    assert [tool.name for tool in runtime["OAKLANG_AGENT"].tools] == (
        registry["OAKTOOLS_NODE"]["metadata"]["tools"]
    )
    assert [tool.name for tool in runtime["SENSITIVE_TOOLS"]] == (
        registry["HUMAN_REVIEW_NODE"]["metadata"]["sensitive_tools"]
    )


def test_layout_declares_nodes_and_edges_from_the_registry(
    published_runtime: Callable[..., LLMRuntime],
) -> None:
    _publish(published_runtime)
    registry = _registry()
    layout = layout_module.OakHumanLoopConfigGraph()

    nodes, edges = layout.get_nodes(), layout.get_edges()

    assert [node.name for node in nodes] == [
        registry[key]["name"]
        for key in ("OAKLANG_NODE", "OAKTOOLS_NODE", "HUMAN_REVIEW_NODE")
    ]
    assert isinstance(nodes[0], SimpleNode)
    assert isinstance(nodes[1], ToolGraphNode)
    assert isinstance(nodes[2], CommandNode)
    assert isinstance(nodes[2].commander, HumanReviewSensitiveToolCall)
    assert (
        nodes[2].commander.destinations == registry["HUMAN_REVIEW_NODE"]["destinations"]
    )
    assert nodes[1].kwargs == {"metadata": registry["OAKTOOLS_NODE"]["metadata"]}

    assert len(edges) == 3
    assert isinstance(edges[0], SimpleEdge) and isinstance(edges[1], SimpleEdge)
    assert isinstance(edges[2], ConditionalEdge)
    assert isinstance(edges[2].evaluator, RouteHumanNode)
    assert edges[2].map_dict == {
        "end": END,
        "review": registry["HUMAN_REVIEW_NODE"]["name"],
    }


def test_build_runtime_requires_the_low_model_runtime(
    published_runtime: Callable[..., LLMRuntime],
) -> None:
    published_runtime(model=cast(BaseChatModel, ScriptedToolCallingModel(DONE)))

    with pytest.raises(RuntimeError, match=r"declares no runtime for \['low_model'\]"):
        layout_module.OakHumanLoopConfigGraph().build_runtime()


@pytest.mark.parametrize(
    ("node", "key"),
    [("OAKTOOLS_NODE", "tools"), ("HUMAN_REVIEW_NODE", "sensitive_tools")],
)
def test_a_declared_tool_nobody_provides_fails_build_runtime(
    published_runtime: Callable[..., LLMRuntime],
    monkeypatch: pytest.MonkeyPatch,
    node: str,
    key: str,
) -> None:
    _publish(published_runtime)
    registry = _registry()
    registry[node]["metadata"][key] = [*registry[node]["metadata"][key], "mega_evolve"]
    monkeypatch.setattr(layout_module, "load_node_registry", lambda _path: registry)

    with pytest.raises(LookupError, match="mega_evolve"):
        layout_module.OakHumanLoopConfigGraph().build_runtime()


def test_a_sensitive_tool_call_pauses_with_the_typed_request_and_schema(
    published_runtime: Callable[..., LLMRuntime],
) -> None:
    graph = _compile(published_runtime, DOMINATE_IRELAND, DONE)

    result = _run(graph, {"messages": [HumanMessage(content="dominate Ireland")]}, "t1")

    interrupt = result["__interrupt__"][0]
    assert interrupt.value["tool_call"]["name"] == "dominate_pokemon"
    assert "Ireland" in interrupt.value["question"]
    assert interrupt.response_schema == HumanReviewDecision.model_json_schema()


def test_resuming_with_continue_runs_the_sensitive_tool(
    published_runtime: Callable[..., LLMRuntime],
) -> None:
    graph = _compile(published_runtime, DOMINATE_IRELAND, DONE)
    _run(graph, {"messages": [HumanMessage(content="dominate Ireland")]}, "t2")

    result = _run(graph, Command(resume={"action": "continue"}), "t2")

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [m.name for m in tool_messages] == ["dominate_pokemon"]
    assert "youtube" in str(tool_messages[0].content)
    assert result["messages"][-1].content == "done"
    assert "__interrupt__" not in result


def test_resuming_with_feedback_answers_the_call_with_the_reviewer_text(
    published_runtime: Callable[..., LLMRuntime],
) -> None:
    graph = _compile(published_runtime, DOMINATE_IRELAND, DONE)
    _run(graph, {"messages": [HumanMessage(content="dominate Ireland")]}, "t3")

    result = _run(
        graph, Command(resume={"action": "feedback", "data": "I meant Iceland"}), "t3"
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [(m.name, m.content) for m in tool_messages] == [
        ("dominate_pokemon", "I meant Iceland")
    ]
    assert result["messages"][-1].content == "done"


@pytest.mark.parametrize("resume", [{"action": "maybe"}, {"action": "feedback"}])
def test_an_invalid_resume_is_rejected_by_the_schema(
    published_runtime: Callable[..., LLMRuntime], resume: dict[str, str]
) -> None:
    graph = _compile(published_runtime, DOMINATE_IRELAND, DONE)
    _run(graph, {"messages": [HumanMessage(content="dominate Ireland")]}, "t4")

    with pytest.raises(ValidationError):
        _run(graph, Command(resume=resume), "t4")


def test_a_turn_without_tool_calls_ends_without_review(
    published_runtime: Callable[..., LLMRuntime],
) -> None:
    graph = _compile(published_runtime, DONE)

    result = _run(graph, {"messages": [HumanMessage(content="hi")]}, "t5")

    assert result["messages"][-1].content == "done"
    assert "__interrupt__" not in result
