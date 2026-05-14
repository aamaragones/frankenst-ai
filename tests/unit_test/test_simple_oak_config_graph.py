import asyncio

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode

from core_examples.components.edges.evaluators.route_tool_condition import (
    RouteToolCondition,
)
from core_examples.components.nodes.enhancers.simple_messages_ainvoke import (
    SimpleMessagesAsyncInvoke,
)
from core_examples.components.runnables.oaklang_agent.oaklang_agent import OakLangAgent
from core_examples.config.layouts import simple_oak_config_graph as simple_oak_module
from core_examples.models.stategraph.stategraph import SharedState
from frankstate import WorkflowBuilder
from frankstate.entity.edge import ConditionalEdge, SimpleEdge
from frankstate.entity.node import SimpleNode
from tests.support.core_doubles import ToolBindingFakeModel


def _patch_simple_oak_runtime(monkeypatch, response_content: str = "oak-response") -> ToolBindingFakeModel:
    fake_model = ToolBindingFakeModel(response_content=response_content)

    def fake_launch() -> None:
        simple_oak_module.LLMServices.model = fake_model

    monkeypatch.setattr(simple_oak_module.LLMServices, "launch", fake_launch)
    return fake_model


def test_simple_oak_config_graph_build_runtime_resolves_all_runtime_dependencies(monkeypatch) -> None:
    fake_model = _patch_simple_oak_runtime(monkeypatch)
    layout = simple_oak_module.SimpleOakConfigGraph()

    runtime = layout.build_runtime()

    assert set(runtime) == {"CONFIG_NODES", "OAKLANG_AGENT"}
    assert runtime["CONFIG_NODES"]["OAKLANG_NODE"]["name"] == "OakLangAgent"
    assert runtime["CONFIG_NODES"]["OAKTOOLS_NODE"]["name"] == "OakTools"
    assert isinstance(runtime["OAKLANG_AGENT"], OakLangAgent)
    assert runtime["OAKLANG_AGENT"].model is fake_model
    assert [tool.name for tool in runtime["OAKLANG_AGENT"].tools] == [
        "GetEvolutionTool",
        "RandomMovementsTool",
    ]


def test_simple_oak_config_graph_declares_nodes_edges_and_runnable_builders(monkeypatch) -> None:
    _patch_simple_oak_runtime(monkeypatch)
    layout = simple_oak_module.SimpleOakConfigGraph()

    nodes = layout.get_nodes()
    edges = layout.get_edges()
    runnable_builders = layout.get_runnable_builders()

    assert [node.name for node in nodes] == ["OakLangAgent", "OakTools"]
    assert isinstance(nodes[0], SimpleNode)
    assert isinstance(nodes[0].enhancer, SimpleMessagesAsyncInvoke)
    assert isinstance(nodes[1], ToolNode)
    assert nodes[0].tags == ["Main agent node. It binds tools and produces the next assistant message."]
    assert nodes[1].tags == ["ToolNode executed when the OakLangAgent node emits tool calls."]

    assert len(edges) == 3
    assert isinstance(edges[0], SimpleEdge)
    assert isinstance(edges[1], SimpleEdge)
    assert isinstance(edges[2], ConditionalEdge)
    assert edges[0].node_path == "OakLangAgent"
    assert edges[1].node_source == "OakTools"
    assert isinstance(edges[2].evaluator, RouteToolCondition)
    assert edges[2].map_dict == {"end": "__end__", "tools": "OakTools"}

    assert runnable_builders == [layout.OAKLANG_AGENT]


def test_simple_oak_config_graph_compiles_and_runs_with_workflow_builder(monkeypatch) -> None:
    _patch_simple_oak_runtime(monkeypatch, response_content="oak-layout-response")
    builder = WorkflowBuilder(
        config=simple_oak_module.SimpleOakConfigGraph,
        state_schema=SharedState,
    )

    compiled = builder.compile()
    result = asyncio.run(compiled.ainvoke({"messages": [HumanMessage(content="Hi Oak")]}))

    assert result["messages"][-1].content == "oak-layout-response"
    assert set(compiled.get_graph().nodes) >= {
        "__start__",
        "OakLangAgent",
        "OakTools",
        "__end__",
    }
    assert compiled.get_graph().nodes["OakLangAgent"].metadata == {
        "tags": ["Main agent node. It binds tools and produces the next assistant message."]
    }