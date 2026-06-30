import asyncio
from typing import Any, cast

import pytest
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START
from langgraph.types import CachePolicy, RetryPolicy, TimeoutPolicy

import frankstate
from frankstate import WorkflowBuilder
from frankstate.entity.edge import ConditionalEdge, SimpleEdge
from frankstate.entity.graph_layout import GraphLayout
from frankstate.entity.node import CommandNode, SimpleNode
from frankstate.entity.runnable_builder import RunnableBuilder
from frankstate.entity.statehandler import StateCommander, StateEnhancer, StateEvaluator
from frankstate.managers.edge_manager import EdgeManager
from frankstate.managers.node_manager import NodeManager
from tests.support.frankstate_doubles.layouts import (
    CommandAsyncLayout,
    ConditionalAsyncEvaluatorLayout,
    ConditionalAsyncLayout,
    FrankTestState,
    LinearAsyncLayout,
    LinearSyncLayout,
    ToolLoopLayout,
)
from tests.support.frankstate_doubles.stub import StaticMessageEnhancer


@pytest.mark.unit
def test_public_api_shortcuts_are_importable() -> None:
    assert WorkflowBuilder is not None


@pytest.mark.unit
def test_frankstate_root_only_exposes_workflow_builder_shortcut() -> None:
    assert frankstate.__all__ == ["WorkflowBuilder"]
    assert frankstate.WorkflowBuilder is WorkflowBuilder
    assert not hasattr(frankstate, "GraphLayout")
    assert not hasattr(frankstate, "SimpleNode")
    assert not hasattr(frankstate, "SimpleEdge")
    assert not hasattr(frankstate, "StateEnhancer")
    assert not hasattr(frankstate, "NodeManager")


@pytest.mark.unit
def test_frankstate_reusable_contracts_are_imported_from_submodules() -> None:
    assert GraphLayout is not None
    assert SimpleNode is not None
    assert CommandNode is not None
    assert SimpleEdge is not None
    assert ConditionalEdge is not None
    assert StateEnhancer is not None
    assert StateEvaluator is not None
    assert StateCommander is not None
    assert RunnableBuilder is not None
    assert NodeManager is not None
    assert EdgeManager is not None


@pytest.mark.unit
def test_to_mermaid_returns_clean_offline_text_with_graph_nodes() -> None:
    builder = WorkflowBuilder(config=LinearSyncLayout, state_schema=FrankTestState)

    mermaid = builder.to_mermaid()

    assert isinstance(mermaid, str)
    assert mermaid.strip() != ""
    assert "linear_sync_node" in mermaid
    # tags are layout metadata and must not clutter the default diagram labels
    assert "linear-sync" not in mermaid


@pytest.mark.unit
def test_to_mermaid_with_metadata_keeps_node_tags() -> None:
    builder = WorkflowBuilder(config=LinearSyncLayout, state_schema=FrankTestState)

    mermaid = builder.to_mermaid(with_metadata=True)

    assert "linear_sync_node" in mermaid
    assert "linear-sync" in mermaid


@pytest.mark.unit
def test_workflow_builder_rejects_non_graphlayout_config() -> None:
    with pytest.raises(TypeError, match="GraphLayout subclass"):
        WorkflowBuilder(config=object, state_schema=FrankTestState)  # type: ignore[arg-type]


@pytest.mark.unit
def test_workflow_builder_compiles_linear_layout_once_and_runs_async_enhancer() -> None:
    builder = WorkflowBuilder(config=LinearAsyncLayout, state_schema=FrankTestState)

    first_compiled = builder.compile()
    second_compiled = builder.compile()
    result = asyncio.run(first_compiled.ainvoke({"messages": [HumanMessage(content="hi")]}))

    assert first_compiled is not None
    assert second_compiled is not None
    assert builder._workflow_configured is True
    assert cast(LinearAsyncLayout, builder.config).runtime_calls == 1
    assert cast(LinearAsyncLayout, builder.config).layout_calls == 1
    assert result["messages"][-1].content == "linear-response"
    assert first_compiled.get_graph().nodes["linear_node"].metadata == {"tags": ["linear"]}


@pytest.mark.unit
def test_workflow_builder_passes_node_kwargs_through_add_node() -> None:
    class NodeKwargsLayout(GraphLayout):
        def build_runtime(self) -> dict[str, object]:
            return {}

        def layout(self) -> None:
            self.KWARGS_NODE = SimpleNode(
                enhancer=StaticMessageEnhancer("kwargs-response"),
                name="kwargs_node",
                defer=True,
                metadata={"owner": "core", "tags": ["kwargs"]},
            )
            self.START_EDGE = SimpleEdge(node_source=START, node_path=self.KWARGS_NODE.name)
            self.END_EDGE = SimpleEdge(node_source=self.KWARGS_NODE.name, node_path=END)

    builder = WorkflowBuilder(config=NodeKwargsLayout, state_schema=FrankTestState)
    compiled = builder.compile()

    assert compiled.get_graph().nodes["kwargs_node"].metadata == {
        "owner": "core",
        "tags": ["kwargs"],
        "defer": True,
    }


@pytest.mark.unit
def test_workflow_builder_forwards_langgraph_v1_2_node_fault_tolerance_kwargs() -> None:
    """BaseNode.kwargs forwards every LangGraph v1.2.0 add_node fault-tolerance arg.

    Covers `retry_policy`, `cache_policy`, `timeout`, `error_handler` and `defer`
    so that the generic passthrough seam stays compatible with LangGraph's
    per-node fault-tolerance controls without any frankstate code change.
    """
    retry_policy = RetryPolicy(max_attempts=2)
    cache_policy: CachePolicy[Any] = CachePolicy(ttl=30)
    timeout = TimeoutPolicy(run_timeout=5)

    def error_handler(state: dict[str, object]) -> dict[str, object]:
        return state

    class FaultTolerantLayout(GraphLayout):
        def build_runtime(self) -> dict[str, object]:
            return {}

        def layout(self) -> None:
            self.FT_NODE = SimpleNode(
                enhancer=StaticMessageEnhancer("ft-response"),
                name="ft_node",
                retry_policy=retry_policy,
                cache_policy=cache_policy,
                timeout=timeout,
                error_handler=error_handler,
                defer=True,
                metadata={"tags": ["ft"]},
            )
            self.START_EDGE = SimpleEdge(node_source=START, node_path=self.FT_NODE.name)
            self.END_EDGE = SimpleEdge(node_source=self.FT_NODE.name, node_path=END)

    builder = WorkflowBuilder(config=FaultTolerantLayout, state_schema=FrankTestState)
    builder.compile()

    spec = builder.workflow.nodes["ft_node"]
    assert spec.retry_policy is retry_policy
    assert spec.cache_policy is cache_policy
    assert spec.timeout is not None
    assert spec.timeout.run_timeout == 5
    assert spec.defer is True
    assert spec.metadata == {"tags": ["ft"]}

    assert spec.error_handler_node == "__error_handler__ft_node"
    handler_spec = builder.workflow.nodes["__error_handler__ft_node"]
    assert handler_spec.is_error_handler is True


@pytest.mark.unit
def test_workflow_builder_compiles_linear_layout_and_runs_sync_enhancer() -> None:
    builder = WorkflowBuilder(config=LinearSyncLayout, state_schema=FrankTestState)

    compiled = builder.compile()
    result = compiled.invoke({"messages": [HumanMessage(content="hi")]})

    assert result["messages"][-1].content == "linear-sync-response"
    assert compiled.get_graph().nodes["linear_sync_node"].metadata == {"tags": ["linear-sync"]}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("route", "expected"),
    [("accept", "accepted"), ("reject", "rejected")],
)
def test_workflow_builder_routes_conditional_edges_through_langgraph(route: str, expected: str) -> None:
    builder = WorkflowBuilder(config=ConditionalAsyncLayout, state_schema=FrankTestState)
    compiled = builder.compile()

    result = asyncio.run(
        compiled.ainvoke(
            {
                "messages": [HumanMessage(content="hi")],
                "route": route,
            }
        )
    )

    assert result["messages"][-1].content == expected
    assert set(compiled.get_graph().nodes) >= {
        "__start__",
        "router_node",
        "accept_node",
        "reject_node",
        "__end__",
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("route", "expected"),
    [("accept", "accepted"), ("reject", "rejected")],
)
def test_workflow_builder_routes_async_evaluators_through_langgraph(route: str, expected: str) -> None:
    builder = WorkflowBuilder(config=ConditionalAsyncEvaluatorLayout, state_schema=FrankTestState)
    compiled = builder.compile()

    result = asyncio.run(
        compiled.ainvoke(
            {
                "messages": [HumanMessage(content="hi")],
                "route": route,
            }
        )
    )

    assert result["messages"][-1].content == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("decision", "expected_message"),
    [("accept", "accepted"), ("reject", "rejected")],
)
def test_workflow_builder_routes_command_nodes_and_applies_updates(decision: str, expected_message: str) -> None:
    builder = WorkflowBuilder(config=CommandAsyncLayout, state_schema=FrankTestState)
    compiled = builder.compile()

    result = asyncio.run(
        compiled.ainvoke(
            {
                "messages": [HumanMessage(content="hi")],
                "decision": decision,
            }
        )
    )

    assert result["decision"] == decision
    assert result["messages"][-2].content == f"command:{decision}"
    assert result["messages"][-1].content == expected_message
    assert cast(CommandNode, builder.node_manager.nodes["command_node"]).destinations == (
        "accept_node",
        "reject_node",
    )


@pytest.mark.unit
def test_workflow_builder_integrates_toolnode_with_frank_nodes() -> None:
    builder = WorkflowBuilder(config=ToolLoopLayout, state_schema=FrankTestState)
    compiled = builder.compile()

    result = asyncio.run(
        compiled.ainvoke(
            {
                "messages": [HumanMessage(content="use a tool")],
                "tool_text": "pikachu",
            }
        )
    )

    assert result["messages"][-1].content == "tool:PIKACHU"
    assert compiled.get_graph().nodes["summary_node"].metadata == {"tags": ["summary"]}