from typing import Any

from langgraph.graph import END, START, MessagesState
from langgraph.prebuilt import ToolNode

from frankstate.entity.edge import ConditionalEdge, SimpleEdge
from frankstate.entity.graph_layout import GraphLayout
from frankstate.entity.node import CommandNode, SimpleNode
from tests.support.frankstate_doubles.builders import FakeRunnableBuilder
from tests.support.frankstate_doubles.stub import (
    AsyncFieldRouteEvaluator,
    FieldRouteEvaluator,
    RoutingCommander,
    RunnableMessageEnhancer,
    StaticMessageEnhancer,
    SyncRunnableMessageEnhancer,
    ToolCallEvaluator,
    ToolCallingEnhancer,
    ToolSummaryEnhancer,
    uppercase_text,
)


class FrankTestState(MessagesState):
    route: str
    decision: str
    tool_text: str


class LinearAsyncLayout(GraphLayout):
    RUNNABLE_BUILDER: FakeRunnableBuilder

    def __init__(self):
        super().__init__()
        self.runtime_calls = 0
        self.layout_calls = 0

    def build_runtime(self) -> dict[str, Any]:
        self.runtime_calls += 1
        return {
            "RUNNABLE_BUILDER": FakeRunnableBuilder(async_result={"content": "linear-response"})
        }

    def layout(self) -> None:
        self.layout_calls += 1
        self.LINEAR_NODE = SimpleNode(
            enhancer=RunnableMessageEnhancer(runnable_builder=self.RUNNABLE_BUILDER),
            name="linear_node",
            tags=["linear"],
        )
        self.START_EDGE = SimpleEdge(node_source=START, node_path=self.LINEAR_NODE.name)
        self.END_EDGE = SimpleEdge(node_source=self.LINEAR_NODE.name, node_path=END)


class ConditionalAsyncLayout(GraphLayout):
    def __init__(self):
        super().__init__()
        self.runtime_calls = 0
        self.layout_calls = 0

    def build_runtime(self) -> dict[str, Any]:
        self.runtime_calls += 1
        return {}

    def layout(self) -> None:
        self.layout_calls += 1
        self.ROUTER_NODE = SimpleNode(
            enhancer=StaticMessageEnhancer("router"),
            name="router_node",
            tags=["router"],
        )
        self.ACCEPT_NODE = SimpleNode(
            enhancer=StaticMessageEnhancer("accepted"),
            name="accept_node",
            tags=["accept"],
        )
        self.REJECT_NODE = SimpleNode(
            enhancer=StaticMessageEnhancer("rejected"),
            name="reject_node",
            tags=["reject"],
        )

        self.START_EDGE = SimpleEdge(node_source=START, node_path=self.ROUTER_NODE.name)
        self.ROUTE_EDGE = ConditionalEdge(
            node_source=self.ROUTER_NODE.name,
            map_dict={
                "accept": self.ACCEPT_NODE.name,
                "reject": self.REJECT_NODE.name,
            },
            evaluator=FieldRouteEvaluator(),
        )
        self.ACCEPT_EDGE = SimpleEdge(node_source=self.ACCEPT_NODE.name, node_path=END)
        self.REJECT_EDGE = SimpleEdge(node_source=self.REJECT_NODE.name, node_path=END)


class ConditionalAsyncEvaluatorLayout(GraphLayout):
    def __init__(self):
        super().__init__()
        self.runtime_calls = 0
        self.layout_calls = 0

    def build_runtime(self) -> dict[str, Any]:
        self.runtime_calls += 1
        return {}

    def layout(self) -> None:
        self.layout_calls += 1
        self.ROUTER_NODE = SimpleNode(
            enhancer=StaticMessageEnhancer("router"),
            name="router_node",
            tags=["router"],
        )
        self.ACCEPT_NODE = SimpleNode(
            enhancer=StaticMessageEnhancer("accepted"),
            name="accept_node",
            tags=["accept"],
        )
        self.REJECT_NODE = SimpleNode(
            enhancer=StaticMessageEnhancer("rejected"),
            name="reject_node",
            tags=["reject"],
        )

        self.START_EDGE = SimpleEdge(node_source=START, node_path=self.ROUTER_NODE.name)
        self.ROUTE_EDGE = ConditionalEdge(
            node_source=self.ROUTER_NODE.name,
            map_dict={
                "accept": self.ACCEPT_NODE.name,
                "reject": self.REJECT_NODE.name,
            },
            evaluator=AsyncFieldRouteEvaluator(),
        )
        self.ACCEPT_EDGE = SimpleEdge(node_source=self.ACCEPT_NODE.name, node_path=END)
        self.REJECT_EDGE = SimpleEdge(node_source=self.REJECT_NODE.name, node_path=END)


class CommandAsyncLayout(GraphLayout):
    def __init__(self):
        super().__init__()
        self.runtime_calls = 0
        self.layout_calls = 0

    def build_runtime(self) -> dict[str, Any]:
        self.runtime_calls += 1
        return {}

    def layout(self) -> None:
        self.layout_calls += 1
        self.COMMAND_NODE = CommandNode(
            commander=RoutingCommander(
                destinations={
                    "accept": "accept_node",
                    "reject": "reject_node",
                }
            ),
            name="command_node",
            tags=["command"],
        )
        self.ACCEPT_NODE = SimpleNode(
            enhancer=StaticMessageEnhancer("accepted"),
            name="accept_node",
            tags=["accept"],
        )
        self.REJECT_NODE = SimpleNode(
            enhancer=StaticMessageEnhancer("rejected"),
            name="reject_node",
            tags=["reject"],
        )

        self.START_EDGE = SimpleEdge(node_source=START, node_path=self.COMMAND_NODE.name)
        self.ACCEPT_EDGE = SimpleEdge(node_source=self.ACCEPT_NODE.name, node_path=END)
        self.REJECT_EDGE = SimpleEdge(node_source=self.REJECT_NODE.name, node_path=END)


class LinearSyncLayout(GraphLayout):
    RUNNABLE_BUILDER: FakeRunnableBuilder

    def __init__(self):
        super().__init__()
        self.runtime_calls = 0
        self.layout_calls = 0

    def build_runtime(self) -> dict[str, Any]:
        self.runtime_calls += 1
        return {
            "RUNNABLE_BUILDER": FakeRunnableBuilder(sync_result={"content": "linear-sync-response"})
        }

    def layout(self) -> None:
        self.layout_calls += 1
        self.LINEAR_NODE = SimpleNode(
            enhancer=SyncRunnableMessageEnhancer(runnable_builder=self.RUNNABLE_BUILDER),
            name="linear_sync_node",
            tags=["linear-sync"],
        )
        self.START_EDGE = SimpleEdge(node_source=START, node_path=self.LINEAR_NODE.name)
        self.END_EDGE = SimpleEdge(node_source=self.LINEAR_NODE.name, node_path=END)


class ToolLoopLayout(GraphLayout):
    def __init__(self):
        super().__init__()
        self.runtime_calls = 0
        self.layout_calls = 0

    def build_runtime(self) -> dict[str, Any]:
        self.runtime_calls += 1
        return {}

    def layout(self) -> None:
        self.layout_calls += 1
        self.AGENT_NODE = SimpleNode(
            enhancer=ToolCallingEnhancer(),
            name="agent_node",
            tags=["agent"],
        )
        self.TOOL_NODE = ToolNode(
            tools=[uppercase_text],
            name="tool_node",
            tags=["tool"],
        )
        self.SUMMARY_NODE = SimpleNode(
            enhancer=ToolSummaryEnhancer(),
            name="summary_node",
            tags=["summary"],
        )

        self.START_EDGE = SimpleEdge(node_source=START, node_path=self.AGENT_NODE.name)
        self.ROUTE_EDGE = ConditionalEdge(
            node_source=self.AGENT_NODE.name,
            map_dict={
                "tools": self.TOOL_NODE.name,
                "end": END,
            },
            evaluator=ToolCallEvaluator(),
        )
        self.TOOL_EDGE = SimpleEdge(node_source=self.TOOL_NODE.name, node_path=self.SUMMARY_NODE.name)
        self.END_EDGE = SimpleEdge(node_source=self.SUMMARY_NODE.name, node_path=END)