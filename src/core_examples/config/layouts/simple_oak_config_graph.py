from typing import Any

from langgraph.graph import END, START
from langgraph.prebuilt import ToolNode

from core_examples.components.edges.evaluators.route_tool_condition import (
    RouteToolCondition,
)
from core_examples.components.nodes.enhancers.simple_messages_ainvoke import (
    SimpleMessagesAsyncInvoke,
)
from core_examples.components.runnables.oaklang_agent.oaklang_agent import OakLangAgent
from core_examples.components.tools.get_evolution.get_evolution_tool import (
    GetEvolutionTool,
)
from core_examples.components.tools.random_movements.random_movements_tool import (
    RandomMovementsTool,
)
from core_examples.constants import CONFIG_NODES_FILE_PATH
from core_examples.utils.config_loader import load_node_registry
from frankstate.entity.edge import ConditionalEdge, SimpleEdge
from frankstate.entity.graph_layout import GraphLayout
from frankstate.entity.node import SimpleNode
from services.foundry.llms import LLMServices


# NOTE: This is an example implementation for illustration purposes
# NOTE: Here you can add other subgraphs as nodes
class SimpleOakConfigGraph(GraphLayout):
    """Minimal agent-with-tools layout.

    State expectations:
        - Uses `SharedState` or another messages-compatible schema.
        - The agent node reads `messages` and appends a new assistant message.

    Flow:
        START -> OakLangAgent -> (OakTools | END)
        OakTools -> OakLangAgent

    This layout is the simplest starting point when the graph only needs a tool
    loop and does not require human review or retrieval-specific state.
    """

    CONFIG_NODES: dict[str, Any]
    OAKLANG_AGENT: OakLangAgent

    def build_runtime(self) -> dict[str, Any]:
        LLMServices.launch()
        if LLMServices.model is None:
            raise RuntimeError("LLMServices.launch() did not initialize model.")
        
        return {
            "CONFIG_NODES": load_node_registry(CONFIG_NODES_FILE_PATH),
            "OAKLANG_AGENT": OakLangAgent(
                model=LLMServices.model,
                tools=[GetEvolutionTool(), RandomMovementsTool()],
            ),
        }

    def layout(self) -> None:
        ## NODES
        self.OAKLANG_NODE = SimpleNode(
            enhancer=SimpleMessagesAsyncInvoke(self.OAKLANG_AGENT),
            name=self.CONFIG_NODES["OAKLANG_NODE"]["name"],
            tags=[self.CONFIG_NODES["OAKLANG_NODE"]["description"]],
        )
        self.OAKTOOLS_NODE = ToolNode(
            tools=self.OAKLANG_AGENT.tools or [],
            name=self.CONFIG_NODES["OAKTOOLS_NODE"]["name"],
            tags=[self.CONFIG_NODES["OAKTOOLS_NODE"]["description"]],
        )

        ## EDGES
        self._EDGE_1 = SimpleEdge(node_source=START, node_path=self.OAKLANG_NODE.name)
        self._EDGE_2 = SimpleEdge(
            node_source=self.OAKTOOLS_NODE.name,
            node_path=self.OAKLANG_NODE.name,
        )
        self._EDGE_3 = ConditionalEdge(
            evaluator=RouteToolCondition(),
            map_dict={
                "end": END,
                "tools": self.OAKTOOLS_NODE.name,
            },
            node_source=self.OAKLANG_NODE.name,
        )