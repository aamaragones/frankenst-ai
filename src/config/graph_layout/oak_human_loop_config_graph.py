"""Tool-calling agent layout with a human review step before sensitive tools."""

from typing import Any

from langchain_core.tools import BaseTool
from langgraph.graph import END, START
from langgraph.prebuilt import ToolNode

from config.settings import get_settings
from core_ai_examples.components.edges.evaluators.route_human_node import RouteHumanNode
from core_ai_examples.components.nodes.commands.human_review_sensitive_tool_call import (
    HumanReviewSensitiveToolCall,
)
from core_ai_examples.components.nodes.enhancers.simple_messages_ainvoke import (
    SimpleMessagesAsyncInvoke,
)
from core_ai_examples.components.runnables.oaklang_agent.oaklang_agent import (
    OakLangAgent,
)
from core_ai_examples.components.tools.dominate_pokemon.dominate_pokemon_tool import (
    DominatePokemonTool,
)
from core_ai_examples.components.tools.get_evolution.get_evolution_tool import (
    GetEvolutionTool,
)
from core_ai_examples.components.tools.random_movements.random_movements_tool import (
    RandomMovementsTool,
)
from frankstate.entity.edge import ConditionalEdge, SimpleEdge
from frankstate.entity.graph_layout import GraphLayout
from frankstate.entity.node import CommandNode, SimpleNode, ToolGraphNode
from services.llm.llm_services import LLMServices
from utils.config_loader import load_node_registry


# NOTE: This is an example implementation for illustration purposes
# NOTE: Here you can add other subgraphs as nodes
class OakHumanLoopConfigGraph(GraphLayout):
    """Tool-calling agent layout with an explicit human review step.

    State expectations:
        - Uses `SharedState` or another messages-compatible schema.
        - The command node inspects the latest tool call and may return a
          LangGraph `Command` with feedback updates.

    Flow:
        START -> OakLangAgent -> (HumanReview | END)
        HumanReview -> (OakTools | OakLangAgent)
        OakTools -> OakLangAgent

    Tools are bound by the names `config_nodes.yaml` declares under
    `OAKTOOLS_NODE.metadata.tools`; `HUMAN_REVIEW_NODE.metadata.sensitive_tools` names
    the ones that pause for review. `HumanReview` calls `interrupt()`, so compile with
    a checkpointer or the pause has nowhere to resume from.
    """

    CONFIG_NODES: dict[str, Any]
    OAKLANG_AGENT: OakLangAgent
    SENSITIVE_TOOLS: list[BaseTool]

    def build_runtime(self) -> dict[str, Any]:
        settings = get_settings()
        (model,) = LLMServices.launch().require("low_model")
        nodes = load_node_registry(settings.config_nodes_file_path)

        available: dict[str, BaseTool] = {
            tool.name: tool
            for tool in (
                GetEvolutionTool(),
                RandomMovementsTool(),
                DominatePokemonTool(),
            )
        }
        tools = [
            available[name] for name in nodes["OAKTOOLS_NODE"]["metadata"]["tools"]
        ]
        bound = {tool.name: tool for tool in tools}
        return {
            "CONFIG_NODES": nodes,
            "OAKLANG_AGENT": OakLangAgent(model=model, tools=tools),
            "SENSITIVE_TOOLS": [
                bound[name]
                for name in nodes["HUMAN_REVIEW_NODE"]["metadata"]["sensitive_tools"]
            ],
        }

    def layout(self) -> None:
        ## NODES
        self.OAKLANG_NODE = SimpleNode(
            enhancer=SimpleMessagesAsyncInvoke(self.OAKLANG_AGENT),
            name=self.CONFIG_NODES["OAKLANG_NODE"]["name"],
            metadata=self.CONFIG_NODES["OAKLANG_NODE"]["metadata"],
        )
        self.OAKTOOLS_NODE = ToolGraphNode(
            tool_node=ToolNode(
                tools=self.OAKLANG_AGENT.tools or [],
                name=self.CONFIG_NODES["OAKTOOLS_NODE"]["name"],
            ),
            name=self.CONFIG_NODES["OAKTOOLS_NODE"]["name"],
            metadata=self.CONFIG_NODES["OAKTOOLS_NODE"]["metadata"],
        )
        self.HUMAN_REVIEW_NODE = CommandNode(
            commander=HumanReviewSensitiveToolCall(
                sensitive_tools=self.SENSITIVE_TOOLS,
                destinations=self.CONFIG_NODES["HUMAN_REVIEW_NODE"]["destinations"],
            ),
            name=self.CONFIG_NODES["HUMAN_REVIEW_NODE"]["name"],
            metadata=self.CONFIG_NODES["HUMAN_REVIEW_NODE"]["metadata"],
        )

        ## EDGES
        self._EDGE_1 = SimpleEdge(node_source=START, node_path=self.OAKLANG_NODE.name)
        self._EDGE_2 = SimpleEdge(
            node_source=self.OAKTOOLS_NODE.name, node_path=self.OAKLANG_NODE.name
        )
        self._EDGE_3 = ConditionalEdge(
            evaluator=RouteHumanNode(),
            map_dict={
                "end": END,
                "review": self.HUMAN_REVIEW_NODE.name,
            },
            node_source=self.OAKLANG_NODE.name,
        )
