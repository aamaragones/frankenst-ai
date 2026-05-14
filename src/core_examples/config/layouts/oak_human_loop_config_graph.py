from typing import Any

from langchain_core.tools import BaseTool
from langgraph.graph import END, START
from langgraph.prebuilt import ToolNode

from core_examples.components.edges.evaluators.route_human_node import RouteHumanNode
from core_examples.components.nodes.commands.human_review_sensitive_tool_call import (
    HumanReviewSensitiveToolCall,
)
from core_examples.components.nodes.enhancers.simple_messages_ainvoke import (
    SimpleMessagesAsyncInvoke,
)
from core_examples.components.runnables.oaklang_agent.oaklang_agent import OakLangAgent
from core_examples.components.tools.dominate_pokemon.dominate_pokemon_tool import (
    DominatePokemonTool,
)
from core_examples.components.tools.get_evolution.get_evolution_tool import (
    GetEvolutionTool,
)
from core_examples.components.tools.random_movements.random_movements_tool import (
    RandomMovementsTool,
)
from core_examples.config.settings import get_settings
from core_examples.utils.config_loader import load_node_registry
from frankstate.entity.edge import ConditionalEdge, SimpleEdge
from frankstate.entity.graph_layout import GraphLayout
from frankstate.entity.node import CommandNode, SimpleNode
from services.foundry.llms import LLMServices


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

    Use this layout as the reference pattern for human-in-the-loop routing.
    """

    CONFIG_NODES: dict[str, Any]
    OAKLANG_AGENT: OakLangAgent
    SENSITIVE_TOOLS: list[BaseTool]

    def build_runtime(self) -> dict[str, Any]:
        settings = get_settings()
        LLMServices.launch()
        if LLMServices.model is None:
            raise RuntimeError("LLMServices.launch() did not initialize model.")

        dominate_pokemon_tool = DominatePokemonTool()

        return {
            "CONFIG_NODES": load_node_registry(settings.config_nodes_file_path),
            "OAKLANG_AGENT": OakLangAgent(
                model=LLMServices.model,
                tools=[GetEvolutionTool(), RandomMovementsTool(), dominate_pokemon_tool],
            ),
            "SENSITIVE_TOOLS": [dominate_pokemon_tool],
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
        self.HUMAN_REVIEW_NODE = CommandNode(
            commander=HumanReviewSensitiveToolCall(
                sensitive_tools=self.SENSITIVE_TOOLS,
                destinations=self.CONFIG_NODES["HUMAN_REVIEW_NODE"]["destinations"],
            ),
            name=self.CONFIG_NODES["HUMAN_REVIEW_NODE"]["name"],
            tags=[self.CONFIG_NODES["HUMAN_REVIEW_NODE"]["description"]],
        )

        ## EDGES
        self._EDGE_1 = SimpleEdge(
            node_source=START, 
            node_path=self.OAKLANG_NODE.name)
        self._EDGE_2 = SimpleEdge(
            node_source=self.OAKTOOLS_NODE.name, 
            node_path=self.OAKLANG_NODE.name)
        self._EDGE_3 = ConditionalEdge(
            evaluator=RouteHumanNode(),
            map_dict={
                "end": END,
                "review": self.HUMAN_REVIEW_NODE.name,
            },
            node_source=self.OAKLANG_NODE.name,
        )