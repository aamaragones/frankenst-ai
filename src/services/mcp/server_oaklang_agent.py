"""FastMCP server exposing the reference graphs as tools; runs in the `mcp` env."""

import logging
from collections.abc import Mapping
from uuid import uuid4

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from config.graph_layout.local_vectorstore_adaptive_rag_config_graph import (
    LocalVectorStoreAdaptiveRAGConfigGraph,
)
from config.graph_layout.oak_human_loop_config_graph import OakHumanLoopConfigGraph
from core_ai_examples.models.stategraph.ragstategraph import RAGState
from core_ai_examples.models.stategraph.stategraph import SharedState
from frankstate import WorkflowBuilder

logger = logging.getLogger(__name__)
mcp = FastMCP("CustomMCPServer")


async def run_oaklang_agent(input: str, headers: Mapping[str, str]) -> str:
    """Answer through the human-review graph; a pause returns the review question.

    The header check is a stand-in for real authorization. Each call compiles with
    its own `InMemorySaver` and thread: nothing resumes a pause from here, so a
    shared saver would only accumulate dead threads.
    """
    if headers.get("x-custom-header") != "admin":
        return "Unauthorized: Invalid permission to access OakLangAgent."

    graph = WorkflowBuilder(
        config=OakHumanLoopConfigGraph, state_schema=SharedState
    ).compile(checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
    response = await graph.ainvoke(
        {"messages": [{"role": "human", "content": input}]}, config
    )
    if interrupts := response.get("__interrupt__"):
        return str(interrupts[0].value["question"])
    return str(response["messages"][-1].content)


async def run_adaptive_rag(input: str) -> str:
    graph = WorkflowBuilder(
        config=LocalVectorStoreAdaptiveRAGConfigGraph, state_schema=RAGState
    ).compile()
    response = await graph.ainvoke({"messages": [{"role": "human", "content": input}]})
    return str(response["messages"][-1].content)


@mcp.tool(
    "handoff_oaklang_agent",
    description="Tool to use OakLangAgent about any Pokemon question. The input is a question.",
)
async def handoff_oaklang_agent(input: str) -> str:
    headers = get_http_headers()
    if headers.get("x-custom-header"):
        logger.info("Received custom header: %s", headers["x-custom-header"])
    return await run_oaklang_agent(input, headers)


@mcp.tool(
    "adaptive_rag_tool",
    description="Tool to use RAG about Pokémon series questions. The input is a question.",
)
async def adaptive_rag_tool(input: str) -> str:
    return await run_adaptive_rag(input)


if __name__ == "__main__":
    from utils.logger import configure_logging

    configure_logging()
    mcp.run(transport="http")
