import logging

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers

logger = logging.getLogger(__name__)
mcp = FastMCP("CustomMCPServer")


@mcp.tool(
    "handoff_oaklang_agent",
    description="Tool to use OakLangAgent about any Pokemon question. The input is a question.",
)
async def handoff_oaklang_agent(input: str) -> str:
    from config.graph_layout.oak_human_loop_config_graph import (
        OakHumanLoopConfigGraph,
    )
    from core_ai_examples.models.stategraph.stategraph import SharedState
    from frankstate import WorkflowBuilder
    from utils.logger import configure_logging

    # Get HTTP headers or bearer token (if needed for authentication or other purposes)
    headers = get_http_headers()
    # auth_token = headers.get("authorization", None) # Example of extracting a bearer token from the Authorization header
    custom_header = headers.get("x-custom-header", None)  # Example of a custom header
    if custom_header:
        logger.info("Received custom header: %s", custom_header)

    if custom_header != "admin":  # dummy check, replace with actual logic as needed
        return "Unauthorized: Invalid permission to access OakLangAgent."

    configure_logging()

    workflow_builder = WorkflowBuilder(
        config=OakHumanLoopConfigGraph,
        state_schema=SharedState,
    )
    oaklang_agent_graph = workflow_builder.compile()

    message_input = {"messages": [{"role": "human", "content": input}]}
    response = await oaklang_agent_graph.ainvoke(message_input)
    return str(response["messages"][-1].content)


@mcp.tool(
    "adaptive_rag_tool",
    description="Tool to use RAG about Pokémon series questions. The input is a question.",
)
async def adaptive_rag_tool(input: str) -> str:
    from config.graph_layout.local_vectorstore_adaptive_rag_config_graph import (
        LocalVectorStoreAdaptiveRAGConfigGraph,
    )
    from core_ai_examples.models.stategraph.ragstategraph import RAGState
    from frankstate import WorkflowBuilder
    from utils.logger import configure_logging

    configure_logging()

    workflow_builder = WorkflowBuilder(
        config=LocalVectorStoreAdaptiveRAGConfigGraph,
        state_schema=RAGState,
    )
    adaptive_rag_graph = workflow_builder.compile()

    message_input = {"messages": [{"role": "human", "content": input}]}
    response = await adaptive_rag_graph.ainvoke(message_input)
    return str(response["messages"][-1].content)


if __name__ == "__main__":
    mcp.run(transport="http")
