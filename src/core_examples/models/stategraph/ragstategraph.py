from langgraph.graph import MessagesState


class RAGState(MessagesState):
    """
    State schema for the adaptive RAG layouts.

    LangGraph merges the partial updates returned by the project enhancers into
    this schema. The fields below are the keys expected by the retriever,
    grader, rewrite and generation nodes.

    Attributes:
        question: Original or rewritten user question.
        context: Retrieved context with recommended text and image lists for multimodal content.
        generation: Final answer returned by the generation node.
        iterations: Number of retrieve-grade-rewrite loops executed.

    Ownership notes:
        - retriever nodes populate `question` and `context`
        - rewrite nodes update `question` and increment `iterations`
        - generation nodes populate `generation` and append a final AI message
    """
    question: str
    context: dict[str, list]
    generation: str
    iterations: int = 0