from typing import Any, cast

from langchain_core.messages import AIMessage, AnyMessage
from pydantic import BaseModel

from core_ai_examples.components.retrievers.ai_search_multivector_retriever.ai_search_multivector_retriever import (
    AISearchMultiVectorRetriever,
)
from frankstate.entity.statehandler import StateEnhancer


class RetrieveContextAISearch(StateEnhancer):
    """Retrieve context from Azure AI Search using the current question.

    The layout composes the retriever and passes it as `retriever=`; the class
    annotation below declares that keyword. This keeps node execution focused on
    state transforms instead of infrastructure setup.

    Reads:
        - `messages` on the first retrieval pass
        - `question` and `iterations` on subsequent passes

    Returns:
        - `context`: retrieved multimodal context from AI Search
        - `question`: the question that should be used by downstream nodes
    """

    retriever: AISearchMultiVectorRetriever

    async def enhance(
        self, state: list[AnyMessage] | dict[str, Any] | BaseModel
    ) -> dict[str, Any]:
        state = cast(dict[str, Any], state)

        if state.get("iterations", 0) > 0:
            question = state["question"]
        else:
            last_message = cast(AIMessage, state["messages"][-1])
            question = last_message.text

        retrieved_docs_context = self.retriever.get_context(question)

        return {"context": retrieved_docs_context, "question": question}
