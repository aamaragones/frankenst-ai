from typing import Any, cast

from langchain_core.messages import AIMessage, AnyMessage
from pydantic import BaseModel

from frankstate.entity.statehandler import StateEnhancer


class RetrieveContextAsyncInvoke(StateEnhancer):
    """Retrieve context from the configured runnable retriever.

    Reads:
        - `messages` on the first retrieval pass
        - `question` and `iterations` on subsequent passes

    Returns:
        - `context`: retrieved multimodal context
        - `question`: the question that should be used by downstream nodes
    """

    async def enhance(self, state: list[AnyMessage] | dict[str, Any] | BaseModel) -> dict[str, Any]:
        state = cast(dict[str, Any], state)
        runnable = self.runnable
        if runnable is None:
            raise TypeError("RetrieveContextAsyncInvoke requires a runnable_builder at initialization time")

        if state.get("iterations", 0) > 0:
            question = state["question"]
        else:
            last_message = cast(AIMessage, state["messages"][-1])
            question = last_message.content

        retrieved_docs_context = await runnable.ainvoke(question)

        return {"context": retrieved_docs_context, "question": question}