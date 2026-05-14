from typing import Any, cast

from langchain_core.messages import AnyMessage
from pydantic import BaseModel

from frankstate.entity.statehandler import StateEnhancer


class RewriteQuestionAsyncInvoke(StateEnhancer):
    """Rewrite the current question before another retrieval attempt.

    Reads:
        - `question`
        - `iterations`

    Returns:
        - `messages`: a list containing the rewritten-question response
        - `question`: the rewritten question text
        - `iterations`: incremented loop counter
    """

    async def enhance(self, state: list[AnyMessage] | dict[str, Any] | BaseModel) -> dict[str, Any]:
        state = cast(dict[str, Any], state)
        runnable = self.runnable
        if runnable is None:
            raise TypeError("RewriteQuestionAsyncInvoke requires a runnable_builder at initialization time")

        question = state["question"]
        response = await runnable.ainvoke(question)
        better_question = response.content

        current_iterations = state.get("iterations", 0)

        return {"messages": [response], "question": better_question, "iterations": current_iterations + 1}
        