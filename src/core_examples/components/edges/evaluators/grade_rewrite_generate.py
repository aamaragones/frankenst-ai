from typing import Any, Literal, cast

from langchain_core.messages import AnyMessage
from pydantic import BaseModel

from frankstate.entity.statehandler import StateEvaluator


class GradeRewriteGenerate(StateEvaluator):
    """Choose between answer generation and question rewriting.

    Reads:
        - `question`
        - `context`
        - `iterations`

    Returns:
        - `"generate"` when the retrieved context is relevant or the graph has
            already retried enough times
        - `"rewrite"` when the question should be refined before another
            retrieval attempt
    """

    async def evaluate(self, state: list[AnyMessage] | dict[str, Any] | BaseModel) -> Literal["generate", "rewrite"]:
        state = cast(dict[str, Any], state)
        runnable = self.runnable
        if runnable is None:
            raise TypeError("GradeRewriteGenerate requires a runnable_builder at initialization time")

        response = await runnable.ainvoke({
            "context": state["context"],
            "question": state["question"],
        })

        score = response.binary_score

        if score == "yes" or state.get("iterations", 0) >= 1:
            return "generate"
        else:
            return "rewrite"