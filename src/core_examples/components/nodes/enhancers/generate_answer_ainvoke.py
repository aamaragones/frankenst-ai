from typing import Any, cast

from langchain_core.messages import AnyMessage
from pydantic import BaseModel

from frankstate.entity.statehandler import StateEnhancer


class GenerateAnswerAsyncInvoke(StateEnhancer):
    """Generate the final answer for the current retrieval iteration.

    Reads:
        - `context`
        - `question`

    Returns:
        - `messages`: a list containing the final AI response
        - `generation`: the response content stored as a scalar graph field
    """

    async def enhance(self, state: list[AnyMessage] | dict[str, Any] | BaseModel) -> dict[str, Any]:
        state = cast(dict[str, Any], state)
        runnable = self.runnable
        if runnable is None:
            raise TypeError("GenerateAnswerAsyncInvoke requires a runnable_builder at initialization time")

        response = await runnable.ainvoke({
            "context": state["context"],
            "question": state["question"],
        })
    
        return {"messages": [response], "generation": response.content}