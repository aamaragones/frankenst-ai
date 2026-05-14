from typing import Any, cast

from langchain_core.messages import AIMessage, AnyMessage
from pydantic import BaseModel

from frankstate.entity.statehandler import StateEnhancer


class SimpleMessagesInvoke(StateEnhancer):
    """Invoke an agent runnable with the current message history.

    Reads:
        - `messages`

    Returns:
        - `messages`: a list containing the new AI response so LangGraph can
          append it to the running conversation state.
    """
    
    def enhance(self, state: list[AnyMessage] | dict[str, Any] | BaseModel) -> dict[str, Any]:
        state = cast(dict[str, Any], state)
        runnable = self.runnable
        if runnable is None:
            raise TypeError("SimpleMessagesInvoke requires a runnable_builder at initialization time")

        messages = cast(AIMessage, state["messages"])
        response = runnable.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}
