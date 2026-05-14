from langgraph.graph import MessagesState


class SharedState(MessagesState):
    """Minimal state schema for message-only agent graphs.

    This is the default schema used by the simple agent and human-in-the-loop
    layouts. Components working with this schema normally read and append the
    `messages` key inherited from `MessagesState`.
    """