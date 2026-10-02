import asyncio

import pytest
from langchain_core.messages import HumanMessage

from core_ai_examples.components.nodes.enhancers.retrieve_context_ai_search import (
    RetrieveContextAISearch,
)

pytestmark = pytest.mark.unit


class FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def get_context(self, query: str) -> dict[str, list[str]]:
        self.queries.append(query)
        return {"texts": [f"context for {query}"]}


def test_the_layout_declares_the_retriever_on_the_node_line() -> None:
    retriever = FakeRetriever()
    node = RetrieveContextAISearch(retriever=retriever)

    result = asyncio.run(
        node.enhance({"messages": [HumanMessage(content="Who evolves from Feebas?")]})
    )

    assert retriever.queries == ["Who evolves from Feebas?"]
    assert result == {
        "context": {"texts": ["context for Who evolves from Feebas?"]},
        "question": "Who evolves from Feebas?",
    }


def test_later_iterations_reuse_the_rewritten_question() -> None:
    retriever = FakeRetriever()
    node = RetrieveContextAISearch(retriever=retriever)

    result = asyncio.run(node.enhance({"question": "rewritten", "iterations": 1}))

    assert retriever.queries == ["rewritten"]
    assert result["question"] == "rewritten"


def test_a_misspelt_keyword_fails_on_the_layout_line() -> None:
    with pytest.raises(
        TypeError, match=r"does not declare \['retreiver'\].*'retriever'"
    ):
        RetrieveContextAISearch(retreiver=FakeRetriever())
