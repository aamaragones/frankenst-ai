from typing import Any

from core_ai_examples.components.tools.retriever_pokeseriex.retriever_pokeseriex import (
    RetrieverPokeSeriex,
)
from services.llm.llm_services import LLMServices


class Orchestrator:
    @staticmethod
    def run(query: str) -> list[Any]:
        (embeddings,) = LLMServices.launch().require("embeddings")
        return RetrieverPokeSeriex.run(query=query, embeddings=embeddings)
