from core_examples.components.tools.retriever_pokeseriex.retriever_pokeseriex import (
    RetrieverPokeSeriex,
)


class Orchestrator:
    @staticmethod
    def run(query: str):
        # Logic
        return RetrieverPokeSeriex.run(query=query)