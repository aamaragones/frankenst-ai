from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain_core.embeddings import Embeddings

from config.settings import get_settings
from core_ai_examples.components.retrievers.ai_search_simple_semantic_retriever.ai_search_simple_semantic_retriever import (
    AISearchSimpleSemanticRetriever,
)

INDEX_NAME = "pokeseriex-index"


class RetrieverPokeSeriex:
    @staticmethod
    def run(query: str, embeddings: Embeddings) -> list[Any]:
        """Embed a natural-language query and retrieve PokeSeriex documents and schemas."""
        settings = get_settings()
        service_endpoint = settings.resolve_secret("AZURE_SEARCH_SERVICE_ENDPOINT")
        key = settings.resolve_secret("AZURE_SEARCH_API_KEY")
        search_client = SearchClient(
            service_endpoint, INDEX_NAME, AzureKeyCredential(key)
        )
        retriever = AISearchSimpleSemanticRetriever(
            search_client=search_client, embeddings=embeddings
        )
        return retriever.retrieve(query)
