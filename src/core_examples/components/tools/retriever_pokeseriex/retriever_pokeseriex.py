from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from core_examples.components.retrievers.ai_search_simple_semantic_retriever.ai_search_simple_semantic_retriever import (
    AISearchSimpleSemanticRetriever,
)
from core_examples.utils.key_vault import get_secret
from services.foundry.llms import LLMServices

INDEX_NAME = "pokeseriex-index"

class RetrieverPokeSeriex:
    @staticmethod
    def run(query: str) -> list:
        """
        This tool transforms a natural language query into a vector and retrieves context from the content and schemas of a Teradata database.

        Args:
            query (str): Question or query in natural language about the PokeSeriex database.

        Returns:
            str: List containing the most relevant documents and schemas within the PokeSeriex database.
        """
        
        LLMServices.launch()
        if LLMServices.embeddings is None:
            raise RuntimeError("LLMServices.launch() did not initialize embeddings.")
   
        # Create search client
        service_endpoint = get_secret("AZURE_SEARCH_SERVICE_ENDPOINT")
        key = get_secret("AZURE_SEARCH_API_KEY")
        search_client = SearchClient(service_endpoint, INDEX_NAME, AzureKeyCredential(key))
        
        # Use the retriever
        retriever = AISearchSimpleSemanticRetriever(search_client=search_client, embeddings=LLMServices.embeddings)

        results = retriever.retrieve(query)
        return results