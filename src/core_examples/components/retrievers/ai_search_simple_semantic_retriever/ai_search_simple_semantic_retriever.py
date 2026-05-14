from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from langchain_core.embeddings import Embeddings


class AISearchSimpleSemanticRetriever:
    def __init__(
        self,
        search_client: SearchClient,
        embeddings: Embeddings,
        k: int = 5,
    ):
        """
        Args:
            search_client: Azure Search client configured with endpoint, index, and credentials.
            embeddings: Model used to generate query embeddings.
        """
        self.search_client = search_client
        self.k = k
        self.embeddings = embeddings

    def retrieve(self, query: str) -> list[dict]:
        """
        Ejecuta una búsqueda semántica usando embeddings y Azure Cognitive Search.
        """
        vector = self.embeddings.embed_query(query)

        vector_query = VectorizedQuery(
            vector=vector,
            kind="vector",
            fields="embeddings",
            k_nearest_neighbors=self.k,
            exhaustive=True,
            # weight=None, # for hybrid search 
        )

        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["content"],
            include_total_count=True,
        )

        return list(results)
