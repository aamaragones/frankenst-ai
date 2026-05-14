from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from langchain_core.embeddings import Embeddings


class AISearchMultiVectorRetriever:
    def __init__(
        self,
        search_client: SearchClient,
        embeddings: Embeddings,
    ):
        """
        Args:
            search_client: Azure Search client configured with endpoint, index, and credentials.
            embeddings: Model used to generate query embeddings.
        """
        self.search_client = search_client
        self.embeddings = embeddings

    def _search(self, query: str, k: int = 5, embed: bool = True) -> list[dict]:
        """Performs a vector search with the given query."""
        if embed:
            vector = self.embeddings.embed_query(query)

            vector_query = VectorizedQuery(
                vector=vector,
                kind="vector",
                fields="embeddings",
                k_nearest_neighbors=k,
                exhaustive=True,
                weight=0.5, # for hybrid search
            )

            results = self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                select=["type", "content", "metadata"],
                include_total_count=True
            )
        else:
            # TODO: search text 
            raise NotImplementedError("Text search in progress.")

        return list(results)

    def _parse_results(self, results: list[dict], metadata_as_content: bool = True) -> dict[str, list[str]]:
        """Groups documents by type, optionally appending metadata to content."""
        grouped = {"texts": [], "images": []}
        for doc in results:
            t = doc.get("type")
            c = doc.get("content", "")
            
            if metadata_as_content:
                metadata = doc.get("metadata")
                if metadata:
                    c += f"\n\nMetadata: {metadata}"

            if t == "texts" or t == "tables":
                grouped["texts"].append(c)
            elif t == "images":
                grouped["images"].append(c)

        return grouped

    def get_context(self, query: str, k: int = 5, embed: bool = True) -> dict[str, object]:
        """
        Performs the search and builds the context.

        Returns:
            dict with keys "texts" (concatenated string) and "images" (list of dicts for prompt).
        """
        results = self._search(query, k, embed)
        docs_by_type = self._parse_results(results)

        context_text = "\n".join(docs_by_type.get("texts", []))

        context_images = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"}
            }
            for img in docs_by_type.get("images", [])
        ]

        return {
            "texts": context_text.strip(),
            "images": context_images
        }
