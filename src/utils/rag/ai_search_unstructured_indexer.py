"""Azure AI Search sink for the unstructured PDF indexer, plus the index manager."""

import datetime as dt
import os
import uuid
from typing import Any

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
from azure.search.documents.models import IndexingResult
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from utils.rag.ai_search_schemas import load_registered_ai_search_index_definition
from utils.rag.unstructured_pdf_indexer import Bucket, UnstructuredPDFIndexerBase


class AISearchMultiVectorDocumentIndexer(UnstructuredPDFIndexerBase):
    """Index PDF texts, tables and images into Azure AI Search, one summary vector each."""

    def __init__(
        self,
        search_client: SearchClient,
        llm_multimodal: BaseLanguageModel[Any] | None = None,
        embeddings: Embeddings | None = None,
    ):
        super().__init__(text_model=llm_multimodal, image_model=llm_multimodal)
        self.search_client = search_client
        self.llm_multimodal = llm_multimodal
        self.embeddings = embeddings
        self.documents: list[dict[str, Any]] = []

    def _classify_chunk(self, chunk: Any) -> Bucket | None:
        # A composite chunk that wraps a table is indexed as a table, not as prose.
        if "Table" in str(type(chunk)):
            return "tables"
        if "CompositeElement" in str(type(chunk)):
            elements = getattr(chunk.metadata, "orig_elements", None) or []
            return (
                "tables" if any("Table" in str(type(e)) for e in elements) else "texts"
            )
        return None

    def _format_table(self, chunk: Any) -> str:
        return str(chunk.text) + str(chunk.metadata.text_as_html)

    def embed_ai_search_index_documents(self) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []

        for content_type in ["texts", "tables", "images"]:
            chunks_list = self.elements.get(content_type)
            summaries_list = self.summaries.get(content_type)
            if self.embeddings is None:
                raise RuntimeError(
                    "An embeddings model is required to index documents."
                )
            embeddings_summaries_list = self.embeddings.embed_documents(
                summaries_list or []
            )

            if (
                chunks_list
                and summaries_list
                and len(chunks_list) == len(summaries_list)
            ):
                for i, chunk in enumerate(chunks_list):
                    doc_id = str(uuid.uuid4())
                    doc_type = content_type
                    summary = summaries_list[i]
                    embeddings_summary = embeddings_summaries_list[i]
                    metadata: dict[str, Any] = {}

                    try:
                        chunk_metadata = getattr(chunk, "metadata", None)
                        languages = getattr(chunk_metadata, "languages", None) or [
                            "und"
                        ]
                        metadata = {
                            "languages": ",".join(languages),
                            "last_modified": dt.datetime.now(dt.UTC),
                            "page_number": getattr(chunk_metadata, "page_number", 1),
                            "file_directory": getattr(
                                chunk_metadata, "file_directory", None
                            ),
                            "filename": os.path.basename(self.file_path or ""),
                            "filetype": getattr(chunk_metadata, "filetype", None),
                            "uri": f"https://www.devops.wiki/{getattr(chunk_metadata, 'filename', None)}",  # TODO: improve dinamic url
                        }
                        metadata = {k: v for k, v in metadata.items() if v is not None}
                    except Exception:
                        pass

                    document = {
                        "id": doc_id,
                        "type": doc_type,
                        "summary": summary,
                        "content": str(chunk),
                        "metadata": metadata,
                        "embeddings": embeddings_summary,
                    }
                    documents.append(document)

        self.documents = documents

        return documents

    def upload_documents(
        self, documents: list[dict[str, Any]] | None = None
    ) -> list[IndexingResult]:
        if documents:
            return self.search_client.upload_documents(documents=documents)
        else:
            return self.search_client.upload_documents(documents=self.documents)

    def delete_document_by_filename(
        self, filename: str, filter: str | None = None
    ) -> None:
        if not filter:
            filter = f"metadata/filename eq '{filename}'"  # TODO: filter and search filename in other index...s

        results = self.search_client.search(
            search_text="*", filter=filter, select=["id"]
        )

        doc_ids_to_delete = [{"id": doc["id"]} for doc in results]

        if doc_ids_to_delete:
            self.search_client.delete_documents(documents=doc_ids_to_delete)
        else:
            raise ValueError(f"No documents found with filename: {filename}")


class AISearchIndexManager:
    """Create, update, delete and read one Azure AI Search index from its registered schema."""

    def __init__(self, index_client: SearchIndexClient, index_name: str):
        self.index_client = index_client
        self.index_name = index_name

    def get_index(self) -> SearchIndex:
        """Retrieves the definition of the current index.

        Returns:
            SearchIndex: The current index definition.
        """
        return self.index_client.get_index(name=self.index_name)

    def index_exists(self) -> bool:
        """Checks whether the specified search index exists in the Azure Cognitive Search service.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        try:
            self.index_client.get_index(name=self.index_name)
            return True
        except Exception:
            return False

    def create_index(self, registered_index_name: str | None = None) -> None:
        """Creates a new search index from a registered Azure AI Search name if it does not already exist.

        Args:
            registered_index_name (str | None): Registered Azure AI Search index
                name used to resolve the shared schema. Defaults to
                `self.index_name`.

        Raises:
            RuntimeError: If the index already exists.
        """
        if not self.index_exists():
            index = self._load_index_definition(registered_index_name)
            self.index_client.create_index(index)
        else:
            raise RuntimeError(
                f"The index '{self.index_name}' already exists. Please update it instead."
            )

    def update_index(self, registered_index_name: str | None = None) -> None:
        """Updates an existing index from a registered Azure AI Search name.

        Args:
            registered_index_name (str | None): Registered Azure AI Search index
                name used to resolve the shared schema. Defaults to
                `self.index_name`.
        """
        index = self._load_index_definition(registered_index_name)
        self.index_client.create_or_update_index(index=index)

    def delete_index(self) -> None:
        """Deletes the current search index."""
        self.index_client.delete_index(self.index_name)

    def _load_index_definition(
        self, registered_index_name: str | None = None
    ) -> SearchIndex:
        """Load a registered index definition and bind it to `self.index_name`."""
        return load_registered_ai_search_index_definition(
            index_name=registered_index_name or self.index_name,
            runtime_index_name=self.index_name,
        )
