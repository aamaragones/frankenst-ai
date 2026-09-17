"""LangChain `MultiVectorRetriever` sink for the unstructured PDF indexer."""

import os
import uuid
from typing import Any

from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.stores import BaseStore, InMemoryStore
from langchain_core.vectorstores import VectorStore

from utils.rag.unstructured_pdf_indexer import UnstructuredPDFIndexerBase


class LangChainMultiVectorDocumentIndexer(UnstructuredPDFIndexerBase):
    """Index PDF texts, tables and images into a LangChain `MultiVectorRetriever`."""

    def __init__(
        self,
        llm: BaseLanguageModel[Any],
        llm_multimodal: BaseLanguageModel[Any],
        vectorstore: VectorStore,
        store: BaseStore[str, Document] | None = None,
        id_key: str = "doc_id",
        metadata_retriever: dict[str, Any] | None = None,
    ):
        super().__init__(text_model=llm, image_model=llm_multimodal)
        self.llm = llm
        self.llm_multimodal = llm_multimodal
        self.vectorstore = vectorstore
        self.store: BaseStore[str, Document] = store or InMemoryStore()
        self.id_key = id_key
        self.metadata_retriever = metadata_retriever
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            id_key=self.id_key,
            metadata=self.metadata_retriever,
        )

    def load_pdf(
        self, path: str | None = None, azure_blob: dict[str, Any] | None = None
    ) -> None:
        if path and not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        if path:
            self.file_path = path
        elif azure_blob:
            raise NotImplementedError("Azure blob loading is not yet implemented.")
        else:
            raise ValueError("Provide a path or azure_blob info.")

    def embed_store_documents(self) -> None:
        """Store each chunk under its summary's vector so a hit returns the full parent."""
        for content_type in ["texts", "tables", "images"]:
            chunks_list = self.elements[content_type]
            summaries_list = self.summaries[content_type]
            if (
                chunks_list
                and summaries_list
                and len(chunks_list) == len(summaries_list)
            ):
                chunk_ids = [str(uuid.uuid4()) for _ in chunks_list]
                summary_docs = [
                    Document(
                        page_content=summaries_list[i],
                        metadata={self.id_key: chunk_ids[i]},
                    )
                    for i in range(len(summaries_list))
                ]
                parent_docs = [
                    Document(
                        page_content=self._serialize_parent_chunk(
                            chunks_list[i], content_type
                        ),
                        metadata={"content_type": content_type},
                    )
                    for i in range(len(chunks_list))
                ]
                self.retriever.vectorstore.add_documents(summary_docs)
                self.retriever.docstore.mset(
                    list(zip(chunk_ids, parent_docs, strict=True))
                )

    def _serialize_parent_chunk(self, chunk: Any, content_type: str) -> str:
        """Serialize a raw chunk into a persistable Document page_content value."""
        if content_type == "images":
            return str(chunk)
        if hasattr(chunk, "text"):
            return str(chunk.text)
        return str(chunk)

    def get_retriever(self) -> MultiVectorRetriever:
        """Returns the retriever populated during the workflow.

        Returns:
            MultiVectorRetriever: Ready-to-query retriever.
        """
        has_elements = any(
            self.elements.get(content_type)
            for content_type in ("texts", "tables", "images")
        )
        has_summaries = any(
            self.summaries.get(content_type)
            for content_type in ("texts", "tables", "images")
        )

        if has_elements or has_summaries:
            return self.retriever
        else:
            raise ValueError(
                "Not chunks detected. If already exist in store/vectorstore please use get_prebuild_retriever."
            )

    def get_prebuilt_retriever(self) -> MultiVectorRetriever:
        """Returns an existing retriever assuming documents already exist in store/vectorstore.

        Returns:
            MultiVectorRetriever: The retriever ready for querying.
        """
        return self.retriever
