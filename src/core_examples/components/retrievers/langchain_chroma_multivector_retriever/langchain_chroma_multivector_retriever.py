from pathlib import Path

from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.stores import BaseStore
from langchain_core.vectorstores import VectorStore

from core_examples.utils.rag.local_chroma import (
    LOCAL_CHROMA_COLLECTION_NAME,
    LOCAL_CHROMA_DIRECTORY,
    LOCAL_DOC_ID_KEY,
    LOCAL_DOCSTORE_DIRECTORY,
    get_local_retriever_storage,
)


class LangchainChromaMultiVectorRetriever:
    """Build a local persistent MultiVectorRetriever backed by Chroma and a file docstore."""

    def __init__(
        self,
        embeddings: Embeddings,
        vectordb: VectorStore | None = None,
        docstore: BaseStore | None = None,
        id_key: str = LOCAL_DOC_ID_KEY,
        collection_name: str = LOCAL_CHROMA_COLLECTION_NAME,
        persist_directory: str | Path = LOCAL_CHROMA_DIRECTORY,
        docstore_directory: str | Path = LOCAL_DOCSTORE_DIRECTORY,
    ) -> None:
        """Create the local retriever from the persisted Chroma collection and docstore."""

        if vectordb is None or docstore is None:
            persistent_vectordb, persistent_docstore = get_local_retriever_storage(
                embeddings=embeddings,
                collection_name=collection_name,
                persist_directory=persist_directory,
                docstore_directory=docstore_directory,
            )
            self.vectordb = vectordb or persistent_vectordb
            self.docstore = docstore or persistent_docstore
        else:
            self.vectordb = vectordb
            self.docstore = docstore

        self.id_key = id_key
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectordb,
            docstore=self.docstore,
            id_key=self.id_key,
        )

    def get_retriever(self) -> BaseRetriever:
        """Return the configured persistent local retriever."""

        return self.retriever