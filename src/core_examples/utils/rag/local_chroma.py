from collections.abc import Callable
from pathlib import Path

from langchain_chroma.vectorstores import Chroma
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.stores import BaseStore

from core_examples.utils.common import (
    get_default_artifacts_directory,
    get_project_root_path,
    resolve_configured_path,
)
from core_examples.utils.rag.langchain_unstructured_indexer import (
    LangChainMultiVectorDocumentIndexer,
)

LOCAL_CHROMA_COLLECTION_NAME = "local_pokemon_collection"
LOCAL_DOC_ID_KEY = "doc_id"
LOCAL_MIN_IMAGE_SIZE = (128, 128)


def get_default_local_chroma_directory() -> Path:
    """Return the default persistent Chroma directory for local storage."""

    return get_default_artifacts_directory() / ".chromadb"


def get_default_local_docstore_directory() -> Path:
    """Return the default companion docstore directory for local storage."""

    return get_default_artifacts_directory() / ".docstore"


def get_default_local_rag_docs_directory() -> Path:
    """Return the default local PDF corpus directory for RAG bootstrapping."""

    return get_default_artifacts_directory() / "rag_docs"


LOCAL_CHROMA_DIRECTORY = get_default_local_chroma_directory()
LOCAL_DOCSTORE_DIRECTORY = get_default_local_docstore_directory()
LOCAL_RAG_DOCS_DIRECTORY = get_default_local_rag_docs_directory()


def _resolve_local_storage_path(
    path_value: str | Path | None,
    default_factory: Callable[[], Path],
) -> Path:
    """Resolve local persistence paths against the project root instead of the cwd."""

    return resolve_configured_path(
        path_value if path_value is not None else default_factory(),
        get_project_root_path(),
    )


def create_local_docstore(docstore_directory: str | Path | None = None) -> BaseStore:
    """Create or open the persistent local docstore used by the local retriever."""

    directory = _resolve_local_storage_path(
        docstore_directory,
        get_default_local_docstore_directory,
    )
    directory.mkdir(parents=True, exist_ok=True)
    return create_kv_docstore(LocalFileStore(str(directory)))


def create_local_vectorstore(
    embeddings: Embeddings,
    collection_name: str = LOCAL_CHROMA_COLLECTION_NAME,
    persist_directory: str | Path | None = None,
) -> Chroma:
    """Create or open the persistent local Chroma collection."""

    directory = _resolve_local_storage_path(
        persist_directory,
        get_default_local_chroma_directory,
    )
    directory.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(directory),
    )


def get_local_vectorstore(
    embeddings: Embeddings,
    collection_name: str = LOCAL_CHROMA_COLLECTION_NAME,
    persist_directory: str | Path | None = None,
) -> Chroma:
    """Open the persistent local Chroma collection without bootstrapping documents."""

    return create_local_vectorstore(
        embeddings=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )


def get_local_retriever_storage(
    embeddings: Embeddings,
    collection_name: str = LOCAL_CHROMA_COLLECTION_NAME,
    persist_directory: str | Path | None = None,
    docstore_directory: str | Path | None = None,
) -> tuple[Chroma, BaseStore]:
    """Open the persistent Chroma collection and companion docstore used by the local retriever."""

    vectorstore = get_local_vectorstore(
        embeddings=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    docstore = create_local_docstore(docstore_directory)
    return vectorstore, docstore


def _docstore_has_data(docstore_directory: str | Path | None) -> bool:
    directory = _resolve_local_storage_path(
        docstore_directory,
        get_default_local_docstore_directory,
    )
    return directory.exists() and any(directory.iterdir())


def _get_pdf_paths(rag_docs_directory: str | Path | None) -> list[Path]:
    directory = _resolve_local_storage_path(
        rag_docs_directory,
        get_default_local_rag_docs_directory,
    )
    pdf_paths = sorted(directory.glob("*.pdf"))

    if not pdf_paths:
        raise ValueError(f"No PDF documents found in {directory} to bootstrap the local Chroma collection.")

    return pdf_paths


def bootstrap_local_vectorstore(
    llm: BaseLanguageModel,
    llm_multimodal: BaseLanguageModel,
    embeddings: Embeddings,
    collection_name: str = LOCAL_CHROMA_COLLECTION_NAME,
    persist_directory: str | Path | None = None,
    docstore_directory: str | Path | None = None,
    rag_docs_directory: str | Path | None = None,
    min_image_size: tuple[int, int] = LOCAL_MIN_IMAGE_SIZE,
) -> Chroma:
    """Bootstrap the local persistent Chroma collection and docstore from the local PDF corpus.

    Images smaller than `min_image_size` are ignored during extraction to avoid indexing icons
    and decorative assets that add noise to the retriever.
    """

    vectorstore = create_local_vectorstore(
        embeddings=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    docstore = create_local_docstore(docstore_directory)

    for pdf_path in _get_pdf_paths(rag_docs_directory):
        indexer = LangChainMultiVectorDocumentIndexer(
            llm=llm,
            llm_multimodal=llm_multimodal,
            vectorstore=vectorstore,
            store=docstore,
            id_key=LOCAL_DOC_ID_KEY,
        )
        indexer.load_pdf(str(pdf_path))
        indexer.split_pdf(min_image_size=min_image_size)
        indexer.summarize_elements()
        indexer.embed_store_documents()

    return vectorstore


def get_or_create_local_vectorstore(
    llm: BaseLanguageModel,
    llm_multimodal: BaseLanguageModel,
    embeddings: Embeddings,
    collection_name: str = LOCAL_CHROMA_COLLECTION_NAME,
    persist_directory: str | Path | None = None,
    docstore_directory: str | Path | None = None,
    rag_docs_directory: str | Path | None = None,
    min_image_size: tuple[int, int] = LOCAL_MIN_IMAGE_SIZE,
) -> Chroma:
    """Return the local persistent Chroma collection, bootstrapping it when no persistence exists yet."""

    vectorstore = create_local_vectorstore(
        embeddings=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    has_collection_data = vectorstore._collection.count() > 0
    has_docstore_data = _docstore_has_data(docstore_directory)

    if has_collection_data and has_docstore_data:
        return vectorstore

    if not has_collection_data and not has_docstore_data:
        return bootstrap_local_vectorstore(
            llm=llm,
            llm_multimodal=llm_multimodal,
            embeddings=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory,
            docstore_directory=docstore_directory,
            rag_docs_directory=rag_docs_directory,
            min_image_size=min_image_size,
        )

    raise RuntimeError(
        "Local Chroma persistence is inconsistent. Keep the Chroma persistence and docstore directories in sync "
        "or remove both to bootstrap the local collection again."
    )